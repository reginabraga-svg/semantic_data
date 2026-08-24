"""
Gemini-backed implementations of the two LLM-dependent stages of the
Intelligent Layer:
  - GeminiNLToSPARQL : translates a natural-language question into a SPARQL
                       query, conditioned on the CorpOnto schema.
  - GeminiResponseSynthesizer : produces a natural-language answer that cites
                                each returned individual and a provenance footer.

Both classes call Google's Gemini API. The API has a generous free tier
(typically 1,500 requests/day) — no credit card required.

To get an API key:
  1. Go to https://aistudio.google.com/apikey
  2. Click "Create API key" (uses your Google account)
  3. Copy the key, export it: `export GEMINI_API_KEY=...`

Default models:
  - SPARQL generation : gemini-2.5-flash  (fast, low-latency)
  - response synthesis: gemini-2.5-flash  (same model works well for both)

Usage:
    >>> from gemini_clients import GeminiNLToSPARQL
    >>> tr = GeminiNLToSPARQL(schema_text="...", model="gemini-2.5-flash")
    >>> sparql, meta = tr.translate("Quais funcionários sabem Python?")
"""
import re
import os
import time
from google import genai
from google.genai import types


# ============================================================
# SYSTEM PROMPTS
# ============================================================

NS_URI = "http://www.semanticweb.org/anon/ontologies/2025/8/CorpOnto.owl#"

SPARQL_SYSTEM_PROMPT = """You translate natural-language questions (English or Portuguese) into SPARQL queries against a corporate ontology. The ontology schema is shown below.

{schema}

RULES:
1. Output ONLY a valid SPARQL query wrapped in ```sparql ... ``` fences. No explanation, no commentary.
2. Always include the namespace prefix:
   PREFIX : <{ns}>
3. Use SELECT DISTINCT to avoid duplicates.
4. Employee instances do NOT have a hasName property; they are identified by IRI (e.g. :Employee_001). Only Profile, Knowledge, Customer, BusinessPartner have :hasName.
5. For numeric thresholds use FILTER with comparison operators.
6. If the question cannot be answered from this ontology, output:
   ```sparql
   # CANNOT_ANSWER: <one-sentence reason>
   ```
7. Portuguese terminology mapping: "funcionário"=Employee, "projeto"=Project, "oportunidade"=Opportunity, "proposta técnica"=TechnicalProposal, "proposta comercial"=CommercialProposal, "perfil"=Profile, "conhecimento"=Knowledge, "margem"=margin, "cliente"=Customer, "fornecedor"=Supplier, "fatura"=Invoice, "pagamento"=Payment.

EXAMPLES:

Q: Which opportunities have a technical proposal with a cost estimate?
```sparql
PREFIX : <{ns}>
SELECT DISTINCT ?opp WHERE {{
  ?opp a :Opportunity .
  ?opp :hasTechnicalProposal ?tp .
  ?tp :technicalProposalNeedsToWorkHourProfile ?whp .
  ?whp :expectedHours ?h .
}}
```

Q: Quais funcionários conhecem Python?
```sparql
PREFIX : <{ns}>
SELECT DISTINCT ?emp ?profile WHERE {{
  ?emp a :Employee .
  ?emp :hasKnowledge ?k .
  ?k :hasName "Python" .
  OPTIONAL {{ ?emp :hasProfile ?p . ?p :hasName ?profile . }}
}}
```

Q: Quais projetos tiveram margem real acima de 25%?
```sparql
PREFIX : <{ns}>
SELECT DISTINCT ?project ?margin WHERE {{
  ?project a :Project .
  ?project :hasRealMargin ?margin .
  FILTER(?margin > 0.25)
}}
ORDER BY DESC(?margin)
```
"""


RESPONSE_SYSTEM_PROMPT = """You are an enterprise analytics assistant grounded in a corporate ontology. You receive a user question, the SPARQL query that was executed, the results returned by the ontology, and the provenance lineage of the data.

Produce a concise natural-language answer that:
1. Directly answers the user's question.
2. Cites each ontological individual using the format [IndividualName].
3. Includes a brief note at the end indicating that the underlying data traversed the Source -> Kafka -> Bronze -> Silver -> Gold pipeline.
4. Uses the same language as the question (Portuguese or English).
5. Does NOT invent facts not present in the results. If the result set is empty, say so plainly.
6. Plain prose, no markdown headers or bold.
7. Be brief: 2 to 4 sentences total.
"""



def _extract_sparql(text):
    """Robustly extract a SPARQL query from an LLM response that may contain
    markdown fences, thinking text, or be truncated.

    Strategy:
      1. If a ```sparql ... ``` fence exists (even without closing), take its body.
      2. Otherwise, take from the first PREFIX/SELECT/ASK/CONSTRUCT/DESCRIBE
         keyword to the end.
      3. Strip any trailing markdown fence or thinking commentary.
    """
    if not text:
        return ""
    s = text
    # 1. Try fenced block: opening ```sparql, optional closing ```
    m = re.search(r"```sparql\s*\n(.*?)(?:```|$)", s, re.DOTALL)
    if m and m.group(1).strip():
        candidate = m.group(1).strip()
    else:
        # 2. Fall back: cut from first query keyword
        m2 = re.search(r"(PREFIX\s|SELECT\s|ASK\s|CONSTRUCT\s|DESCRIBE\s)",
                       s, re.IGNORECASE)
        candidate = s[m2.start():] if m2 else s.strip()
    # 3. Remove any stray fences and trailing commentary
    candidate = candidate.replace("```sparql", "").replace("```", "")
    # Cut trailing thinking that sometimes follows the closing brace
    candidate = re.split(r"\n\s*(?:Wait\b|Is this correct|Note:|Actually\b)",
                         candidate)[0]
    return candidate.strip()


# ============================================================
# GEMINI CLIENT WRAPPERS
# ============================================================

class GeminiNLToSPARQL:
    def __init__(self, schema_text, model="gemini-2.5-flash",
                 api_key=None, temperature=0.0):
        # API key resolution: explicit param > GEMINI_API_KEY env > GOOGLE_API_KEY env
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY environment variable "
                "or pass api_key=... to the constructor. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=api_key)
        self.system_prompt = SPARQL_SYSTEM_PROMPT.format(
            schema=schema_text, ns=NS_URI
        )
        self.model = model
        self.temperature = temperature

    def translate(self, question):
        """Return (sparql_string, meta_dict).
        meta_dict contains: latency_ms, model, prompt_tokens, response_tokens."""
        t0 = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=self.temperature,
                    max_output_tokens=2048,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except Exception as e:
            return f"# CANNOT_ANSWER: Gemini call failed: {e}", {
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "model": self.model,
                "error": str(e),
            }
        elapsed_ms = (time.perf_counter() - t0) * 1000

        text = response.text or ""
        sparql = _extract_sparql(text)

        usage = getattr(response, "usage_metadata", None)
        meta = {
            "latency_ms": round(elapsed_ms, 2),
            "model": self.model,
            "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "response_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            "raw_response_first200": text[:200],
        }
        return sparql, meta


class GeminiResponseSynthesizer:
    def __init__(self, model="gemini-2.5-flash", api_key=None, temperature=0.2):
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY environment variable."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def synthesize(self, question, sparql, exec_result):
        """Return (nl_answer_string, meta_dict)."""
        context = self._format_context(question, sparql, exec_result)
        t0 = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=context,
                config=types.GenerateContentConfig(
                    system_instruction=RESPONSE_SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_output_tokens=1024,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except Exception as e:
            return f"[Synthesis failed: {e}]", {
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "error": str(e),
            }
        elapsed_ms = (time.perf_counter() - t0) * 1000
        text = (response.text or "").strip()

        usage = getattr(response, "usage_metadata", None)
        meta = {
            "latency_ms": round(elapsed_ms, 2),
            "model": self.model,
            "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "response_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
        }
        return text, meta

    def _format_context(self, question, sparql, exec_result):
        parts = [
            f"USER QUESTION: {question}",
            "",
            "SPARQL QUERY EXECUTED:",
            sparql,
            "",
            f"RESULTS ({exec_result['n_rows']} rows):",
        ]
        if exec_result["n_rows"] == 0:
            parts.append("(empty result set)")
        else:
            for row in exec_result["rows"][:50]:
                parts.append(
                    "  " + " | ".join((c["value"] or "-") for c in row)
                )
        if exec_result.get("lineage"):
            parts.append("")
            parts.append("PROVENANCE LINEAGE:")
            for ind, lineage in list(exec_result["lineage"].items())[:5]:
                chain = " <- ".join(
                    f"[{s['stage']}] {s['entity']}" for s in lineage
                )
                parts.append(f"  {ind}: {chain}")
        return "\n".join(parts)


# ============================================================
# Helper: quick connectivity check
# ============================================================

def check_gemini_available(api_key=None):
    """Return True if Gemini API responds, else False with reason."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return False, "No API key set in GEMINI_API_KEY or GOOGLE_API_KEY"
    try:
        client = genai.Client(api_key=api_key)
        # Cheap probe
        list(client.models.list())
        return True, "OK"
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    print("Gemini connectivity check...")
    ok, reason = check_gemini_available()
    if not ok:
        print(f"  NOT reachable: {reason}")
        print("  Get a free API key at https://aistudio.google.com/apikey")
        print("  Then: export GEMINI_API_KEY=AIza...")
    else:
        print(f"  Reachable: {reason}")
