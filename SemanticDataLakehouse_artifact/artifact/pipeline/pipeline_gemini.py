"""
Full Intelligent Layer pipeline with Gemini API backing.

Pipeline:
  user question -> [GeminiNLToSPARQL]            -> SPARQL
                -> [OntologyExecutor]            -> structured rows
                -> [PROV-O lineage enrichment]   -> rows + lineage
                -> [GeminiResponseSynthesizer]   -> NL answer with citations

USAGE:
    export GEMINI_API_KEY=AIza...           # get from https://aistudio.google.com/apikey
    python3 pipeline_gemini.py              # full run, default settings
    python3 pipeline_gemini.py --sparql-model gemini-2.5-flash
    python3 pipeline_gemini.py --synth-model gemini-2.5-flash
    python3 pipeline_gemini.py --questions 1,3,7    # subset
    python3 pipeline_gemini.py --no-synth           # SPARQL only, fewer API calls
    python3 pipeline_gemini.py --delay 2            # seconds between calls (for rate limits)

Requirements:
    pip install owlready2 rdflib google-genai
    Gemini API key (free, get at https://aistudio.google.com/apikey)
"""
import sys
import os
import json
import time
import argparse
import statistics
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from schema_extractor import extract_schema, format_schema_for_llm
from executor import OntologyExecutor
from gemini_clients import (
    GeminiNLToSPARQL, GeminiResponseSynthesizer, check_gemini_available
)

# Same 18-question test set as pipeline_ollama.py
NS = "PREFIX : <http://www.semanticweb.org/anon/ontologies/2025/8/CorpOnto.owl#>"

QUESTIONS = [
    # Paper CQs - English
    {"id": "EN-CQ1", "lang": "en", "category": "paper_cq",
     "question": "Which opportunities have a technical proposal with a cost estimate?",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?opp WHERE {{
  ?opp a :Opportunity .
  ?opp :hasTechnicalProposal ?tp .
  ?tp :technicalProposalNeedsToWorkHourProfile ?whp .
  ?whp :expectedHours ?h .
}} ORDER BY ?opp"""},
    {"id": "EN-CQ2", "lang": "en", "category": "paper_cq",
     "question": "Which technical proposals require a Backend Developer profile?",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?tp WHERE {{
  ?tp a :TechnicalProposal .
  ?tp :technicalProposalNeedsToWorkHourProfile ?whp .
  ?whp :workHourProfileNeedsToProfile ?p .
  ?p :hasName "Backend Developer" .
}}"""},
    {"id": "EN-CQ3", "lang": "en", "category": "paper_cq",
     "question": "Which projects are linked to opportunities?",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?proj ?opp WHERE {{
  ?proj a :Project .
  ?proj :projectRelatedToOpportunity ?opp .
}} ORDER BY ?proj"""},
    {"id": "EN-CQ4", "lang": "en", "category": "paper_cq",
     "question": "Which employees know Python?",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?emp ?profName WHERE {{
  ?emp a :Employee .
  ?emp :hasKnowledge ?k .
  ?k :hasName "Python" .
  OPTIONAL {{ ?emp :hasProfile ?p . ?p :hasName ?profName }}
}} ORDER BY ?emp"""},
    {"id": "EN-CQ5", "lang": "en", "category": "paper_cq",
     "question": "Which profiles are needed for the project linked to opportunity OPP001?",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?profName ?hours WHERE {{
  :Opportunity_OPP001 :hasTechnicalProposal ?tp .
  ?tp :technicalProposalNeedsToWorkHourProfile ?whp .
  ?whp :expectedHours ?hours .
  ?whp :workHourProfileNeedsToProfile ?p .
  ?p :hasName ?profName .
}}"""},
    {"id": "EN-CQ6", "lang": "en", "category": "paper_cq",
     "question": "Which projects have real margin above 25%?",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?proj ?margin WHERE {{
  ?proj a :Project .
  ?proj :hasRealMargin ?margin .
  FILTER(?margin > 0.25)
}} ORDER BY DESC(?margin)"""},

    # Paper CQs - Portuguese
    {"id": "PT-CQ1", "lang": "pt", "category": "paper_cq_pt",
     "question": "Quais oportunidades já têm proposta técnica com estimativa de custo?",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?opp WHERE {{
  ?opp a :Opportunity .
  ?opp :hasTechnicalProposal ?tp .
  ?tp :technicalProposalNeedsToWorkHourProfile ?whp .
  ?whp :expectedHours ?h .
}} ORDER BY ?opp"""},
    {"id": "PT-CQ2", "lang": "pt", "category": "paper_cq_pt",
     "question": "Quais propostas técnicas precisam de um Backend Developer?",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?tp WHERE {{
  ?tp a :TechnicalProposal .
  ?tp :technicalProposalNeedsToWorkHourProfile ?whp .
  ?whp :workHourProfileNeedsToProfile ?p .
  ?p :hasName "Backend Developer" .
}}"""},
    {"id": "PT-CQ3", "lang": "pt", "category": "paper_cq_pt",
     "question": "Quais projetos estão ligados a oportunidades?",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?proj ?opp WHERE {{
  ?proj a :Project .
  ?proj :projectRelatedToOpportunity ?opp .
}} ORDER BY ?proj"""},
    {"id": "PT-CQ4", "lang": "pt", "category": "paper_cq_pt",
     "question": "Quais funcionários sabem Python?",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?emp ?profName WHERE {{
  ?emp a :Employee .
  ?emp :hasKnowledge ?k .
  ?k :hasName "Python" .
  OPTIONAL {{ ?emp :hasProfile ?p . ?p :hasName ?profName }}
}} ORDER BY ?emp"""},
    {"id": "PT-CQ5", "lang": "pt", "category": "paper_cq_pt",
     "question": "Quais perfis são necessários para o projeto ligado à oportunidade OPP001?",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?profName ?hours WHERE {{
  :Opportunity_OPP001 :hasTechnicalProposal ?tp .
  ?tp :technicalProposalNeedsToWorkHourProfile ?whp .
  ?whp :expectedHours ?hours .
  ?whp :workHourProfileNeedsToProfile ?p .
  ?p :hasName ?profName .
}}"""},
    {"id": "PT-CQ6", "lang": "pt", "category": "paper_cq_pt",
     "question": "Quais projetos tiveram margem real acima de 25%?",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?proj ?margin WHERE {{
  ?proj a :Project .
  ?proj :hasRealMargin ?margin .
  FILTER(?margin > 0.25)
}} ORDER BY DESC(?margin)"""},

    # Natural questions
    {"id": "NAT-1", "lang": "en", "category": "natural",
     "question": "Show me the projects whose real margin is below their expected margin.",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?proj ?real ?expected WHERE {{
  ?proj a :Project .
  ?proj :hasRealMargin ?real .
  ?proj :expectedMargin ?expected .
  FILTER(?real < ?expected)
}}"""},
    {"id": "NAT-2", "lang": "pt", "category": "natural",
     "question": "Quantos funcionários temos por perfil?",
     "reference_sparql": f"""{NS}
SELECT ?profName (COUNT(?emp) AS ?count) WHERE {{
  ?emp a :Employee .
  ?emp :hasProfile ?p .
  ?p :hasName ?profName .
}} GROUP BY ?profName ORDER BY DESC(?count)"""},
    {"id": "NAT-3", "lang": "en", "category": "natural",
     "question": "What is the average expected hours across all technical proposals?",
     "reference_sparql": f"""{NS}
SELECT (AVG(?h) AS ?avg_hours) WHERE {{
  ?tp a :TechnicalProposal .
  ?tp :technicalProposalNeedsToWorkHourProfile ?whp .
  ?whp :expectedHours ?h .
}}"""},
    {"id": "NAT-4", "lang": "pt", "category": "natural",
     "question": "Quais funcionários trabalham em mais de um projeto?",
     "reference_sparql": f"""{NS}
SELECT ?emp (COUNT(DISTINCT ?proj) AS ?n_projects) WHERE {{
  ?emp a :Employee .
  ?proj :employs ?emp .
}} GROUP BY ?emp HAVING (COUNT(DISTINCT ?proj) > 1)"""},
    {"id": "NAT-5", "lang": "en", "category": "natural",
     "question": "Which opportunities have both a technical and a commercial proposal?",
     "reference_sparql": f"""{NS}
SELECT DISTINCT ?opp WHERE {{
  ?opp a :Opportunity .
  ?opp :hasTechnicalProposal ?tp .
  ?opp :hasCommercialProposal ?cp .
}}"""},
    {"id": "NAT-6", "lang": "pt", "category": "natural",
     "question": "Qual o valor total dos pagamentos recebidos?",
     "reference_sparql": f"""{NS}
SELECT (SUM(?a) AS ?total) WHERE {{
  ?p a :IncomingPayment .
  ?p :amount ?a .
}}"""},
]


def result_set_signature(rows):
    """Comparable signature of a SPARQL result set (set of identifiers/values)."""
    sig = set()
    for row in rows:
        for cell in row:
            if cell["name"]:
                sig.add(cell["name"])
            elif cell["value"]:
                try:
                    sig.add(round(float(cell["value"]), 2))
                except (ValueError, TypeError):
                    sig.add(cell["value"])
    return sig


def run(args):
    # Resolve ontology paths
    corponto_path = args.corponto or "/home/claude/CorpOnto.owl"
    prov_path = args.proveniencia or "/home/claude/proveniencia.owl"
    for p, lbl in [(corponto_path, "CorpOnto"), (prov_path, "proveniencia")]:
        if not Path(p).exists():
            print(f"ERROR: {lbl} not found at {p}")
            print("  Pass --corponto and --proveniencia to point at your local copies.")
            sys.exit(1)

    # Gemini connectivity
    ok, reason = check_gemini_available()
    if not ok:
        print(f"ERROR: Gemini API not reachable: {reason}")
        print("  1. Get a free API key at https://aistudio.google.com/apikey")
        print("  2. export GEMINI_API_KEY=AIza...")
        sys.exit(1)
    print(f"Gemini OK. Models:")
    print(f"  SPARQL model       : {args.sparql_model}")
    print(f"  Synthesis model    : {args.synth_model if not args.no_synth else '(skipped)'}")
    if args.delay > 0:
        print(f"  Delay between calls: {args.delay}s (to stay under free-tier rate limits)")

    # Schema extraction
    print("\n[1/4] Extracting schema from CorpOnto...")
    t0 = time.perf_counter()
    schema = extract_schema(f"file://{corponto_path}")
    schema_text = format_schema_for_llm(schema)
    schema_ms = (time.perf_counter() - t0) * 1000
    print(f"  Done in {schema_ms:.1f} ms ({len(schema_text)} chars)")

    # Initialize components
    print("\n[2/4] Initializing Gemini-backed translator and synthesizer...")
    translator = GeminiNLToSPARQL(schema_text, model=args.sparql_model)
    synth = None
    if not args.no_synth:
        synth = GeminiResponseSynthesizer(model=args.synth_model)

    # Ontology executor
    print("\n[3/4] Loading ontologies into triplestore...")
    executor = OntologyExecutor(
        f"file://{corponto_path}", f"file://{prov_path}"
    )
    print("  Loaded.")

    # Select questions
    if args.questions:
        idxs = [int(x) - 1 for x in args.questions.split(",")]
        questions = [QUESTIONS[i] for i in idxs if 0 <= i < len(QUESTIONS)]
    else:
        questions = QUESTIONS
    print(f"\n[4/4] Running {len(questions)} questions through the pipeline...")
    print("=" * 78)

    results = []
    for q_idx, q in enumerate(questions, start=1):
        print(f"\n[{q_idx}/{len(questions)} | {q['id']}] {q['question']}")
        t_total = time.perf_counter()

        # Stage 1: NL -> SPARQL via Gemini
        sparql, sparql_meta = translator.translate(q["question"])
        print(f"  Generated SPARQL ({sparql_meta['latency_ms']:.0f} ms):")
        for line in sparql.splitlines()[:10]:
            print(f"    {line}")
        if len(sparql.splitlines()) > 10:
            print(f"    ... ({len(sparql.splitlines())-10} more lines)")

        # Stage 2: execute live SPARQL
        try:
            exec_result = executor.execute_sparql(sparql)
            exec_error = None
        except Exception as e:
            exec_result = {"n_rows": 0, "rows": [], "latency_ms": 0,
                           "lineage": {}, "lineage_latency_ms": 0}
            exec_error = str(e)
            print(f"  EXECUTION ERROR: {e}")

        # Stage 3: enrich with lineage
        if exec_result["n_rows"] > 0:
            executor.enrich_results_with_lineage(exec_result)

        # Reference result for correctness comparison
        try:
            ref_result = executor.execute_sparql(q["reference_sparql"])
            sig_llm = result_set_signature(exec_result["rows"])
            sig_ref = result_set_signature(ref_result["rows"])
            correctness = "exact_match" if sig_llm == sig_ref else (
                "partial_match" if sig_llm & sig_ref else "no_match"
            )
        except Exception as e:
            ref_result = {"n_rows": 0, "rows": []}
            correctness = f"reference_failed:{e}"
        print(f"  LLM rows: {exec_result['n_rows']}, "
              f"reference rows: {ref_result['n_rows']}, "
              f"correctness: {correctness}")

        # Optional pacing delay between calls (helps stay under free-tier RPM limits)
        if args.delay > 0 and not args.no_synth:
            time.sleep(args.delay)

        # Stage 4: synthesize NL answer
        answer, synth_meta = None, None
        if synth and not exec_error:
            answer, synth_meta = synth.synthesize(q["question"], sparql, exec_result)
            print(f"  Answer ({synth_meta['latency_ms']:.0f} ms): {answer[:200]}")

        total_ms = (time.perf_counter() - t_total) * 1000

        results.append({
            "id": q["id"],
            "lang": q["lang"],
            "category": q["category"],
            "question": q["question"],
            "generated_sparql": sparql,
            "reference_sparql": q["reference_sparql"],
            "exec_error": exec_error,
            "n_rows_llm": exec_result["n_rows"],
            "n_rows_reference": ref_result["n_rows"],
            "correctness": correctness,
            "rows_sample": [
                [c["value"] for c in row]
                for row in exec_result["rows"][:5]
            ],
            "answer": answer,
            "latency_ms": {
                "nl_to_sparql": sparql_meta["latency_ms"],
                "sparql_execution": exec_result["latency_ms"],
                "lineage_enrichment": exec_result.get("lineage_latency_ms", 0),
                "synthesis": synth_meta["latency_ms"] if synth_meta else None,
                "total": round(total_ms, 2),
            },
            "sparql_meta": sparql_meta,
            "synth_meta": synth_meta,
        })

        # Pacing delay
        if args.delay > 0 and q_idx < len(questions):
            time.sleep(args.delay)

    # Save JSON
    out_json = args.output or "intelligent_layer_gemini_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "config": {
                "sparql_model": args.sparql_model,
                "synth_model": args.synth_model if not args.no_synth else None,
                "schema_chars": len(schema_text),
                "schema_extract_ms": round(schema_ms, 1),
                "n_questions": len(questions),
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    # Aggregate
    print("\n" + "=" * 78)
    print("AGGREGATE STATISTICS")
    print("=" * 78)
    by_correctness = {"exact_match": 0, "partial_match": 0, "no_match": 0}
    for r in results:
        key = r["correctness"]
        if key in by_correctness:
            by_correctness[key] += 1
    print(f"  SPARQL correctness vs hand-curated reference:")
    print(f"    Exact match   : {by_correctness['exact_match']} / {len(results)}")
    print(f"    Partial match : {by_correctness['partial_match']} / {len(results)}")
    print(f"    No match      : {by_correctness['no_match']} / {len(results)}")

    sparql_lats = [r["latency_ms"]["nl_to_sparql"] for r in results]
    exec_lats = [r["latency_ms"]["sparql_execution"] for r in results]
    synth_lats = [r["latency_ms"]["synthesis"] for r in results if r["latency_ms"]["synthesis"]]
    total_lats = [r["latency_ms"]["total"] for r in results]
    print(f"\n  Latency (ms, median across {len(results)} questions):")
    print(f"    NL -> SPARQL (LLM)   : {statistics.median(sparql_lats):.0f}")
    print(f"    SPARQL execution     : {statistics.median(exec_lats):.2f}")
    if synth_lats:
        print(f"    Response synthesis   : {statistics.median(synth_lats):.0f}")
    print(f"    End-to-end           : {statistics.median(total_lats):.0f}")

    # By category
    print("\n  Correctness by category:")
    for cat in ['paper_cq', 'paper_cq_pt', 'natural']:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results: continue
        n_exact = sum(1 for r in cat_results if r["correctness"] == "exact_match")
        n_partial = sum(1 for r in cat_results if r["correctness"] == "partial_match")
        print(f"    {cat:15s}: {n_exact}/{len(cat_results)} exact, "
              f"{n_partial}/{len(cat_results)} partial")

    print(f"\nFull results saved to: {out_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the Intelligent Layer pipeline with Gemini API."
    )
    parser.add_argument("--corponto", help="Path to CorpOnto.owl")
    parser.add_argument("--proveniencia", help="Path to proveniencia.owl")
    parser.add_argument("--sparql-model", default="gemini-2.5-flash",
                        help="Gemini model for NL->SPARQL translation")
    parser.add_argument("--synth-model", default="gemini-2.5-flash",
                        help="Gemini model for response synthesis")
    parser.add_argument("--questions", help="Comma-separated 1-based indexes")
    parser.add_argument("--no-synth", action="store_true",
                        help="Skip response synthesis stage (halves API calls)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to wait between API calls (helps with rate limits)")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
