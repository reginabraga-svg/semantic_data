# Ontology-Grounded Graph RAG for a Semantic Data Lakehouse — Code, Ontologies, and Evaluation Data

This repository contains the ontologies, source code, and raw evaluation results
accompanying the paper *"A Semantic Data Lakehouse Architecture for Enterprise
Information Systems Interoperability."* It allows independent reproduction of the
**Intelligent Layer** evaluation reported in the paper: an ontology-grounded
Graph RAG pipeline that translates natural-language questions into SPARQL,
executes them against a curated OWL ontology, enriches results with PROV-O
lineage, and synthesizes a natural-language answer.

> **Anonymized for double-blind review.** All author names, affiliations, and
> personal identifiers have been removed from these files. The ontology
> namespace IRIs have been neutralized to `…/anon/…`. This deposit will be
> updated with full attribution upon paper acceptance.

## What the Intelligent Layer does

The pipeline has four stages:

1. **Schema extraction** — produces a compact textual rendering of the OWL
   ontology (classes, properties, domains/ranges) for use as LLM context.
2. **NL → SPARQL translation** — an LLM, conditioned on the extracted schema,
   converts a user question (English or Portuguese) into a SPARQL query.
3. **SPARQL execution + PROV-O lineage** — the query runs against the populated
   Gold-stage ontology; result rows are enriched with cross-stage data lineage
   drawn from the PROV-O provenance ontology.
4. **Response synthesis** — an LLM composes a natural-language answer that cites
   each ontological individual and appends a provenance footer.

The central design property is that the **expert-curated OWL ontology** acts as
a structural guardrail: queries against typed properties either parse correctly
against existing properties or fail at parse time, so the language model cannot
fabricate facts that are not present in the curated graph.

## Repository layout

```
.
├── README.md                 ← this file
├── ollama/                   ← fully self-contained local-SLM deployment
│   ├── CorpOnto.owl          ← Gold-stage domain ontology (RDF/XML, load this)
│   ├── CorpOnto.owx          ← same ontology in OWL/XML (Protégé native)
│   ├── proveniencia.owl      ← PROV-O lineage ontology (RDF/XML, load this)
│   ├── proveniencia.owx      ← same ontology in OWL/XML (Protégé native)
│   ├── schema_extractor.py   ← stage 1
│   ├── ollama_clients.py     ← LLM clients (Ollama: Qwen 2.5 Coder + Llama 3.2)
│   ├── executor.py           ← stage 3 (SPARQL + PROV-O lineage)
│   ├── pipeline_ollama.py    ← end-to-end driver
│   ├── README_ollama.md      ← step-by-step run guide (local CPU)
│   └── intelligent_layer_ollama_results.json   ← raw results (18 questions)
└── gemini/                   ← fully self-contained cloud-LLM deployment
    ├── CorpOnto.owl / .owx
    ├── proveniencia.owl / .owx
    ├── schema_extractor.py
    ├── gemini_clients.py     ← LLM client (Gemini 2.5 Flash, free tier)
    ├── executor.py
    ├── pipeline_gemini.py
    ├── README_gemini.md      ← step-by-step run guide (cloud API)
    └── intelligent_layer_gemini_results.json   ← raw results (18 questions)
```

The two subfolders are intentionally self-contained so that each experiment can
be reproduced in isolation. `schema_extractor.py`, `executor.py`, and the two
ontology files are identical across the two folders.

### Ontology file formats

Two formats of each ontology are provided. The pipeline code loads the **`.owl`
(RDF/XML)** files via `rdflib`/`owlready2`; the **`.owx` (OWL/XML)** files are the
Protégé-native exports, provided for convenience if you wish to open and inspect
the ontologies in Protégé. The two formats encode the same logical content
(CorpOnto: 2,785 RDF triples; proveniencia: 1,084 RDF triples).

## How to reproduce

Detailed, platform-specific instructions are in `ollama/README_ollama.md`
(local, no API key, no cost) and `gemini/README_gemini.md` (cloud, free-tier API
key required). In brief:

**Local SLM deployment (Ollama):**
```bash
cd ollama
pip install owlready2 rdflib ollama
ollama pull qwen2.5-coder:1.5b
ollama pull llama3.2:3b
python pipeline_ollama.py --corponto CorpOnto.owl --proveniencia proveniencia.owl
```

**Cloud LLM deployment (Gemini):**
```bash
cd gemini
pip install owlready2 rdflib google-genai
export GEMINI_API_KEY=<your-key>          # see README_gemini.md
python pipeline_gemini.py --corponto CorpOnto.owl --proveniencia proveniencia.owl --delay 20
```

## Evaluation data

Each `intelligent_layer_*_results.json` contains, for all 18 evaluation
questions: the question text, the LLM-generated SPARQL, the hand-curated
reference SPARQL, result-set comparison signatures, the correctness label
(`exact_match` / `partial_match` / `no_match`), per-stage timings, and the
synthesized answer. The 18 questions comprise the six paper competency questions
in English and in Portuguese, plus six additional questions exercising
aggregation, multi-hop joins, and looser vocabulary correspondence.

**Configuration recorded in each run:**

- Ollama run: SPARQL model `qwen2.5-coder:1.5b`, synthesis model `llama3.2:3b`,
  local CPU.
- Gemini run: SPARQL and synthesis model `gemini-2.5-flash`, Google AI Studio
  free tier.

Note that the Gemini free tier imposes a daily request quota; the run reported
in the paper was constrained by this quota, as documented in the paper's threats
to validity. To reproduce a full uninterrupted Gemini run, use `--delay 20` or a
paid-tier key.

## Dependencies

- Python 3.10+
- `owlready2`, `rdflib` (deterministic components)
- `ollama` (local deployment) **or** `google-genai` (cloud deployment)
- For the local deployment: an Ollama installation with the two models pulled.

## License

The code is released under the MIT License. The ontologies and evaluation data
are released under the Creative Commons Attribution 4.0 International (CC BY 4.0)
license. See the Zenodo record metadata for details.
