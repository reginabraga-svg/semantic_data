# Semantic Data Lakehouse — Research Artifact

This repository accompanies the paper *"A Semantic Data Lakehouse Architecture
for Enterprise Information Systems Interoperability."* It contains the ontologies,
the Intelligent Layer pipeline, the comparative baseline, the scale sensitivity
test, and the raw evaluation results needed to reproduce the reported findings.

> **Note on data.** All instance data is a controlled, synthetic instantiation.
> It contains no confidential partner-company data. Identifiers such as
> `Opportunity_OPP001` or `TechnicalProposal_001` are synthetic entities of the
> controlled dataset.

## Repository layout

```
ontologies/       The two OWL ontologies (Protégé OWL/XML, .owx)
  CorpOnto.owx        Domain ontology (31 classes, 246 individuals)
  proveniencia.owx    PROV-O provenance ontology

pipeline/         The Intelligent Layer (ontology-grounded Graph RAG)
  pipeline_gemini.py    End-to-end pipeline: schema extraction -> NL-to-SPARQL
                        -> execution with PROV-O lineage -> answer synthesis
  gemini_clients.py     Gemini-backed translator/synthesizer (thinking disabled,
                        robust SPARQL extraction)
  schema_extractor.py   Ontology schema extraction for LLM context
  executor.py           SPARQL execution against the triplestore

scale_test/       Scale sensitivity check (synthetic, structure-preserving)
  scale_test.py             Generates an N x scaled instantiation and measures
                            reasoner and competency-question SPARQL latency
  scale_test_results.json   Results at 10x (2,460 individuals)

evaluation/       Raw evaluation results
  intelligent_layer_ollama_results.json    Local SLM deployment (18 questions)
  intelligent_layer_gemini_35flash.json    Cloud LLM (Gemini 3.5 Flash, 18 questions)
  second_rater/
    second_rater_results.xlsx              Blind second-rater assessment + kappa

baseline/         Comparative baseline (LLM-extracted knowledge graph)
  extracted_kg_ollama.json       The KG extracted by the LLM
  baseline_results_ollama.json   Baseline query/answer results
```

## Requirements

```
python >= 3.10
pip install owlready2 rdflib google-genai
```

The reasoner step (HermiT, bundled with owlready2) requires a Java runtime.
If Java is unavailable, that step is skipped and SPARQL latencies are still
measured.

## Reproducing the results

### Intelligent Layer (cloud LLM)

```
export GEMINI_API_KEY=your_key_here      # never hard-code the key
python pipeline/pipeline_gemini.py \
    --corponto ontologies/CorpOnto.owx \
    --proveniencia ontologies/proveniencia.owx \
    --sparql-model gemini-3.5-flash \
    --synth-model  gemini-3.5-flash \
    --delay 2 \
    --output intelligent_layer_gemini.json
```

Reported result: 16/18 semantically correct (89%), 0 hallucinated facts,
median end-to-end latency ~5.1 s.

### Scale sensitivity check

```
python scale_test/scale_test.py \
    --corponto ontologies/CorpOnto.owx \
    --factor 10
```

Reported result: at 2,460 individuals, median competency-question SPARQL latency
remains in the single-digit-millisecond range (~6.3 ms on the reference machine),
confirming the sub-30 ms query latencies are not an artifact of instantiation size.
This is a synthetic sensitivity check, not an enterprise-scale benchmark.

### Second-rater agreement

`evaluation/second_rater/second_rater_results.xlsx` contains the blind
second-rater assessment (33 of 36 items) and the agreement summary:
Cohen's kappa = 1.00 on the correct-vs-incorrect distinction (perfect),
and 0.52 on the three-level classification (moderate), with all disagreements
at the exact-vs-partial boundary.

## Notes

- The ontology namespace uses `anon` in place of author-identifying information.
- No API keys are stored in any file; all model access reads `GEMINI_API_KEY`
  from the environment.
