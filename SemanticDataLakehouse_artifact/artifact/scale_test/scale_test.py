"""
scale_test.py — Scale sensitivity check for the Semantic Data Lakehouse paper.

Generates a synthetically scaled instantiation (~10x) that preserves the SAME
class structure and property distributions as the primary CorpOnto instantiation,
then measures:
  - reasoner (HermiT) classification/closure time
  - competency-question SPARQL latency (median)
at the larger scale, to test whether the sub-30ms query latencies reported in the
paper hold as the dataset grows.

IMPORTANT (honest framing for the paper):
  This is a SENSITIVITY CHECK using SYNTHETIC data, NOT an enterprise-scale
  benchmark and NOT real company data. The scaled individuals are generated
  programmatically to mirror the original distributions. Report it as such.

The script SCALES the existing CorpOnto by cloning its individuals N times with
fresh IRIs, preserving each individual's class and property assertions. This keeps
the ontology's TBox identical and multiplies the ABox ~10x.

Usage:
    python scale_test.py --corponto CorpOnto.owx --factor 10 --out CorpOnto_scaled.owl

Requirements:
    pip install owlready2 rdflib
    (HermiT ships with owlready2; it needs Java installed for the reasoner step.)

Notes:
  - Input is .owx (Protégé OWL/XML). owlready2 loads it directly.
  - Output is written as RDF/XML (.owl) so rdflib can query it.
  - If Java is not available, the reasoner step is skipped with a clear message
    and only SPARQL latencies are measured (still useful).
"""
import argparse
import time
import statistics
import os
import sys


def scale_ontology(onto, factor):
    """Clone every individual `factor-1` extra times with fresh IRIs, preserving
    class membership and property assertions. Returns count of individuals after."""
    from owlready2 import Thing

    originals = list(onto.individuals())
    print(f"  Original individuals: {len(originals)}")

    # Build a name->new-individual map per copy so object properties can be rewired
    with onto:
        for copy_idx in range(1, factor):
            # First pass: create the cloned individuals with their classes
            clones = {}
            for ind in originals:
                cls = ind.is_a[0] if ind.is_a else Thing
                new_name = f"{ind.name}_s{copy_idx}"
                clone = cls(new_name)
                clones[ind.name] = clone
            # Second pass: copy property assertions
            for ind in originals:
                clone = clones[ind.name]
                # Object properties
                for prop in onto.object_properties():
                    for val in prop[ind]:
                        if hasattr(val, 'name') and val.name in clones:
                            # rewire to the cloned target (keep the copy self-contained)
                            prop[clone].append(clones[val.name])
                        else:
                            prop[clone].append(val)
                # Data properties
                for prop in onto.data_properties():
                    for val in prop[ind]:
                        prop[clone].append(val)

    total = len(list(onto.individuals()))
    print(f"  After {factor}x scaling: {total} individuals")
    return total


def run_reasoner(onto):
    """Run HermiT and time it. Returns milliseconds or None if unavailable."""
    from owlready2 import sync_reasoner
    try:
        t0 = time.perf_counter()
        with onto:
            sync_reasoner(infer_property_values=False)
        return (time.perf_counter() - t0) * 1000
    except Exception as e:
        print(f"  Reasoner skipped ({type(e).__name__}: {str(e)[:80]})")
        print("  (HermiT needs Java. Install a JRE to include this measurement.)")
        return None


# The 6 competency questions as SPARQL (same as the paper's CQ1-CQ6)
NS = "http://www.semanticweb.org/anon/ontologies/2025/8/CorpOnto.owl#"
CQ_QUERIES = {
    "CQ1_opportunities_with_tech_proposal": f"""
        PREFIX : <{NS}>
        SELECT DISTINCT ?opp WHERE {{
          ?opp a :Opportunity . ?opp :hasTechnicalProposal ?tp .
          ?tp :technicalProposalNeedsToWorkHourProfile ?whp . ?whp :expectedHours ?h .
        }}""",
    "CQ3_projects_linked_opportunities": f"""
        PREFIX : <{NS}>
        SELECT DISTINCT ?project ?opp WHERE {{
          ?project a :Project . ?project :projectRelatedToOpportunity ?opp .
        }}""",
    "CQ4_employees_knowing_python": f"""
        PREFIX : <{NS}>
        SELECT DISTINCT ?emp WHERE {{
          ?emp a :Employee . ?emp :hasKnowledge ?k . ?k :hasName ?n .
          FILTER(CONTAINS(?n, "Python"))
        }}""",
    "CQ6_projects_margin_above_25": f"""
        PREFIX : <{NS}>
        SELECT DISTINCT ?project ?margin WHERE {{
          ?project a :Project . ?project :hasRealMargin ?margin .
          FILTER(?margin > 0.25)
        }}""",
}


def measure_sparql_latencies(owl_path, repeats=5):
    """Load the (scaled) ontology into rdflib and time each CQ."""
    import rdflib
    g = rdflib.Graph()
    g.parse(owl_path)
    print(f"  Loaded scaled graph: {len(g)} triples")

    results = {}
    for name, q in CQ_QUERIES.items():
        times = []
        nrows = 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            rows = list(g.query(q))
            times.append((time.perf_counter() - t0) * 1000)
            nrows = len(rows)
        results[name] = {"median_ms": statistics.median(times), "rows": nrows}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corponto", required=True, help="CorpOnto.owx (Protégé OWL/XML)")
    ap.add_argument("--factor", type=int, default=10, help="scaling factor (default 10)")
    ap.add_argument("--out", default="CorpOnto_scaled.owl")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    from owlready2 import get_ontology

    print(f"[1/4] Loading {args.corponto} ...")
    onto = get_ontology("file://" + os.path.abspath(args.corponto)).load()

    print(f"[2/4] Scaling {args.factor}x (synthetic, structure-preserving) ...")
    total = scale_ontology(onto, args.factor)

    print(f"[3/4] Saving scaled ontology to {args.out} (RDF/XML) ...")
    onto.save(file=args.out, format="rdfxml")

    print(f"[4/4] Measuring ...")
    reasoner_ms = run_reasoner(onto)
    sparql = measure_sparql_latencies(args.out, args.repeats)

    print("\n" + "=" * 60)
    print("SCALE SENSITIVITY RESULTS (synthetic, ~{}x)".format(args.factor))
    print("=" * 60)
    print(f"  Individuals: {total}")
    if reasoner_ms is not None:
        print(f"  HermiT reasoning/closure: {reasoner_ms:.0f} ms")
    else:
        print(f"  HermiT reasoning/closure: (skipped — no Java)")
    print(f"  Competency-question SPARQL latency (median of {args.repeats}):")
    all_medians = []
    for name, r in sparql.items():
        print(f"    {name}: {r['median_ms']:.2f} ms ({r['rows']} rows)")
        all_medians.append(r["median_ms"])
    print(f"  Overall median CQ latency: {statistics.median(all_medians):.2f} ms")
    print("=" * 60)
    print("\nReport this as a SYNTHETIC sensitivity check, not an enterprise benchmark.")

    # Save JSON
    import json
    out = {
        "factor": args.factor, "individuals": total,
        "reasoner_ms": reasoner_ms,
        "sparql": sparql,
        "overall_median_cq_ms": statistics.median(all_medians),
    }
    json.dump(out, open("scale_test_results.json", "w"), indent=2)
    print("Saved scale_test_results.json")


if __name__ == "__main__":
    main()
