"""
scripts/measure_channel_b_coverage.py

Measures how far SNOMED hierarchy traversal (Channel B) lifts guideline
coverage above the direct-code baseline, across the full 272-note gold set.
Read-only, no LLM calls, no pipeline run.

WHY THIS IS THE FIRST EXPERIMENT TO RUN.
docs/MoLLM_Stage3_Retrieval_Design.md S8 names this as "the largest single
uncertainty in this design". The measured direct-code baseline is 10.80% of
gold annotations (the curated corpus holds only 447 distinct SNOMED codes), so
~87.5% of entities have no directly-matching guideline rule. Channel B is the
mechanism that is supposed to close that gap by connecting a specific entity
("Stage 2 AKI") to a rule stated more generally ("Acute Kidney Injury"). Until
this number exists, the whole guideline-grounding claim rests on 10.8% and an
assumption.

WHAT IS MEASURED, AND WHAT IS NOT.
This runs over the GOLD concept_ids, not over pipeline output. That is
deliberate and it measures a different thing:

  * Using gold concept_ids isolates RETRIEVAL from extraction and normalisation
    error. It answers "given a correctly-identified concept, can we reach a
    guideline rule for it?" -- which is the property of the KG and the
    traversal, and the thing this design decision hinges on.
  * The end-to-end number will be LOWER, because Stage 2 will mis-normalise
    some entities. That is a separate measurement (run the pipeline over the
    272 notes and repeat this against normalized_entities), and conflating the
    two would make it impossible to tell a retrieval problem from a
    normalisation problem.

So: report this as an upper bound on guideline reachability, explicitly.

Two figures are produced for each channel, and the gap between them matters:
  RAW      -- the concept (or an ancestor) has guideline rules attached.
  GUARDED  -- those rules survive name_agreement_guard(), i.e. the guideline
              node's name actually agrees with the gold span. Given that 47% of
              multi-name codes in this corpus attach clinically unrelated names
              to one code, RAW alone would overstate real coverage. GUARDED is
              the number to quote.

Run:  python3 scripts/measure_channel_b_coverage.py [--out report.json]
"""

import argparse
import collections
import csv
import json
import os
import sys
import time

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import (  # noqa: E402
    HIERARCHY_STOP_CODES,
    MAX_HIERARCHY_HOPS,
    GuidelineIndex,
    is_boilerplate,
    name_agreement_guard,
)

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

# PATHS ARE DISCOVERED, NOT HARDCODED. The deployed tree on EC2 does not match
# the repository layout -- the repo nests the gold set under
# `data/evaluaiton-dataset/`, the EC2 copy has it directly under `data/` -- and
# a single hardcoded path fails with a bare FileNotFoundError that says nothing
# about which alternatives were tried. Each candidate list below is ordered
# most-specific first.
GOLD_CANDIDATES = [
    os.path.join(PROJECT_DIR, "data", "evaluaiton-dataset",
                 "snomed-ct-entity-linking-challenge-1.2.0", "train_annotations.csv"),
    os.path.join(PROJECT_DIR, "data", "snomed-ct-entity-linking-challenge-1.2.0",
                 "train_annotations.csv"),
    os.path.join(PROJECT_DIR, "data", "evaluation-dataset",
                 "snomed-ct-entity-linking-challenge-1.2.0", "train_annotations.csv"),
]

# `_cleaned_grounded` first: if the SNOMED/ICD10 grounding backfill has been
# run, that corpus has codes on nodes that were `N/A` in `_cleaned`, and more
# grounded nodes means more reachable guideline rules. Measuring against the
# ungrounded corpus when a grounded one exists would understate coverage.
# Overridable with --triplets so the two can be compared directly.
TRIPLETS_CANDIDATES = [
    # 2026-08-11 Stage3 Issue1 rule backfill -- see
    # docs/Stage3_Issue1_Rule_Backfill.md. Non-destructive; fallbacks below
    # still work if this dir is ever missing.
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded_rules_added"),
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded"),
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned"),
]


def _first_existing(candidates, what):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise SystemExit(
        f"Could not locate {what}. Tried:\n  " + "\n  ".join(candidates)
        + "\nPass an explicit path, or check the deployed data/ layout."
    )


def load_gold(path):
    """Gold annotations, plus a representative surface form per concept_id.

    The span text is needed for the name-agreement guard. The MOST FREQUENT
    span for each concept is used rather than the first encountered, so the
    guard is evaluated against the phrasing that actually dominates the corpus
    rather than an incidental one-off.
    """
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    freq = collections.Counter(r["concept_id"] for r in rows)
    spans = collections.defaultdict(collections.Counter)
    for r in rows:
        spans[r["concept_id"]][(r["span"] or "").strip().lower()] += 1
    rep = {cid: c.most_common(1)[0][0] for cid, c in spans.items()}
    return rows, freq, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write full JSON report here")
    ap.add_argument("--hops", type=int, default=MAX_HIERARCHY_HOPS)
    ap.add_argument("--triplets", default=None,
                    help="guideline corpus dir (default: prefer *_cleaned_grounded)")
    ap.add_argument("--gold", default=None, help="path to train_annotations.csv")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    print("=" * 78)
    print("CHANNEL B COVERAGE MEASUREMENT")
    print("=" * 78)

    triplets = args.triplets or _first_existing(TRIPLETS_CANDIDATES, "guideline triplets dir")
    gold_path = args.gold or _first_existing(GOLD_CANDIDATES, "gold annotations CSV")
    print(f"\ntriplets: {triplets}")
    print(f"gold:     {gold_path}")

    index = GuidelineIndex(triplets)
    guideline_codes = set(index.nodes_by_code)
    ungrounded = index.stats["nodes"] - index.stats["grounded_nodes"]
    print(f"\nguideline KG: {index.stats['nodes']} nodes "
          f"({index.stats['grounded_nodes']} grounded, {ungrounded} ungrounded), "
          f"{index.stats['rules']} rules, {len(guideline_codes)} distinct SNOMED codes")

    rows, freq, rep_span = load_gold(gold_path)
    total_ann = len(rows)
    gold_codes = set(freq)
    print(f"gold set: {total_ann:,} annotations, {len(gold_codes):,} distinct concept_ids")

    conn = duckdb.connect(args.db, read_only=True)

    # ---------------------------------------------------------- Channel A
    direct = gold_codes & guideline_codes
    ann_direct = sum(freq[c] for c in direct)
    print(f"\n--- CHANNEL A (direct code match) ---")
    print(f"  concepts: {len(direct)}/{len(gold_codes)}")
    print(f"  annotations: {ann_direct:,}/{total_ann:,} = {ann_direct/total_ann*100:.2f}%")

    # ---------------------------------------------------------- Channel B
    # ONE bulk query rather than 6,595 per-concept lookups. The ancestor table
    # has 78.4M rows; issuing thousands of separate queries against it would
    # turn a seconds-long measurement into a very long one for no extra
    # information.
    print(f"\n--- CHANNEL B (hierarchy, up to {args.hops} hops) ---")
    t0 = time.time()
    conn.execute("CREATE TEMP TABLE gold_codes(code VARCHAR)")
    conn.executemany("INSERT INTO gold_codes VALUES (?)", [(c,) for c in gold_codes])
    conn.execute("CREATE TEMP TABLE gl_codes(code VARCHAR)")
    conn.executemany("INSERT INTO gl_codes VALUES (?)", [(c,) for c in guideline_codes])

    anc_rows = conn.sql(f"""
        SELECT g.code AS gold_code, anc.concept_code AS anc_code,
               min(a.min_levels_of_separation) AS hops
        FROM gold_codes g
        JOIN athena_concept d
          ON d.concept_code = g.code AND d.vocabulary_id = 'SNOMED'
        JOIN athena_concept_ancestor a
          ON a.descendant_concept_id = d.concept_id
        JOIN athena_concept anc
          ON anc.concept_id = a.ancestor_concept_id AND anc.vocabulary_id = 'SNOMED'
        JOIN gl_codes gl ON gl.code = anc.concept_code
        WHERE a.min_levels_of_separation BETWEEN 1 AND {args.hops}
        GROUP BY 1, 2
    """).fetchall()
    print(f"  [bulk ancestor join: {time.time()-t0:.1f}s, {len(anc_rows):,} "
          f"(gold, ancestor) pairs]")

    # Stop codes are excluded here rather than in SQL so the exclusion uses the
    # SAME list src/retrieval.py traverses with -- one definition, not two that
    # can drift.
    by_gold = collections.defaultdict(list)
    for gold_code, anc_code, hops in anc_rows:
        if str(anc_code) in HIERARCHY_STOP_CODES:
            continue
        by_gold[str(gold_code)].append((str(anc_code), int(hops)))

    hier_only = set(by_gold) - direct
    ann_hier_only = sum(freq[c] for c in hier_only)
    combined = direct | set(by_gold)
    ann_combined = sum(freq[c] for c in combined)

    print(f"  concepts reachable ONLY via hierarchy: {len(hier_only)}")
    print(f"  annotations added: {ann_hier_only:,} = {ann_hier_only/total_ann*100:.2f}%")
    print(f"  hop distribution (nearest guideline ancestor per concept):")
    nearest = collections.Counter(
        min(h for _, h in v) for c, v in by_gold.items() if c in hier_only)
    for h in sorted(nearest):
        n_ann = sum(freq[c] for c in hier_only
                    if min(x[1] for x in by_gold[c]) == h)
        print(f"    {h} hop(s): {nearest[h]:>4} concepts, {n_ann:>6,} annotations")

    print(f"\n--- COMBINED A + B (RAW, before name-agreement guard) ---")
    print(f"  concepts: {len(combined)}/{len(gold_codes)}")
    print(f"  annotations: {ann_combined:,}/{total_ann:,} = {ann_combined/total_ann*100:.2f}%")
    print(f"  >>> lift over direct-code baseline: "
          f"{(ann_combined-ann_direct)/total_ann*100:+.2f} percentage points")

    # ------------------------------------------------------------- guarded
    # The number to quote. RAW counts a concept as covered if ANY guideline node
    # carries its code (or an ancestor's); GUARDED additionally requires the
    # node's NAME to agree with the gold span, which is what stops the
    # NSTEMI/STEMI class of collision from being counted as coverage.
    print(f"\n--- GUARDED (name_agreement_guard applied) ---")
    # FSNs for every code the guard will need, fetched in ONE query. Channel B
    # guards each node against ITS ANCESTOR's name, not the entity's -- see
    # GroundingRetriever.channel_b_hierarchy for why. Fetching per-code inside
    # the loop would issue thousands of round-trips for data that is a single
    # join away.
    need = set(combined) | {a for v in by_gold.values() for a, _ in v}
    conn.execute("CREATE TEMP TABLE need_codes(code VARCHAR)")
    conn.executemany("INSERT INTO need_codes VALUES (?)", [(c,) for c in need])
    fsn = {str(c): n for c, n in conn.sql("""
        SELECT ac.concept_code, min(ac.concept_name)
        FROM need_codes n
        JOIN athena_concept ac
          ON ac.concept_code = n.code AND ac.vocabulary_id = 'SNOMED'
        GROUP BY 1
    """).fetchall()}
    print(f"  [resolved FSNs for {len(fsn)}/{len(need)} codes]")
    # Synonyms in bulk. The guard now scores against the concept's whole name
    # set, not just its FSN -- 'WBC' is a registered synonym of 'White blood
    # cell count' and was being rejected as a mismatch. Fetched in one query
    # for the same reason the FSNs are.
    conn.execute("CREATE TEMP TABLE syn_codes(code VARCHAR)")
    conn.executemany("INSERT INTO syn_codes VALUES (?)", [(c,) for c in need])
    syns = collections.defaultdict(list)
    for code, name in conn.sql("""
        SELECT ac.concept_code, s.concept_synonym_name
        FROM syn_codes n
        JOIN athena_concept ac
          ON ac.concept_code = n.code AND ac.vocabulary_id = 'SNOMED'
        JOIN athena_concept_synonym s ON s.concept_id = ac.concept_id
    """).fetchall():
        if name and len(syns[str(code)]) < 25:
            syns[str(code)].append(name)
    print(f"  [resolved synonyms for {len(syns)} codes]")


    guarded_direct, guarded_hier = set(), set()
    for code in combined:
        span = rep_span.get(code, "")
        sources = [(code, 0)] if code in direct else []
        sources += by_gold.get(code, [])
        for src_code, hops in sources:
            ok = False
            # Direct match: compare against the ENTITY (span + its own FSN).
            # Hierarchy match: compare against the ANCESTOR's FSN, because the
            # ancestor's name is SUPPOSED to differ from the entity's -- that
            # difference is the generalisation, not a collision.
            if hops == 0:
                left, right = span, fsn.get(src_code, span)
            else:
                anc_fsn = fsn.get(src_code, "")
                left, right = anc_fsn, anc_fsn
            for node in index.nodes_for_code(src_code):
                if is_boilerplate(node["name"]):
                    continue
                if node.get("quality_flag") == "same_snomed_type_mismatch_not_merged":
                    continue
                if not index.rules_touching(node["uid"]):
                    continue
                _, status = name_agreement_guard(
                    left, right, node["name"], concept_synonyms=syns.get(src_code))
                if status != "reject":
                    ok = True
                    break
            if ok:
                (guarded_direct if hops == 0 else guarded_hier).add(code)
                break

    guarded = guarded_direct | guarded_hier
    ann_guarded = sum(freq[c] for c in guarded)
    ann_gd = sum(freq[c] for c in guarded_direct)
    print(f"  via direct code:   {len(guarded_direct):>5} concepts, {ann_gd:>7,} annotations "
          f"= {ann_gd/total_ann*100:.2f}%")
    print(f"  via hierarchy:     {len(guarded_hier):>5} concepts, "
          f"{ann_guarded-ann_gd:>7,} annotations = {(ann_guarded-ann_gd)/total_ann*100:.2f}%")
    print(f"  >>> TOTAL GUARDED: {len(guarded):>5} concepts, {ann_guarded:>7,} annotations "
          f"= {ann_guarded/total_ann*100:.2f}%")
    print(f"  suppressed by guard: {ann_combined-ann_guarded:,} annotations "
          f"({(ann_combined-ann_guarded)/max(1,ann_combined)*100:.1f}% of raw coverage)")

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print(f"  direct-code baseline ......... {ann_direct/total_ann*100:6.2f}%")
    print(f"  + hierarchy (raw) ............ {ann_combined/total_ann*100:6.2f}%")
    print(f"  + hierarchy (guarded) ........ {ann_guarded/total_ann*100:6.2f}%   <- quote this")
    print(f"  no guideline evidence ........ {100-ann_guarded/total_ann*100:6.2f}%")
    print("\n  Upper bound on guideline reachability: measured over GOLD concept_ids,")
    print("  so it excludes extraction and normalisation error. The end-to-end")
    print("  figure will be lower; measure it separately against normalized_entities.")

    if args.out:
        json.dump({
            "total_annotations": total_ann,
            "distinct_concepts": len(gold_codes),
            "guideline_codes": len(guideline_codes),
            "channel_a": {"concepts": len(direct), "annotations": ann_direct,
                          "pct": ann_direct / total_ann * 100},
            "channel_b_only": {"concepts": len(hier_only), "annotations": ann_hier_only,
                               "pct": ann_hier_only / total_ann * 100},
            "combined_raw": {"concepts": len(combined), "annotations": ann_combined,
                             "pct": ann_combined / total_ann * 100},
            "guarded": {"concepts": len(guarded), "annotations": ann_guarded,
                        "pct": ann_guarded / total_ann * 100,
                        "via_direct": len(guarded_direct), "via_hierarchy": len(guarded_hier)},
            "hop_distribution": {str(k): v for k, v in sorted(nearest.items())},
            "max_hops": args.hops,
            "triplets_dir": triplets,
            "grounded_nodes": index.stats["grounded_nodes"],
            "ungrounded_nodes": ungrounded,
        }, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
