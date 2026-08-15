"""
scripts/diagnose_guard_suppression.py

Decomposes WHY guideline evidence is unreachable for a gold concept.

WHY THIS MATTERS. measure_channel_b_coverage.py reports that the
name-agreement guard suppresses ~74% of raw coverage, but "suppressed" lumps
together four causes with completely different implications:

  node_has_no_rules          -- the node exists but nothing is attached to it.
                                Not a guard issue at all; the KG simply has no
                                statement about that concept.
  name_reject                -- the guard judged the node's name inconsistent
                                with the concept's. CORRECT when it catches the
                                NSTEMI/STEMI class of collision; a FALSE
                                rejection when it is only a lexical mismatch
                                ("sepsis" vs "septic").
  unverified_code_assertion  -- node flagged same_snomed_type_mismatch_not_merged
                                during cleaning.
  boilerplate                -- journal headers and methodology text.

Quoting a coverage figure without this breakdown risks either overstating the
guard's cost (if most suppression is really "no rules exist") or hiding a
miscalibrated guard (if most is lexical false rejection). The sampled
name_reject pairs at the end are there so the distinction can be judged by
looking at real examples rather than argued from the aggregate.

Run:  python3 scripts/diagnose_guard_suppression.py [--triplets DIR] [--sample N]
"""

import argparse
import collections
import csv
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import (  # noqa: E402
    GuidelineIndex,
    is_boilerplate,
    name_agreement_guard,
)

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

GOLD_CANDIDATES = [
    os.path.join(PROJECT_DIR, "data", "snomed-ct-entity-linking-challenge-1.2.0",
                 "train_annotations.csv"),
    os.path.join(PROJECT_DIR, "data", "evaluaiton-dataset",
                 "snomed-ct-entity-linking-challenge-1.2.0", "train_annotations.csv"),
]
TRIPLETS_CANDIDATES = [
    # 2026-08-11 Stage3 Issue1 rule backfill -- see
    # docs/Stage3_Issue1_Rule_Backfill.md. Non-destructive; fallbacks below
    # still work if this dir is ever missing.
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded_rules_added"),
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded"),
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned"),
]


def _first_existing(cands, what):
    for p in cands:
        if os.path.exists(p):
            return p
    raise SystemExit(f"Could not locate {what}. Tried:\n  " + "\n  ".join(cands))


def classify(index, code, fsn, synonyms=None):
    """Best outcome across all nodes carrying this code, plus the reason.

    BEST, not first: a code may carry several nodes, and if ANY of them yields
    usable evidence the concept is covered. Reporting the first node's failure
    would overstate suppression.
    """
    order = {"ACCEPTED": 0, "name_reject": 1, "node_has_no_rules": 2,
             "unverified_code_assertion": 3, "boilerplate": 4, "no_nodes": 5}
    best, best_node = "no_nodes", None
    for node in index.nodes_for_code(code):
        if is_boilerplate(node["name"]):
            reason = "boilerplate"
        elif node.get("quality_flag") == "same_snomed_type_mismatch_not_merged":
            reason = "unverified_code_assertion"
        elif not index.rules_touching(node["uid"]):
            reason = "node_has_no_rules"
        elif name_agreement_guard(fsn, fsn, node["name"],
                                  concept_synonyms=synonyms)[1] == "reject":
            reason = "name_reject"
        else:
            reason = "ACCEPTED"
        if order[reason] < order[best]:
            best, best_node = reason, node
    return best, best_node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", default=None)
    ap.add_argument("--gold", default=None)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--sample", type=int, default=12)
    args = ap.parse_args()

    triplets = args.triplets or _first_existing(TRIPLETS_CANDIDATES, "triplets dir")
    gold = args.gold or _first_existing(GOLD_CANDIDATES, "gold CSV")

    index = GuidelineIndex(triplets)
    rows = list(csv.DictReader(open(gold, encoding="utf-8")))
    freq = collections.Counter(r["concept_id"] for r in rows)
    total = len(rows)

    codes = [c for c in freq if c in index.nodes_by_code]
    print("=" * 78)
    print("GUARD SUPPRESSION DIAGNOSIS (direct-code matches)")
    print("=" * 78)
    print(f"triplets: {triplets}")
    print(f"gold concepts with a direct code match: {len(codes)}")

    conn = duckdb.connect(args.db, read_only=True)
    conn.execute("CREATE TEMP TABLE nc(code VARCHAR)")
    conn.executemany("INSERT INTO nc VALUES (?)", [(c,) for c in codes])
    fsn = {
        str(a): b for a, b in conn.sql(
            "SELECT ac.concept_code, min(ac.concept_name) "
            "FROM nc n JOIN athena_concept ac "
            "  ON ac.concept_code = n.code AND ac.vocabulary_id = 'SNOMED' "
            "GROUP BY 1"
        ).fetchall()
    }
    print(f"resolved FSNs: {len(fsn)}/{len(codes)}")
    syns = collections.defaultdict(list)
    for code, name in conn.sql(
        "SELECT ac.concept_code, s.concept_synonym_name "
        "FROM nc n JOIN athena_concept ac "
        "  ON ac.concept_code = n.code AND ac.vocabulary_id = 'SNOMED' "
        "JOIN athena_concept_synonym s ON s.concept_id = ac.concept_id"
    ).fetchall():
        if name and len(syns[str(code)]) < 25:
            syns[str(code)].append(name)
    print(f"resolved synonyms: {len(syns)} codes")

    why, ann, examples = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    for code in codes:
        f = fsn.get(code, "")
        reason, node = classify(index, code, f, synonyms=syns.get(code))
        why[reason] += 1
        ann[reason] += freq[code]
        if node is not None and len(examples[reason]) < args.sample:
            examples[reason].append((f, node["name"], freq[code]))

    print("\n--- why each concept's evidence was or wasn't usable ---")
    for k, v in why.most_common():
        print(f"  {k:<28} {v:>5} concepts  {ann[k]:>7,} annotations "
              f"({ann[k]/total*100:5.2f}% of gold)")

    print("\n--- sampled name_reject pairs (SNOMED FSN vs guideline node name) ---")
    print("    Judge these directly: a genuine collision is a CORRECT rejection;")
    print("    a mere wording difference means the guard is too tight.")
    for f, n, c in examples.get("name_reject", []):
        print(f"  [{c:>5} ann] {f[:44]!r}")
        print(f"              vs {n[:44]!r}")

    if examples.get("node_has_no_rules"):
        print("\n--- sampled node_has_no_rules (KG mentions the concept, states nothing) ---")
        for f, n, c in examples["node_has_no_rules"][:6]:
            print(f"  [{c:>5} ann] {n[:60]!r}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
