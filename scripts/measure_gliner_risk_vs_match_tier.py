"""scripts/measure_gliner_risk_vs_match_tier.py -- does a confirmed Tier 1/2
match_tier predict correctness EVEN WHEN gliner_confidence is high?

WHY THIS EXISTS. compute_confidence_tier() flags `high_gliner_risk` whenever
gliner_confidence >= HIGH_GLINER_RISK_FLOOR (0.70) -- normalization.py's own
comment says this floor came from evaluation/stage_calibration.py's ECE
measurement of EXTRACTION accuracy (is the span/label right), not
normalization accuracy (does the resolved text map to the right concept).
Those are different questions. This script tests whether they're actually
independent: among HIGH-gliner-confidence entities, is LINKED accuracy
(correct SNOMED concept per scripts/score_gold_recall.py's own methodology)
meaningfully higher for match_tier 1/2 (Exact/Synonym) than for 3/4/Failed?
If yes, a confirmed Tier 1/2 match is real evidence the entity is trustworthy
even under high_gliner_risk, and the tier-gate exemption implemented for
short_token/isupper/alnum_mix (2026-08-12) could reasonably be extended to
also cover high_gliner_risk. If no, the risk signal is genuinely orthogonal
to match_tier and must NOT be overridden by it.

Deliberately reuses load_gold(), attach_snomed_codes(), overlaps() from
score_gold_recall.py rather than reimplementing gold-linkage logic --
that module is the one place in this codebase that's already been checked
against the actual challenge gold set; duplicating it risks a second,
subtly different implementation drifting out of sync (the same failure class
docs/MoLLM_Redesign_Proposal.md and this session's other work has repeatedly
guarded against).

Run on EC2:
    cd ~/clinical_neuro_symbolic_pipeline/code
    source ~/.venv/bin/activate
    python3 scripts/measure_gliner_risk_vs_match_tier.py --note-ids <ids>
    python3 scripts/measure_gliner_risk_vs_match_tier.py --out reports/gliner_risk_report.json

Read-only. No LLM calls, no pipeline run -- run test_pipeline_e2e.py /
score_gold_recall.py's own prerequisites first so extracted_entities /
normalized_entities are populated for the target notes.
"""
import argparse
import csv
import json
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
GOLD_PATH = os.path.join(
    PROJECT_DIR, "code", "evaluaiton-dataset",
    "snomed-ct-entity-linking-challenge-1.2.0", "train_annotations.csv",
)

sys.path.insert(0, os.path.join(PROJECT_DIR, "code", "scripts"))
from score_gold_recall import load_gold, attach_snomed_codes, overlaps  # noqa: E402

HIGH_GLINER_RISK_FLOOR = 0.70  # mirrors src/normalization.py's constant
CONFIRMED_TIERS = {"1 (Exact)", "2 (Synonym)"}


def load_predictions_with_confidence(conn, note_ids):
    """Same join as score_gold_recall.load_predictions(), plus
    n.gliner_confidence -- the one column that script doesn't need but this
    measurement does."""
    rows = conn.execute("""
        SELECT e.note_id, e.orig_start, e.orig_end, e.entity_label,
               e.original_text, e.expanded_text, e.entity_id,
               n.omop_concept_id, n.omop_concept_name, n.omop_vocab,
               n.match_tier, n.gliner_confidence
        FROM extracted_entities e
        JOIN normalized_entities n
          ON n.note_id = e.note_id
         AND n.original_text = e.original_text
         AND n.expanded_text = e.expanded_text
         AND n.gliner_label = e.entity_label
         AND n.is_test = TRUE
        WHERE e.is_test = TRUE
          AND e.note_id IN ({})
          AND (e.below_threshold IS NULL OR e.below_threshold = FALSE)
          AND (e.superseded_by_split IS NULL OR e.superseded_by_split = FALSE)
          AND (e.superseded_by_growth IS NULL OR e.superseded_by_growth = FALSE)
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()
    cols = ["note_id", "orig_start", "orig_end", "entity_label", "original_text",
            "expanded_text", "entity_id", "omop_concept_id", "omop_concept_name",
            "omop_vocab", "match_tier", "gliner_confidence"]
    return [dict(zip(cols, r)) for r in rows]


def bucket(p):
    risk = "high_gliner_risk" if (p["gliner_confidence"] or 0) >= HIGH_GLINER_RISK_FLOOR else "low_gliner_risk"
    tier = "confirmed_tier_1_2" if p["match_tier"] in CONFIRMED_TIERS else "unconfirmed_tier_3_4_0"
    return f"{risk} / {tier}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", help="comma-separated note_ids; default all gold notes present in extracted_entities")
    ap.add_argument("--out", help="write full JSON report here")
    args = ap.parse_args()

    conn = duckdb.connect(DB_PATH, read_only=True)

    if args.note_ids:
        note_ids = args.note_ids.split(",")
    else:
        note_ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test = TRUE").fetchall()]

    if not note_ids:
        print("No is_test=TRUE note_ids found in extracted_entities -- nothing to measure.")
        return 1

    gold = load_gold(GOLD_PATH, note_ids)
    gold_by_note = {}
    for g in gold:
        gold_by_note.setdefault(g["note_id"], []).append(g)

    preds = load_predictions_with_confidence(conn, note_ids)
    attach_snomed_codes(conn, preds)

    buckets = {}  # bucket_name -> {"n": int, "linked_correct": int}
    for p in preds:
        b = bucket(p)
        buckets.setdefault(b, {"n": 0, "linked_correct": 0, "examples_wrong": []})
        buckets[b]["n"] += 1
        note_gold = gold_by_note.get(p["note_id"], [])
        overlapping_gold = [g for g in note_gold
                            if overlaps(p["orig_start"], p["orig_end"], g["start"], g["end"])]
        correct = any(g["concept_id"] == p["snomed_code"] for g in overlapping_gold) if overlapping_gold else None
        if correct is True:
            buckets[b]["linked_correct"] += 1
        elif correct is False and len(buckets[b]["examples_wrong"]) < 5:
            buckets[b]["examples_wrong"].append({
                "text": p["original_text"], "predicted_concept": p["omop_concept_name"],
                "predicted_snomed": p["snomed_code"],
                "gold_concepts": [g["concept_id"] for g in overlapping_gold],
            })
        # correct is None (no overlapping gold at all) -- excluded from the
        # denominator below, same as score_gold_recall's own scoping: no gold
        # label exists to grade against, so it neither confirms nor refutes.

    print("=" * 78)
    print(f"{'BUCKET':<40} {'N (gradable)':>12} {'LINKED ACC':>12}")
    print("-" * 78)
    report = {}

    # Gradable denominator: only rows with >=1 overlapping gold annotation
    # count towards linked_accuracy -- an entity with no gold annotation at
    # all neither confirms nor refutes correctness (same scoping
    # score_gold_recall.py itself uses).
    gradable_counts = {b: 0 for b in buckets}
    for p in preds:
        b = bucket(p)
        note_gold = gold_by_note.get(p["note_id"], [])
        overlapping_gold = [g for g in note_gold
                            if overlaps(p["orig_start"], p["orig_end"], g["start"], g["end"])]
        if overlapping_gold:
            gradable_counts[b] += 1

    for b, d in sorted(buckets.items()):
        gradable = gradable_counts[b]
        acc = (d["linked_correct"] / gradable) if gradable else None
        print(f"{b:<40} {gradable:>12} {f'{acc:.1%}' if acc is not None else 'n/a':>12}")
        report[b] = {"n_total_in_bucket": d["n"], "n_gradable": gradable,
                     "linked_correct": d["linked_correct"],
                     "linked_accuracy": acc, "example_errors": d["examples_wrong"]}
    print("=" * 78)

    hi_confirmed = report.get("high_gliner_risk / confirmed_tier_1_2", {}).get("linked_accuracy")
    hi_unconfirmed = report.get("high_gliner_risk / unconfirmed_tier_3_4_0", {}).get("linked_accuracy")
    print("\nKey comparison (does match_tier rescue high_gliner_risk entities?):")
    print(f"  high_gliner_risk + confirmed Tier 1/2:     {hi_confirmed}")
    print(f"  high_gliner_risk + unconfirmed Tier 3/4/0: {hi_unconfirmed}")
    if hi_confirmed is not None and hi_unconfirmed is not None:
        if hi_confirmed - hi_unconfirmed >= 0.20:
            print("  -> Meaningful gap. Extending the tier-gate to also exempt "
                  "high_gliner_risk under confirmed Tier 1/2 has real support.")
        else:
            print("  -> No meaningful gap. high_gliner_risk looks independent of "
                  "match_tier -- do NOT extend the tier-gate on this evidence.")
    else:
        print("  -> Not enough gradable examples in one or both buckets yet to conclude anything.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nFull report written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
