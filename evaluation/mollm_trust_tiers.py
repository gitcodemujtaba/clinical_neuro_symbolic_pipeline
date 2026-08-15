"""
evaluation/mollm_trust_tiers.py — tags every ACCEPTED (non-HITL) resolution-mode
mollm_decisions row with a trust tier, so a batch of MoLLM-resolved entities
can be safely repurposed as pipeline feedback data without pseudo-labeling on
itself. Implements the 4-tier framework this project settled on:

  TIER A -- gold-annotation match, OR human-audited (see --human-audit-csv).
            Equally trustworthy either way: an independent ground truth
            confirmed the verdict. Safe basis for dictionary/synonym
            promotion (unreviewed, pipeline-wide blast radius) as well as
            MoLLMCalibrator training data and aggregate recalibration.
  TIER B -- not gold-checkable, but grounding_basis == "guideline_rule" AND
            citation_verified == True: the verdict cited a real KG rule and
            that citation passed containment verification (verify_citations()
            in src/mollm_ensemble.py). This rules out a FABRICATED citation,
            not a shared misreading of a real one -- de-risked, not
            independently verified. Safe for MoLLMCalibrator training data;
            NOT safe for dictionary/synonym promotion.
  TIER C -- ensemble agreement alone, grounding_basis == "model_terminology_
            knowledge" or no real evidence retrieved. The weakest tier:
            circular (the two models' shared training-data belief is the only
            reason it's here), not safe for feedback without independent
            review. Usable only for aggregate d_anno/tier_reasons statistical
            recalibration, per the trust-tier framework's own gating.
  REJECTED_GOLD_MISMATCH -- not a trust tier, a distinct, louder category:
            the decision WAS gold-checkable and gold DISAGREED with the
            accepted verdict. This is evidence Stage 3 got an accepted
            (non-HITL) decision wrong, not just "insufficiently verified" --
            reported separately so it is never silently folded into Tier C's
            count, which would understate how often accepted decisions are
            actually wrong.

SCOPE: only mollm_routing_decision IN ('AUTO_VALIDATED', 'MOLLM_RESOLVED') --
i.e. decisions Stage 3 actually accepted -- are tiered at all. HITL_REQUIRED
rows were never resolved by MoLLM in the first place; they are counted and
excluded, not tiered, since there is no "verified label" to reuse yet.

HUMAN-AUDIT PATH. No `human_audit_result`-style column exists anywhere in
mollm_decisions today (checked directly against store_decision()'s DDL,
2026-08-13) -- the trust-tier framework's "human-audited" arm of Tier A has
no persisted data source yet. --human-audit-csv lets a reviewer supply one
externally (mollm_call_id,human_verdict with human_verdict in
correct/incorrect) without waiting on a schema change; rows it covers are
tiered from it with priority over the gold check (a human reviewer looking at
the actual note is a stronger signal than an automatic gold-span match, and
should be able to override one, e.g. when gold itself is wrong or ambiguous).

GOLD GRADING IS REUSED, NOT REIMPLEMENTED. official_verdict()/grade() are
imported directly from evaluation.cal_eval -- the exact same function that
computes cal_eval.py's own resolution-mode accuracy number, so a row this
script calls Tier A is graded identically to how cal_eval.py's ECE/accuracy
figures already treat it. Two different "was this correct" answers for the
same decision would be a bug, not a modeling choice.

Run:
  python3 evaluation/mollm_trust_tiers.py --out reports/trust_tiers.csv
  python3 evaluation/mollm_trust_tiers.py --note-ids 10043750-DS-6 --human-audit-csv audit.csv --out reports/trust_tiers.csv
"""

import argparse
import collections
import csv
import json
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

sys.path.insert(0, PROJECT_DIR)

from src.retrieval import VocabularyRetriever  # noqa: E402
from scripts.score_gold_recall import load_gold, _first_existing, GOLD_CANDIDATES  # noqa: E402
from evaluation.cal_eval import (  # noqa: E402
    load_gradable_decisions, load_ungradable_counts, grade,
)
from evaluation.splits import (  # noqa: E402
    add_split_args, assert_no_contamination, resolve_note_ids,
)

# route()'s two non-HITL routing decisions (src/mollm_ensemble.py: INGESTION_AUTO
# = "AUTO_VALIDATED", INGESTION_RESOLVED = "MOLLM_RESOLVED"). HITL_REQUIRED rows
# are excluded from tiering entirely -- see module docstring "SCOPE".
ACCEPTED_ROUTING = {"AUTO_VALIDATED", "MOLLM_RESOLVED"}

TIER_A = "A_gold_or_human_verified"
TIER_B = "B_cited_rule_verified"
TIER_C = "C_agreement_only"
REJECTED = "REJECTED_GOLD_MISMATCH"


def load_human_audit(path):
    """--human-audit-csv: mollm_call_id,human_verdict (correct/incorrect).
    Returns {mollm_call_id: "correct"|"incorrect"}. Unknown/blank
    human_verdict values are ignored (not silently coerced to a guess) so a
    typo in the audit sheet fails to apply rather than mis-tiering a row.
    """
    if not path:
        return {}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            v = (r.get("human_verdict") or "").strip().lower()
            if v in ("correct", "incorrect"):
                out[r["mollm_call_id"]] = v
    return out


def classify(decision, outcome, human_audit):
    """Returns (tier, basis) for one ACCEPTED decision.

    `outcome` is cal_eval.grade()'s own "correct" / "incorrect" / None for
    this decision -- computed once by the caller and passed in rather than
    re-derived, so tiering can never silently disagree with the accuracy
    number cal_eval.py reports for the same row.
    """
    human = human_audit.get(decision["mollm_call_id"])
    if human == "correct":
        return TIER_A, "human_audit"
    if human == "incorrect":
        return REJECTED, "human_audit"

    if outcome == "correct":
        return TIER_A, "gold_match"
    if outcome == "incorrect":
        return REJECTED, "gold_mismatch"

    # outcome is None: gold could not check this decision (no overlapping
    # gold span, model disagreement never reaches here since disagreement
    # never routes to ACCEPTED_ROUTING in the first place, etc -- see
    # cal_eval.grade()'s own reason codes for the exact cause per row,
    # carried through as gold_reason below for the CSV).
    if decision.get("grounding_basis") == "guideline_rule" and decision.get("citation_verified") is True:
        return TIER_B, "cited_rule_plus_verified_citation"
    return TIER_C, "ensemble_agreement_only"


def build_rows(decisions, gold_by_note, vocab, human_audit):
    """One output row per ACCEPTED decision, tiered. Returns (rows, skipped_hitl_count)."""
    rows = []
    skipped_hitl = 0
    for d in decisions:
        routing = d.get("mollm_routing_decision")
        if routing not in ACCEPTED_ROUTING:
            skipped_hitl += 1
            continue
        outcome, gold_reason = grade(d, gold_by_note, vocab)
        tier, basis = classify(d, outcome, human_audit)
        rows.append({
            "mollm_call_id": d["mollm_call_id"],
            "entity_id": d["entity_id"],
            "note_id": d["note_id"],
            "original_text": d["original_text"],
            "entity_label": d["entity_label"],
            "mollm_routing_decision": routing,
            "composite_confidence": d.get("composite_confidence"),
            "grounding_basis": d.get("grounding_basis"),
            "citation_verified": d.get("citation_verified"),
            "gold_outcome": outcome,
            "gold_reason": gold_reason,
            "trust_tier": tier,
            "tier_basis": basis,
        })
    return rows, skipped_hitl


def print_report(rows, skipped_hitl, ungraded_modes, note_ids):
    print("=" * 78)
    print("MoLLM TRUST-TIER CLASSIFICATION — ACCEPTED (non-HITL) resolution decisions")
    print("=" * 78)
    print(f"\nnotes: {note_ids}")
    print(f"ACCEPTED decisions tiered: {len(rows)}")
    print(f"excluded (mollm_routing_decision == HITL_REQUIRED, never resolved): {skipped_hitl}")
    if ungraded_modes:
        print(f"non-resolution-mode decisions for these notes (not covered by this "
              f"script -- no gold concept-identity label exists for them):")
        for mode, n in ungraded_modes.items():
            print(f"    {mode}: {n}")

    by_tier = collections.Counter(r["trust_tier"] for r in rows)
    print(f"\n--- Tier breakdown ---")
    for tier in (TIER_A, TIER_B, TIER_C, REJECTED):
        n = by_tier.get(tier, 0)
        pct = f"{n / len(rows) * 100:.1f}%" if rows else "-"
        print(f"  {tier:<28} {n:>5}  ({pct})")

    by_basis = collections.Counter(r["tier_basis"] for r in rows)
    print(f"\n--- Basis breakdown ---")
    for basis, n in sorted(by_basis.items()):
        print(f"  {basis:<32} {n:>5}")

    if by_tier.get(REJECTED, 0):
        print(f"\n--- Sample REJECTED_GOLD_MISMATCH rows (accepted decision, gold disagreed) ---")
        shown = 0
        for r in rows:
            if r["trust_tier"] == REJECTED and shown < 10:
                print(f"  [{r['note_id']}] '{r['original_text']}' ({r['entity_label']}) "
                      f"routing={r['mollm_routing_decision']} basis={r['tier_basis']}")
                shown += 1

    print(f"\nGATING REMINDER (per this project's trust-tier framework):")
    print(f"  Tier A only        -> safe for dictionary/synonym promotion (unreviewed, pipeline-wide)")
    print(f"  Tier A + B         -> safe for MoLLMCalibrator training data")
    print(f"  Tier A + B + C     -> safe only for aggregate d_anno/tier_reasons statistical recalibration")
    print(f"  REJECTED_GOLD_MISMATCH -> never feedback data; a quality signal on Stage 3 itself")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None,
                     help="Comma-separated note_ids. Default: every note_id with "
                          "is_test=TRUE resolution-mode rows in mollm_decisions.")
    ap.add_argument("--gold", default=None, help="path to train_annotations.csv")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--human-audit-csv", default=None,
                     help="optional CSV: mollm_call_id,human_verdict "
                          "(correct/incorrect) -- overrides gold check, see "
                          "module docstring 'HUMAN-AUDIT PATH'.")
    ap.add_argument("--out", default=None, help="write per-decision tier CSV here")
    # 2026-08-13 (docs/2026-08-13_Code_Improvement_Proposals.md P0.1). This was
    # the last evaluation script still defaulting to "every note_id in the
    # database". Its 47.6% REJECTED_GOLD_MISMATCH headline is the closest thing
    # this project has to a false-deflection measurement, which makes it
    # exactly the number that must not be computed over an ad hoc note pool.
    add_split_args(ap, default="val")
    args = ap.parse_args()

    gold_path = args.gold or _first_existing(GOLD_CANDIDATES, "gold annotations CSV")
    human_audit = load_human_audit(args.human_audit_csv)

    conn = duckdb.connect(args.db, read_only=True)
    try:
        note_ids, split_prov = resolve_note_ids(
            args, conn=conn, table="mollm_decisions",
            where="is_test = TRUE AND mode = 'resolution'")
        assert_no_contamination(conn)
        if not note_ids:
            raise SystemExit(
                f"No is_test=TRUE resolution-mode rows in mollm_decisions for "
                f"split '{split_prov['split']}'.\n"
                "Run scripts/run_stage3_batch.py over that split, or pass "
                "--split all to tier whatever is present\n(a development "
                "measurement, not a reportable one).")

        print(f"gold:  {gold_path}")
        print(f"db:    {args.db}")
        print(f"notes: {note_ids}")

        gold_rows = load_gold(gold_path, note_ids)
        gold_by_note = collections.defaultdict(list)
        for g in gold_rows:
            gold_by_note[g["note_id"]].append(g)

        decisions = load_gradable_decisions(conn, note_ids)
        ungraded_modes = load_ungradable_counts(conn, note_ids)
        vocab = VocabularyRetriever(conn)

        rows, skipped_hitl = build_rows(decisions, gold_by_note, vocab, human_audit)
    finally:
        conn.close()

    print_report(rows, skipped_hitl, ungraded_modes, note_ids)

    if args.out:
        fieldnames = ["mollm_call_id", "entity_id", "note_id", "original_text",
                      "entity_label", "mollm_routing_decision", "composite_confidence",
                      "grounding_basis", "citation_verified", "gold_outcome",
                      "gold_reason", "trust_tier", "tier_basis"]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} row(s) to {args.out}")


if __name__ == "__main__":
    main()
