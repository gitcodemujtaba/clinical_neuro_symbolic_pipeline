"""
evaluation/stage1_disambiguation_eval.py — Stage 1 (preprocessing) reliability:
does abbreviation-expansion disambiguation actually pick the right meaning?

WHY THIS IS NOT AN ECE SCRIPT. evaluation/stage2a_cal_eval.py (GLiNER
confidence) and evaluation/stage2b_cal_eval.py (SapBERT similarity, Tier 3)
both grade a CONTINUOUS, per-decision probability against empirical accuracy
-- that is what Expected Calibration Error measures (Guo et al. 2017).
Stage 1's abbreviation-expansion tiebreak (src/preprocessing.py's
expand_text_and_track_offsets()) is deterministic rule-based logic, not a
probabilistic model: it has no confidence score to bin, only a categorical
`selection_basis` recording WHICH of three tiebreaks fired --
"numeric_context:<kind>", "omop_groundability", or "alphabetical_default"
(the last being "no real tiebreak resolved it, arbitrary default"). ECE does
not apply to a method that never claimed a probability in the first place.

The honest analogous question, asked the same way evaluation/stage2b_cal_eval.py
asks it for match_tier ("does 'Exact' actually mean near-100% right"): does
'selection_basis' actually mean what its name implies -- is numeric_context a
more reliable tiebreak than omop_groundability, and is alphabetical_default
(no real tiebreak fired) measurably worse than either, as its "we're guessing"
framing (src/preprocessing.py's _select_by_groundability() docstring) would
predict? That is a discrete reliability table, the same shape
evaluation/stage2b_cal_eval.py already uses for match_tier.

CORRECTNESS SIGNAL: Stage 1 does not itself produce a gradable label -- an
abbreviation expansion is only "right" or "wrong" via its DOWNSTREAM effect,
i.e. whether the entity's final Stage 2b OMOP concept matches gold. This is
the same indirection evaluation/stage2b_cal_eval.py uses to grade Stage 2b
against gold and cal_eval.py uses to grade MoLLM -- correctness always means
"the concept ultimately assigned to this span matches an overlapping gold
annotation's concept_id", regardless of which stage is under the microscope.

PREREQUISITE: `selection_basis` was computed in src/preprocessing.py from the
start but never persisted to extracted_entities until 2026-08-13 (see
src/entity_extraction.py's store_entities()/ensure_extracted_entities_table()
-- additive ALTER TABLE, no migration risk to existing rows). Existing rows
extracted before that patch will have selection_basis = NULL even for
genuinely ambiguous abbreviations; re-run extraction (test_pipeline_e2e.py or
the normal ingestion path) on any note you want graded here AFTER syncing
that patch, or this script will simply report those notes have 0 ambiguous
rows to grade (reported explicitly, never silently).

Run:
  python3 evaluation/stage1_disambiguation_eval.py
  python3 evaluation/stage1_disambiguation_eval.py --note-ids 10060142-DS-9,10097089-DS-8 --out stage1_eval.json
"""

import argparse
import collections
import csv
import json
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

sys.path.insert(0, PROJECT_DIR)

from src.retrieval import VocabularyRetriever  # noqa: E402
from evaluation.splits import (  # noqa: E402
    add_split_args, assert_no_contamination, resolve_note_ids,
)
from scripts.score_gold_recall import (  # noqa: E402
    load_gold, overlaps, _first_existing, GOLD_CANDIDATES,
)


def load_stage1_predictions(conn, note_ids):
    """Same accepted-row shape as scripts/score_gold_recall.py's
    load_predictions(), plus expansion_ambiguous and selection_basis --
    kept local to this script rather than widening a stable, widely-reused
    function's column list for two fields only this script needs.
    """
    rows = conn.execute("""
        SELECT e.note_id, e.orig_start, e.orig_end, e.entity_label,
               e.original_text, e.expanded_text, e.entity_id,
               e.expansion_ambiguous, e.selection_basis,
               n.omop_concept_id, n.omop_concept_name, n.omop_vocab, n.match_tier
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
            "expanded_text", "entity_id", "expansion_ambiguous", "selection_basis",
            "omop_concept_id", "omop_concept_name", "omop_vocab", "match_tier"]
    return [dict(zip(cols, r)) for r in rows]


def attach_snomed_and_grade(conn, predictions, gold_by_note):
    """Mutates predictions in place: adds snomed_code and stage_correct
    (True/False/None -- None means no overlapping gold span, i.e. ungradable,
    same convention as evaluation/stage2b_cal_eval.py)."""
    vocab = VocabularyRetriever(conn)
    for p in predictions:
        cid = p["omop_concept_id"]
        p["snomed_code"] = vocab.snomed_code_for_concept(cid) if cid is not None else None

        gold = gold_by_note.get(p["note_id"], [])
        overlapping_gold = [g for g in gold
                            if overlaps(p["orig_start"], p["orig_end"], g["start"], g["end"])]
        if not overlapping_gold:
            p["stage_correct"] = None
            continue
        gold_concept_ids = {g["concept_id"] for g in overlapping_gold}
        p["stage_correct"] = p["snomed_code"] in gold_concept_ids if p["snomed_code"] else False
    return predictions


def _basis_bucket(selection_basis):
    """Groups "numeric_context:<kind>" values under one "numeric_context"
    bucket for the headline table (there are several <kind>s -- dose,
    lab_value, etc. -- each individually too small to read a rate off), while
    the raw per-kind counts are still reported separately so nothing is
    hidden, only summarized."""
    if selection_basis is None:
        return None
    if selection_basis.startswith("numeric_context"):
        return "numeric_context"
    return selection_basis


def reliability_by_basis(predictions):
    """Discrete n/correct/accuracy per selection_basis bucket, over
    predictions with a determinable stage_correct AND a non-null
    selection_basis (i.e. entities where Stage 1 actually had to
    disambiguate). Also returns the raw (un-bucketed) breakdown and the
    ambiguous-vs-not comparison."""
    by_bucket = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    by_raw = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    ambiguous_totals = {"n": 0, "correct": 0}
    unambiguous_totals = {"n": 0, "correct": 0}

    for p in predictions:
        if p["stage_correct"] is None:
            continue
        basis = p["selection_basis"]
        if basis:
            b = by_bucket[_basis_bucket(basis)]
            b["n"] += 1
            b["correct"] += int(p["stage_correct"])
            r = by_raw[basis]
            r["n"] += 1
            r["correct"] += int(p["stage_correct"])
            ambiguous_totals["n"] += 1
            ambiguous_totals["correct"] += int(p["stage_correct"])
        else:
            unambiguous_totals["n"] += 1
            unambiguous_totals["correct"] += int(p["stage_correct"])

    def _acc(d):
        return {**d, "accuracy": round(d["correct"] / d["n"], 4) if d["n"] else None}

    return (
        {k: _acc(v) for k, v in by_bucket.items()},
        {k: _acc(v) for k, v in by_raw.items()},
        _acc(ambiguous_totals),
        _acc(unambiguous_totals),
    )


def print_report(bucket_table, raw_table, ambiguous_totals, unambiguous_totals, note_ids):
    print("=" * 78)
    print("STAGE 1 RELIABILITY — abbreviation-expansion disambiguation vs gold")
    print("=" * 78)
    print(f"\nnotes: {note_ids}")

    print("\n--- Ambiguous-abbreviation entities vs unambiguous entities (headline) ---")
    for label, t in [("ambiguous (Stage 1 had to disambiguate)", ambiguous_totals),
                     ("unambiguous (single/no known meaning)", unambiguous_totals)]:
        acc = f"{t['accuracy']*100:.2f}%" if t["accuracy"] is not None else "-"
        print(f"  {label:<42} n={t['n']:>5}  correct={t['correct']:>5}  accuracy={acc}")

    print("\n--- By selection_basis (bucketed) ---")
    print(f"  {'basis':<20} | {'n':>5} | {'correct':>7} | {'accuracy':>9}")
    order = ["numeric_context", "omop_groundability", "alphabetical_default"]
    for basis in order:
        b = bucket_table.get(basis)
        if not b:
            print(f"  {basis:<20} | {'0':>5} | {'-':>7} | {'-':>9}  (no gradable rows)")
            continue
        acc = f"{b['accuracy']*100:.2f}%" if b["accuracy"] is not None else "-"
        print(f"  {basis:<20} | {b['n']:>5} | {b['correct']:>7} | {acc:>9}")

    print("\n--- By raw selection_basis value (numeric_context split by kind) ---")
    for basis, b in sorted(raw_table.items()):
        acc = f"{b['accuracy']*100:.2f}%" if b["accuracy"] is not None else "-"
        print(f"  {basis:<30} | n={b['n']:>4} | correct={b['correct']:>4} | accuracy={acc}")

    if bucket_table.get("alphabetical_default") and bucket_table.get("omop_groundability"):
        a = bucket_table["alphabetical_default"]["accuracy"]
        g = bucket_table["omop_groundability"]["accuracy"]
        if a is not None and g is not None and a >= g:
            print(f"\n  NOTE: alphabetical_default ({a*100:.1f}%) is NOT lower than "
                  f"omop_groundability ({g*100:.1f}%) -- the 'arbitrary default' tiebreak "
                  f"is not measurably worse than the deliberate one on this slice. Worth "
                  f"more data before concluding groundability isn't earning its keep, but "
                  f"flagged rather than assumed away.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None,
                     help="Comma-separated note_ids. Default: every note_id with "
                          "is_test=TRUE rows in extracted_entities.")
    ap.add_argument("--gold", default=None, help="path to train_annotations.csv")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=None, help="write full JSON report here")
    add_split_args(ap, default="val")
    args = ap.parse_args()

    gold_path = args.gold or _first_existing(GOLD_CANDIDATES, "gold annotations CSV")
    conn = duckdb.connect(args.db, read_only=True)
    try:
        note_ids, split_prov = resolve_note_ids(
            args, conn=conn, table="extracted_entities", where="is_test = TRUE")
        assert_no_contamination(conn)
        if not note_ids:
            raise SystemExit(
                f"No is_test=TRUE extracted_entities rows for split "
                f"'{split_prov['split']}'.")

        print(f"gold:  {gold_path}")
        print(f"db:    {args.db}")
        print(f"notes: {len(note_ids)}")

        gold_rows = load_gold(gold_path, note_ids)
        gold_by_note = collections.defaultdict(list)
        for g in gold_rows:
            gold_by_note[g["note_id"]].append(g)

        predictions = load_stage1_predictions(conn, note_ids)
        n_with_basis = sum(1 for p in predictions if p["selection_basis"])
        print(f"loaded {len(predictions)} accepted, normalized entities "
              f"({n_with_basis} with a non-null selection_basis)")
        if n_with_basis == 0:
            print("\nWARNING: 0 rows have selection_basis set. Either these notes were "
                  "extracted before the 2026-08-13 persistence patch (re-run extraction "
                  "after syncing src/entity_extraction.py), or none of them happened to "
                  "contain an ambiguous abbreviation. Nothing to grade.")

        attach_snomed_and_grade(conn, predictions, gold_by_note)
        bucket_table, raw_table, ambiguous_totals, unambiguous_totals = \
            reliability_by_basis(predictions)
    finally:
        conn.close()

    print_report(bucket_table, raw_table, ambiguous_totals, unambiguous_totals, note_ids)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "bucket_reliability": bucket_table,
                "raw_reliability": raw_table,
                "ambiguous_totals": ambiguous_totals,
                "unambiguous_totals": unambiguous_totals,
            }, f, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
