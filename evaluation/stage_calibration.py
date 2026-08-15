"""
evaluation/stage_calibration.py — per-stage confidence calibration (ECE),
so each stage's accept/reject threshold is set from measured data rather
than picked once and never checked.

WHY THIS EXISTS. Every stage in this pipeline that emits a numeric
confidence also enforces a hardcoded cutoff on it: Stage 2a's
EXTRACTION_THRESHOLD = 0.5 (src/entity_extraction.py), Stage 2b's
TIER3_SIMILARITY_FLOOR = 0.72 (src/normalization.py), Stage 3's
AUTO_VALIDATE_THRESHOLD = 0.85 / MOLLM_RESOLVE_THRESHOLD = 0.60
(src/mollm_ensemble.py). Only Stage 3's has ever been checked against
data (evaluation/cal_eval.py) -- the other two are documented constants,
not measured ones. This script asks the same question of all three:
does confidence X actually mean "correct" X% of the time, and if not,
where would the threshold need to move.

THREE DIFFERENT "CORRECT"S, BECAUSE THE STAGES ANSWER DIFFERENT QUESTIONS.
  Stage 2a -- is this predicted SPAN real at all (overlaps ANY gold
             annotation, regardless of concept)? A precision question.
             This is NOT recall calibration: a gold span GLiNER never
             proposed has no confidence value to bin in the first place --
             scripts/score_gold_recall.py's span_recall already covers
             that separate question.
  Stage 2b -- GIVEN a span with a gold match, is the CONCEPT this specific
             candidate names the right one? Graded per CANDIDATE, not just
             the tier's top-1 pick -- src/normalization.py's _result()
             keeps the full candidates[] list even on a "0 (Failed)" top
             result (below TIER3_SIMILARITY_FLOOR), so calibration data
             exists on both sides of that boundary, not just above it.
             Tier 1/2 are exact-string matches with similarity_score
             ALWAYS 1.0 by construction -- no spread to calibrate, so
             those get an accuracy sanity-check instead of an ECE curve.
  Stage 3  -- does the ensemble's resolution verdict match gold, as a
             function of composite_confidence? Not recomputed here --
             evaluation/cal_eval.py's own load_gradable_decisions()/
             grade() are imported and reused directly so this can't drift
             from that script's methodology, and results appear side by
             side with Stages 2a/2b rather than in a separate report.

STAGE 1 IS DELIBERATELY NOT INCLUDED. Abbreviation expansion is
deterministic dictionary lookup with only a binary `ambiguous` flag, not
a continuous confidence score -- there is no calibration curve to draw.

SAMPLE SIZE. Same caveat as every other evaluation/ script: as of
2026-08-11 the corpus run through this pipeline is small (a double-digit
number of notes for Stage 1/2, a handful for Stage 3). Treat every
threshold this script suggests as a methodology check, not a production
value, until re-run on a larger, dedicated validation slice -- see
docs/Evaluation_Criteria.md's "Validation Set" definition, which this
script does not itself enforce (it scores whatever --note-ids you give
it, train-slice or test-slice alike; keep the validation slice separate
from anything used for a final reported number yourself).

Run:
  python3 evaluation/stage_calibration.py
  python3 evaluation/stage_calibration.py --note-ids 10000032-DS-21 --out calib_report.json
"""

import argparse
import collections
import json
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

sys.path.insert(0, PROJECT_DIR)

from src.retrieval import VocabularyRetriever  # noqa: E402
from scripts.score_gold_recall import (  # noqa: E402
    load_gold, overlaps, _first_existing, GOLD_CANDIDATES,
)
from evaluation.cal_eval import (  # noqa: E402
    compute_ece, threshold_sweep, ECE_BINS,
    load_gradable_decisions as stage3_load_decisions,
    grade as stage3_grade,
)

# Current production cutoffs, imported where possible so this script can
# never silently drift from what the pipeline actually enforces.
try:
    from src.entity_extraction import EXTRACTION_THRESHOLD, SUBTHRESHOLD_FLOOR  # noqa: E402
except ImportError:
    EXTRACTION_THRESHOLD, SUBTHRESHOLD_FLOOR = 0.5, 0.35  # fallback if unimportable here

try:
    from src.normalization import TIER3_SIMILARITY_FLOOR  # noqa: E402
except ImportError:
    TIER3_SIMILARITY_FLOOR = 0.72

try:
    from src.mollm_ensemble import AUTO_VALIDATE_THRESHOLD, MOLLM_RESOLVE_THRESHOLD  # noqa: E402
except ImportError:
    AUTO_VALIDATE_THRESHOLD, MOLLM_RESOLVE_THRESHOLD = 0.85, 0.60


def _json(v, default):
    if not v:
        return default
    try:
        return json.loads(v) if isinstance(v, str) else v
    except (ValueError, TypeError):
        return default


# ==========================================================================
# Stage 2a -- extraction confidence vs. "is this span real"
# ==========================================================================

def load_stage2a_rows(conn, note_ids):
    """Every extracted_entities row for these notes, including
    below_threshold ones (retained down to SUBTHRESHOLD_FLOOR precisely so
    analysis like this can see both sides of EXTRACTION_THRESHOLD).
    Excludes superseded_by_split/superseded_by_growth rows for the same
    double-count reason scripts/score_gold_recall.py excludes them --
    a replaced entity and its replacement would otherwise both contribute
    a confidence data point for what is really one decision.
    """
    rows = conn.execute("""
        SELECT confidence, orig_start, orig_end, note_id
        FROM extracted_entities
        WHERE is_test = TRUE AND note_id IN ({})
          AND (superseded_by_split IS NULL OR superseded_by_split = FALSE)
          AND (superseded_by_growth IS NULL OR superseded_by_growth = FALSE)
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()
    return [{"confidence": r[0], "orig_start": r[1], "orig_end": r[2], "note_id": r[3]}
            for r in rows]


def grade_stage2a(rows, gold_by_note):
    """(confidence, is_real_span) pairs. is_real_span = this predicted span
    overlaps AT LEAST ONE gold annotation, any concept -- a precision
    question about the span itself, deliberately blind to whether Stage 2b
    later picked the right concept for it."""
    graded = []
    for r in rows:
        gold = gold_by_note.get(r["note_id"], [])
        correct = any(overlaps(r["orig_start"], r["orig_end"], g["start"], g["end"]) for g in gold)
        graded.append((r["confidence"], correct))
    return graded


# ==========================================================================
# Stage 2b -- per-candidate similarity score vs. "is this concept right"
# ==========================================================================

def load_stage2b_candidates(conn, note_ids):
    """Every individual candidate (not just each entity's top-1 pick) from
    normalized_entities.candidates, joined back to its span offsets on the
    same safe composite key every other evaluation/ script uses (note_id,
    original_text, expanded_text, gliner_label -- never entity_id, see
    scripts/score_gold_recall.py's module docstring for why).

    Returns ALL candidates regardless of match_tier; callers filter by
    tier, because Tier 1/2's similarity_score is always 1.0 by
    construction (exact string match) and has nothing to calibrate --
    only Tier 3 (SapBERT) and Tier 4 (fuzzy typo) carry a real score
    distribution.
    """
    rows = conn.execute("""
        SELECT n.candidates, e.orig_start, e.orig_end, e.note_id
        FROM normalized_entities n
        JOIN extracted_entities e
          ON e.note_id = n.note_id
         AND e.original_text = n.original_text
         AND e.expanded_text = n.expanded_text
         AND e.entity_label = n.gliner_label
        WHERE n.is_test = TRUE AND n.note_id IN ({})
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    out = []
    for cands_json, start, end, note_id in rows:
        for c in _json(cands_json, []):
            out.append({
                "similarity_score": c.get("similarity_score"),
                "omop_concept_id": c.get("omop_concept_id"),
                "match_tier": c.get("match_tier"),
                "orig_start": start, "orig_end": end, "note_id": note_id,
            })
    return out


def grade_stage2b_candidates(candidates, tier_label, gold_by_note, vocab):
    """(similarity_score, is_correct_concept) pairs for candidates at
    tier_label. Only counts candidates whose span overlaps a gold
    annotation AND whose concept crosswalks to a SNOMED code (same
    exclusions as scripts/score_gold_recall.py's uncrosswalked bucket --
    a candidate that can't even be checked isn't evidence either way)."""
    graded = []
    n_no_gold = n_uncrosswalked = 0
    for c in candidates:
        if c["match_tier"] != tier_label or c["similarity_score"] is None:
            continue
        gold = gold_by_note.get(c["note_id"], [])
        overlapping = [g for g in gold
                       if overlaps(c["orig_start"], c["orig_end"], g["start"], g["end"])]
        if not overlapping:
            n_no_gold += 1
            continue
        gold_ids = {g["concept_id"] for g in overlapping}
        code = vocab.snomed_code_for_concept(c["omop_concept_id"])
        if code is None:
            n_uncrosswalked += 1
            continue
        graded.append((c["similarity_score"], code in gold_ids))
    return graded, {"no_overlapping_gold": n_no_gold, "uncrosswalked": n_uncrosswalked}


def grade_stage2b_exact_tier(candidates, tier_label, gold_by_note, vocab):
    """Accuracy-only sanity check for Tier 1/2 (similarity_score constant
    at 1.0 -- no ECE curve possible, but a low accuracy here would mean
    exact lexical matching itself is unreliable, e.g. the documented
    'ED' -> 'Ed District' domain collision, which is worth surfacing even
    without a threshold to tune."""
    graded, excluded = grade_stage2b_candidates(candidates, tier_label, gold_by_note, vocab)
    n = len(graded)
    correct = sum(1 for _, ok in graded if ok)
    return {"n": n, "correct": correct, "accuracy": (correct / n) if n else None,
            "excluded": excluded}


# ==========================================================================
# Stage 3 -- thin wrapper around cal_eval.py's own functions
# ==========================================================================

def stage3_confidences(conn, note_ids, gold_by_note, vocab):
    """(composite_confidence, is_correct) pairs for resolution-mode Stage 3
    decisions, computed by calling cal_eval.py's own load_gradable_decisions()
    and grade() rather than re-deriving that logic -- if cal_eval.py's
    grading rules change, this follows automatically instead of silently
    diverging."""
    decisions = stage3_load_decisions(conn, note_ids)
    graded = []
    for d in decisions:
        outcome, _reason = stage3_grade(d, gold_by_note, vocab)
        if outcome is None or d["composite_confidence"] is None:
            continue
        graded.append((d["composite_confidence"], outcome == "correct"))
    return graded, len(decisions)


# ==========================================================================
# Reporting
# ==========================================================================

def _curve_report(name, graded, current_threshold, thresholds):
    if not graded:
        return {"name": name, "n": 0, "current_threshold": current_threshold,
                "ece": None, "reliability_table": [], "threshold_sweep": []}
    ece, table = compute_ece(graded)
    sweep = threshold_sweep(graded, thresholds)
    return {
        "name": name, "n": len(graded), "current_threshold": current_threshold,
        "ece": ece, "reliability_table": table, "threshold_sweep": sweep,
    }


def print_curve(report):
    print(f"\n--- {report['name']} (n={report['n']}, current threshold={report['current_threshold']}) ---")
    if report["n"] == 0:
        print("  no gradable data -- nothing to calibrate yet")
        return
    print(f"  ECE = {report['ece']}")
    print(f"  {'bin':<12} | {'n':>4} | {'mean conf':>10} | {'accuracy':>9}")
    for row in report["reliability_table"]:
        if row["n"] == 0:
            continue
        print(f"  {row['bin']:<12} | {row['n']:>4} | {row['mean_confidence']:>10.4f} | "
              f"{row['accuracy']:>9.4f}")
    # Row nearest the current production threshold, for a direct read. Some
    # stages (Tier 4 fuzzy) have no documented floor yet -- current_threshold
    # is None there, so there is nothing to snap to and this is skipped
    # rather than crashing on abs(x - None).
    if report["current_threshold"] is None:
        print("  (no current production threshold documented for this stage yet -- "
              "read the full sweep above to propose one)")
        return
    nearest = min(report["threshold_sweep"], key=lambda r: abs(r["threshold"] - report["current_threshold"]))
    print(f"  at current threshold ({report['current_threshold']}): "
          f"coverage={nearest['coverage']}, precision_if_admitted={nearest['precision_if_admitted']}")


def print_report(report):
    print("=" * 78)
    print("PER-STAGE CONFIDENCE CALIBRATION")
    print("=" * 78)
    print("\nSee module docstring: three different stages, three different")
    print("'correct's -- do not compare ECE values across stages directly,")
    print("only each stage's curve against its own threshold.\n")

    print_curve(report["stage2a_extraction"])

    print(f"\n--- Stage 2b Tier 1/2 (exact match, accuracy sanity check only) ---")
    for tier in ("1 (Exact)", "2 (Synonym)"):
        s = report["stage2b_exact"][tier]
        acc = f"{s['accuracy']*100:.1f}%" if s["accuracy"] is not None else "-"
        print(f"  {tier:<14} n={s['n']:>5}  accuracy={acc}  "
              f"(excluded: no_gold={s['excluded']['no_overlapping_gold']}, "
              f"uncrosswalked={s['excluded']['uncrosswalked']})")

    print_curve(report["stage2b_tier3"])
    print_curve(report["stage2b_tier4"])
    print_curve(report["stage3_composite"])

    print(f"\nSAMPLE SIZE CAVEAT: see module docstring -- these are methodology")
    print(f"checks at current corpus size, not fitted production thresholds.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None,
                     help="Comma-separated note_ids. Default: every note_id with "
                          "is_test=TRUE rows in extracted_entities.")
    ap.add_argument("--gold", default=None, help="path to train_annotations.csv")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=None, help="write full JSON report here")
    args = ap.parse_args()

    gold_path = args.gold or _first_existing(GOLD_CANDIDATES, "gold annotations CSV")
    conn = duckdb.connect(args.db, read_only=True)
    try:
        if args.note_ids:
            note_ids = [n.strip() for n in args.note_ids.split(",") if n.strip()]
        else:
            note_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test = TRUE"
            ).fetchall()]
        if not note_ids:
            raise SystemExit("No is_test=TRUE rows in extracted_entities. "
                             "Run scripts/test_pipeline_e2e.py first.")

        print(f"gold:  {gold_path}")
        print(f"db:    {args.db}")
        print(f"notes: {note_ids}")

        gold_rows = load_gold(gold_path, note_ids)
        gold_by_note = collections.defaultdict(list)
        for g in gold_rows:
            gold_by_note[g["note_id"]].append(g)

        vocab = VocabularyRetriever(conn)
        thresholds = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05..0.95

        # Stage 2a
        stage2a_rows = load_stage2a_rows(conn, note_ids)
        stage2a_graded = grade_stage2a(stage2a_rows, gold_by_note)

        # Stage 2b
        stage2b_candidates = load_stage2b_candidates(conn, note_ids)
        tier1_report = grade_stage2b_exact_tier(stage2b_candidates, "1 (Exact)", gold_by_note, vocab)
        tier2_report = grade_stage2b_exact_tier(stage2b_candidates, "2 (Synonym)", gold_by_note, vocab)
        tier3_graded, _ = grade_stage2b_candidates(stage2b_candidates, "3 (Semantic)", gold_by_note, vocab)
        tier4_graded, _ = grade_stage2b_candidates(stage2b_candidates, "4 (Fuzzy)", gold_by_note, vocab)

        # Stage 3 -- guarded separately: if MoLLM has never been run on this
        # DB at all, mollm_decisions may not exist yet (it's created lazily
        # by src.mollm_ensemble.store_decision()'s CREATE TABLE IF NOT
        # EXISTS), and a bare SELECT against a missing table would raise a
        # duckdb.Error that should not take down the Stage 2a/2b results
        # computed above -- those are valid and reportable independent of
        # whether Stage 3 has run at all yet.
        try:
            stage3_graded, stage3_n_decisions = stage3_confidences(conn, note_ids, gold_by_note, vocab)
            stage3_error = None
        except Exception as exc:
            # Deliberately broad: duckdb's exception class names/hierarchy
            # have changed across versions (requirements.txt only pins
            # duckdb>=0.9.0), and the failure mode this guards against --
            # mollm_decisions not existing yet -- must not take down the
            # already-computed Stage 2a/2b results below regardless of
            # which exact exception type this duckdb build raises for it.
            stage3_graded, stage3_n_decisions = [], 0
            stage3_error = str(exc)
            print(f"\nStage 3 skipped: {stage3_error}\n"
                  f"(expected if scripts/test_stage3_live.py --store has never been "
                  f"run on this DB -- mollm_decisions is created lazily on first write)")

        report = {
            "n_notes": len(note_ids),
            "stage2a_extraction": _curve_report(
                "Stage 2a extraction confidence (span is real)",
                stage2a_graded, EXTRACTION_THRESHOLD, thresholds),
            "stage2b_exact": {"1 (Exact)": tier1_report, "2 (Synonym)": tier2_report},
            "stage2b_tier3": _curve_report(
                "Stage 2b Tier 3 SapBERT similarity (concept is right)",
                tier3_graded, TIER3_SIMILARITY_FLOOR, thresholds),
            "stage2b_tier4": _curve_report(
                "Stage 2b Tier 4 fuzzy-typo similarity (concept is right)",
                tier4_graded, None, thresholds),
            "stage3_composite": _curve_report(
                "Stage 3 composite_confidence (resolution verdict is right)",
                stage3_graded, AUTO_VALIDATE_THRESHOLD, thresholds),
            "stage3_total_decisions_loaded": stage3_n_decisions,
            "stage3_error": stage3_error,
        }
    finally:
        conn.close()

    print_report(report)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
