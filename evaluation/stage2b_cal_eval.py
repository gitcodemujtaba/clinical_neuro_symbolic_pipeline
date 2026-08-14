"""
evaluation/stage2b_cal_eval.py — Stage 2b (OMOP concept-linking) reliability,
PLUS a direct cross-tabulation against Stage 3 (MoLLM) for every entity that
reached both stages: does MoLLM actually add value over Stage 2b alone?

WHY THIS IS TWO QUESTIONS IN ONE SCRIPT, NOT TWO SCRIPTS.
"Is Stage 2b's own confidence signal (match_tier / similarity_score)
trustworthy" and "does MoLLM's ensemble improve on Stage 2b's raw pick" are
different questions, but the second is uninterpretable without the first --
if Stage 2b's own top-1 pick were already highly accurate, MoLLM's job is
mostly confirmation (which is a fine, valid job); if Stage 2b's top-1 is
weak, MoLLM's job is genuine error-correction, and a failure to correct
matters much more. Reporting them separately, side by side, is what
evaluation/stage2a_cal_eval.py and evaluation/cal_eval.py already do for
their own stages -- this is the same pattern extended across the seam
between them.

WHY NO SINGLE CONTINUOUS ECE FOR ALL OF STAGE 2B. match_tier is categorical
("1 (Exact)", "2 (Synonym)", "3 (Semantic)", "0 (Failed)"), not a probability
-- Tier 1/2 hits are exact/synonym STRING matches with no numeric confidence
attached at all (src/normalization.py's _lookup_tier12() returns a fixed
1.0 "confidence" for any hit, which is a placeholder, not a measured
probability). Only Tier 3 (SapBERT cosine similarity) has a real continuous
score. So this script reports:
  (a) a DISCRETE reliability table -- accuracy per match_tier, answering
      "does 'Exact' actually mean near-100% right, does 'Semantic' mean
      something much lower" -- which is the honest way to check calibration
      when the confidence signal is categorical, not a score;
  (b) a continuous ECE computed ONLY on the Tier-3 subset's similarity_score,
      the one place a real Guo et al. 2017-style binning is meaningful here.

CROSS-TAB DEFINITIONS.
  CONFIRMED_CORRECT      -- Stage 2b's top-1 was already right, MoLLM's
                            resolution-mode verdict is also right. The
                            "nothing broken" case.
  CAUGHT_AND_FIXED       -- Stage 2b's top-1 was WRONG, MoLLM's verdict is
                            right (either picked a different, correct
                            candidate, or correctly said NONE_CORRECT when
                            genuinely none of the candidates matched gold).
                            This is MoLLM's entire reason for existing --
                            the number to watch.
  INTRODUCED_ERROR        -- Stage 2b's top-1 was RIGHT, but MoLLM's verdict
                            is WRONG. MoLLM made something that was already
                            correct worse. The number that should be zero
                            or very small; if it isn't, MoLLM is actively
                            harmful on this slice, not just unhelpful.
  MISSED_ERROR            -- Stage 2b's top-1 was wrong AND MoLLM's verdict
                            is also wrong (whether or not they agree on the
                            SAME wrong answer -- this bucket doesn't
                            distinguish "same mistake" from "different
                            mistake", both are "did not fix it").
  UNGRADABLE combinations -- either side lacking a comparable outcome (no
                            overlapping gold span for Stage 2b, or MoLLM
                            verdict not a resolution pick / model
                            disagreement / no candidates) -- reported
                            separately, never silently dropped.

Both sides of the join use grade()/load_gradable_decisions() from
evaluation/cal_eval.py and load_predictions()/attach_snomed_codes() from
scripts/score_gold_recall.py UNCHANGED -- this script adds no new grading
logic of its own, only joins the two existing, already-verified gradings on
the same safe composite key (note_id, original_text, expanded_text,
gliner_label/entity_label) every other cross-table script in this project
uses instead of entity_id, for the same documented reason (normalized_
entities' UNIQUE constraint fan-out, see score_gold_recall.py's module
docstring).

Run:
  python3 evaluation/stage2b_cal_eval.py
  python3 evaluation/stage2b_cal_eval.py --note-ids 17751158-DS-19,19442119-DS-15 --out reports/stage2b_cal.json
"""

import argparse
import collections
import json
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

sys.path.insert(0, PROJECT_DIR)

from src.retrieval import VocabularyRetriever  # noqa: E402
from scripts.score_gold_recall import (  # noqa: E402
    load_gold, overlaps, _first_existing, GOLD_CANDIDATES,
)
from evaluation.cal_eval import (  # noqa: E402
    load_gradable_decisions, grade, ECE_BINS,
)
from evaluation.metrics import (  # noqa: E402
    accuracy as _accuracy, bootstrap_ci, compute_ece_report, format_ci,
    group_by_note, print_interpretation_block,
)
from evaluation.splits import (  # noqa: E402
    add_split_args, assert_no_contamination, resolve_note_ids,
)


def load_stage2b_predictions(conn, note_ids):
    """Same accepted-row shape as score_gold_recall.py's load_predictions(),
    plus similarity_score (not selected there -- only needed for Stage 2b's
    own Tier-3 ECE, so kept local to this script rather than widening a
    stable, widely-reused function's column list for one extra field).
    """
    rows = conn.execute("""
        SELECT e.note_id, e.orig_start, e.orig_end, e.entity_label,
               e.original_text, e.expanded_text, e.entity_id,
               n.omop_concept_id, n.omop_concept_name, n.omop_vocab,
               n.match_tier, n.similarity_score
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
            "omop_vocab", "match_tier", "similarity_score"]
    return [dict(zip(cols, r)) for r in rows]


def attach_snomed_and_grade(conn, predictions, gold_by_note):
    """Mutates predictions in place: adds snomed_code (RxNorm->SNOMED
    crosswalked where needed, same method as score_gold_recall.py) and
    stage2b_correct (True/False/None -- None means no overlapping gold span
    at all, i.e. ungradable, not 'wrong')."""
    vocab = VocabularyRetriever(conn)
    for p in predictions:
        cid = p["omop_concept_id"]
        p["snomed_code"] = vocab.snomed_code_for_concept(cid) if cid is not None else None

        gold = gold_by_note.get(p["note_id"], [])
        overlapping_gold = [g for g in gold
                            if overlaps(p["orig_start"], p["orig_end"], g["start"], g["end"])]
        if not overlapping_gold:
            p["stage2b_correct"] = None
            continue
        gold_concept_ids = {g["concept_id"] for g in overlapping_gold}
        p["stage2b_correct"] = p["snomed_code"] in gold_concept_ids if p["snomed_code"] else False
    return predictions


def discrete_reliability_table(predictions):
    """Per-match_tier n/correct/accuracy, over predictions with a
    determinable stage2b_correct (i.e. excluding the None/ungradable ones --
    tallied separately by the caller)."""
    by_tier = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    for p in predictions:
        if p["stage2b_correct"] is None:
            continue
        b = by_tier[p["match_tier"]]
        b["n"] += 1
        b["correct"] += int(p["stage2b_correct"])
    out = {}
    for tier, b in by_tier.items():
        out[tier] = {"n": b["n"], "correct": b["correct"],
                     "accuracy": round(b["correct"] / b["n"], 4) if b["n"] else None}
    return out


def tier3_ece(predictions, n_bins=ECE_BINS):
    """Continuous ECE on similarity_score, restricted to Tier 3 ('3
    (Semantic)') predictions with a determinable stage2b_correct -- see
    module docstring for why Tier 1/2/0 are excluded (no continuous
    confidence signal to bin)."""
    pairs = [(p["similarity_score"], p["stage2b_correct"]) for p in predictions
             if p["match_tier"] == "3 (Semantic)" and p["stage2b_correct"] is not None
             and p["similarity_score"] is not None]
    if not pairs:
        return None, [], 0

    bins = [[] for _ in range(n_bins)]
    for conf, correct in pairs:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))

    n = len(pairs)
    ece = 0.0
    table = []
    for i, b in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not b:
            table.append({"bin": f"[{lo:.1f}, {hi:.1f})", "n": 0,
                          "mean_similarity": None, "accuracy": None})
            continue
        mean_conf = sum(c for c, _ in b) / len(b)
        acc = sum(1 for _, ok in b if ok) / len(b)
        ece += (len(b) / n) * abs(acc - mean_conf)
        table.append({"bin": f"[{lo:.1f}, {hi:.1f})", "n": len(b),
                      "mean_similarity": round(mean_conf, 4), "accuracy": round(acc, 4)})
    return round(ece, 4), table, n


# ==========================================================================
# PER-CANDIDATE calibration -- ported from evaluation/stage_calibration.py
# (2026-08-13 reconciliation, docs/2026-08-13_Code_Improvement_Proposals.md P5)
# ==========================================================================
#
# WHY THIS WAS PORTED RATHER THAN DELETED. The 2026-08-13 report S9.3 flagged
# stage_calibration.py as "substantially overlapping" this script and left the
# reconciliation outstanding. Diffed: the overlap is real for Stage 2a (and
# stage_calibration.py's version of it is DEFECTIVE -- see that file's
# deprecation banner), but its Stage 2b analysis asks a genuinely different
# question that nothing else in evaluation/ asks:
#
#   this script's tier3_ece()      -- grades each ENTITY's top-1 pick.
#                                     "Given what Stage 2b actually chose,
#                                      how often was it right?"
#   per_candidate_calibration()    -- grades EVERY candidate in the list.
#                                     "Does similarity_score rank candidates
#                                      correctly, independent of which one
#                                      the top-1 rule happened to select?"
#
# The second question is the one that matters for P1.1's Tier 1/2 ranker: if
# similarity ranks well but top-1 accuracy is poor, the SELECTION rule is the
# bug, not the signal. Deleting this would have thrown away the measurement
# that distinguishes those two diagnoses.

def load_stage2b_all_candidates(conn, note_ids):
    """Every individual candidate from normalized_entities.candidates, not
    just each entity's top-1, joined back to span offsets on the safe
    composite key (note_id, original_text, expanded_text, gliner_label --
    never entity_id; see scripts/score_gold_recall.py's module docstring).

    Also returns each candidate's INDEX within its entity's list, which
    stage_calibration.py's version did not carry. That index is what makes
    the "is the ranking right" question answerable: rank-0 accuracy versus
    best-available-in-list accuracy is exactly the gap a better tiebreak
    could close, and it cannot be computed without knowing position.
    """
    rows = conn.execute("""
        SELECT n.candidates, e.orig_start, e.orig_end, e.note_id, n.match_tier
        FROM normalized_entities n
        JOIN extracted_entities e
          ON e.note_id = n.note_id
         AND e.original_text = n.original_text
         AND e.expanded_text = n.expanded_text
         AND e.entity_label = n.gliner_label
        WHERE n.is_test = TRUE AND n.note_id IN ({})
          AND (e.below_threshold IS NULL OR e.below_threshold = FALSE)
          AND (e.superseded_by_split IS NULL OR e.superseded_by_split = FALSE)
          AND (e.superseded_by_growth IS NULL OR e.superseded_by_growth = FALSE)
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    out = []
    for cands_json, start, end, note_id, entity_tier in rows:
        try:
            cands = json.loads(cands_json) if isinstance(cands_json, str) else (cands_json or [])
        except (TypeError, ValueError):
            cands = []
        for rank, c in enumerate(cands):
            out.append({
                "rank": rank,
                "similarity_score": c.get("similarity_score"),
                "omop_concept_id": c.get("omop_concept_id"),
                "match_tier": c.get("match_tier") or entity_tier,
                "orig_start": start, "orig_end": end, "note_id": note_id,
            })
    return out


def grade_candidates(conn, candidates, gold_by_note):
    """Adds `correct` to each candidate: does its crosswalked SNOMED code
    match an overlapping gold annotation's concept_id.

    None (not False) when there is no overlapping gold span or the concept
    does not crosswalk -- "cannot be checked" is not "wrong", the same
    exclusion policy attach_snomed_and_grade() already applies to top-1
    predictions and score_gold_recall.py applies to its uncrosswalked bucket.
    Excluded counts are returned so they can be reported loudly rather than
    silently shrinking the denominator.
    """
    vocab = VocabularyRetriever(conn)
    excluded = {"no_overlapping_gold": 0, "uncrosswalked": 0}
    for c in candidates:
        gold = gold_by_note.get(c["note_id"], [])
        overlapping = [g for g in gold
                       if overlaps(c["orig_start"], c["orig_end"], g["start"], g["end"])]
        if not overlapping:
            c["correct"] = None
            excluded["no_overlapping_gold"] += 1
            continue
        cid = c["omop_concept_id"]
        code = vocab.snomed_code_for_concept(cid) if cid is not None else None
        if code is None:
            c["correct"] = None
            excluded["uncrosswalked"] += 1
            continue
        c["correct"] = code in {g["concept_id"] for g in overlapping}
    return candidates, excluded


def ranking_quality(candidates):
    """The measurement P1.1 exists to move: how much accuracy is left on the
    table by the SELECTION rule, as opposed to by the candidate list.

    Groups graded candidates by entity (note_id + span), then reports:
      * top1_accuracy   -- accuracy of rank 0, i.e. what Stage 2b actually ships
      * oracle_accuracy -- fraction of entities where ANY candidate in the list
                           is correct, i.e. the ceiling a perfect tiebreak
                           would reach with today's retrieval unchanged
      * headroom        -- oracle - top1. This number IS the value of a better
                           Tier 1/2 ranker. If it is ~0, the retrieval is the
                           problem and no reordering will help; if it is large,
                           the arbitrary `ORDER BY concept_id ASC` tiebreak is
                           throwing away matches it already found.

    Reported per match_tier as well as overall, because the 2026-08-13 report
    S5.3's headline (Tier 1 "Exact" at 52.71%, BELOW Tier 2's 61.84%) predicts
    the headroom should be concentrated in Tier 1 specifically.
    """
    by_entity = collections.defaultdict(list)
    for c in candidates:
        if c.get("correct") is None:
            continue
        by_entity[(c["note_id"], c["orig_start"], c["orig_end"])].append(c)

    overall = {"n_entities": 0, "top1_correct": 0, "oracle_correct": 0}
    by_tier = collections.defaultdict(
        lambda: {"n_entities": 0, "top1_correct": 0, "oracle_correct": 0})

    for cands in by_entity.values():
        cands = sorted(cands, key=lambda c: c["rank"])
        # Only entities whose rank-0 candidate is itself gradable can
        # contribute a top-1 number; otherwise "top1_correct" would be
        # comparing a missing value against an oracle that has one.
        if cands[0]["rank"] != 0:
            continue
        tier = cands[0]["match_tier"]
        top1 = bool(cands[0]["correct"])
        oracle = any(bool(c["correct"]) for c in cands)
        for bucket in (overall, by_tier[tier]):
            bucket["n_entities"] += 1
            bucket["top1_correct"] += int(top1)
            bucket["oracle_correct"] += int(oracle)

    def _finish(b):
        n = b["n_entities"]
        if not n:
            return {**b, "top1_accuracy": None, "oracle_accuracy": None, "headroom": None}
        t, o = b["top1_correct"] / n, b["oracle_correct"] / n
        return {**b, "top1_accuracy": round(t, 4), "oracle_accuracy": round(o, 4),
                "headroom": round(o - t, 4)}

    return {"overall": _finish(overall),
            "by_tier": {k: _finish(v) for k, v in sorted(by_tier.items())}}


def per_candidate_ece(candidates, tier_label="3 (Semantic)", n_bins=ECE_BINS):
    """ECE over EVERY candidate at `tier_label`, not just top-1 picks.

    Tier 1/2 candidates carry similarity_score 1.0 by construction (exact
    string match), so they have no spread to calibrate -- an accuracy-only
    figure is the honest output for those, which ranking_quality() above
    already provides per tier. Kept restricted to a scored tier for that
    reason, defaulting to Tier 3.
    """
    pairs = [(c["similarity_score"], c["correct"]) for c in candidates
             if c["match_tier"] == tier_label and c.get("correct") is not None
             and c.get("similarity_score") is not None]
    if not pairs:
        return None
    return compute_ece_report(pairs, n_bins=n_bins, scheme="equal_width",
                              value_name="similarity")


def cross_tab(stage2b_predictions, mollm_decisions_graded):
    """Joins Stage 2b's per-entity correctness against MoLLM's per-decision
    outcome on (note_id, original_text, expanded_text, entity_label) -- the
    established safe composite key (see module docstring). Returns the
    4-category confusion breakdown plus example rows for each category (for
    manual spot-checking, same "show concrete examples, don't just report a
    count" pattern score_gold_recall.py uses throughout).
    """
    stage2b_by_key = {}
    for p in stage2b_predictions:
        key = (p["note_id"], p["original_text"], p["expanded_text"], p["entity_label"])
        stage2b_by_key[key] = p

    counts = collections.Counter()
    examples = collections.defaultdict(list)
    # 2026-08-13 (verification follow-up, docs/2026-08-13_Implementation_Verification.md):
    # the aggregate NET VALUE figure hid that INTRODUCED_ERROR concentrates in
    # specific entity types (Lab Test, Anatomy) rather than spreading evenly --
    # found only by manually re-deriving it from `examples`' truncated sample,
    # since nothing printed the full breakdown. Counted over EVERY graded row,
    # not capped like `examples`, so this number is exact rather than a sample
    # extrapolation.
    by_entity_label = collections.defaultdict(collections.Counter)
    n_no_stage2b_match = 0

    for decision, outcome in mollm_decisions_graded:
        key = (decision["note_id"], decision["original_text"],
               decision["expanded_text"], decision["entity_label"])
        s2b = stage2b_by_key.get(key)
        if s2b is None:
            # A resolution-mode MoLLM decision with no matching Stage 2b row
            # at all -- shouldn't normally happen (Stage 3 reads its
            # candidates from Stage 2b's own output) but reported rather
            # than silently skipped in case it does (e.g. a stale decision
            # from before a Stage 2b rerun changed which rows exist).
            n_no_stage2b_match += 1
            continue

        s2b_correct = s2b["stage2b_correct"]
        mollm_correct = outcome  # "correct" / "incorrect" / None

        if s2b_correct is None or mollm_correct is None:
            counts["ungradable"] += 1
            continue

        mollm_bool = (mollm_correct == "correct")
        if s2b_correct and mollm_bool:
            cat = "CONFIRMED_CORRECT"
        elif not s2b_correct and mollm_bool:
            cat = "CAUGHT_AND_FIXED"
        elif s2b_correct and not mollm_bool:
            cat = "INTRODUCED_ERROR"
        else:
            cat = "MISSED_ERROR"

        counts[cat] += 1
        by_entity_label[decision["entity_label"]][cat] += 1
        if len(examples[cat]) < 8:
            examples[cat].append({
                "note_id": decision["note_id"],
                "text": decision["original_text"],
                "entity_label": decision["entity_label"],
                "stage2b_concept": s2b["omop_concept_name"],
                "stage2b_match_tier": s2b["match_tier"],
                "mollm_routing_decision": decision.get("mollm_routing_decision"),
            })

    return {
        "counts": dict(counts),
        "examples": dict(examples),
        "no_stage2b_match": n_no_stage2b_match,
        "by_entity_label": {label: dict(cats)
                            for label, cats in sorted(by_entity_label.items())},
    }


def print_report(discrete_table, tier3_ece_val, tier3_table, tier3_n, xtab, note_ids):
    print("=" * 78)
    print("STAGE 2B RELIABILITY — OMOP concept-linking (match_tier) vs gold")
    print("=" * 78)
    print(f"\nnotes: {note_ids}")

    print(f"\n--- Discrete reliability by match_tier ---")
    print(f"  {'tier':<16} | {'n':>5} | {'correct':>7} | {'accuracy':>9}")
    tier_order = ["1 (Exact)", "2 (Synonym)", "3 (Semantic)", "0 (Failed)"]
    for tier in tier_order:
        b = discrete_table.get(tier)
        if not b:
            continue
        print(f"  {tier:<16} | {b['n']:>5} | {b['correct']:>7} | {b['accuracy']*100:>8.2f}%")
    for tier, b in discrete_table.items():
        if tier not in tier_order:
            print(f"  {tier or '(null)':<16} | {b['n']:>5} | {b['correct']:>7} | "
                  f"{(b['accuracy']*100 if b['accuracy'] is not None else 0):>8.2f}%")

    print(f"\n--- Tier-3 (Semantic) continuous ECE on similarity_score, n={tier3_n} ---")
    if tier3_ece_val is not None:
        print(f"  ECE = {tier3_ece_val}")
        print(f"  {'bin':<12} | {'n':>4} | {'mean sim':>9} | {'accuracy':>9}")
        for row in tier3_table:
            ms = f"{row['mean_similarity']:.4f}" if row["mean_similarity"] is not None else "-"
            ac = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "-"
            print(f"  {row['bin']:<12} | {row['n']:>4} | {ms:>9} | {ac:>9}")
    else:
        print("  no Tier-3 predictions with both a gold-comparable outcome and a "
              "similarity_score -- nothing to report.")

    print(f"\n{'=' * 78}")
    print("STAGE 2B vs STAGE 3 — does MoLLM add value over Stage 2b's own top-1 pick?")
    print("=" * 78)
    c = xtab["counts"]
    total_graded = sum(c.get(k, 0) for k in
                       ("CONFIRMED_CORRECT", "CAUGHT_AND_FIXED", "INTRODUCED_ERROR", "MISSED_ERROR"))
    print(f"\n  entities gradable on BOTH sides (Stage 2b has a gold-comparable "
          f"top-1, MoLLM has a gradable resolution verdict): {total_graded}")
    print(f"  ungradable on at least one side (excluded above): {c.get('ungradable', 0)}")
    if xtab["no_stage2b_match"]:
        print(f"  MoLLM decisions with no matching Stage 2b row at all "
              f"(unexpected, worth investigating): {xtab['no_stage2b_match']}")

    if total_graded == 0:
        print("\n  Nothing gradable on both sides -- no cross-tab to report.")
        return

    for cat, desc in [
        ("CONFIRMED_CORRECT", "Stage 2b already right, MoLLM agrees"),
        ("CAUGHT_AND_FIXED", "Stage 2b WRONG, MoLLM fixed it -- MoLLM's value-add"),
        ("INTRODUCED_ERROR", "Stage 2b RIGHT, MoLLM broke it -- should be ~0"),
        ("MISSED_ERROR", "Stage 2b wrong, MoLLM did not fix it"),
    ]:
        n = c.get(cat, 0)
        pct = f"{n / total_graded * 100:.1f}%" if total_graded else "-"
        print(f"\n  {cat:<20} {n:>4} ({pct})  -- {desc}")
        for ex in xtab["examples"].get(cat, [])[:5]:
            print(f"      [{ex['note_id']}] '{ex['text']}' ({ex['entity_label']}) "
                  f"stage2b->{ex['stage2b_concept']} (tier {ex['stage2b_match_tier']}) "
                  f"mollm_routing={ex['mollm_routing_decision']}")

    caught = c.get("CAUGHT_AND_FIXED", 0)
    introduced = c.get("INTRODUCED_ERROR", 0)
    print(f"\n  NET VALUE (CAUGHT_AND_FIXED - INTRODUCED_ERROR): {caught - introduced}")
    if introduced > caught:
        print("  WARNING: MoLLM is introducing more errors than it fixes on this "
              "gradable slice -- do not treat MoLLM's resolution-mode output as a "
              "strict improvement over Stage 2b alone without addressing this.")

    # 2026-08-13: NET VALUE alone hides WHERE the damage concentrates. Printed
    # unconditionally (not just when negative) so a healthy net value can't
    # hide one bad entity_label offsetting several good ones either.
    by_label = xtab.get("by_entity_label") or {}
    if by_label:
        print(f"\n  --- by entity_label (CONFIRMED / CAUGHT+FIXED / INTRODUCED_ERR / MISSED_ERR / net) ---")
        for label, cats in sorted(by_label.items(),
                                  key=lambda kv: -sum(kv[1].values())):
            cc = cats.get("CONFIRMED_CORRECT", 0)
            cf = cats.get("CAUGHT_AND_FIXED", 0)
            ie = cats.get("INTRODUCED_ERROR", 0)
            me = cats.get("MISSED_ERROR", 0)
            flag = "  <-- MoLLM net-harmful for this type" if ie > cf else ""
            print(f"    {label:<14} {cc:>4} / {cf:>4} / {ie:>4} / {me:>4}  "
                  f"net={cf - ie:+d}{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None,
                     help="Comma-separated note_ids. Default: every note_id with "
                          "is_test=TRUE resolution-mode rows in mollm_decisions "
                          "(so the cross-tab always has something to join against).")
    ap.add_argument("--gold", default=None, help="path to train_annotations.csv")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=None, help="write full JSON report here")
    ap.add_argument("--per-candidate", action="store_true",
                    help="ALSO run the per-candidate analysis ported from the "
                         "deprecated evaluation/stage_calibration.py: grades every "
                         "candidate in each entity's list, not just the top-1 pick, "
                         "and reports ranking headroom (oracle minus top-1). This is "
                         "the measurement that says whether Stage 2b's accuracy "
                         "problem is the RETRIEVAL or the TIEBREAK.")
    add_split_args(ap, default="val")
    args = ap.parse_args()

    gold_path = args.gold or _first_existing(GOLD_CANDIDATES, "gold annotations CSV")
    conn = duckdb.connect(args.db, read_only=True)
    try:
        note_ids, split_prov = resolve_note_ids(
            args, conn=conn, table="mollm_decisions",
            where="is_test = TRUE AND mode = 'resolution'")
        assert_no_contamination(conn)
        if not note_ids:
            raise SystemExit(
                f"No is_test=TRUE resolution-mode rows in mollm_decisions for "
                f"split '{split_prov['split']}'.")

        print(f"gold:  {gold_path}")
        print(f"db:    {args.db}")
        print(f"notes: {note_ids}")

        gold_rows = load_gold(gold_path, note_ids)
        gold_by_note = collections.defaultdict(list)
        for g in gold_rows:
            gold_by_note[g["note_id"]].append(g)

        stage2b_predictions = load_stage2b_predictions(conn, note_ids)
        attach_snomed_and_grade(conn, stage2b_predictions, gold_by_note)

        discrete_table = discrete_reliability_table(stage2b_predictions)
        tier3_ece_val, tier3_table, tier3_n = tier3_ece(stage2b_predictions)

        mollm_decisions = load_gradable_decisions(conn, note_ids)
        vocab = VocabularyRetriever(conn)
        mollm_graded = [(d, grade(d, gold_by_note, vocab)[0]) for d in mollm_decisions]

        xtab = cross_tab(stage2b_predictions, mollm_graded)

        # Note-level bootstrap CI per match_tier. The 2026-08-13 report
        # presented Tier 1 at 52.71% and Tier 2 at 61.84% as a real inversion
        # without ever testing it; these intervals are what make that claim
        # (or refute it) rather than assert it.
        tier_cis = {}
        for tier in sorted({p["match_tier"] for p in stage2b_predictions
                            if p["stage2b_correct"] is not None}):
            rows = [{"note_id": p["note_id"], "correct": p["stage2b_correct"]}
                    for p in stage2b_predictions
                    if p["match_tier"] == tier and p["stage2b_correct"] is not None]
            tier_cis[tier] = bootstrap_ci(
                group_by_note(rows),
                lambda rs: _accuracy([r["correct"] for r in rs]))

        tier3_pairs = [(p["similarity_score"], p["stage2b_correct"])
                       for p in stage2b_predictions
                       if p["match_tier"] == "3 (Semantic)"
                       and p["stage2b_correct"] is not None
                       and p["similarity_score"] is not None]

        per_candidate = None
        if args.per_candidate:
            cands = load_stage2b_all_candidates(conn, note_ids)
            grade_candidates(conn, cands, gold_by_note)
            per_candidate = {
                "ranking_quality": ranking_quality(cands),
                "tier3_ece": per_candidate_ece(cands),
                "n_candidates_loaded": len(cands),
                "n_gradable": sum(1 for c in cands if c.get("correct") is not None),
            }
    finally:
        conn.close()

    print_report(discrete_table, tier3_ece_val, tier3_table, tier3_n, xtab, note_ids)

    print(f"\n--- Per-tier accuracy, note-level 95% CI ---")
    for tier, ci in tier_cis.items():
        print(f"  {tier:<16} {format_ci(ci) if ci else 'n/a (needs >=2 notes)'}")
    print("  Overlapping intervals mean the tier ordering is not established "
          "by this sample.")

    if tier3_pairs:
        print(f"\n--- Tier 3 (SapBERT similarity) interpretation ---")
        print_interpretation_block(tier3_pairs)

    if per_candidate:
        print(f"\n{'=' * 78}")
        print("PER-CANDIDATE ANALYSIS (ported from the deprecated "
              "evaluation/stage_calibration.py)")
        print("=" * 78)
        rq = per_candidate["ranking_quality"]
        print(f"\n  candidates loaded: {per_candidate['n_candidates_loaded']}  "
              f"gradable: {per_candidate['n_gradable']}")
        print(f"\n  {'tier':<16} | {'entities':>8} | {'top-1':>7} | "
              f"{'oracle':>7} | {'headroom':>8}")
        rows = list(rq["by_tier"].items()) + [("OVERALL", rq["overall"])]
        for tier, b in rows:
            if b["top1_accuracy"] is None:
                continue
            print(f"  {tier:<16} | {b['n_entities']:>8} | "
                  f"{b['top1_accuracy']*100:>6.2f}% | "
                  f"{b['oracle_accuracy']*100:>6.2f}% | "
                  f"{b['headroom']*100:>7.2f}%")
        print("\n  HEADROOM = oracle - top-1: the accuracy a PERFECT tiebreak would")
        print("  recover from candidate lists Stage 2b has ALREADY retrieved.")
        print("  Large headroom on Tier 1 => the `ORDER BY concept_id ASC` tiebreak")
        print("  is the bug (fix: src/normalization.py RANKED_TIER12).")
        print("  Near-zero headroom => retrieval is the bug and no reordering helps.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "split_provenance": split_prov,
                "discrete_reliability_by_tier": discrete_table,
                "accuracy_ci_by_tier": tier_cis,
                "tier3_ece": tier3_ece_val,
                "tier3_ece_full": compute_ece_report(tier3_pairs,
                                                     value_name="similarity"),
                "tier3_ece_equal_mass": compute_ece_report(
                    tier3_pairs, scheme="equal_mass", value_name="similarity"),
                "tier3_reliability_table": tier3_table,
                "tier3_n": tier3_n,
                "cross_tab": xtab,
                "per_candidate": per_candidate,
            }, f, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
