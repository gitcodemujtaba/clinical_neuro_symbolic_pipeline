"""scripts/shadow_validate_tier2b.py -- 2026-08-20 retroactive shadow
validation for the new TIER_2B_CALIBRATED_AUTO_RESOLVED escape hatch
(src.mollm_tier_gate route_tier(), calibrator consultation for unanimous
re-rank decisions). Does NOT re-run the pipeline or the LLM ensemble --
reconstructs the exact entity/model_results shape build_feature_context()
needs from what's already stored in mollm_tier_gate_decisions for the
fresh25 batch's 94 TIER_2_AUTO_RESOLVED entities, scores each with the
production ConsensusCalibrator, and compares against gold (same clean-span
methodology as evaluation/grade_fresh25_by_tier.py) to answer: if this
mechanism HAD been active during that run, what precision/coverage would
TIER_2B actually have shown? This is the validation step
TIER_2B_CALIBRATED_AUTO_RESOLVED's own comment says it needs before ever
being added to AUTO_TIERS.
"""
import collections
import json
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
sys.path.insert(0, PROJECT_DIR)

from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from evaluation.grade_fresh25_by_tier import NOTE_IDS  # noqa: E402
from scripts.score_gold_recall import load_gold, overlaps  # noqa: E402
from src.mollm_tier_calibrator import DEFAULT_MODEL_PATH, ConsensusCalibrator, build_feature_context  # noqa: E402
from src.mollm_tier_gate import (  # noqa: E402
    CALIBRATED_AUTO_THRESHOLD, _fragile_shorthand_trap, _score_with_calibrator)
from src.retrieval import VocabularyRetriever  # noqa: E402


def pct(n, d):
    return f"{n}/{d} = {n/d*100:.1f}%" if d else f"{n}/{d} = n/a"


def main():
    conn = duckdb.connect(DB_PATH, read_only=True)
    vocab = VocabularyRetriever(conn)

    # scoring_note_ids intentionally omitted: fresh25 is disjoint from the
    # calibrator's own training notes (that's the whole point of "fresh"),
    # so the leakage guard has nothing to refuse here -- confirmed, not
    # assumed, since ConsensusCalibrator.load() would silently degrade to
    # untrained if it found any overlap.
    calibrator = ConsensusCalibrator.load(DEFAULT_MODEL_PATH, scoring_note_ids=NOTE_IDS)
    if calibrator.model is None:
        print("FATAL: calibrator failed to load or degraded to untrained "
              "(leakage guard tripped?). Aborting.")
        return

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, NOTE_IDS)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    note_ph = ",".join("?" * len(NOTE_IDS))
    rows = conn.execute(f"""
        SELECT d.entity_id, d.note_id, d.final_candidate_index, d.models,
               e.original_text, e.orig_start, e.orig_end, e.expansion_ambiguous,
               n.candidates, n.match_tier, n.is_ambiguous, n.domain_conflict,
               n.normalized_from
        FROM mollm_tier_gate_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.tier = 'TIER_2_AUTO_RESOLVED' AND d.note_id IN ({note_ph})
    """, NOTE_IDS).fetchall()
    print(f"{len(rows)} TIER_2_AUTO_RESOLVED decisions in the fresh25 batch\n")

    n_trapped = 0
    n_scored = 0
    n_promoted = 0
    n_promoted_gradable = 0
    n_promoted_correct = 0
    n_unpromoted_gradable = 0
    n_unpromoted_correct = 0
    score_buckets = collections.Counter()

    for (entity_id, note_id, final_idx, models_json, text, s, e, expansion_ambiguous,
         cands_json, match_tier, is_ambiguous, domain_conflict,
         normalized_from) in rows:

        candidates = cands_json if isinstance(cands_json, list) else json.loads(cands_json)
        model_results = models_json if isinstance(models_json, list) else json.loads(models_json)

        entity = {
            "original_text": text, "candidates": candidates, "match_tier": match_tier,
            "is_ambiguous": is_ambiguous, "domain_conflict": domain_conflict,
            "normalized_from": normalized_from, "expansion_ambiguous": expansion_ambiguous,
        }

        trapped, trap_reason = _fragile_shorthand_trap(entity, final_idx, candidates)
        if trapped:
            n_trapped += 1
            score = None
        else:
            context = build_feature_context(entity, model_results, prior_confirmation_count=0)
            score = calibrator.score(context)
            n_scored += 1
            score_buckets[f"{int(score*10)/10:.1f}" if score is not None else "None"] += 1

        promoted = score is not None and score >= CALIBRATED_AUTO_THRESHOLD
        if promoted:
            n_promoted += 1

        # Grade against gold, same clean-span discipline as grade_fresh25_by_tier.py.
        gold = gold_by_note.get(note_id, [])
        overlapping = [g for g in gold if overlaps(s, e, g["start"], g["end"])]
        if len(overlapping) != 1:
            continue
        g0 = overlapping[0]
        if (e - s) < (g0["end"] - g0["start"]):
            continue
        idx = (final_idx or 0) - 1
        if idx < 0 or idx >= len(candidates):
            continue
        concept_id = candidates[idx].get("omop_concept_id")
        pred_code = vocab.snomed_code_for_concept(concept_id) if concept_id else None
        correct = pred_code is not None and str(pred_code) == str(g0["concept_id"])

        if promoted:
            n_promoted_gradable += 1
            n_promoted_correct += int(correct)
        else:
            n_unpromoted_gradable += 1
            n_unpromoted_correct += int(correct)

    print(f"trapped (calibrator never consulted): {n_trapped}")
    print(f"scored by calibrator: {n_scored}")
    print(f"score distribution (0.1 buckets): {dict(sorted(score_buckets.items()))}\n")

    print("=" * 78)
    print(f"IF TIER_2B_CALIBRATED_AUTO_RESOLVED had been active "
          f"(threshold={CALIBRATED_AUTO_THRESHOLD}):")
    print("=" * 78)
    print(f"  would-be-promoted (TIER_2B): {n_promoted}/{len(rows)} decisions")
    print(f"  TIER_2B precision (clean-span, gradable only): "
          f"{pct(n_promoted_correct, n_promoted_gradable)}")
    print(f"  remaining at TIER_2 (unpromoted) precision: "
          f"{pct(n_unpromoted_correct, n_unpromoted_gradable)}")
    print(f"\n  For reference, the ORIGINAL (no calibrator) TIER_2 precision "
          f"measured on this batch was 11/68 = 16.2%.")


if __name__ == "__main__":
    main()
