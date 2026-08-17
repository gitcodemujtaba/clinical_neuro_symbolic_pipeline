"""
evaluation/tier_gate_calibrator_ood_check.py -- read-only diagnostic testing
whether ConsensusCalibrator generalizes to unanimous-vote feature shapes it
has never been trained on (2026-08-18, "Fix #1" of the tier-gate audit).

CONTEXT. evaluation/tier_gate_cal_eval.py's own docstring is explicit: the
calibrator's training population is ONLY TIER_4_ENSEMBLE_SPLIT (non-unanimous)
decisions -- "training on [AUTO tiers] would teach the model an input
distribution it never sees at inference." A proposed fix (route unanimous
SUPPORTED_1 votes with composite_confidence < TIER1_CONFIDENCE_FLOOR=0.70
through the calibrator instead of auto-parking them as "Unrouted") assumes
the calibrator would score these sensibly despite never having trained on a
single 3-0 vote-count feature pattern. This script tests that assumption
empirically instead of trusting the feature-shape intuition: reconstructs
the real 16-feature vector for every historical Unrouted decision, scores
it with the CURRENTLY DEPLOYED calibrator .pkl, and reports the resulting
distribution. Writes nothing to the DB -- calibrator.score() is read-only.

Run: python3 -m evaluation.tier_gate_calibrator_ood_check
"""
import collections
import json
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
sys.path.insert(0, PROJECT_DIR)

from src.mollm_tier_calibrator import (  # noqa: E402
    ConsensusCalibrator, DEFAULT_MODEL_PATH, build_feature_context,
    count_prior_confirmations, featurize)
from src.mollm_tier_gate import (  # noqa: E402
    CALIBRATED_AUTO_THRESHOLD, _is_coronary_segment_trap, _is_short_alphanumeric_code)


def load_unrouted_decisions(conn):
    """Every historical decision that landed as 'Unrouted' specifically
    because it was a unanimous SUPPORTED_1 vote whose composite_confidence
    fell short of TIER1_CONFIDENCE_FLOOR (0.70) -- the exact population
    Fix #1 proposes routing through the calibrator instead."""
    rows = conn.execute("""
        SELECT d.entity_id, d.models, d.composite_confidence,
               e.original_text, n.candidates, n.match_tier, n.is_ambiguous,
               n.domain_conflict, n.normalized_from
        FROM mollm_tier_gate_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.tier IS NULL AND d.queue_reason = 'below_confidence_threshold'
    """).fetchall()
    cols = [c[0] for c in conn.description]
    return [dict(zip(cols, r)) for r in rows]


def main():
    conn = duckdb.connect(DB_PATH, read_only=True)
    calibrator = ConsensusCalibrator.load(DEFAULT_MODEL_PATH)
    if calibrator.model is None:
        print(f"No fitted calibrator at {DEFAULT_MODEL_PATH} -- nothing to test.")
        return

    decisions = load_unrouted_decisions(conn)
    print(f"{len(decisions)} historical Unrouted (unanimous SUPPORTED_1, "
          f"composite_confidence < 0.70) decisions found.\n")

    scores = []
    trapped_count = 0
    skipped = collections.Counter()
    for d in decisions:
        candidates = d["candidates"]
        if isinstance(candidates, str):
            candidates = json.loads(candidates)
        if not candidates:
            skipped["no_candidates"] += 1
            continue

        models = d["models"]
        if isinstance(models, str):
            models = json.loads(models)

        entity = {
            "candidates": candidates,
            "match_tier": d["match_tier"],
            "is_ambiguous": d["is_ambiguous"],
            "domain_conflict": d["domain_conflict"],
            "normalized_from": d["normalized_from"],
            "expansion_ambiguous": False,
            "original_text": d["original_text"],
        }

        # Unanimous SUPPORTED_1 always means final_candidate_index = 1 (see
        # route_tier()'s own branch for this exact case).
        candidate_index = 1
        if _is_coronary_segment_trap(entity, candidate_index, candidates) or \
           _is_short_alphanumeric_code(entity):
            trapped_count += 1
            # Still score it -- we want to see what the calibrator WOULD say
            # even for entities the production hard traps would bypass, to
            # know if the OOD problem and the trap problem overlap or are
            # separate populations.

        chosen_concept_id = candidates[0].get("omop_concept_id")
        prior_count = count_prior_confirmations(conn, d["original_text"], chosen_concept_id)
        context = build_feature_context(entity, models, prior_count)
        vector = featurize(context)
        score = calibrator.score(context)
        scores.append({
            "entity_id": d["entity_id"], "text": d["original_text"],
            "composite_confidence": d["composite_confidence"],
            "calibrated_score": score,
            "trapped": _is_coronary_segment_trap(entity, candidate_index, candidates)
                      or _is_short_alphanumeric_code(entity),
        })

    if skipped:
        print(f"Skipped (no reconstructable candidates): {dict(skipped)}\n")

    if not scores:
        print("Nothing scoreable -- cannot test the OOD assumption with this data.")
        return

    vals = [s["calibrated_score"] for s in scores if s["calibrated_score"] is not None]
    n_none = sum(1 for s in scores if s["calibrated_score"] is None)
    print(f"Scored {len(scores)} entities ({trapped_count} would additionally be "
          f"hard-trap-caught in production, scored anyway here for comparison).")
    print(f"calibrator.score() returned None for {n_none} of them.\n")

    if not vals:
        print("Every score came back None -- calibrator.score()'s own untrained/"
              "error-guard is firing on this feature shape. That is itself a "
              "data point: the model may be refusing to score something this "
              "far outside its training distribution.")
        conn.close()
        return

    vals_sorted = sorted(vals)
    n = len(vals_sorted)
    mean = sum(vals_sorted) / n
    median = vals_sorted[n // 2]
    n_above_threshold = sum(1 for v in vals if v >= CALIBRATED_AUTO_THRESHOLD)

    print(f"mean={mean:.4f}  median={median:.4f}  "
          f"min={vals_sorted[0]:.4f}  max={vals_sorted[-1]:.4f}")
    print(f"Would clear CALIBRATED_AUTO_THRESHOLD ({CALIBRATED_AUTO_THRESHOLD}): "
          f"{n_above_threshold}/{n} ({n_above_threshold/n*100:.1f}%)\n")

    print("Histogram (0.05-wide buckets):")
    buckets = collections.Counter()
    for v in vals:
        buckets[round(v * 20) / 20] += 1
    for b in sorted(buckets):
        bar = "#" * buckets[b]
        marker = "  <-- threshold" if abs(b - CALIBRATED_AUTO_THRESHOLD) < 0.025 else ""
        print(f"  {b:.2f}: {bar} ({buckets[b]}){marker}")

    print("\nSample of individual scores (first 15, sorted by score descending):")
    for s in sorted(scores, key=lambda x: -(x["calibrated_score"] or -1))[:15]:
        trap_flag = " [TRAP]" if s["trapped"] else ""
        print(f"  {s['text']!r:<40s} composite_confidence={s['composite_confidence']:.3f}  "
              f"calibrated_score={s['calibrated_score']}{trap_flag}")

    conn.close()


if __name__ == "__main__":
    main()
