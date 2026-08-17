"""
evaluation/tier_gate_cal_eval.py -- Phase 6 build-order steps 3-4: builds
training data for src.mollm_tier_calibrator.ConsensusCalibrator from the
2026-08-17 overnight corpus run's mollm_tier_gate_decisions, fits it, and
validates on a held-out, note-disjoint split.

TRAINING POPULATION. Only TIER_4_ENSEMBLE_SPLIT decisions -- NOT the AUTO
tiers (route_tier() never consults the calibrator for those; training on
them would teach the model an input distribution it never sees at inference)
and NOT TIER_5_TRUE_AMBIGUITY (a unanimous NONE_CORRECT plurality has no
candidate to promote, so route_tier() never reaches the calibrator-
consultation block for that shape either -- see src/mollm_tier_gate.py's own
comment at the calibrator call site). This mirrors exactly the population
evaluation/grade_overnight_corpus_run.py's "Tier 4 shadow precision" section
measured (668 clean-span gradable, 470 correct = 70.4%) -- reuses that
script's plurality_candidate_index() rather than re-deriving it, so the
label definition here and the shadow-precision number already reported can
never silently diverge.

LABELS. 1 if the plurality-vote candidate (the same one route_tier() would
promote to TIER_1B on a high enough calibrator score) matches gold via the
established SNOMED-crosswalk + clean-span methodology; 0 otherwise.
Compound-span, no-gold-overlap, narrower-than-gold, and no-plurality-
candidate (NONE_CORRECT) rows are excluded -- unlabelable, not 0.

SPLIT. Strictly by note_id, not by row -- entities inside one discharge
note are not independent draws, and the calibrator's own load()-time
leakage guard (trained_on_any_of()) checks note_id overlap specifically, so
partitioning any other way would defeat that guard's purpose. Deterministic
(sorted note_ids, held-out = every 4th), not random, so a re-run is
reproducible without a stored seed.

Read-only against the DB except for the final model .pkl written to disk.
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
from evaluation.grade_overnight_corpus_run import (  # noqa: E402
    NOTE_IDS, KNOWN_GOLD_ERRORS, plurality_candidate_index)
from scripts.score_gold_recall import load_gold, overlaps  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402
from src.mollm_tier_calibrator import (  # noqa: E402
    ConsensusCalibrator, DEFAULT_MODEL_PATH, FEATURE_NAMES, build_feature_context,
    count_prior_confirmations, featurize)
from src.mollm_tier_gate import _is_coronary_segment_trap, _is_short_alphanumeric_code  # noqa: E402


def _code_version():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_DIR,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def build_labeled_examples(conn, vocab, gold_by_note):
    """Returns a list of dicts: note_id, text, label (1/0), feature_context,
    vector (pre-featurized, so fitting doesn't re-run featurize() per split),
    plus diagnostic fields (top_verdict, vote_counts) for the printed report.
    """
    note_ph = ",".join("?" * len(NOTE_IDS))
    rows = conn.execute(f"""
        SELECT d.entity_id, d.note_id, d.models,
               e.original_text, e.expanded_text, e.entity_label,
               e.orig_start, e.orig_end,
               n.candidates, n.match_tier, n.similarity_score, n.is_ambiguous,
               n.domain_conflict, n.normalized_from, n.omop_concept_id
        FROM mollm_tier_gate_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.note_id IN ({note_ph}) AND d.tier = 'TIER_4_ENSEMBLE_SPLIT'
    """, NOTE_IDS).fetchall()
    cols = [c[0] for c in conn.description]
    decisions = [dict(zip(cols, row)) for row in rows]

    skipped = collections.Counter()
    examples = []
    for d in decisions:
        note_id = d["note_id"]
        gold = gold_by_note.get(note_id, [])
        overlapping = [g for g in gold
                       if overlaps(d["orig_start"], d["orig_end"], g["start"], g["end"])]
        if not overlapping:
            skipped["no_gold_overlap"] += 1
            continue
        if len(overlapping) != 1:
            skipped["compound_span"] += 1
            continue
        g0 = overlapping[0]
        if (d["orig_end"] - d["orig_start"]) < (g0["end"] - g0["start"]):
            skipped["narrower_than_gold"] += 1
            continue

        idx, top_verdict, vote_counts = plurality_candidate_index(d["models"])
        if idx is None:
            skipped["no_candidate"] += 1
            continue

        candidates = d["candidates"]
        if isinstance(candidates, str):
            candidates = json.loads(candidates)
        i = idx - 1
        if candidates is None or i < 0 or i >= len(candidates):
            skipped["candidate_index_out_of_range"] += 1
            continue
        chosen = candidates[i]
        concept_id = chosen.get("omop_concept_id") or chosen.get("concept_id")

        pred_code = vocab.snomed_code_for_concept(concept_id) if concept_id else None
        gold_code = g0["concept_id"]
        is_gold_error = (note_id, str(gold_code)) in KNOWN_GOLD_ERRORS
        label = 1 if ((pred_code is not None and str(pred_code) == str(gold_code))
                       or is_gold_error) else 0

        # Reconstruct the "entity" shape build_feature_context()/featurize()
        # expect (see src/mollm_tier_calibrator.py's FEATURE_NAMES) directly
        # from normalized_entities' own columns -- this is exactly what
        # route_tier() has in hand at its calibrator call site, not a re-
        # derivation of it.
        entity = {
            "candidates": candidates,
            "match_tier": d["match_tier"],
            "is_ambiguous": d["is_ambiguous"],
            "domain_conflict": d["domain_conflict"],
            "normalized_from": d["normalized_from"],
            "expansion_ambiguous": False,  # not carried on normalized_entities; safe default
            "original_text": d["original_text"],
        }
        models = d["models"]
        if isinstance(models, str):
            models = json.loads(models)
        prior_count = count_prior_confirmations(conn, d["original_text"], concept_id)
        context = build_feature_context(entity, models, prior_count)

        examples.append({
            "note_id": note_id, "text": d["original_text"], "label": label,
            "vector": featurize(context), "top_verdict": top_verdict,
            "vote_counts": dict(vote_counts or {}), "prior_confirmation_count": prior_count,
            "is_trapped": (_is_coronary_segment_trap(entity, idx, candidates)
                          or _is_short_alphanumeric_code(entity)),
        })

    print(f"labeled examples built: {len(examples)}  (skipped: {dict(skipped)})")
    return examples


def split_by_note(examples, holdout_every=4):
    """Deterministic, note-disjoint split. Every `holdout_every`-th note
    (by sorted note_id) goes to val; the rest to train. Not random, so a
    re-run reproduces the same split without a stored seed.
    """
    notes = sorted({e["note_id"] for e in examples})
    val_notes = set(notes[holdout_every - 1::holdout_every])
    train_notes = set(notes) - val_notes
    train = [e for e in examples if e["note_id"] in train_notes]
    val = [e for e in examples if e["note_id"] in val_notes]
    return train, val, sorted(train_notes), sorted(val_notes)


def threshold_sweep(val, thresholds, respect_hard_traps=False):
    """For each threshold: how many val examples score >= it (coverage of
    the val Tier-4 population), and precision among those. This is exactly
    what CALIBRATED_AUTO_THRESHOLD trades off in route_tier().

    respect_hard_traps=True mirrors src.mollm_tier_gate.route_tier()'s
    actual production behavior: a trapped entity (coronary-segment OR
    short-alphanumeric-code) is never promoted regardless of score, since
    the real gate bypasses calibrator.score() for it entirely -- this
    reproduces that exclusion for the val sweep rather than just filtering
    by score after the fact.
    """
    rows = []
    for t in thresholds:
        promoted = [e for e in val if e["score"] is not None and e["score"] >= t
                   and not (respect_hard_traps and e.get("is_trapped"))]
        n = len(promoted)
        correct = sum(1 for e in promoted if e["label"] == 1)
        rows.append({
            "threshold": t, "n_promoted": n, "n_val": len(val),
            "coverage_pct": round(n / len(val) * 100, 1) if val else None,
            "precision_pct": round(correct / n * 100, 1) if n else None,
        })
    return rows


def fit_and_report(train, val, train_notes, ablate_indices=(), label="FULL",
                   save_path=None, fp_threshold=0.65, respect_hard_traps=False):
    """Fits one ConsensusCalibrator and prints its val report. ablate_indices
    zeroes those feature positions in every vector before fit/score -- for a
    linear model this is equivalent to dropping the feature entirely (a
    constant-zero column has no discriminative signal to learn from and
    contributes nothing to predict_proba()), without touching
    FEATURE_NAMES/FEATURE_SET_VERSION in the production module for what is
    explicitly a diagnostic ablation, not a feature-set change.
    """
    import copy
    import numpy as np
    from sklearn.metrics import roc_auc_score

    def _ablated(vector):
        v = list(vector)
        for i in ablate_indices:
            v[i] = 0.0
        return v

    train = copy.deepcopy(train)
    val = copy.deepcopy(val)
    for e in train:
        e["vector"] = _ablated(e["vector"])
    for e in val:
        e["vector"] = _ablated(e["vector"])

    calibrator = ConsensusCalibrator()
    calibrator.fit_vectors([e["vector"] for e in train], [e["label"] for e in train],
                           min_examples=100)

    model = calibrator.model
    classes = list(model.classes_)
    positive_index = classes.index(1) if 1 in classes else 1
    val_vectors = np.array([e["vector"] for e in val])
    val_probs = model.predict_proba(val_vectors)[:, positive_index]
    for e, p in zip(val, val_probs):
        e["score"] = round(float(p), 6)

    try:
        auc = roc_auc_score([e["label"] for e in val], [e["score"] for e in val])
    except ValueError:
        auc = None

    print("=" * 78)
    print(f"MODEL: {label}  (fitted on {calibrator.n_training_examples} train examples, "
          f"ablated feature indices={list(ablate_indices)}, "
          f"hard_traps={'ON' if respect_hard_traps else 'off'})")
    print("=" * 78)
    print(f"val AUROC: {auc}")
    print(f"{'thresh':>8s} {'n_promoted':>11s} {'coverage%':>10s} {'precision%':>11s}")
    for row in threshold_sweep(val, [round(0.5 + 0.05 * i, 2) for i in range(10)],
                              respect_hard_traps=respect_hard_traps):
        print(f"{row['threshold']:>8.2f} {row['n_promoted']:>11d} "
              f"{row['coverage_pct']:>9.1f}% {row['precision_pct']!s:>11s}")

    print(f"\n--- {label}: false positives at threshold={fp_threshold} ---")
    fps = [e for e in val if e["score"] is not None and e["score"] >= fp_threshold
           and e["label"] == 0
           and not (respect_hard_traps and e.get("is_trapped"))]
    if not fps:
        print("  (none)")
    for e in fps:
        print(f"  [{e['note_id']}] {e['text']!r} score={e['score']} "
              f"votes={e['vote_counts']} prior_count={e['prior_confirmation_count']}")

    if save_path:
        code_version = _code_version()
        calibrator.save(save_path, training_note_ids=train_notes,
                        training_split="overnight_2026-08-17_train", code_version=code_version)
        print(f"\nsaved to {save_path} (code_version={code_version})")

    return calibrator, val, auc


def main():
    conn = duckdb.connect(DB_PATH, read_only=True)
    vocab = VocabularyRetriever(conn)

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, NOTE_IDS)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    examples = build_labeled_examples(conn, vocab, gold_by_note)
    n_pos = sum(e["label"] for e in examples)
    print(f"base rate: {n_pos}/{len(examples)} = {n_pos/len(examples)*100:.1f}% correct\n")

    train, val, train_notes, val_notes = split_by_note(examples)
    print(f"split: {len(train_notes)} train notes ({len(train)} examples), "
          f"{len(val_notes)} val notes ({len(val)} examples)")
    print(f"  train notes: {train_notes}")
    print(f"  val notes:   {val_notes}")
    n_pos_train = sum(e["label"] for e in train)
    n_pos_val = sum(e["label"] for e in val)
    print(f"  train base rate: {n_pos_train}/{len(train)} = {n_pos_train/len(train)*100:.1f}%")
    print(f"  val base rate:   {n_pos_val}/{len(val)} = {n_pos_val/len(val)*100:.1f}%\n")

    n_trapped_val = sum(1 for e in val if e["is_trapped"])
    print(f"hard-trap (coronary + short-code) entities in val: {n_trapped_val}/{len(val)}\n")

    calibrator, val_full, auc = fit_and_report(
        train, val, train_notes, ablate_indices=(), label="FULL (16 features), hard traps OFF")

    prior_idx = FEATURE_NAMES.index("prior_confirmation_count")
    fit_and_report(
        train, val, train_notes, ablate_indices=(prior_idx,),
        label="ABLATED (prior_confirmation_count zeroed) -- diagnostic only, not saved")

    # The production candidate: FULL features (ablation showed dropping
    # prior_confirmation_count doesn't fix the coronary false positives and
    # measurably costs precision elsewhere) + the coronary trap gating
    # calibrator.score() at inference time. No re-fit needed -- the trap
    # changes ROUTING, not training, so this reuses val_full's already-fitted
    # scores and just re-sweeps with the trap's exclusion applied, exactly
    # mirroring what route_tier() itself now does.
    print("\n" + "=" * 78)
    print("FULL MODEL + HARD SAFETY TRAPS (coronary + short-code) -- the production candidate")
    print("=" * 78)
    print(f"{'thresh':>8s} {'n_promoted':>11s} {'coverage%':>10s} {'precision%':>11s}")
    for row in threshold_sweep(val_full, [round(0.5 + 0.05 * i, 2) for i in range(10)],
                              respect_hard_traps=True):
        print(f"{row['threshold']:>8.2f} {row['n_promoted']:>11d} "
              f"{row['coverage_pct']:>9.1f}% {row['precision_pct']!s:>11s}")

    print("\n--- projected corpus-wide impact (1,629 Tier-4 entities), trap ON ---")
    for t in [0.55, 0.60, 0.65, 0.70]:
        row = [r for r in threshold_sweep(val_full, [t], respect_hard_traps=True)][0]
        if row["coverage_pct"] is not None:
            projected = round(1629 * row["coverage_pct"] / 100)
            print(f"  threshold={t}: val coverage {row['coverage_pct']}%, precision "
                  f"{row['precision_pct']}% -> projected ~{projected} of 1,629 entities "
                  f"promotable corpus-wide")

    code_version = _code_version()
    calibrator.save(DEFAULT_MODEL_PATH, training_note_ids=train_notes,
                    training_split="overnight_2026-08-17_train", code_version=code_version)
    print(f"\nsaved FULL (untrapped-training, trap applied at inference by route_tier() "
          f"itself) calibrator to {DEFAULT_MODEL_PATH} (code_version={code_version})")


if __name__ == "__main__":
    main()
