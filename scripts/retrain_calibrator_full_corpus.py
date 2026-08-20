"""scripts/retrain_calibrator_full_corpus.py -- 2026-08-20. Retrain
ConsensusCalibrator on the FULL current corpus (114 notes with Stage 3
decisions, well past the ~100-note threshold noted as a planned follow-up),
compare val AUROC against the production baseline (0.74), and report
whether it should be adopted. Diagnostic by default -- does NOT overwrite
models/consensus_calibrator_v1.pkl unless explicitly told to save.

Reuses evaluation/tier_gate_cal_eval.py's proven build_labeled_examples()/
split_by_note()/fit_and_report() machinery rather than re-deriving it --
only the note_ids population differs (full corpus, not the hardcoded
31-note "overnight" set that module defaults to).
"""
import argparse
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402
from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from evaluation.tier_gate_cal_eval import (  # noqa: E402
    build_labeled_examples, split_by_note, fit_and_report, FEATURE_NAMES,
    threshold_sweep, DEFAULT_MODEL_PATH)
from scripts.score_gold_recall import load_gold  # noqa: E402
from src.mollm_tier_calibrator import ConsensusCalibrator  # noqa: E402

DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
BASELINE_AUROC = 0.74  # the current production v1 calibrator's own val AUROC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="Overwrite the production calibrator if the new fit beats baseline.")
    args = ap.parse_args()

    conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=300)
    vocab = VocabularyRetriever(conn)

    all_notes = sorted(r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM mollm_tier_gate_decisions WHERE tier = 'TIER_4_ENSEMBLE_SPLIT'"
    ).fetchall())
    print(f"{len(all_notes)} notes have TIER_4_ENSEMBLE_SPLIT decisions (the calibrator's training population)\n")

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, all_notes)
    import collections
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    examples = build_labeled_examples(conn, vocab, gold_by_note, note_ids=all_notes)
    n_pos = sum(e["label"] for e in examples)
    print(f"labeled examples: {len(examples)}, base rate {n_pos}/{len(examples)} = {n_pos/len(examples)*100:.1f}% correct\n")

    train, val, train_notes, val_notes = split_by_note(examples)
    print(f"split: {len(train_notes)} train notes ({len(train)} examples), "
          f"{len(val_notes)} val notes ({len(val)} examples)\n")

    print("=" * 78)
    print("RETRAIN on full current corpus -- diagnostic, NOT saved yet")
    print("=" * 78)
    calibrator, val_scored, auc = fit_and_report(
        train, val, train_notes, label=f"RETRAIN ({len(all_notes)}-note corpus)",
        save_path=None, respect_hard_traps=True)

    print(f"\n{'='*78}")
    print(f"COMPARISON: new val AUROC = {auc}  vs  production baseline = {BASELINE_AUROC}")
    if auc is None:
        verdict = "INCONCLUSIVE (val set too small/one-class -- AUROC undefined)"
    elif auc > BASELINE_AUROC:
        verdict = f"BETTER by {auc - BASELINE_AUROC:.3f} -- candidate for adoption, pending review"
    elif auc >= BASELINE_AUROC - 0.01:
        verdict = "ROUGHLY EQUIVALENT (within 0.01) -- more data did not clearly help"
    else:
        verdict = f"WORSE by {BASELINE_AUROC - auc:.3f} -- do NOT adopt, matches the 2026-08-17 51-note finding"
    print(f"VERDICT: {verdict}")
    print("=" * 78)

    if args.save and auc is not None and auc > BASELINE_AUROC:
        import datetime
        code_version = f"full_corpus_retrain_{datetime.date.today().isoformat()}"
        calibrator.save(DEFAULT_MODEL_PATH, training_note_ids=train_notes,
                        training_split=f"full_corpus_{len(all_notes)}_notes_2026-08-20",
                        code_version=code_version)
        print(f"\nSAVED to {DEFAULT_MODEL_PATH} (code_version={code_version}, "
              f"trained on {len(train_notes)} notes, val AUROC {auc:.3f} vs baseline {BASELINE_AUROC})")
    elif args.save:
        print("\n--save was passed but the new fit did NOT beat baseline -- NOT saved.")
    else:
        print("\nNOT saved to disk. Re-run with --save to overwrite the production calibrator "
              "if you decide to adopt this after reviewing the numbers above.")

    conn.close()


if __name__ == "__main__":
    main()
