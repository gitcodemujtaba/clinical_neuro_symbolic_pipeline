"""scripts/retrain_calibrator_full_corpus.py -- 2026-08-20. Retrain
ConsensusCalibrator on the FULL current corpus (114 notes with Stage 3
decisions, well past the ~100-note threshold noted as a planned follow-up),
compare val AUROC against the production baseline, and report whether it
should be adopted. Diagnostic by default -- does NOT overwrite
models/consensus_calibrator_v1.pkl unless explicitly told to save.

Reuses evaluation/tier_gate_cal_eval.py's proven build_labeled_examples()/
split_by_note()/fit_and_report() machinery rather than re-deriving it --
only the note_ids population differs (full corpus, not the hardcoded
31-note "overnight" set that module defaults to).

BASELINE_AUROC (2026-08-30 update): 0.845 -- the 114-note retrain this
exact script produced on 2026-08-20, which WAS adopted (per
memory/calibrator-retrain-at-100-notes.md) and is the actual currently-
deployed models/consensus_calibrator_v1.pkl as of this update (confirmed
live: its pickled metadata reads training_split=
"full_corpus_114_notes_2026-08-20"). The original 0.74 baseline this
script compared against was itself superseded by that same run and should
not still be the comparison target for any FUTURE retrain.

2026-08-30 (FEATURE_SET_VERSION=2): this run also feeds the new
kg3_confirmation_count feature (src.kg3_query.count_kg3_confirmations())
via a live Memgraph driver, best-effort -- see the driver setup in main()
below. The currently-deployed .pkl was fit under FEATURE_SET_VERSION=1
(16 features) and therefore ALWAYS reports "untrained (no-op)" when
loaded by any FEATURE_SET_VERSION=2 code now, regardless of this script --
running this retrain (with --save, after reviewing the numbers) is what
actually produces a usable 17-feature model again.
"""
import argparse
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402
from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from evaluation.splits import load_split  # noqa: E402
from evaluation.tier_gate_cal_eval import (  # noqa: E402
    build_labeled_examples, split_by_note, fit_and_report, FEATURE_NAMES,
    threshold_sweep, DEFAULT_MODEL_PATH)
from scripts.score_gold_recall import load_gold  # noqa: E402
from src.mollm_tier_calibrator import ConsensusCalibrator  # noqa: E402

DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
BASELINE_AUROC = 0.845  # the current production v1 calibrator's own val AUROC
                        # (114-note retrain, 2026-08-20 -- see this module's
                        # docstring for how this was confirmed still current)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true",
                    help="Overwrite the production calibrator if the new fit beats baseline.")
    args = ap.parse_args()

    conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=300)
    vocab = VocabularyRetriever(conn)

    from src.kg3_ingestion import get_memgraph_driver
    kg3_driver = None
    try:
        kg3_driver = get_memgraph_driver()
        with kg3_driver.session() as s:
            s.run("RETURN 1")
        print("Memgraph: reachable -- kg3_confirmation_count will use real data\n")
    except Exception as exc:
        print(f"Memgraph: NOT reachable ({exc}) -- kg3_confirmation_count will be 0 "
              f"for every example (still a valid fit, just no real signal on "
              f"the new feature)\n")
        kg3_driver = None

    all_tier4_notes = {r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM mollm_tier_gate_decisions WHERE tier = 'TIER_4_ENSEMBLE_SPLIT'"
    ).fetchall()}
    # 2026-08-31 FIX: previously used every TIER_4-bearing note unconditionally
    # -- confirmed live that 39 of 149 (26%) were from data/splits/note_splits.
    # csv's OFFICIAL locked test split (70 notes reserved for the T0/T1/T2
    # benchmark), meaning the deployed calibrator's own fit was contaminated
    # with locked-test-split gold outcomes as training labels. Excluded here,
    # same evaluation.splits.load_split() discipline every other evaluation
    # script in this codebase already follows.
    #
    # SECOND exclusion, found while re-verifying the first fix: fresh-10/
    # fresh-5 (ui/components/fresh10_notes.py, docs/FINAL_RESULTS_Single_
    # Source_Of_Truth.md S2/S10) are this project's own "genuinely held-out"
    # validation populations -- fresh-10 IS entirely inside the official
    # test split (so the exclusion above already covers it), but fresh-5's
    # real 5 notes are NOT (they're train/val-split notes used for a
    # different, date-based "never processed before" held-out property).
    # Confirmed live: without this second exclusion, today's retrain pulled
    # in 4 of fresh-5's 5 notes as training data the moment their own
    # 2026-08-30 validation run gave them TIER_4_ENSEMBLE_SPLIT decisions --
    # silently destroying the exact held-out property that made fresh-5
    # useful, the same failure mode the first fix exists to prevent, just
    # via a different mechanism (a project-specific holdout, not the
    # official split). Both are excluded explicitly so this calibrator can
    # still be honestly validated against either population later.
    from ui.components.fresh10_notes import FRESH10_NOTE_IDS
    FRESH5_NOTE_IDS = {"13397956-DS-5", "17739994-DS-31", "16410990-DS-12",
                       "16795604-DS-17", "17309807-DS-20"}
    locked_test_split = load_split("test") | set(FRESH10_NOTE_IDS) | FRESH5_NOTE_IDS
    all_notes = sorted(all_tier4_notes - locked_test_split)
    n_excluded = len(all_tier4_notes) - len(all_notes)
    print(f"{len(all_tier4_notes)} notes have TIER_4_ENSEMBLE_SPLIT decisions; "
         f"excluded {n_excluded} (locked test split + fresh-10/fresh-5 holdouts) -- "
         f"{len(all_notes)} notes remain as the calibrator's training population\n")

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, all_notes)
    import collections
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    examples = build_labeled_examples(conn, vocab, gold_by_note, note_ids=all_notes,
                                      kg3_driver=kg3_driver)
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
        today = datetime.date.today().isoformat()
        code_version = f"full_corpus_retrain_{today}"
        # 2026-08-30 fix: this date suffix was hardcoded to "2026-08-20" (the
        # day this script was first written) regardless of when it actually
        # ran -- caught live when a 2026-08-30 retrain saved metadata falsely
        # claiming it was trained on 2026-08-20. code_version two lines above
        # already used datetime.date.today() correctly; training_split now
        # does too, so the two can't drift apart again.
        calibrator.save(DEFAULT_MODEL_PATH, training_note_ids=train_notes,
                        training_split=f"full_corpus_{len(all_notes)}_notes_{today}",
                        code_version=code_version)
        print(f"\nSAVED to {DEFAULT_MODEL_PATH} (code_version={code_version}, "
              f"trained on {len(train_notes)} notes, val AUROC {auc:.3f} vs baseline {BASELINE_AUROC})")
    elif args.save:
        print("\n--save was passed but the new fit did NOT beat baseline -- NOT saved.")
    else:
        print("\nNOT saved to disk. Re-run with --save to overwrite the production calibrator "
              "if you decide to adopt this after reviewing the numbers above.")

    conn.close()
    if kg3_driver is not None:
        kg3_driver.close()


if __name__ == "__main__":
    main()
