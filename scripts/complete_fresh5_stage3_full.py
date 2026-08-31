"""scripts/complete_fresh5_stage3_full.py -- 2026-08-31/09-01: grades
EVERY remaining Stage-2b-eligible entity in both fresh-5 batches through
Stage 3 -- no per-note cap. Both prior runs (scripts/run_fresh5_final_
validation.py's own history for the original batch, scripts/run_fresh5_
gazetteer_validation.py's PER_NOTE_GRADE_CAP=30 for the gazetteer batch)
stopped early for time-budget reasons, leaving a real, non-full-coverage
sample. This finishes both to genuine full coverage so the two batches
(and any comparison against them, e.g. the Evaluation Metrics page's
Batch comparison tab) reflect the whole note, not a capped slice.

Confirmed live before writing this (2026-09-01):
  Fresh-5 original:   453 eligible, 373 already decided, 80 remaining.
  Fresh-5 gazetteer:  634 eligible, 150 already decided, 364 remaining.

Idempotent -- only processes entities with no existing
mollm_tier_gate_decisions row, so a re-run after an interruption picks up
exactly where it left off, same discipline as every other Stage 3 runner
in this codebase.

Run: python3 scripts/complete_fresh5_stage3_full.py
"""
import sys
import time

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

FRESH5_ORIGINAL_NOTE_IDS = [
    "13397956-DS-5", "17739994-DS-31", "16410990-DS-12",
    "16795604-DS-17", "17309807-DS-20",
]
FRESH5_GAZETTEER_NOTE_IDS = [
    "15285988-DS-7", "15906604-DS-2", "14809657-DS-15",
    "19015466-DS-9", "19884924-DS-14",
]


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    from src.db_utils import connect_with_retry
    from src.mollm_ensemble import load_validation_records
    from src.mollm_tier_gate import build_clients, route_tier, store_tier_decision
    from src.mollm_tier_calibrator import ConsensusCalibrator, DEFAULT_MODEL_PATH

    all_note_ids = FRESH5_ORIGINAL_NOTE_IDS + FRESH5_GAZETTEER_NOTE_IDS
    calibrator = ConsensusCalibrator.load(DEFAULT_MODEL_PATH, scoring_note_ids=all_note_ids)
    print(f"calibrator: {'fitted' if calibrator.model is not None else 'untrained (no-op)'}")
    clients = build_clients()

    n_processed, n_errors = 0, 0
    start_time = time.time()

    for batch_label, note_ids in [("Fresh-5 original (2026-08-30)", FRESH5_ORIGINAL_NOTE_IDS),
                                  ("Fresh-5 gazetteer (2026-08-31)", FRESH5_GAZETTEER_NOTE_IDS)]:
        print(f"\n=== {batch_label} ===")
        for note_idx, note_id in enumerate(note_ids, 1):
            note_conn = connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=1800)
            try:
                records = load_validation_records(note_conn, note_id, tier=None)
            finally:
                note_conn.close()
            already_done = {r[0] for r in connect_with_retry(
                DB_PATH, read_only=True, max_wait_seconds=300).execute(
                "SELECT entity_id FROM mollm_tier_gate_decisions WHERE note_id = ?",
                [note_id]).fetchall()}
            todo = [r for r in records if r["entity_id"] not in already_done]
            print(f"[{batch_label} {note_idx}/{len(note_ids)}] {note_id}: "
                 f"{len(records)} total, {len(already_done)} already decided, "
                 f"{len(todo)} remaining -- NO cap, processing all of them")

            for rec in todo:
                elapsed_min = (time.time() - start_time) / 60
                try:
                    decision = route_tier(
                        rec, clients=clients, calibrator=calibrator,
                        conn_factory=lambda: connect_with_retry(
                            DB_PATH, read_only=False, max_wait_seconds=1800))
                    write_conn = connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=1800)
                    try:
                        decision = store_tier_decision(decision, rec["entity_id"], note_id,
                                                       write_conn, is_test=True)
                    finally:
                        write_conn.close()
                except Exception as exc:
                    n_errors += 1
                    print(f"    ERROR on {rec['original_text']!r}: {exc.__class__.__name__}: {exc}")
                    continue
                n_processed += 1
                if n_processed % 10 == 0:
                    print(f"  [{elapsed_min:.1f}m] {n_processed} processed so far "
                         f"({n_errors} errors)...")

    print(f"\ndone: {n_processed} processed, {n_errors} errors, "
         f"{(time.time()-start_time)/60:.1f} min total")


if __name__ == "__main__":
    main()
