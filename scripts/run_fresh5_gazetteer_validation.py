"""scripts/run_fresh5_gazetteer_validation.py -- 2026-08-31: genuinely
fresh, real, end-to-end validation of the new GLiNER gazetteer-fallback
mechanism (src.gliner_gazetteer_fallback, CNSP_GLINER_GAZETTEER_FALLBACK)
on 5 short, never-before-processed notes -- same "smallest for speed,
genuinely unseen" methodology as scripts/run_fresh5_final_validation.py.

Confirmed live before writing this: none of these 5 notes have ever been
extracted (0 rows in extracted_entities), and none appear in any other
note-ID list used elsewhere this session (grade_overnight_corpus_run.py,
grade_fresh25_by_tier.py, grade_fresh5_by_tier.py, fresh10_notes.py) --
genuinely unseen, not just unseen by this specific mechanism.

Sets CNSP_GLINER_GAZETTEER_FALLBACK=1 BEFORE any project import, since
src.gliner_gazetteer_fallback.GAZETTEER_FALLBACK_ENABLED is read once at
import time (same convention as every other env-gated flag in this
codebase) -- setting it after import would silently no-op.

Run: python3 scripts/run_fresh5_gazetteer_validation.py
"""
import csv
import os
import sys
import time

os.environ.setdefault("CNSP_GLINER_GAZETTEER_FALLBACK", "1")

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
GOLD_NOTES_PATH = f"{PROJECT_DIR}/data/raw_notes/gold_notes.csv"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

NOTE_IDS = [
    # 2026-08-31, gazetteer-fallback fresh-5 batch: the 5 smallest gold
    # notes never processed at all (checked live against extracted_entities
    # AND every other note-ID list used this session before selection).
    "15285988-DS-7", "15906604-DS-2", "14809657-DS-15",
    "19015466-DS-9", "19884924-DS-14",
]

PER_NOTE_GRADE_CAP = 30  # same time-budget discipline as the earlier fresh5 run


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    from src.db_utils import connect_with_retry
    from src.gliner_gazetteer_fallback import GAZETTEER_FALLBACK_ENABLED
    print(f"CNSP_GLINER_GAZETTEER_FALLBACK active: {GAZETTEER_FALLBACK_ENABLED}")
    assert GAZETTEER_FALLBACK_ENABLED, "flag did not take -- check env var was set before import"

    conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=60)
    existing = {r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM extracted_entities WHERE note_id IN ({})".format(
            ",".join("?" * len(NOTE_IDS))), NOTE_IDS).fetchall()}
    conn.close()
    print(f"already extracted (should be none): {sorted(existing)}")

    # ==== Part A: full Stage 1-2b pipeline on all 5 (all genuinely new) ====
    from src.clinical_pipeline import run_pipeline
    text_by_note = {}
    with open(GOLD_NOTES_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("note_id") in NOTE_IDS:
                text_by_note[row["note_id"]] = row.get("text", "")

    gazetteer_counts = {}
    for note_id in NOTE_IDS:
        raw_text = text_by_note.get(note_id)
        if not raw_text:
            print(f"  SKIP {note_id}: not found in {GOLD_NOTES_PATH}")
            continue
        print(f"  running full pipeline for {note_id} ({len(raw_text)} chars)...")
        t0 = time.time()
        conn = connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=300)
        try:
            run_pipeline(note_id, raw_text, conn, is_test=True)
        finally:
            conn.close()
        conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=60)
        n_gaz = conn.execute(
            "SELECT COUNT(*) FROM extracted_entities WHERE note_id = ? "
            "AND extraction_source = 'gazetteer_fallback_gliner_miss'", [note_id]).fetchone()[0]
        n_total = conn.execute(
            "SELECT COUNT(*) FROM extracted_entities WHERE note_id = ?", [note_id]).fetchone()[0]
        conn.close()
        gazetteer_counts[note_id] = (n_gaz, n_total)
        print(f"  done in {(time.time()-t0)/60:.1f} min -- {n_gaz}/{n_total} entities "
             f"from the gazetteer fallback")

    print(f"\ngazetteer contribution by note: {gazetteer_counts}")
    print(f"total gazetteer-recovered entities: {sum(g for g, t in gazetteer_counts.values())}")

    # ==== Part B: Stage 3 tier-gate on all 5 notes ====
    from src.mollm_ensemble import load_validation_records
    from src.mollm_tier_gate import build_clients, route_tier, store_tier_decision
    from src.mollm_tier_calibrator import ConsensusCalibrator, DEFAULT_MODEL_PATH

    calibrator = ConsensusCalibrator.load(DEFAULT_MODEL_PATH, scoring_note_ids=NOTE_IDS)
    print(f"\ncalibrator: {'fitted' if calibrator.model is not None else 'untrained (no-op)'}")
    clients = build_clients()

    n_processed, n_errors = 0, 0
    start_time = time.time()
    for note_idx, note_id in enumerate(NOTE_IDS, 1):
        note_conn = connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=1800)
        try:
            records = load_validation_records(note_conn, note_id, tier=None)
        finally:
            note_conn.close()
        todo = records[:PER_NOTE_GRADE_CAP]
        print(f"[note {note_idx}/{len(NOTE_IDS)}] {note_id}: {len(records)} total, "
             f"{len(todo)} to grade (capped at {PER_NOTE_GRADE_CAP}/note)")

        for rec in todo:
            elapsed_min = (time.time() - start_time) / 60
            try:
                decision = route_tier(
                    rec, clients=clients, calibrator=calibrator,
                    conn_factory=lambda: connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=1800))
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
                print(f"  [{elapsed_min:.1f}m] {n_processed} processed so far...")

    print(f"\ndone: {n_processed} processed, {n_errors} errors, "
         f"{(time.time()-start_time)/60:.1f} min total")


if __name__ == "__main__":
    main()
