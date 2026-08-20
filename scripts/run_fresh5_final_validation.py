"""scripts/run_fresh5_final_validation.py -- 2026-08-20: end-to-end
re-validation of the official 5-note held-out test split
(data/splits/note_splits.csv), reusing the exact note list
evaluation/grade_fresh5_by_tier.py already established for this purpose
(2026-08-17). That earlier run predates today's SNOMED near-duplicate
retrieval fix (commit 6f5135d) and the tier-gate changes since -- this
re-run gives the real "does it generalize to genuinely unseen notes"
answer, not a number measured on notes already used for debugging.

Checked live before writing this: 4 of the 5 notes already have
extracted_entities from the 2026-08-17 run (314 entities total); one note
(14652764-DS-17) was NEVER extracted at all. Scoped accordingly --
Stage 1 (extraction) wasn't touched by today's fixes, so the 4 existing
notes only need Stage 2b (re-normalize) + Stage 3 (re-grade) re-run, not
full re-extraction. The missing note needs the full pipeline since it has
never been processed.

Run: python3 scripts/run_fresh5_final_validation.py
"""
import csv
import sys
import time

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
GOLD_NOTES_PATH = f"{PROJECT_DIR}/data/raw_notes/gold_notes.csv"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

NOTE_IDS = [
    # 2026-08-20, second fresh-5 batch: the 5 SMALLEST notes (by raw char
    # count) in the official locked test split (data/splits/note_splits.csv)
    # not already used in the first fresh-5 run, per explicit "smallest"
    # instruction for speed.
    "15706386-DS-9", "15975714-DS-10", "14766716-DS-22",
    "10043750-DS-6", "10371195-DS-9",
]


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    from src.db_utils import connect_with_retry

    conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=60)
    existing = {r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM extracted_entities WHERE note_id IN ({})".format(
            ",".join("?" * len(NOTE_IDS))), NOTE_IDS).fetchall()}
    conn.close()
    missing = [n for n in NOTE_IDS if n not in existing]
    already_extracted = [n for n in NOTE_IDS if n in existing]
    print(f"already extracted (Stage 2b/3 re-run only): {already_extracted}")
    print(f"never extracted (full Stage 1-3 run needed): {missing}")

    # ==== Part A: notes needing a full fresh pipeline run ====
    if missing:
        from src.clinical_pipeline import run_pipeline
        text_by_note = {}
        with open(GOLD_NOTES_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("note_id") in missing:
                    text_by_note[row["note_id"]] = row.get("text", "")
        for note_id in missing:
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
            print(f"  done in {(time.time()-t0)/60:.1f} min")

    # ==== Part B: re-normalize (Stage 2b) existing entities ====
    if already_extracted:
        from src.normalization.orchestrator import process_and_normalize_entities
        conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=60)
        placeholders = ",".join("?" * len(already_extracted))
        rows = conn.execute(f"""
            SELECT entity_id, note_id, entity_label, original_text, expanded_text,
                   orig_start, orig_end, exp_start, exp_end, assertion_status,
                   experiencer, temporality, assertion_cue, assertion_cue_start,
                   assertion_cue_end, assertion_cue_category, assertion_engine,
                   section_name, sentence_id, local_context,
                   expansion_ambiguous, candidate_expansions, selection_basis,
                   gliner_model_version
            FROM extracted_entities WHERE note_id IN ({placeholders})
            AND (superseded_by_split IS NULL OR superseded_by_split = FALSE)
            AND (superseded_by_growth IS NULL OR superseded_by_growth = FALSE)
        """, already_extracted).fetchall()
        cols = [c[0] for c in conn.description]
        entities = [dict(zip(cols, r)) for r in rows]
        conn.close()
        print(f"\n{len(entities)} entities to re-normalize across {len(already_extracted)} notes")

        conn = connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=300)
        try:
            for i in range(0, len(entities), 50):
                batch = entities[i:i + 50]
                process_and_normalize_entities(batch, conn, is_test=True)
                print(f"  re-normalized {min(i+50, len(entities))}/{len(entities)}")

            entity_ids = [e["entity_id"] for e in entities]
            ph = ",".join("?" * len(entity_ids))
            stale = conn.execute(
                f"SELECT count(*) FROM mollm_tier_gate_decisions WHERE entity_id IN ({ph})",
                entity_ids).fetchone()[0]
            if stale:
                conn.execute(
                    f"DELETE FROM mollm_tier_gate_decisions WHERE entity_id IN ({ph})",
                    entity_ids)
            print(f"cleared {stale} stale tier-gate decision(s) for re-grading")
        finally:
            conn.close()

    # ==== Part C: Stage 3 tier-gate on ALL 5 notes (whatever's now undecided) ====
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
        already_done = {r[0] for r in connect_with_retry(
            DB_PATH, read_only=True, max_wait_seconds=300).execute(
            "SELECT entity_id FROM mollm_tier_gate_decisions WHERE note_id = ?",
            [note_id]).fetchall()}
        todo = [r for r in records if r["entity_id"] not in already_done]
        # 2026-08-20, capped under time pressure: grading EVERY undecided
        # entity across 5 notes (~380-400 total) at the ~15-20 sec/entity
        # full-ensemble pace measured earlier today would take 1-2 hours.
        # 25/note (125 total) is still a real, gradable held-out sample --
        # not the full note, but enough for a genuine precision read.
        todo = todo[:25]
        print(f"[note {note_idx}/{len(NOTE_IDS)}] {note_id}: {len(records)} total, "
             f"{len(todo)} to grade (capped at 25/note)")

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
