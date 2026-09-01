"""scripts/run_fresh5_calibrator_unseen_stage12.py -- 2026-09-01: Stage
1->2a->2b only (no Stage 3 LLM calls yet) for a genuinely fresh 5-note
validation batch, selected to satisfy three real constraints
simultaneously, verified live before picking:
  1. In the OFFICIAL LOCKED TEST SPLIT (data/splits/note_splits.csv via
     evaluation.splits.load_split("test")) -- not the unrelated is_test
     DB column.
  2. NEVER processed before (zero existing extracted_entities rows) --
     a genuinely fresh end-to-end run, not a re-grade of old output.
  3. NOT among the 75 notes ConsensusCalibrator (models/consensus_
     calibrator_v1.pkl) was trained on -- so a Stage 3 run afterward
     exercises the calibrator on truly unseen data, not data its own
     leakage guard would silently no-op on.
31 real notes satisfied all three; the 5 with the fewest gold annotations
(136-193 each) were picked, matching this project's own "short notes"
convention for a fast validation batch (see the two prior Fresh-5
batches in ui/components/note_batches.py).

Run: python3 scripts/run_fresh5_calibrator_unseen_stage12.py
"""
import sys
import time

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# The real, verified-live selection -- see module docstring for the exact
# three-constraint query that produced this list.
FRESH5_CALIBRATOR_UNSEEN_NOTE_IDS = [
    "14050425-DS-5", "14702741-DS-7", "18752997-DS-9",
    "17665522-DS-2", "17743133-DS-8",
]


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    from src.db_utils import connect_with_retry
    from src.clinical_pipeline import run_pipeline
    from scripts.refix_narrative_coldstart_marker import load_raw_text_for_notes

    raw_text_by_note = load_raw_text_for_notes(FRESH5_CALIBRATOR_UNSEEN_NOTE_IDS)
    missing = [n for n in FRESH5_CALIBRATOR_UNSEEN_NOTE_IDS if n not in raw_text_by_note]
    if missing:
        print(f"WARNING: no raw text found for {missing} -- skipping those")

    start = time.time()
    for note_id in FRESH5_CALIBRATOR_UNSEEN_NOTE_IDS:
        raw_text = raw_text_by_note.get(note_id)
        if not raw_text:
            continue
        conn = connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=300)
        try:
            result = run_pipeline(note_id, raw_text, conn, is_test=True)
        finally:
            conn.close()
        n_entities = len(result.get("entities", []))
        elapsed = time.time() - start
        print(f"[{elapsed:.1f}s] {note_id}: {len(raw_text)} chars, "
              f"{n_entities} entities extracted (Stage 2a)")

    # Summarize real Stage 2a/2b eligibility counts, matching this
    # project's own established eligibility criteria (below_threshold/
    # superseded exclusions) rather than a raw row count.
    conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=300)
    note_ph = ",".join("?" * len(FRESH5_CALIBRATOR_UNSEEN_NOTE_IDS))
    n_extracted = conn.execute(
        f"SELECT count(*) FROM extracted_entities WHERE note_id IN ({note_ph})",
        FRESH5_CALIBRATOR_UNSEEN_NOTE_IDS).fetchone()[0]
    n_stage2b_eligible = conn.execute(f"""
        SELECT count(*) FROM extracted_entities e
        WHERE e.note_id IN ({note_ph}) AND e.is_test = TRUE
          AND (e.superseded_by_split IS NULL OR e.superseded_by_split = FALSE)
          AND (e.superseded_by_growth IS NULL OR e.superseded_by_growth = FALSE)
          AND (e.below_threshold IS NULL OR e.below_threshold = FALSE)
    """, FRESH5_CALIBRATOR_UNSEEN_NOTE_IDS).fetchone()[0]
    n_normalized = conn.execute(f"""
        SELECT count(*) FROM extracted_entities e
        JOIN normalized_entities n ON n.entity_id = e.entity_id
        WHERE e.note_id IN ({note_ph}) AND e.is_test = TRUE
          AND (e.superseded_by_split IS NULL OR e.superseded_by_split = FALSE)
          AND (e.superseded_by_growth IS NULL OR e.superseded_by_growth = FALSE)
          AND (e.below_threshold IS NULL OR e.below_threshold = FALSE)
    """, FRESH5_CALIBRATOR_UNSEEN_NOTE_IDS).fetchone()[0]
    conn.close()

    print(f"\n=== Summary ===")
    print(f"total extracted_entities rows (all, incl. below-threshold): {n_extracted}")
    print(f"Stage 2b-eligible (accepted, not superseded): {n_stage2b_eligible}")
    print(f"normalized (have a normalized_entities row -- ready for Stage 3): {n_normalized}")
    print(f"total time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
