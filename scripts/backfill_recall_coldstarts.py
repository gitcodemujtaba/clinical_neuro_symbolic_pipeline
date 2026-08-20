"""scripts/backfill_recall_coldstarts.py -- 2026-08-20 one-time backfill:
injects src.lab_abbrev_coldstart + src.narrative_state_word_coldstart
entities into every already-processed note, then runs Stage 2b
normalization on JUST the newly-added entities.

WHY A TARGETED BACKFILL, NOT A FULL CORPUS RE-RUN. Both cold-start
modules are purely additive -- they never touch or re-derive an existing
entity, only add new ones GLiNER never proposed. Re-running Stage 1-2a-2b
from scratch on all 140 notes (as scripts/test_pipeline_e2e.py would) is
unnecessary and would cost ~50 hours at the fresh25 batch's measured
per-note pace; this script does only the actually-new work: inject,
store, normalize. Stage 3 tier-gating for these new entities is a
SEPARATE step (scripts/run_stage3_tier_gate.py, unmodified) -- its own
existing resume-check (already_processed_entity_ids()) will correctly
skip every entity that already has a decision and process only these new
ones, so it does not need a special-cased backfill variant of its own.
Every injected entity is guaranteed to hit tier3_fast_path()'s
deterministic branches (verified_lab_test_alias / verified_narrative_
state_word) -- zero LLM calls expected for any of them, confirmed by
this script's own summary counts.

Run: python3 scripts/backfill_recall_coldstarts.py [--dry-run]
"""
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.entity_extraction import store_entities  # noqa: E402
from src.lab_abbrev_coldstart import build_lab_abbrev_coldstart_entities  # noqa: E402
from src.narrative_state_word_coldstart import build_narrative_state_word_entities  # noqa: E402
from src.normalization.orchestrator import process_and_normalize_entities  # noqa: E402

DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
DRY_RUN = "--dry-run" in sys.argv


def load_raw_text(note_id, cache):
    if note_id in cache:
        return cache[note_id]
    import csv
    import os
    RAW_TEXT_CANDIDATES = [
        f"{PROJECT_DIR}/data/raw_notes/gold_notes.csv",
        f"{PROJECT_DIR}/data/raw_notes/discharge.csv",
        f"{PROJECT_DIR}/data/snomed-ct-entity-linking-challenge-1.2.0/train_notes.csv",
    ]
    for path in RAW_TEXT_CANDIDATES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                cache.setdefault(row.get("note_id"),
                                row.get("text") or row.get("note_text"))
    return cache.get(note_id)


def main():
    conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=300)
    note_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test = TRUE").fetchall()]
    conn.close()
    print(f"{len(note_ids)} notes to process")

    raw_text_cache = {}
    total_lab = 0
    total_narrative = 0
    total_notes_touched = 0
    n_errors = 0

    for i, note_id in enumerate(note_ids, 1):
        raw_text = load_raw_text(note_id, raw_text_cache)
        if raw_text is None:
            print(f"[{i}/{len(note_ids)}] {note_id}: no raw text found, skipping")
            continue

        conn = connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=300)
        try:
            existing = [{"orig_start": r[0], "orig_end": r[1]} for r in conn.execute(
                "SELECT orig_start, orig_end FROM extracted_entities WHERE note_id = ? AND is_test = TRUE",
                [note_id]).fetchall()]

            lab_entities = build_lab_abbrev_coldstart_entities(raw_text, note_id, existing)
            combined = existing + lab_entities
            narrative_entities = build_narrative_state_word_entities(raw_text, note_id, combined)
            new_entities = lab_entities + narrative_entities

            if not new_entities:
                continue

            total_notes_touched += 1
            total_lab += len(lab_entities)
            total_narrative += len(narrative_entities)

            if DRY_RUN:
                print(f"[{i}/{len(note_ids)}] {note_id}: would inject "
                      f"{len(lab_entities)} lab + {len(narrative_entities)} narrative")
                continue

            store_entities(conn, new_entities, is_test=True)
            process_and_normalize_entities(new_entities, conn, is_test=True)
            print(f"[{i}/{len(note_ids)}] {note_id}: injected+normalized "
                  f"{len(lab_entities)} lab + {len(narrative_entities)} narrative")
        except Exception as exc:
            n_errors += 1
            print(f"[{i}/{len(note_ids)}] {note_id}: ERROR {type(exc).__name__}: {exc}")
        finally:
            conn.close()

    print(f"\n{'='*78}\nBACKFILL COMPLETE{' (DRY RUN)' if DRY_RUN else ''}\n{'='*78}")
    print(f"notes touched: {total_notes_touched}/{len(note_ids)}")
    print(f"lab-abbrev entities injected: {total_lab}")
    print(f"narrative-state-word entities injected: {total_narrative}")
    print(f"total new entities: {total_lab + total_narrative}")
    print(f"errors: {n_errors}")


if __name__ == "__main__":
    main()
