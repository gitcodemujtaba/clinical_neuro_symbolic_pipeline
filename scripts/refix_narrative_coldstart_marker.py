"""scripts/refix_narrative_coldstart_marker.py -- 2026-08-20 second-order
fix: scripts/refix_coldstart_hierarchy_collapse.py (itself a fix for the
_collapse_hierarchy_duplicates() bug) reconstructed entity dicts by
SELECTing them back from extracted_entities -- but the `narrative_coldstart`
bypass marker (src.narrative_state_word_coldstart's whole mechanism for
skipping Tier 1-3 search entirely) only ever existed in-memory, in the
dict build_narrative_state_word_entities() itself returns, never as a
real extracted_entities column (confirmed: same as local_context_basis,
silently dropped by store_entities() since it isn't in the table schema).
Reconstructing from a DB SELECT therefore silently stripped the marker,
which knocked every narrative-state-word cold-start entity off its
deterministic bypass and onto ordinary Tier 1-3 search instead --
confirmed live: entities that should carry exactly 1 candidate
(match_basis="verified_narrative_state_word") were showing 5-6 candidates
via plain semantic_similarity.

Lab-abbreviation cold-start entities are NOT affected by this specific
issue (they never used a marker-field bypass -- they rely on
_LAB_TEST_ALIASES, a DB-independent lookup keyed on text alone, reused
correctly regardless of how the entity dict was reconstructed).

Fix: re-normalize narrative-state-word entities using the REAL builder
(build_narrative_state_word_entities()) called fresh per note, not
reconstructed from a DB row -- the only way to correctly regenerate the
marker field.

Run: python3 scripts/refix_narrative_coldstart_marker.py
"""
import csv
import os
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.narrative_state_word_coldstart import build_narrative_state_word_entities  # noqa: E402
from src.normalization.orchestrator import process_and_normalize_entities  # noqa: E402

RAW_TEXT_CANDIDATES = [
    f"{PROJECT_DIR}/data/raw_notes/gold_notes.csv",
    f"{PROJECT_DIR}/data/raw_notes/discharge.csv",
    f"{PROJECT_DIR}/data/snomed-ct-entity-linking-challenge-1.2.0/train_notes.csv",
]


def load_raw_text_for_notes(note_ids):
    wanted = set(note_ids)
    out = {}
    for path in RAW_TEXT_CANDIDATES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                nid = row.get("note_id")
                if nid in wanted and nid not in out:
                    out[nid] = row.get("text") or row.get("note_text")
        if len(out) == len(wanted):
            break
    return out


def main():
    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb",
                              read_only=True, max_wait_seconds=300)
    note_ids = [r[0] for r in conn.execute("""
        SELECT DISTINCT note_id FROM extracted_entities
        WHERE gliner_model_version = 'narrative_state_word_coldstart' AND is_test = TRUE
    """).fetchall()]
    conn.close()  # 2026-08-20: must close before opening any read_only=False
                  # connection below -- DuckDB refuses a second connection to
                  # the same file with a different configuration while this
                  # one is still open (confirmed live: ConnectionException).
    print(f"{len(note_ids)} notes with narrative-state-word cold-start entities")

    raw_text_by_note = load_raw_text_for_notes(note_ids)
    print(f"raw text loaded for {len(raw_text_by_note)}/{len(note_ids)} notes")

    total_reentities = 0
    total_cleared = 0
    for i, note_id in enumerate(note_ids, 1):
        raw_text = raw_text_by_note.get(note_id)
        if raw_text is None:
            print(f"[{i}/{len(note_ids)}] {note_id}: no raw text, skipping")
            continue

        conn_w = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb",
                                    read_only=False, max_wait_seconds=300)
        try:
            existing = [{"orig_start": r[0], "orig_end": r[1]} for r in conn_w.execute(
                "SELECT orig_start, orig_end FROM extracted_entities "
                "WHERE note_id = ? AND is_test = TRUE "
                "AND (gliner_model_version IS NULL OR gliner_model_version != 'narrative_state_word_coldstart')",
                [note_id]).fetchall()]

            fresh_entities = build_narrative_state_word_entities(raw_text, note_id, existing)
            if not fresh_entities:
                continue

            process_and_normalize_entities(fresh_entities, conn_w, is_test=True)
            total_reentities += len(fresh_entities)

            entity_ids = [e["entity_id"] for e in fresh_entities]
            placeholders = ",".join("?" * len(entity_ids))
            stale = conn_w.execute(
                f"SELECT count(*) FROM mollm_tier_gate_decisions WHERE entity_id IN ({placeholders})",
                entity_ids).fetchone()[0]
            if stale:
                conn_w.execute(
                    f"DELETE FROM mollm_tier_gate_decisions WHERE entity_id IN ({placeholders})",
                    entity_ids)
                total_cleared += stale
            print(f"[{i}/{len(note_ids)}] {note_id}: re-fixed {len(fresh_entities)} entities, "
                  f"cleared {stale} stale decision(s)")
        finally:
            conn_w.close()

    print(f"\nTOTAL: re-fixed {total_reentities} narrative entities, "
          f"cleared {total_cleared} stale tier-gate decisions")


if __name__ == "__main__":
    main()
