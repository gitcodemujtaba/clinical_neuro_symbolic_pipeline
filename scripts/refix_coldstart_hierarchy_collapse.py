"""scripts/refix_coldstart_hierarchy_collapse.py -- 2026-08-20 targeted
re-fix: _collapse_hierarchy_duplicates() was silently discarding a
curated cold-start alias candidate in favor of a plain-similarity
SNOMED-hierarchy sibling (found live on "HCO3" -- see the fix's own
commit message / tier_retrieval.py comment for the full story). Fixed in
src/normalization/tier_retrieval.py. This script re-normalizes every
lab-abbreviation/narrative-state-word cold-start entity already written
by scripts/backfill_recall_coldstarts.py (BEFORE the fix landed) so their
candidate pools reflect the corrected logic, then clears any stale
tier-gate decision the partial (interrupted) Stage 3 backfill run already
wrote for them, so the next Stage 3 pass re-grades with fresh candidates.

Run: python3 scripts/refix_coldstart_hierarchy_collapse.py
"""
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.normalization.orchestrator import process_and_normalize_entities  # noqa: E402


def main():
    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb",
                              read_only=True, max_wait_seconds=300)
    rows = conn.execute("""
        SELECT entity_id, note_id, entity_label, original_text, expanded_text,
               orig_start, orig_end, exp_start, exp_end, assertion_status,
               experiencer, temporality, assertion_cue, assertion_cue_start,
               assertion_cue_end, assertion_cue_category, assertion_engine,
               section_name, sentence_id, local_context,
               expansion_ambiguous, candidate_expansions, selection_basis,
               gliner_model_version
        FROM extracted_entities
        WHERE gliner_model_version IN ('lab_abbrev_coldstart', 'narrative_state_word_coldstart')
        AND is_test = TRUE
    """).fetchall()
    cols = [c[0] for c in conn.description]
    entities = [dict(zip(cols, r)) for r in rows]
    conn.close()
    print(f"{len(entities)} cold-start entities to re-normalize")

    if not entities:
        return

    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb",
                              read_only=False, max_wait_seconds=300)
    try:
        process_and_normalize_entities(entities, conn, is_test=True)

        entity_ids = [e["entity_id"] for e in entities]
        placeholders = ",".join("?" * len(entity_ids))
        stale = conn.execute(
            f"SELECT count(*) FROM mollm_tier_gate_decisions WHERE entity_id IN ({placeholders})",
            entity_ids).fetchone()[0]
        if stale:
            conn.execute(
                f"DELETE FROM mollm_tier_gate_decisions WHERE entity_id IN ({placeholders})",
                entity_ids)
        print(f"re-normalized {len(entities)} entities; cleared {stale} stale tier-gate decision(s)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
