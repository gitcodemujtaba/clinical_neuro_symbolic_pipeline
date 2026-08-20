"""scripts/refix_uk_extension_lab_suffixed.py -- 2026-08-20: second-half
follow-up to scripts/refix_uk_extension_lab_candidates.py. That script
re-normalized bare-text matches to _LAB_TEST_ALIASES's 26 curated terms
(e.g. original_text == "MCHC"). It did NOT catch value-suffixed variants
like "MCHC-31"/"RDW-19"/"MCV-97" -- these only reduce to a curated alias
term INSIDE normalize_entity() via strip_lab_value_suffix()
(src.normalization.compound_span), never in the bare original_text column,
so the earlier `lower(original_text) IN (...)` filter silently excluded
them. Confirmed live via evaluation/exhaustive_candidate_eval_impact.py:
several value-suffixed entities were still resolving to the pre-fix
UK-extension/Qualifier-Value duplicate concepts because they'd simply
never been re-normalized since the fix landed -- not a gap in
strip_lab_value_suffix() itself (it already handles this shape correctly
and was already wired into the pipeline before today).

NOT a new regex. Reuses strip_lab_value_suffix() directly to find exactly
which entities are affected (any entity_label='Lab Test' entity whose
stripped candidate(s) land in _LAB_TEST_ALIASES), avoiding a second,
parallel implementation of suffix-stripping that could drift from the
original.

Run: python3 scripts/refix_uk_extension_lab_suffixed.py
"""
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.normalization.compound_span import strip_lab_value_suffix  # noqa: E402
from src.normalization.orchestrator import process_and_normalize_entities  # noqa: E402
from src.normalization.tier_retrieval import _LAB_TEST_ALIASES  # noqa: E402


def main():
    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb",
                              read_only=True, max_wait_seconds=300)
    all_rows = conn.execute("""
        SELECT entity_id, original_text FROM extracted_entities
        WHERE entity_label = 'Lab Test' AND is_test = TRUE
        AND (superseded_by_split IS NULL OR superseded_by_split = FALSE)
        AND (superseded_by_growth IS NULL OR superseded_by_growth = FALSE)
    """).fetchall()

    target_ids = set()
    for entity_id, text in all_rows:
        for cand in strip_lab_value_suffix(text or ""):
            if cand.strip().lower() in _LAB_TEST_ALIASES:
                target_ids.add(entity_id)
                break
    print(f"{len(target_ids)} value-suffixed entities reduce to a curated alias term")

    if not target_ids:
        conn.close()
        return

    placeholders = ",".join("?" * len(target_ids))
    rows = conn.execute(f"""
        SELECT entity_id, note_id, entity_label, original_text, expanded_text,
               orig_start, orig_end, exp_start, exp_end, assertion_status,
               experiencer, temporality, assertion_cue, assertion_cue_start,
               assertion_cue_end, assertion_cue_category, assertion_engine,
               section_name, sentence_id, local_context,
               expansion_ambiguous, candidate_expansions, selection_basis,
               gliner_model_version
        FROM extracted_entities WHERE entity_id IN ({placeholders})
    """, list(target_ids)).fetchall()
    cols = [c[0] for c in conn.description]
    entities = [dict(zip(cols, r)) for r in rows]
    conn.close()

    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb",
                              read_only=False, max_wait_seconds=300)
    try:
        done = 0
        for i in range(0, len(entities), 50):
            batch = entities[i:i + 50]
            process_and_normalize_entities(batch, conn, is_test=True)
            done += len(batch)
            print(f"  re-normalized {done}/{len(entities)}")

        entity_ids = list(target_ids)
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
