"""scripts/refix_uk_extension_lab_candidates.py -- 2026-08-20: re-normalizes
every Lab-Test-labeled test-corpus entity so its Stage 2b candidate pool
reflects the SCTID-namespace UK-extension exclusion and the Qualifier-Value
class extension to _prefer_lab_procedure_over_observable() (both in
src/normalization/tier_retrieval.py, commit 6f5135d).

WHY THIS SCRIPT EXISTS AT ALL, NOT JUST "RE-RUN STAGE 3". Discovered live
(2026-08-20): scripts/run_stage3_tier_gate.py does not recompute
candidates -- it reads the ALREADY-STORED normalized_entities.candidates
JSON, written once by src.normalization.orchestrator (Stage 2b) at
extraction time. Restarting the Stage 3 backfill after the retrieval fix
landed therefore did NOT exercise the fix at all: MCHC/RDW decisions
written minutes into the restarted run were still picking the pre-fix
UK-extension concepts (37393850/37397924), confirmed directly against
mollm_tier_gate_decisions. The fix only takes effect the next time Stage
2b itself runs -- this script is that re-run, scoped narrowly.

SCOPE: narrowed further, 2026-08-20 same day, after a first attempt at
entity_label='Lab Test' (all 6,172 entities) proved far slower than
expected (~5GB read, still under 200 entities after 40+ minutes wall
time -- root cause not fully diagnosed under time pressure, but the vast
majority of those 6,172 entities are plain lab-test NAMES like
"potassium"/"glucose" that were never near a UK-extension or Qualifier-
Value duplicate in the first place; only the ~26 curated abbreviation
terms in _LAB_TEST_ALIASES are actually force-included candidates that
could collide with either near-duplicate pattern). Rescoped to
lower(original_text) IN (the alias dict's own keys) -- 2,484 entities,
directly the population the fix targets, not a superset padded with
irrelevant lab-test mentions.

Follows the exact re-normalize-then-clear-stale-decisions pattern already
proven this session in scripts/refix_coldstart_hierarchy_collapse.py.

Run: python3 scripts/refix_uk_extension_lab_candidates.py
"""
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.normalization.orchestrator import process_and_normalize_entities  # noqa: E402
from src.normalization.tier_retrieval import _LAB_TEST_ALIASES  # noqa: E402


def main():
    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb",
                              read_only=True, max_wait_seconds=300)
    terms = list(_LAB_TEST_ALIASES.keys())
    placeholders = ",".join("?" * len(terms))
    rows = conn.execute(f"""
        SELECT entity_id, note_id, entity_label, original_text, expanded_text,
               orig_start, orig_end, exp_start, exp_end, assertion_status,
               experiencer, temporality, assertion_cue, assertion_cue_start,
               assertion_cue_end, assertion_cue_category, assertion_engine,
               section_name, sentence_id, local_context,
               expansion_ambiguous, candidate_expansions, selection_basis,
               gliner_model_version
        FROM extracted_entities
        WHERE entity_label = 'Lab Test' AND is_test = TRUE
        AND lower(original_text) IN ({placeholders})
        AND (superseded_by_split IS NULL OR superseded_by_split = FALSE)
        AND (superseded_by_growth IS NULL OR superseded_by_growth = FALSE)
    """, terms).fetchall()
    cols = [c[0] for c in conn.description]
    entities = [dict(zip(cols, r)) for r in rows]
    conn.close()
    print(f"{len(entities)} Lab Test entities to re-normalize")

    if not entities:
        return

    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb",
                              read_only=False, max_wait_seconds=300)
    try:
        done = 0
        for i in range(0, len(entities), 50):
            batch = entities[i:i + 50]
            process_and_normalize_entities(batch, conn, is_test=True)
            done += len(batch)
            print(f"  re-normalized {done}/{len(entities)}")

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
