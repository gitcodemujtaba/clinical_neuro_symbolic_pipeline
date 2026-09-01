"""scripts/fix_normalized_entities_dedup_key.py -- 2026-09-01: one-time
migration + backfill for a real bug found this session while diagnosing the
recall gap: normalized_entities' row identity was keyed on
(note_id, original_text, expanded_text, gliner_label), NOT entity_id.

THE BUG. Two different real entity_ids (two separate mentions of, say,
"HTN" in the same note -- extremely common) can share an identical
(note_id, original_text, expanded_text, gliner_label) tuple. The old
`ON CONFLICT (note_id, original_text, expanded_text, gliner_label) DO
UPDATE SET entity_id = EXCLUDED.entity_id, ...` collapsed every such
duplicate mention into ONE physical row, whose entity_id column got
overwritten to whichever entity_id was processed last. Every earlier
duplicate mention's entity_id was left pointing at nothing -- Stage 3,
HITL, and KG3 all read normalized_entities by entity_id, so those
mentions were invisible to everything downstream despite GLiNER having
correctly extracted them. Verified live: 100% of the 8,471 corpus-wide
"accepted but never normalized" extracted_entities rows are explained by
exactly this collision (see the recall-breakdown diagnostic that found
this, same session).

WHY NOT (entity_id) ALONE. A second real, legitimate case exists: a
multi-drug regimen abbreviation (e.g. "R-CHOP") expands to SEVERAL
different drug names for the SAME physical span/entity_id (Rituximab,
Cyclophosphamide, Hydroxydaunomycin, Oncovin, Prednisone -- a real,
verified example from note 12465457-DS-18), each independently
normalized to its own concept. These five rows correctly share one
entity_id but have five different expanded_text values -- collapsing to
a single row per entity_id would be a NEW regression, deleting four of
five real drug resolutions. Checked directly against live data: zero
collisions exist today on (entity_id, expanded_text) across all 22,177
existing rows -- this is the correct, verified key.

DuckDB 1.4.5 has no ALTER TABLE ADD/DROP CONSTRAINT support, so this
does a rename-recreate-copy migration rather than an in-place ALTER --
standard DuckDB pattern for a constraint change. The pre-migration table
is kept (renamed, not dropped) as a safety net, matching this project's
own "archive, don't hard-delete" convention elsewhere (docs/2026-08-14_
Dead_Code_Audit.md).

BACKFILL. After the schema fix, this script also inserts the missing
rows for every currently-orphaned entity_id, by COPYING its sibling
row's already-computed normalization result (same note_id/original_text/
expanded_text/gliner_label -> deterministically identical outcome) --
not by re-running normalize_entity(), which would mean thousands of
redundant SapBERT/tier-search calls for a result already known.

Run: python3 scripts/fix_normalized_entities_dedup_key.py
Idempotent: re-running after a successful prior run is a no-op (the
migration step no-ops if normalized_entities already has the new
UNIQUE(entity_id, expanded_text) constraint -- detected by checking
whether the backup table already exists; the backfill step no-ops
because ON CONFLICT (entity_id, expanded_text) DO NOTHING on the insert
means already-backfilled rows are simply skipped).
"""
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

BACKUP_TABLE = "normalized_entities_backup_predupfix_20260901"

# Exact column list, in order, captured live from the production schema
# before this migration -- copied verbatim, not hand-retyped, so a column
# this script's author forgot about can't silently vanish.
COLUMNS = [
    ("note_id", "VARCHAR"), ("original_text", "VARCHAR"), ("expanded_text", "VARCHAR"),
    ("gliner_label", "VARCHAR"), ("gliner_confidence", "FLOAT"),
    ("omop_concept_id", "BIGINT"), ("omop_concept_name", "VARCHAR"),
    ("omop_domain", "VARCHAR"), ("omop_vocab", "VARCHAR"), ("match_tier", "VARCHAR"),
    ("similarity_score", "FLOAT"), ("is_test", "BOOLEAN"), ("entity_id", "VARCHAR"),
    ("candidates", "JSON"), ("is_ambiguous", "BOOLEAN"), ("ambiguity_reason", "VARCHAR"),
    ("confidence_tier_in", "VARCHAR"), ("tier_reasons", "JSON"), ("domain_conflict", "JSON"),
    ("tier_trace", "JSON"), ("domain_id_queried", "JSON"), ("vocab_queried", "JSON"),
    ("sapbert_pooling_method", "VARCHAR"), ("matched", "BOOLEAN"), ("normalized_from", "VARCHAR"),
    ("athena_vocabulary_release", "VARCHAR"), ("tier12_rank_basis", "VARCHAR"),
    ("created_at", "TIMESTAMP"), ("run_id", "VARCHAR"), ("code_version", "VARCHAR"),
    ("candidates_hash", "VARCHAR"), ("is_stale", "BOOLEAN"),
]


def main():
    from src.db_utils import connect_with_retry
    conn = connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=300)

    existing_tables = {r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()}

    if BACKUP_TABLE not in existing_tables:
        print("=== Step 1: migrating normalized_entities to UNIQUE(entity_id, expanded_text) ===")
        before_count = conn.execute("SELECT count(*) FROM normalized_entities").fetchone()[0]
        conn.execute(f"ALTER TABLE normalized_entities RENAME TO {BACKUP_TABLE}")

        col_defs = ",\n            ".join(f"{name} {dtype}" for name, dtype in COLUMNS)
        conn.execute(f"""
            CREATE TABLE normalized_entities (
                {col_defs},
                UNIQUE(entity_id, expanded_text)
            )
        """)
        col_list = ", ".join(name for name, _ in COLUMNS)
        conn.execute(f"INSERT INTO normalized_entities ({col_list}) "
                     f"SELECT {col_list} FROM {BACKUP_TABLE}")
        after_count = conn.execute("SELECT count(*) FROM normalized_entities").fetchone()[0]
        assert before_count == after_count, (
            f"row count mismatch after migration: {before_count} -> {after_count}")
        print(f"  migrated {after_count} existing rows unchanged, old table kept as "
              f"{BACKUP_TABLE!r} for safety")
    else:
        print("=== Step 1: already migrated (backup table exists) -- skipping ===")

    print("\n=== Step 2: backfilling orphaned entity_ids from their sibling's result ===")
    # Source of each target column, kept explicit (not positionally generated)
    # to avoid a silent column-order mismatch between INSERT and SELECT.
    FROM_EE = {"note_id", "original_text", "expanded_text", "entity_id"}
    select_exprs = []
    for name, _ in COLUMNS:
        if name == "entity_id":
            select_exprs.append("ee.entity_id")
        elif name == "gliner_label":
            select_exprs.append("ee.entity_label")
        elif name in FROM_EE:
            select_exprs.append(f"ee.{name}")
        elif name == "is_test":
            select_exprs.append("ee.is_test")
        else:
            select_exprs.append(f"n.{name}")
    col_list = ", ".join(name for name, _ in COLUMNS)
    select_list = ", ".join(select_exprs)
    before_backfill = conn.execute("SELECT count(*) FROM normalized_entities").fetchone()[0]
    conn.execute(f"""
        INSERT INTO normalized_entities ({col_list})
        SELECT {select_list}
        FROM extracted_entities ee
        JOIN normalized_entities n
          ON n.note_id = ee.note_id AND n.original_text = ee.original_text
         AND n.expanded_text = ee.expanded_text AND n.gliner_label = ee.entity_label
        WHERE NOT EXISTS (
            SELECT 1 FROM normalized_entities n2 WHERE n2.entity_id = ee.entity_id
        )
        ON CONFLICT (entity_id, expanded_text) DO NOTHING
    """)
    after_backfill = conn.execute("SELECT count(*) FROM normalized_entities").fetchone()[0]
    print(f"  inserted {after_backfill - before_backfill} backfilled rows "
          f"({before_backfill} -> {after_backfill})")

    n_still_orphaned = conn.execute("""
        SELECT count(*) FROM extracted_entities ee
        WHERE (ee.below_threshold IS NULL OR ee.below_threshold = FALSE)
          AND (ee.superseded_by_split IS NULL OR ee.superseded_by_split = FALSE)
          AND (ee.superseded_by_growth IS NULL OR ee.superseded_by_growth = FALSE)
          AND NOT EXISTS (SELECT 1 FROM normalized_entities n WHERE n.entity_id = ee.entity_id)
    """).fetchone()[0]
    print(f"  accepted extracted_entities still with zero normalized_entities row "
          f"after backfill: {n_still_orphaned} (expected: only genuine zero-candidate "
          f"cases, i.e. entities whose sibling ALSO never normalized -- not a bug)")

    conn.close()


if __name__ == "__main__":
    main()
