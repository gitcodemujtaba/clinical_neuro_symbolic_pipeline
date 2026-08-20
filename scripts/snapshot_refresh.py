"""scripts/snapshot_refresh.py — keeps db/kg2_lexical_store_snapshot.duckdb
in sync with the live DB, so Streamlit can browse data continuously while
a background pipeline batch job holds the live DB's write lock.

WHY THIS EXISTS. DuckDB is single-writer-exclusive: while a batch job
(scripts/test_pipeline_e2e.py, scripts/run_stage3_tier_gate.py) holds the
live DB open for writing, EVERY other connection attempt -- even a
read-only one -- fails immediately (confirmed empirically this session).
A single note's Stage 3 run can hold that lock continuously for 10-45+
minutes, which makes ui/pages/*.py effectively unusable for the whole
batch's duration if they point at the live file directly. Pointing
Streamlit at a SEPARATE snapshot file instead means it never contends with
the writer at all -- the cost is staleness (data as of the last refresh),
not availability.

TWO MODES, deliberately different cost:
  --full   One-time bootstrap: copies table SCHEMA + rows for every table,
           column by column, EXCLUDING any `embedding` column. The live
           DB's ~31GB is almost entirely SapBERT embedding vectors on the
           SNOMED/RxNorm/OMOP vocabulary tables (millions of rows x
           FLOAT[768], confirmed by column inspection) -- data the UI
           never reads (embeddings are only used during LIVE Tier 3
           retrieval by the pipeline itself, never for displaying or
           grading already-computed results). Dropping just that one
           column turns a many-minute, whole-database copy into a copy of
           the actual few-hundred-MB of text/int columns everything else
           needs. An earlier version of this script used DuckDB's native
           `COPY FROM DATABASE` (copies everything, no column control) --
           replaced after the user pointed out copying 31GB to serve a
           UI that only shows recent notes' data made no sense.
  (default) Incremental refresh: re-copies ONLY the small, fast-changing
           pipeline-output tables (DYNAMIC_TABLES below) -- thousands of
           rows, not tens of millions, so this is fast (well under a
           second) and cheap enough to run after every note in a batch
           queue. The large static tables are never touched again after
           the one-time --full bootstrap, since they don't change (and
           don't carry an embedding column that could go stale anyway --
           the UI-visible columns of static reference data are immutable).

LOCK COORDINATION. Both modes open the live DB read-only, which will fail
with a lock conflict if a writer is mid-note -- this script does NOT retry
internally (keeps it simple, single-responsibility); callers (e.g.
scripts/process notes in a loop) should wrap invocations in their own
retry-on-lock-conflict logic, exactly like scripts/test_pipeline_e2e.py's
own callers already do for writes.

Run:
  python3 scripts/snapshot_refresh.py --full      # one-time bootstrap
  python3 scripts/snapshot_refresh.py             # incremental refresh
"""
import argparse
import os
import time

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
SNAPSHOT_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store_snapshot.duckdb")

# Small, fast-changing pipeline-output tables -- everything ui/pages/*.py
# and evaluation/*.py actually write-or-recently-write to. NOT the large
# static vocabulary tables (athena_concept, athena_concept_synonym,
# athena_concept_relationship, athena_concept_ancestor, ...), which never
# change after the one-time --full bootstrap and would make an incremental
# refresh just as slow as a full one if re-copied every time.
DYNAMIC_TABLES = [
    "extracted_entities",
    "normalized_entities",
    "mollm_tier_gate_decisions",
    "mollm_decisions",
    "hitl_review_queue",
    "note_expansions",
]

# Column names dropped from EVERY table during the --full copy, regardless
# of which table they're on -- consistently named "embedding" everywhere
# this codebase creates one (src/normalization/tier_retrieval.py,
# scripts/backfill_guideline_grounding.py), so matching by name is simpler
# and more robust than hardcoding a table-specific column list, and
# automatically covers any embedding-bearing table added later.
DROPPED_COLUMNS = {"embedding"}


def _catalog_name(conn):
    return conn.execute("SELECT current_catalog()").fetchone()[0]


def _table_names(conn, catalog):
    # schema_name = 'main' only -- the BM25 full-text-search extension
    # (src/normalization/bm25_index.py) creates its own internal index
    # tables (fts_main_athena_concept.dict/docs/fields/stats/terms, ...) in
    # a SEPARATE schema. Confirmed live: including them pulls in short,
    # colliding table names like "dict" that fail to resolve unqualified
    # (DuckDB suggests the schema-qualified name in the error). The UI
    # never queries these directly (only via match_bm25(), and
    # HYBRID_RETRIEVAL_ENABLED is off anyway) so they're simply excluded,
    # not worth the complexity of schema-qualifying every copy.
    return [r[0] for r in conn.execute(
        "SELECT table_name FROM duckdb_tables() WHERE database_name = ? AND schema_name = 'main'",
        [catalog]
    ).fetchall()]


def _copyable_column_list(conn, catalog, table):
    cols = [r[0] for r in conn.execute(
        "SELECT column_name FROM duckdb_columns() WHERE database_name = ? AND schema_name = 'main' "
        "AND table_name = ? ORDER BY column_index", [catalog, table]
    ).fetchall()]
    return [c for c in cols if c.lower() not in DROPPED_COLUMNS]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true",
                    help="One-time full-database bootstrap copy (embedding columns dropped).")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--snapshot", default=SNAPSHOT_PATH)
    args = ap.parse_args()

    t0 = time.time()
    conn = duckdb.connect(args.db, read_only=True)
    try:
        # Explicit READ_WRITE: a read-only PARENT connection otherwise
        # defaults an ATTACHed database to read-only too, which fails
        # outright the first time (can't create a new file in read-only
        # mode) -- confirmed live, this is not hypothetical.
        conn.execute(f"ATTACH '{args.snapshot}' AS snap (READ_WRITE)")
        catalog = _catalog_name(conn)

        if args.full:
            print(f"full bootstrap: copying every table (minus embedding columns) to "
                 f"{args.snapshot} ...")
            copied = []
            for table in _table_names(conn, catalog):
                cols = _copyable_column_list(conn, catalog, table)
                if not cols:
                    continue
                col_list = ", ".join(f'"{c}"' for c in cols)
                conn.execute(f'CREATE OR REPLACE TABLE snap."{table}" AS '
                            f'SELECT {col_list} FROM "{table}"')
                copied.append(table)
            print(f"full bootstrap done: {len(copied)} tables copied in {time.time()-t0:.1f}s")
        else:
            existing = set(_table_names(conn, catalog))
            refreshed = []
            for table in DYNAMIC_TABLES:
                if table not in existing:
                    continue  # e.g. mollm_decisions may not exist yet on a fresh DB
                cols = _copyable_column_list(conn, catalog, table)
                col_list = ", ".join(f'"{c}"' for c in cols)
                conn.execute(f'CREATE OR REPLACE TABLE snap."{table}" AS '
                            f'SELECT {col_list} FROM "{table}"')
                refreshed.append(table)
            print(f"incremental refresh done: {refreshed} in {time.time()-t0:.2f}s")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
