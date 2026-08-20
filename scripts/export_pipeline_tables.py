"""scripts/export_pipeline_tables.py -- 2026-08-20: exports this project's
OWN pipeline-generated tables to Parquet, excluding the 5 large imported
OMOP/Athena reference-vocabulary tables (athena_concept,
athena_concept_relationship, athena_concept_ancestor,
athena_concept_synonym, concept_embeddings -- 131.9M rows combined, ~31GB,
licensed reference data, not this project's own research output).

Exported tables are small (~77,600 rows combined across all of them) --
suitable for a git commit, for supplementary paper data, or for a
standalone backup, unlike the full 31GB DB file (which stays gitignored,
see db/.gitignore).

Run: python3 scripts/export_pipeline_tables.py
"""
import os
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
OUT_DIR = f"{PROJECT_DIR}/exports"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Excluded deliberately: the imported OMOP/Athena reference vocabulary.
# Everything else in the DB is this project's own generated output (or a
# small supporting reference table, e.g. the abbreviation dictionary).
EXCLUDED_TABLES = {
    "athena_concept", "athena_concept_relationship", "athena_concept_ancestor",
    "athena_concept_synonym", "concept_embeddings",
}


def main():
    from src.db_utils import connect_with_retry

    os.makedirs(OUT_DIR, exist_ok=True)
    conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=60)

    tables = [r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()]
    to_export = sorted(t for t in tables if t not in EXCLUDED_TABLES)

    print(f"exporting {len(to_export)}/{len(tables)} tables "
         f"({len(tables) - len(to_export)} large reference tables excluded)")

    total_rows = 0
    for t in to_export:
        n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        out_path = os.path.join(OUT_DIR, f"{t}.parquet")
        conn.execute(f"COPY (SELECT * FROM \"{t}\") TO '{out_path}' (FORMAT PARQUET)")
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  {t:40s} {n:>10,} rows  ->  {size_kb:>10,.1f} KB")
        total_rows += n

    conn.close()
    total_size = sum(os.path.getsize(os.path.join(OUT_DIR, f))
                     for f in os.listdir(OUT_DIR) if f.endswith(".parquet"))
    print(f"\n{total_rows:,} total rows exported, {total_size/1024/1024:.1f} MB total")


if __name__ == "__main__":
    main()
