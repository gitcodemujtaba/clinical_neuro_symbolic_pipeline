"""
scripts/run_kg3_ingestion.py — Stage 4 batch driver: writes every
HUMAN_VERIFIED hitl_review_queue case to Memgraph (KG3).

WHY A SEPARATE SCRIPT FROM src/kg3_ingestion.py. Same split as
src/mollm_ensemble.py (the per-decision logic) vs scripts/run_stage3_batch.py
(the corpus loop) -- src/kg3_ingestion.py's ingest_reviewed_case() takes an
already-assembled case + entity_fields dict so it has no DuckDB dependency
of its own; this script owns the DuckDB read side (hitl_review_queue,
extracted_entities, athena_concept) and the loop.

RESUMABLE, LOUD ON FAILURE. Mirrors run_stage3_batch.py's own posture: a
single UningestibleCase (see src/kg3_ingestion.py) is reported and skipped,
not fatal to the batch -- but it is printed, not swallowed, since a case
that should have been reviewable-as-CORRECTED silently vanishing from KG3
coverage is exactly the kind of gap this project's evaluation work has
repeatedly found expensive to diagnose after the fact.
"""
import argparse
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
sys.path.insert(0, PROJECT_DIR)

from src.hitl_queue import load_ingestible_cases, mark_ingested  # noqa: E402
from src.kg3_ingestion import (  # noqa: E402
    UningestibleCase, get_memgraph_driver, ingest_reviewed_case, resolve_concept_id,
)


def _entity_fields(conn, entity_id: str, omop_concept_id: int) -> dict:
    """Joins extracted_entities (per-mention offsets/confidence) and
    athena_concept (vocabulary_id/domain_id/concept_name for the RESOLVED
    concept, which for a CORRECTED case is not necessarily the concept Stage
    2b originally proposed) into the flat dict ingest_reviewed_case() needs.
    """
    ent = conn.execute(
        "SELECT orig_start, orig_end, confidence FROM extracted_entities WHERE entity_id = ?",
        [entity_id],
    ).fetchone()
    concept = conn.execute(
        "SELECT vocabulary_id, domain_id, concept_name FROM athena_concept WHERE concept_id = ?",
        [omop_concept_id],
    ).fetchone()
    return {
        "orig_start": ent[0] if ent else None,
        "orig_end": ent[1] if ent else None,
        "confidence": ent[2] if ent else None,
        "vocabulary_id": concept[0] if concept else None,
        "domain_id": concept[1] if concept else None,
        "concept_name": concept[2] if concept else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--dry-run", action="store_true",
                     help="resolve and print what would be ingested without "
                          "writing to Memgraph or marking anything ingested.")
    args = ap.parse_args()

    print("=" * 78)
    print("STAGE 4 KG3 INGESTION")
    print("=" * 78)

    conn = duckdb.connect(args.db, read_only=False)
    cases = load_ingestible_cases(conn)
    print(f"ingestible cases (HUMAN_VERIFIED, not yet ingested): {len(cases)}")
    if not cases:
        return 0

    driver = None if args.dry_run else get_memgraph_driver()

    ingested = 0
    uningestible = 0
    errors = 0
    for case in cases:
        try:
            omop_concept_id = resolve_concept_id(case)
            fields = _entity_fields(conn, case["entity_id"], omop_concept_id)
            if args.dry_run:
                print(f"  [DRY RUN] {case['hitl_case_id']} -> concept_id={omop_concept_id} "
                     f"({fields.get('concept_name')})")
                continue
            ingest_reviewed_case(driver, case, fields)
            mark_ingested(conn, case["hitl_case_id"])
            ingested += 1
        except UningestibleCase as exc:
            uningestible += 1
            print(f"  SKIPPED (uningestible): {exc}")
        except Exception as exc:
            errors += 1
            print(f"  ERROR on {case['hitl_case_id']}: {type(exc).__name__}: {exc}")

    if driver:
        driver.close()

    print()
    print("=" * 78)
    print("INGESTION COMPLETE")
    print("=" * 78)
    print(f"ingested: {ingested}   uningestible: {uningestible}   errors: {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
