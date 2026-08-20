"""scripts/backfill_lowered_threshold.py — one-time backfill (2026-08-18):
EXTRACTION_THRESHOLD dropped from 0.5 to 0.35 (see src/entity_extraction.py's
own comment for the corpus-wide threshold-sweep evidence). Already-processed
notes have their 0.35-0.50-confidence entities correctly STORED (that's the
whole point of SUBTHRESHOLD_FLOOR retention) but flagged below_threshold=TRUE
and never normalized, since that flag was computed against the OLD 0.5
constant at extraction time and isn't recomputed retroactively just by
changing the constant.

For each target note: flips below_threshold to FALSE on the newly-admitted
rows (confidence >= 0.35, the new floor == new threshold) and runs
process_and_normalize_entities() on exactly those rows -- no re-extraction,
no re-running GLiNER, just normalizing entities that were already sitting in
the DB waiting for exactly this.

Run:
  python3 scripts/backfill_lowered_threshold.py --note-ids 13538696-DS-11
  python3 scripts/backfill_lowered_threshold.py --all-processed
"""
import argparse
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
sys.path.insert(0, PROJECT_DIR)

from src.entity_extraction import EXTRACTION_THRESHOLD  # noqa: E402
from src.normalization import process_and_normalize_entities  # noqa: E402
from src.db_utils import connect_with_retry  # noqa: E402


def backfill_note(conn, note_id: str) -> int:
    cur = conn.execute("""
        SELECT * FROM extracted_entities
        WHERE is_test = TRUE AND note_id = ? AND below_threshold = TRUE
          AND confidence >= ?
    """, [note_id, EXTRACTION_THRESHOLD])
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    if not rows:
        return 0
    entities = [dict(zip(cols, r)) for r in rows]

    process_and_normalize_entities(entities, conn, is_test=True)

    entity_ids = [e["entity_id"] for e in entities]
    placeholders = ",".join("?" * len(entity_ids))
    conn.execute(f"UPDATE extracted_entities SET below_threshold = FALSE "
                f"WHERE entity_id IN ({placeholders})", entity_ids)
    return len(entities)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None, help="Comma-separated note_ids.")
    ap.add_argument("--all-processed", action="store_true",
                    help="Every is_test=TRUE note_id with newly-admittable entities.")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    conn = connect_with_retry(args.db, read_only=False)
    try:
        if args.note_ids:
            note_ids = [n.strip() for n in args.note_ids.split(",") if n.strip()]
        elif args.all_processed:
            note_ids = [r[0] for r in conn.execute("""
                SELECT DISTINCT note_id FROM extracted_entities
                WHERE is_test = TRUE AND below_threshold = TRUE AND confidence >= ?
            """, [EXTRACTION_THRESHOLD]).fetchall()]
        else:
            raise SystemExit("Pass --note-ids or --all-processed.")

        print(f"EXTRACTION_THRESHOLD: {EXTRACTION_THRESHOLD}")
        print(f"notes: {len(note_ids)}")
        total = 0
        for i, note_id in enumerate(note_ids, 1):
            n = backfill_note(conn, note_id)
            total += n
            print(f"[{i}/{len(note_ids)}] {note_id}: {n} entities newly normalized")
        print(f"\nTotal newly-normalized entities: {total}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
