"""scripts/mark_notes_stale.py — one-time migration (2026-08-18): adds
is_stale to normalized_entities and marks every note EXCEPT the ones
processed by current code as stale, so the Streamlit UI (ui/pages/1, 3, 4)
can filter to "processed by current code" via a durable DB-level flag
instead of a hand-maintained cutoff timestamp in application code (the
earlier ui/components/data_freshness.py approach, replaced by this).

WHY THIS EXISTS. Confirmed directly against normalized_entities.code_version
this session: every previously-processed note (the 5-note validation batch
and everything before it) shows code_version 'c9d2d60-dirty' or older --
predating the LMCA/CAPS fragile-shorthand-trap fix (commit 1826d7e) and
several other fixes earlier the same session. Only one note,
16795604-DS-17, ran with the real fixed logic (code_version
'f190ac3-dirty' -- still "-dirty" because the fix was uncommitted at the
time it ran, committed shortly after as 1826d7e; the git-dirty suffix
alone can't distinguish "old code" from "new code, not yet committed", so
this migration marks by NOTE_ID explicitly rather than trying to parse
code_version strings).

DEFAULT FOR FUTURE ROWS. is_stale defaults to FALSE at the column level, so
every note the pipeline processes FROM NOW ON is automatically "fresh"
without any pipeline code changes -- src/normalization/orchestrator.py's
INSERT statements don't (and don't need to) reference this column at all.

Run once: python3 scripts/mark_notes_stale.py
Idempotent: safe to re-run (ADD COLUMN IF NOT EXISTS, UPDATE is a no-op if
already applied to the same note set).
"""
import argparse
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

# The one note confirmed to have run with current (fixed) pipeline code as
# of this migration -- see the module docstring. Extend this list by hand
# if a future session identifies more genuinely-fresh notes processed
# before their own is_stale=FALSE default would have applied.
FRESH_NOTE_IDS = ["16795604-DS-17"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    conn = duckdb.connect(args.db)
    try:
        conn.execute("ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS "
                     "is_stale BOOLEAN DEFAULT FALSE")
        placeholders = ",".join("?" * len(FRESH_NOTE_IDS))
        conn.execute(f"UPDATE normalized_entities SET is_stale = TRUE "
                    f"WHERE note_id NOT IN ({placeholders})", FRESH_NOTE_IDS)
        conn.execute(f"UPDATE normalized_entities SET is_stale = FALSE "
                    f"WHERE note_id IN ({placeholders})", FRESH_NOTE_IDS)

        stale_notes = conn.execute(
            "SELECT count(DISTINCT note_id) FROM normalized_entities WHERE is_stale = TRUE"
        ).fetchone()[0]
        fresh_notes = conn.execute(
            "SELECT count(DISTINCT note_id) FROM normalized_entities WHERE is_stale = FALSE"
        ).fetchone()[0]
        print(f"stale notes: {stale_notes}")
        print(f"fresh notes: {fresh_notes}")
        fresh_ids = conn.execute(
            "SELECT DISTINCT note_id FROM normalized_entities WHERE is_stale = FALSE"
        ).fetchall()
        print(f"fresh note_ids: {[r[0] for r in fresh_ids]}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
