"""
scripts/regression_check_kg3.py — snapshot gold-recall + KG3 coverage,
flag regressions against the previous snapshot.

WHY THIS EXISTS NOW, BEFORE STAGE 5'S FEEDBACK MECHANISM DOES.
docs/Implementation_Checklist.md's Stage 5 checklist item: "Regression check
against held-out gold-evaluation set on each KG 3 update." No feedback
mechanism writes anything back to Stage 2b yet (see src/kg3_query.py's
module docstring for why), but the HABIT and the TOOL should exist before
that mechanism does, not be bolted on afterward once there's something to
regress. This script's gold-recall number cannot move on its own today
(nothing feeds KG3 signal back into Stage 2b yet) -- it exists so the very
first change that DOES close that loop has something to diff against
immediately, rather than that being the first time this check gets written.

REUSES scripts/score_gold_recall.py's SCORING, DOES NOT REIMPLEMENT IT.
load_gold/load_predictions/attach_snomed_codes/score() are imported
directly -- this script's only original contribution is the snapshot/diff
mechanics and the KG3-side counts, matching this codebase's own repeated
"two copies of one idea drifting apart" warning (see e.g. src/provenance.py,
docs/Databases.md's own note re: Cypher query duplication).
"""
import argparse
import datetime
import glob
import json
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
SNAPSHOT_DIR = os.path.join(PROJECT_DIR, "reports", "kg3_regression_snapshots")
sys.path.insert(0, PROJECT_DIR)

from scripts.score_gold_recall import (  # noqa: E402
    GOLD_CANDIDATES, DEFAULT_NOTE_IDS, _first_existing,
    attach_snomed_codes, load_gold, load_predictions, score,
)
from src.kg3_ingestion import get_memgraph_driver  # noqa: E402
from src.kg3_query import count_by_label  # noqa: E402


def _latest_snapshot():
    files = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "*.json")))
    if not files:
        return None
    with open(files[-1]) as fh:
        return json.load(fh)


def take_snapshot(note_ids, db_path=DB_PATH) -> dict:
    gold_path = _first_existing(GOLD_CANDIDATES, "gold annotations CSV")
    gold_rows = load_gold(gold_path, note_ids)
    conn = duckdb.connect(db_path, read_only=True)
    try:
        predictions = load_predictions(conn, note_ids)
        attach_snomed_codes(conn, predictions)
        report = score(gold_rows, predictions)
    finally:
        conn.close()

    kg3_counts = {}
    try:
        driver = get_memgraph_driver()
        kg3_counts = count_by_label(driver)
        driver.close()
    except Exception as exc:
        kg3_counts = {"__error__": f"{type(exc).__name__}: {exc}"}

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note_ids": note_ids,
        "gold_recall_combined": report["combined"],
        "kg3_observation_counts_by_label": kg3_counts,
    }


def compare(previous: dict, current: dict) -> list:
    """Returns a list of human-readable regression warnings, empty if none.
    Only flags DROPS -- an improvement is never a regression, even if it
    also shifts the number in a way a naive != comparison would flag.
    """
    warnings = []
    prev_combined = (previous or {}).get("gold_recall_combined", {})
    curr_combined = current.get("gold_recall_combined", {})
    for key in ("linked_recall", "span_recall"):
        prev_val = prev_combined.get(key)
        curr_val = curr_combined.get(key)
        if prev_val is not None and curr_val is not None and curr_val < prev_val:
            warnings.append(
                f"REGRESSION: {key} dropped {prev_val:.4f} -> {curr_val:.4f} "
                f"(-{prev_val - curr_val:.4f})"
            )
    return warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=",".join(DEFAULT_NOTE_IDS))
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()
    note_ids = [n.strip() for n in args.note_ids.split(",") if n.strip()]

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    previous = _latest_snapshot()
    current = take_snapshot(note_ids, db_path=args.db)

    print("=" * 78)
    print("KG3 REGRESSION CHECK")
    print("=" * 78)
    print(f"notes: {note_ids}")
    print(f"gold_recall_combined: {current['gold_recall_combined']}")
    print(f"kg3_observation_counts_by_label: {current['kg3_observation_counts_by_label']}")

    warnings = compare(previous, current)
    if warnings:
        print("\n".join(warnings))
    elif previous:
        print("no regression vs. previous snapshot")
    else:
        print("no previous snapshot to compare against -- this is the baseline")

    out_path = os.path.join(
        SNAPSHOT_DIR,
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json",
    )
    with open(out_path, "w") as fh:
        json.dump(current, fh, indent=2, default=str)
    print(f"\nsnapshot written: {out_path}")

    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
