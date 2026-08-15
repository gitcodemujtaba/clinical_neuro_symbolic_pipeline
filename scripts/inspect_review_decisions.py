"""
scripts/inspect_review_decisions.py — prints each mollm_review_decisions row's
routing decision alongside what each ensemble model actually said, so a
disagreement can be read and judged rather than just counted.

2026-08-14: reads the `models` JSON array generically (however many entries
it has) rather than assuming exactly two -- the ensemble is now three models
(qwen2.5:3b, llama3.2:3b, phi4-mini; see src/llm_client.py) and hardcoding
`$[0]`/`$[1]` indices would have silently dropped the third model's opinion.

Printed rather than conn.sql(...).show()'d: reasoning text routinely runs
past the width .show() truncates columns to, and the whole point here is to
read the reasoning, not just confirm it exists.

Run:
  python3 scripts/inspect_review_decisions.py --note-id 10000032-DS-21
  python3 scripts/inspect_review_decisions.py --note-id 10000032-DS-21 --routing AL_HITL_REQUIRED
  python3 scripts/inspect_review_decisions.py   # every note, every routing decision
"""

import argparse
import json
import os

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-id", default=None,
                     help="Restrict to one note_id. Default: every note.")
    ap.add_argument("--routing", default=None,
                     choices=["AL_ACCEPTED", "AL_ACCEPTED_PROVISIONAL", "AL_HITL_REQUIRED"],
                     help="Restrict to one al_routing_decision. Default: all.")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found at {args.db}")
        return 1

    where = []
    params = []
    if args.note_id:
        where.append("note_id = ?")
        params.append(args.note_id)
    if args.routing:
        where.append("al_routing_decision = ?")
        params.append(args.routing)
    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    conn = duckdb.connect(args.db, read_only=True)
    try:
        rows = conn.execute(f"""
            SELECT entity_id, note_id, original_text, al_routing_decision, queue_reason, models
            FROM mollm_review_decisions
            {where_clause}
            ORDER BY note_id, entity_id
        """, params).fetchall()

        if not rows:
            print("No matching rows in mollm_review_decisions.")
            return 0

        for entity_id, note_id, text, routing, reason, models_json in rows:
            models = json.loads(models_json) if isinstance(models_json, str) else (models_json or [])
            print("=" * 78)
            print(f"[{note_id}] {entity_id}  {text!r}")
            print(f"routing: {routing}   reason: {reason}")
            for m in models:
                concept = m.get("proposed_concept_name")
                print(f"\n  {m.get('model')}: {m.get('assessment')}"
                      + (f"  -> proposed: {concept}" if concept else ""))
                print(f"    {m.get('reasoning')}")
            print()

        print(f"{len(rows)} row(s).")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
