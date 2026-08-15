"""
Run on EC2, from the project root:
    python3 inspect_citations.py

Pulls the two records this needs (Abdominal pain / Paracentesis, plus the
gum-bleeding dash case) from mollm_decisions and pretty-prints each model's
raw cited_evidence[].quote next to what verify_citations() actually checked
it against, so the paraphrase-vs-literal-quote question can be read directly
instead of guessed at.
"""
import os
import json
import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

conn = duckdb.connect(DB_PATH, read_only=True)

rows = conn.sql("""
    SELECT d.entity_id, e.original_text, d.models, d.citation_checks, d.mollm_call_id
    FROM mollm_decisions d
    JOIN extracted_entities e ON e.entity_id = d.entity_id
    WHERE d.note_id = '10000032-DS-21' AND d.is_test = TRUE
    ORDER BY d.mollm_call_id
""").fetchall()

for entity_id, entity_text, models_json, checks_json, call_id in rows:
    models = json.loads(models_json) if isinstance(models_json, str) else models_json
    checks = json.loads(checks_json) if isinstance(checks_json, str) else checks_json
    print("=" * 90)
    print(f"entity_text={entity_text!r}  entity_id={entity_id}  call_id={call_id}")
    print("-" * 90)
    for m in models or []:
        name = m.get("model") or "?"
        for c in (m.get("cited_evidence") or []):
            print(f"  [{name}] rule_id={c.get('rule_id')!r}")
            print(f"      quote: {c.get('quote')!r}")
    print("  --- verify_citations() results ---")
    for chk in checks or []:
        print(f"  rule_id={chk.get('rule_id')!r} verified={chk.get('verified')} "
              f"reason={chk.get('reason')} containment={chk.get('containment')}")
    print()