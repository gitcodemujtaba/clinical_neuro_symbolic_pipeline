"""scripts/smoke_test_8b_kg_escalation.py -- 2026-08-20 smoke test: 10
entities from each of 5 hard-case categories, run through the KG-aided
8B-model escalation (src.tier4_kg_escalation), with the FULL prompt, KG
context, and verdict written out per entity for review -- not just a
pass/fail summary.

Run: python3 scripts/smoke_test_8b_kg_escalation.py
"""
import json
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.tier4_kg_escalation import escalate_to_8b  # noqa: E402
from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from scripts.score_gold_recall import load_gold, overlaps  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402

DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
N_PER_CATEGORY = 10

CATEGORIES = [
    ("TIER_4_ENSEMBLE_SPLIT", "ensemble_split"),
    ("TIER_5_TRUE_AMBIGUITY", "verdict_none_correct"),
    ("TIER_5_TRUE_AMBIGUITY", "below_similarity_floor"),
    (None, "below_confidence_threshold"),
    ("TIER_2_AUTO_RESOLVED", "tier2_auto_resolved_pending_revalidation"),
]


def main():
    conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=300)
    client = LLMClient("llama3.1:8b", timeout=180.0)
    vocab = VocabularyRetriever(conn)
    gold_path = _first_existing(GOLD_CANDIDATES, "gold")

    out_lines = []
    n_correct = 0
    n_graded = 0
    n_errors = 0

    for tier, queue_reason in CATEGORIES:
        tier_clause = "d.tier = ?" if tier else "d.tier IS NULL"
        params = [tier] if tier else []
        params += [queue_reason]
        rows = conn.execute(f"""
            SELECT d.entity_id, d.note_id, e.original_text, e.entity_label,
                   e.section_name, e.local_context, e.assertion_status, n.candidates
            FROM mollm_tier_gate_decisions d
            JOIN extracted_entities e ON e.entity_id = d.entity_id
            JOIN normalized_entities n ON n.entity_id = d.entity_id
            WHERE {tier_clause} AND d.queue_reason = ?
            AND n.candidates IS NOT NULL AND n.candidates != '[]'
            ORDER BY d.entity_id
            LIMIT {N_PER_CATEGORY}
        """, params).fetchall()

        header = f"\n{'='*100}\nCATEGORY: tier={tier} queue_reason={queue_reason}  ({len(rows)} entities)\n{'='*100}"
        print(header)
        out_lines.append(header)

        gold_rows = load_gold(gold_path, [r[1] for r in rows])
        gold_by_note = {}
        for g in gold_rows:
            gold_by_note.setdefault(g["note_id"], []).append(g)

        for entity_id, note_id, orig_text, label, section, local_ctx, assertion, cands_json in rows:
            candidates = cands_json if isinstance(cands_json, list) else json.loads(cands_json)
            entity = {"entity_id": entity_id, "note_id": note_id, "original_text": orig_text,
                      "gliner_label": label, "section_name": section,
                      "local_context": local_ctx, "assertion_status": assertion}

            result = escalate_to_8b(client, conn, entity, candidates)

            block = [
                f"\n--- entity_id={entity_id}  note={note_id}  text={orig_text!r}  label={label} ---",
                f"PROMPT SENT:\n{result['prompt']}",
                f"\nKG CONTEXT (raw): {json.dumps(result['kg_context'], indent=2, default=str)}",
                f"\nVERDICT: index={result['index']}  error={result['error']}",
                f"REASONING: {result['reasoning']}",
            ]

            if result["index"] is not None:
                chosen = candidates[result["index"] - 1]
                block.append(f"CHOSEN CONCEPT: {chosen.get('omop_concept_id')} {chosen.get('concept_name')}")

                # Grade against gold (clean-span only, same methodology as
                # evaluation/tier_gate_grading.py).
                orig_start = conn.execute(
                    "SELECT orig_start, orig_end FROM extracted_entities WHERE entity_id = ?",
                    [entity_id]).fetchone()
                gold_matches = [g for g in gold_by_note.get(note_id, [])
                                if overlaps(orig_start[0], orig_start[1], g["start"], g["end"])]
                if len(gold_matches) == 1:
                    gold = gold_matches[0]
                    pred_snomed = vocab.snomed_code_for_concept(chosen.get("omop_concept_id"))
                    correct = pred_snomed == gold["concept_id"]
                    n_graded += 1
                    n_correct += int(correct)
                    block.append(f"GOLD: {gold['concept_id']}  |  CORRECT: {correct}")
                else:
                    block.append(f"GOLD: not clean-gradable ({len(gold_matches)} overlapping gold spans)")
            else:
                n_errors += 1

            text = "\n".join(block)
            print(text)
            out_lines.append(text)

    summary = (f"\n{'='*100}\nSMOKE TEST SUMMARY: {n_correct}/{n_graded} clean-gradable correct "
              f"({100*n_correct/n_graded:.1f}%)  |  {n_errors} escalation errors\n{'='*100}")
    print(summary)
    out_lines.append(summary)

    out_path = f"{PROJECT_DIR}/logs/smoke_test_8b_kg_escalation.log"
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"\nFull transcript written to {out_path}")

    conn.close()


if __name__ == "__main__":
    main()
