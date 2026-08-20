"""scripts/smoke_test_kg_search_loop.py -- 2026-08-20 smoke test for
src.kg_search_loop against real Tier 4 (TIER_4_ENSEMBLE_SPLIT) and
Tier 5 (TIER_5_TRUE_AMBIGUITY) entities from the fresh25 batch. Small N by
design (a smoke test, not the batch validation this module's own docstring
says it still needs) -- prints the full prompt/search-trace/verdict
transcript for each entity so it's directly inspectable, same discipline
established for every other escalation module this session.

Run: python3 scripts/smoke_test_kg_search_loop.py [N_PER_TIER]
"""
import json
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.kg_search_loop import run_kg_search_loop  # noqa: E402
from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from evaluation.grade_fresh25_by_tier import NOTE_IDS  # noqa: E402
from scripts.score_gold_recall import load_gold, overlaps  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402

DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
N_PER_TIER = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def main():
    conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=300)
    client = LLMClient("llama3.1:8b", timeout=180.0)
    vocab = VocabularyRetriever(conn)

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, NOTE_IDS)
    gold_by_note = {}
    for g in gold_rows:
        gold_by_note.setdefault(g["note_id"], []).append(g)

    note_ph = ",".join("?" * len(NOTE_IDS))
    out_lines = []

    for tier, queue_reason_filter in [
        ("TIER_4_ENSEMBLE_SPLIT", "ensemble_split"),
        ("TIER_5_TRUE_AMBIGUITY", None),
    ]:
        clause = "AND d.queue_reason = ?" if queue_reason_filter else ""
        params = NOTE_IDS + ([queue_reason_filter] if queue_reason_filter else [])
        rows = conn.execute(f"""
            SELECT d.entity_id, d.note_id, e.original_text, e.entity_label,
                   e.section_name, e.local_context, e.assertion_status,
                   e.orig_start, e.orig_end, n.candidates
            FROM mollm_tier_gate_decisions d
            JOIN extracted_entities e ON e.entity_id = d.entity_id
            JOIN normalized_entities n ON n.entity_id = d.entity_id
            WHERE d.tier = ? AND d.note_id IN ({note_ph}) {clause}
            AND n.candidates IS NOT NULL AND n.candidates != '[]'
            ORDER BY d.entity_id LIMIT ?
        """, [tier] + params + [N_PER_TIER]).fetchall()

        header = f"\n{'='*100}\n{tier} ({len(rows)} sampled)\n{'='*100}"
        print(header)
        out_lines.append(header)

        for (entity_id, note_id, text, label, section, local_ctx, assertion,
             s, e, cands_json) in rows:
            candidates = cands_json if isinstance(cands_json, list) else json.loads(cands_json)
            entity = {"original_text": text, "gliner_label": label, "section_name": section,
                      "local_context": local_ctx, "assertion_status": assertion}

            result = run_kg_search_loop(client, conn, entity, candidates)

            gold = gold_by_note.get(note_id, [])
            overlapping = [g for g in gold if overlaps(s, e, g["start"], g["end"])]
            gold_code = overlapping[0]["concept_id"] if len(overlapping) == 1 else None

            pred_code = None
            if result["index"]:
                cid = candidates[result["index"] - 1].get("omop_concept_id")
                pred_code = vocab.snomed_code_for_concept(cid) if cid else None
            correct = (gold_code is not None and pred_code is not None
                      and str(pred_code) == str(gold_code))

            block = [
                f"\n--- {entity_id}  text={text!r}  label={label} ---",
                f"n_candidates={len(candidates)}  rounds={result['n_rounds']}  "
                f"searches={result['n_searches']}  error={result['error']}",
                f"VERDICT index={result['index']}  reasoning={result['reasoning']}",
                f"gold_concept_id={gold_code}  correct={correct if gold_code else 'ungradable'}",
            ]
            for t in result["trace"]:
                block.append(f"  [round {t['round']}] action="
                             f"{(t['raw_response'] or {}).get('action')}  "
                             f"n_searches_requested={len(t['searches_requested'])}")
                for sr in t["searches_requested"]:
                    block.append(f"      requested: {sr}")
            text_out = "\n".join(block)
            print(text_out)
            out_lines.append(text_out)

    out_path = f"{PROJECT_DIR}/logs/smoke_test_kg_search_loop.log"
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"\nFull transcript: {out_path}")
    conn.close()


if __name__ == "__main__":
    main()
