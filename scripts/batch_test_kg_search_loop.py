"""scripts/batch_test_kg_search_loop.py -- 2026-08-20 graded batch test of
src.kg_search_loop against TIER_4_ENSEMBLE_SPLIT, same "better vs worse"
metric framework as scripts/batch_test_8b_arbiter.py: compares the loop's
verdict against (a) gold, (b) the ORIGINAL plurality verdict route_tier()
actually recorded (does the search loop do better than what shipped).

Same note-diversified sampling discipline as batch_test_8b_arbiter.py
(round-robin across notes, max N_PER_NOTE per note) -- this session
already hit a real single-note sampling-bias bug once with a naive
ORDER BY entity_id LIMIT N, not repeating it here.

Run: python3 scripts/batch_test_kg_search_loop.py [N]
"""
import collections
import json
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.kg_search_loop import run_kg_search_loop  # noqa: E402
from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from evaluation.tier_gate_grading import plurality_candidate_index  # noqa: E402
from scripts.score_gold_recall import load_gold, overlaps  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402

DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
N_PER_NOTE = 3


def main():
    conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=300)
    client = LLMClient("llama3.1:8b", timeout=180.0)
    vocab = VocabularyRetriever(conn)
    gold_path = _first_existing(GOLD_CANDIDATES, "gold")

    all_rows = conn.execute("""
        SELECT d.entity_id, d.note_id, d.models, e.original_text, e.entity_label,
               e.section_name, e.local_context, e.assertion_status,
               e.orig_start, e.orig_end, n.candidates
        FROM mollm_tier_gate_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.tier = 'TIER_4_ENSEMBLE_SPLIT' AND d.queue_reason = 'ensemble_split'
        AND n.candidates IS NOT NULL AND n.candidates != '[]'
    """).fetchall()
    print(f"{len(all_rows)} total ensemble_split entities in DB")

    all_notes = sorted({r[1] for r in all_rows})
    gold_rows = load_gold(gold_path, all_notes)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    gradable_by_note = collections.defaultdict(list)
    n_gradable_total = 0
    for row in all_rows:
        note_id, s, e = row[1], row[8], row[9]
        matches = [g for g in gold_by_note.get(note_id, [])
                  if overlaps(s, e, g["start"], g["end"])]
        if len(matches) == 1:
            gradable_by_note[note_id].append((row, matches[0]))
            n_gradable_total += 1

    note_ids_with_gradable = sorted(gradable_by_note.keys())
    sample = []
    round_idx = 0
    while len(sample) < N and round_idx < N_PER_NOTE:
        for note_id in note_ids_with_gradable:
            if len(sample) >= N:
                break
            bucket = gradable_by_note[note_id]
            if round_idx < len(bucket):
                sample.append(bucket[round_idx])
        round_idx += 1

    n_notes_used = len({row[1] for row, _ in sample})
    print(f"{n_gradable_total} clean-gradable (single gold overlap) across "
          f"{len(note_ids_with_gradable)} notes -- sampling {len(sample)} "
          f"(max {N_PER_NOTE}/note) spanning {n_notes_used} distinct notes")

    out_lines = []
    n_graded = 0
    n_loop_correct = 0
    n_original_correct = 0
    n_loop_changed_verdict = 0
    n_loop_better = 0
    n_loop_worse = 0
    n_loop_errors = 0
    total_rounds = 0
    total_searches = 0

    for i, (row, gold) in enumerate(sample, 1):
        (entity_id, note_id, models_json, orig_text, label, section, local_ctx,
         assertion, s, e, cands_json) = row
        candidates = cands_json if isinstance(cands_json, list) else json.loads(cands_json)
        model_results = json.loads(models_json) if isinstance(models_json, str) else models_json
        entity = {"original_text": orig_text, "gliner_label": label, "section_name": section,
                  "local_context": local_ctx, "assertion_status": assertion}

        result = run_kg_search_loop(client, conn, entity, candidates)
        n_graded += 1
        total_rounds += result["n_rounds"]
        total_searches += result["n_searches"]
        if result["error"]:
            n_loop_errors += 1

        gold_snomed = gold["concept_id"]
        orig_idx, _, _ = plurality_candidate_index(model_results)
        orig_correct = False
        if orig_idx and orig_idx <= len(candidates):
            orig_snomed = vocab.snomed_code_for_concept(candidates[orig_idx - 1].get("omop_concept_id"))
            orig_correct = orig_snomed == gold_snomed
        n_original_correct += int(orig_correct)

        loop_correct = False
        if result["index"]:
            loop_snomed = vocab.snomed_code_for_concept(candidates[result["index"] - 1].get("omop_concept_id"))
            loop_correct = loop_snomed == gold_snomed
        n_loop_correct += int(loop_correct)

        changed = result["index"] != orig_idx
        n_loop_changed_verdict += int(changed)
        if changed and loop_correct and not orig_correct:
            n_loop_better += 1
        if changed and not loop_correct and orig_correct:
            n_loop_worse += 1

        block = (f"\n[{i}/{len(sample)}] {entity_id}  text={orig_text!r}  label={label}\n"
                f"  ORIGINAL plurality index={orig_idx}  correct={orig_correct}\n"
                f"  LOOP index={result['index']}  correct={loop_correct}  "
                f"rounds={result['n_rounds']}  searches={result['n_searches']}  "
                f"error={result['error']}\n"
                f"  LOOP reasoning: {result['reasoning']}\n"
                f"  gold_snomed={gold_snomed}")
        print(block)
        out_lines.append(block)

    summary = (f"\n{'='*100}\nSUMMARY (n={n_graded})\n"
              f"original plurality precision: {n_original_correct}/{n_graded} "
              f"({100*n_original_correct/n_graded:.1f}%)\n"
              f"KG search loop precision:      {n_loop_correct}/{n_graded} "
              f"({100*n_loop_correct/n_graded:.1f}%)\n"
              f"loop changed the verdict: {n_loop_changed_verdict}/{n_graded}\n"
              f"  of those changes: {n_loop_better} made it correct (was wrong), "
              f"{n_loop_worse} made it wrong (was correct)\n"
              f"errors: {n_loop_errors}/{n_graded}\n"
              f"avg rounds/entity: {total_rounds/n_graded:.2f}  "
              f"avg searches/entity: {total_searches/n_graded:.2f}\n{'='*100}")
    print(summary)
    out_lines.append(summary)

    out_path = f"{PROJECT_DIR}/logs/batch_test_kg_search_loop.log"
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"\nFull transcript: {out_path}")
    conn.close()


if __name__ == "__main__":
    main()
