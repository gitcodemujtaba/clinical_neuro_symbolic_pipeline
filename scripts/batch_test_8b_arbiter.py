"""scripts/batch_test_8b_arbiter.py -- 2026-08-20 test of src.tier4_arbiter_8b
(8B model arbitrating the 3B ensemble's own verdicts+reasoning) against
TIER_4_ENSEMBLE_SPLIT specifically (both 1:1:1 and 2:1 shapes).

SAMPLING FIX vs earlier tests: every prior smoke/batch test in this session
used `ORDER BY entity_id LIMIT N`, which happened to pull entities with no
clean gold overlap for this exact category three times in a row -- not bad
luck, a real sampling bias (entity_id order correlates with something,
likely note processing order, that also correlates with span/compound-span
shape). This script PRE-FILTERS for clean gold-gradability first (a single
gold span cleanly overlapping the entity), THEN samples from that already-
gradable pool, so every sampled entity is guaranteed comparable.

Compares the arbiter's verdict against: (a) gold, (b) the ORIGINAL
plurality verdict route_tier() actually recorded (i.e. does the arbiter
do better than what shipped).

Run: python3 scripts/batch_test_8b_arbiter.py [N]
"""
import collections
import json
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.tier4_arbiter_8b import arbitrate  # noqa: E402
from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from evaluation.tier_gate_grading import plurality_candidate_index  # noqa: E402
from scripts.score_gold_recall import load_gold, overlaps  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402

DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def main():
    conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=300)
    client = LLMClient("llama3.1:8b", timeout=180.0)
    vocab = VocabularyRetriever(conn)
    gold_path = _first_existing(GOLD_CANDIDATES, "gold")

    all_rows = conn.execute("""
        SELECT d.entity_id, d.note_id, d.models, d.final_candidate_index,
               e.original_text, e.entity_label, e.section_name, e.local_context,
               e.assertion_status, e.orig_start, e.orig_end, n.candidates
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
        note_id, orig_start, orig_end = row[1], row[9], row[10]
        matches = [g for g in gold_by_note.get(note_id, [])
                  if overlaps(orig_start, orig_end, g["start"], g["end"])]
        if len(matches) == 1:
            gradable_by_note[note_id].append((row, matches[0]))
            n_gradable_total += 1

    # 2026-08-20 sampling fix: every prior test in this session used
    # ORDER BY entity_id LIMIT N, which repeatedly clustered on entities
    # from a single note (entity_id order correlates with note-processing
    # order). Round-robin across notes instead -- one entity per note per
    # pass, capped at MAX_PER_NOTE -- so a large sample is guaranteed to
    # span many distinct notes rather than being dominated by whichever
    # note happens to sort first.
    MAX_PER_NOTE = 3
    note_ids_with_gradable = sorted(gradable_by_note.keys())
    sample = []
    round_idx = 0
    while len(sample) < N and round_idx < MAX_PER_NOTE:
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
          f"(max {MAX_PER_NOTE}/note) spanning {n_notes_used} distinct notes")

    out_lines = []
    n_arbiter_correct = 0
    n_original_correct = 0
    n_arbiter_changed_verdict = 0
    n_arbiter_better = 0
    n_arbiter_worse = 0
    n_graded = 0

    for row, gold in sample:
        (entity_id, note_id, models_json, orig_final_idx, orig_text, label,
         section, local_ctx, assertion, orig_start, orig_end, cands_json) = row
        candidates = cands_json if isinstance(cands_json, list) else json.loads(cands_json)
        model_results = json.loads(models_json) if isinstance(models_json, str) else models_json
        entity = {"original_text": orig_text, "gliner_label": label, "section_name": section,
                  "local_context": local_ctx, "assertion_status": assertion}

        result = arbitrate(client, conn, entity, candidates, model_results)
        n_graded += 1

        gold_snomed = gold["concept_id"]
        orig_idx, _, _ = plurality_candidate_index(model_results)
        orig_correct = False
        if orig_idx and orig_idx <= len(candidates):
            orig_snomed = vocab.snomed_code_for_concept(candidates[orig_idx - 1].get("omop_concept_id"))
            orig_correct = orig_snomed == gold_snomed
        n_original_correct += int(orig_correct)

        arbiter_correct = False
        if result["index"]:
            arb_snomed = vocab.snomed_code_for_concept(candidates[result["index"] - 1].get("omop_concept_id"))
            arbiter_correct = arb_snomed == gold_snomed
        n_arbiter_correct += int(arbiter_correct)

        changed = result["index"] != orig_idx
        n_arbiter_changed_verdict += int(changed)
        if changed and arbiter_correct and not orig_correct:
            n_arbiter_better += 1
        if changed and not arbiter_correct and orig_correct:
            n_arbiter_worse += 1

        block = [
            f"\n--- {entity_id}  text={orig_text!r}  label={label} ---",
            f"ORIGINAL plurality index={orig_idx}  correct={orig_correct}",
            f"ARBITER index={result['index']}  correct={arbiter_correct}  "
            f"agrees_with_majority={result['agrees_with_majority']}  error={result['error']}",
            f"ARBITER reasoning: {result['reasoning']}",
            f"gold_snomed={gold_snomed}",
        ]
        text = "\n".join(block)
        print(text)
        out_lines.append(text)

    summary = (f"\n{'='*100}\nSUMMARY (n={n_graded})\n"
              f"original plurality precision: {n_original_correct}/{n_graded} "
              f"({100*n_original_correct/n_graded:.1f}%)\n"
              f"arbiter precision:             {n_arbiter_correct}/{n_graded} "
              f"({100*n_arbiter_correct/n_graded:.1f}%)\n"
              f"arbiter changed the verdict: {n_arbiter_changed_verdict}/{n_graded}\n"
              f"  of those changes: {n_arbiter_better} made it correct (was wrong), "
              f"{n_arbiter_worse} made it wrong (was correct)\n{'='*100}")
    print(summary)
    out_lines.append(summary)

    out_path = f"{PROJECT_DIR}/logs/batch_test_8b_arbiter.log"
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"\nFull transcript: {out_path}")
    conn.close()


if __name__ == "__main__":
    main()
