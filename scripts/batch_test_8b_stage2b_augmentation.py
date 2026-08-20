"""scripts/batch_test_8b_stage2b_augmentation.py -- 2026-08-20 properly-
scoped batch test of src.tier2b_llm_candidate_generation (the 8B
generate-then-verify Stage 2b augmentation), with correct gold grading
(reusing evaluation/tier_gate_grading.py's proven SNOMED-crosswalk +
clean-span methodology, not the ad-hoc manual pulls used during initial
smoke testing).

METRIC: for each entity, checks whether gold's SNOMED code is present
ANYWHERE in the candidate pool -- BEFORE augmentation (Stage 2b's existing
candidates alone) and AFTER (existing + the new LLM-verified candidate, if
any). This measures RECALL RECOVERY: did augmentation put the correct
answer somewhere in reach that wasn't there before? (Whether Stage 3 would
then actually PICK it is a separate question this script doesn't test --
that needs a full route_tier() re-run, a bigger and more expensive next
step if this recall signal looks real.)

Also reports, separately: of the NEW candidates actually added, how many
are themselves exactly correct (precision of the augmentation itself).

Run: python3 scripts/batch_test_8b_stage2b_augmentation.py
"""
import collections
import json
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.tier2b_llm_candidate_generation import augment_candidates_with_llm  # noqa: E402
from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from scripts.score_gold_recall import load_gold, overlaps  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402

DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
N_PER_CATEGORY = 15

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
    totals = collections.Counter()
    by_category = collections.defaultdict(collections.Counter)

    for tier, queue_reason in CATEGORIES:
        tier_clause = "d.tier = ?" if tier else "d.tier IS NULL"
        params = [tier] if tier else []
        params += [queue_reason]
        rows = conn.execute(f"""
            SELECT d.entity_id, d.note_id, e.original_text, e.entity_label,
                   e.section_name, e.local_context, e.assertion_status,
                   e.orig_start, e.orig_end, n.candidates
            FROM mollm_tier_gate_decisions d
            JOIN extracted_entities e ON e.entity_id = d.entity_id
            JOIN normalized_entities n ON n.entity_id = d.entity_id
            WHERE {tier_clause} AND d.queue_reason = ?
            AND n.candidates IS NOT NULL AND n.candidates != '[]'
            ORDER BY d.entity_id
            LIMIT {N_PER_CATEGORY}
        """, params).fetchall()

        cat_key = f"{tier}/{queue_reason}"
        header = f"\n{'='*100}\nCATEGORY: {cat_key}  ({len(rows)} entities)\n{'='*100}"
        print(header)
        out_lines.append(header)

        gold_rows = load_gold(gold_path, [r[1] for r in rows])
        gold_by_note = collections.defaultdict(list)
        for g in gold_rows:
            gold_by_note[g["note_id"]].append(g)

        for (entity_id, note_id, orig_text, label, section, local_ctx, assertion,
             orig_start, orig_end, cands_json) in rows:
            candidates = cands_json if isinstance(cands_json, list) else json.loads(cands_json)
            entity = {"entity_id": entity_id, "note_id": note_id, "original_text": orig_text,
                      "gliner_label": label, "section_name": section,
                      "local_context": local_ctx, "assertion_status": assertion}

            gold_matches = [g for g in gold_by_note.get(note_id, [])
                            if overlaps(orig_start, orig_end, g["start"], g["end"])]
            totals["n_entities"] += 1
            by_category[cat_key]["n_entities"] += 1
            if len(gold_matches) != 1:
                totals["n_not_clean_gradable"] += 1
                by_category[cat_key]["n_not_clean_gradable"] += 1
                continue
            gold_snomed = gold_matches[0]["concept_id"]

            existing_snomed = {vocab.snomed_code_for_concept(c.get("omop_concept_id"))
                               for c in candidates}
            had_correct_before = gold_snomed in existing_snomed
            totals["n_gradable"] += 1
            by_category[cat_key]["n_gradable"] += 1
            if had_correct_before:
                totals["n_had_correct_before"] += 1
                by_category[cat_key]["n_had_correct_before"] += 1

            result = augment_candidates_with_llm(client, conn, entity, candidates)
            new_cand = result["new_candidate"]

            block = [f"\n--- {entity_id}  text={orig_text!r}  label={label} "
                    f"had_correct_before={had_correct_before} ---"]
            block.append(f"generated: {result['generated']}")
            if new_cand:
                new_snomed = vocab.snomed_code_for_concept(new_cand["omop_concept_id"])
                new_is_correct = new_snomed == gold_snomed
                has_correct_after = had_correct_before or new_is_correct
                block.append(f"NEW CANDIDATE: {new_cand['omop_concept_id']} "
                            f"{new_cand['concept_name']}  snomed={new_snomed}  "
                            f"gold={gold_snomed}  NEW_CANDIDATE_CORRECT={new_is_correct}")
                totals["n_new_candidate_added"] += 1
                by_category[cat_key]["n_new_candidate_added"] += 1
                if new_is_correct:
                    totals["n_new_candidate_matches_gold"] += 1
                    by_category[cat_key]["n_new_candidate_matches_gold"] += 1
                if has_correct_after and not had_correct_before:
                    totals["n_recall_recovered"] += 1
                    by_category[cat_key]["n_recall_recovered"] += 1
            else:
                block.append(f"NEW CANDIDATE: none (verified={result['verified']}, "
                            f"already_present={result['already_present']})")

            text = "\n".join(block)
            print(text)
            out_lines.append(text)

    summary_lines = [f"\n{'='*100}", "OVERALL SUMMARY", "=" * 100]
    summary_lines.append(f"entities sampled: {totals['n_entities']}  "
                         f"clean-gradable: {totals['n_gradable']}  "
                         f"not clean-gradable (skipped): {totals['n_not_clean_gradable']}")
    summary_lines.append(f"already had correct candidate in pool BEFORE augmentation: "
                         f"{totals['n_had_correct_before']}/{totals['n_gradable']}")
    summary_lines.append(f"new candidate added: {totals['n_new_candidate_added']}/{totals['n_gradable']}")
    summary_lines.append(f"of those, new candidate itself matches gold: "
                         f"{totals['n_new_candidate_matches_gold']}/{totals['n_new_candidate_added'] or 1}")
    summary_lines.append(f"RECALL RECOVERED (correct answer now in pool, wasn't before): "
                         f"{totals['n_recall_recovered']}/{totals['n_gradable']}")
    summary_lines.append("\nBy category:")
    for cat_key, c in by_category.items():
        summary_lines.append(f"  {cat_key}: gradable={c['n_gradable']} "
                            f"had_before={c['n_had_correct_before']} "
                            f"new_added={c['n_new_candidate_added']} "
                            f"new_correct={c['n_new_candidate_matches_gold']} "
                            f"recall_recovered={c['n_recall_recovered']}")

    summary = "\n".join(summary_lines)
    print(summary)
    out_lines.append(summary)

    out_path = f"{PROJECT_DIR}/logs/batch_test_8b_stage2b_augmentation.log"
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"\nFull transcript written to {out_path}")

    conn.close()


if __name__ == "__main__":
    main()
