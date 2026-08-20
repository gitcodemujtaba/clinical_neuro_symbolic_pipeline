"""evaluation/exhaustive_candidate_eval_impact.py -- 2026-08-20: the
planned follow-up to EXHAUSTIVE_CANDIDATE_EVAL_ENABLED (src.mollm_tier_gate),
per the project's own tracked memory note. That flag was flipped default-on
to fix one verified pattern (wound-dehiscence-class SNOMED duplicate
concepts, 11/14 corpus-measured) but its cost was already quantified
separately (~34% more LLM calls, 16.88% of model-runs hit a genuine
2+-independently-accepted-candidate situation) WITHOUT a matching accuracy
read. This script provides that missing half: grades the tiebreak-eligible
subset against gold and compares its precision to the non-eligible
population, so the flag's net impact (not just its cost) is measured.

TIEBREAK-ELIGIBLE definition (same detection logic already used earlier
this session to confirm _resolve_tiebreak() fired correctly for a specific
entity): a model's eval_trail shows n_accepts =
sum(1 for t in trail if t.get("match")) >= 2 -- i.e. that model
independently accepted 2+ candidates as clinically matching before the
comparative tiebreak call (if any) resolved it. An ENTITY counts as
tiebreak-eligible if ANY of its 3 models hit this.

Run: python3 -m evaluation.exhaustive_candidate_eval_impact --notes 5
"""
import argparse
import collections
import json
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def is_tiebreak_eligible(models: list) -> bool:
    """Pure function -- no DB, directly unit-testable."""
    for m in models or []:
        trail = m.get("eval_trail") or []
        n_accepts = sum(1 for t in trail if t.get("match"))
        if n_accepts >= 2:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", type=int, default=5,
                        help="Number of notes to scope this run to (test run default: 5)")
    args = parser.parse_args()

    from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing
    from scripts.score_gold_recall import load_gold, overlaps
    from src.db_utils import connect_with_retry
    from src.retrieval import VocabularyRetriever

    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb",
                              read_only=True, max_wait_seconds=60)
    vocab = VocabularyRetriever(conn)

    note_ids = [r[0] for r in conn.execute("""
        SELECT note_id, COUNT(*) c FROM mollm_tier_gate_decisions
        WHERE models IS NOT NULL AND models != '[]'
        GROUP BY note_id ORDER BY c DESC LIMIT ?
    """, [args.notes]).fetchall()]
    print(f"scoped to {len(note_ids)} notes: {note_ids}")

    placeholders = ",".join("?" * len(note_ids))
    rows = conn.execute(f"""
        SELECT d.entity_id, d.note_id, e.orig_start, e.orig_end, d.tier,
               d.final_candidate_index, d.models, n.candidates
        FROM mollm_tier_gate_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.note_id IN ({placeholders})
        AND d.models IS NOT NULL AND d.models != '[]'
    """, note_ids).fetchall()
    print(f"{len(rows)} decisions in scope")

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, note_ids)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    buckets = {"eligible": collections.Counter(), "not_eligible": collections.Counter()}

    for entity_id, note_id, s, e, tier, final_idx, models_json, cands_json in rows:
        models = models_json if isinstance(models_json, list) else json.loads(models_json)
        cands = cands_json if isinstance(cands_json, list) else json.loads(cands_json)
        if not final_idx or not cands or not (1 <= final_idx <= len(cands)):
            continue

        gold = gold_by_note.get(note_id, [])
        overlapping = [g for g in gold if overlaps(s, e, g["start"], g["end"])]
        if len(overlapping) != 1:
            continue
        g0 = overlapping[0]
        if (e - s) < (g0["end"] - g0["start"]):
            continue

        picked = cands[final_idx - 1]
        snomed = vocab.snomed_code_for_concept(picked.get("omop_concept_id"))
        correct = snomed is not None and str(snomed) == str(g0["concept_id"])

        bucket = "eligible" if is_tiebreak_eligible(models) else "not_eligible"
        buckets[bucket]["n"] += 1
        if correct:
            buckets[bucket]["correct"] += 1

    conn.close()

    print("\n=== EXHAUSTIVE_CANDIDATE_EVAL_ENABLED net-impact assessment ===")
    for name in ("eligible", "not_eligible"):
        c = buckets[name]
        prec = c["correct"] / c["n"] if c["n"] else None
        prec_str = f"{prec:.1%}" if prec is not None else "n/a"
        print(f"{name:14s}: n={c['n']:4d}  correct={c['correct']:4d}  precision={prec_str}")

    e, ne = buckets["eligible"], buckets["not_eligible"]
    if e["n"] and ne["n"]:
        gap = (e["correct"] / e["n"]) - (ne["correct"] / ne["n"])
        print(f"\nprecision gap (eligible - not_eligible): {gap:+.1%}")
    print("\nRemember: this is a scoped test run (--notes "
         f"{args.notes}) -- treat as a sanity check on the methodology, "
         "widen the note count for a number worth citing in the paper.")


if __name__ == "__main__":
    main()
