"""scripts/sweep_tier3_floor_downward.py -- 2026-09-01: real re-computation
sweep of TIER3_SIMILARITY_FLOOR (currently 0.72) BELOW its current value.

WHY THIS NEEDS REAL RE-COMPUTATION, UNLIKE THE UPWARD DIRECTION. Below
the floor, orchestrator.py reports NO_CANDIDATE and drops the candidate
list entirely -- nothing is persisted for that population beyond a
tier_trace.top_score (real, but doesn't say WHICH concept, so precision
can't be graded from stored data alone). This script re-runs the real
production normalize_entity() (not a reimplementation) with
TIER3_SIMILARITY_FLOOR monkeypatched to 0.0 (effectively disabled) for a
sample of entities currently getting NO_CANDIDATE, to recover their real
top-1 concept and grade it against gold.

SAMPLE, NOT THE FULL 2,467 -- stated honestly, not silently. Real
SapBERT calls, not free; scoped to a bounded, still-statistically-
meaningful sample rather than the full below-floor population.

Run: python3 scripts/sweep_tier3_floor_downward.py [--sample-size 300]
"""
import argparse
import sys
import time
import collections

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

FLOORS = [0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.72]


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=300)
    args = ap.parse_args()

    from src.db_utils import connect_with_retry
    from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing
    from scripts.score_gold_recall import load_gold, overlaps
    from src.retrieval import VocabularyRetriever
    import src.normalization.orchestrator as orch
    import src.normalization.tier_retrieval as tr

    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb", read_only=True)
    vocab = VocabularyRetriever(conn)

    # Population: currently NO_CANDIDATE (match_tier '0 (Failed)') entities
    # that actually have gold coverage (span overlap), so precision is
    # gradable. Real, current entity_ids -- not simulated.
    note_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test = TRUE").fetchall()]
    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, note_ids)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    failed = conn.execute("""
        SELECT n.entity_id, n.note_id, n.expanded_text, n.gliner_label,
               e.orig_start, e.orig_end
        FROM normalized_entities n
        JOIN extracted_entities e ON e.entity_id = n.entity_id
        WHERE n.match_tier = '0 (Failed)' AND n.is_test = TRUE
    """).fetchall()
    print(f"{len(failed)} total NO_CANDIDATE entities")

    gradable = []
    for entity_id, note_id, expanded_text, label, s, e in failed:
        golds = gold_by_note.get(note_id, [])
        gold_hit = next((g for g in golds if overlaps(s, e, g["start"], g["end"])), None)
        if gold_hit is not None:
            gradable.append((entity_id, note_id, expanded_text, label, gold_hit["concept_id"]))
    print(f"{len(gradable)} of those have gold span coverage (gradable)")

    import random
    random.seed(42)
    sample = random.sample(gradable, min(args.sample_size, len(gradable)))
    print(f"sampling {len(sample)} for real re-computation "
          f"(TIER3_SIMILARITY_FLOOR temporarily disabled)\n")

    # Monkeypatch the floor to 0.0 so normalize_entity()'s real code path
    # returns whatever top-1 candidate actually exists, at any score --
    # the exact production function, not a reimplementation.
    orig_floor = orch.TIER3_SIMILARITY_FLOOR
    orch.TIER3_SIMILARITY_FLOOR = 0.0
    tr.TIER3_SIMILARITY_FLOOR = 0.0

    results = []  # (top_score, correct: bool)
    start = time.time()
    n_errors = 0
    for i, (entity_id, note_id, expanded_text, label, gold_concept_id) in enumerate(sample, 1):
        try:
            mapping = orch.normalize_entity(expanded_text, conn, gliner_label=label)
        except Exception as exc:
            n_errors += 1
            continue
        score = mapping.get("score")
        concept_id = mapping.get("concept_id")
        if score is None or concept_id is None:
            continue
        # gold_concept_id is a raw SNOMED code; normalize_entity() returns
        # an OMOP internal concept_id -- two different id spaces. Must
        # crosswalk before comparing (the same real bug this project
        # already found once this session for gazetteer concept lookups).
        snomed_code = vocab.snomed_code_for_concept(concept_id)
        correct = snomed_code is not None and str(snomed_code) == str(gold_concept_id)
        results.append((score, correct))
        if i % 50 == 0:
            print(f"  [{time.time()-start:.0f}s] {i}/{len(sample)} done, {n_errors} errors")

    orch.TIER3_SIMILARITY_FLOOR = orig_floor
    tr.TIER3_SIMILARITY_FLOOR = orig_floor
    conn.close()

    print(f"\n{len(results)} real, gradable re-computed results "
          f"({n_errors} errors), {time.time()-start:.0f}s total\n")

    print(f"{'floor':>8} {'n_clearing':>11} {'n_correct':>10} {'precision':>11} "
          f"{'cum_recall_gain':>16}")
    for floor in FLOORS:
        clearing = [c for s, c in results if s >= floor]
        n_clear = len(clearing)
        n_correct = sum(clearing)
        precision = n_correct / n_clear if n_clear else 0.0
        # scale sample recall-gain up to the real below-floor population
        est_gain = n_clear / len(sample) * len(gradable) if sample else 0
        print(f"{floor:>8.2f} {n_clear:>11} {n_correct:>10} {precision*100:>10.2f}% "
              f"{est_gain:>15.0f} (extrapolated)")


if __name__ == "__main__":
    main()
