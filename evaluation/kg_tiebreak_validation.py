"""evaluation/kg_tiebreak_validation.py -- 2026-08-20: validates
src.kg_embedding_tiebreak's pool-consistency tiebreak against real gold
data, sweeping a TIE_THRESHOLD parameter, and compares it head-to-head
against the existing hardcoded _prefer_lab_procedure_over_observable() rule
(src.normalization.tier_retrieval) on the subset where both mechanisms
apply.

ZERO LIVE SAPBERT/EMBEDDING CALLS. Reuses each entity's ALREADY-STORED
normalized_entities.candidates -- similarity_score per candidate was
computed once at Stage 2b time and is just read back here. This matters
concretely, not just as a performance nicety: an earlier attempt to verify
a related question live (recomputing a fresh SapBERT embedding in a
parallel process while a Stage 3 backfill was running) measurably stalled
that backfill for ~22 minutes via disk/CPU contention on this box. This
harness is designed to never repeat that mistake -- the only "load" it
does is the TransE checkpoint itself (small, CPU, plain PyTorch).

NEEDS A TRAINED MODEL CHECKPOINT (models/kg_transe_v1.pt, written by
scripts/build_kg_embeddings.py's save_model() call) to run for real.
Without one, main() raises a clear error rather than silently producing
empty/misleading results -- this is expected to be run for the first time
only after the current Stage 3 backfill finishes and the KGE model is
retrained on the larger post-backfill TP record pool, per the standing
plan. The pure classification logic below has its own unit tests
(tests/test_kg_tiebreak_validation.py) that need no DB, no model, and no
live run at all -- exercised now, while the backfill is still running.

THE THREE-WAY OUTCOME, per entity in a tied-pair subpopulation:
  WIN:     baseline (SapBERT top-1) was gold-WRONG, the mechanism's pick
           is gold-CORRECT.
  LOSS:    baseline was gold-CORRECT, the mechanism's pick is gold-WRONG
           (the fatal case -- a mechanism that causes this on net should
           not ship, no matter how many wins it also produces).
  NEUTRAL: both right, both wrong, or the mechanism didn't change the pick
           (includes "unresolved" cases where KGE had no usable score for
           either tied candidate and fell back to the baseline unchanged).

Run: python3 -m evaluation.kg_tiebreak_validation [--thresholds 0.01,0.02,0.03,0.05,0.08]
"""
import argparse
import collections
import json
import os
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

DEFAULT_MODEL_PATH = f"{PROJECT_DIR}/models/kg_transe_v1.pt"
DEFAULT_THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.08]

# Same class pair src.normalization.tier_retrieval._prefer_lab_procedure_
# over_observable() penalizes (as of the 2026-08-20 Qualifier Value
# extension) -- duplicated here rather than imported so this harness can
# replicate the RULE'S decision in pure Python without re-running the real
# ranking function against a partial/reconstructed candidate list.
_HARDCODED_RULE_PENALIZED_CLASSES = {"Observable Entity", "Qualifier Value"}
_HARDCODED_RULE_LABEL = "Lab Test"


def classify_outcome(baseline_correct, new_correct):
    """Pure three-way classifier, the actual metric this whole harness
    exists to compute. No DB, no model -- unit-testable directly."""
    if baseline_correct and not new_correct:
        return "loss"
    if not baseline_correct and new_correct:
        return "win"
    return "neutral"


def hardcoded_rule_applicable(entity_label, top1_class, top2_class):
    """True exactly when _prefer_lab_procedure_over_observable() would fire
    on this tied pair: a Lab-Test-labeled entity with one candidate in the
    penalized classes and the other 'Procedure'-class."""
    if entity_label != _HARDCODED_RULE_LABEL:
        return False
    classes = {top1_class, top2_class}
    return "Procedure" in classes and bool(classes & _HARDCODED_RULE_PENALIZED_CLASSES)


def hardcoded_rule_pick(top1_id, top1_class, top2_id, top2_class):
    """Replicates the rule's own winner selection for a two-candidate tied
    pair: the Procedure-class one wins outright (bonus large enough to
    flip every observed gap, per the rule's own docstring)."""
    if top1_class == "Procedure":
        return top1_id
    if top2_class == "Procedure":
        return top2_id
    return top1_id  # rule doesn't apply -- shouldn't be reached if caller checked applicable() first


def _load_candidate_pools(conn):
    """One row per gradable entity: entity_id, note_id, span, entity_label,
    and its Stage 2b candidate list (already-computed similarity_score per
    candidate, no live embedding calls). Excludes superseded rows, same
    discipline as every other grading script this session
    (evaluation/iou_metrics.py, evaluation/stage2a_cal_eval.py)."""
    rows = conn.execute("""
        SELECT e.entity_id, e.note_id, e.orig_start, e.orig_end, e.entity_label, n.candidates
        FROM extracted_entities e
        JOIN normalized_entities n ON n.entity_id = e.entity_id
        WHERE e.is_test = TRUE
          AND (e.superseded_by_split IS NULL OR e.superseded_by_split = FALSE)
          AND (e.superseded_by_growth IS NULL OR e.superseded_by_growth = FALSE)
          AND n.candidates IS NOT NULL AND n.candidates != '[]'
    """).fetchall()
    out = []
    for entity_id, note_id, s, e, label, cands_json in rows:
        candidates = cands_json if isinstance(cands_json, list) else json.loads(cands_json)
        if len(candidates) < 2:
            continue
        candidates = sorted(candidates, key=lambda c: c.get("similarity_score") or 0.0, reverse=True)
        out.append({"entity_id": entity_id, "note_id": note_id, "start": s, "end": e,
                    "entity_label": label, "candidates": candidates})
    return out


def _concept_class_map(conn, concept_ids):
    ids = [c for c in set(concept_ids) if c]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT concept_id, concept_class_id FROM athena_concept WHERE concept_id IN ({placeholders})",
        ids).fetchall()
    return dict(rows)


def run_validation(conn, model, entity2idx, vocab, gold_by_note, thresholds):
    """Returns {threshold: {"kge": {...counts}, "hardcoded_subset": {...counts}}}."""
    from scripts.score_gold_recall import overlaps
    from src.kg_embedding_tiebreak import pick_via_kg_tiebreak

    pools = _load_candidate_pools(conn)
    print(f"{len(pools)} gradable entities with >=2 Stage 2b candidates")

    all_concept_ids = set()
    for p in pools:
        all_concept_ids.update(c.get("omop_concept_id") for c in p["candidates"])
    class_map = _concept_class_map(conn, all_concept_ids)
    print(f"{len(class_map)} distinct concept_ids resolved to a concept_class_id")

    results = {t: {"kge": collections.Counter(), "hardcoded_subset": collections.Counter()}
               for t in thresholds}
    max_threshold = max(thresholds)

    for p in pools:
        top1, top2 = p["candidates"][0], p["candidates"][1]
        s1, s2 = top1.get("similarity_score") or 0.0, top2.get("similarity_score") or 0.0
        delta = s1 - s2
        if delta < 0 or delta > max_threshold:
            continue  # not a tie under even the widest threshold being swept

        top1_id, top2_id = top1.get("omop_concept_id"), top2.get("omop_concept_id")
        if not top1_id or not top2_id:
            continue

        gold = gold_by_note.get(p["note_id"], [])
        overlapping = [g for g in gold if overlaps(p["start"], p["end"], g["start"], g["end"])]
        if len(overlapping) != 1:
            continue
        g0 = overlapping[0]
        if (p["end"] - p["start"]) < (g0["end"] - g0["start"]):
            continue  # not a clean span, same discipline as every other grading script

        def _is_gold(concept_id):
            snomed = vocab.snomed_code_for_concept(concept_id)
            return snomed is not None and str(snomed) == str(g0["concept_id"])

        baseline_correct = _is_gold(top1_id)

        pool_ids = [c.get("omop_concept_id") for c in p["candidates"] if c.get("omop_concept_id")]
        kge_result = pick_via_kg_tiebreak(model, entity2idx, tied_concept_ids=[top1_id, top2_id],
                                          full_pool_concept_ids=pool_ids)
        kge_pick = kge_result["winner"] if kge_result["resolved"] else top1_id
        kge_correct = _is_gold(kge_pick)
        kge_outcome = classify_outcome(baseline_correct, kge_correct)

        top1_class, top2_class = class_map.get(top1_id), class_map.get(top2_id)
        rule_applies = hardcoded_rule_applicable(p["entity_label"], top1_class, top2_class)
        if rule_applies:
            rule_pick = hardcoded_rule_pick(top1_id, top1_class, top2_id, top2_class)
            rule_correct = _is_gold(rule_pick)
            rule_outcome = classify_outcome(baseline_correct, rule_correct)

        for t in thresholds:
            if delta > t:
                continue
            results[t]["kge"][kge_outcome] += 1
            results[t]["kge"]["n"] += 1
            results[t]["kge"]["n_resolved"] += 1 if kge_result["resolved"] else 0
            if rule_applies:
                results[t]["hardcoded_subset"]["kge_" + kge_outcome] += 1
                results[t]["hardcoded_subset"]["rule_" + rule_outcome] += 1
                results[t]["hardcoded_subset"]["n"] += 1

    return results


def _print_report(results, thresholds):
    print("\n=== TIE_THRESHOLD sweep: KGE tiebreak vs. baseline (SapBERT top-1), full population ===")
    print(f"{'threshold':>10} {'n':>6} {'resolved':>9} {'win':>5} {'loss':>5} {'neutral':>8}")
    for t in thresholds:
        c = results[t]["kge"]
        print(f"{t:>10} {c['n']:>6} {c['n_resolved']:>9} {c['win']:>5} {c['loss']:>5} {c['neutral']:>8}")

    print("\n=== Head-to-head: KGE vs. hardcoded _prefer_lab_procedure_over_observable() "
         "rule, ONLY on the subset where the rule applies ===")
    print(f"{'threshold':>10} {'n':>6} {'kge_win':>8} {'kge_loss':>9} "
         f"{'rule_win':>9} {'rule_loss':>10}")
    for t in thresholds:
        c = results[t]["hardcoded_subset"]
        print(f"{t:>10} {c['n']:>6} {c['kge_win']:>8} {c['kge_loss']:>9} "
             f"{c['rule_win']:>9} {c['rule_loss']:>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    thresholds = sorted(float(x) for x in args.thresholds.split(","))

    if not os.path.exists(args.model_path):
        raise SystemExit(
            f"No trained KGE checkpoint at {args.model_path}. Run "
            f"scripts/build_kg_embeddings.py first (it now writes this checkpoint via "
            f"src.kg_embedding.save_model() as of 2026-08-20) -- this harness deliberately "
            f"does not fall back to training a throwaway model, since the whole point is "
            f"to validate the SAME weights that would be wired into production.")

    from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing
    from scripts.score_gold_recall import load_gold
    from src.db_utils import connect_with_retry
    from src.kg_embedding import load_model
    from src.retrieval import VocabularyRetriever

    model, entity2idx, relation2idx = load_model(args.model_path)
    print(f"loaded checkpoint: {len(entity2idx)} entities, {len(relation2idx)} relations")

    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb",
                              read_only=True, max_wait_seconds=120)
    vocab = VocabularyRetriever(conn)

    note_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test=TRUE").fetchall()]
    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, note_ids)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    results = run_validation(conn, model, entity2idx, vocab, gold_by_note, thresholds)
    conn.close()

    _print_report(results, thresholds)


if __name__ == "__main__":
    main()
