"""evaluation/guideline_evidence_ab_test.py -- 2026-08-30 (plan: "Guideline
Evidence Integration + Gold-Graded A/B Validation", see
/home/ec2-user/.claude/plans/here-is-details-of-peppy-pixel.md).

Real, gold-graded A/B test of src.guideline_evidence's GUIDELINE_EVIDENCE_ENABLED
flag: does injecting curated clinical-guideline rules into the tiebreak prompt
(src.mollm_tier_gate._tiebreak_prompt(), via _guideline_evidence_block()) change
whether the ensemble's plurality/tiebreak-resolved candidate is actually correct
against gold?

WHY THIS NEEDS FRESH LLM CALLS, UNLIKE evaluation/kg3_feature_ablation.py OR
evaluation/kg3_calibrator_previous_regrade.py. Those two replay ALREADY-STORED
mollm_tier_gate_decisions.models JSON -- the calibrator is a pure post-hoc
scoring step over votes that already exist. Guideline evidence changes the
PROMPT TEXT sent to the 3 ensemble models -- there is nothing to replay against
a counterfactual "what if the prompt had been different" question. Each entity
in the population must be run live, twice (flag off, flag on), through
route_tier() -> run_two_step_ensemble().

DELIBERATELY NO CALIBRATOR IN THIS TEST (calibrator=None, conn=None,
kg3_driver=None passed to route_tier()). Two reasons: (1) it isolates exactly
what's being tested -- the ensemble/tiebreak's own candidate choice, matching
evaluation/grade_overnight_corpus_run.py's existing "Tier 4 shadow precision"
methodology (grade the plurality-resolved candidate even though it's routed to
HITL); (2) it avoids a real cross-arm contamination risk -- if the "off" arm's
decisions were persisted and then the "on" arm's calibrator consulted
prior_confirmation_count/kg3_confirmation_count, the on-arm could see evidence
seeded by the off-arm's own run, which would confound the comparison this test
exists to make.

POPULATION: entities currently at TIER_4_ENSEMBLE_SPLIT or TIER_2_AUTO_RESOLVED
(the only two tiers whose prompt reaches _tiebreak_prompt() at all) whose
candidate list has >=1 real guideline-corpus name match, via
guideline_evidence_for_candidates() itself (not a re-derived matching rule --
this guarantees the selected population is exactly what production code would
actually inject evidence for). Real, measured count as of 2026-08-30: 258
entities, 88 distinct mention texts, corpus-wide.

ENTITY RECONSTRUCTION: reuses src.mollm_ensemble.load_validation_records()
directly -- the exact same function scripts/run_stage3_tier_gate.py already
uses to build route_tier()'s input, read fresh from extracted_entities/
normalized_entities (NOT from the stored tier-gate decision, which may predate
the current code version or candidate list).

GRADING: reuses evaluation/grade_overnight_corpus_run.py's grade_population()/
plurality_candidate_index()/KNOWN_GOLD_ERRORS unmodified -- same clean-span +
SNOMED-crosswalk methodology as every other precision figure in this project.

STORAGE: both arms' decisions are persisted via store_tier_decision(...,
is_test=True) for audit, but grading itself works directly off the in-memory
decision dicts returned by route_tier() -- no DB read-back required, avoiding
any risk of a stored-but-uncommitted-connection race.
"""
import argparse
import collections
import json
import os
import sys
import time

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.retrieval import GuidelineIndex, VocabularyRetriever  # noqa: E402
from src.guideline_evidence import (  # noqa: E402
    guideline_evidence_for_candidates, _type_compatible)
from src.mollm_tier_gate import (  # noqa: E402
    TIER_4_ENSEMBLE_SPLIT, TIER_2_AUTO_RESOLVED, route_tier, store_tier_decision,
    build_clients)
from src.mollm_ensemble import load_validation_records  # noqa: E402
from evaluation.grade_overnight_corpus_run import (  # noqa: E402
    grade_population, plurality_candidate_index, KNOWN_GOLD_ERRORS)
from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from scripts.score_gold_recall import load_gold  # noqa: E402

DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def select_population(conn, guideline_index, limit=None):
    """(note_id, entity_id) pairs currently at TIER_4_ENSEMBLE_SPLIT or
    TIER_2_AUTO_RESOLVED whose candidate list gets a real guideline-evidence
    hit -- reusing guideline_evidence_for_candidates() itself so the selected
    population is exactly what production code would inject evidence for,
    not a re-derived approximation of it.
    """
    rows = conn.execute("""
        SELECT d.note_id, d.entity_id, n.candidates
        FROM mollm_tier_gate_decisions d
        JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.tier IN (?, ?)
    """, [TIER_4_ENSEMBLE_SPLIT, TIER_2_AUTO_RESOLVED]).fetchall()

    hits = []
    for note_id, entity_id, cands_json in rows:
        candidates = cands_json
        if isinstance(candidates, str):
            candidates = json.loads(candidates)
        if not candidates:
            continue
        # Shape guideline_evidence_for_candidates() expects: a list of
        # {"index": i, "candidate": {...}} dicts, matching _tiebreak_prompt()'s
        # own `accepted` list. Every candidate is checked here (a superset of
        # what any single model's own `accepted` subset would be) -- fine for
        # POPULATION SELECTION (a conservative "could this candidate list ever
        # get evidence" check), not used for grading.
        accepted_shape = [{"index": i + 1, "candidate": c} for i, c in enumerate(candidates)]
        if guideline_evidence_for_candidates(guideline_index, accepted_shape):
            hits.append((note_id, entity_id))
    if limit:
        hits = hits[:limit]
    return hits


def _reconstruct(rec):
    """load_validation_records() row -> the entity dict route_tier() expects,
    plus the extra fields grade_population() needs (entity_label, not
    gliner_label; orig_start/orig_end already match)."""
    entity = dict(rec)
    entity["entity_label"] = rec.get("gliner_label")
    return entity


def run_arm(conn, population_by_note, clients, flag_on):
    if flag_on:
        os.environ["CNSP_GUIDELINE_EVIDENCE"] = "1"
    else:
        os.environ.pop("CNSP_GUIDELINE_EVIDENCE", None)

    results = []
    for note_id, entity_ids in population_by_note.items():
        records = load_validation_records(conn, note_id, tier=None)
        by_id = {r["entity_id"]: r for r in records}
        for entity_id in entity_ids:
            rec = by_id.get(entity_id)
            if rec is None:
                log(f"  SKIP {entity_id} ({note_id}): not found in current "
                    f"load_validation_records() (re-normalized since? superseded?)")
                continue
            entity = _reconstruct(rec)
            decision = route_tier(entity, clients=clients)  # model_results=None -> forces a live call
            row = dict(rec)
            row["entity_label"] = rec.get("gliner_label")
            row["tier"] = decision.get("tier")
            row["final_candidate_index"] = decision.get("final_candidate_index")
            row["models"] = decision.get("models")
            row["routing_basis"] = decision.get("routing_basis")
            results.append(row)

            store_tier_decision(decision, entity_id, note_id, conn, is_test=True)
    return results


def _idx_fn(d):
    if d.get("final_candidate_index"):
        return d["final_candidate_index"], {}
    idx, top_verdict, vote_counts = plurality_candidate_index(d["models"])
    return idx, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the population (dry-run mode, e.g. --limit 8)")
    args = ap.parse_args()

    # 2026-08-30: a SINGLE read-write connection for the whole script, not a
    # read-only one used for reads plus separate read-write connections
    # opened per store_tier_decision() call. Verified live: mixing a
    # long-lived read-only connection with per-operation read-write
    # connections in the same process raises duckdb.ConnectionException
    # ("different configuration than existing connections") -- the exact
    # bug ui/components/db_status.py's render_mixed_connection_status()
    # already documents for the Streamlit UI, hit here too. This script
    # both reads (load_validation_records) and writes (store_tier_decision)
    # throughout, so one read-write connection for the whole run is the
    # correct fix, not a workaround.
    conn = connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=120)
    idx = GuidelineIndex()
    log(f"guideline corpus: {idx.stats}")

    pop = select_population(conn, idx, limit=args.limit)
    log(f"population: {len(pop)} entities "
        f"({'DRY RUN, --limit applied' if args.limit else 'FULL'})")

    population_by_note = collections.defaultdict(list)
    for note_id, entity_id in pop:
        population_by_note[note_id].append(entity_id)
    log(f"spanning {len(population_by_note)} notes")

    all_notes = sorted(population_by_note.keys())
    gold_rows = load_gold(_first_existing(GOLD_CANDIDATES, "gold"), all_notes)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    vocab = VocabularyRetriever(conn)
    clients = build_clients()

    log("=== OFF arm (CNSP_GUIDELINE_EVIDENCE unset) ===")
    t0 = time.time()
    off_results = run_arm(conn, population_by_note, clients, flag_on=False)
    log(f"off arm: {len(off_results)} entities, {(time.time()-t0)/60:.1f} min")

    log("=== ON arm (CNSP_GUIDELINE_EVIDENCE=1) ===")
    t0 = time.time()
    on_results = run_arm(conn, population_by_note, clients, flag_on=True)
    log(f"on arm: {len(on_results)} entities, {(time.time()-t0)/60:.1f} min")

    os.environ.pop("CNSP_GUIDELINE_EVIDENCE", None)  # leave the env clean

    _, off_clean, off_skip = grade_population(off_results, gold_by_note, vocab, _idx_fn)
    _, on_clean, on_skip = grade_population(on_results, gold_by_note, vocab, _idx_fn)

    off_correct = sum(1 for r in off_clean if r["correct"])
    on_correct = sum(1 for r in on_clean if r["correct"])

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"OFF: gradable={len(off_clean)}  correct={off_correct}  "
          f"precision={off_correct/len(off_clean)*100:.1f}%" if off_clean else "OFF: no gradable")
    print(f"ON:  gradable={len(on_clean)}  correct={on_correct}  "
          f"precision={on_correct/len(on_clean)*100:.1f}%" if on_clean else "ON: no gradable")
    print(f"OFF skip breakdown: {dict(off_skip)}")
    print(f"ON  skip breakdown: {dict(on_skip)}")

    off_by_key = {(r["note_id"], r["text"]): r for r in off_clean}
    on_by_key = {(r["note_id"], r["text"]): r for r in on_clean}
    print("\n--- per-entity flips (off vs on disagree on correctness) ---")
    n_flip = 0
    for key, r_off in off_by_key.items():
        r_on = on_by_key.get(key)
        if r_on is None:
            continue
        if r_off["correct"] != r_on["correct"]:
            n_flip += 1
            direction = "OFF wrong -> ON right" if r_on["correct"] else "OFF right -> ON wrong"
            print(f"  [{key[0]}] {key[1]!r}  {direction}  "
                  f"off_pred={r_off['pred_concept_name']}  on_pred={r_on['pred_concept_name']}  "
                  f"gold={r_off['gold_snomed']}")
    if n_flip == 0:
        print("  (none -- identical correctness on every paired entity)")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
