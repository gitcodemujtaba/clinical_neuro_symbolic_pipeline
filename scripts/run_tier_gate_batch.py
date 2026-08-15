"""
scripts/run_tier_gate_batch.py — Phase 2 validation: measure src/mollm_tier_gate.py's
Tier 1-5 gate against gold, the way docs/2026-08-15_Stage4_Stage5_Build.md
measured the OLD binary route() (52.6% AUTO_VALIDATED precision, the baseline
this module exists to beat).

WHAT THIS DOES. Loads Stage 2b LOW-tier records for the given notes (same
loader production Stage 3 uses, src.mollm_ensemble.load_validation_records()),
runs each through src.mollm_tier_gate.route_tier() (the real two-step CoT
ensemble, not a mock), grades the resulting tier's chosen candidate against
gold the same way scripts/experiment_3b_voting.py's grade() does (SNOMED
crosswalk via VocabularyRetriever, overlap-based span matching), and reports
per-tier precision plus the overall tier distribution against the spec's
~70/15/5/5/5 target.

READ-ONLY. Connects to DuckDB read_only=True and never calls
src.kg3_ingestion.ingest_auto_decision() -- this script measures whether the
gate SHOULD be trusted to write, it does not write. See that module's own
dry_run=True default for the next step once a batch here looks good.

Run:  python3 scripts/run_tier_gate_batch.py --note-ids 10000032-DS-21 --limit-per-note 10
      python3 scripts/run_tier_gate_batch.py --note-ids 17751158-DS-19,19442119-DS-15,14490470-DS-11
"""
import argparse
import collections
import json
import os
import sys
import time

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
sys.path.insert(0, PROJECT_DIR)

from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from scripts.score_gold_recall import load_gold, overlaps  # noqa: E402
from src.mollm_ensemble import load_validation_records  # noqa: E402
from src.mollm_tier_gate import AUTO_TIERS, build_clients, route_tier  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402


def grade(record, decision, gold_by_note, vocab):
    """Same crosswalk logic as scripts/experiment_3b_voting.py's grade():
    does the routed decision's chosen candidate's SNOMED code match an
    overlapping gold span? Returns 'correct' / 'incorrect' / None (ungradable,
    e.g. HITL decisions with no chosen candidate, or no overlapping gold)."""
    gold = gold_by_note.get(record["note_id"], [])
    overlapping = [g for g in gold
                   if overlaps(record["orig_start"], record["orig_end"], g["start"], g["end"])]
    if not overlapping:
        return None
    gold_codes = {g["concept_id"] for g in overlapping}

    idx = decision.get("final_candidate_index")
    candidates = record.get("candidates") or []
    if idx is None:
        # HITL / unresolved decisions carry no chosen candidate -- not
        # gradable as correct/incorrect the way an AUTO decision is; the
        # interesting question for these is precision of the DECISION TO
        # ROUTE HITL, which is a separate (recall-of-ambiguity) analysis
        # this script does not attempt.
        return None
    if idx < 1 or idx > len(candidates):
        return None
    concept_id = candidates[idx - 1].get("omop_concept_id")
    code = vocab.snomed_code_for_concept(concept_id) if concept_id is not None else None
    if code is None:
        return None
    return "correct" if code in gold_codes else "incorrect"


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default="10000032-DS-21", help="comma-separated note_ids")
    ap.add_argument("--limit-per-note", type=int, default=None)
    ap.add_argument("--tier", default="LOW",
                    help="confidence_tier_in filter passed to load_validation_records "
                         "(HIGH/MEDIUM/LOW), or 'ALL' to load every tier -- 2026-08-15 "
                         "follow-up: production Stage 3 (and this script's own default) "
                         "has only ever tested LOW, which is Stage 2b's own hardest, "
                         "pre-filtered-toward-disagreement subset. The spec's 70/15/5/5/5 "
                         "target almost certainly describes the FULL entity population, "
                         "where HIGH/MEDIUM-tier confirmations make up most of Tier 1's "
                         "~70%% -- see docs/2026-08-15_Phase2_TierGate_Validation.md.")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    note_ids = [n.strip() for n in args.note_ids.split(",")]
    conn = duckdb.connect(args.db, read_only=True)
    vocab = VocabularyRetriever(conn)

    tier_filter = None if args.tier.upper() == "ALL" else args.tier.upper()
    records = []
    for note_id in note_ids:
        records.extend(load_validation_records(conn, note_id, limit=args.limit_per_note,
                                                tier=tier_filter))
    # route_tier() expects >=1 candidate to be worth evaluating (0 candidates
    # is Tier 5's own "no_candidates" precheck, which needs no filtering
    # here -- kept in the sample deliberately so that precheck path is
    # exercised too, unlike experiment_3b_voting.py's load_entities() which
    # requires len(cands) >= 2.
    print(f"records to route: {len(records)} (LOW-tier, notes: {note_ids})")

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, note_ids)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    clients = build_clients()
    print(f"models: {list(clients.keys())}\n")

    results = []
    t0 = time.time()
    for i, record in enumerate(records, 1):
        decision = route_tier(record, clients=clients)
        outcome = grade(record, decision, gold_by_note, vocab)
        results.append({"record": record, "decision": decision, "outcome": outcome})
        elapsed = time.time() - t0
        n_calls = sum(1 for m in (decision.get("models") or []) if m.get("verdict"))
        print(f"[{i}/{len(records)}] [{elapsed:.0f}s] "
              f"in_tier={record.get('confidence_tier_in')} {record['original_text']!r} "
              f"({len(record.get('candidates') or [])} cands) -> "
              f"tier={decision.get('tier')} routing={decision['mollm_routing_decision']} "
              f"reason={decision.get('queue_reason')} outcome={outcome} "
              f"model_calls={n_calls}")

    print()
    print("=" * 78)
    print("TIER DISTRIBUTION (target ~70/15/5/5/5 across Tier 1/2/3/4/5)")
    print("=" * 78)
    tier_counts = collections.Counter(r["decision"].get("tier") or "HITL_NO_TIER" for r in results)
    for tier, n in sorted(tier_counts.items()):
        print(f"  {tier}: {n} ({n / len(results) * 100:.1f}%)")

    print()
    print("=" * 78)
    print("PER-TIER PRECISION (against gold, gradable decisions only)")
    print("=" * 78)
    for tier in sorted(tier_counts):
        rows = [r for r in results if (r["decision"].get("tier") or "HITL_NO_TIER") == tier]
        graded = [r for r in rows if r["outcome"] in ("correct", "incorrect")]
        correct = sum(1 for r in graded if r["outcome"] == "correct")
        prec = f"{correct / len(graded) * 100:.1f}%" if graded else "n/a"
        print(f"  {tier}: {len(rows)} total, {len(graded)} gradable, "
              f"{correct} correct -- precision {prec}")

    auto_count = sum(1 for r in results if (r["decision"].get("tier") in AUTO_TIERS))
    print(f"\nAUTO coverage (Tier 1+2+3, fraction skipping human review): "
          f"{auto_count / len(results) * 100:.1f}% (target ~90%)")

    print()
    print("=" * 78)
    print("AUTO COVERAGE BY INPUT (Stage 2b) CONFIDENCE TIER")
    print("=" * 78)
    in_tier_counts = collections.Counter(r["record"].get("confidence_tier_in") for r in results)
    for in_tier in sorted(in_tier_counts):
        rows = [r for r in results if r["record"].get("confidence_tier_in") == in_tier]
        auto = sum(1 for r in rows if r["decision"].get("tier") in AUTO_TIERS)
        print(f"  {in_tier}: {auto}/{len(rows)} auto ({auto / len(rows) * 100:.1f}%)")

    reasons = collections.Counter(r["decision"].get("queue_reason") for r in results
                                  if r["decision"].get("queue_reason"))
    print(f"\nHITL queue_reason breakdown: {dict(reasons)}")

    out_path = os.path.join(PROJECT_DIR, "reports", "tier_gate_batch_results.json")
    with open(out_path, "w") as f:
        json.dump([{"note_id": r["record"]["note_id"], "text": r["record"]["original_text"],
                    "confidence_tier_in": r["record"].get("confidence_tier_in"),
                    "tier": r["decision"].get("tier"),
                    "routing": r["decision"]["mollm_routing_decision"],
                    "queue_reason": r["decision"].get("queue_reason"),
                    "composite_confidence": r["decision"].get("composite_confidence"),
                    "final_candidate_index": r["decision"].get("final_candidate_index"),
                    "outcome": r["outcome"], "models": r["decision"].get("models")}
                   for r in results], f, indent=2, default=str)
    print(f"\nfull results written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
