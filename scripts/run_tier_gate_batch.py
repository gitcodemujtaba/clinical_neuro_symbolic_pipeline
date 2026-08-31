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
import random
import re
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


_WORD_RE = re.compile(r"[a-z]+")


def _is_precoordination_match(candidate_name: str, gold_fragment_texts: list) -> bool:
    """Heuristic pre-coordination check (2026-08-16, user proposal). True when
    the candidate's own concept name textually subsumes EVERY gold fragment's
    span text -- e.g. 'Fracture of clavicle' subsumes both 'clavicular' and
    'fracture', the two texts gold split into separate annotations.

    NOT an authoritative SNOMED compositional-semantics check. A rigorous
    version would walk athena_concept_relationship's ATTRIBUTE relationships
    (e.g. 'Finding site': 'Fracture of clavicle' -> 'Clavicle structure' is
    an attribute relationship, not an IS-A one, so the existing IS-A ancestor
    hierarchy this codebase's DuckDBHierarchy already uses elsewhere would
    not even catch this specific example). This is a cheap, transparent
    proxy instead: every gold fragment's significant words (len > 2,
    lowercased) must appear as a substring somewhere in the candidate name.
    Deliberately conservative in one direction (a real compositional match
    with very different wording would be missed) and permissive in another
    (a coincidental word overlap could pass) -- good enough to separate
    "the model union-mapped the compound phrase correctly" from "the model
    mapped it to something unrelated", not a certified ontology check.
    """
    name_lower = (candidate_name or "").lower()
    for frag_text in gold_fragment_texts:
        words = [w for w in _WORD_RE.findall(frag_text.lower()) if len(w) > 2]
        if words and not any(w in name_lower for w in words):
            return False
    return True


def grade(record, decision, gold_by_note, vocab):
    """Same crosswalk logic as scripts/experiment_3b_voting.py's grade():
    does the routed decision's chosen candidate's SNOMED code match an
    overlapping gold span? Returns (outcome, scoring_note):
      outcome: 'correct' / 'incorrect' / None (ungradable, e.g. HITL
        decisions with no chosen candidate, or no overlapping gold).
      scoring_note: None (clean 1:1 span match) / 'compound_span' /
        'compound_span_precoordinated' / 'narrower_than_gold'.

    2026-08-16 (found while diagnosing a batch of "incorrect" Tier 1 results
    that all turned out to be genuinely correct model judgments on inspection
    -- e.g. 'clavicular fracture' -> 'Fracture of clavicle', graded WRONG
    because gold annotates 'left clavicular' and 'fracture' as two SEPARATE
    spans/codes and a single-candidate entity cannot match both). This is
    scripts/score_gold_recall.py's own documented, pre-existing caveat
    ("COMPOUND-SPAN CASES ARE COUNTED SEPARATELY... the fix is Stage 2a
    splitting compound spans... not normalization tuning") -- that script's
    find_compound_spans() flags exactly this pattern; the same per-entity
    check is reimplemented here rather than importing it, since that
    function is batch-shaped (preds_by_note) and this script grades one
    record at a time.

    For a compound_span case specifically, _is_precoordination_match() (see
    its own docstring for the heuristic and its limits) checks whether the
    chosen candidate's name plausibly UNIONS the gold fragments' text -- if
    so, an otherwise-"incorrect" outcome (its code cannot literally equal
    either fragment's code) is promoted to "correct" with scoring_note
    'compound_span_precoordinated', so a model correctly recognizing
    "clavicular fracture" as one pre-coordinated concept is not penalized
    for gold's post-coordinated annotation choice. A real code match against
    either fragment (rare, but possible if Stage 2a's candidate list happens
    to carry one fragment's own code) is never overridden by this -- the
    heuristic only ever promotes incorrect-by-code-comparison outcomes, it
    never demotes a real code match.

    'narrower_than_gold' (e.g. entity span 'fixation' inside gold span
    'intermaxillary fixation') is a related but distinct caveat: not a
    compound mismatch, but the entity's own span is a strict substring of
    what gold annotated, so the candidate list it was ever shown may not
    have contained the more specific concept gold expects -- a Stage 2a
    span-boundary/Stage 2b retrieval question, not necessarily a Tier 1-5
    gating failure. All three are surfaced, not swept under a single
    'incorrect', so the reported precision distinguishes real gate errors
    from these known, separately-attributable upstream limitations.
    """
    gold = gold_by_note.get(record["note_id"], [])
    overlapping = [g for g in gold
                   if overlaps(record["orig_start"], record["orig_end"], g["start"], g["end"])]
    if not overlapping:
        return None, None
    gold_codes = {g["concept_id"] for g in overlapping}

    idx = decision.get("final_candidate_index")
    candidates = record.get("candidates") or []
    if idx is None:
        # HITL / unresolved decisions carry no chosen candidate -- not
        # gradable as correct/incorrect the way an AUTO decision is; the
        # interesting question for these is precision of the DECISION TO
        # ROUTE HITL, which is a separate (recall-of-ambiguity) analysis
        # this script does not attempt.
        return None, None
    if idx < 1 or idx > len(candidates):
        return None, None
    chosen = candidates[idx - 1]
    concept_id = chosen.get("omop_concept_id")
    code = vocab.snomed_code_for_concept(concept_id) if concept_id is not None else None
    if code is None:
        return None, None

    outcome = "correct" if code in gold_codes else "incorrect"

    scoring_note = None
    if len(overlapping) > 1:
        scoring_note = "compound_span"
        if outcome == "incorrect" and _is_precoordination_match(
                chosen.get("concept_name"), [g["span"] for g in overlapping]):
            outcome = "correct"
            scoring_note = "compound_span_precoordinated"
    else:
        g = overlapping[0]
        entity_len = record["orig_end"] - record["orig_start"]
        gold_len = g["end"] - g["start"]
        if entity_len < gold_len:
            scoring_note = "narrower_than_gold"

    return outcome, scoring_note


def select_atomic_entities(conn, note_ids, gold_by_note, target_n=36,
                           labels=("Condition", "Procedure", "Medication"), seed=1):
    """Curated 'atomic-only' entity selection (2026-08-16, user proposal), to
    get a clean precision measurement uncontaminated by the compound_span/
    narrower_than_gold confounds grade() otherwise has to flag after the
    fact. Pulls candidates across every already-processed note (not just the
    3 canonical gold notes), keeping only entities where:
      - gliner_label is a substantive clinical noun type (Condition/
        Procedure/Medication by default -- Qualifier is never in this set,
        so standalone modifier spans are excluded without needing to
        special-case them here; route_tier()'s own
        qualifier_fragment_precheck() would catch them anyway).
      - exactly ONE gold annotation overlaps this entity's span (an
        unambiguous mapping target, not a compound phrase).
      - the entity's own span is at least as long as that gold annotation's
        (not a narrower substring missing the specificity gold expects).
    Shuffled with a fixed seed for a reproducible-but-varied sample across
    notes/entity types rather than an accidentally note-ordered one.
    """
    pool = []
    for note_id in note_ids:
        for r in load_validation_records(conn, note_id, tier=None):
            if (r.get("gliner_label") or "") not in labels:
                continue
            gold = gold_by_note.get(r["note_id"], [])
            overlapping = [g for g in gold
                          if overlaps(r["orig_start"], r["orig_end"], g["start"], g["end"])]
            if len(overlapping) != 1:
                continue
            g = overlapping[0]
            if (r["orig_end"] - r["orig_start"]) < (g["end"] - g["start"]):
                continue
            pool.append(r)
    random.Random(seed).shuffle(pool)
    return pool[:target_n] if target_n else pool


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
    ap.add_argument("--curated-atomic", type=int, default=0, metavar="N",
                    help="2026-08-16 (user proposal): instead of --note-ids/--tier, pull "
                         "N entities via select_atomic_entities() -- unambiguous single-"
                         "gold-span Condition/Procedure/Medication entities across EVERY "
                         "already-processed note, not just the 3 canonical gold notes -- "
                         "for a precision measurement uncontaminated by the compound_span/ "
                         "narrower_than_gold confounds. 0 (default) uses --note-ids/--tier "
                         "as before.")
    args = ap.parse_args()

    conn = duckdb.connect(args.db, read_only=True)
    vocab = VocabularyRetriever(conn)

    if args.curated_atomic:
        # 2026-08-31 FIX: excludes the locked test split by default -- see
        # evaluation/splits.py; this script's default previously had no
        # such guard.
        from evaluation.splits import load_split
        all_note_ids_unfiltered = {r[0] for r in conn.execute(
            "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test = TRUE").fetchall()}
        all_note_ids = sorted(all_note_ids_unfiltered - load_split("test"))
        gold_path = _first_existing(GOLD_CANDIDATES, "gold")
        gold_rows = load_gold(gold_path, all_note_ids)
        gold_by_note = collections.defaultdict(list)
        for g in gold_rows:
            gold_by_note[g["note_id"]].append(g)
        records = select_atomic_entities(conn, all_note_ids, gold_by_note,
                                         target_n=args.curated_atomic)
        print(f"records to route: {len(records)} (curated atomic-only, "
              f"pooled across {len(all_note_ids)} processed notes)")
    else:
        note_ids = [n.strip() for n in args.note_ids.split(",")]
        tier_filter = None if args.tier.upper() == "ALL" else args.tier.upper()
        records = []
        for note_id in note_ids:
            records.extend(load_validation_records(conn, note_id, limit=args.limit_per_note,
                                                    tier=tier_filter))
        # route_tier() expects >=1 candidate to be worth evaluating (0
        # candidates is Tier 5's own "no_candidates" precheck, which needs no
        # filtering here -- kept in the sample deliberately so that precheck
        # path is exercised too, unlike experiment_3b_voting.py's
        # load_entities() which requires len(cands) >= 2.
        print(f"records to route: {len(records)} (tier={args.tier}, notes: {note_ids})")

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
        outcome, scoring_note = grade(record, decision, gold_by_note, vocab)
        results.append({"record": record, "decision": decision, "outcome": outcome,
                        "scoring_note": scoring_note})
        elapsed = time.time() - t0
        n_calls = sum(1 for m in (decision.get("models") or []) if m.get("verdict"))
        note_flag = f" [{scoring_note}]" if scoring_note else ""
        print(f"[{i}/{len(records)}] [{elapsed:.0f}s] "
              f"in_tier={record.get('confidence_tier_in')} {record['original_text']!r} "
              f"({len(record.get('candidates') or [])} cands) -> "
              f"tier={decision.get('tier')} routing={decision['mollm_routing_decision']} "
              f"reason={decision.get('queue_reason')} outcome={outcome}{note_flag} "
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
    print("PER-TIER PRECISION (against gold; 'clean' excludes compound_span/")
    print("narrower_than_gold cases -- see grade()'s docstring)")
    print("=" * 78)
    for tier in sorted(tier_counts):
        rows = [r for r in results if (r["decision"].get("tier") or "HITL_NO_TIER") == tier]
        graded = [r for r in rows if r["outcome"] in ("correct", "incorrect")]
        correct = sum(1 for r in graded if r["outcome"] == "correct")
        prec = f"{correct / len(graded) * 100:.1f}%" if graded else "n/a"
        clean = [r for r in graded if r["scoring_note"] is None]
        clean_correct = sum(1 for r in clean if r["outcome"] == "correct")
        clean_prec = f"{clean_correct / len(clean) * 100:.1f}%" if clean else "n/a"
        print(f"  {tier}: {len(rows)} total, {len(graded)} gradable, "
              f"{correct} correct -- precision {prec}  "
              f"(clean-span only: {clean_correct}/{len(clean)} -- {clean_prec})")

    note_counts = collections.Counter(r["scoring_note"] for r in results if r["scoring_note"])
    if note_counts:
        print(f"\nscoring caveats among gradable decisions: {dict(note_counts)}")

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
                    "outcome": r["outcome"], "scoring_note": r["scoring_note"],
                    "models": r["decision"].get("models")}
                   for r in results], f, indent=2, default=str)
    print(f"\nfull results written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
