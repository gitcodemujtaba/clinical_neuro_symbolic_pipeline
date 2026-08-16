"""
evaluation/stage2b_hybrid_ab.py — Phase 3 validation: A/B oracle-accuracy
measurement, dense-only Tier 3 (src.normalization.orchestrator.normalize_entity
with HYBRID_RETRIEVAL_ENABLED=False) vs. hybrid BM25+SapBERT+prior RRF
(CNSP_HYBRID_RETRIEVAL=1), against the SAME already-extracted entities.

WHY A FRESH normalize_entity() CALL, NOT THE STORED normalized_entities.candidates.
Every row already in the DB was written under the OLD dense-only path
(CANDIDATE_LIMIT=3, no hybrid code path even existed). Comparing dense vs.
hybrid needs BOTH arms computed under today's code, at today's
CANDIDATE_LIMIT=5, on the same entities -- so this script re-runs Stage 2b's
normalization step only (not Stage 2a extraction, which is untouched by this
phase and expensive to redo) directly against already-extracted_entities rows.

WHY TWO SEPARATE PROCESSES, NOT A RUNTIME TOGGLE. HYBRID_RETRIEVAL_ENABLED is
read from os.environ ONCE at src.normalization.tier_retrieval's import time
(same pattern RANKED_TIER12/TIER12_RANK_SEMANTIC already use) -- flipping
os.environ mid-process after that module is imported has no effect. Run this
script twice, CNSP_HYBRID_RETRIEVAL unset then =1, each writing its own JSON
report, and diff the two reports' oracle_accuracy.

Run:  python3 evaluation/stage2b_hybrid_ab.py --note-ids 17751158-DS-19,19442119-DS-15,14490470-DS-11 --out reports/stage2b_ab_dense.json
      CNSP_HYBRID_RETRIEVAL=1 python3 evaluation/stage2b_hybrid_ab.py --note-ids 17751158-DS-19,19442119-DS-15,14490470-DS-11 --out reports/stage2b_ab_hybrid.json
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
from src.retrieval import VocabularyRetriever  # noqa: E402
import src.normalization as N  # noqa: E402


def load_entities_for_ab(conn, note_ids):
    """Already-extracted, non-superseded entities for the given notes --
    the SAME population src.mollm_ensemble.load_validation_records() reads
    from, minus the tier/limit filters this script doesn't need."""
    rows = conn.execute(f"""
        SELECT note_id, original_text, expanded_text, entity_label,
               orig_start, orig_end
        FROM extracted_entities
        WHERE note_id IN ({",".join("?" * len(note_ids))})
          AND (superseded_by_split IS NULL OR superseded_by_split = FALSE)
          AND (superseded_by_growth IS NULL OR superseded_by_growth = FALSE)
    """, note_ids).fetchall()
    return [{"note_id": r[0], "original_text": r[1], "expanded_text": r[2] or r[1],
            "entity_label": r[3], "orig_start": r[4], "orig_end": r[5]}
            for r in rows]


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", required=True, help="comma-separated note_ids")
    ap.add_argument("--limit", type=int, default=None, help="cap total entities processed")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    note_ids = [n.strip() for n in args.note_ids.split(",")]
    conn = duckdb.connect(args.db, read_only=True)
    vocab = VocabularyRetriever(conn)

    entities = load_entities_for_ab(conn, note_ids)
    if args.limit:
        entities = entities[:args.limit]

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, note_ids)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    mode = "hybrid" if N.HYBRID_RETRIEVAL_ENABLED else "dense"
    print(f"mode: {mode}  entities: {len(entities)}  notes: {note_ids}")

    results = []
    t0 = time.time()
    for i, e in enumerate(entities, 1):
        gold = gold_by_note.get(e["note_id"], [])
        overlapping = [g for g in gold
                      if overlaps(e["orig_start"], e["orig_end"], g["start"], g["end"])]
        if not overlapping:
            continue  # not gradable -- no gold span to check candidates against
        gold_codes = {g["concept_id"] for g in overlapping}

        result = N.normalize_entity(e["expanded_text"], conn, gliner_label=e["entity_label"])
        candidates = result.get("candidates") or []
        cand_codes = []
        for c in candidates:
            code = vocab.snomed_code_for_concept(c.get("omop_concept_id"))
            if code:
                cand_codes.append(code)

        top1_correct = bool(cand_codes) and cand_codes[0] in gold_codes
        oracle_correct = any(code in gold_codes for code in cand_codes)
        results.append({
            "note_id": e["note_id"], "text": e["original_text"],
            "entity_label": e["entity_label"], "n_candidates": len(candidates),
            "match_tier": candidates[0].get("match_tier") if candidates else None,
            "top1_correct": top1_correct, "oracle_correct": oracle_correct,
            "gold_rank": (cand_codes.index(next(c for c in cand_codes if c in gold_codes)) + 1
                         if oracle_correct else None),
        })
        if i % 20 == 0 or i == len(entities):
            print(f"  [{i}/{len(entities)}] [{time.time()-t0:.0f}s]")

    n = len(results)
    top1 = sum(1 for r in results if r["top1_correct"])
    oracle = sum(1 for r in results if r["oracle_correct"])
    tier3_results = [r for r in results if r["match_tier"] in ("3 (Semantic)", "3 (Hybrid)")]
    tier3_top1 = sum(1 for r in tier3_results if r["top1_correct"])
    tier3_oracle = sum(1 for r in tier3_results if r["oracle_correct"])

    print()
    print(f"mode={mode}  n={n} gradable entities")
    print(f"  overall top1_accuracy={top1/n*100:.1f}%  oracle_accuracy={oracle/n*100:.1f}%")
    if tier3_results:
        print(f"  Tier-3-only ({len(tier3_results)} entities): "
              f"top1_accuracy={tier3_top1/len(tier3_results)*100:.1f}%  "
              f"oracle_accuracy={tier3_oracle/len(tier3_results)*100:.1f}%")
    rank_dist = collections.Counter(r["gold_rank"] for r in results if r["gold_rank"])
    print(f"  gold-correct-candidate rank distribution: {dict(sorted(rank_dist.items()))}")

    report = {
        "mode": mode, "n": n, "top1_accuracy": top1 / n if n else None,
        "oracle_accuracy": oracle / n if n else None,
        "tier3_n": len(tier3_results),
        "tier3_top1_accuracy": tier3_top1 / len(tier3_results) if tier3_results else None,
        "tier3_oracle_accuracy": tier3_oracle / len(tier3_results) if tier3_results else None,
        "rank_distribution": dict(rank_dist), "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nreport written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
