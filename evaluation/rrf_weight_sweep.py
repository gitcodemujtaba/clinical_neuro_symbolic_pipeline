"""
evaluation/rrf_weight_sweep.py — Phase 3 follow-up: grid search over
src.normalization.tier_retrieval's RRF_WEIGHT_DENSE/RRF_WEIGHT_SPARSE,
charting Top-1 vs. Top-5-oracle accuracy per weight setting, per the Phase 3
findings doc's explicit "next step, not attempted this session" (now being
attempted).

WHY THIS IS FAST DESPITE SWEEPING MANY WEIGHT COMBINATIONS. RRF fusion is
pure arithmetic over two already-fetched RANKED LISTS (dense cosine order,
BM25 order) -- it does not require a new DB query per weight combination.
This script fetches each entity's dense and sparse rankings ONCE (the
expensive part -- one SapBERT embedding + one BM25 query per entity), then
re-fuses them in-memory under every weight combination in the grid. A naive
approach (re-running scripts/stage2b_hybrid_ab.py once per weight) would
re-pay the DB query cost for every grid point; this pays it once.

SCOPE: only entities that reach Tier 3 under dense-only retrieval matter
here (Tier 1/2 exact/synonym hits are identical regardless of RRF weights).
Reuses the same 300-entity population evaluation/stage2b_hybrid_ab.py
already validated Phase 3 against, for direct comparability.

Run:  python3 evaluation/rrf_weight_sweep.py --note-ids <32 notes> --limit 300
"""
import argparse
import collections
import itertools
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
from src.normalization.tier_retrieval import (  # noqa: E402
    RRF_K, RRF_POOL_SIZE, _rrf_scores, _tier3_semantic_rows, get_sapbert_embedding,
)
from src.normalization.bm25_index import query_bm25  # noqa: E402
from src.normalization.constants import VOCAB_BY_LABEL, DEFAULT_VOCAB, GLINER_LABEL_TO_DOMAIN  # noqa: E402
from src.mollm_ensemble import load_validation_records  # noqa: E402


def fetch_rankings(conn, entity):
    """One dense + one sparse ranking per entity, fetched once. Returns
    (dense_rows, sparse_rows) in the same 5-tuple/(id,name,domain,vocab,score)
    shape _tier3_hybrid_rows() itself consumes, or None if this entity
    wouldn't even reach Tier 3 (no text) -- caller skips those.
    """
    text = entity["expanded_text"]
    label = entity["gliner_label"]
    vocabs = VOCAB_BY_LABEL.get(label, DEFAULT_VOCAB)
    domains = GLINER_LABEL_TO_DOMAIN.get(label)
    vector = get_sapbert_embedding(text)
    dense_rows = _tier3_semantic_rows(conn, vector, vocabs, domains, limit=RRF_POOL_SIZE)
    sparse_rows = query_bm25(conn, text, vocabs=vocabs, domains=domains, limit=RRF_POOL_SIZE)
    return dense_rows, sparse_rows


def fused_top1(dense_rows, sparse_rows, w_dense, w_sparse):
    """Top-ranked concept_id under RRF fusion with the given weights
    (w_prior is always 0 -- Phase 4's Empirical Prior Matrix doesn't exist
    yet, same as production _tier3_hybrid_rows()). Returns None if both
    rankings are empty.
    """
    dense_rrf = _rrf_scores([r[0] for r in dense_rows])
    sparse_rrf = _rrf_scores([r[0] for r in sparse_rows])
    all_ids = set(dense_rrf) | set(sparse_rrf)
    if not all_ids:
        return None, []
    scored = [(cid, w_dense * dense_rrf.get(cid, 0.0) + w_sparse * sparse_rrf.get(cid, 0.0))
             for cid in all_ids]
    scored.sort(key=lambda t: t[1], reverse=True)
    ranked_ids = [cid for cid, _ in scored]
    return ranked_ids[0], ranked_ids[:5]


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    note_ids = [n.strip() for n in args.note_ids.split(",")]
    conn = duckdb.connect(args.db, read_only=True)
    vocab = VocabularyRetriever(conn)

    entities = []
    for note_id in note_ids:
        entities.extend(load_validation_records(conn, note_id, tier=None))
    if args.limit:
        entities = entities[:args.limit]

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, note_ids)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    print(f"entities: {len(entities)}  notes: {len(note_ids)}")
    print("fetching dense+sparse rankings once per entity (this is the slow part)...")

    t0 = time.time()
    fetched = []  # (entity, dense_rows, sparse_rows, gold_codes) for gradable Tier-3-eligible entities
    dense_only_tier3_count = 0
    for i, e in enumerate(entities, 1):
        gold = gold_by_note.get(e["note_id"], [])
        overlapping = [g for g in gold
                      if overlaps(e["orig_start"], e["orig_end"], g["start"], g["end"])]
        if len(overlapping) != 1:
            continue  # skip compound/no-gold cases for this sweep -- clean-span only
        g0 = overlapping[0]
        if (e["orig_end"] - e["orig_start"]) < (g0["end"] - g0["start"]):
            continue  # skip narrower-than-gold too
        dense_rows, sparse_rows = fetch_rankings(conn, e)
        if not dense_rows and not sparse_rows:
            continue
        dense_only_tier3_count += 1
        gold_codes = {g["concept_id"] for g in overlapping}
        fetched.append((e, dense_rows, sparse_rows, gold_codes))
        if i % 50 == 0:
            print(f"  [{i}/{len(entities)}] [{time.time()-t0:.0f}s] "
                 f"{len(fetched)} clean-span Tier-3-eligible so far")

    print(f"\nfetched rankings for {len(fetched)} clean-span, Tier-3-eligible entities "
         f"in {time.time()-t0:.0f}s")

    def grade(cid):
        if cid is None:
            return None
        code = vocab.snomed_code_for_concept(cid)
        return code is not None, code

    grid = [
        (1.0, 0.0), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5),
        (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.0, 1.0),
    ]
    print("\n" + "=" * 78)
    print(f"{'w_dense':>8} {'w_sparse':>9} {'top1_acc':>9} {'oracle_acc':>11} {'n_gradable':>11}")
    print("=" * 78)
    results = []
    for w_dense, w_sparse in grid:
        top1_correct, oracle_correct, n_gradable = 0, 0, 0
        for e, dense_rows, sparse_rows, gold_codes in fetched:
            top1_id, top5_ids = fused_top1(dense_rows, sparse_rows, w_dense, w_sparse)
            top1_code = vocab.snomed_code_for_concept(top1_id) if top1_id else None
            top5_codes = {vocab.snomed_code_for_concept(cid) for cid in top5_ids}
            top5_codes.discard(None)
            if top1_code is None and not top5_codes:
                continue
            n_gradable += 1
            if top1_code and top1_code in gold_codes:
                top1_correct += 1
            if top5_codes & gold_codes:
                oracle_correct += 1
        top1_acc = top1_correct / n_gradable * 100 if n_gradable else 0.0
        oracle_acc = oracle_correct / n_gradable * 100 if n_gradable else 0.0
        results.append({"w_dense": w_dense, "w_sparse": w_sparse, "top1_acc": top1_acc,
                        "oracle_acc": oracle_acc, "n_gradable": n_gradable})
        print(f"{w_dense:>8.1f} {w_sparse:>9.1f} {top1_acc:>8.1f}% {oracle_acc:>10.1f}% "
             f"{n_gradable:>11}")

    dense_only_row = next(r for r in results if r["w_dense"] == 1.0)
    print(f"\ndense-only (w_dense=1.0, w_sparse=0.0) baseline: "
         f"top1={dense_only_row['top1_acc']:.1f}%  oracle={dense_only_row['oracle_acc']:.1f}%")
    best_top1 = max(results, key=lambda r: r["top1_acc"])
    print(f"best top1 in grid: w_dense={best_top1['w_dense']} w_sparse={best_top1['w_sparse']} "
         f"-> top1={best_top1['top1_acc']:.1f}% oracle={best_top1['oracle_acc']:.1f}%")

    out_path = os.path.join(PROJECT_DIR, "reports", "rrf_weight_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nfull results written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
