"""scripts/build_kg_embeddings.py -- 2026-08-20: trains and evaluates the
TransE KG embedding pipeline (src.kg_embedding) on the real SNOMED
subgraph this pipeline's own candidate pools touch, then evaluates it
two ways: standard KGE link-prediction (MRR/Hits@10 on held-out real
SNOMED triples) and an extrinsic check tied directly to this project's
own graded true-positive decisions (does the embedding space actually
separate a competing-but-wrong candidate from an unrelated concept more
than chance -- the real question for whether this would help Stage 2b).

Read-only DB access throughout -- safe to run alongside another script
holding the write lock (e.g. a concurrent Stage 3 batch), since this
never opens a write connection.

Run: python3 scripts/build_kg_embeddings.py
"""
import collections
import json
import random
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.kg_embedding import (  # noqa: E402
    build_vocab, evaluate_against_tp_records, evaluate_link_prediction,
    load_snomed_subgraph, save_model, train_transe)

MODEL_CHECKPOINT_PATH = f"{PROJECT_DIR}/models/kg_transe_v1.pt"


def gather_tp_records(conn):
    """One record per gold-graded, tier-gate-correct entity with >=2 real
    candidates -- the wrong candidate(s) are genuine alternatives the
    correct concept had to beat, not synthetic. Reuses this session's
    standard clean-span grading methodology throughout.

    2026-08-31 FIX: previously scoped to every `is_test=TRUE` note -- a DB
    flag meaning only "processed by the pipeline", unrelated to
    `data/splits/note_splits.csv`'s OFFICIAL locked test split (70 notes
    reserved for the T0/T1/T2 benchmark). Confirmed live: 39 of the 149
    is_test=TRUE notes (26%) were from that locked split, meaning this
    function's output -- consumed as RotatE's `gold`/`combined` TRAINING
    data (src/kg_embedding_rotate.py) and as the extrinsic-eval population
    for BOTH TransE and RotatE -- was partly built from gold annotations
    the proposal reserves for final evaluation only. Now excludes the
    locked split via evaluation.splits.load_split(), same discipline every
    other evaluation script in this codebase already follows (evaluation/
    splits.py's own docstring: "the safe choice required remembering to
    type something, and the unsafe choice was what you got by pressing
    enter" -- this function was exactly that unsafe default)."""
    import os

    from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing
    from evaluation.splits import load_split
    from scripts.score_gold_recall import load_gold, overlaps
    from src.mollm_tier_gate import AUTO_TIERS
    from src.retrieval import VocabularyRetriever

    vocab = VocabularyRetriever(conn)
    all_test_notes = {r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test=TRUE").fetchall()}
    locked_test_split = load_split("test")
    note_ids = sorted(all_test_notes - locked_test_split)
    n_excluded = len(all_test_notes) - len(note_ids)
    if n_excluded:
        print(f"gather_tp_records(): excluded {n_excluded} note(s) from the "
             f"locked test split (data/splits/note_splits.csv) -- {len(note_ids)} "
             f"notes remain in scope")
    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, note_ids)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    tier_ph = ",".join("?" * len(AUTO_TIERS))
    rows = conn.execute(f"""
        SELECT d.note_id, d.final_candidate_index, e.orig_start, e.orig_end, n.candidates
        FROM mollm_tier_gate_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.tier IN ({tier_ph}) AND n.candidates IS NOT NULL AND n.candidates != '[]'
    """, list(AUTO_TIERS)).fetchall()

    records = []
    for note_id, final_idx, s, e, cands_json in rows:
        candidates = cands_json if isinstance(cands_json, list) else json.loads(cands_json)
        if len(candidates) < 2 or not final_idx or not (1 <= final_idx <= len(candidates)):
            continue
        gold = gold_by_note.get(note_id, [])
        overlapping = [g for g in gold if overlaps(s, e, g["start"], g["end"])]
        if len(overlapping) != 1:
            continue
        g0 = overlapping[0]
        if (e - s) < (g0["end"] - g0["start"]):
            continue
        chosen = candidates[final_idx - 1]
        chosen_cid = chosen.get("omop_concept_id")
        if not chosen_cid:
            continue
        chosen_snomed = vocab.snomed_code_for_concept(chosen_cid)
        if not chosen_snomed or str(chosen_snomed) != str(g0["concept_id"]):
            continue  # not actually correct -- only true positives count
        wrong_ids = [c.get("omop_concept_id") for i, c in enumerate(candidates, 1)
                    if i != final_idx and c.get("omop_concept_id")]
        if not wrong_ids:
            continue
        records.append({"correct_concept_id": chosen_cid, "wrong_candidate_ids": wrong_ids})
    return records


def main():
    random.seed(42)
    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb",
                              read_only=True, max_wait_seconds=120)

    touched = conn.execute("""
        SELECT DISTINCT omop_concept_id FROM (
            SELECT unnest(json_extract(candidates, '$[*].omop_concept_id')::BIGINT[]) AS omop_concept_id
            FROM normalized_entities WHERE is_test=TRUE AND candidates IS NOT NULL
        ) WHERE omop_concept_id IS NOT NULL
    """).fetchall()
    touched_ids = [r[0] for r in touched]
    print(f"{len(touched_ids)} distinct concept_ids touched by this pipeline")

    triples = load_snomed_subgraph(conn, touched_ids)
    print(f"{len(triples)} real SNOMED relationship triples (both endpoints touched)")

    print("\ngathering this project's own true-positive records for extrinsic eval...")
    tp_records = gather_tp_records(conn)
    print(f"{len(tp_records)} TP records (auto-tier, gold-confirmed correct, >=2 real candidates)")
    conn.close()

    random.shuffle(triples)
    split = int(len(triples) * 0.9)
    train_triples, test_triples = triples[:split], triples[split:]
    print(f"\ntrain: {len(train_triples)}  test (held out): {len(test_triples)}")

    entity2idx, relation2idx = build_vocab(triples)
    print(f"vocab: {len(entity2idx)} entities, {len(relation2idx)} relation types")

    print("\ntraining TransE...")
    model = train_transe(train_triples, entity2idx, relation2idx, dim=100, epochs=50)

    save_model(model, entity2idx, relation2idx, MODEL_CHECKPOINT_PATH, dim=100)
    print(f"\nmodel checkpoint written to {MODEL_CHECKPOINT_PATH}")

    print("\n--- intrinsic evaluation (standard KGE protocol, RAW setting) ---")
    link_pred = evaluate_link_prediction(model, test_triples, entity2idx, relation2idx, k=10)
    print(json.dumps(link_pred, indent=2))

    print("\n--- extrinsic evaluation (this project's own TP records) ---")
    extrinsic = evaluate_against_tp_records(model, entity2idx, tp_records)
    print(json.dumps(extrinsic, indent=2))

    out_path = f"{PROJECT_DIR}/logs/kg_embedding_results.json"
    with open(out_path, "w") as f:
        json.dump({"n_touched_concepts": len(touched_ids), "n_triples": len(triples),
                  "n_train": len(train_triples), "n_test": len(test_triples),
                  "n_entities": len(entity2idx), "n_relations": len(relation2idx),
                  "link_prediction": link_pred, "extrinsic": extrinsic}, f, indent=2)
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()
