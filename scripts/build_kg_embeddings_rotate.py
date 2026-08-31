"""scripts/build_kg_embeddings_rotate.py -- 2026-08-31: trains and
evaluates RotatE (src.kg_embedding_rotate) on FOUR independent training
configurations, as a genuine ablation -- not one config picked in advance:

  guideline   -- curated clinical-guideline graph (Memgraph), 355 real
                 grounded edges, 31 relation types.
  gold        -- this project's own gold-confirmed candidate-competition
                 signal (scripts.build_kg_embeddings.gather_tp_records()),
                 1,593 PREFERRED_OVER triples, 452 source TP records.
  combined    -- guideline + gold, concatenated.
  snomed_is_a -- the full SNOMED IS_A hierarchy from the separate KG1
                 Neo4j instance (bolt://localhost:7687), 641,727 real
                 edges before crosswalk-drop -- added as a fourth arm per
                 explicit direction, kept separate from "combined" since
                 it's raw ontology structure, not repurposed/curated data
                 like the other two.

All four configs run through IDENTICAL training/evaluation code
(train_rotate(), evaluate_link_prediction(), evaluate_against_tp_records())
-- only the training triples differ. This is the actual controlled
ablation: does training data source/character change what RotatE learns,
holding architecture and hyperparameters fixed.

Every config's held-out set is genuinely reported, however small --
resist any temptation to round expectations up.

Run: python3 scripts/build_kg_embeddings_rotate.py [--config guideline|gold|combined|snomed_is_a|all]
"""
import argparse
import json
import random
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.kg_embedding_rotate import (  # noqa: E402
    build_vocab, evaluate_against_tp_records, evaluate_link_prediction,
    load_combined_subgraph, load_gold_competition_triples,
    load_guideline_subgraph, load_snomed_is_a_subgraph, save_model,
    train_rotate)

ALL_CONFIGS = ["guideline", "gold", "combined", "snomed_is_a"]


def _load_triples(config, memgraph_driver, neo4j_driver, conn):
    if config == "guideline":
        return load_guideline_subgraph(memgraph_driver, conn)
    if config == "gold":
        return load_gold_competition_triples(conn)
    if config == "combined":
        return load_combined_subgraph(memgraph_driver, conn)
    if config == "snomed_is_a":
        return load_snomed_is_a_subgraph(neo4j_driver, conn)
    raise ValueError(f"unknown config {config!r}")


def run_one_config(config, memgraph_driver, neo4j_driver, conn, tp_records):
    print(f"\n{'='*70}\nCONFIG: {config}\n{'='*70}")
    random.seed(42)

    triples = _load_triples(config, memgraph_driver, neo4j_driver, conn)
    print(f"{len(triples)} real triples loaded")
    if len(triples) < 20:
        print(f"WARNING: {config} has too few triples to train/evaluate meaningfully "
             f"({len(triples)} < 20) -- reporting honestly, not skipping silently.")
        return {"config": config, "n_triples": len(triples), "skipped": True}

    relation_types = sorted({r for _, r, _ in triples})
    print(f"{len(relation_types)} distinct relation types")

    random.shuffle(triples)
    split = max(1, int(len(triples) * 0.9))
    train_triples, test_triples = triples[:split], triples[split:]
    print(f"train: {len(train_triples)}  test (held out): {len(test_triples)}")

    entity2idx, relation2idx = build_vocab(triples)
    print(f"vocab: {len(entity2idx)} entities, {len(relation2idx)} relation types")

    print(f"\ntraining RotatE ({config})...")
    model = train_rotate(train_triples, entity2idx, relation2idx, dim=100, epochs=50)

    ckpt_path = f"{PROJECT_DIR}/models/kg_rotate_{config}_v1.pt"
    save_model(model, entity2idx, relation2idx, ckpt_path, dim=100)
    print(f"model checkpoint written to {ckpt_path}")

    print("\n--- intrinsic evaluation (standard KGE protocol, RAW setting) ---")
    link_pred = evaluate_link_prediction(model, test_triples, entity2idx, relation2idx, k=10) \
        if test_triples else {"mrr": None, "hits_at_10": None, "n_evaluated": 0}
    print(json.dumps(link_pred, indent=2))

    print("\n--- extrinsic evaluation (this project's own TP records) ---")
    extrinsic = evaluate_against_tp_records(model, entity2idx, tp_records)
    print(json.dumps(extrinsic, indent=2))

    result = {"config": config, "n_triples": len(triples), "n_relation_types": len(relation_types),
              "n_train": len(train_triples), "n_test": len(test_triples),
              "n_entities": len(entity2idx), "n_relations": len(relation2idx),
              "link_prediction": link_pred, "extrinsic": extrinsic, "skipped": False}

    out_path = f"{PROJECT_DIR}/logs/kg_embedding_rotate_{config}_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"results written to {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="all", choices=ALL_CONFIGS + ["all"])
    args = parser.parse_args()
    configs = ALL_CONFIGS if args.config == "all" else [args.config]

    from src.kg3_ingestion import get_memgraph_driver
    from neo4j import GraphDatabase
    import os

    memgraph_driver = get_memgraph_driver()
    neo4j_driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "secure_password_here")))

    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb",
                              read_only=True, max_wait_seconds=120)

    print("gathering this project's own true-positive records for extrinsic eval "
         "(shared across all configs -- the eval population, not the training data)...")
    from scripts.build_kg_embeddings import gather_tp_records
    tp_records = gather_tp_records(conn)
    print(f"{len(tp_records)} TP records (auto-tier, gold-confirmed correct, >=2 real candidates)")

    all_results = {}
    for config in configs:
        all_results[config] = run_one_config(config, memgraph_driver, neo4j_driver, conn, tp_records)

    conn.close()
    memgraph_driver.close()
    neo4j_driver.close()

    print(f"\n{'='*70}\nSUMMARY (all configs run this session)\n{'='*70}")
    for config, r in all_results.items():
        if r.get("skipped"):
            print(f"{config:>12}: SKIPPED ({r['n_triples']} triples, too few)")
            continue
        lp, ex = r["link_prediction"], r["extrinsic"]
        mrr = lp.get("mrr")
        frac = ex.get("frac_wrong_candidate_closer_than_random")
        mrr_str = f"{mrr:.4f}" if mrr is not None else "n/a"
        frac_str = f"{frac:.3f}" if frac is not None else "n/a"
        print(f"{config:>12}: n_triples={r['n_triples']:>7}  MRR={mrr_str}  frac_correct_signal={frac_str}")

    summary_path = f"{PROJECT_DIR}/logs/kg_embedding_rotate_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nfull ablation summary written to {summary_path}")


if __name__ == "__main__":
    main()
