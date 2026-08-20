"""
tests/test_kg_embedding.py — src/kg_embedding.py's TransE implementation:
scoring function, vocab building, training convergence on a tiny
synthetic graph, and both evaluation functions.

No DB dependency -- load_snomed_subgraph() (the one function that needs a
live connection) is exercised in scripts/build_kg_embeddings.py directly,
not here; everything else operates on plain Python triples.

Run: python3 -m pytest tests/test_kg_embedding.py -v
"""
import sys

import torch

from src.kg_embedding import (
    TransE,
    build_vocab,
    evaluate_against_tp_records,
    evaluate_link_prediction,
    train_transe,
)


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # build_vocab
    # ======================================================================
    triples = [(1, "IS_A", 2), (2, "IS_A", 3), (1, "TREATS", 4)]
    e2i, r2i = build_vocab(triples)
    check("build_vocab collects every distinct entity from both head and tail",
          set(e2i.keys()) == {1, 2, 3, 4})
    check("build_vocab collects every distinct relation",
          set(r2i.keys()) == {"IS_A", "TREATS"})
    check("indices are dense (0..n-1)",
          set(e2i.values()) == set(range(4)) and set(r2i.values()) == set(range(2)))

    # ======================================================================
    # TransE scoring function
    # ======================================================================
    model = TransE(n_entities=5, n_relations=2, dim=8)
    h = torch.tensor([0, 1])
    r = torch.tensor([0, 1])
    t = torch.tensor([2, 3])
    scores = model.score(h, r, t)
    check("score() returns one scalar per input triple",
          scores.shape == (2,))
    check("entity embeddings are unit-normalized at init",
          torch.allclose(model.entity_emb.weight.norm(dim=1), torch.ones(5), atol=1e-4))

    # ======================================================================
    # train_transe -- a genuinely learnable tiny synthetic graph. A simple
    # 3-entity chain (0 -IS_A-> 1 -IS_A-> 2) plus its structural
    # consequence under TransE's own additive geometry: since
    # h+r≈t for both edges, (0, IS_A, 2) should score highly too once
    # trained, EVEN THOUGH it was never in the training set -- this is
    # the actual generalization property TransE is supposed to learn,
    # not just memorization.
    # ======================================================================
    chain_triples = [(0, "IS_A", 1), (1, "IS_A", 2)] * 20  # repeat for enough gradient signal
    e2i2, r2i2 = build_vocab(chain_triples)
    trained = train_transe(chain_triples, e2i2, r2i2, dim=16, epochs=100,
                           batch_size=8, lr=0.05, device="cpu")
    with torch.no_grad():
        implied_score = trained.score(
            torch.tensor(e2i2[0]), torch.tensor(r2i2["IS_A"]), torch.tensor(e2i2[2]))
        # compare against a clearly-wrong triple (2 -IS_A-> 0, reversed)
        wrong_score = trained.score(
            torch.tensor(e2i2[2]), torch.tensor(r2i2["IS_A"]), torch.tensor(e2i2[0]))
    check("trained TransE generalizes the transitive IS_A structure -- "
         "the UNSEEN implied triple (0 IS_A 2) scores higher than its "
         "reversed (clearly wrong) counterpart",
          implied_score.item() > wrong_score.item())

    # ======================================================================
    # evaluate_link_prediction -- sanity on a trained model
    # ======================================================================
    result = evaluate_link_prediction(trained, chain_triples[:2], e2i2, r2i2, k=2, device="cpu")
    check("link-prediction eval returns a valid MRR in (0, 1]",
          result["mrr"] is not None and 0 < result["mrr"] <= 1.0)
    check("link-prediction eval reports how many triples were evaluated",
          result["n_evaluated"] == 2)

    check("triples referencing an unknown entity/relation are skipped, not crashed on",
          evaluate_link_prediction(trained, [(999, "IS_A", 1)], e2i2, r2i2, device="cpu")["n_evaluated"] == 0)

    # ======================================================================
    # evaluate_against_tp_records
    # ======================================================================
    tp_records = [
        {"correct_concept_id": 0, "wrong_candidate_ids": [1]},
        {"correct_concept_id": 1, "wrong_candidate_ids": [2]},
    ]
    extrinsic = evaluate_against_tp_records(trained, e2i2, tp_records, device="cpu")
    check("extrinsic eval processes both provided TP records",
          extrinsic["n_tp_records_provided"] == 2 and extrinsic["n_usable_records"] == 2)
    check("extrinsic eval produces at least one comparison",
          extrinsic["n_comparisons"] >= 1)
    check("the reported fraction is a valid probability",
          0.0 <= extrinsic["frac_wrong_candidate_closer_than_random"] <= 1.0)

    check("a TP record whose correct concept isn't in the vocab is skipped, not crashed on",
          evaluate_against_tp_records(
              trained, e2i2, [{"correct_concept_id": 999, "wrong_candidate_ids": [1]}],
              device="cpu")["n_usable_records"] == 0)

    check("a TP record with no wrong candidates at all is skipped",
          evaluate_against_tp_records(
              trained, e2i2, [{"correct_concept_id": 0, "wrong_candidate_ids": []}],
              device="cpu")["n_usable_records"] == 0)

    print(f"kg-embedding tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_kg_embedding():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
