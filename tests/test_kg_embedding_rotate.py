"""
tests/test_kg_embedding_rotate.py — src/kg_embedding_rotate.py's RotatE
implementation: scoring function, packed-vector shape, training
convergence on a tiny synthetic graph, save/load roundtrip, both
evaluation functions (imported unchanged from src.kg_embedding, proving
the model-agnostic contract holds in practice) -- plus the three
Memgraph/Neo4j-backed loaders, exercised against mocked driver/connection
objects so no live DB/graph is needed to run this file.

Run: python3 -m pytest tests/test_kg_embedding_rotate.py -v
"""
import sys
import tempfile
from unittest.mock import MagicMock

import torch

from src.kg_embedding_rotate import (
    RotatE,
    build_vocab,
    evaluate_against_tp_records,
    evaluate_link_prediction,
    load_combined_subgraph,
    load_gold_competition_triples,
    load_guideline_subgraph,
    load_model,
    load_snomed_is_a_subgraph,
    save_model,
    train_rotate,
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
    # RotatE scoring function / packed-vector shape
    # ======================================================================
    model = RotatE(n_entities=5, n_relations=2, dim=8)
    check("entity_emb is packed real+imaginary: width == 2*dim",
          model.entity_emb.embedding_dim == 16)
    check("relation_phase width == dim (one phase angle per dimension, not 2*dim)",
          model.relation_phase.embedding_dim == 8)
    check("relation_phase initialized within [-pi, pi]",
          model.relation_phase.weight.min().item() >= -torch.pi - 1e-4
          and model.relation_phase.weight.max().item() <= torch.pi + 1e-4)
    check("entities are NOT unit-normalized at init (deliberate RotatE deviation from TransE)",
          not torch.allclose(model.entity_emb.weight.norm(dim=1), torch.ones(5), atol=1e-2))

    h = torch.tensor([0, 1])
    r = torch.tensor([0, 1])
    t = torch.tensor([2, 3])
    scores = model.score(h, r, t)
    check("score() returns one scalar per input triple", scores.shape == (2,))

    mean_norm, max_norm = model.entity_norm_stats()
    check("entity_norm_stats returns real positive floats",
          mean_norm > 0 and max_norm >= mean_norm)

    # ======================================================================
    # train_rotate -- NOT src.kg_embedding's own transitive-chain-
    # generalization check (0 IS_A 1, 1 IS_A 2 => untrained 0 IS_A 2 scores
    # higher than its reverse). That check is TransE-specific: proved
    # algebraically (see module docstring's caveat) that TransE's h+r=t
    # additive geometry makes it a mathematical certainty once training
    # converges near-exactly (e2 = e0+2r implies score(0,r,2) = -||r|| vs.
    # score(2,r,0) = -||3r||, always closer for the implied direction).
    # RotatE's rotational composition has NO equivalent guarantee -- the
    # relative ordering depends on where the trained phase angle lands
    # (chord length 2sin(theta/2) vs 2sin(3theta/2) is not monotonic), so
    # copying that exact test would be checking the wrong property for
    # this architecture. Instead: the direct thing RotatE's own margin-
    # ranking loss actually optimizes -- a real training triple should
    # score higher than a corrupted (wrong-tail) version of itself.
    # ======================================================================
    chain_triples = [(0, "IS_A", 1), (1, "IS_A", 2)] * 20
    e2i, r2i = build_vocab(chain_triples)
    trained = train_rotate(chain_triples, e2i, r2i, dim=16, epochs=100,
                           batch_size=8, lr=0.05, device="cpu")
    with torch.no_grad():
        real_score = trained.score(
            torch.tensor(e2i[0]), torch.tensor(r2i["IS_A"]), torch.tensor(e2i[1]))
        corrupted_score = trained.score(
            torch.tensor(e2i[0]), torch.tensor(r2i["IS_A"]), torch.tensor(e2i[2]))
    check("trained RotatE scores a real training triple (0 IS_A 1) higher than "
         "the same head/relation with a corrupted (wrong) tail",
          real_score.item() > corrupted_score.item())

    # ======================================================================
    # save_model / load_model roundtrip
    # ======================================================================
    with tempfile.NamedTemporaryFile(suffix=".pt") as tf:
        save_model(trained, e2i, r2i, tf.name, dim=16)
        reloaded, e2i_r, r2i_r = load_model(tf.name)
        check("load_model reconstructs the same vocab", e2i_r == e2i and r2i_r == r2i)
        with torch.no_grad():
            reloaded_score = reloaded.score(
                torch.tensor(e2i[0]), torch.tensor(r2i["IS_A"]), torch.tensor(e2i[1]))
        check("reloaded model reproduces the same score",
              abs(reloaded_score.item() - real_score.item()) < 1e-5)

    # ======================================================================
    # evaluate_link_prediction / evaluate_against_tp_records -- imported
    # UNCHANGED from src.kg_embedding; proves the model-agnostic contract
    # (only ever call .score()/.entity_emb()) holds for RotatE too.
    # ======================================================================
    result = evaluate_link_prediction(trained, chain_triples[:2], e2i, r2i, k=2, device="cpu")
    check("link-prediction eval returns a valid MRR in (0, 1]",
          result["mrr"] is not None and 0 < result["mrr"] <= 1.0)

    tp_records = [{"correct_concept_id": 0, "wrong_candidate_ids": [1]}]
    extrinsic = evaluate_against_tp_records(trained, e2i, tp_records, device="cpu")
    check("extrinsic eval works unchanged against a RotatE model",
          extrinsic["n_usable_records"] == 1)

    # ======================================================================
    # load_guideline_subgraph -- mocked Memgraph driver + DuckDB conn
    # ======================================================================
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value.data.return_value = [
        {"h": "111", "r": "INDICATES", "t": "222"},
        {"h": "333", "r": "N/A_CODE", "t": "444"},  # will fail to crosswalk below
    ]
    mock_conn = MagicMock()

    def fake_resolve_snomed_cui(conn, cui, domains=None):
        mapping = {"111": (1001, "concept a", "Condition", "SNOMED"),
                  "222": (1002, "concept b", "Condition", "SNOMED")}
        return mapping.get(cui)

    import scripts.backfill_guideline_grounding as bgg
    orig = bgg.resolve_snomed_cui
    bgg.resolve_snomed_cui = fake_resolve_snomed_cui
    try:
        triples = load_guideline_subgraph(mock_driver, mock_conn)
    finally:
        bgg.resolve_snomed_cui = orig

    check("load_guideline_subgraph grounds a real edge to OMOP concept_ids",
          (1001, "INDICATES", 1002) in triples)
    check("load_guideline_subgraph drops an edge whose endpoint fails to crosswalk",
          len(triples) == 1)

    # ======================================================================
    # load_gold_competition_triples -- mocked gather_tp_records()
    # ======================================================================
    import scripts.build_kg_embeddings as bke
    orig_gather = bke.gather_tp_records
    bke.gather_tp_records = lambda conn: [
        {"correct_concept_id": 10, "wrong_candidate_ids": [11, 12]},
        {"correct_concept_id": 20, "wrong_candidate_ids": [21]},
    ]
    try:
        gold_triples = load_gold_competition_triples(mock_conn)
    finally:
        bke.gather_tp_records = orig_gather

    check("load_gold_competition_triples flattens each wrong candidate into its own triple",
          set(gold_triples) == {(10, "PREFERRED_OVER", 11), (10, "PREFERRED_OVER", 12),
                                (20, "PREFERRED_OVER", 21)})

    # ======================================================================
    # load_combined_subgraph -- concatenation
    # ======================================================================
    bgg.resolve_snomed_cui = fake_resolve_snomed_cui
    bke.gather_tp_records = lambda conn: [{"correct_concept_id": 10, "wrong_candidate_ids": [11]}]
    try:
        combined = load_combined_subgraph(mock_driver, mock_conn)
    finally:
        bgg.resolve_snomed_cui = orig
        bke.gather_tp_records = orig_gather

    check("load_combined_subgraph concatenates guideline + gold triples",
          (1001, "INDICATES", 1002) in combined and (10, "PREFERRED_OVER", 11) in combined)

    # ======================================================================
    # load_snomed_is_a_subgraph -- mocked Neo4j driver + DuckDB conn
    # ======================================================================
    mock_neo4j_driver = MagicMock()
    mock_neo4j_session = MagicMock()
    mock_neo4j_driver.session.return_value.__enter__.return_value = mock_neo4j_session
    mock_neo4j_session.run.return_value.data.return_value = [
        {"h": "111", "t": "222"},
        {"h": "555", "t": "666"},  # 555/666 won't crosswalk below
    ]
    mock_conn2 = MagicMock()
    mock_conn2.execute.return_value.fetchall.return_value = [("111", 1001), ("222", 1002)]

    is_a_triples = load_snomed_is_a_subgraph(mock_neo4j_driver, mock_conn2)
    check("load_snomed_is_a_subgraph grounds a real edge to OMOP concept_ids",
          (1001, "IS_A", 1002) in is_a_triples)
    check("load_snomed_is_a_subgraph drops an edge whose endpoints don't crosswalk",
          len(is_a_triples) == 1)

    print(f"kg-embedding-rotate tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_kg_embedding_rotate():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
