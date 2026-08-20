"""
tests/test_kg_embedding_tiebreak.py — src/kg_embedding_tiebreak.py's
pool-consistency tiebreak for the SNOMED near-duplicate-concept pattern.
Trains a tiny, genuinely structured synthetic TransE model (not the real
SNOMED-scale one -- that lives in scripts/build_kg_embeddings.py) so the
tiebreak logic itself can be verified against a KNOWN-correct answer.

Run: python3 -m pytest tests/test_kg_embedding_tiebreak.py -v
"""
import sys

from src.kg_embedding import build_vocab, train_transe
from src.kg_embedding_tiebreak import kg_tiebreak_score, pick_via_kg_tiebreak


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # Build a tiny synthetic graph with a KNOWN-correct tiebreak answer:
    # a "cardiac cluster" (0,1,2 all connected to each other via
    # RELATED_TO) and candidate 10 is IS_A-tied to candidate 11, but 10 is
    # ALSO connected into the cardiac cluster while 11 is an isolated
    # outlier with no other connections. If the entity's broader candidate
    # pool is [0, 1, 2] (the cardiac cluster, standing in for "what
    # SapBERT independently proposed"), 10 should win the tiebreak over
    # 11 -- it's structurally consistent with that neighborhood, 11 isn't.
    # ======================================================================
    triples = (
        [(0, "RELATED_TO", 1), (1, "RELATED_TO", 2), (0, "RELATED_TO", 2)] * 10
        + [(10, "RELATED_TO", 0), (10, "RELATED_TO", 1)] * 10  # 10 is IN the cluster
        + [(10, "IS_A", 11)] * 10  # 10 and 11 are the hierarchy-tied pair
        + [(11, "UNRELATED_TO", 99)] * 10  # 11 only connects to an outlier
    )
    e2i, r2i = build_vocab(triples)
    model = train_transe(triples, e2i, r2i, dim=16, epochs=150, batch_size=8,
                         lr=0.05, device="cpu")

    # ======================================================================
    # kg_tiebreak_score
    # ======================================================================
    score_10 = kg_tiebreak_score(model, e2i, 10, [0, 1, 2])
    score_11 = kg_tiebreak_score(model, e2i, 11, [0, 1, 2])
    check("both scores are real floats", isinstance(score_10, float) and isinstance(score_11, float))
    check("the cluster-connected candidate (10) scores closer to the pool "
         "than the outlier-connected one (11)",
          score_10 < score_11)

    check("candidate_id not in vocab -> None, no crash",
          kg_tiebreak_score(model, e2i, 99999, [0, 1, 2]) is None)
    check("empty pool (nothing to compare against) -> None",
          kg_tiebreak_score(model, e2i, 10, []) is None)
    check("pool containing only candidate_id itself -> None (self-excluded)",
          kg_tiebreak_score(model, e2i, 10, [10]) is None)

    # ======================================================================
    # pick_via_kg_tiebreak -- the actual decision function
    # ======================================================================
    result = pick_via_kg_tiebreak(model, e2i, tied_concept_ids=[10, 11],
                                  full_pool_concept_ids=[0, 1, 2, 10, 11])
    check("pick_via_kg_tiebreak resolves (both tied candidates had usable scores)",
          result["resolved"] is True)
    check("pick_via_kg_tiebreak picks the structurally-consistent candidate (10), "
         "not its hierarchy-tied sibling (11) -- the actual point of this module",
          result["winner"] == 10)
    check("both tied candidates' individual scores are reported for audit",
          set(result["scores"].keys()) == {10, 11})

    unresolvable = pick_via_kg_tiebreak(model, e2i, tied_concept_ids=[10, 99999],
                                        full_pool_concept_ids=[0, 1, 2])
    check("fewer than 2 usable tied candidates -> resolved=False, caller falls "
         "back to its own existing tiebreak",
          unresolvable["resolved"] is False and unresolvable["winner"] is None)

    print(f"kg-embedding-tiebreak tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_kg_embedding_tiebreak():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
