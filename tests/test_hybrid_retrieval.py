"""
tests/test_hybrid_retrieval.py — src/normalization/tier_retrieval.py's Pass 3
hybrid (BM25 + SapBERT + prior) Reciprocal Rank Fusion (2026-08-16).

Tests only the pure ranking arithmetic (_rrf_scores) and, via a fake DuckDB
connection, the fusion/truncation/alias-guarantee logic in
_tier3_hybrid_rows() -- not a live BM25 index or a real SapBERT embedding.
Same "stub the heavy imports, exercise the shipped code, not a copy of it"
approach tests/test_tier12_ranking.py already established for this same
module (importing src.normalization loads an actual SapBERT model at import
time otherwise).

Run: python3 -m pytest tests/test_hybrid_retrieval.py -v
     (or: python3 tests/test_hybrid_retrieval.py for a plain-output run)
"""

import os
import sys
import types

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def _install_stubs():
    """Returns the list of module names this call actually stubbed (i.e.
    weren't already real, present modules). 2026-08-17 fix: this used to
    leave a fake `duckdb` module permanently in sys.modules with no
    cleanup -- caught live when it made tests/test_tier_gate_grading.py
    fail with "module 'duckdb' has no attribute 'connect'" ONLY when run
    as part of the full suite (this file runs earlier, alphabetically),
    never standalone. Same bug, same fix, as
    tests/test_tier12_ranking.py's identical _install_stubs().
    """
    installed = []

    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")

        class _NoGrad:
            def __enter__(self): return None
            def __exit__(self, *a): return False

        torch.no_grad = _NoGrad
        sys.modules["torch"] = torch
        installed.append("torch")

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")

        class _FromPretrained:
            @classmethod
            def from_pretrained(cls, *a, **k):
                return cls()

            def __call__(self, *a, **k):
                raise AssertionError(
                    "the real tokenizer/model must never be called in these tests")

        transformers.AutoTokenizer = _FromPretrained
        transformers.AutoModel = _FromPretrained
        sys.modules["transformers"] = transformers
        installed.append("transformers")

    if "duckdb" not in sys.modules:
        duckdb_stub = types.ModuleType("duckdb")
        duckdb_stub.Error = Exception
        sys.modules["duckdb"] = duckdb_stub
        installed.append("duckdb")

    return installed


_stubbed_modules = _install_stubs()

import src.normalization as N  # noqa: E402

# See tests/test_tier12_ranking.py's identical cleanup for why this matters:
# only remove the stubs THIS file installed, so a later test file's real
# import gets the genuine library, not a leaked stand-in.
for _name in _stubbed_modules:
    sys.modules.pop(_name, None)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Fake connection for _tier3_hybrid_rows(): .sql()/.execute() both
    return canned rows keyed off a simple substring match on the query text,
    so the test controls exactly what the "dense missing-embedding lookup"
    sub-query returns without needing a real DuckDB file.
    """
    def __init__(self, missing_dense_rows=None):
        self.missing_dense_rows = missing_dense_rows or []

    def sql(self, query, params=None):
        return _FakeResult(self.missing_dense_rows)

    def execute(self, query, params=None):
        return _FakeResult(self.missing_dense_rows)


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # _rrf_scores
    # ======================================================================
    scores = N._rrf_scores([10, 20, 30])
    check("rank 1 scores highest", scores[10] > scores[20] > scores[30])
    check("RRF formula matches 1/(k+rank)",
          abs(scores[10] - 1.0 / (N.RRF_K + 1)) < 1e-9
          and abs(scores[30] - 1.0 / (N.RRF_K + 3)) < 1e-9)
    check("empty list -> empty dict", N._rrf_scores([]) == {})

    # ======================================================================
    # _tier3_hybrid_rows -- fusion, truncation, alias guarantee
    # ======================================================================
    import unittest.mock as mock

    # Two concepts found by BOTH dense and sparse (should fuse to a higher
    # combined score than a concept found by only one signal).
    row_a = (1, "Concept A", "Condition", "SNOMED", 0.9)
    row_b = (2, "Concept B", "Condition", "SNOMED", 0.5)
    row_c = (3, "Concept C", "Condition", "SNOMED", 0.6)
    # An alias-only concept: not in the dense OR sparse top-K pool at all,
    # force-included via alias_ids, its cosine looked up in the
    # missing-dense follow-up query.
    row_alias = (4, "Concept Alias", "Condition", "RxNorm", 0.4)

    with mock.patch("src.normalization.tier_retrieval._tier3_semantic_rows",
                    return_value=[row_a, row_c]), \
         mock.patch("src.normalization.bm25_index.query_bm25",
                    return_value=[(1, "Concept A", "Condition", "SNOMED", 8.0),
                                  (2, "Concept B", "Condition", "SNOMED", 6.0)]):
        conn = _FakeConn(missing_dense_rows=[row_b, row_alias])
        cands = N._tier3_hybrid_rows(conn, "entity text", [0.1, 0.2], ["SNOMED"],
                                     ["Condition"], alias_ids=[4])

    names = [c["concept_name"] for c in cands]
    check("concept found by BOTH dense+sparse ranks first (A)",
          names[0] == "Concept A")
    check("alias-only concept is force-included even though absent from "
          "both dense and sparse pools", "Concept Alias" in names)
    check("alias concept carries verified_brand_alias match_basis",
          next(c for c in cands if c["concept_name"] == "Concept Alias")
          ["match_basis"] == "verified_brand_alias")
    check("non-alias concepts keep the default match_basis (unknown)",
          next(c for c in cands if c["concept_name"] == "Concept A")
          ["match_basis"] != "verified_brand_alias")
    check("similarity_score is the DENSE cosine, not the RRF score",
          next(c for c in cands if c["concept_name"] == "Concept A")
          ["similarity_score"] == 0.9)
    check("rrf_score field is present and distinct from similarity_score",
          all("rrf_score" in c for c in cands))
    check("retrieval_method is tagged hybrid_rrf",
          all(c["retrieval_method"] == "hybrid_rrf" for c in cands))
    check("result never exceeds CANDIDATE_LIMIT", len(cands) <= N.CANDIDATE_LIMIT)

    # Truncation still guarantees the alias slot even when there are more
    # than CANDIDATE_LIMIT other, higher-RRF-scoring candidates.
    many_dense = [(i, f"Dense {i}", "Condition", "SNOMED", 0.99) for i in range(10, 10 + 10)]
    with mock.patch("src.normalization.tier_retrieval._tier3_semantic_rows",
                    return_value=many_dense), \
         mock.patch("src.normalization.bm25_index.query_bm25",
                    return_value=[]):
        conn = _FakeConn(missing_dense_rows=[row_alias])
        cands = N._tier3_hybrid_rows(conn, "entity text", [0.1, 0.2], ["SNOMED"],
                                     ["Condition"], alias_ids=[4])
    check("alias survives truncation even outranked by CANDIDATE_LIMIT+ "
          "higher-scoring dense-only hits",
          "Concept Alias" in [c["concept_name"] for c in cands])
    check("truncated result still respects CANDIDATE_LIMIT",
          len(cands) <= N.CANDIDATE_LIMIT)

    print(f"hybrid-retrieval tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_hybrid_retrieval():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
