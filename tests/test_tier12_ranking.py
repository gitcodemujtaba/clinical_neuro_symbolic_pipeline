"""
tests/test_tier12_ranking.py — isolated tests for src/normalization.py's
Tier 1/2 ranker (2026-08-13, docs/2026-08-13_Code_Improvement_Proposals.md P1.1).

WHY THESE ARE STUB TESTS WITH A HAND-WRITTEN FAKE CONNECTION, NOT LIVE-DB
TESTS. Importing src/normalization.py loads SapBERT (a ~400MB transformer) and
expects an Athena-populated DuckDB, neither of which belongs in a unit test.
The ranking logic is pure ordering over rows plus two lookups, so the lookups
are faked and the ordering is asserted directly -- the same approach
tests/test_offset_mapping.py takes with its fake spaCy tokenizer, and the same
approach the 2026-08-13 session used for _select_by_groundability().

The functions under test are re-implemented here?  NO -- they are imported. The
module-level SapBERT load is stubbed out first (see _install_stubs) so the
import succeeds without the model. That keeps these tests honest: they exercise
the shipped code, not a copy of it.

Run:  python3 tests/test_tier12_ranking.py
"""

import os
import sys
import types

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def _install_stubs():
    """Stubs torch/transformers/duckdb so `import src.normalization` does not
    download or load a model. Only the names normalization.py touches at
    import time are provided; anything else it reaches for should fail loudly
    rather than be silently faked.

    Returns the list of module names THIS CALL actually stubbed (i.e. that
    weren't already real, present modules) -- the caller uses this to clean
    up afterward. 2026-08-17 fix: this used to leave a fake `duckdb` module
    permanently installed in sys.modules with no cleanup at all, so any test
    FILE that happened to run later in the same pytest session (alphabetical
    collection order can put this file before one that needs a real
    duckdb.connect()) would silently get the empty stub instead of the real
    library -- caught live via tests/test_tier_gate_grading.py failing with
    "module 'duckdb' has no attribute 'connect'" only when run as part of
    the full suite, never standalone. Real test-isolation bug, not
    specific to that one new test.
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
        sys.modules["duckdb"] = types.ModuleType("duckdb")
        installed.append("duckdb")

    return installed


_stubbed_modules = _install_stubs()

import src.normalization as N  # noqa: E402

# Remove exactly the stubs THIS FILE installed, now that the one import that
# needed them has completed -- any other test file (this one or a later one
# in the same pytest session) that does a real `import torch`/`transformers`/
# `duckdb` must get the genuine library, not this file's throwaway stand-in.
# Modules that were ALREADY real before _install_stubs() ran are untouched
# (never in _stubbed_modules), so this can't accidentally evict a real,
# already-loaded module some earlier test file depended on.
for _name in _stubbed_modules:
    sys.modules.pop(_name, None)


class FakeConn:
    """Answers exactly the two queries _rank_tier12_candidates() issues:
    the concept_class_id map and the ancestor-count map. Anything else raises,
    so a future change that starts issuing a third query fails visibly here
    instead of silently returning nothing and degrading the ranking.
    """

    def __init__(self, classes=None, depths=None, fail=False):
        self.classes = classes or {}
        self.depths = depths or {}
        self.fail = fail
        self.queries = []

    def sql(self, query, params=None):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("simulated database failure")
        outer = self

        class _R:
            def fetchall(self_):
                if "concept_class_id" in query:
                    return [(cid, outer.classes.get(cid))
                            for cid in (params or []) if cid in outer.classes]
                if "athena_concept_ancestor" in query:
                    return [(cid, outer.depths.get(cid))
                            for cid in (params or []) if cid in outer.depths]
                raise AssertionError(f"unexpected query: {query[:80]}")
        return _R()


def fake_embedding(text):
    """Deterministic 3-d 'embedding' keyed off the text, so cosine similarity
    is predictable without a model. "fracture"-like strings point one way,
    "fraction"-like strings another, everything else in between.
    """
    t = (text or "").lower()
    if "fracture" in t or t == "fx":
        return [1.0, 0.0, 0.0]
    if "fraction" in t:
        return [0.0, 1.0, 0.0]
    if "left" in t:
        return [0.0, 0.0, 1.0]
    return [0.5, 0.5, 0.5]


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    original = N.get_sapbert_embedding
    N.get_sapbert_embedding = fake_embedding
    try:
        # ---------------------------------------------------------------
        # 1. Single row: no ranking attempted, no queries issued at all.
        # ---------------------------------------------------------------
        conn = FakeConn()
        rows = [(100, "Fracture", "Condition", "SNOMED")]
        out, basis, tie = N._rank_tier12_candidates(conn, rows, "Condition", "fx")
        check("single row returns unchanged", out == rows)
        check("single row basis is legacy", basis == "concept_id_asc")
        check("single row not a tie", tie is False)
        check("single row costs no queries", conn.queries == [])

        # ---------------------------------------------------------------
        # 2. THE HEADLINE CASE. Class preference beats concept_id order.
        #    Lower concept_id (200) is a Qualifier Value; the clinically
        #    right answer (900) is a Clinical Finding. Legacy ordering picks
        #    200; the ranker must pick 900.
        # ---------------------------------------------------------------
        conn = FakeConn(classes={200: "Qualifier Value", 900: "Clinical Finding"})
        rows = [(200, "Left", "Observation", "SNOMED"),
                (900, "Left", "Condition", "SNOMED")]
        out, basis, tie = N._rank_tier12_candidates(conn, rows, "Condition", "left")
        check("class preference beats concept_id", out[0][0] == 900)
        check("basis records the ranker ran", basis == "ranked_v1")

        # ---------------------------------------------------------------
        # 3. Domain agreement decides when class ties.
        # ---------------------------------------------------------------
        conn = FakeConn(classes={300: "Clinical Finding", 400: "Clinical Finding"})
        rows = [(300, "X", "Observation", "SNOMED"),
                (400, "X", "Condition", "SNOMED")]
        out, _b, _t = N._rank_tier12_candidates(conn, rows, "Condition", "x",
                                                domains=["Condition"])
        check("domain agreement decides", out[0][0] == 400)

        # ---------------------------------------------------------------
        # 4. THE `fx` CASE from the 2026-08-13 report S3.1. Both readings are
        #    real OMOP concepts; semantic similarity to the entity text is
        #    what separates them, and "fracture" must win over "fractions".
        # ---------------------------------------------------------------
        #    2026-08-13: the semantic criterion is now opt-in
        #    (TIER12_RANK_SEMANTIC, default OFF) because it is the only
        #    criterion that costs an embedding call, and Tier 1/2 is called
        #    per candidate boundary during compound-split search. Passed
        #    explicitly here so this case tests the semantic path regardless
        #    of the process-wide default.
        conn = FakeConn(classes={500: "Clinical Finding", 600: "Clinical Finding"})
        rows = [(500, "Fractions", "Condition", "SNOMED"),
                (600, "Fracture", "Condition", "SNOMED")]
        out, basis4, tie = N._rank_tier12_candidates(conn, rows, "Condition", "fracture",
                                                     use_semantic=True)
        check("semantics picks fracture over fractions", out[0][0] == 600)
        check("clear semantic winner is not a tie", tie is False)
        check("semantic basis is distinguishable", basis4 == "ranked_v1_semantic")

        # ---------------------------------------------------------------
        # 4b. METADATA-ONLY IS THE DEFAULT, AND IT COSTS NO EMBEDDINGS.
        #     With semantics off the same two candidates are indistinguishable
        #     on class, domain and depth, so the ranker must fall back to
        #     concept_id AND report the tie as unresolved rather than pretend
        #     it settled anything. This is the honest-abstention property the
        #     whole ranker rests on: an arbitrary pick that ADMITS it is
        #     arbitrary routes to Stage 3; one that does not is the 52.71% bug.
        # ---------------------------------------------------------------
        calls = []
        N.get_sapbert_embedding = lambda t: (calls.append(t), fake_embedding(t))[1]
        conn = FakeConn(classes={500: "Clinical Finding", 600: "Clinical Finding"})
        out, basis4b, tie4b = N._rank_tier12_candidates(
            conn, rows, "Condition", "fracture", use_semantic=False)
        check("metadata-only makes NO embedding calls", calls == [])
        check("metadata-only basis omits the semantic marker",
              basis4b == "ranked_v1")
        check("metadata-only falls back to concept_id", out[0][0] == 500)
        check("metadata-only flags the unresolved tie", tie4b is True)
        N.get_sapbert_embedding = fake_embedding

        check("semantic ranking is OFF by default", N.TIER12_RANK_SEMANTIC is False)

        # ---------------------------------------------------------------
        # 5. Specificity breaks a class+domain+semantic tie: deeper in the
        #    hierarchy (more ancestors) wins, even though its id is higher.
        # ---------------------------------------------------------------
        conn = FakeConn(classes={700: "Clinical Finding", 800: "Clinical Finding"},
                        depths={700: 3, 800: 11})
        rows = [(700, "Zzz", "Condition", "SNOMED"),
                (800, "Zzz", "Condition", "SNOMED")]
        out, _b, tie = N._rank_tier12_candidates(conn, rows, "Condition", "zzz")
        check("specificity breaks the tie", out[0][0] == 800)
        check("identical-name tie IS flagged unresolved", tie is True)

        # ---------------------------------------------------------------
        # 6. concept_id remains the FINAL tiebreak -- determinism preserved.
        # ---------------------------------------------------------------
        conn = FakeConn(classes={11: "Clinical Finding", 22: "Clinical Finding"})
        rows = [(22, "Same", "Condition", "SNOMED"),
                (11, "Same", "Condition", "SNOMED")]
        out, _b, tie = N._rank_tier12_candidates(conn, rows, "Condition", "same")
        check("concept_id is the final tiebreak", out[0][0] == 11)
        check("total tie is unresolved", tie is True)
        out2, _b2, _t2 = N._rank_tier12_candidates(conn, list(rows), "Condition", "same")
        check("deterministic across calls", [r[0] for r in out] == [r[0] for r in out2])

        # ---------------------------------------------------------------
        # 7. A DB failure must never raise, and the basis must SAY that the
        #    ranking was made without concept_class_id -- otherwise the A/B
        #    would silently pool full-information and degraded rankings under
        #    one label and compare two different things.
        # ---------------------------------------------------------------
        conn = FakeConn(fail=True)
        rows = [(1, "A", "Condition", "SNOMED"), (2, "B", "Condition", "SNOMED")]
        out, basis, tie = N._rank_tier12_candidates(conn, rows, "Condition", "a")
        check("db failure does not raise", len(out) == 2)
        check("db failure marks the basis degraded", basis == "ranked_v1_no_class")

        # Same, but with the class lookup succeeding: full-strength basis.
        conn = FakeConn(classes={1: "Clinical Finding", 2: "Clinical Finding"})
        out, basis, _t = N._rank_tier12_candidates(conn, rows, "Condition", "a")
        check("full information marks the basis ranked_v1", basis == "ranked_v1")

        # ---------------------------------------------------------------
        # 8. Unknown class is NOT demoted below an explicitly demoted one.
        # ---------------------------------------------------------------
        check("demoted ranks below unknown",
              N._class_rank("Qualifier Value", "Condition")
              > N._class_rank("Some New Class", "Condition"))
        check("preferred ranks above unknown",
              N._class_rank("Clinical Finding", "Condition")
              < N._class_rank("Some New Class", "Condition"))
        check("missing class is unknown-ranked",
              N._class_rank(None, "Condition") == 90)
        check("unmapped label falls through to unknown",
              N._class_rank("Clinical Finding", "NoSuchLabel") == 90)

        # ---------------------------------------------------------------
        # 9. Cosine helper edge cases.
        # ---------------------------------------------------------------
        check("cosine identical == 1", abs(N._cosine([1, 0], [1, 0]) - 1.0) < 1e-9)
        check("cosine orthogonal == 0", abs(N._cosine([1, 0], [0, 1])) < 1e-9)
        check("cosine zero vector safe", N._cosine([0, 0], [1, 0]) == 0.0)
        check("cosine length mismatch safe", N._cosine([1, 0], [1, 0, 0]) == 0.0)
        check("cosine empty safe", N._cosine([], []) == 0.0)

        # ---------------------------------------------------------------
        # 10. The flag itself defaults OFF -- the whole point of shipping it
        #     this way. A regression that flips the default silently would
        #     invalidate every Stage 3 row in the database.
        # ---------------------------------------------------------------
        check("RANKED_TIER12 defaults off",
              N.RANKED_TIER12 is False or os.environ.get("CNSP_RANKED_TIER12"))
    finally:
        N.get_sapbert_embedding = original

    print(f"tier12 ranking tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
