"""
tests/test_collapse_hierarchy_duplicates.py —
src.normalization.tier_retrieval._collapse_hierarchy_duplicates()'s
2026-08-20 fix: a curated match_basis (verified_lab_test_alias/
verified_brand_alias/lab_procedure_preferred) must win its SNOMED-
hierarchy root unconditionally, never lose to a plain semantic_similarity
sibling merely because that sibling scored higher. Found live via the
lab-abbreviation cold-start backfill ("HCO3" losing to its own SNOMED
parent concept) -- see the fix's own tier_retrieval.py comment for the
full story.

Uses a fake connection (no live DB needed) since the only thing this
function's DB access does is fetch "Is a"/"Subsumes" edges among the
candidate set -- fully controllable with a scripted fake.

Run: python3 -m pytest tests/test_collapse_hierarchy_duplicates.py -v
"""
import sys

from src.normalization.tier_retrieval import _collapse_hierarchy_duplicates


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Scripted `conn.sql(...)` -- returns `edges` regardless of the
    actual query text, since this function only ever issues one shape of
    query (fetch "Is a"/"Subsumes" rows among the given candidate ids)."""
    def __init__(self, edges):
        self.edges = edges

    def sql(self, query, params=None):
        return _FakeResult(self.edges)


def _cand(concept_id, name, similarity, match_basis=None):
    return {"omop_concept_id": concept_id, "concept_name": name,
           "similarity_score": similarity, "match_basis": match_basis}


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # The exact bug scenario: a curated alias with LOWER raw similarity
    # loses to its own SNOMED parent/child with HIGHER raw similarity,
    # under the OLD "highest similarity wins" rule -- must now win instead.
    # ======================================================================
    curated_lower = _cand(4194291, "Blood bicarbonate measurement", 0.7441,
                          match_basis="verified_lab_test_alias")
    plain_higher = _cand(4227915, "Bicarbonate measurement", 0.8857,
                         match_basis="semantic_similarity")
    conn = _FakeConn([(4194291, 4227915)])  # connected by an "Is a" edge
    result = _collapse_hierarchy_duplicates(conn, [curated_lower, plain_higher])
    check("curated candidate wins its root even with LOWER raw similarity "
         "than the plain sibling (the actual HCO3 bug)",
          len(result) == 1 and result[0]["omop_concept_id"] == 4194291)

    # ======================================================================
    # Reversed order in the input list -- must not depend on iteration order.
    # ======================================================================
    conn2 = _FakeConn([(4194291, 4227915)])
    result2 = _collapse_hierarchy_duplicates(conn2, [plain_higher, curated_lower])
    check("curated candidate wins regardless of input order",
          len(result2) == 1 and result2[0]["omop_concept_id"] == 4194291)

    # ======================================================================
    # Two curated candidates connected -- falls back to similarity between
    # them (no regression to the original tie-break rule when both sides
    # are curated).
    # ======================================================================
    curated_a = _cand(100, "A", 0.70, match_basis="verified_lab_test_alias")
    curated_b = _cand(200, "B", 0.90, match_basis="verified_brand_alias")
    conn3 = _FakeConn([(100, 200)])
    result3 = _collapse_hierarchy_duplicates(conn3, [curated_a, curated_b])
    check("between two curated candidates, higher similarity still wins "
         "(no curated-vs-curated special case needed)",
          len(result3) == 1 and result3[0]["omop_concept_id"] == 200)

    # ======================================================================
    # Original behavior preserved: two PLAIN candidates, highest similarity
    # still wins (this is the pre-existing, still-correct rule).
    # ======================================================================
    plain_a = _cand(300, "C", 0.60, match_basis="semantic_similarity")
    plain_b = _cand(400, "D", 0.85, match_basis="semantic_similarity")
    conn4 = _FakeConn([(300, 400)])
    result4 = _collapse_hierarchy_duplicates(conn4, [plain_a, plain_b])
    check("between two plain candidates, highest similarity still wins "
         "(original behavior unchanged)",
          len(result4) == 1 and result4[0]["omop_concept_id"] == 400)

    # ======================================================================
    # No hierarchy edge at all -- both survive untouched, regardless of
    # match_basis.
    # ======================================================================
    conn5 = _FakeConn([])
    result5 = _collapse_hierarchy_duplicates(conn5, [curated_lower, plain_higher])
    check("no hierarchy edge -- both candidates survive, curated fix "
         "irrelevant here (nothing to collapse)",
          len(result5) == 2)

    # ======================================================================
    # match_basis of None (e.g. exact_text/synonym default) never treated
    # as curated.
    # ======================================================================
    none_basis = _cand(500, "E", 0.60, match_basis=None)
    plain_higher_2 = _cand(600, "F", 0.80, match_basis="semantic_similarity")
    conn6 = _FakeConn([(500, 600)])
    result6 = _collapse_hierarchy_duplicates(conn6, [none_basis, plain_higher_2])
    check("match_basis=None is never treated as curated -- ordinary "
         "highest-similarity rule applies",
          len(result6) == 1 and result6[0]["omop_concept_id"] == 600)

    print(f"collapse-hierarchy-duplicates tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_collapse_hierarchy_duplicates():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
