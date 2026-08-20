"""
tests/test_guideline_evidence.py — src/guideline_evidence.py's name+type
matching against the real guideline-triplet corpus (src.retrieval.
GuidelineIndex), and the off-by-default gating.

Real (not mocked) GuidelineIndex load throughout -- file-backed, loads in
well under a second, no DB/model dependency.

Run: python3 -m pytest tests/test_guideline_evidence.py -v
"""
import sys

from src.guideline_evidence import (
    GUIDELINE_EVIDENCE_ENABLED,
    _type_compatible,
    get_guideline_index,
    guideline_evidence_for_candidates,
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
    # off-by-default flag
    # ======================================================================
    check("GUIDELINE_EVIDENCE_ENABLED defaults to False (env var unset in "
         "this test process)",
          GUIDELINE_EVIDENCE_ENABLED is False)

    # ======================================================================
    # get_guideline_index -- real, file-backed load
    # ======================================================================
    idx = get_guideline_index()
    check("real corpus loads with a non-trivial number of nodes",
          idx.stats["nodes"] > 1000)
    check("get_guideline_index() returns the same singleton on repeated calls",
          get_guideline_index() is idx)

    # ======================================================================
    # _type_compatible -- soft filter
    # ======================================================================
    check("Condition node type is compatible with Condition domain",
          _type_compatible("Condition", "Condition"))
    check("Finding node type is compatible with BOTH Condition and Observation",
          _type_compatible("Finding", "Condition") and _type_compatible("Finding", "Observation"))
    check("Medication node type is compatible with Drug domain",
          _type_compatible("Medication", "Drug"))
    check("Medication node type is NOT compatible with Procedure domain",
          not _type_compatible("Medication", "Procedure"))
    check("an unrecognized node type never false-excludes (no known constraint)",
          _type_compatible("Acuity", "Condition") and _type_compatible("Acuity", "Drug"))
    check("a candidate with no domain_id never false-excludes",
          _type_compatible("Medication", None))

    # ======================================================================
    # guideline_evidence_for_candidates -- real corpus, real matches
    # ======================================================================
    real_hit = [{"index": 1, "candidate": {"concept_name": "Anemia", "domain_id": "Condition",
                                           "omop_concept_id": 1}}]
    evidence = guideline_evidence_for_candidates(idx, real_hit)
    check("a real, name-matched, type-compatible candidate (Anemia) produces "
         "non-empty evidence text",
          bool(evidence))
    check("evidence text names the matched candidate's own index",
          "[1]" in evidence)
    check("evidence text carries the framing header (evidence to weigh, not decide)",
          "OFFICIAL GUIDELINE EVIDENCE" in evidence and "does not decide the answer" in evidence)

    no_hit = [{"index": 1, "candidate": {"concept_name": "Xyzzyplugh Nonexistent Concept 12345",
                                         "domain_id": "Condition", "omop_concept_id": 2}}]
    check("a candidate with no name match returns empty string, not None",
          guideline_evidence_for_candidates(idx, no_hit) == "")

    mismatched_type = [{"index": 1, "candidate": {"concept_name": "Anemia",
                                                   "domain_id": "Drug", "omop_concept_id": 3}}]
    check("a name match with an incompatible domain (Anemia found under "
         "Drug domain -- nonsensical) is filtered out by the type check",
          guideline_evidence_for_candidates(idx, mismatched_type) == "")

    no_name = [{"index": 1, "candidate": {"domain_id": "Condition", "omop_concept_id": 4}}]
    check("a candidate with no concept_name at all does not crash",
          guideline_evidence_for_candidates(idx, no_name) == "")

    check("empty candidate list returns empty string, no crash",
          guideline_evidence_for_candidates(idx, []) == "")

    print(f"guideline-evidence tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_guideline_evidence():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
