"""
tests/test_mollm_tier_calibrator.py -- src/mollm_tier_calibrator.py's
ConsensusCalibrator and its pure feature-extraction functions.

Pure-logic tests for featurize()/build_feature_context()/
count_prior_confirmations(), which need no DB and no scikit-learn. A
handful of fit()/score() round-trip tests DO import scikit-learn (via
ConsensusCalibrator.fit()) -- if that's unavailable in a given environment
those specific checks are skipped, not failed, matching this module's own
"scikit-learn is optional until you actually fit/load" contract.

Run: python3 -m pytest tests/test_mollm_tier_calibrator.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mollm_tier_calibrator import (  # noqa: E402
    FEATURE_NAMES, build_feature_context, count_prior_confirmations,
    featurize, usable_votes, ConsensusCalibrator,
)


class FakeConn:
    """Backs mollm_tier_gate_decisions/hitl_review_queue with an in-memory
    row count, just enough to exercise count_prior_confirmations()'s two
    queries without a real DuckDB connection."""

    def __init__(self, auto_count=0, hitl_count=0, raise_on=None):
        self.auto_count = auto_count
        self.hitl_count = hitl_count
        self.raise_on = raise_on or set()  # {"auto"} and/or {"hitl"}
        self.queries = []

    def execute(self, query, params):
        self.queries.append(query)
        if "mollm_tier_gate_decisions" in query:
            if "auto" in self.raise_on:
                raise RuntimeError("simulated DB error")
            return _Result(self.auto_count)
        if "hitl_review_queue" in query:
            if "hitl" in self.raise_on:
                raise RuntimeError("simulated DB error")
            return _Result(self.hitl_count)
        raise AssertionError(f"unexpected query: {query!r}")


class _Result:
    def __init__(self, count):
        self._count = count

    def fetchone(self):
        return (self._count,)


def _entity(**overrides):
    base = {
        "candidates": [{"similarity_score": 0.9}],
        "match_tier": "3 (Semantic)",
        "is_ambiguous": False,
        "domain_conflict": False,
        "normalized_from": "expanded",
        "expansion_ambiguous": False,
    }
    base.update(overrides)
    return base


def _vote(verdict, confidence=0.9, degenerate=False):
    return {"verdict": verdict, "logprob_confidence": confidence,
            "degenerate_generation": degenerate}


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # usable_votes
    # ======================================================================
    votes = [_vote("SUPPORTED_1"), _vote("ERROR"), _vote("NONE_CORRECT", degenerate=True),
             _vote("RE_RANK_TO_CANDIDATE_2")]
    usable = usable_votes(votes)
    check("usable_votes drops ERROR verdicts", all(v.get("verdict") != "ERROR" for v in usable))
    check("usable_votes drops degenerate generations",
          all(not v.get("degenerate_generation") for v in usable))
    check("usable_votes keeps the two genuinely usable votes", len(usable) == 2)
    check("usable_votes on empty/None input returns []",
          usable_votes([]) == [] and usable_votes(None) == [])

    # ======================================================================
    # featurize -- vote-pattern features
    # ======================================================================
    ctx = build_feature_context(
        _entity(),
        [_vote("SUPPORTED_1", 0.9), _vote("SUPPORTED_1", 0.8), _vote("NONE_CORRECT", 0.3)])
    vec = featurize(ctx)
    values = dict(zip(FEATURE_NAMES, vec))
    check("featurize returns exactly len(FEATURE_NAMES) values", len(vec) == len(FEATURE_NAMES))
    check("frac_supported_1 correctly computed (2 of 3 usable votes)",
          abs(values["frac_supported_1"] - 2 / 3) < 1e-9)
    check("frac_none_correct correctly computed (1 of 3)",
          abs(values["frac_none_correct"] - 1 / 3) < 1e-9)
    check("frac_usable_votes is 1.0 when all 3 models produced a usable vote",
          values["frac_usable_votes"] == 1.0)
    check("mean_logprob_confidence averages the usable votes only",
          abs(values["mean_logprob_confidence"] - (0.9 + 0.8 + 0.3) / 3) < 1e-9)
    check("min_logprob_confidence is the weakest vote", values["min_logprob_confidence"] == 0.3)
    check("confidence_spread is max-min", abs(values["confidence_spread"] - 0.6) < 1e-9)

    # RE_RANK agreement: two models pick the SAME alternate candidate.
    ctx2 = build_feature_context(
        _entity(),
        [_vote("RE_RANK_TO_CANDIDATE_3"), _vote("RE_RANK_TO_CANDIDATE_3"),
         _vote("RE_RANK_TO_CANDIDATE_2")])
    values2 = dict(zip(FEATURE_NAMES, featurize(ctx2)))
    check("frac_rerank_same_target counts only the AGREEING re-rank votes (2 of 3, not 3 of 3)",
          abs(values2["frac_rerank_same_target"] - 2 / 3) < 1e-9)

    # No usable votes at all -- every vote-derived feature must default to 0.0, no crash.
    ctx3 = build_feature_context(_entity(), [_vote("ERROR"), _vote("ERROR")])
    values3 = dict(zip(FEATURE_NAMES, featurize(ctx3)))
    check("zero usable votes -> vote-derived features all default to 0.0, no crash/NaN",
          values3["frac_supported_1"] == 0.0 and values3["mean_logprob_confidence"] == 0.0
          and values3["confidence_spread"] == 0.0)

    # ======================================================================
    # featurize -- entity/provenance features
    # ======================================================================
    ctx4 = build_feature_context(
        _entity(match_tier="1 (Exact)", is_ambiguous=True, domain_conflict=True,
               normalized_from="value_stripped_from_expanded:WBC (upgraded_from_0 (Failed))",
               expansion_ambiguous=True),
        [_vote("SUPPORTED_1")])
    values4 = dict(zip(FEATURE_NAMES, featurize(ctx4)))
    check("match_tier_is_exact_or_synonym true for Tier 1", values4["match_tier_is_exact_or_synonym"] == 1.0)
    check("is_ambiguous carried through", values4["is_ambiguous"] == 1.0)
    check("domain_conflict carried through", values4["domain_conflict"] == 1.0)
    check("resolved_via_value_stripped_fallback detected from normalized_from",
          values4["resolved_via_value_stripped_fallback"] == 1.0)
    check("expansion_ambiguous carried through", values4["expansion_ambiguous"] == 1.0)

    for suffix, feature_name in [("+acronym_mollm", "resolved_via_acronym_escalation"),
                                 ("+acronym_cache", "resolved_via_acronym_escalation"),
                                 ("original_after_expanded_failed",
                                  "resolved_via_original_text_fallback")]:
        ctx5 = build_feature_context(_entity(normalized_from=f"expanded{suffix}"), [_vote("SUPPORTED_1")])
        v5 = dict(zip(FEATURE_NAMES, featurize(ctx5)))
        check(f"{feature_name} detected for normalized_from containing {suffix!r}",
              v5[feature_name] == 1.0)

    ctx6 = build_feature_context(_entity(candidates=[]), [_vote("SUPPORTED_1")])
    v6 = dict(zip(FEATURE_NAMES, featurize(ctx6)))
    check("no candidates -> top_candidate_similarity_score defaults to 0.0, no crash",
          v6["top_candidate_similarity_score"] == 0.0)

    # prior_confirmation_count scaling: capped at 10, scaled to [0,1].
    ctx7 = build_feature_context(_entity(), [_vote("SUPPORTED_1")], prior_confirmation_count=25)
    v7 = dict(zip(FEATURE_NAMES, featurize(ctx7)))
    check("prior_confirmation_count is capped at 10 before scaling (25 -> 1.0, not 2.5)",
          v7["prior_confirmation_count"] == 1.0)

    # ======================================================================
    # count_prior_confirmations
    # ======================================================================
    check("conn=None -> 0, no crash", count_prior_confirmations(None, "ED", 123) == 0)
    check("missing entity_text -> 0", count_prior_confirmations(FakeConn(5, 5), "", 123) == 0)
    check("missing concept_id -> 0", count_prior_confirmations(FakeConn(5, 5), "ED", None) == 0)

    conn = FakeConn(auto_count=3, hitl_count=2)
    check("sums AUTO + HITL confirmation counts",
          count_prior_confirmations(conn, "ED", 123) == 5)

    conn_partial_failure = FakeConn(auto_count=3, hitl_count=2, raise_on={"hitl"})
    check("a failure in ONE query still returns the other query's count, not 0",
          count_prior_confirmations(conn_partial_failure, "ED", 123) == 3)

    conn_both_fail = FakeConn(raise_on={"auto", "hitl"})
    check("both queries failing -> 0, no exception propagates",
          count_prior_confirmations(conn_both_fail, "ED", 123) == 0)

    # ======================================================================
    # ConsensusCalibrator -- untrained state, fit/score, save/load, leakage
    # ======================================================================
    c = ConsensusCalibrator()
    check("a fresh calibrator is untrained", c.model is None)
    check("untrained score() returns None, never raises", c.score(ctx) is None)

    try:
        import sklearn  # noqa: F401
        have_sklearn = True
    except ImportError:
        have_sklearn = False

    if have_sklearn:
        import random
        random.seed(0)
        contexts, labels = [], []
        for i in range(120):
            strong = i % 2 == 0
            votes_i = [_vote("SUPPORTED_1" if strong else "NONE_CORRECT",
                             0.9 if strong else 0.4) for _ in range(3)]
            contexts.append(build_feature_context(_entity(), votes_i,
                                                  prior_confirmation_count=5 if strong else 0))
            labels.append(1 if strong else 0)

        check("fit() below min_examples raises",
              _raises(lambda: ConsensusCalibrator().fit(contexts[:10], labels[:10],
                                                        min_examples=100)))
        check("fit() with mismatched lengths raises",
              _raises(lambda: ConsensusCalibrator().fit(contexts, labels[:5], min_examples=5)))
        check("fit() on single-class labels raises",
              _raises(lambda: ConsensusCalibrator().fit(contexts, [1] * len(contexts),
                                                        min_examples=5)))

        c2 = ConsensusCalibrator()
        c2.fit(contexts, labels, min_examples=100)
        check("fit() succeeds and records n_training_examples",
              c2.n_training_examples == len(contexts))
        strong_score = c2.score(contexts[0])
        weak_score = c2.score(contexts[1])
        check("a trained calibrator scores the strong-consensus example higher "
              "than the weak-consensus one",
              strong_score is not None and weak_score is not None
              and strong_score > weak_score)

        # save/load round trip + leakage guard
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            c2.save(path, training_note_ids=["note-A", "note-B"], training_split="train",
                   code_version="test-v1")
            loaded = ConsensusCalibrator.load(path)
            check("a loaded calibrator scores identically to the one that was saved",
                  loaded.score(contexts[0]) == c2.score(contexts[0]))
            check("training_note_ids survive the save/load round trip",
                  loaded.training_note_ids == ["note-A", "note-B"])

            check("trained_on_any_of finds real overlap",
                  c2.trained_on_any_of(["note-A", "note-Z"]) == ["note-A"])
            check("trained_on_any_of returns [] for no overlap",
                  c2.trained_on_any_of(["note-X", "note-Y"]) == [])

            leaked = ConsensusCalibrator.load(path, scoring_note_ids=["note-A"],
                                              refuse_on_leakage=True)
            check("load() refuses (degrades to untrained) on detected leakage by default",
                  leaked.model is None)

            not_refused = ConsensusCalibrator.load(path, scoring_note_ids=["note-A"],
                                                    refuse_on_leakage=False)
            check("refuse_on_leakage=False keeps the model usable despite overlap",
                  not_refused.model is not None)

            clean = ConsensusCalibrator.load(path, scoring_note_ids=["note-Z"])
            check("load() with no overlapping notes keeps the model usable",
                  clean.model is not None)
        finally:
            os.unlink(path)

        check("load() of a nonexistent file returns an untrained instance, not an exception",
              ConsensusCalibrator.load("/nonexistent/path/x.pkl").model is None)
    else:
        print("  (scikit-learn not installed -- skipping fit/score/save/load checks)")

    print(f"mollm-tier-calibrator tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def _raises(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


def test_mollm_tier_calibrator():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
