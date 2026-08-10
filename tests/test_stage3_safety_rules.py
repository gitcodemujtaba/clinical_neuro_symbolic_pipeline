"""
tests/test_stage3_safety_rules.py

Pins the three functions that carry every clinical guarantee Stage 3 makes:
verify_citations(), combine() and route().

WHY THESE THREE, AND WHY NOW. Stage 3's output is written to a provenance
ledger and is meant to be defensible after the fact. Everything else in the
stage can degrade visibly -- a bad prompt produces a bad verdict, and a human
reading the artifact can see it. These three degrade INVISIBLY: a routing bug
does not produce a wrong verdict, it produces a correct verdict that was
allowed past the gate, and nothing in the artifact says so.

The first live run (docs/Stage3_Open_Issues.md, 2026-08-10) made this concrete.
On record 3 the ensemble agreed, both models self-reported HIGH, and
composite_confidence was 0.916 -- above AUTO_VALIDATE_THRESHOLD. The verdict
was wrong (`spirnolactone` resolved to SPIRAPRILAT, an ACE inhibitor, when the
drug is spironolactone). The ONLY thing that stopped it being auto-validated
was citation verification catching a fabricated rule_id. test_record3_* below
reproduces that case exactly, because it is the single most load-bearing
behaviour in the stage and nothing else tests it.

No models, no database, no network -- fixtures only, so this runs in CI and in
a second.

Run:  python3 -m pytest tests/test_stage3_safety_rules.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mollm_ensemble import (  # noqa: E402
    AUTO_VALIDATE_THRESHOLD,
    CITATION_CONTAINMENT_THRESHOLD,
    CONFIDENCE_SPREAD_PENALTY,
    CONFIDENCE_SPREAD_TRIGGER,
    INGESTION_AUTO,
    INGESTION_HITL,
    INGESTION_RESOLVED,
    MOLLM_RESOLVE_THRESHOLD,
    combine,
    route,
    verify_citations,
)


# ---------------------------------------------------------------- fixtures --

def mk_rule(rule_id="R1", citation_type="verbatim", excerpt=None, citation=None):
    return {
        "rule_id": rule_id,
        "citation_type": citation_type,
        "citation_verbatim_excerpt": excerpt,
        "citation": citation,
        "predicate": "TREATED_WITH",
        "source_name": "Heart failure",
        "target_name": "ACE inhibitor",
    }


def mk_retrieval(rules=None, released=None):
    return {"rules": rules or [], "suppressed_rules_released": released or []}


def mk_model(verdict, conf=0.9, model="m"):
    return {"model": model, "verdict": verdict, "logprob_confidence": conf}


def cite(rule_id, quote=""):
    return {"rule_id": rule_id, "quote": quote}


# ------------------------------------------------------ verify_citations() --

class TestVerifyCitations:
    """The hallucination detector. Its job is to distinguish a model that used
    the evidence from one that invented it."""

    def test_no_citations_passes(self):
        """An empty citation list is not a failure. A model can legitimately
        reach INSUFFICIENT_EVIDENCE without citing anything, and treating that
        as a verification failure would route every honest abstention to HITL
        for the wrong reason."""
        out = verify_citations({"cited_evidence": []}, mk_retrieval())
        assert out["citation_verified"] is True
        assert out["citations_made"] == 0
        assert out["citation_checks"] == []

    def test_missing_key_treated_as_no_citations(self):
        out = verify_citations({}, mk_retrieval())
        assert out["citation_verified"] is True
        assert out["citations_made"] == 0

    def test_fabricated_rule_id_fails(self):
        """THE case this mechanism exists for. Observed in 6 of 10 records on
        the first live run, always when the model had been shown no evidence at
        all."""
        out = verify_citations(
            {"cited_evidence": [cite("RULE_ID_1", "some quote")]},
            mk_retrieval(rules=[]))
        assert out["citation_verified"] is False
        assert out["citation_checks"][0]["reason"] == "fabricated_rule_id_not_in_evidence"

    def test_verbatim_quote_present_verifies(self):
        rule = mk_rule("R1", "verbatim",
                       excerpt="ACE inhibitors are recommended for all patients "
                               "with reduced ejection fraction.")
        out = verify_citations(
            {"cited_evidence": [cite("R1", "recommended for all patients")]},
            mk_retrieval([rule]))
        assert out["citation_verified"] is True
        assert out["citation_checks"][0]["mode"] == "strict"
        assert out["citation_checks"][0]["containment"] >= CITATION_CONTAINMENT_THRESHOLD

    def test_verbatim_quote_absent_fails(self):
        """A real rule_id with an invented quote. Subtler than a fabricated id
        and exactly what strict mode is for."""
        rule = mk_rule("R1", "verbatim", excerpt="ACE inhibitors are recommended.")
        out = verify_citations(
            {"cited_evidence": [cite("R1", "beta blockers are contraindicated")]},
            mk_retrieval([rule]))
        assert out["citation_verified"] is False
        assert out["citation_checks"][0]["reason"] == "quote_not_found_in_source"
        assert out["citation_checks"][0]["containment"] < CITATION_CONTAINMENT_THRESHOLD

    def test_recovered_excerpt_checked_strictly(self):
        rule = mk_rule("R1", "paraphrase_with_recovered_excerpt",
                       excerpt="Spironolactone reduces mortality in heart failure.")
        ok = verify_citations({"cited_evidence": [cite("R1", "reduces mortality")]},
                              mk_retrieval([rule]))
        bad = verify_citations({"cited_evidence": [cite("R1", "increases mortality")]},
                               mk_retrieval([rule]))
        assert ok["citation_verified"] is True
        assert bad["citation_verified"] is False

    def test_paraphrase_verifies_loosely(self):
        """Containment cannot be demanded of text that was never a verbatim
        quote. Loose mode degrades to 'was this rule_id actually shown', which
        still catches fabricated attribution -- the thing that matters."""
        rule = mk_rule("R1", "paraphrase", citation="Some guideline, 2019.")
        out = verify_citations(
            {"cited_evidence": [cite("R1", "anything at all")]},
            mk_retrieval([rule]))
        assert out["citation_verified"] is True
        assert out["citation_checks"][0]["mode"] == "loose"

    def test_pointer_unverifiable_is_not_citable(self):
        """No source text exists to check against, so permitting the citation
        would admit one that can never be verified either way."""
        rule = mk_rule("R1", "pointer_unverifiable")
        out = verify_citations({"cited_evidence": [cite("R1", "q")]},
                               mk_retrieval([rule]))
        assert out["citation_verified"] is False
        assert "not_citable" in out["citation_checks"][0]["reason"]

    def test_malformed_citation_fails(self):
        """Guided decoding should make this impossible, but the unguided
        fallback path in llm_client exists and must not crash here."""
        out = verify_citations({"cited_evidence": ["just a string"]}, mk_retrieval())
        assert out["citation_verified"] is False
        assert out["citation_checks"][0]["reason"] == "malformed_citation"

    def test_aggregate_is_conjunctive(self):
        """One bad citation among several good ones must fail the record. An
        'any' rule would let a model launder a fabrication behind real ones."""
        rule = mk_rule("R1", "verbatim", excerpt="ACE inhibitors are recommended.")
        out = verify_citations(
            {"cited_evidence": [cite("R1", "ACE inhibitors"), cite("GHOST", "x")]},
            mk_retrieval([rule]))
        assert out["citation_verified"] is False
        assert out["citations_made"] == 2
        assert [c["verified"] for c in out["citation_checks"]] == [True, False]

    def test_released_suppressed_rule_is_citable_but_flagged(self):
        """Evidence we deliberately handed over on request must be citable, or
        every expansion round would fail verification. Its use is surfaced
        rather than forbidden."""
        rule = mk_rule("S1", "verbatim", excerpt="Suppressed but released text.")
        out = verify_citations({"cited_evidence": [cite("S1", "released text")]},
                               mk_retrieval(rules=[], released=[rule]))
        assert out["citation_verified"] is True
        assert out["cited_suppressed_rules"] == ["S1"]

    def test_normal_rule_not_reported_as_suppressed(self):
        rule = mk_rule("R1", "verbatim", excerpt="Normal evidence text.")
        out = verify_citations({"cited_evidence": [cite("R1", "Normal evidence")]},
                               mk_retrieval([rule]))
        assert out["cited_suppressed_rules"] == []


# --------------------------------------------------------------- combine() --

class TestCombine:
    """Agreement detection and composite confidence."""

    def test_agreement_averages_confidences(self):
        out = combine([mk_model("SUPPORTED", 0.90), mk_model("SUPPORTED", 0.80)])
        assert out["ensemble_agreement"] is True
        assert out["composite_confidence"] == pytest.approx(0.85)
        assert out["confidence_basis"] == "mean_logprob_agreeing_verdicts"

    def test_disagreement_marks_composite_unusable(self):
        """The number is still reported for analysis but the basis string says
        plainly it must not drive routing. A discounted average of two opposite
        clinical verdicts is not a meaningful quantity."""
        out = combine([mk_model("SUPPORTED", 0.90), mk_model("CONTRADICTED", 0.80)])
        assert out["ensemble_agreement"] is False
        assert "not_used_for_routing" in out["confidence_basis"]

    def test_no_logprobs_yields_none_not_a_default(self):
        """Critical: a fabricated default (say 1.0) would silently inflate
        composite_confidence and could push a record past AUTO_VALIDATE on no
        evidence whatsoever. 'Unmeasured' must stay distinct from 'low'."""
        out = combine([mk_model("SUPPORTED", None), mk_model("SUPPORTED", None)])
        assert out["composite_confidence"] is None
        assert out["confidence_basis"] == "no_logprobs_available"

    def test_partial_logprobs_uses_what_exists(self):
        out = combine([mk_model("SUPPORTED", 0.80), mk_model("SUPPORTED", None)])
        assert out["composite_confidence"] == pytest.approx(0.80)

    def test_wide_spread_is_penalised(self):
        """Agreement in verdict but not in degree is weaker evidence than
        agreement in both, and averaging alone would hide the difference."""
        lo, hi = 0.50, 0.50 + CONFIDENCE_SPREAD_TRIGGER + 0.01
        out = combine([mk_model("SUPPORTED", hi), mk_model("SUPPORTED", lo)])
        assert "spread_penalty" in out["confidence_basis"]
        assert out["composite_confidence"] == pytest.approx(
            ((hi + lo) / 2) * CONFIDENCE_SPREAD_PENALTY, rel=1e-4)

    def test_narrow_spread_is_not_penalised(self):
        out = combine([mk_model("SUPPORTED", 0.90), mk_model("SUPPORTED", 0.88)])
        assert "spread_penalty" not in out["confidence_basis"]
        assert out["confidence_spread"] == pytest.approx(0.02)

    def test_single_model_has_zero_spread(self):
        out = combine([mk_model("SUPPORTED", 0.90)])
        assert out["ensemble_agreement"] is True
        assert out["confidence_spread"] == 0.0


# ----------------------------------------------------------------- route() --

VERIFIED = {"citation_verified": True}
FAILED = {"citation_verified": False}


def agreeing(conf=0.95):
    return {"ensemble_agreement": True, "composite_confidence": conf}


class TestRouteSafetyRules:
    """The three safety rules short-circuit BEFORE any threshold is consulted.
    Each test therefore pairs the safety condition with a confidence high
    enough to auto-validate, so a regression that reorders the checks fails
    here rather than in production."""

    def test_disagreement_forces_hitl_despite_high_confidence(self):
        out = route({"ensemble_agreement": False, "composite_confidence": 0.99},
                    VERIFIED, [mk_model("SUPPORTED", 0.99)])
        assert out["mollm_routing_decision"] == INGESTION_HITL
        assert out["queue_reason"] == "model_disagreement"

    def test_failed_citation_forces_hitl_despite_high_confidence(self):
        out = route(agreeing(0.99), FAILED, [mk_model("SUPPORTED", 0.99)])
        assert out["mollm_routing_decision"] == INGESTION_HITL
        assert out["queue_reason"] == "citation_verification_failed"

    def test_contradiction_never_auto_resolves(self):
        """A flagged guideline contradiction is a clinical finding for a human,
        not something to auto-accept because the model was sure."""
        out = route(agreeing(0.99), VERIFIED, [mk_model("CONTRADICTED", 0.99)])
        assert out["mollm_routing_decision"] == INGESTION_HITL
        assert out["queue_reason"] == "guideline_contradiction"

    def test_unmeasurable_confidence_forces_hitl(self):
        out = route({"ensemble_agreement": True, "composite_confidence": None},
                    VERIFIED, [mk_model("SUPPORTED", None)])
        assert out["mollm_routing_decision"] == INGESTION_HITL
        assert out["queue_reason"] == "confidence_unmeasurable"

    @pytest.mark.parametrize("verdict", ["INSUFFICIENT_EVIDENCE", "NONE_CORRECT"])
    def test_non_resolutions_go_to_hitl(self, verdict):
        out = route(agreeing(0.99), VERIFIED, [mk_model(verdict, 0.99)])
        assert out["mollm_routing_decision"] == INGESTION_HITL
        assert out["queue_reason"] == f"verdict_{verdict.lower()}"


class TestRouteThresholds:
    """Only reached once every safety rule has passed."""

    def test_high_confidence_auto_validates(self):
        out = route(agreeing(AUTO_VALIDATE_THRESHOLD + 0.01), VERIFIED,
                    [mk_model("SUPPORTED")])
        assert out["mollm_routing_decision"] == INGESTION_AUTO
        assert out["queue_reason"] is None

    def test_threshold_is_inclusive(self):
        out = route(agreeing(AUTO_VALIDATE_THRESHOLD), VERIFIED, [mk_model("SUPPORTED")])
        assert out["mollm_routing_decision"] == INGESTION_AUTO

    def test_middle_band_is_mollm_resolved(self):
        mid = (AUTO_VALIDATE_THRESHOLD + MOLLM_RESOLVE_THRESHOLD) / 2
        out = route(agreeing(mid), VERIFIED, [mk_model("RESOLVED_TO_CANDIDATE_1")])
        assert out["mollm_routing_decision"] == INGESTION_RESOLVED

    def test_below_floor_goes_to_hitl(self):
        out = route(agreeing(MOLLM_RESOLVE_THRESHOLD - 0.01), VERIFIED,
                    [mk_model("SUPPORTED")])
        assert out["mollm_routing_decision"] == INGESTION_HITL
        assert out["queue_reason"] == "below_confidence_threshold"


class TestRecord3Regression:
    """Reproduces the observed near-miss from the first live run.

    `spirnolactone` (a misspelling of spironolactone) was resolved by BOTH
    models to SPIRAPRILAT -- spirapril, an ACE inhibitor. The correct concept
    was not among the candidates at all, so NONE_CORRECT was the only correct
    verdict and it was available in the allowed vocabulary.

    Every signal intended to catch this failed: the models agreed, both
    self-reported HIGH, and composite_confidence came out at 0.916 -- above
    AUTO_VALIDATE_THRESHOLD. Citation verification alone forced HITL, because
    the model cited RULE_ID_1 having been shown no evidence at all.

    If a future change ever lets this combination auto-validate, that is a
    patient-safety regression and it must fail loudly here.
    """

    def test_wrong_but_confident_and_agreeing_is_saved_by_citation_check(self):
        models = [mk_model("RESOLVED_TO_CANDIDATE_3", 0.922471, "biomistral"),
                  mk_model("RESOLVED_TO_CANDIDATE_3", 0.910462, "openbiollm")]
        ensemble = combine(models)

        # Preconditions: every non-citation signal says "accept".
        assert ensemble["ensemble_agreement"] is True
        assert ensemble["composite_confidence"] > AUTO_VALIDATE_THRESHOLD

        citation = verify_citations(
            {"cited_evidence": [cite("RULE_ID_1", "guideline says so")]},
            mk_retrieval(rules=[]))          # nothing was retrieved -- 0 rules
        assert citation["citation_verified"] is False

        out = route(ensemble, citation, models)
        assert out["mollm_routing_decision"] == INGESTION_HITL
        assert out["queue_reason"] == "citation_verification_failed"

    def test_without_the_citation_check_it_would_have_auto_validated(self):
        """Demonstrates what was at stake, and documents that agreement plus
        confidence alone are NOT sufficient. This is the counterfactual the
        dissertation's argument for symbolic verification rests on."""
        models = [mk_model("RESOLVED_TO_CANDIDATE_3", 0.922471),
                  mk_model("RESOLVED_TO_CANDIDATE_3", 0.910462)]
        out = route(combine(models), VERIFIED, models)
        assert out["mollm_routing_decision"] == INGESTION_AUTO
