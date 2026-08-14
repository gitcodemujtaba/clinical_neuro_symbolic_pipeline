"""
tests/test_reasoning_verdict_mismatch.py — the 2026-08-13 reasoning/verdict
consistency check (src/mollm_ensemble.py reasoning_verdict_mismatch, wired
into route() as a fourth hard safety rule).

WHAT THIS PROTECTS. Hand-tracing evaluation/stage2b_cal_eval.py's
INTRODUCED_ERROR rows (docs/2026-08-13_Implementation_Verification.md
follow-up) found note 11838076-DS-20, entity 'aortic valve leaflets': BOTH
ensemble models' free-text `reasoning` named candidate [2] ("Structure of
cusp of aortic valve") verbatim, then independently voted
RESOLVED_TO_CANDIDATE_3 -- a DIFFERENT candidate. Traced to _format_candidates()
in src/mollm_ensemble.py: candidate [3]'s own is-a hint line repeated
candidate [2]'s name verbatim, which a guided-JSON decode's structured
`verdict` field can lock onto independently of what the free-text `reasoning`
generated in the same pass actually argued for.

The check is intentionally NOT a fix to the model's behavior (nothing here
changes what the LLM outputs) -- it is a safety net that catches the
resulting self-inconsistent verdict and routes it to a human, the same
posture as the citation-verification and model-disagreement rules it sits
alongside in route().

Run:  python3 tests/test_reasoning_verdict_mismatch.py
"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.mollm_ensemble import (  # noqa: E402
    reasoning_verdict_mismatch, route,
)

AGREE = {"ensemble_agreement": True, "composite_confidence": 0.95}
VERIFIED = {"citation_verified": True}

AORTIC_CANDIDATES = [
    {"concept_name": "Structure of cardiac valve leaflet"},
    {"concept_name": "Structure of cusp of aortic valve"},
    {"concept_name": "Entire cusp of aortic valve"},
]
# The actual reasoning text recorded in mollm_decisions for this entity.
MODEL_A_REASONING = (
    "The context and section both indicate the patient is having an "
    "echocardiogram done. The entity is in the conclusion, which is a "
    "summary of the findings. The closest candidate is 'Structure of cusp "
    "of aortic valve' which is a sub-concept of the entity. The other "
    "candidates are not the correct concept, but are related concepts."
)
MODEL_B_REASONING = (
    "The entity 'aortic valve leaflets' is most consistent with the concept "
    "'Structure of cusp of aortic valve' as it is a specific part of the "
    "aortic valve mentioned in the context."
)


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ==================================================================
    # reasoning_verdict_mismatch() -- unit behavior
    # ==================================================================

    # 1. THE REAL CASE. Reasoning names candidate [2] verbatim; verdict picks
    #    candidate [3].
    r = reasoning_verdict_mismatch(
        "RESOLVED_TO_CANDIDATE_3", MODEL_A_REASONING, AORTIC_CANDIDATES)
    check("aortic valve replay: checked", r["checked"] is True)
    check("aortic valve replay: mismatch detected", r["mismatch"] is True)
    check("aortic valve replay: names candidate 2", r["reasoning_names_candidate"] == 2)
    check("aortic valve replay: verdict was candidate 3", r["verdict_candidate"] == 3)

    r2 = reasoning_verdict_mismatch(
        "RESOLVED_TO_CANDIDATE_3", MODEL_B_REASONING, AORTIC_CANDIDATES)
    check("aortic valve replay (2nd model): mismatch detected", r2["mismatch"] is True)

    # 2. CONSISTENT CASE: reasoning names the SAME candidate the verdict picks.
    r = reasoning_verdict_mismatch(
        "RESOLVED_TO_CANDIDATE_2",
        "The best match is 'Structure of cusp of aortic valve'.",
        AORTIC_CANDIDATES)
    check("consistent reasoning is checked but not flagged",
          r["checked"] is True and r["mismatch"] is False)

    # 3. A reasoning text that mentions multiple candidates to RULE THEM OUT,
    #    but argues most strongly for the one actually chosen, must not
    #    false-positive.
    r = reasoning_verdict_mismatch(
        "RESOLVED_TO_CANDIDATE_1",
        "This is not 'Structure of cusp of aortic valve' or 'Entire cusp of "
        "aortic valve'. Structure of cardiac valve leaflet Structure of "
        "cardiac valve leaflet is the correct match here.",
        AORTIC_CANDIDATES)
    check("chosen candidate scoring highest is not a false positive",
          r["mismatch"] is False)

    # 4. NOT APPLICABLE cases: no sibling candidate name a mismatch could
    #    point to. checked=False, distinct from checked=True/mismatch=False.
    check("NONE_CORRECT is not applicable",
          reasoning_verdict_mismatch(
              "NONE_CORRECT", "none of these fit", AORTIC_CANDIDATES
          )["checked"] is False)
    check("contradiction-mode verdict is not applicable",
          reasoning_verdict_mismatch(
              "SUPPORTED", "the evidence supports this", AORTIC_CANDIDATES
          )["checked"] is False)
    check("missing reasoning is not applicable",
          reasoning_verdict_mismatch(
              "RESOLVED_TO_CANDIDATE_1", None, AORTIC_CANDIDATES
          )["checked"] is False)
    check("single-candidate list is not applicable",
          reasoning_verdict_mismatch(
              "RESOLVED_TO_CANDIDATE_1", "yes this one", AORTIC_CANDIDATES[:1]
          )["checked"] is False)
    check("out-of-range verdict index is not applicable",
          reasoning_verdict_mismatch(
              "RESOLVED_TO_CANDIDATE_9", "yes this one", AORTIC_CANDIDATES
          )["checked"] is False)

    # 5. A reasoning text that never names ANY candidate concept must not
    #    flag -- there is nothing for it to be inconsistent WITH.
    r = reasoning_verdict_mismatch(
        "RESOLVED_TO_CANDIDATE_1",
        "The context strongly supports this reading given the section.",
        AORTIC_CANDIDATES)
    check("reasoning naming no candidate is not flagged", r["mismatch"] is False)

    # ==================================================================
    # route() -- wired as a fourth hard safety rule
    # ==================================================================

    def _models_with_check(verdict, reasoning, candidates):
        chk = reasoning_verdict_mismatch(verdict, reasoning, candidates)
        return [{"model": "model_a", "verdict": verdict,
                 "reasoning_verdict_check": chk},
                {"model": "model_b", "verdict": verdict,
                 "reasoning_verdict_check": chk}]

    # 6. A mismatched verdict is forced to HITL even at high confidence and
    #    full agreement -- same posture as the other three hard rules.
    models = _models_with_check(
        "RESOLVED_TO_CANDIDATE_3", MODEL_A_REASONING, AORTIC_CANDIDATES)
    r = route(AGREE, VERIFIED, models)
    check("mismatch forces HITL despite high confidence",
          r["mollm_routing_decision"] == "HITL_REQUIRED")
    check("mismatch has its own queue_reason",
          r["queue_reason"] == "reasoning_verdict_mismatch")
    check("routing_basis names the models and candidates involved",
          "model_a" in r["routing_basis"] and "candidate 2" in r["routing_basis"])

    # 7. A consistent verdict is NOT blocked by this rule. Uses candidate 1
    #    (a confirmation of Stage 2b's own top-1) so the separate P2.1
    #    override gate -- which requires a calibrator + guideline evidence
    #    for any candidate 2/3 pick, tested on its own in
    #    tests/test_override_gate.py -- does not also fire here and confound
    #    this check.
    models = _models_with_check(
        "RESOLVED_TO_CANDIDATE_1",
        "The best match is 'Structure of cardiac valve leaflet'.",
        AORTIC_CANDIDATES)
    r = route(AGREE, VERIFIED, models)
    check("consistent verdict is not blocked",
          r["queue_reason"] != "reasoning_verdict_mismatch")
    check("consistent verdict still auto-validates",
          r["mollm_routing_decision"] == "AUTO_VALIDATED")

    # 8. The other three hard safety rules still short-circuit BEFORE this
    #    one -- a disagreement must not be relabeled as a reasoning mismatch.
    r = route({"ensemble_agreement": False, "composite_confidence": 0.9},
              VERIFIED, models)
    check("disagreement still wins over the mismatch check",
          r["queue_reason"] == "model_disagreement")

    # 9. Full end-to-end replay: note 11838076-DS-20, 'aortic valve leaflets',
    #    both models' actual recorded reasoning and verdict.
    models = [
        {"model": "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM",
         "verdict": "RESOLVED_TO_CANDIDATE_3",
         "reasoning_verdict_check": reasoning_verdict_mismatch(
             "RESOLVED_TO_CANDIDATE_3", MODEL_A_REASONING, AORTIC_CANDIDATES)},
        {"model": "bartowski/OpenBioLLM-Llama3-8B-AWQ",
         "verdict": "RESOLVED_TO_CANDIDATE_3",
         "reasoning_verdict_check": reasoning_verdict_mismatch(
             "RESOLVED_TO_CANDIDATE_3", MODEL_B_REASONING, AORTIC_CANDIDATES)},
    ]
    r = route({"ensemble_agreement": True, "composite_confidence": 0.93},
              VERIFIED, models, grounding_basis="model_terminology_knowledge")
    check("aortic valve leaflets replay: routed to HITL",
          r["mollm_routing_decision"] == "HITL_REQUIRED")
    check("aortic valve leaflets replay: caught by the mismatch rule specifically",
          r["queue_reason"] == "reasoning_verdict_mismatch")

    print(f"reasoning-verdict-mismatch tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
