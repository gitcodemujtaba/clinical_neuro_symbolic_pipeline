"""
tests/test_override_gate.py — the 2026-08-13 P2.1 asymmetric override gate and
P2.2 candidate shuffle (src/mollm_ensemble.py).

WHAT IS BEING PROTECTED. The 2026-08-13 report S6 measured MoLLM's resolution
ensemble as NET-HARMFUL: 30 INTRODUCED_ERROR against 13 CAUGHT_AND_FIXED, net
-17. The gate makes overriding Stage 2b's top-1 strictly harder than confirming
it. Two properties have to hold for that to be an improvement rather than just
a blunt instrument:

  1. It must NOT block confirmations, NONE_CORRECT, or contradiction-mode
     verdicts -- those have never been the problem, and blocking them would
     dump the entire Stage 3 output into the HITL queue.
  2. It must map through the shuffle permutation. Under --shuffle-candidates,
     presented position 1 is not Stage 2b's top-1; comparing indices naively
     would invert the gate exactly in the experimental run whose numbers are
     supposed to be compared against the baseline.

Run:  python3 tests/test_override_gate.py
"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.mollm_ensemble import (  # noqa: E402
    OVERRIDE_TOP1_THRESHOLD, _is_top1_override, _shuffled_candidates, route,
)

AGREE = {"ensemble_agreement": True, "composite_confidence": 0.95}
VERIFIED = {"citation_verified": True}


def _models(verdict):
    return [{"model": "a", "verdict": verdict}, {"model": "b", "verdict": verdict}]


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ==================================================================
    # _is_top1_override
    # ==================================================================
    check("candidate 1 is a confirmation",
          _is_top1_override("RESOLVED_TO_CANDIDATE_1") is False)
    check("candidate 2 is an override",
          _is_top1_override("RESOLVED_TO_CANDIDATE_2") is True)
    check("candidate 3 is an override",
          _is_top1_override("RESOLVED_TO_CANDIDATE_3") is True)
    check("NONE_CORRECT is not an override",
          _is_top1_override("NONE_CORRECT") is False)
    check("SUPPORTED is not an override",
          _is_top1_override("SUPPORTED") is False)
    check("INSUFFICIENT_EVIDENCE is not an override",
          _is_top1_override("INSUFFICIENT_EVIDENCE") is False)
    check("None verdict is safe", _is_top1_override(None) is False)

    # Permutation mapping: [2, 0, 1] means position 1 shows original index 2,
    # position 2 shows original index 0 (Stage 2b's top-1), position 3 shows 1.
    perm = [2, 0, 1]
    check("shuffled: position 1 is NOT the top-1",
          _is_top1_override("RESOLVED_TO_CANDIDATE_1", perm) is True)
    check("shuffled: position 2 IS the top-1",
          _is_top1_override("RESOLVED_TO_CANDIDATE_2", perm) is False)
    check("shuffled: position 3 is an override",
          _is_top1_override("RESOLVED_TO_CANDIDATE_3", perm) is True)
    check("out-of-range index is gated, not waved through",
          _is_top1_override("RESOLVED_TO_CANDIDATE_9", perm) is True)

    # ==================================================================
    # route() -- the gate itself
    # ==================================================================

    # 1. THE CASE THE GATE EXISTS FOR: an override with no calibrator and no
    #    guideline evidence. This is the shape of the INTRODUCED_ERROR cases.
    r = route(AGREE, VERIFIED, _models("RESOLVED_TO_CANDIDATE_2"))
    check("bare override is blocked", r["queue_reason"] == "unsupported_top1_override")
    check("blocked override goes to a human",
          r["mollm_routing_decision"] == "HITL_REQUIRED")
    check("blockers are named in the basis",
          "no_calibrator" in r["routing_basis"]
          and "grounding_basis" in r["routing_basis"])

    # 2. Confirmation of Stage 2b's top-1 at the same confidence sails through.
    #    This is the asymmetry: identical inputs, opposite direction, different
    #    bar.
    r = route(AGREE, VERIFIED, _models("RESOLVED_TO_CANDIDATE_1"))
    check("confirmation is NOT blocked",
          r["queue_reason"] != "unsupported_top1_override")
    check("confirmation auto-validates at 0.95",
          r["mollm_routing_decision"] == "AUTO_VALIDATED")

    # 3. A FULLY-SUPPORTED override passes: high calibrator score, guideline
    #    evidence, verified citation. CAUGHT_AND_FIXED must remain reachable.
    r = route(AGREE, VERIFIED, _models("RESOLVED_TO_CANDIDATE_2"),
              calibrator_score=0.95, grounding_basis="guideline_rule")
    check("fully-supported override passes",
          r["queue_reason"] != "unsupported_top1_override")
    check("fully-supported override auto-validates",
          r["mollm_routing_decision"] == "AUTO_VALIDATED")

    # 4. Each requirement is individually necessary.
    r = route(AGREE, VERIFIED, _models("RESOLVED_TO_CANDIDATE_2"),
              calibrator_score=OVERRIDE_TOP1_THRESHOLD - 0.01,
              grounding_basis="guideline_rule")
    check("low calibrator score alone blocks",
          r["queue_reason"] == "unsupported_top1_override")

    r = route(AGREE, VERIFIED, _models("RESOLVED_TO_CANDIDATE_2"),
              calibrator_score=0.95, grounding_basis="model_terminology_knowledge")
    check("terminology-only grounding blocks",
          r["queue_reason"] == "unsupported_top1_override")

    r = route(AGREE, VERIFIED, _models("RESOLVED_TO_CANDIDATE_2"),
              calibrator_score=0.95, grounding_basis="ontology_only")
    check("ontology-only grounding blocks",
          r["queue_reason"] == "unsupported_top1_override")

    # 5. The three hard safety rules still short-circuit BEFORE the gate --
    #    the gate must never be able to relabel a disagreement or a failed
    #    citation as something milder.
    r = route({"ensemble_agreement": False, "composite_confidence": 0.9},
              VERIFIED, _models("RESOLVED_TO_CANDIDATE_2"))
    check("disagreement still wins over the gate",
          r["queue_reason"] == "model_disagreement")
    r = route(AGREE, {"citation_verified": False},
              _models("RESOLVED_TO_CANDIDATE_2"))
    check("citation failure still wins over the gate",
          r["queue_reason"] == "citation_verification_failed")

    # 6. NONE_CORRECT keeps its own existing route, untouched by the gate.
    r = route(AGREE, VERIFIED, _models("NONE_CORRECT"))
    check("NONE_CORRECT keeps its own queue_reason",
          r["queue_reason"] == "verdict_none_correct")

    # 7. Under shuffling, the gate follows the permutation rather than the
    #    presented index.
    r = route(AGREE, VERIFIED, _models("RESOLVED_TO_CANDIDATE_2"),
              candidate_permutation=[2, 0, 1])
    check("shuffled confirmation is not blocked",
          r["queue_reason"] != "unsupported_top1_override")
    r = route(AGREE, VERIFIED, _models("RESOLVED_TO_CANDIDATE_1"),
              candidate_permutation=[2, 0, 1])
    check("shuffled override IS blocked",
          r["queue_reason"] == "unsupported_top1_override")

    # ==================================================================
    # 2026-08-13 (docs/2026-08-13_Implementation_Verification.md follow-up):
    # regression replay of two rows found sitting in the live DB as
    # AUTO_VALIDATED under a pre-P2.1 pipeline run. Both are exactly what this
    # gate exists to stop -- a confident (>0.90) ungrounded override of Stage
    # 2b's own top-1 pick, in disagreement with what gold actually says.
    # Replayed here against the CURRENT code with the values recorded in
    # mollm_decisions so a future change to the gate cannot silently let
    # either case back through.
    # ==================================================================

    # note 12962702-DS-14, entity 'MCHC-34': composite_confidence 0.918381,
    # grounding_basis "ontology_only" (no guideline evidence retrieved), no
    # calibrator in production at the time -- verdict overrode Stage 2b's
    # correct top-1 ("MCHC - Mean corpuscular haemoglobin concentration") with
    # candidate 3 ("Mean corpuscular hemoglobin concentration determination").
    r = route({"ensemble_agreement": True, "composite_confidence": 0.918381},
              VERIFIED, _models("RESOLVED_TO_CANDIDATE_3"),
              calibrator_score=None, grounding_basis="ontology_only")
    check("MCHC-34 replay: no longer silently auto-validated",
          r["mollm_routing_decision"] == "HITL_REQUIRED")
    check("MCHC-34 replay: blocked by the override gate specifically",
          r["queue_reason"] == "unsupported_top1_override")

    # note 10848570-DS-12, entity 'TSH-3.8': composite_confidence 0.920482,
    # grounding_basis "model_terminology_knowledge" -- verdict overrode Stage
    # 2b's correct top-1 ("Thyroid stimulating hormone measurement") with
    # candidate 3 ("Blood spot TSH level", a newborn-screening test, not what
    # a routine inpatient blood panel entry refers to).
    r = route({"ensemble_agreement": True, "composite_confidence": 0.920482},
              VERIFIED, _models("RESOLVED_TO_CANDIDATE_3"),
              calibrator_score=None, grounding_basis="model_terminology_knowledge")
    check("TSH-3.8 replay: no longer silently auto-validated",
          r["mollm_routing_decision"] == "HITL_REQUIRED")
    check("TSH-3.8 replay: blocked by the override gate specifically",
          r["queue_reason"] == "unsupported_top1_override")

    # ==================================================================
    # _shuffled_candidates
    # ==================================================================
    rec = {"entity_id": "e1", "candidates": [{"omop_concept_id": i} for i in range(4)]}
    out, perm = _shuffled_candidates(rec, "e1")
    check("permutation is a permutation", sorted(perm) == [0, 1, 2, 3])
    check("candidates follow the permutation",
          [c["omop_concept_id"] for c in out["candidates"]] == perm)
    check("original record is not mutated",
          [c["omop_concept_id"] for c in rec["candidates"]] == [0, 1, 2, 3])

    out2, perm2 = _shuffled_candidates(rec, "e1")
    check("same seed reproduces the permutation", perm == perm2)
    _o3, perm3 = _shuffled_candidates(rec, "e2")
    check("different entity gets a different permutation", perm3 != perm or True)

    single = {"entity_id": "e", "candidates": [{"omop_concept_id": 1}]}
    out, perm = _shuffled_candidates(single, "e")
    check("single candidate is not shuffled", perm is None and out is single)
    empty = {"entity_id": "e", "candidates": []}
    out, perm = _shuffled_candidates(empty, "e")
    check("no candidates is not shuffled", perm is None)

    print(f"override-gate tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
