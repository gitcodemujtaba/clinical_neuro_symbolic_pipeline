"""
tests/test_tier_gate.py — src/mollm_tier_gate.py's Tier 1-5 routing table.

Pure-logic tests only: tier3_fast_path(), tier5_precheck(), and route_tier()'s
aggregation of pre-supplied model_results (never calling run_two_step_ensemble()
itself, so no Ollama server is required to run this file).

WHY AST-EXTRACTED RATHER THAN IMPORTED NORMALLY: src/mollm_tier_gate.py imports
TIER3_SIMILARITY_FLOOR from src.normalization.constants, which -- via that
package's __init__.py -- loads the actual SapBERT model at import time
(~5.7s, confirmed empirically). tests/test_offset_mapping.py hit the same
problem for GLiNER/scispaCy and solved it with _load_pure_functions(): parse
the module's AST and execute only the named functions plus literal module-
level constants, with no imports run at all. Reused here verbatim rather than
duplicated, with TIER3_SIMILARITY_FLOOR's real value (0.72,
src/normalization/constants.py) injected directly as an extra_global -- the
one piece of state route_tier()'s pure logic actually depends on.

Run: python3 -m pytest tests/test_tier_gate.py -v
     (or: python3 tests/test_tier_gate.py for a plain-output run)
"""

import ast
import collections
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _load_pure_functions(module_filename: str, wanted: set, extra_globals: dict = None) -> dict:
    """See tests/test_offset_mapping.py's identical helper -- executes only the
    named functions plus literal module-level constants, running no imports."""
    path = os.path.join(SRC_DIR, module_filename)
    tree = ast.parse(open(path, encoding="utf-8").read())

    def _is_literal_assign(node):
        if not isinstance(node, ast.Assign):
            return False
        try:
            ast.literal_eval(node.value)
            return True
        except (ValueError, SyntaxError, TypeError):
            return False

    body = [
        n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted) or _is_literal_assign(n)
    ]
    ns = {"collections": collections}
    ns.update(extra_globals or {})
    exec(compile(ast.Module(body=body, type_ignores=[]), f"<{module_filename}>", "exec"), ns)
    return ns


TG = _load_pure_functions(
    "mollm_tier_gate.py",
    {"qualifier_fragment_precheck", "tier3_fast_path", "tier5_precheck", "route_tier"},
    extra_globals={"TIER3_SIMILARITY_FLOOR": 0.72, "TIER1_CONFIDENCE_FLOOR": 0.70},
)

qualifier_fragment_precheck = TG["qualifier_fragment_precheck"]
tier3_fast_path = TG["tier3_fast_path"]
tier5_precheck = TG["tier5_precheck"]
route_tier = TG["route_tier"]


def _entity(**overrides):
    base = {
        "original_text": "lasix", "expanded_text": "lasix",
        "gliner_label": "Medication", "assertion_status": "PRESENT",
        "experiencer": "PATIENT", "section_name": "Medications",
        "local_context": "started on lasix 40mg", "expansion_ambiguous": False,
        "candidates": [{"concept_name": "Furosemide", "match_basis": "verified_brand_alias",
                        "similarity_score": 1.0, "domain_id": "Drug",
                        "vocabulary_id": "RxNorm"}],
    }
    base.update(overrides)
    return base


def _vote(model, verdict, confidence=0.9, degenerate=False):
    return {"model": model, "verdict": verdict, "logprob_confidence": confidence,
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
    # qualifier_fragment_precheck (2026-08-16, Option 1)
    # ======================================================================
    qualifier_entity = _entity(gliner_label="Qualifier", original_text="left",
                               candidates=[{"concept_name": "Left", "match_basis": "exact_text",
                                            "similarity_score": 1.0}])
    r = qualifier_fragment_precheck(qualifier_entity)
    check("standalone Qualifier span -> Tier 5, no model calls",
          r is not None and r["queue_reason"] == "standalone_qualifier_span")

    anatomy_entity = _entity(gliner_label="Anatomy", original_text="chest",
                             candidates=[{"concept_name": "Chest structure"}])
    check("Anatomy-labeled single word is NOT caught by the qualifier filter",
          qualifier_fragment_precheck(anatomy_entity) is None)

    symptom_entity = _entity(gliner_label="Symptom", original_text="pain",
                             candidates=[{"concept_name": "Pain"}])
    check("Symptom-labeled single word is NOT caught by the qualifier filter",
          qualifier_fragment_precheck(symptom_entity) is None)

    r = route_tier(qualifier_entity, model_results=[_vote("a", "SUPPORTED_1")])
    check("route_tier: qualifier precheck short-circuits before model_results too",
          r["queue_reason"] == "standalone_qualifier_span")

    # ======================================================================
    # tier3_fast_path
    # ======================================================================
    r = tier3_fast_path(_entity())
    check("sole verified_brand_alias -> Tier 3 auto-validated",
          r is not None and r["tier"] == "TIER_3_AUTO_VALIDATED")
    check("Tier 3 picks the alias candidate index",
          r["final_candidate_index"] == 1)

    two_alias = _entity(candidates=[
        {"concept_name": "A", "match_basis": "verified_brand_alias"},
        {"concept_name": "B", "match_basis": "verified_brand_alias"},
    ])
    check("multiple alias hits do not fast-path (real ambiguity)",
          tier3_fast_path(two_alias) is None)

    no_alias = _entity(candidates=[{"concept_name": "X", "match_basis": "semantic_similarity"}])
    check("exact-text-only match does NOT fast-path (52.48% measured unsafe)",
          tier3_fast_path(no_alias) is None)

    ambiguous = _entity(expansion_ambiguous=True)
    check("ambiguous expansion blocks the fast path even with a clean alias",
          tier3_fast_path(ambiguous) is None)

    absent = _entity(assertion_status="ABSENT")
    check("non-PRESENT assertion blocks the fast path",
          tier3_fast_path(absent) is None)

    # ======================================================================
    # tier5_precheck
    # ======================================================================
    check("no candidates -> Tier 5",
          tier5_precheck(_entity(candidates=[]))["tier"] == "TIER_5_TRUE_AMBIGUITY")

    low_score = _entity(candidates=[{"concept_name": "X", "similarity_score": 0.5,
                                     "match_basis": "semantic_similarity"}])
    r = tier5_precheck(low_score)
    check("below TIER3_SIMILARITY_FLOOR -> Tier 5",
          r is not None and r["queue_reason"] == "below_similarity_floor")

    high_score = _entity(candidates=[{"concept_name": "X", "similarity_score": 0.9,
                                      "match_basis": "semantic_similarity"}])
    check("above floor, no ambiguity -> no precheck fires",
          tier5_precheck(high_score) is None)

    unresolved_acronym = _entity(expansion_ambiguous=True,
                                 candidates=[{"concept_name": "X", "similarity_score": 0.9}])
    r = tier5_precheck(unresolved_acronym)
    check("unresolved ambiguous expansion -> Tier 5",
          r is not None and r["queue_reason"] == "unresolved_acronym")

    resolved_acronym = _entity(expansion_ambiguous=True, mollm_escalation_resolved=True,
                               candidates=[{"concept_name": "X", "similarity_score": 0.9}])
    check("Pass-1-resolved ambiguous expansion does not trip Tier 5",
          tier5_precheck(resolved_acronym) is None)

    # ======================================================================
    # route_tier -- aggregation of pre-supplied model_results
    # ======================================================================
    plain = _entity(candidates=[{"concept_name": "X", "similarity_score": 0.9,
                                 "match_basis": "semantic_similarity"}])

    # Tier 1: 3/3 unanimous SUPPORTED_1 above the confidence floor.
    votes = [_vote("a", "SUPPORTED_1", 0.9), _vote("b", "SUPPORTED_1", 0.85),
             _vote("c", "SUPPORTED_1", 0.95)]
    r = route_tier(plain, model_results=votes)
    check("3/3 SUPPORTED_1 -> Tier 1 AUTO_VALIDATED",
          r["tier"] == "TIER_1_AUTO_VALIDATED"
          and r["mollm_routing_decision"] == "AUTO_VALIDATED")
    check("Tier 1 final_candidate_index is 1", r["final_candidate_index"] == 1)

    # Tier 1 confidence floor: unanimous but under-confident routes to HITL,
    # not silently promoted.
    votes_low_conf = [_vote("a", "SUPPORTED_1", 0.5), _vote("b", "SUPPORTED_1", 0.4),
                      _vote("c", "SUPPORTED_1", 0.3)]
    r = route_tier(plain, model_results=votes_low_conf)
    check("unanimous but low-confidence SUPPORTED_1 does not auto-validate",
          r["mollm_routing_decision"] == "HITL_REQUIRED"
          and r["queue_reason"] == "below_confidence_threshold")

    # Tier 2: 3/3 unanimous re-rank to the SAME candidate N.
    votes = [_vote("a", "RE_RANK_TO_CANDIDATE_2", 0.9),
             _vote("b", "RE_RANK_TO_CANDIDATE_2", 0.8),
             _vote("c", "RE_RANK_TO_CANDIDATE_2", 0.9)]
    r = route_tier(plain, model_results=votes)
    check("3/3 same re-rank target -> Tier 2 AUTO_RESOLVED",
          r["tier"] == "TIER_2_AUTO_RESOLVED"
          and r["mollm_routing_decision"] == "AUTO_RESOLVED")
    check("Tier 2 final_candidate_index follows the agreed re-rank target",
          r["final_candidate_index"] == 2)

    # Tier 2 requires the SAME N, not just "all disagreed with candidate 1".
    votes_diff_n = [_vote("a", "RE_RANK_TO_CANDIDATE_2"),
                    _vote("b", "RE_RANK_TO_CANDIDATE_3"),
                    _vote("c", "RE_RANK_TO_CANDIDATE_2")]
    r = route_tier(plain, model_results=votes_diff_n)
    check("re-rank to DIFFERENT candidates is a split, not Tier 2",
          r["tier"] == "TIER_4_ENSEMBLE_SPLIT")

    # Tier 4: 2-1 split.
    votes = [_vote("a", "SUPPORTED_1"), _vote("b", "SUPPORTED_1"),
             _vote("c", "RE_RANK_TO_CANDIDATE_2")]
    r = route_tier(plain, model_results=votes)
    check("2-1 split -> Tier 4 HITL", r["tier"] == "TIER_4_ENSEMBLE_SPLIT")
    check("Tier 4 queue_reason is ensemble_split", r["queue_reason"] == "ensemble_split")

    # Tier 4: 1-1-1 three-way split.
    votes = [_vote("a", "SUPPORTED_1"), _vote("b", "RE_RANK_TO_CANDIDATE_2"),
             _vote("c", "NONE_CORRECT")]
    r = route_tier(plain, model_results=votes)
    check("1-1-1 split -> Tier 4 HITL", r["tier"] == "TIER_4_ENSEMBLE_SPLIT")

    # Tier 5: unanimous NONE_CORRECT is not a split, it's true ambiguity.
    votes = [_vote("a", "NONE_CORRECT"), _vote("b", "NONE_CORRECT"),
             _vote("c", "NONE_CORRECT")]
    r = route_tier(plain, model_results=votes)
    check("3/3 NONE_CORRECT -> Tier 5, not Tier 4",
          r["tier"] == "TIER_5_TRUE_AMBIGUITY"
          and r["queue_reason"] == "verdict_none_correct")

    # Degenerate/error exclusion mirrors src.mollm_ensemble.combine(): a
    # degenerate vote is dropped, not counted as a real disagreement.
    votes = [_vote("a", "SUPPORTED_1", 0.9), _vote("b", "SUPPORTED_1", 0.9),
             _vote("c", "NONE_CORRECT", degenerate=True)]
    r = route_tier(plain, model_results=votes)
    check("degenerate vote excluded, but 2 usable SUPPORTED_1 is not unanimous-3 "
          "-> still Tier 4 (2 of 2 usable is not the same as 3/3)",
          r["tier"] == "TIER_4_ENSEMBLE_SPLIT")

    votes_all_bad = [_vote("a", "ERROR"), _vote("b", "NONE_CORRECT", degenerate=True),
                     _vote("c", "ERROR")]
    r = route_tier(plain, model_results=votes_all_bad)
    check("every model errored/degenerated -> HITL, not a false unanimous vote",
          r["mollm_routing_decision"] == "HITL_REQUIRED"
          and r["queue_reason"] == "model_unavailable_or_degenerate")

    # Fast paths take priority over any model_results passed in.
    alias_entity = _entity()
    r = route_tier(alias_entity, model_results=[_vote("a", "NONE_CORRECT")])
    check("Tier 3 fast path short-circuits before model_results is even consulted",
          r["tier"] == "TIER_3_AUTO_VALIDATED")

    print(f"tier-gate tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_tier_gate():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
