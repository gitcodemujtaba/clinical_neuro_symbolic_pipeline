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
import json
import os
import re
import sys

from src.llm_client import LLMUnavailable, extract_verdict_confidence, parse_json_response

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
    {"qualifier_fragment_precheck", "tier3_fast_path", "_lab_procedure_fast_path",
     "tier5_precheck", "route_tier",
     "_clinical_meaning_prompt", "_binary_match_prompt", "_is_coronary_segment_trap",
     "_is_short_alphanumeric_code", "_fragile_shorthand_trap"},
    extra_globals={"TIER3_SIMILARITY_FLOOR": 0.72, "TIER1_CONFIDENCE_FLOOR": 0.70,
                  "SHORT_ALPHANUMERIC_CODE_RE": re.compile(r"^[A-Za-z]{1,2}[0-9]{1,2}$"),
                  "SHORT_ALPHA_CODE_RE": re.compile(r"^[A-Z]{3,4}$"),
                  "PHYSEXAM_SHORTHAND_MATCH_BASIS": "verified_physexam_shorthand"},
)

qualifier_fragment_precheck = TG["qualifier_fragment_precheck"]
tier3_fast_path = TG["tier3_fast_path"]
tier5_precheck = TG["tier5_precheck"]
route_tier = TG["route_tier"]
_clinical_meaning_prompt = TG["_clinical_meaning_prompt"]
_binary_match_prompt = TG["_binary_match_prompt"]
_is_short_alphanumeric_code = TG["_is_short_alphanumeric_code"]
_fragile_shorthand_trap = TG["_fragile_shorthand_trap"]
ALLERGY_MEANING_INSTRUCTION = TG["ALLERGY_MEANING_INSTRUCTION"]
ALLERGY_CONTEXT_CLAUSE = TG["ALLERGY_CONTEXT_CLAUSE"]


# Separate extraction, EXHAUSTIVE_CANDIDATE_EVAL_ENABLED forced True (2026-08-18,
# "cold start" fix) -- _evaluate_one_model()/_resolve_tiebreak() actually call an
# LLM client, so real src.llm_client helpers (fast import, no SapBERT/GLiNER --
# confirmed ~0.3s, unlike src.normalization.constants) are injected for real
# rather than faked, same discipline as everything else in this file.
TG_EXHAUSTIVE = _load_pure_functions(
    "mollm_tier_gate.py",
    {"_evaluate_one_model", "_resolve_tiebreak", "_tiebreak_prompt", "_tiebreak_schema",
     "_condition_vs_observation_duplicate",
     "_clinical_meaning_prompt", "_binary_match_prompt", "_meaning_schema", "_match_schema"},
    extra_globals={"EXHAUSTIVE_CANDIDATE_EVAL_ENABLED": True,
                  "LLMUnavailable": LLMUnavailable,
                  "extract_verdict_confidence": extract_verdict_confidence,
                  "parse_json_response": parse_json_response},
)
_evaluate_one_model_exhaustive = TG_EXHAUSTIVE["_evaluate_one_model"]

# Second copy of the same functions with the flag False, to prove the default
# (legacy stop-at-first) behavior is untouched when the flag is off.
TG_SEQUENTIAL = _load_pure_functions(
    "mollm_tier_gate.py",
    {"_evaluate_one_model", "_resolve_tiebreak", "_tiebreak_prompt", "_tiebreak_schema",
     "_condition_vs_observation_duplicate",
     "_clinical_meaning_prompt", "_binary_match_prompt", "_meaning_schema", "_match_schema"},
    extra_globals={"EXHAUSTIVE_CANDIDATE_EVAL_ENABLED": False,
                  "LLMUnavailable": LLMUnavailable,
                  "extract_verdict_confidence": extract_verdict_confidence,
                  "parse_json_response": parse_json_response},
)
_evaluate_one_model_sequential = TG_SEQUENTIAL["_evaluate_one_model"]


class _FakeClient:
    """Scripted client.complete() -- returns each entry in `responses` in
    order, raising if the script runs out (a test bug, not a real failure
    path) or if that specific entry says to raise."""

    def __init__(self, responses, model_name="fake-model"):
        self.responses = list(responses)
        self.calls = []
        self.model_name = model_name

    def complete(self, system_prompt, user_prompt, schema=None, max_tokens=None):
        self.calls.append({"system": system_prompt, "user": user_prompt, "schema": schema})
        if not self.responses:
            raise AssertionError("FakeClient script exhausted -- test wired up too few responses")
        resp = self.responses.pop(0)
        if "raise" in resp:
            raise resp["raise"]
        return {"text": json.dumps(resp["json"]), "tokens": resp.get("tokens", []),
                "degenerate_generation": resp.get("degenerate", False)}


def _token(text, logprob):
    return [{"token": text, "logprob": logprob}]


_WOUND_DEHISCENCE_ENTITY = {
    "original_text": "wound dehiscence", "expanded_text": "wound dehiscence",
    "gliner_label": "Condition", "assertion_status": "PRESENT", "experiencer": "PATIENT",
    "section_name": "Hospital Course", "local_context": "noted to have wound dehiscence",
    "candidates": [
        {"concept_name": "Wound dehiscence", "domain_id": "Condition",
         "concept_class_id": "Disorder", "vocabulary_id": "SNOMED",
         "match_basis": "exact_text", "similarity_score": 1.0, "omop_concept_id": 111},
        {"concept_name": "Wound dehiscence", "domain_id": "Observation",
         "concept_class_id": "Morph Abnormality", "vocabulary_id": "SNOMED",
         "match_basis": "exact_text", "similarity_score": 1.0, "omop_concept_id": 222},
    ],
}


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
    # tier3_fast_path -- verified_lab_test_alias (2026-08-20, "Lab Test
    # near-duplicate-concept" fix, evaluation/grade_fresh25_by_tier.py)
    # ======================================================================
    lab_alias = _entity(gliner_label="Lab Test", original_text="Calcium-8.3",
                        candidates=[{"concept_name": "Blood calcium measurement",
                                    "match_basis": "verified_lab_test_alias",
                                    "similarity_score": 0.62}])
    r = tier3_fast_path(lab_alias)
    check("sole verified_lab_test_alias, not ambiguous -> Tier 3 auto-validated",
          r is not None and r["tier"] == "TIER_3_AUTO_VALIDATED")

    lab_alias_rescued = _entity(gliner_label="Lab Test", original_text="HCT-32",
                                is_ambiguous=True,
                                ambiguity_reason="verified_lab_test_alias_below_floor",
                                candidates=[{"concept_name": "Hematocrit determination",
                                            "match_basis": "verified_lab_test_alias",
                                            "similarity_score": 0.6625}])
    check("is_ambiguous=True with reason=...below_floor still fast-paths "
          "(the curated identity, not the raw score, is the trust signal)",
          tier3_fast_path(lab_alias_rescued) is not None)

    lab_alias_other_ambiguity = _entity(
        gliner_label="Lab Test", original_text="MCH-28",
        is_ambiguous=True, ambiguity_reason="tier3_top2_margin_below_threshold",
        candidates=[{"concept_name": "Mean corpuscular hemoglobin determination",
                    "match_basis": "verified_lab_test_alias", "similarity_score": 0.75}])
    check("is_ambiguous=True with a DIFFERENT reason (genuine near-miss) still "
          "blocks the fast path, same discipline as _lab_procedure_fast_path()",
          tier3_fast_path(lab_alias_other_ambiguity) is None)

    two_lab_alias = _entity(gliner_label="Lab Test", candidates=[
        {"concept_name": "A", "match_basis": "verified_lab_test_alias"},
        {"concept_name": "B", "match_basis": "verified_lab_test_alias"},
    ])
    check("multiple verified_lab_test_alias hits do not fast-path (real ambiguity)",
          tier3_fast_path(two_lab_alias) is None)

    lab_alias_ambiguous_expansion = _entity(
        gliner_label="Lab Test", expansion_ambiguous=True,
        candidates=[{"concept_name": "X", "match_basis": "verified_lab_test_alias"}])
    check("ambiguous expansion blocks the lab-alias fast path too",
          tier3_fast_path(lab_alias_ambiguous_expansion) is None)

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

    # 2026-08-16 (Phase 3 hybrid-retrieval A/B, user-diagnosed bug): under
    # RRF fusion, candidates[0] can be a WEAKER dense match than one further
    # down the list (BM25 promoted it). The floor must check the POOL's best
    # dense score, not just candidates[0]'s -- otherwise a genuinely passing
    # match gets floor-rejected purely because fusion ranked it 2nd.
    rrf_demoted_good_match = _entity(candidates=[
        {"concept_name": "BM25-favored but weak dense match", "similarity_score": 0.65},
        {"concept_name": "True match, demoted by RRF", "similarity_score": 0.90},
    ])
    check("a passing dense score elsewhere in the pool prevents floor-rejection "
          "even when candidates[0]'s own score is below floor",
          tier5_precheck(rrf_demoted_good_match) is None)

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
    # 2026-08-19 (temporary, conservative gating -- see AUTO_TIERS' own
    # comment in mollm_tier_gate.py): still tagged TIER_2_AUTO_RESOLVED for
    # audit continuity, but routed to HITL_REQUIRED, not auto-written,
    # pending post-fix re-validation of this tier's measured 20% precision.
    check("3/3 same re-rank target -> tier TIER_2_AUTO_RESOLVED, routed HITL pending revalidation",
          r["tier"] == "TIER_2_AUTO_RESOLVED"
          and r["mollm_routing_decision"] == "HITL_REQUIRED")
    check("Tier 2 final_candidate_index still follows the agreed re-rank target",
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

    # ======================================================================
    # 2026-08-16: allergy-context ensemble-split fix. Empirically diagnosed
    # via mollm_tier_gate_decisions.models trail data (see
    # ALLERGY_CONTEXT_CLAUSE's own docstring/comment) -- assertion_status
    # was already in both prompts, but Step B's rule 3 told models to ignore
    # it, and Step A gave no guidance on what ALLERGY status means. Both
    # gaps are entity-content, not routing-logic, so these are prompt-text
    # assertions rather than route_tier() aggregation checks.
    # ======================================================================
    allergy_entity = _entity(assertion_status="ALLERGY", original_text="morphine",
                             candidates=[{"concept_name": "Allergy to morphine",
                                          "match_basis": "allergy_nonstandard_exact"}])
    present_entity = _entity(assertion_status="PRESENT", original_text="morphine",
                             candidates=[{"concept_name": "Morphine"}])

    check("Step A prompt includes the ALLERGY meaning instruction for an "
          "ALLERGY-status entity",
          "documenting a known or reported patient allergy" in
          _clinical_meaning_prompt(allergy_entity))
    check("Step A prompt omits the ALLERGY instruction for a PRESENT-status entity",
          "documenting a known or reported patient allergy" not in
          _clinical_meaning_prompt(present_entity))

    step_b_with_clause = _binary_match_prompt(
        allergy_entity, allergy_entity["candidates"][0],
        "the patient has a documented allergy to morphine",
        extra_rule=ALLERGY_CONTEXT_CLAUSE)
    check("Step B prompt, with the allergy clause supplied, tells the model NOT "
          "to reject an allergy candidate as 'a different concept'",
          "Do NOT reject it under rule 4" in step_b_with_clause)
    check("Step B base rule 3 (ignore assertion for concept match) is still "
          "present alongside the allergy exception -- clause augments, doesn't "
          "replace, the base rules",
          "Ignore assertion/negation status when judging the CONCEPT match"
          in step_b_with_clause)

    step_b_without_clause = _binary_match_prompt(
        present_entity, present_entity["candidates"][0], "morphine, an opioid medication")
    check("Step B prompt for a non-ALLERGY entity (no extra_rule passed) "
          "does not mention the allergy exception",
          "ALLERGY EXCEPTION" not in step_b_without_clause)

    # ======================================================================
    # 2026-08-18: rule 2 (brand alias) is now conditional on THIS candidate's
    # own match_basis, not printed unconditionally -- the prompt-bleed fix
    # (live hallucination caught this session: a exact_text candidate got
    # justified in a model's reasoning as "verified to be a brand alias"
    # purely because the rule text was always present regardless of basis).
    # ======================================================================
    brand_alias_candidate = {"concept_name": "Furosemide", "match_basis": "verified_brand_alias"}
    exact_text_candidate = {"concept_name": "Wound dehiscence", "match_basis": "exact_text"}

    prompt_with_alias = _binary_match_prompt(present_entity, brand_alias_candidate, "furosemide")
    check("Step B prompt DOES mention verified_brand_alias when the candidate's "
          "own basis actually is verified_brand_alias",
          "verified_brand_alias" in prompt_with_alias)

    prompt_without_alias = _binary_match_prompt(present_entity, exact_text_candidate, "wound dehiscence")
    check("Step B prompt does NOT mention verified_brand_alias when the candidate's "
          "own basis is exact_text -- no rule for the model to misattribute",
          "verified_brand_alias" not in prompt_without_alias)
    check("base rules 1/3/4 are still present even when rule 2 is omitted",
          "SEMANTIC MATCH" in prompt_without_alias
          and "Ignore assertion/negation status" in prompt_without_alias
          and "STRICT DOMAIN MISMATCH" in prompt_without_alias)

    # -- 2026-08-18: verified_lab_test_alias (the "CHEM-7" fix) gets the
    # same trust rule as verified_brand_alias, with its own basis string
    # named explicitly rather than a generic paraphrase.
    lab_panel_candidate = {"concept_name": "Basic metabolic panel, Blood",
                           "match_basis": "verified_lab_test_alias"}
    prompt_with_panel_alias = _binary_match_prompt(present_entity, lab_panel_candidate, "CHEM-7")
    check("Step B prompt DOES mention verified_lab_test_alias when the "
          "candidate's own basis actually is verified_lab_test_alias",
          "verified_lab_test_alias" in prompt_with_panel_alias)
    check("the lab-panel-alias rule does not falsely claim it's a brand alias",
          "verified_brand_alias" not in prompt_with_panel_alias)

    # ======================================================================
    # 2026-08-18: Step A tightened to a single-phrase definition.
    # ======================================================================
    meaning_prompt = _clinical_meaning_prompt(present_entity)
    check("Step A prompt asks for a single-phrase definition, not a "
          "free sentence or two",
          "single-phrase" in meaning_prompt and "Define the term only" in meaning_prompt)

    # ======================================================================
    # 2026-08-17: ConsensusCalibrator escape hatch for TIER_4_ENSEMBLE_SPLIT.
    # ======================================================================
    class _FakeCalibrator:
        def __init__(self, fixed_score):
            self.fixed_score = fixed_score
            self.calls = []

        def score(self, context):
            self.calls.append(context)
            return self.fixed_score

    calibrator_entity = _entity(candidates=[
        {"concept_name": "X", "similarity_score": 0.9, "omop_concept_id": 111}])
    split_votes = [_vote("a", "SUPPORTED_1", 0.9), _vote("b", "SUPPORTED_1", 0.85),
                  _vote("c", "RE_RANK_TO_CANDIDATE_2", 0.9)]

    r = route_tier(calibrator_entity, model_results=split_votes)
    check("calibrator=None, conn=None (default) reproduces existing Tier 4 "
          "behavior exactly, no signature-change regression",
          r["tier"] == "TIER_4_ENSEMBLE_SPLIT")
    check("2026-08-18: calibrated_score is explicitly None (not just absent) "
          "when the calibrator was never supplied at all",
          r["calibrated_score"] is None)

    high_calibrator = _FakeCalibrator(0.95)
    r = route_tier(calibrator_entity, model_results=split_votes,
                   calibrator=high_calibrator, conn="FAKE_CONN")
    check("a high-scoring calibrator promotes a SUPPORTED_1-plurality split "
          "to TIER_1B_CALIBRATED_AUTO_VALIDATED",
          r["tier"] == "TIER_1B_CALIBRATED_AUTO_VALIDATED"
          and r["mollm_routing_decision"] == "AUTO_VALIDATED")
    check("the promoted decision's final_candidate_index follows the "
          "plurality verdict (SUPPORTED_1 -> candidate 1)",
          r["final_candidate_index"] == 1)
    check("routing_basis records the calibrator's own contribution, auditable",
          "ConsensusCalibrator" in r["routing_basis"]
          or "calibrat" in r["routing_basis"].lower())
    check("2026-08-18: calibrated_score is persisted on the promoted decision "
          "itself, not just embedded in routing_basis text",
          r["calibrated_score"] == 0.95)

    low_calibrator = _FakeCalibrator(0.40)
    r = route_tier(calibrator_entity, model_results=split_votes,
                   calibrator=low_calibrator, conn="FAKE_CONN")
    check("a low-scoring calibrator leaves the split at Tier 4, unpromoted",
          r["tier"] == "TIER_4_ENSEMBLE_SPLIT")
    check("2026-08-18 (tier-gate audit fix #2): the calibrator's real score "
          "is STILL persisted even when it does NOT clear the threshold -- "
          "previously this exact case computed calibrated_score then threw "
          "it away, making 'never consulted' indistinguishable from "
          "'consulted and scored low' after the fact",
          r["calibrated_score"] == 0.40)

    check("supplying a calibrator WITHOUT a conn is still a no-op (both required)",
          route_tier(calibrator_entity, model_results=split_votes,
                    calibrator=high_calibrator, conn=None)["tier"] == "TIER_4_ENSEMBLE_SPLIT")
    check("supplying a conn WITHOUT a calibrator is still a no-op (both required)",
          route_tier(calibrator_entity, model_results=split_votes,
                    calibrator=None, conn="FAKE_CONN")["tier"] == "TIER_4_ENSEMBLE_SPLIT")

    # A plurality of NONE_CORRECT has no candidate to promote -- the
    # calibrator must never be consulted for this shape, regardless of what
    # it would have scored.
    none_correct_votes = [_vote("a", "NONE_CORRECT", 0.9), _vote("b", "NONE_CORRECT", 0.85),
                          _vote("c", "SUPPORTED_1", 0.9)]
    never_called_calibrator = _FakeCalibrator(0.99)
    r = route_tier(calibrator_entity, model_results=none_correct_votes,
                   calibrator=never_called_calibrator, conn="FAKE_CONN")
    check("NONE_CORRECT plurality stays Tier 4 even with a high-scoring "
          "calibrator available -- there is no candidate to promote",
          r["tier"] == "TIER_4_ENSEMBLE_SPLIT")
    check("the calibrator is never even called for a NONE_CORRECT plurality",
          never_called_calibrator.calls == [])

    # ======================================================================
    # 2026-08-17: coronary-artery-segment trap -- bypasses the calibrator
    # entirely (not merely overrides its score) for a known-fragile pattern.
    # ======================================================================
    coronary_abbrev_entity = _entity(
        original_text="LCX",
        candidates=[{"concept_name": "Structure of circumflex coronary artery",
                     "similarity_score": 0.9, "omop_concept_id": 222}])
    trap_calibrator = _FakeCalibrator(0.99)
    r = route_tier(coronary_abbrev_entity, model_results=split_votes,
                   calibrator=trap_calibrator, conn="FAKE_CONN")
    check("a coronary-abbreviation mention text stays Tier 4 even with a "
          "high-scoring calibrator available",
          r["tier"] == "TIER_4_ENSEMBLE_SPLIT")
    check("2026-08-18: calibrated_score is None for a trapped entity -- "
          "calibrator.score() genuinely never runs for it (not just "
          "overridden after the fact), so there's nothing to persist",
          r["calibrated_score"] is None and trap_calibrator.calls == [])
    check("queue_reason records the coronary trap specifically, distinguishable "
          "from a plain unpromoted split",
          r["queue_reason"] == "coronary_segment_trap")
    check("the calibrator is never even called for a trapped coronary abbreviation",
          trap_calibrator.calls == [])

    generic_coronary_entity = _entity(
        original_text="the vessel",
        candidates=[{"concept_name": "Coronary artery structure",
                     "similarity_score": 0.9, "omop_concept_id": 333}])
    trap_calibrator_2 = _FakeCalibrator(0.99)
    r = route_tier(generic_coronary_entity, model_results=split_votes,
                   calibrator=trap_calibrator_2, conn="FAKE_CONN")
    check("a top candidate resolving to the generic 'Coronary artery structure' "
          "parent concept also trips the trap, independent of mention text",
          r["tier"] == "TIER_4_ENSEMBLE_SPLIT"
          and r["queue_reason"] == "coronary_segment_trap")
    check("the calibrator is never even called when the generic-concept trap fires",
          trap_calibrator_2.calls == [])

    check("an ordinary (non-coronary) entity is unaffected by the trap -- "
          "the earlier high-scoring-calibrator promotion still fires",
          route_tier(calibrator_entity, model_results=split_votes,
                    calibrator=_FakeCalibrator(0.95), conn="FAKE_CONN")["tier"]
          == "TIER_1B_CALIBRATED_AUTO_VALIDATED")

    # ======================================================================
    # 2026-08-17: short alphanumeric code trap (S2/T1/V12-shaped mentions).
    # ======================================================================
    for text in ["S2", "T1", "v12", "AB99"]:
        code_entity = _entity(original_text=text, candidates=[
            {"concept_name": "Some ambiguous concept", "similarity_score": 0.9,
             "omop_concept_id": 444}])
        trap_calibrator_3 = _FakeCalibrator(0.99)
        r = route_tier(code_entity, model_results=split_votes,
                       calibrator=trap_calibrator_3, conn="FAKE_CONN")
        check(f"a short alphanumeric code mention ({text!r}) stays Tier 4 even "
              f"with a high-scoring calibrator available",
              r["tier"] == "TIER_4_ENSEMBLE_SPLIT"
              and r["queue_reason"] == "short_alphanumeric_code_trap")
        check(f"the calibrator is never even called for {text!r}",
              trap_calibrator_3.calls == [])

    for text in ["hemoglobin", "S", "12", "S200", "ABC1"]:
        non_code_entity = _entity(original_text=text, candidates=[
            {"concept_name": "X", "similarity_score": 0.9, "omop_concept_id": 555}])
        r = route_tier(non_code_entity, model_results=split_votes,
                       calibrator=_FakeCalibrator(0.95), conn="FAKE_CONN")
        check(f"{text!r} does not match the short-code shape -- unaffected by "
              f"this trap, the calibrator promotion still fires",
              r["tier"] == "TIER_1B_CALIBRATED_AUTO_VALIDATED")

    # ======================================================================
    # 2026-08-18 (5-note validation run, 'LMCA' finding): widened short-code
    # regex (pure-alpha 3-4 letter ALL-CAPS shapes) + elevated gate (trap
    # now guards the UNANIMOUS TIER_1_AUTO_VALIDATED path too, not just the
    # calibrator's TIER_1B path).
    # ======================================================================

    # -- _is_short_alphanumeric_code() unit-level: the new pure-alpha shape.
    for text in ["LAD", "LCX", "RCA", "PDA", "LMCA", "XYZ", "WXYZ"]:
        check(f"_is_short_alphanumeric_code({text!r}) -- pure-alpha 3-4 "
             f"letter ALL-CAPS shape is caught even for an abbreviation "
             f"NOT on the coronary enumerated list (XYZ/WXYZ)",
              _is_short_alphanumeric_code(_entity(original_text=text)))
    for text in ["lad", "Lad", "AB", "ABCDE", "hemoglobin"]:
        check(f"_is_short_alphanumeric_code({text!r}) -- lowercase/mixed-case, "
             f"too short, too long, or an ordinary word does not match the "
             f"pure-alpha shape",
              not _is_short_alphanumeric_code(_entity(original_text=text)))

    # -- _fragile_shorthand_trap() unit-level: reason string is specific to
    # which trap fired, and (False, None) when neither does.
    check("_fragile_shorthand_trap: coronary-enumerated text wins even "
         "without a matching candidate concept name",
          _fragile_shorthand_trap(_entity(original_text="lmca"), 1, [])
          == (True, "coronary_segment_trap"))
    check("_fragile_shorthand_trap: a not-on-the-list pure-alpha code falls "
         "through to the short-code reason",
          _fragile_shorthand_trap(_entity(original_text="XYZ"), 1, [])
          == (True, "short_alphanumeric_code_trap"))
    check("_fragile_shorthand_trap: an ordinary entity is untrapped",
          _fragile_shorthand_trap(_entity(original_text="hemoglobin"), 1, [])
          == (False, None))

    # -- elevated gate: a genuinely UNANIMOUS 3/3 SUPPORTED_1 vote on a
    # trapped mention must NOT reach TIER_1_AUTO_VALIDATED -- this is the
    # actual LMCA production case (all 3 models agreed on the wrong
    # candidate; unanimity is not evidence the candidate list was sound).
    # No calibrator/conn supplied at all here, proving the elevated check
    # runs independently of the calibrator machinery.
    # NOTE: `votes` (the module-level name) has been reassigned several
    # times since its original "3/3 unanimous SUPPORTED_1" definition
    # around line 283 -- a fresh, locally-named fixture is used here rather
    # than relying on `votes` still holding that value.
    unanimous_supported_votes = [_vote("a", "SUPPORTED_1", 0.9), _vote("b", "SUPPORTED_1", 0.85),
                                 _vote("c", "SUPPORTED_1", 0.95)]
    lmca_entity = _entity(original_text="LMCA", candidates=[
        {"concept_name": "Coronary artery stenosis", "similarity_score": 0.9,
         "omop_concept_id": 666}])
    r = route_tier(lmca_entity, model_results=unanimous_supported_votes)
    check("unanimous 3/3 SUPPORTED_1 on 'LMCA' (coronary-enumerated) does "
         "NOT reach Tier 1 -- forced to Tier 4 HITL despite full agreement",
          r["tier"] == "TIER_4_ENSEMBLE_SPLIT"
          and r["mollm_routing_decision"] == "HITL_REQUIRED"
          and r["queue_reason"] == "coronary_segment_trap")

    s2_entity_unanimous = _entity(original_text="S2", candidates=[
        {"concept_name": "Some ambiguous concept", "similarity_score": 0.9,
         "omop_concept_id": 777}])
    r = route_tier(s2_entity_unanimous, model_results=unanimous_supported_votes)
    check("unanimous 3/3 SUPPORTED_1 on 'S2' (short-code shape) does NOT "
         "reach Tier 1 either -- same elevated gate, other trap",
          r["tier"] == "TIER_4_ENSEMBLE_SPLIT"
          and r["queue_reason"] == "short_alphanumeric_code_trap")

    xyz_entity_unanimous = _entity(original_text="XYZ", candidates=[
        {"concept_name": "Some concept", "similarity_score": 0.9, "omop_concept_id": 888}])
    r = route_tier(xyz_entity_unanimous, model_results=unanimous_supported_votes)
    check("unanimous 3/3 SUPPORTED_1 on a not-yet-enumerated pure-alpha "
         "code ('XYZ') is ALSO caught by the widened regex -- forward-"
         "looking coverage, not dependent on the coronary list",
          r["tier"] == "TIER_4_ENSEMBLE_SPLIT"
          and r["queue_reason"] == "short_alphanumeric_code_trap")

    # -- regression: an ordinary unanimous entity is completely unaffected
    # by the elevated gate -- Tier 1 still fires normally (same as the
    # pre-existing "3/3 SUPPORTED_1 -> Tier 1" check above, re-asserted
    # here for locality with the new trap tests).
    r = route_tier(plain, model_results=unanimous_supported_votes)
    check("elevated gate does not disturb an ordinary (non-trapped) "
         "unanimous SUPPORTED_1 -- still reaches Tier 1 normally",
          r["tier"] == "TIER_1_AUTO_VALIDATED")

    # ======================================================================
    # 2026-08-18: exhaustive candidate evaluation + comparative tiebreak
    # ("cold start" fix -- see EXHAUSTIVE_CANDIDATE_EVAL_ENABLED's own
    # comment in src/mollm_tier_gate.py for the full rationale).
    # ======================================================================

    # -- default (flag off) behavior is untouched: stops at the FIRST
    # accepted candidate, never even calls the client about candidate 2.
    seq_client = _FakeClient([
        {"json": {"clinical_meaning": "surgical wound has separated",
                  "reasoning": "documented in hospital course"}},
        {"json": {"match": True, "reasoning": "matches"}, "tokens": _token("true", -0.1)},
    ])
    r = _evaluate_one_model_sequential(seq_client, _WOUND_DEHISCENCE_ENTITY)
    check("flag OFF: stops at first accept, verdict is SUPPORTED_1",
          r["verdict"] == "SUPPORTED_1")
    check("flag OFF: only 2 calls made (meaning + candidate 1) -- candidate 2 "
          "never evaluated, exact legacy behavior",
          len(seq_client.calls) == 2)

    # -- exhaustive mode, 0 accepted -> NONE_CORRECT, same as before.
    none_client = _FakeClient([
        {"json": {"clinical_meaning": "an unrelated finding", "reasoning": "x"}},
        {"json": {"match": False, "reasoning": "no"}, "tokens": _token("false", -0.1)},
        {"json": {"match": False, "reasoning": "no"}, "tokens": _token("false", -0.1)},
    ])
    r = _evaluate_one_model_exhaustive(none_client, _WOUND_DEHISCENCE_ENTITY)
    check("exhaustive mode, 0 accepted -> NONE_CORRECT",
          r["verdict"] == "NONE_CORRECT")
    check("exhaustive mode, 0 accepted -> all candidates were actually checked",
          len(none_client.calls) == 3)

    # -- exhaustive mode, exactly 1 accepted -> normal verdict, no tiebreak
    # call spent (only candidate 2 accepted here).
    one_client = _FakeClient([
        {"json": {"clinical_meaning": "surgical wound has separated", "reasoning": "x"}},
        {"json": {"match": False, "reasoning": "wrong domain"}, "tokens": _token("false", -0.1)},
        {"json": {"match": True, "reasoning": "matches"}, "tokens": _token("true", -0.1)},
    ])
    r = _evaluate_one_model_exhaustive(one_client, _WOUND_DEHISCENCE_ENTITY)
    check("exhaustive mode, exactly 1 accepted -> RE_RANK_TO_CANDIDATE_2, no "
          "tiebreak call spent",
          r["verdict"] == "RE_RANK_TO_CANDIDATE_2" and len(one_client.calls) == 3)

    # -- exhaustive mode, BOTH accepted -> genuine tiebreak, model picks
    # candidate 2 (the Observation/Morph-Abnormality one) using context.
    tie_client = _FakeClient([
        {"json": {"clinical_meaning": "surgical wound has separated",
                  "reasoning": "hospital course documents this"}},
        {"json": {"match": True, "reasoning": "plausible"}, "tokens": _token("true", -0.1)},
        {"json": {"match": True, "reasoning": "also plausible"}, "tokens": _token("true", -0.1)},
        {"json": {"best_index": "2",
                  "reasoning": "documentation style matches a morphologic finding"},
         "tokens": _token("2", -0.05)},
    ])
    r = _evaluate_one_model_exhaustive(tie_client, _WOUND_DEHISCENCE_ENTITY)
    check("exhaustive mode, both accepted -> tiebreak call is actually made "
          "(4 total calls: meaning + 2 candidate checks + 1 tiebreak)",
          len(tie_client.calls) == 4)
    check("tiebreak result is honored: verdict is RE_RANK_TO_CANDIDATE_2, "
          "not the lower-indexed candidate 1",
          r["verdict"] == "RE_RANK_TO_CANDIDATE_2")
    check("tiebreak eval_trail entry is recorded for audit",
          any(t.get("tiebreak") for t in r["eval_trail"]))

    # -- 2026-08-18: the data-grounded CONDITION_VS_OBSERVATION_PRIOR fires
    # for exactly the pattern it was measured against (Condition/Disorder vs
    # Observation/Morph Abnormality, same name), and stays silent otherwise.
    _condition_vs_observation_duplicate = TG_EXHAUSTIVE["_condition_vs_observation_duplicate"]
    _tiebreak_prompt_fn = TG_EXHAUSTIVE["_tiebreak_prompt"]
    wound_accepted = [
        {"index": 1, "candidate": _WOUND_DEHISCENCE_ENTITY["candidates"][0], "reasoning": "x"},
        {"index": 2, "candidate": _WOUND_DEHISCENCE_ENTITY["candidates"][1], "reasoning": "y"},
    ]
    check("Condition/Disorder vs Observation/Morph-Abnormality pattern is detected",
          _condition_vs_observation_duplicate(wound_accepted) is True)
    check("the measured prior text is actually injected into the tiebreak prompt "
          "for this pattern",
          "STRICT CORPUS CONVENTION" in _tiebreak_prompt_fn(
              _WOUND_DEHISCENCE_ENTITY, wound_accepted, "a wound that reopened"))

    unrelated_accepted = [
        {"index": 1, "candidate": {"concept_name": "Furosemide", "domain_id": "Drug",
                                   "concept_class_id": "Ingredient"}, "reasoning": "x"},
        {"index": 2, "candidate": {"concept_name": "Lasix", "domain_id": "Drug",
                                   "concept_class_id": "Branded Drug"}, "reasoning": "y"},
    ]
    check("the pattern is NOT detected for an unrelated tie (different names/domains)",
          _condition_vs_observation_duplicate(unrelated_accepted) is False)

    # 2026-08-20 regression guard. _WOUND_DEHISCENCE_ENTITY's own candidates
    # (above) carry concept_class_id, but src.normalization.tier_retrieval's
    # real _candidate() NEVER populates that field -- confirmed live, this
    # let a real bug hide behind a test fixture that didn't match production
    # candidate shape: the ORIGINAL _condition_vs_observation_duplicate()
    # required concept_class_id to match too, which meant it could never
    # fire on a real candidate dict, only on this idealized fixture. Guards
    # against that exact class of bug recurring by testing with a
    # concept_class_id-FREE candidate pair, the shape real candidates
    # actually have.
    wound_accepted_no_class = [
        {"index": 1, "candidate": {"concept_name": "Wound dehiscence", "domain_id": "Condition",
                                   "vocabulary_id": "SNOMED", "match_basis": "exact_text",
                                   "similarity_score": 1.0, "omop_concept_id": 111},
         "reasoning": "x"},
        {"index": 2, "candidate": {"concept_name": "Wound dehiscence", "domain_id": "Observation",
                                   "vocabulary_id": "SNOMED", "match_basis": "exact_text",
                                   "similarity_score": 1.0, "omop_concept_id": 222},
         "reasoning": "y"},
    ]
    check("pattern is STILL detected with no concept_class_id at all -- the real shape "
         "of every actual candidate dict this function is ever called with in production",
          _condition_vs_observation_duplicate(wound_accepted_no_class) is True)
    check("the prior is NOT injected into the prompt for an unrelated tie -- "
          "conditional, not blanket, injection",
          "STRICT CORPUS CONVENTION" not in _tiebreak_prompt_fn(
              _entity(), unrelated_accepted, "furosemide"))

    # -- tiebreak call itself picks candidate 1 -> SUPPORTED_1, proving the
    # tiebreak genuinely decides rather than always preferring the higher
    # index either.
    tie_client_2 = _FakeClient([
        {"json": {"clinical_meaning": "surgical wound has separated", "reasoning": "x"}},
        {"json": {"match": True, "reasoning": "plausible"}, "tokens": _token("true", -0.1)},
        {"json": {"match": True, "reasoning": "also plausible"}, "tokens": _token("true", -0.1)},
        {"json": {"best_index": "1", "reasoning": "context favors the disorder sense"},
         "tokens": _token("1", -0.05)},
    ])
    r = _evaluate_one_model_exhaustive(tie_client_2, _WOUND_DEHISCENCE_ENTITY)
    check("tiebreak can also legitimately choose candidate 1 -> SUPPORTED_1",
          r["verdict"] == "SUPPORTED_1")

    # -- tiebreak call fails (LLMUnavailable) -> falls back to the HIGHEST-
    # CONFIDENCE accepted candidate, not the lowest index (that would just
    # reintroduce the arbitrary-pick bug this mode exists to remove).
    # Candidate 1 is scripted with a much higher token confidence than
    # candidate 2 here.
    fallback_client = _FakeClient([
        {"json": {"clinical_meaning": "surgical wound has separated", "reasoning": "x"}},
        {"json": {"match": True, "reasoning": "plausible"}, "tokens": _token("true", -0.05)},
        {"json": {"match": True, "reasoning": "also plausible"}, "tokens": _token("true", -2.0)},
        {"raise": LLMUnavailable("simulated transport failure")},
    ])
    r = _evaluate_one_model_exhaustive(fallback_client, _WOUND_DEHISCENCE_ENTITY)
    check("tiebreak transport failure falls back to the HIGHER-confidence "
          "accepted candidate (1), not just candidate order",
          r["verdict"] == "SUPPORTED_1")
    check("fallback path is recorded in eval_trail for audit, not silent",
          any(t.get("fallback_reason") for t in r["eval_trail"]))

    # -- tiebreak call returns an out-of-set index -> same fallback path,
    # not a crash and not a silently-accepted invalid answer.
    badindex_client = _FakeClient([
        {"json": {"clinical_meaning": "surgical wound has separated", "reasoning": "x"}},
        {"json": {"match": True, "reasoning": "plausible"}, "tokens": _token("true", -2.0)},
        {"json": {"match": True, "reasoning": "also plausible"}, "tokens": _token("true", -0.05)},
        {"json": {"best_index": "5", "reasoning": "hallucinated index"}},
    ])
    r = _evaluate_one_model_exhaustive(badindex_client, _WOUND_DEHISCENCE_ENTITY)
    check("tiebreak out-of-set response falls back safely (higher-confidence "
          "candidate 2 here) instead of trusting the invalid index",
          r["verdict"] == "RE_RANK_TO_CANDIDATE_2")

    print(f"tier-gate tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_tier_gate():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
