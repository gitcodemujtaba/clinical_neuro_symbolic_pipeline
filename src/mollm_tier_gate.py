"""
src/mollm_tier_gate.py — Pass 4: two-step CoT + Tier 1-5 autonomous gating.

WHAT THIS ADDS ON TOP OF src/mollm_ensemble.py. Production's route()/combine()
implement a strict unanimous-agreement gate with two confidence thresholds
(AUTO_VALIDATE_THRESHOLD/MOLLM_RESOLVE_THRESHOLD) and no distinction between a
2-1 split and a 1-1-1 split -- both just trip the "model_disagreement" safety
rule straight to HITL. Measured AUTO_VALIDATED precision under that scheme was
52.6% (docs/2026-08-15_Stage4_Stage5_Build.md), far short of the 90%
autonomous / ~0% false-positive target this module exists to work toward.

THE TWO-STEP COT. Each ensemble model (qwen2.5:3b, llama3.2:3b, phi4-mini,
src/llm_client.py) answers in two separate calls per entity, not one combined
prompt:
  Step A (_evaluate_one_model's meaning call): shown ONLY the entity + note
    context -- no candidate list at all -- and asked to state the entity's
    clinical meaning in plain language. This is deliberate: a single prompt
    that shows the candidate list ALONGSIDE an instruction to "ignore the
    scores" cannot reliably enforce that isolation -- src/mollm_ensemble.py's
    own SYSTEM_PROMPT has carried an "IGNORE THE SCORE WHILE JUDGING FIT"
    rule for a while and still measured real score/basis-anchoring failures
    (see that module's docstring). Removing the candidate list from Step A's
    prompt entirely makes the isolation structural, not just instructed.
  Step B (the sequential binary loop): candidates are evaluated ONE AT A TIME
    in Stage 2b's own rank order, seeded with Step A's stated meaning, using
    scripts/experiment_3b_voting.py's evaluate_candidates_sequentially()
    pattern -- a 1-to-N multiple-choice prompt measurably let 3B models
    detach a candidate's Basis tag from its own bracket index (see that
    module's _format_candidates() docstring); asking about exactly one
    candidate per call removes bracket-tracking entirely. Stops at the first
    accepted candidate. A model that accepts candidate 1 votes SUPPORTED_1; a
    model that rejects 1 but accepts a later candidate N votes
    RE_RANK_TO_CANDIDATE_N (the spec's own name for what
    src/mollm_ensemble.py calls RESOLVED_TO_CANDIDATE_N -- same concept,
    named per this module's own routing table rather than aliased, since this
    module does not touch mollm_decisions' existing verdict vocabulary at
    all); a model that rejects every candidate votes NONE_CORRECT.

THE TIER 1-5 TABLE (route_tier()). Tier 3 and Tier 5 are free pre-checks that
never spend a model call; Tiers 1/2/4 come out of the two-step ensemble's
three verdicts:
  Tier 1 (~70% target): 3/3 unanimous SUPPORTED_1, confidence over floor
    -> AUTO_VALIDATED.
  Tier 2 (~15% target): 3/3 unanimous RE_RANK_TO_CANDIDATE_N, same N
    -> AUTO_RESOLVED.
  Tier 3 (~5% target): free pre-check, see tier3_fast_path().
  Tier 4 (~5% target): non-unanimous verdicts (2-1 or 1-1-1 split)
    -> HITL_REQUIRED, queue_reason="ensemble_split".
  Tier 5 (~5% target): free pre-check (see tier5_precheck()) OR unanimous
    NONE_CORRECT -> HITL_REQUIRED.

WHY TIER 3 IS NARROWER THAN THE SPEC'S LITERAL WORDING. The spec describes
Tier 3 as "non-acronym exact string matches with zero contradiction cues."
This codebase already measured that criterion directly and found it unsafe:
scripts/experiment_3b_voting.py's check_deterministic_bypass() docstring
records Stage 2b's Tier-1 "1 (Exact)" match accuracy at 52.48% (402/766) --
barely better than chance, because an exact STRING match says nothing about
which DOMAIN/SENSE of an ambiguous term was meant. tier3_fast_path() below
reuses that function's already-validated, narrower criterion instead
(verified_brand_alias, a walked and confirmed KG relationship, not a string
coincidence) rather than implementing the wider spec criterion against
evidence already on file that it is not safe. Loosening this is a real,
measurable calibration question for a future pass once there is Tier 1/2 data
to check a wider rule against -- not a judgment call to make from the spec
text alone (matches this project's own "empirical validation before fixing"
discipline).

WHAT THIS MODULE DOES NOT DO YET. It does not write to KG3 (see
src/kg3_ingestion.py's ingest_auto_decision(), added feature-flagged/log-only
alongside this module) and it does not persist decisions to DuckDB the way
src/mollm_ensemble.py's store_decision() does for production Stage 3 -- both
are deliberately left as the next increment until this module's own tier
distribution and Tier 1/2 precision have been measured against gold on a real
batch (see scripts/run_tier_gate_batch.py), per the plan's own risk note that
direct KG3 write-back is new, higher-stakes code that should ship
feature-flagged and log-only first.
"""

import collections
import concurrent.futures

from src.llm_client import (
    LLMUnavailable,
    build_clients,
    extract_verdict_confidence,
    parse_json_response,
)
from src.normalization.constants import TIER3_SIMILARITY_FLOOR

TIER_1_AUTO_VALIDATED = "TIER_1_AUTO_VALIDATED"
TIER_2_AUTO_RESOLVED = "TIER_2_AUTO_RESOLVED"
TIER_3_AUTO_VALIDATED = "TIER_3_AUTO_VALIDATED"
TIER_4_ENSEMBLE_SPLIT = "TIER_4_ENSEMBLE_SPLIT"
TIER_5_TRUE_AMBIGUITY = "TIER_5_TRUE_AMBIGUITY"

AUTO_TIERS = {TIER_1_AUTO_VALIDATED, TIER_2_AUTO_RESOLVED, TIER_3_AUTO_VALIDATED}

# CALIBRATION-PENDING, same discipline as src/mollm_ensemble.py's
# AUTO_VALIDATE_THRESHOLD/MOLLM_RESOLVE_THRESHOLD: a placeholder until there
# is real Tier 1 decision data to check it against, not a measured value.
TIER1_CONFIDENCE_FLOOR = 0.70


# ==========================================================================
# Step A -- isolated clinical-meaning definition
# ==========================================================================

MEANING_SYSTEM_PROMPT = (
    "You are a clinical terminology expert reading a single clinical note. "
    "Your only job right now is to state what a highlighted text span means "
    "clinically, using the note's own context. You have not been shown any "
    "candidate concept list and must not anticipate or guess at one."
)


def _clinical_meaning_prompt(entity: dict) -> str:
    return (
        "ENTITY:\n"
        f"  text as written: {entity.get('original_text')!r}\n"
        f"  after abbreviation expansion: {entity.get('expanded_text')!r}\n"
        f"  extractor label: {entity.get('gliner_label')}\n"
        f"  assertion: {entity.get('assertion_status', 'PRESENT')} / "
        f"experiencer: {entity.get('experiencer', 'PATIENT')}\n\n"
        f"SECTION: {entity.get('section_name') or 'unknown'}\n"
        f"CONTEXT: ...{entity.get('local_context', '')}...\n\n"
        "TASK: Based ONLY on the note text above, state in one or two "
        "sentences what specific clinical concept (a diagnosis, medication, "
        "lab test, procedure, anatomical structure, symptom, or similar) this "
        "entity refers to. Describe the clinical meaning in plain language; "
        "do not name a database code, ontology term, or vocabulary identity -- "
        "you have not been shown any and must not invent one.\n\n"
        'Reply with JSON: {"clinical_meaning": "<plain-language statement>", '
        '"reasoning": "<one sentence on how the context supports this>"}'
    )


def _meaning_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "clinical_meaning": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["clinical_meaning", "reasoning"],
    }


# ==========================================================================
# Step B -- sequential binary candidate evaluation, seeded with Step A
# ==========================================================================

MATCH_SYSTEM_PROMPT = (
    "You are a clinical terminology validator auditing whether a proposed "
    "concept code correctly labels a text span, given an independent "
    "statement of what that span means."
)


def _binary_match_prompt(entity: dict, candidate: dict, clinical_meaning: str) -> str:
    basis = candidate.get("match_basis", "semantic_similarity")
    return (
        "This entity's clinical meaning was independently determined to be:\n"
        f'  "{clinical_meaning}"\n\n'
        "ENTITY:\n"
        f"  text as written: {entity.get('original_text')!r}\n"
        f"  section: {entity.get('section_name') or 'unknown'}\n"
        f"  assertion: {entity.get('assertion_status', 'PRESENT')} / "
        f"experiencer: {entity.get('experiencer', 'PATIENT')}\n"
        f"  context: ...{entity.get('local_context', '')}...\n\n"
        "CANDIDATE CONCEPT:\n"
        f"  name: {candidate.get('concept_name')}\n"
        f"  domain: {candidate.get('domain_id')}\n"
        f"  vocabulary: {candidate.get('vocabulary_id')}\n"
        f"  basis: {basis}\n\n"
        "RULES:\n"
        "1. Judge the candidate against the CLINICAL MEANING stated above, "
        "not against the raw text spelling or the candidate's match score.\n"
        "2. If basis is verified_brand_alias, it is a mathematically "
        "verified terminology-database link -- do not reject it merely "
        "because the spelling differs from the entity text.\n"
        "3. Ignore assertion/negation status when judging the CONCEPT match "
        '-- a negated entity ("denies fever") still maps to its concept '
        '("Fever") if the name matches; you are labeling which concept the '
        "text refers to, not diagnosing.\n"
        "4. Reject a candidate that is a distinct or clinically unrelated "
        "concept (e.g. mapping a symptom to a biological genus, or a lab "
        "value to an unrelated test). Do not force a match.\n\n"
        "Does this candidate concept match the clinical meaning stated "
        'above? Reply with JSON: {"match": true or false, "reasoning": '
        '"<one sentence>"}'
    )
    # 2026-08-15 REJECTED EXPERIMENT, recorded rather than silently discarded
    # (docs/2026-08-15_Phase2_TierGate_Validation.md). A 5th rule was tried
    # here ("match the core concept, not every detail -- a plain 'Pneumonia'
    # candidate matches even if the note adds laterality/severity") to fix a
    # real, diagnosed problem: qwen2.5:3b was the consistent dissenting vote
    # on obviously-correct cases (plain "pneumonia"/"heart failure"/"sodium"),
    # demanding contextual specificity the candidate name was never going to
    # have. On a 36-entity HIGH-tier batch it worked exactly as intended for
    # coverage (AUTO coverage 2.8% -> 55.6%) and catastrophically for
    # precision (Tier 1 precision 5.9%, 1/17 correct) -- the loosened rule
    # let all three models unanimously rubber-stamp WRONG matches on bare
    # qualifier/fragment entities that are not independently linkable
    # concepts at all ("left", "Removal", "Multiple", "fixation" all
    # auto-validated to some SNOMED code). This is the exact "consensus went
    # up, precision did not" failure this codebase's own Fragile Concept Gate
    # (src/mollm_ensemble.py route()) was built to catch, reproduced here by
    # a different mechanism. Reverted; the stricter 4-rule prompt above is
    # back in force. A future attempt at this same problem should test the
    # specificity relaxation MUCH more narrowly (e.g. only for candidates
    # whose GLiNER label is Condition/Medication/Lab Test, never for
    # single-word qualifier-shaped spans) rather than loosening it globally.


def _match_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "match": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["match", "reasoning"],
    }


def _evaluate_one_model(client, entity: dict) -> dict:
    """Runs the full two-step CoT for one ensemble model against one entity.

    Returns a per-model result dict with `verdict` in
    {"SUPPORTED_1", "RE_RANK_TO_CANDIDATE_{n}", "NONE_CORRECT", "ERROR"}, plus
    `logprob_confidence` (the accepted candidate's match=true/false token
    confidence, via src.llm_client.extract_verdict_confidence -- same
    geometric-mean-over-verdict-tokens machinery production uses, just
    pointed at a boolean's string form instead of an enum verdict) and
    `degenerate_generation` (True if any call in the trail degenerated, so
    the aggregator in route_tier() can exclude this model's vote the same
    way src.mollm_ensemble.combine() does).
    """
    candidates = entity.get("candidates") or []
    try:
        meaning_raw = client.complete(
            MEANING_SYSTEM_PROMPT, _clinical_meaning_prompt(entity),
            schema=_meaning_schema())
        meaning_parsed = parse_json_response(meaning_raw["text"])
        clinical_meaning = (meaning_parsed.get("clinical_meaning") or "").strip()
    except (LLMUnavailable, ValueError) as exc:
        return {"model": client.model_name, "verdict": "ERROR",
                "error": f"step_a: {type(exc).__name__}: {exc}",
                "clinical_meaning": None, "logprob_confidence": None,
                "degenerate_generation": False, "eval_trail": []}

    if not clinical_meaning:
        return {"model": client.model_name, "verdict": "ERROR",
                "error": "step_a: empty clinical_meaning",
                "clinical_meaning": None, "logprob_confidence": None,
                "degenerate_generation": bool(meaning_raw.get("degenerate_generation")),
                "eval_trail": []}

    step_a_degenerate = bool(meaning_raw.get("degenerate_generation"))
    trail = []
    for i, cand in enumerate(candidates, 1):
        try:
            raw = client.complete(
                MATCH_SYSTEM_PROMPT,
                _binary_match_prompt(entity, cand, clinical_meaning),
                schema=_match_schema())
            parsed = parse_json_response(raw["text"])
            matched = bool(parsed.get("match"))
            confidence = extract_verdict_confidence(
                raw["tokens"], "true" if matched else "false")
            step_degenerate = bool(raw.get("degenerate_generation"))
            trail.append({"candidate_index": i, "concept_name": cand.get("concept_name"),
                          "match": matched, "reasoning": parsed.get("reasoning"),
                          "confidence": confidence, "degenerate_generation": step_degenerate})
            if matched:
                verdict = "SUPPORTED_1" if i == 1 else f"RE_RANK_TO_CANDIDATE_{i}"
                return {"model": client.model_name, "verdict": verdict,
                        "clinical_meaning": clinical_meaning,
                        "reasoning": parsed.get("reasoning"),
                        "logprob_confidence": confidence,
                        "degenerate_generation": step_a_degenerate or step_degenerate,
                        "eval_trail": trail}
        except (LLMUnavailable, ValueError) as exc:
            trail.append({"candidate_index": i, "error": f"{type(exc).__name__}: {exc}"})

    any_degenerate = step_a_degenerate or any(t.get("degenerate_generation") for t in trail)
    return {"model": client.model_name, "verdict": "NONE_CORRECT",
            "clinical_meaning": clinical_meaning,
            "reasoning": trail[-1].get("reasoning") if trail and "reasoning" in trail[-1] else None,
            "logprob_confidence": None,
            "degenerate_generation": any_degenerate, "eval_trail": trail}


def run_two_step_ensemble(entity: dict, clients: dict = None) -> list:
    """Runs _evaluate_one_model() for every ensemble member in parallel
    (same concurrent.futures pattern scripts/experiment_3b_voting.py uses),
    independently -- Step A's isolated-context definition is per-model, not
    pooled, so a shared "meaning" would collapse three independent judgments
    into one and defeat the point of a 3-way vote.
    """
    clients = clients if clients is not None else build_clients()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(clients))) as executor:
        futures = {executor.submit(_evaluate_one_model, c, entity): name
                   for name, c in clients.items()}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


# ==========================================================================
# Tier 3 / Tier 5 free pre-checks
# ==========================================================================

def tier3_fast_path(entity: dict) -> dict:
    """AUTO_VALIDATED without spending any model calls. See module docstring
    for why this is narrower than the spec's literal "exact string match"
    wording -- reuses scripts/experiment_3b_voting.py's
    check_deterministic_bypass() criterion (verified_brand_alias, exactly one
    hit) rather than the wider, already-measured-unsafe one. Also requires no
    ambiguous-abbreviation expansion and an asserted-present entity, since
    "zero contradiction cues" in the spec's own wording rules out exactly
    those two cases. Returns None when the fast path does not apply.
    """
    alias_hits = [(i, c) for i, c in enumerate(entity.get("candidates") or [], 1)
                  if c.get("match_basis") == "verified_brand_alias"]
    if len(alias_hits) != 1:
        return None
    if entity.get("expansion_ambiguous"):
        return None
    if entity.get("assertion_status") not in (None, "PRESENT"):
        return None
    i, c = alias_hits[0]
    return {
        "tier": TIER_3_AUTO_VALIDATED,
        "mollm_routing_decision": "AUTO_VALIDATED",
        "queue_reason": None,
        "final_candidate_index": i,
        "composite_confidence": None,
        "routing_basis": (
            f"Tier 3 fast path: candidate [{i}] ({c.get('concept_name')}) is a "
            f"graph-verified brand alias, the sole such hit, with no ambiguous "
            f"expansion or non-PRESENT assertion -- skipped the two-step "
            f"ensemble entirely."),
        "models": [],
    }


def tier5_precheck(entity: dict) -> dict:
    """HITL_REQUIRED without spending any model calls, when the upstream
    signal is already known to be too weak for an ensemble verdict built on
    top of it to be trustworthy. Returns None when neither precheck fires
    (the entity should proceed to the full two-step ensemble).
    """
    candidates = entity.get("candidates") or []
    if not candidates:
        return {"tier": TIER_5_TRUE_AMBIGUITY, "mollm_routing_decision": "HITL_REQUIRED",
                "queue_reason": "no_candidates", "final_candidate_index": None,
                "composite_confidence": None,
                "routing_basis": "Tier 5: Stage 2 produced no candidates to evaluate.",
                "models": []}
    top_score = candidates[0].get("similarity_score")
    if isinstance(top_score, (int, float)) and top_score < TIER3_SIMILARITY_FLOOR:
        return {"tier": TIER_5_TRUE_AMBIGUITY, "mollm_routing_decision": "HITL_REQUIRED",
                "queue_reason": "below_similarity_floor", "final_candidate_index": None,
                "composite_confidence": None,
                "routing_basis": (f"Tier 5: top candidate similarity {top_score} < "
                                  f"{TIER3_SIMILARITY_FLOOR} (TIER3_SIMILARITY_FLOOR)"),
                "models": []}
    # Pass 1 (MoLLM acronym escalation, plan Phase 4) is not built yet in
    # this pipeline -- every ambiguous expansion is therefore "unresolved" by
    # construction until that phase lands. mollm_escalation_resolved is the
    # flag Phase 4 will set once it exists; absent that phase, this branch
    # always fires for an ambiguous entity, which is the honest behavior
    # until there is an escalation step to have resolved it.
    if entity.get("expansion_ambiguous") and not entity.get("mollm_escalation_resolved"):
        return {"tier": TIER_5_TRUE_AMBIGUITY, "mollm_routing_decision": "HITL_REQUIRED",
                "queue_reason": "unresolved_acronym", "final_candidate_index": None,
                "composite_confidence": None,
                "routing_basis": ("Tier 5: ambiguous abbreviation expansion, not "
                                  "resolved by Pass 1 MoLLM escalation (plan Phase 4, "
                                  "not yet built)."),
                "models": []}
    return None


# ==========================================================================
# Full Tier 1-5 gate
# ==========================================================================

def route_tier(entity: dict, model_results: list = None, clients: dict = None) -> dict:
    """Runs the Tier 1-5 gate for one Stage 2b LOW-tier entity record.

    Order: Tier 3 fast path -> Tier 5 pre-check (both free) -> full two-step
    ensemble -> Tier 1/2/4/5 based on the ensemble's three verdicts.

    `model_results`, when passed, skips running the ensemble (used by
    scripts/run_tier_gate_batch.py to separate "call the models" from
    "apply the routing table" for testability and for re-scoring a stored
    run without re-paying for inference).
    """
    fast = tier3_fast_path(entity)
    if fast:
        return fast
    pre = tier5_precheck(entity)
    if pre:
        return pre

    if model_results is None:
        model_results = run_two_step_ensemble(entity, clients=clients)

    usable = [m for m in model_results
              if not m.get("degenerate_generation") and m.get("verdict") != "ERROR"]
    n_excluded = len(model_results) - len(usable)
    if not usable:
        return {"tier": None, "mollm_routing_decision": "HITL_REQUIRED",
                "queue_reason": "model_unavailable_or_degenerate",
                "final_candidate_index": None, "composite_confidence": None,
                "routing_basis": "every ensemble member errored or degenerated; no usable vote",
                "models": model_results}

    verdicts = [m["verdict"] for m in usable]
    vote_counts = collections.Counter(verdicts)
    top_verdict, top_count = vote_counts.most_common(1)[0]

    confs = [m["logprob_confidence"] for m in usable
             if m.get("logprob_confidence") is not None and m["verdict"] == top_verdict]
    composite_confidence = round(sum(confs) / len(confs), 6) if confs else None

    unanimous = len(usable) == 3 and top_count == 3

    if unanimous and top_verdict == "SUPPORTED_1":
        if composite_confidence is not None and composite_confidence < TIER1_CONFIDENCE_FLOOR:
            return {"tier": None, "mollm_routing_decision": "HITL_REQUIRED",
                    "queue_reason": "below_confidence_threshold", "final_candidate_index": 1,
                    "composite_confidence": composite_confidence,
                    "routing_basis": (f"unanimous SUPPORTED_1 but composite_confidence "
                                      f"{composite_confidence} < {TIER1_CONFIDENCE_FLOOR}"),
                    "models": model_results}
        return {"tier": TIER_1_AUTO_VALIDATED, "mollm_routing_decision": "AUTO_VALIDATED",
                "queue_reason": None, "final_candidate_index": 1,
                "composite_confidence": composite_confidence,
                "routing_basis": (f"3/3 unanimous SUPPORTED_1, "
                                  f"composite_confidence {composite_confidence}"),
                "models": model_results}

    if unanimous and top_verdict.startswith("RE_RANK_TO_CANDIDATE_"):
        n = int(top_verdict.rsplit("_", 1)[1])
        return {"tier": TIER_2_AUTO_RESOLVED, "mollm_routing_decision": "AUTO_RESOLVED",
                "queue_reason": None, "final_candidate_index": n,
                "composite_confidence": composite_confidence,
                "routing_basis": (f"3/3 unanimous re-rank to candidate {n}, "
                                  f"composite_confidence {composite_confidence}"),
                "models": model_results}

    if unanimous and top_verdict == "NONE_CORRECT":
        return {"tier": TIER_5_TRUE_AMBIGUITY, "mollm_routing_decision": "HITL_REQUIRED",
                "queue_reason": "verdict_none_correct", "final_candidate_index": None,
                "composite_confidence": composite_confidence,
                "routing_basis": "3/3 unanimous NONE_CORRECT -- no usable resolution produced",
                "models": model_results}

    return {"tier": TIER_4_ENSEMBLE_SPLIT, "mollm_routing_decision": "HITL_REQUIRED",
            "queue_reason": "ensemble_split", "final_candidate_index": None,
            "composite_confidence": composite_confidence,
            "routing_basis": (f"non-unanimous verdicts: {dict(vote_counts)}"
                              + (f" ({n_excluded} model(s) excluded as "
                                 f"degenerate/errored)" if n_excluded else "")),
            "models": model_results}
