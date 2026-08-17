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
import json
import re
import uuid

from src.llm_client import (
    LLMUnavailable,
    build_clients,
    extract_verdict_confidence,
    parse_json_response,
)
from src.normalization.constants import TIER3_SIMILARITY_FLOOR
from src.provenance import (
    provenance_alter_statements,
    provenance_column_sql,
    provenance_params,
    provenance_placeholders,
)

TIER_1_AUTO_VALIDATED = "TIER_1_AUTO_VALIDATED"
TIER_2_AUTO_RESOLVED = "TIER_2_AUTO_RESOLVED"
TIER_3_AUTO_VALIDATED = "TIER_3_AUTO_VALIDATED"
TIER_4_ENSEMBLE_SPLIT = "TIER_4_ENSEMBLE_SPLIT"
TIER_5_TRUE_AMBIGUITY = "TIER_5_TRUE_AMBIGUITY"
# 2026-08-17 (plan Phase 6). A split-vote entity a fitted ConsensusCalibrator
# (src/mollm_tier_calibrator.py) scores as likely-correct promotes here --
# deliberately NOT merged into TIER_1_AUTO_VALIDATED, so every downstream
# count (precision measurement, audit, "how much of AUTO tier came from a
# genuine unanimous vote vs a calibrated guess") can tell the two apart.
TIER_1B_CALIBRATED_AUTO_VALIDATED = "TIER_1B_CALIBRATED_AUTO_VALIDATED"

AUTO_TIERS = {TIER_1_AUTO_VALIDATED, TIER_2_AUTO_RESOLVED, TIER_3_AUTO_VALIDATED,
             TIER_1B_CALIBRATED_AUTO_VALIDATED}

# CALIBRATION-PENDING, same discipline as src/mollm_ensemble.py's
# AUTO_VALIDATE_THRESHOLD/MOLLM_RESOLVE_THRESHOLD: a placeholder until there
# is real Tier 1 decision data to check it against, not a measured value.
TIER1_CONFIDENCE_FLOOR = 0.70

# Fit 2026-08-17 (evaluation/tier_gate_cal_eval.py, Phase 6 steps 3-4)
# against a note-disjoint held-out split of the overnight 31-note corpus
# run's TIER_4_ENSEMBLE_SPLIT population (668 labeled examples, 70.4% base
# rate), WITH both hard traps active (without them, precision tops out
# around 89% at any threshold -- they're load-bearing for every number
# below). Originally locked at 0.65 (98.0% precision / 38.9% coverage on
# that val set), then bumped after a genuinely fresh 5-note run (outside
# both the calibrator's training AND its val notes) surfaced 3 false
# positives -- 'Tenotomy' (0.6997), 'S2' (0.704), 'incontinence' (0.70698)
# -- clustered in a narrow 0.699-0.707 band just above the old threshold.
# 'S2' is now independently caught by the alphanumeric-short-code trap
# below regardless of score, but 'Tenotomy'/'incontinence' are plain words,
# not short codes -- only a threshold bump catches those two.
# RE-MEASURED on the original held-out val set WITH the new alphanumeric
# trap also active: 0.70 already reaches 100% precision at 17.5% coverage
# (22/126); 0.72 still 100% precision but coverage drops sharply to 7.9%
# (10/126) -- a real, measured cost, not a vague "a few points." 0.72 (not
# 0.70) was chosen specifically because 'incontinence' (0.70698) is a
# non-alphanumeric word the short-code trap can't catch, and 0.70 alone
# would still have promoted it. Projected corpus-wide at 0.72: ~129 of the
# 1,629 TIER_4_ENSEMBLE_SPLIT entities promotable (down from ~634 at the
# old 0.65) -- a materially smaller coverage win than the original Phase 6
# estimate; re-confirm this trade-off is still wanted at production scale.
# Deliberately NOT reusing TIER1_CONFIDENCE_FLOOR: this threshold applies to
# a differently-scaled, differently-sourced probability (a trained model's
# P(correct) over the whole disagreement pattern, not a raw mean logprob
# confidence on an already-unanimous vote), and conflating the two would
# silently let one threshold's tuning drag the other's meaning along with
# it. Re-validate before ever changing this: a refit on new data or a
# different held-out split can shift where 0.72 actually sits on the
# precision/coverage curve.
CALIBRATED_AUTO_THRESHOLD = 0.72

# 2026-08-17 (plan Phase 6, coronary safety gate). The calibrator's own
# val-set false positives (evaluation/tier_gate_cal_eval.py) cluster on
# coronary-artery-SEGMENT abbreviations -- LCX/LCx/LMCA repeatedly split-
# voted wrong even after prior_confirmation_count was ablated out, meaning
# the cause isn't a calibrator-feature bug but a retrieval-layer one:
# SapBERT's embedding space doesn't reliably separate a specific named
# branch ("Left circumflex coronary artery") from the generic parent
# concept ("Coronary artery structure"), so the ensemble gets handed a
# muddy candidate list and splits. Same failure shape independently
# confirmed twice more this session -- Phase 4's acronym-escalation grading
# (LAD -> wrong every time) and the plain AUTO-tier grading pass (LCx twice
# resolved to the generic parent instead of its specific segment) -- so this
# is a structural retrieval weak spot, not calibrator noise, and no amount
# of feature engineering on THIS calibrator fixes it. Quarantined here
# rather than left for the calibrator to (over)learn: matches
# TIER3_SIMILARITY_FLOOR/the Lab Value Fragile Concept Gate's own precedent
# of a narrow, evidence-scoped hard exclusion for a known-fragile pattern.
CORONARY_SEGMENT_TRAP_ABBREVIATIONS = {
    "lad", "lcx", "lmca", "rca", "pda", "om", "plv",
}
CORONARY_SEGMENT_TRAP_GENERIC_CONCEPTS = {"coronary artery structure"}


def _is_coronary_segment_trap(entity: dict, candidate_index, candidates: list) -> bool:
    """True when this entity matches the coronary-segment-abbreviation
    pattern above -- either the mention's own text IS one of the known
    abbreviations, or the candidate the calibrator would be asked to score
    resolved to the generic "Coronary artery structure" parent concept
    (the specific failure shape observed: a specific segment mention
    resolving to its own generic parent instead of the named branch).
    """
    text = (entity.get("original_text") or "").strip().lower()
    if text in CORONARY_SEGMENT_TRAP_ABBREVIATIONS:
        return True
    if candidate_index and candidates and 0 < candidate_index <= len(candidates):
        name = (candidates[candidate_index - 1].get("concept_name") or "").strip().lower()
        if name in CORONARY_SEGMENT_TRAP_GENERIC_CONCEPTS:
            return True
    return False


# 2026-08-17 (5-fresh-note validation run). One of that run's 3 TIER_1B
# false positives was 'S2', scored 0.704 -- the SAME embedding-collapse
# shape as the coronary trap (a short code with multiple unrelated clinical
# readings SapBERT can't reliably separate: S2 as a cardiac exam finding
# ["heart sound S2"] vs. a spinal level ["second sacral vertebra"]), just a
# different, more general vocabulary than named coronary branches. This
# pattern generalizes structurally: S1-S4 (heart sounds vs. sacral
# vertebrae), T1/T2 (thoracic vertebrae vs. MRI relaxation times vs. tumor
# stage), V1-V6 (ECG leads vs. cranial nerves) are all the same 1-2-letter-
# plus-1-2-digit shape colliding across unrelated clinical domains. Caught
# structurally via regex rather than an enumerated set (unlike the coronary
# trap, whose abbreviations are a small closed list) -- the SHAPE is the
# risk signal here, not a specific enumerable vocabulary.
SHORT_ALPHANUMERIC_CODE_RE = re.compile(r"^[A-Za-z]{1,2}[0-9]{1,2}$")


def _is_short_alphanumeric_code(entity: dict) -> bool:
    """True when the mention's own text is a bare short alphanumeric code
    (S2, T1, V12, ...) -- see the constant's docstring above. Deliberately
    does not also check candidate concept names (unlike the coronary trap):
    the risk here is inherent to the mention's SHAPE, independent of which
    domain the top candidate happened to land in.
    """
    text = (entity.get("original_text") or "").strip()
    return bool(SHORT_ALPHANUMERIC_CODE_RE.match(text))


# ==========================================================================
# Step A -- isolated clinical-meaning definition
# ==========================================================================

MEANING_SYSTEM_PROMPT = (
    "You are a clinical terminology expert reading a single clinical note. "
    "Your only job right now is to state what a highlighted text span means "
    "clinically, using the note's own context. You have not been shown any "
    "candidate concept list and must not anticipate or guess at one."
)


ALLERGY_MEANING_INSTRUCTION = (
    "ALLERGY NOTE: this entity's assertion status is ALLERGY. That means the "
    "note is documenting a known or reported patient allergy/adverse "
    "reaction to this substance -- NOT that the patient is currently taking "
    "or being prescribed it. State the clinical meaning as the patient's "
    "allergic disposition or reaction to the substance (e.g. \"the patient "
    "has a documented allergy to X\"), not as the substance's use as a "
    "medication.\n\n"
)


def _clinical_meaning_prompt(entity: dict) -> str:
    assertion = entity.get("assertion_status", "PRESENT")
    allergy_instruction = ALLERGY_MEANING_INSTRUCTION if assertion == "ALLERGY" else ""
    return (
        "ENTITY:\n"
        f"  text as written: {entity.get('original_text')!r}\n"
        f"  after abbreviation expansion: {entity.get('expanded_text')!r}\n"
        f"  extractor label: {entity.get('gliner_label')}\n"
        f"  assertion: {assertion} / "
        f"experiencer: {entity.get('experiencer', 'PATIENT')}\n\n"
        f"SECTION: {entity.get('section_name') or 'unknown'}\n"
        f"CONTEXT: ...{entity.get('local_context', '')}...\n\n"
        f"{allergy_instruction}"
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


# 2026-08-16 (Option 2 of the user's 3-path proposal following the 2026-08-15
# rejected experiment -- docs/2026-08-15_Phase2_TierGate_Validation.md).
# ISOLATED A/B CLAUSE, qwen2.5:3b ONLY. The blanket "don't require every
# detail" rule tried on 2026-08-15 was rejected because it let ALL THREE
# models rubber-stamp wrong matches on qualifier/fragment spans -- but
# qualifier_fragment_precheck() (below) now removes those spans from the
# ensemble's job entirely, so the residual risk of a per-model relaxation is
# narrower than it was in that experiment. This clause is deliberately
# scoped to hierarchical subsumption specifically (a genuine parent/child
# SNOMED relationship), not general specificity forgiveness, and is applied
# ONLY to qwen2.5:3b's Step B prompt -- llama3.2:3b/phi4-mini's prompts are
# UNCHANGED, since the diagnosed problem was qwen's asymmetric strictness,
# not theirs. Because Step B evaluates candidates ONE AT A TIME (the whole
# point of the sequential design -- see _evaluate_one_model()'s docstring),
# the model cannot literally verify "no more specific candidate exists
# elsewhere in the list"; the clause instead makes that tradeoff explicit
# (rank order, no second look) rather than pretending the model has
# information it does not.
QWEN_SUBSUMPTION_CLAUSE = (
    "5. HIERARCHICAL SUBSUMPTION: candidates are being checked ONE AT A TIME "
    "in ranked order; if you reject this one, it will not be reconsidered "
    "later. If this candidate is a correct but less-specific (broader) or "
    "more-specific (narrower) SNOMED/RxNorm relative of the precise concept "
    "described -- not a DIFFERENT concept entirely -- accept it as a match "
    "rather than rejecting it purely for lacking the note's full specificity "
    "(severity, laterality, exact subtype). Still reject a candidate that "
    "names a different clinical concept, not merely a less-detailed one."
)

# 2026-08-16, same session as the allergy-context retrieval fix
# (src/assertion.py's STATUS_ALLERGY, src/normalization/orchestrator.py's
# _apply_allergy_nonstandard_exact_override()). Empirically diagnosed via
# mollm_tier_gate_decisions.models trail data on the 6-note allergy re-run
# (docs/2026-08-16_Shadow_Run_Precision_At_Scale.md): with the RETRIEVAL fix
# landed and correctly surfacing "Allergy to X" as candidate #1, the
# ENSEMBLE still split votes on it, 0/19 reaching Tier 1. Root cause was in
# rule 3 below, not a missing-context problem -- assertion_status was
# already in both prompts. Rule 3 tells models to ignore assertion status
# when judging concept identity, which is correct for negation (a denied
# entity still names the same concept) but actively WRONG for ALLERGY: the
# correct concept for an allergy-context substance mention IS a different
# concept (the allergic-disposition finding), not the substance itself, and
# stock rule 3 pushed models toward rejecting the (correct) allergy
# candidate as "a different concept" per rule 4. Confirmed in the raw trail:
# phi4-mini's Step A never even mentioned allergy ("Morphine is an opioid
# medication..."), and qwen2.5:3b rejected "Allergy to morphine" as "too
# specific" after its own Step A hedged into a generic "may cause an
# allergy" framing instead of stating the patient's actual disposition.
ALLERGY_CONTEXT_CLAUSE = (
    "ALLERGY EXCEPTION TO RULE 3: this entity's assertion status is ALLERGY. "
    "Unlike negation, an ALLERGY assertion means the CORRECT concept is the "
    "patient's allergic disposition/reaction to the substance, not the "
    "substance itself -- these are genuinely different concepts here, and "
    "that is expected, not an error. If the candidate names an allergy or "
    "adverse-reaction concept for this same substance (e.g. 'Allergy to X', "
    "'X allergy', 'Allergic reaction caused by X'), treat that as the "
    "correct match. Do NOT reject it under rule 4 on the grounds that it "
    "names a 'different concept' from the substance itself -- for an "
    "ALLERGY-status entity, the allergy/reaction concept IS the correct one."
)


def _binary_match_prompt(entity: dict, candidate: dict, clinical_meaning: str,
                         extra_rule: str = None) -> str:
    basis = candidate.get("match_basis", "semantic_similarity")
    rules = (
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
        "value to an unrelated test). Do not force a match.\n"
    )
    if extra_rule:
        rules += extra_rule + "\n"
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
        f"{rules}\n"
        "Does this candidate concept match the clinical meaning stated "
        'above? Reply with JSON: {"match": true or false, "reasoning": '
        '"<one sentence>"}'
    )


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

    # 2026-08-16 Option 2 A/B clause -- see QWEN_SUBSUMPTION_CLAUSE's own
    # comment. Model-name-keyed rather than a parameter threaded through
    # run_two_step_ensemble()/route_tier(), since this is specifically a
    # per-model prompt difference, not a per-entity or per-run one.
    extra_rules = []
    if client.model_name.startswith("qwen"):
        extra_rules.append(QWEN_SUBSUMPTION_CLAUSE)
    # ALLERGY_CONTEXT_CLAUSE applies to every model (see its own comment) --
    # all three models showed the split-vote failure, not just qwen.
    if entity.get("assertion_status") == "ALLERGY":
        extra_rules.append(ALLERGY_CONTEXT_CLAUSE)
    extra_rule = "\n".join(extra_rules) if extra_rules else None

    step_a_degenerate = bool(meaning_raw.get("degenerate_generation"))
    trail = []
    for i, cand in enumerate(candidates, 1):
        try:
            raw = client.complete(
                MATCH_SYSTEM_PROMPT,
                _binary_match_prompt(entity, cand, clinical_meaning, extra_rule=extra_rule),
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
# Qualifier-fragment precheck (Option 1, 2026-08-16 user proposal)
# ==========================================================================

def qualifier_fragment_precheck(entity: dict) -> dict:
    """HITL_REQUIRED without spending any model calls, for a standalone
    Qualifier-labeled span -- a laterality, generic modifier, or similar
    ("left", "right", "multiple", "third", "R", "Cranial"). These are not
    independently linkable clinical concepts on their own, and asking the
    ensemble to judge whether some SNOMED code IS "left" is the exact
    question that produced both failure modes recorded in
    docs/2026-08-15_Phase2_TierGate_Validation.md: qwen correctly refusing
    (a split, no harm beyond lost coverage) or a loosened match prompt
    letting all three models rubber-stamp a wrong code onto it (the 5.9%
    precision collapse). Removing these spans from the ensemble's job
    entirely is safer than asking any prompt to get them right.

    Scoped to gliner_label == "Qualifier" specifically, confirmed against
    real data (not a hand-rolled word list) -- a DB query for the exact
    fragment entities that caused the 2026-08-15 collapse ('left', 'right',
    'multiple', 'third', 'R', 'Cranial') showed all of them label Qualifier,
    while single-word entities that ARE real, independently linkable
    concepts ('chest' -> Anatomy, 'pain' -> Symptom) carry a different
    label and are untouched by this check.
    """
    if (entity.get("gliner_label") or "").strip() == "Qualifier":
        return {"tier": TIER_5_TRUE_AMBIGUITY, "mollm_routing_decision": "HITL_REQUIRED",
                "queue_reason": "standalone_qualifier_span", "final_candidate_index": None,
                "composite_confidence": None,
                "routing_basis": (
                    "Tier 5: gliner_label=Qualifier -- a standalone laterality/"
                    "modifier span is not an independently linkable clinical "
                    "concept; routed to HITL without spending an ensemble call."),
                "models": []}
    return None


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
    # 2026-08-16 (same fix as src/normalization/orchestrator.py's
    # normalize_entity(), same root cause): under hybrid retrieval,
    # candidates[0] is whichever concept RRF fusion ranked first, not
    # whichever has the best dense score -- checking the floor against only
    # candidates[0] would re-reject entities normalize_entity()'s own
    # (already-fixed) floor check just let through, undoing that fix at the
    # one place it actually matters for routing. Checked against the pool's
    # best dense score instead; entities with no numeric similarity_score at
    # all (e.g. Tier 1/2 exact/synonym hits, similarity_score=1.0 by
    # convention) are unaffected since 1.0 is always >= any real floor.
    scored = [c.get("similarity_score") for c in candidates
              if isinstance(c.get("similarity_score"), (int, float))]
    pool_max_dense = max(scored) if scored else None
    if pool_max_dense is not None and pool_max_dense < TIER3_SIMILARITY_FLOOR:
        return {"tier": TIER_5_TRUE_AMBIGUITY, "mollm_routing_decision": "HITL_REQUIRED",
                "queue_reason": "below_similarity_floor", "final_candidate_index": None,
                "composite_confidence": None,
                "routing_basis": (f"Tier 5: best candidate similarity {pool_max_dense} < "
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

def route_tier(entity: dict, model_results: list = None, clients: dict = None,
               calibrator=None, conn=None) -> dict:
    """Runs the Tier 1-5 gate for one Stage 2b LOW-tier entity record.

    Order: qualifier-fragment precheck -> Tier 3 fast path -> Tier 5
    pre-check (all three free) -> full two-step ensemble -> Tier 1/2/4/5
    based on the ensemble's three verdicts.

    `model_results`, when passed, skips running the ensemble (used by
    scripts/run_tier_gate_batch.py to separate "call the models" from
    "apply the routing table" for testability and for re-scoring a stored
    run without re-paying for inference).

    `calibrator`/`conn` (2026-08-17, plan Phase 6): both default to None,
    which reproduces every existing behavior exactly -- a non-unanimous
    vote always falls through to TIER_4_ENSEMBLE_SPLIT, unchanged. Passing
    a fitted src.mollm_tier_calibrator.ConsensusCalibrator (and a DuckDB
    connection for its prior-confirmation-count feature) is what activates
    the new TIER_1B_CALIBRATED_AUTO_VALIDATED escape hatch below -- and even
    then, ONLY for entities that already failed every existing Tier 1/2/3
    rule; nothing above this point in the function changes at all.
    """
    qualifier = qualifier_fragment_precheck(entity)
    if qualifier:
        return qualifier
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
                "calibrated_score": None,
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
                    "calibrated_score": None,
                    "routing_basis": (f"unanimous SUPPORTED_1 but composite_confidence "
                                      f"{composite_confidence} < {TIER1_CONFIDENCE_FLOOR}"),
                    "models": model_results}
        return {"tier": TIER_1_AUTO_VALIDATED, "mollm_routing_decision": "AUTO_VALIDATED",
                "queue_reason": None, "final_candidate_index": 1,
                "composite_confidence": composite_confidence,
                "calibrated_score": None,
                "routing_basis": (f"3/3 unanimous SUPPORTED_1, "
                                  f"composite_confidence {composite_confidence}"),
                "models": model_results}

    if unanimous and top_verdict.startswith("RE_RANK_TO_CANDIDATE_"):
        n = int(top_verdict.rsplit("_", 1)[1])
        return {"tier": TIER_2_AUTO_RESOLVED, "mollm_routing_decision": "AUTO_RESOLVED",
                "queue_reason": None, "final_candidate_index": n,
                "composite_confidence": composite_confidence,
                "calibrated_score": None,
                "routing_basis": (f"3/3 unanimous re-rank to candidate {n}, "
                                  f"composite_confidence {composite_confidence}"),
                "models": model_results}

    if unanimous and top_verdict == "NONE_CORRECT":
        return {"tier": TIER_5_TRUE_AMBIGUITY, "mollm_routing_decision": "HITL_REQUIRED",
                "queue_reason": "verdict_none_correct", "final_candidate_index": None,
                "composite_confidence": composite_confidence,
                "calibrated_score": None,
                "routing_basis": "3/3 unanimous NONE_CORRECT -- no usable resolution produced",
                "models": model_results}

    # 2026-08-17 (plan Phase 6). Every non-unanimous case reaches here.
    # Before falling through to the unconditional HITL routing below, give
    # a fitted calibrator one chance to say this specific split is likely
    # correct anyway -- ONLY when calibrator/conn were both actually
    # supplied (None/None reproduces prior behavior exactly, see this
    # function's own docstring) and ONLY when the plurality verdict names an
    # actual candidate (SUPPORTED_1 or a RE_RANK target) -- a plurality of
    # NONE_CORRECT has no candidate to promote, so the calibrator is never
    # consulted for that shape regardless of score.
    # 2026-08-18 (tier-gate audit fix #2): tracked at this scope (not just
    # inside the `if candidate_index is not None:` block below) so the final
    # fallback return can report it accurately either way -- None when the
    # calibrator was never reached at all (candidate_index was None, or
    # calibrator/conn weren't supplied), the real computed value when it
    # was consulted but didn't clear CALIBRATED_AUTO_THRESHOLD. Previously
    # this second case computed the score then discarded it entirely,
    # making "never consulted" indistinguishable from "consulted and
    # scored low" after the fact.
    calibrated_score = None

    if calibrator is not None and conn is not None:
        candidate_index = None
        if top_verdict == "SUPPORTED_1":
            candidate_index = 1
        elif top_verdict.startswith("RE_RANK_TO_CANDIDATE_"):
            candidate_index = int(top_verdict.rsplit("_", 1)[1])

        if candidate_index is not None:
            candidates = entity.get("candidates") or []

            # Coronary segment trap (see constant block above): a known-
            # fragile retrieval pattern, quarantined BEFORE the calibrator
            # ever sees it -- calibrator.score() is not called at all for a
            # trapped entity, not merely overridden after the fact, so no
            # training data (evaluation/tier_gate_cal_eval.py) or future fit
            # can accidentally re-learn its way around this gate.
            if _is_coronary_segment_trap(entity, candidate_index, candidates):
                return {"tier": TIER_4_ENSEMBLE_SPLIT, "mollm_routing_decision": "HITL_REQUIRED",
                        "queue_reason": "coronary_segment_trap", "final_candidate_index": None,
                        "composite_confidence": composite_confidence,
                        "calibrated_score": None,  # bypassed BEFORE calibrator.score() is called
                        "routing_basis": (
                            f"non-unanimous verdicts {dict(vote_counts)}; calibrator bypassed -- "
                            f"coronary-artery-segment trap (known SapBERT embedding-collapse "
                            f"pattern, see CORONARY_SEGMENT_TRAP_ABBREVIATIONS)"),
                        "models": model_results}

            # Short alphanumeric code trap (see constant block above): same
            # bypass-before-scoring discipline as the coronary trap, for the
            # structurally similar S2/T1/V12-shaped collision pattern.
            if _is_short_alphanumeric_code(entity):
                return {"tier": TIER_4_ENSEMBLE_SPLIT, "mollm_routing_decision": "HITL_REQUIRED",
                        "queue_reason": "short_alphanumeric_code_trap",
                        "final_candidate_index": None,
                        "composite_confidence": composite_confidence,
                        "calibrated_score": None,  # bypassed BEFORE calibrator.score() is called
                        "routing_basis": (
                            f"non-unanimous verdicts {dict(vote_counts)}; calibrator bypassed -- "
                            f"short alphanumeric code trap (known SapBERT embedding-collapse "
                            f"pattern, see SHORT_ALPHANUMERIC_CODE_RE)"),
                        "models": model_results}

            from src.mollm_tier_calibrator import (
                build_feature_context, count_prior_confirmations)
            chosen_concept_id = None
            if 0 < candidate_index <= len(candidates):
                chosen_concept_id = candidates[candidate_index - 1].get("omop_concept_id")
            prior_count = count_prior_confirmations(
                conn, entity.get("original_text"), chosen_concept_id)
            context = build_feature_context(entity, model_results, prior_count)
            calibrated_score = calibrator.score(context)

            if calibrated_score is not None and calibrated_score >= CALIBRATED_AUTO_THRESHOLD:
                return {"tier": TIER_1B_CALIBRATED_AUTO_VALIDATED,
                        "mollm_routing_decision": "AUTO_VALIDATED", "queue_reason": None,
                        "final_candidate_index": candidate_index,
                        "composite_confidence": composite_confidence,
                        "calibrated_score": calibrated_score,
                        "routing_basis": (
                            f"non-unanimous verdicts {dict(vote_counts)}, but "
                            f"ConsensusCalibrator scored {calibrated_score} >= "
                            f"{CALIBRATED_AUTO_THRESHOLD} (prior_confirmation_count="
                            f"{prior_count})"),
                        "models": model_results}

    return {"tier": TIER_4_ENSEMBLE_SPLIT, "mollm_routing_decision": "HITL_REQUIRED",
            "queue_reason": "ensemble_split", "final_candidate_index": None,
            "composite_confidence": composite_confidence,
            "calibrated_score": calibrated_score,
            "routing_basis": (f"non-unanimous verdicts: {dict(vote_counts)}"
                              + (f" ({n_excluded} model(s) excluded as "
                                 f"degenerate/errored)" if n_excluded else "")
                              + (f"; calibrator scored {calibrated_score} < "
                                 f"{CALIBRATED_AUTO_THRESHOLD}" if calibrated_score is not None else "")),
            "models": model_results}


# ==========================================================================
# Persistence (2026-08-16, production deploy -- gate wired in, KG3 writes
# stay dry-run per user decision)
# ==========================================================================

def store_tier_decision(decision: dict, entity_id: str, note_id: str, conn,
                        is_test: bool = False) -> dict:
    """Persists one route_tier() decision to its own table,
    mollm_tier_gate_decisions -- deliberately SEPARATE from
    src.mollm_ensemble.store_decision()'s mollm_decisions table rather than
    shoehorned into it: the two artifacts have different shapes (route_tier()
    has no ensemble_agreement/citation_verified/mode -- those are
    contradiction-audit concepts the two-step CoT doesn't use) and mixing
    them would make every downstream reader (evaluation/cal_eval.py,
    src/hitl_queue.py) guess which schema a given row follows.

    Mutates a COPY of `decision` with a freshly generated mollm_call_id (plus
    entity_id/note_id) and returns it -- callers pass the returned dict
    straight to src.kg3_ingestion.ingest_auto_decision() for Tier 1/2/3
    decisions, so the call_id that got written here is the same one that
    would appear in a KG3 write (real or dry-run).

    `is_test` mirrors store_decision()'s own flag: True for smoke-test/
    diagnostic runs against synthetic or held-out data, False for real
    corpus processing -- kept as an explicit parameter (not inferred) so a
    caller can never write a real-looking row by accident.
    """
    decision = dict(decision)
    decision["mollm_call_id"] = decision.get("mollm_call_id") or str(uuid.uuid4())
    decision["entity_id"] = entity_id
    decision["note_id"] = note_id

    conn.sql("""
    CREATE TABLE IF NOT EXISTS mollm_tier_gate_decisions (
        mollm_call_id VARCHAR PRIMARY KEY,
        entity_id VARCHAR,
        note_id VARCHAR,
        tier VARCHAR,
        mollm_routing_decision VARCHAR,
        queue_reason VARCHAR,
        final_candidate_index INTEGER,
        composite_confidence DOUBLE,
        routing_basis VARCHAR,
        models JSON,
        is_test BOOLEAN DEFAULT FALSE
    );
    """)
    # 2026-08-18 (tier-gate audit fix #2): calibrator.score() was being
    # computed for every non-unanimous, non-trapped entity but only ever
    # surfacing in routing_basis TEXT when it cleared CALIBRATED_AUTO_THRESHOLD
    # -- for anything that scored below the threshold (i.e. every entity that
    # STAYED in TIER_4_ENSEMBLE_SPLIT after calibrator consultation), the
    # score was computed then silently discarded, making it impossible to
    # tell "the calibrator wasn't consulted" apart from "the calibrator
    # scored it low" after the fact. ADD COLUMN IF NOT EXISTS because
    # CREATE TABLE IF NOT EXISTS above is a no-op against the real,
    # already-existing production table.
    conn.sql("ALTER TABLE mollm_tier_gate_decisions "
            "ADD COLUMN IF NOT EXISTS calibrated_score DOUBLE;")
    for stmt in provenance_alter_statements("mollm_tier_gate_decisions"):
        conn.sql(stmt)
    conn.sql(f"""
    INSERT INTO mollm_tier_gate_decisions
    (mollm_call_id, entity_id, note_id, tier, mollm_routing_decision,
     queue_reason, final_candidate_index, composite_confidence,
     routing_basis, models, is_test, calibrated_score, {provenance_column_sql()})
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {provenance_placeholders()})
    ON CONFLICT (mollm_call_id) DO NOTHING;
    """, params=[
        decision["mollm_call_id"], entity_id, note_id, decision.get("tier"),
        decision.get("mollm_routing_decision"), decision.get("queue_reason"),
        decision.get("final_candidate_index"), decision.get("composite_confidence"),
        decision.get("routing_basis"),
        json.dumps(decision.get("models"), default=str), is_test,
        decision.get("calibrated_score"),
        *provenance_params(),
    ])
    return decision
