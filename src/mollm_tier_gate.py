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
import os
import re
import uuid

from src.llm_client import (
    LLMUnavailable,
    build_clients,
    extract_verdict_confidence,
    parse_json_response,
)
from src.normalization.constants import TIER3_SIMILARITY_FLOOR
from src.physexam_shorthand import PHYSEXAM_SHORTHAND_MATCH_BASIS
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

# 2026-08-20 (fresh25 root-cause finding, evaluation/grade_fresh25_by_tier.py
# and docs/2026-08-20_Session_Results_And_Status.md). A unanimous re-rank
# (TIER_2_AUTO_RESOLVED) a fitted ConsensusCalibrator scores as
# likely-correct anyway promotes here -- structurally separate from
# TIER_1B for the same reason TIER_1B is separate from TIER_1: the
# calibrator was fit on split-vote (non-unanimous) examples, a materially
# different feature distribution than Tier 2's 100%-is_ambiguous,
# zero-vote-disagreement population (confirmed directly: every single
# TIER_2_AUTO_RESOLVED decision in the DB has is_ambiguous=True, whereas
# "unanimous" here reflects the 3 models sharing a common bias on an
# already-shaky retrieval signal, not independent verification -- see the
# TIER_2_AUTO_RESOLVED comment below). NOT added to AUTO_TIERS by default;
# see that set's own comment for why, and for the shadow-validation this
# tier needs before it ever could be.
TIER_2B_CALIBRATED_AUTO_RESOLVED = "TIER_2B_CALIBRATED_AUTO_RESOLVED"

# 2026-08-19 (temporary, conservative gating change -- see
# docs/2026-08-19_Lab_Procedure_Vs_Observable_Entity_Finding.md), UPDATED
# 2026-08-20 with the fresh25 re-measurement. TIER_2_AUTO_RESOLVED stays
# excluded from AUTO_TIERS: the fresh-note re-evaluation this exclusion was
# pending measured 16.2% clean-span precision (11/68) -- essentially
# unchanged from the original ~20% baseline, confirming the earlier fixes
# did NOT recover it and this exclusion should stay in place, not be a
# transitional state. Root cause dug into directly: 100% of
# TIER_2_AUTO_RESOLVED decisions in the DB are flagged is_ambiguous=True by
# retrieval -- structural, not incidental. Tier 2 requires all 3 models to
# unanimously re-rank AWAY from retrieval's own top candidate, which only
# happens when that candidate already looked shaky; "3/3 unanimous" in
# this population is more likely to reflect the 3 models sharing a common
# bias than independently verifying the same correct answer. A
# deterministic fast path therefore isn't the right tool here (see
# TIER_2B_CALIBRATED_AUTO_RESOLVED above) -- route_tier() now gives a
# fitted calibrator a chance to promote a specific Tier-2-shaped decision
# there instead of a blanket rule, but TIER_2B is not in this set either,
# pending the shadow-validation its own comment describes.
AUTO_TIERS = {TIER_1_AUTO_VALIDATED, TIER_3_AUTO_VALIDATED,
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

# 2026-08-18 (5-note validation run, 'LMCA' finding). 'LMCA' reached 3/3
# unanimous TIER_1_AUTO_VALIDATED on the WRONG candidate ("Coronary artery
# stenosis" instead of "Structure of left main coronary artery") --
# CORONARY_SEGMENT_TRAP_ABBREVIATIONS already lists "lmca", but at the time
# neither trap ran on the unanimous path at all (see the elevated gate
# below). This regex is the complementary, forward-looking half of that
# fix: a short (3-4 letter), ALL-CAPS, pure-alphabetic mention is the same
# SapBERT-collapse risk shape as the digit-suffixed one above, for
# abbreviations NOT yet on the coronary list (a new segment name, a
# different organ system's shorthand) -- caught structurally rather than
# requiring every future collision to be found the hard way and manually
# enumerated first. Deliberately case-sensitive (upper only): a lowercase
# or Title-Case 3-4 letter word is far more likely to be ordinary prose
# than clinical shorthand, and the note-writing convention for these
# abbreviations is consistently all-caps in this corpus.
SHORT_ALPHA_CODE_RE = re.compile(r"^[A-Z]{3,4}$")


def _is_short_alphanumeric_code(entity: dict) -> bool:
    """True when the mention's own text is a bare short alphanumeric code
    (S2, T1, V12, ...) OR a short, ALL-CAPS pure-alphabetic clinical
    shorthand (LAD, LCX, RCA, PDA, LMCA, ...) -- see the constants'
    docstrings above. Deliberately does not also check candidate concept
    names (unlike the coronary trap): the risk here is inherent to the
    mention's SHAPE, independent of which domain the top candidate happened
    to land in.
    """
    text = (entity.get("original_text") or "").strip()
    return bool(SHORT_ALPHANUMERIC_CODE_RE.match(text) or SHORT_ALPHA_CODE_RE.match(text))


def _fragile_shorthand_trap(entity: dict, candidate_index, candidates: list):
    """Checks the coronary-segment trap and the short-code trap together,
    returning (True, queue_reason) on whichever fires first, else
    (False, None).

    Factored out 2026-08-18 ("elevate the gate" fix, 5-note validation run)
    so the SAME check guards every AUTO-tier grant this fragile-shorthand
    pattern can reach -- TIER_1_AUTO_VALIDATED (unanimous SUPPORTED_1) and
    TIER_1B_CALIBRATED_AUTO_VALIDATED (calibrator-rescued split) -- without
    the two call sites drifting out of sync. Previously only the calibrator
    path ran these checks at all, so a genuinely unanimous 3/3 vote (LMCA ->
    "Coronary artery stenosis") sailed straight into TIER_1_AUTO_VALIDATED
    with zero trap protection: unanimity was treated as proof the models
    got it right, but the failure mode here is that all three models see
    the SAME muddy, SapBERT-collapsed candidate list -- unanimous agreement
    on a bad candidate list is not evidence the candidate is correct.
    """
    if _is_coronary_segment_trap(entity, candidate_index, candidates):
        return True, "coronary_segment_trap"
    if _is_short_alphanumeric_code(entity):
        return True, "short_alphanumeric_code_trap"
    return False, None


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
        # 2026-08-18 (user proposal, "cold start" prompt tightening): a
        # single-phrase definition rather than a free sentence or two gives
        # Step B a tighter, more atomic string to compare each candidate
        # against -- "a beta-blocker medication" is less ambiguous to judge
        # a candidate against than a paragraph that drifts into patient
        # history. Kept to one field, same schema, same downstream contract
        # -- every existing consumer of clinical_meaning is unaffected.
        "TASK: Based ONLY on the note text above, provide a concise, "
        "single-phrase clinical definition of what this entity refers to "
        '(e.g. "a beta-blocker medication", "a diagnosis of high blood '
        'pressure", "a surgical procedure on the knee"). Do not name a '
        "database code, ontology term, or vocabulary identity. Do not "
        "explain the patient's history. Define the term only.\n\n"
        'Reply with JSON: {"clinical_meaning": "<single-phrase definition>", '
        '"reasoning": "<one short sentence>"}'
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


# 2026-08-19 ("MCV/MCHC/RDW problem", grading-pass finding). Corpus-scale
# grading of TIER_2_AUTO_RESOLVED (evaluation/tier_gate_grading.py) found
# 20% precision -- traced to a specific, repeatable pattern: Stage 2b's Tier
# 3 semantic search already ranks the correct Procedure/"determination"-
# class concept at candidate #1 (src.normalization.tier_retrieval's
# _prefer_lab_procedure_over_observable(), 78/78-exceptionless corpus
# evidence for this exact class pair), but all 3 ensemble models
# unanimously REJECT candidate #1 and RE_RANK to an Observable-Entity-class
# sibling instead (e.g. "RDW-13" -> correct "Red cell distribution width
# determination" at #1, rejected; wrong "Red blood cell distribution width"
# at #2, unanimously accepted) -- every model's own reasoning cites the
# Observable-Entity name's closer literal wording to the abbreviation as
# the deciding factor, the same anchoring-on-surface-form failure mode
# CONDITION_VS_OBSERVATION_PRIOR already exists to counteract for a
# different class pair. Applied unconditionally for Lab-Test-labeled
# entities (cheap -- a no-op instruction when this specific pattern isn't
# present in the candidate set) rather than gated on detecting the pattern
# first, since the check would need the model's OWN judgment of "which
# candidate is which class" to gate on, circular for what this fixes.
def _binary_match_prompt(entity: dict, candidate: dict, clinical_meaning: str,
                         extra_rule: str = None) -> str:
    basis = candidate.get("match_basis", "semantic_similarity")
    rules = (
        "RULES:\n"
        "1. SEMANTIC MATCH: judge the candidate strictly against the CLINICAL "
        "MEANING stated above, not the raw text spelling or the candidate's "
        "match score -- does it represent the exact same clinical idea, even "
        "if spelled completely differently? Do not reject a candidate just "
        "because the words do not match the original text.\n"
    )
    # 2026-08-18 (user proposal, "cold start" prompt-bleed fix): rule 2 used
    # to be printed for EVERY candidate regardless of its own match_basis --
    # confirmed live to cause exactly the bleed this guards against: a
    # candidate whose real basis was exact_text got justified in a model's
    # own reasoning as "verified to be a brand alias in the SNOMED
    # vocabulary", language borrowed straight from this rule's text despite
    # not applying to that candidate at all. Only ever include it now when
    # THIS candidate's own basis is one of the verified-alias kinds.
    # Numbering intentionally stays 1/[2]/3/4/5 rather than renumbering when
    # omitted -- ALLERGY_CONTEXT_CLAUSE hardcodes "RULE 3" and
    # QWEN_SUBSUMPTION_CLAUSE hardcodes "5.", both must keep referring to the
    # same rules whether or not rule 2 is present.
    #
    # 2026-08-18, extended beyond verified_brand_alias to also cover
    # verified_lab_test_alias (src.normalization.tier_retrieval's
    # _LAB_TEST_ALIASES -- panel shorthand like "CHEM-7" -> the Basic
    # Metabolic Panel LOINC concept, and historical single-test synonyms
    # like "SGPT" -> ALT's concept) -- same underlying shape (a curated,
    # verified lookup that legitimately differs from the raw entity text),
    # so it gets the same trust instruction rather than being silently
    # treated as an unverified semantic_similarity guess. The rule names
    # the SPECIFIC basis string rather than a generic "trust this" so the
    # model sees the real value, not a paraphrase it could misattribute to
    # a different candidate.
    _VERIFIED_ALIAS_BASES = {
        "verified_brand_alias": "a mathematically verified terminology-database link",
        "verified_lab_test_alias": "a curated, human-verified lab-test-shorthand mapping "
                                   "(a panel abbreviation or a historical single-test synonym)",
    }
    if basis in _VERIFIED_ALIAS_BASES:
        rules += (
            f"2. This candidate's basis is {basis} -- {_VERIFIED_ALIAS_BASES[basis]}. "
            "Do not reject it merely because the spelling differs from the "
            "entity text.\n"
        )
    rules += (
        "3. Ignore assertion/negation status when judging the CONCEPT match "
        '-- a negated entity ("denies fever") still maps to its concept '
        '("Fever") if the name matches; you are labeling which concept the '
        "text refers to, not diagnosing.\n"
        "4. STRICT DOMAIN MISMATCH: reject a candidate that is a distinct or "
        "clinically unrelated concept (e.g. mapping a symptom to a "
        "biological genus, or a medication to a surgical tool). Do not "
        "force a match.\n"
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


# ==========================================================================
# Contextual candidate disambiguation (2026-08-18, "cold start" fix). See
# src.normalization.constants.CONTEXTUAL_CANDIDATES_ENABLED (its own comment
# there) -- the two flags share one env var and must be enabled together:
# widening the SNOMED domain restriction without this evaluation change just
# hands Step B a bigger list to blindly accept-first from, which can make the
# arbitrary-pick problem WORSE (more candidates, still no real comparison).
#
# WHY NOT A SINGLE 1-TO-N COMPARATIVE PROMPT (the naive version of "let
# MoLLM see everything"). Already tried and rejected in this exact codebase,
# 2026-08-14 (scripts/experiment_3b_voting.py's _format_candidates()/
# evaluate_candidates_sequentially() docstrings): a dense 1-to-N candidate
# list measurably let 3B models detach a candidate's evidence tag from its
# own bracket index and misattribute it to the highest-scored candidate
# instead (two real 'lasix' cases both moved to the highest-scored
# LASCUFLOXACIN candidate instead of the correctly-tagged furosemide one) --
# a formatting fix (isolating each tag on its own line) was tried and still
# not reliable enough. That is why Step B asks about exactly ONE candidate
# per call: no bracket index for the model to mis-track.
#
# THE FIX HERE keeps that one-candidate-per-call isolation for every
# individual judgment (unchanged, same measured-safe shape), but stops
# discarding information: instead of returning at the FIRST accepted
# candidate, every candidate is evaluated independently. Only when 2+ come
# back independently accepted (the genuine "cold start" case -- Stage 2b
# handed over real ambiguity it couldn't resolve on its own, e.g. two SNOMED
# concepts sharing an identical name across the Condition/Disorder vs.
# Observation/Morph-Abnormality axes) does a SEPARATE, small comparative
# call run, scoped to just that short accepted subset (usually 2, rarely 3
# candidates) rather than the full original list -- far less surface for the
# same index-confusion failure mode, and it only fires when actually needed,
# so the common (already-unambiguous) case pays no extra cost.
# 2026-08-18: promoted to default-ON alongside
# src.normalization.constants.CONTEXTUAL_CANDIDATES_ENABLED (same env var,
# same opt-out semantics -- see that flag's own comment for why leaving this
# off-by-default was silently reproducing the wound-dehiscence-class bug on
# every note that didn't manually export the var).
EXHAUSTIVE_CANDIDATE_EVAL_ENABLED = os.environ.get(
    "CNSP_CONTEXTUAL_CANDIDATES", "1").strip().lower() not in ("0", "false", "no")


TIEBREAK_SYSTEM_PROMPT = (
    "You are a clinical terminology validator. Several candidate concept "
    "codes have each independently been judged a plausible match for the "
    "same text span -- your job now is to pick the single best one using "
    "the note's own context, not to re-decide whether any of them match."
)


# 2026-08-18 ("cold start" tiebreak, data-grounded version). A user-proposed
# hard rule ("PREFER CONDITIONS ... if context implies a diagnosis") was
# evaluated against this corpus's own sizing data BEFORE being adopted, per
# this codebase's own "verify hypotheses against real data first" discipline
# -- and rejected as-is: of 14 known Condition/Disorder-vs-Observation/Morph-
# Abnormality same-name pairs (measured across the 109-note test corpus,
# 2026-08-18), 11 go the OPPOSITE direction (gold prefers Observation/Morph-
# Abnormality), confirmed live by an unprompted 3/3 unanimous WRONG
# auto-resolve on 'Osteopenia' the very session this was found. This constant
# encodes only the WEAK, corpus-measured prior (11/14 = 78.6%, not the
# 78/78-exceptionless bar _prefer_lab_procedure_over_observable was held to)
# -- injected as a tie-breaking HINT the model can still override with real
# context, never as an absolute rule, and only when the actual pattern (same
# name, Condition/Disorder vs Observation/Morph-Abnormality) is present in
# THIS candidate set specifically, not a blanket addition to every tiebreak.
# 2026-08-18, v2 ("sledgehammer" strengthening). v1's softly-worded, stats-
# cited prior ("11 of 14... treat as a weak prior") measurably UNDER-powered
# against a 3B model's pretraining association between disease-sounding
# terms ("Carcinoma") and Condition/Disorder framing: live-tested on
# 'Metastatic Renal Cell Cancer', all 3 models cited "diagnosis"/
# "documentation style" to override the prior toward the WRONG (Condition)
# answer, despite gold wanting Observation. A follow-up section_name check
# across all 11 known cases also killed the section-conditioned rule
# proposed as an alternative: the only two Chief Complaint cases both want
# Observation, and 10/11 want Observation regardless of section -- the
# signal is not section-dependent, it is just strongly one-directional.
# v2 replaces the statistic (a number small models don't weigh probabilistically)
# with an imperative default plus a specific, high-bar override condition,
# instead of a low-bar one ("clearly supports diagnosis") a model could
# talk itself into from thin/generic context alone.
CONDITION_VS_OBSERVATION_PRIOR = (
    "STRICT CORPUS CONVENTION: when evaluating SNOMED near-duplicates where "
    "one candidate is a Condition/Disorder and the other is an Observation/"
    "Morphologic-Abnormality classification of the identical concept name, "
    "this dataset's strict convention is to prefer the Observation/"
    "Morphologic-Abnormality concept. You MUST default to the Observation/"
    "Morphologic-Abnormality candidate. Do not override this and choose the "
    "Condition/Disorder candidate unless the surrounding text contains "
    "overwhelming, explicit evidence that it is being coded as a formal "
    "disease state rather than a clinical finding -- the section heading "
    "alone (e.g. 'Chief Complaint', 'Problem List') is NOT sufficient "
    "evidence on its own; that heading appears on both kinds of cases in "
    "this corpus. Only override on specific language in the entity's own "
    "sentence, not on section type or general medical association."
)


def _condition_vs_observation_duplicate(accepted: list) -> bool:
    """True when `accepted` is exactly the same-name Condition-vs-Observation
    duplicate pattern CONDITION_VS_OBSERVATION_PRIOR was measured against.

    2026-08-20 (real, previously-undetected bug, found via corpus-scale
    grading + a direct pull of a stored decision's eval_trail). This
    function originally also required concept_class_id to match
    ("Disorder"/"Morph Abnormality" specifically) -- but candidate dicts
    (src.normalization.tier_retrieval._candidate()) NEVER carry
    concept_class_id at all; nothing populates it. That made this check
    permanently return False regardless of input, silently disabling
    CONDITION_VS_OBSERVATION_PRIOR's injection ENTIRELY since it was built
    -- confirmed live on the "wound dehiscence" case (13538696-DS-11): the
    tiebreak correctly fired for all 3 models (eval_trail shows
    'tiebreak': True, candidates_considered: [1, 2] for every one of them),
    but with no prior injected each model reasoned unguided ("more commonly
    used", "aligns with documentation style") and all 3 happened to agree
    on the wrong (Condition/Disorder) candidate.

    Relaxed to match on domain_id alone (a field every candidate dict
    actually carries) combined with the exact-name-match already required
    below -- two DIFFERENT concept_ids sharing the EXACT SAME name string
    while spanning Condition and Observation domains is already a narrow,
    specific signal on its own; the concept_class_id check was extra
    precision this codebase never actually had wired up.
    """
    if len(accepted) != 2:
        return False
    names = {(a["candidate"].get("concept_name") or "").strip().lower() for a in accepted}
    if len(names) != 1:
        return False
    domains = {a["candidate"].get("domain_id") for a in accepted}
    return domains == {"Condition", "Observation"}


def _tiebreak_prompt(entity: dict, accepted: list, clinical_meaning: str) -> str:
    indices = [a["index"] for a in accepted]
    block = "\n\n".join(
        f"[{a['index']}] name: {a['candidate'].get('concept_name')}\n"
        f"    domain: {a['candidate'].get('domain_id')}\n"
        f"    concept class: {a['candidate'].get('concept_class_id') or 'unknown'}\n"
        f"    vocabulary: {a['candidate'].get('vocabulary_id')}\n"
        f"    basis: {a['candidate'].get('match_basis', 'semantic_similarity')}\n"
        f"    your own earlier independent judgment: MATCH "
        f"(\"{a.get('reasoning') or ''}\")"
        for a in accepted
    )
    return (
        "This entity's clinical meaning was independently determined to be:\n"
        f'  "{clinical_meaning}"\n\n'
        "ENTITY:\n"
        f"  text as written: {entity.get('original_text')!r}\n"
        f"  section: {entity.get('section_name') or 'unknown'}\n"
        f"  assertion: {entity.get('assertion_status', 'PRESENT')} / "
        f"experiencer: {entity.get('experiencer', 'PATIENT')}\n"
        f"  context: ...{entity.get('local_context', '')}...\n\n"
        f"CANDIDATES THAT EACH INDEPENDENTLY MATCHED:\n{block}\n\n"
        "Multiple candidates can independently look correct when they are "
        "SNOMED near-duplicates -- most commonly the same clinical idea "
        "filed once as a Condition/Disorder concept and once as an "
        "Observation/Morphologic-Abnormality concept. Use the domain/class "
        "distinction shown above plus the note's own documentation style "
        "and context to pick the single better fit. Do not default to the "
        f"lowest-numbered candidate without a reason. Valid indices: {indices}.\n\n"
        + (CONDITION_VS_OBSERVATION_PRIOR + "\n\n"
           if _condition_vs_observation_duplicate(accepted) else "")
        + 'Reply with JSON: {"best_index": "<one of the valid indices, as a '
        'string>", "reasoning": "<one sentence>"}'
    )


def _tiebreak_schema(indices: list) -> dict:
    return {
        "type": "object",
        "properties": {
            "best_index": {"type": "string", "enum": [str(i) for i in indices]},
            "reasoning": {"type": "string"},
        },
        "required": ["best_index", "reasoning"],
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

    Default behavior stops Step B at the first accepted candidate (unchanged
    since this module's original build). When EXHAUSTIVE_CANDIDATE_EVAL_ENABLED
    is set, every candidate is evaluated instead, and a genuine 2+-way tie is
    resolved by _resolve_tiebreak() rather than silently keeping whichever
    candidate happened to be checked first -- see that flag's own comment
    block above for why this exists and why it is not just a full-list
    comparative prompt.
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
    accepted = []  # only populated/used when EXHAUSTIVE_CANDIDATE_EVAL_ENABLED
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
                if not EXHAUSTIVE_CANDIDATE_EVAL_ENABLED:
                    verdict = "SUPPORTED_1" if i == 1 else f"RE_RANK_TO_CANDIDATE_{i}"
                    return {"model": client.model_name, "verdict": verdict,
                            "clinical_meaning": clinical_meaning,
                            "reasoning": parsed.get("reasoning"),
                            "logprob_confidence": confidence,
                            "degenerate_generation": step_a_degenerate or step_degenerate,
                            "eval_trail": trail}
                # Exhaustive mode: record the accept and keep going -- do NOT
                # return here. Stopping at the first "yes" is exactly the bug
                # this mode exists to fix (see the module comment above
                # EXHAUSTIVE_CANDIDATE_EVAL_ENABLED).
                accepted.append({"index": i, "candidate": cand, "confidence": confidence,
                                 "reasoning": parsed.get("reasoning"),
                                 "degenerate": step_a_degenerate or step_degenerate})
        except (LLMUnavailable, ValueError) as exc:
            trail.append({"candidate_index": i, "error": f"{type(exc).__name__}: {exc}"})

    any_degenerate = step_a_degenerate or any(t.get("degenerate_generation") for t in trail)

    if EXHAUSTIVE_CANDIDATE_EVAL_ENABLED and accepted:
        if len(accepted) == 1:
            a = accepted[0]
            verdict = "SUPPORTED_1" if a["index"] == 1 else f"RE_RANK_TO_CANDIDATE_{a['index']}"
            return {"model": client.model_name, "verdict": verdict,
                    "clinical_meaning": clinical_meaning, "reasoning": a["reasoning"],
                    "logprob_confidence": a["confidence"],
                    "degenerate_generation": a["degenerate"], "eval_trail": trail}
        # 2+ independently accepted -- genuine ambiguity Stage 2b/Step B's
        # own per-candidate judgment could not resolve on its own. Run a
        # single comparative call scoped to just this short accepted subset
        # (see _tiebreak_prompt's own docstring for why this is safe where a
        # full-list comparative prompt was measured NOT to be).
        return _resolve_tiebreak(client, entity, accepted, clinical_meaning,
                                 trail, any_degenerate)

    return {"model": client.model_name, "verdict": "NONE_CORRECT",
            "clinical_meaning": clinical_meaning,
            "reasoning": trail[-1].get("reasoning") if trail and "reasoning" in trail[-1] else None,
            "logprob_confidence": None,
            "degenerate_generation": any_degenerate, "eval_trail": trail}


def _resolve_tiebreak(client, entity: dict, accepted: list, clinical_meaning: str,
                      trail: list, any_degenerate: bool) -> dict:
    """The comparative call for EXHAUSTIVE_CANDIDATE_EVAL_ENABLED's 2+-accepted
    case. On any transport/parse failure, or a response outside the valid
    index set, falls back to the accepted candidate with the HIGHEST
    individual logprob_confidence (never the lowest index -- defaulting to
    list order is the exact arbitrary-pick behavior this mode exists to
    remove) rather than raising, so a single flaky tiebreak call degrades to
    a defensible answer instead of losing the entity's vote entirely.
    """
    indices = [a["index"] for a in accepted]

    def _fallback(reason):
        best = max(accepted, key=lambda a: (a["confidence"] if a["confidence"] is not None else -1))
        verdict = "SUPPORTED_1" if best["index"] == 1 else f"RE_RANK_TO_CANDIDATE_{best['index']}"
        trail.append({"tiebreak": True, "candidates_considered": indices,
                      "fallback_reason": reason, "chosen_index": best["index"]})
        return {"model": client.model_name, "verdict": verdict,
                "clinical_meaning": clinical_meaning,
                "reasoning": f"tiebreak fallback ({reason}): highest-confidence "
                            f"accepted candidate was [{best['index']}]",
                "logprob_confidence": best["confidence"],
                "degenerate_generation": any_degenerate, "eval_trail": trail}

    try:
        raw = client.complete(
            TIEBREAK_SYSTEM_PROMPT,
            _tiebreak_prompt(entity, accepted, clinical_meaning),
            schema=_tiebreak_schema(indices))
        parsed = parse_json_response(raw["text"])
        chosen_str = str(parsed.get("best_index"))
        if chosen_str not in {str(i) for i in indices}:
            return _fallback(f"model returned out-of-set index {chosen_str!r}")
        chosen = int(chosen_str)
        confidence = extract_verdict_confidence(raw["tokens"], chosen_str)
        step_degenerate = bool(raw.get("degenerate_generation"))
        trail.append({"tiebreak": True, "candidates_considered": indices,
                      "chosen_index": chosen, "reasoning": parsed.get("reasoning"),
                      "confidence": confidence, "degenerate_generation": step_degenerate})
        verdict = "SUPPORTED_1" if chosen == 1 else f"RE_RANK_TO_CANDIDATE_{chosen}"
        return {"model": client.model_name, "verdict": verdict,
                "clinical_meaning": clinical_meaning, "reasoning": parsed.get("reasoning"),
                "logprob_confidence": confidence,
                "degenerate_generation": any_degenerate or step_degenerate,
                "eval_trail": trail}
    except (LLMUnavailable, ValueError) as exc:
        return _fallback(f"{type(exc).__name__}: {exc}")


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

    2026-08-18: also recognizes PHYSEXAM_SHORTHAND_MATCH_BASIS
    (src.physexam_shorthand) -- same "curated, pre-verified lookup, not a
    similarity guess" trust tier as verified_brand_alias, but deliberately
    ALLOWS assertion_status=ABSENT for this basis specifically: 'NT'/'ND'
    are correctly tagged ABSENT (see that module's docstring on
    inherently_negated terms), and a non-PRESENT assertion there is the
    CORRECT, expected state, not a contradiction cue -- unlike the general
    case, where ABSENT on an otherwise-PRESENT-shaped candidate would
    signal something real to double-check.
    """
    physexam_hits = [(i, c) for i, c in enumerate(entity.get("candidates") or [], 1)
                     if c.get("match_basis") == PHYSEXAM_SHORTHAND_MATCH_BASIS]
    if len(physexam_hits) == 1 and not entity.get("expansion_ambiguous"):
        i, c = physexam_hits[0]
        return {
            "tier": TIER_3_AUTO_VALIDATED,
            "mollm_routing_decision": "AUTO_VALIDATED",
            "queue_reason": None,
            "final_candidate_index": i,
            "composite_confidence": None,
            "routing_basis": (
                f"Tier 3 fast path: candidate [{i}] ({c.get('concept_name')}) is a "
                f"gold-mined physical-exam-shorthand alias, the sole such hit, "
                f"with no ambiguous expansion -- skipped the two-step ensemble "
                f"entirely."),
            "models": [],
        }

    alias_hits = [(i, c) for i, c in enumerate(entity.get("candidates") or [], 1)
                  if c.get("match_basis") == "verified_brand_alias"]
    if len(alias_hits) == 1 and not entity.get("expansion_ambiguous") \
            and entity.get("assertion_status") in (None, "PRESENT"):
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

    # 2026-08-20 ("Lab Test near-duplicate-concept" finding,
    # evaluation/grade_fresh25_by_tier.py): the module docstring already
    # claimed verified_lab_test_alias sits in the same curated/verified
    # trust tier as verified_brand_alias, but this bypass never actually
    # checked for it -- confirmed live the gap was real: existing "HCT"
    # entities (already in _LAB_TEST_ALIASES, in production before this
    # fix) were landing in TIER_5_TRUE_AMBIGUITY, not any auto-write tier,
    # because a bare lab abbreviation's own SapBERT similarity is
    # routinely too low to pass tier5_precheck's floor even after the
    # correct concept is force-included into the pool (confirmed: none of
    # Calcium/Na's correct concepts even appear in the raw top-5; ALT/MCV
    # land at raw rank #2, not #1) -- force-inclusion alone was never
    # going to be enough to reach Tier 1 on raw ranking.
    #
    # is_ambiguous is explicitly ALLOWED here, unlike the general case,
    # but ONLY when ambiguity_reason is specifically
    # "verified_lab_test_alias_below_floor" (orchestrator.py's own
    # "rescue" flag for exactly this situation: the curated alias
    # candidate's raw score fell below TIER3_SIMILARITY_FLOOR). That
    # reason is a symptom of embedding a bare abbreviation, not evidence
    # against the curated identity -- the curation IS the trust signal.
    # Any OTHER ambiguity reason (e.g. a genuine top-2-candidate near-miss,
    # the MCH/MCHC pattern _lab_procedure_fast_path() already guards
    # against) still declines, same discipline as that fast path's own
    # is_ambiguous check -- this does not blindly trust every ambiguous
    # lab entity, only this one specific, already-understood reason.
    lab_alias_hits = [(i, c) for i, c in enumerate(entity.get("candidates") or [], 1)
                      if c.get("match_basis") == "verified_lab_test_alias"]
    if len(lab_alias_hits) != 1:
        return None
    if entity.get("expansion_ambiguous"):
        return None
    if entity.get("assertion_status") not in (None, "PRESENT"):
        return None
    if entity.get("is_ambiguous") and \
            entity.get("ambiguity_reason") != "verified_lab_test_alias_below_floor":
        return None
    i, c = lab_alias_hits[0]
    return {
        "tier": TIER_3_AUTO_VALIDATED,
        "mollm_routing_decision": "AUTO_VALIDATED",
        "queue_reason": None,
        "final_candidate_index": i,
        "composite_confidence": None,
        "routing_basis": (
            f"Tier 3 fast path: candidate [{i}] ({c.get('concept_name')}) is a "
            f"curated, gold-verified lab-test-shorthand alias, the sole such hit, "
            f"with no ambiguous expansion, non-PRESENT assertion, or unexplained "
            f"ambiguity -- skipped the two-step ensemble entirely."),
        "models": [],
    }


def _lab_procedure_fast_path(entity: dict) -> dict:
    """AUTO_VALIDATED without spending any model calls, for a Lab-Test
    entity whose top candidate is tagged match_basis="lab_procedure_preferred"
    (src.normalization.tier_retrieval's _prefer_lab_procedure_over_
    observable()). Split out from tier3_fast_path() below rather than
    folded into its single `return`, since this pattern's own justification
    (a corpus-measured naming CONVENTION, not a KG-verified identity fact
    like a brand alias) is different enough to want its own docstring and
    its own routing_basis wording, not a paraphrase of the alias branch's.

    Why a deterministic bypass, after a prompt-based attempt already failed:
    see the tag's own comment in _prefer_lab_procedure_over_observable() --
    live-tested, all 3 models unanimously re-ranked away from this exact
    winner regardless of instruction wording, because _binary_match_prompt()
    judges one candidate in isolation with no visibility into the sibling it
    would need to be told NOT to prefer. The 78/78-exceptionless corpus
    evidence behind the tag itself is strong enough to trust directly,
    matching this codebase's existing precedent for curated/verified trust
    tiers (verified_brand_alias, verified_lab_test_alias,
    PHYSEXAM_SHORTHAND_MATCH_BASIS) rather than routing every occurrence
    through an ensemble that has already been shown not to honor it.
    """
    if entity.get("gliner_label") != "Lab Test":
        return None
    candidates = entity.get("candidates") or []
    if not candidates or candidates[0].get("match_basis") != "lab_procedure_preferred":
        return None
    if entity.get("expansion_ambiguous"):
        return None
    # 2026-08-20 ("MCH/MCHC problem", corpus-scale grading finding). Found
    # live: bare "MCH" force-includes the CORRECT verified_lab_test_alias
    # candidate (4182871, "Mean corpuscular hemoglobin determination"), but
    # _prefer_lab_procedure_over_observable()'s tagging only compares raw
    # similarity scores among same-class candidates -- it has no way to know
    # the alias candidate is more authoritative than a same-class sibling
    # that merely scores higher by coincidence (here, MCHC's own
    # "...concentration determination" concept, 0.7458 vs the correct
    # concept's 0.7233 -- a genuine SapBERT near-miss between two
    # closely-named tests). Stage 2b's OWN ambiguity check already catches
    # this specific case (is_ambiguous=True, reason=
    # tier3_top2_margin_below_threshold, since the top two scores are only
    # 0.0225 apart) -- this fast-path just never consulted that flag before
    # bypassing the ensemble entirely. Checking it now closes the gap
    # without needing to fix the tagging logic's own class-blind scoring.
    if entity.get("is_ambiguous"):
        return None
    if entity.get("assertion_status") not in (None, "PRESENT"):
        return None
    score = candidates[0].get("similarity_score")
    if not isinstance(score, (int, float)) or score < TIER3_SIMILARITY_FLOOR:
        return None
    return {
        "tier": TIER_3_AUTO_VALIDATED,
        "mollm_routing_decision": "AUTO_VALIDATED",
        "queue_reason": None,
        "final_candidate_index": 1,
        "composite_confidence": None,
        "routing_basis": (
            f"Tier 3 fast path: candidate [1] ({candidates[0].get('concept_name')}) "
            f"is the determination/measurement-class concept for this lab test, "
            f"preferred over an Observable-Entity sibling by a corpus-measured "
            f"naming convention (78/78 exceptionless) -- skipped the two-step "
            f"ensemble entirely after it was found to unanimously re-rank away "
            f"from this exact pattern regardless of prompt instruction."),
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


def _score_with_calibrator(entity: dict, model_results: list, candidate_index: int,
                           calibrator, conn, conn_factory) -> dict:
    """Shared calibrator-consultation logic, factored out 2026-08-20 so the
    TIER_2_AUTO_RESOLVED (unanimous re-rank) and non-unanimous-split paths
    in route_tier() apply the EXACT same safety checks -- the
    fragile-shorthand trap and prior-confirmation lookup -- rather than
    risking the two call sites silently drifting apart over time. Returns
    {"trapped": bool, "trap_reason": str|None, "calibrated_score": float|None,
    "prior_count": int|None}. Caller decides what tier a promotion lands in
    (TIER_1B for a split, TIER_2B for a unanimous re-rank) -- this function
    only computes the score.
    """
    candidates = entity.get("candidates") or []
    trapped, trap_reason = _fragile_shorthand_trap(entity, candidate_index, candidates)
    if trapped:
        return {"trapped": True, "trap_reason": trap_reason,
                "calibrated_score": None, "prior_count": None}

    from src.mollm_tier_calibrator import build_feature_context, count_prior_confirmations
    chosen_concept_id = None
    if 0 < candidate_index <= len(candidates):
        chosen_concept_id = candidates[candidate_index - 1].get("omop_concept_id")

    if conn_factory is not None:
        lookup_conn = conn_factory()
        try:
            prior_count = count_prior_confirmations(
                lookup_conn, entity.get("original_text"), chosen_concept_id)
        finally:
            lookup_conn.close()
    else:
        prior_count = count_prior_confirmations(
            conn, entity.get("original_text"), chosen_concept_id)
    context = build_feature_context(entity, model_results, prior_count)
    calibrated_score = calibrator.score(context)
    return {"trapped": False, "trap_reason": None,
            "calibrated_score": calibrated_score, "prior_count": prior_count}


# ==========================================================================
# Full Tier 1-5 gate
# ==========================================================================

def route_tier(entity: dict, model_results: list = None, clients: dict = None,
               calibrator=None, conn=None, conn_factory=None) -> dict:
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

    `conn_factory` (2026-08-18, "don't lock Streamlit out" fix): a zero-arg
    callable returning a fresh connection, preferred over `conn` by callers
    that care about lock duration (scripts/run_stage3_tier_gate.py). The
    ONLY DB access anywhere in this function is the brief
    count_prior_confirmations() lookup below, which happens AFTER
    run_two_step_ensemble()'s 3 LLM calls have already returned -- passing
    an already-OPEN `conn` (the old contract) means that connection sits
    open, holding DuckDB's single-writer lock, for the ENTIRE ensemble call
    too, even though the ensemble itself never touches it (confirmed live:
    this was measured and is not hypothetical -- the entity-level
    connection-cycling fix that preceded this one reduced average lock
    duration but left near-zero free gaps between entities, so Streamlit
    was still effectively always locked out during Stage 3). With
    `conn_factory`, the connection is opened fresh and closed immediately
    around just that one query -- milliseconds, not the whole call. `conn`
    is kept for backward compatibility (existing tests, any caller that
    doesn't care about lock duration); `conn_factory` takes priority when
    both are supplied.
    """
    qualifier = qualifier_fragment_precheck(entity)
    if qualifier:
        return qualifier
    fast = tier3_fast_path(entity)
    if fast:
        return fast
    lab_fast = _lab_procedure_fast_path(entity)
    if lab_fast:
        return lab_fast
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
        # 2026-08-18 ("elevate the gate" fix): checked BEFORE the confidence
        # floor and BEFORE granting TIER_1_AUTO_VALIDATED -- unanimity does
        # not protect against a known-fragile candidate list, so a trapped
        # entity is forced to HITL regardless of how confident all three
        # models claim to be. See _fragile_shorthand_trap()'s own docstring.
        trapped, trap_reason = _fragile_shorthand_trap(entity, 1, entity.get("candidates") or [])
        if trapped:
            return {"tier": TIER_4_ENSEMBLE_SPLIT, "mollm_routing_decision": "HITL_REQUIRED",
                    "queue_reason": trap_reason, "final_candidate_index": None,
                    "composite_confidence": composite_confidence,
                    "calibrated_score": None,
                    "routing_basis": (
                        f"3/3 unanimous SUPPORTED_1 but bypassed -- fragile-shorthand trap "
                        f"({trap_reason}), see _fragile_shorthand_trap()"),
                    "models": model_results}
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
        # 2026-08-20 (see TIER_2B_CALIBRATED_AUTO_RESOLVED's own comment):
        # a unanimous re-rank is NOT a deterministic auto-write on its own
        # -- give a fitted calibrator a chance to say THIS SPECIFIC decision
        # is likely correct anyway, same mechanism (and same safety checks:
        # fragile-shorthand trap, prior-confirmation count) as the
        # non-unanimous split path below, just landing in a structurally
        # separate tier since Tier 2's feature distribution (100%
        # is_ambiguous, zero vote disagreement) is not what the calibrator
        # was fit/validated on.
        tier2_calibrated_score = None
        if calibrator is not None and (conn is not None or conn_factory is not None):
            result = _score_with_calibrator(entity, model_results, n, calibrator, conn, conn_factory)
            if result["trapped"]:
                return {"tier": TIER_2_AUTO_RESOLVED, "mollm_routing_decision": "HITL_REQUIRED",
                        "queue_reason": result["trap_reason"], "final_candidate_index": n,
                        "composite_confidence": composite_confidence,
                        "calibrated_score": None,
                        "routing_basis": (
                            f"3/3 unanimous re-rank to candidate {n}; calibrator bypassed -- "
                            f"fragile-shorthand trap ({result['trap_reason']})"),
                        "models": model_results}
            tier2_calibrated_score = result["calibrated_score"]
            if tier2_calibrated_score is not None and tier2_calibrated_score >= CALIBRATED_AUTO_THRESHOLD:
                return {"tier": TIER_2B_CALIBRATED_AUTO_RESOLVED,
                        "mollm_routing_decision": "AUTO_VALIDATED", "queue_reason": None,
                        "final_candidate_index": n,
                        "composite_confidence": composite_confidence,
                        "calibrated_score": tier2_calibrated_score,
                        "routing_basis": (
                            f"3/3 unanimous re-rank to candidate {n}, but ConsensusCalibrator "
                            f"scored {tier2_calibrated_score} >= {CALIBRATED_AUTO_THRESHOLD} "
                            f"(prior_confirmation_count={result['prior_count']})"),
                        "models": model_results}
        # Still tagged TIER_2_AUTO_RESOLVED for audit/grading continuity
        # (the underlying signal is real and worth distinguishing), routed
        # to HITL_REQUIRED since it didn't clear the calibrator (or no
        # calibrator was supplied at all).
        return {"tier": TIER_2_AUTO_RESOLVED, "mollm_routing_decision": "HITL_REQUIRED",
                "queue_reason": "tier2_auto_resolved_pending_revalidation",
                "final_candidate_index": n,
                "composite_confidence": composite_confidence,
                "calibrated_score": tier2_calibrated_score,
                "routing_basis": (f"3/3 unanimous re-rank to candidate {n}, "
                                  f"composite_confidence {composite_confidence} -- "
                                  f"queued for review pending TIER_2_AUTO_RESOLVED "
                                  f"post-fix re-validation (see AUTO_TIERS comment)"
                                  + (f"; calibrator scored {tier2_calibrated_score} < "
                                     f"{CALIBRATED_AUTO_THRESHOLD}"
                                     if tier2_calibrated_score is not None else "")),
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

    if calibrator is not None and (conn is not None or conn_factory is not None):
        candidate_index = None
        if top_verdict == "SUPPORTED_1":
            candidate_index = 1
        elif top_verdict.startswith("RE_RANK_TO_CANDIDATE_"):
            candidate_index = int(top_verdict.rsplit("_", 1)[1])

        if candidate_index is not None:
            # Fragile-shorthand trap (coronary-segment enumerated list +
            # short-code shape regex, see _fragile_shorthand_trap()): a
            # known-fragile retrieval pattern, quarantined BEFORE the
            # calibrator ever sees it -- calibrator.score() is not called at
            # all for a trapped entity, not merely overridden after the
            # fact, so no training data (evaluation/tier_gate_cal_eval.py)
            # or future fit can accidentally re-learn its way around this
            # gate. Same _score_with_calibrator() helper the unanimous
            # re-rank (TIER_2) branch above uses, so both stay in sync.
            result = _score_with_calibrator(entity, model_results, candidate_index,
                                            calibrator, conn, conn_factory)
            if result["trapped"]:
                return {"tier": TIER_4_ENSEMBLE_SPLIT, "mollm_routing_decision": "HITL_REQUIRED",
                        "queue_reason": result["trap_reason"], "final_candidate_index": None,
                        "composite_confidence": composite_confidence,
                        "calibrated_score": None,  # bypassed BEFORE calibrator.score() is called
                        "routing_basis": (
                            f"non-unanimous verdicts {dict(vote_counts)}; calibrator bypassed -- "
                            f"fragile-shorthand trap ({result['trap_reason']})"),
                        "models": model_results}
            calibrated_score = result["calibrated_score"]
            prior_count = result["prior_count"]

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
