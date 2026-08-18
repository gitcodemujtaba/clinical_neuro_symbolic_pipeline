"""
src/mollm_ensemble.py — Stage 3: MoLLM confidence-gated validation.

Objective 2 in its entirety, and the primary novelty claim of the thesis. Takes
a Stage 2 ValidationRecord, retrieves KG-grounded evidence (src/retrieval.py),
puts it to two independent local LLMs (src/llm_client.py), verifies that what
they cited actually exists, and produces a decision artifact matching
docs/Provenance_Schema.md's Stage 3 schema.

Runs PER ENTITY, not per note. OpenBioLLM 8B fine-tunes base Meta-Llama-3-8B,
so its context window is 8,192 tokens while notes run 2,374-24,858 characters
(~6,200 tokens for the longest). Both ensemble members must see identical input
for their votes to be comparable, so 8,192 is the binding budget and a
per-entity call with a sentence-bounded context window is the only shape that
keeps prompt size independent of note length.

THREE HARD SAFETY RULES, each bypassing the confidence arithmetic entirely.
They are rules rather than model judgements because each concerns a situation
where a confident-looking number would be actively misleading:

  1. ASSERTION GATING. Guideline rules describe what to do when a finding IS
     PRESENT. ~20.9% of gold-annotated spans are non-assertive (14.2% negated),
     so applying guideline logic to "denies chest pain" is a high-frequency
     category error, not an edge case. Negated / non-patient records never
     reach a contradiction check, and CONTRADICTED is not an available verdict
     when no guideline evidence was retrieved at all -- a contradiction with
     nothing to contradict is definitionally a hallucination.
  2. MODEL DISAGREEMENT. Opposite verdicts from two independent validators go
     to a human regardless of how confident either model was. "The discounted
     average happened to fall below threshold" is a much weaker thing to defend
     than "the two validators disagreed, so a clinician looked at it".
  3. FAILED CITATION. A model citing evidence it was never given is the exact
     hallucination this mechanism exists to catch. It must not be recoverable
     by a high confidence score.

WHY LOGPROBS DRIVE THE MATH AND SELF-REPORTED CONFIDENCE DOES NOT: see
src/llm_client.py. raw_confidence_label is recorded but never used in a
decision, purely so the dissertation can report whether the two agree.
"""

import json
import os
import re
import uuid
import warnings

from src.llm_client import (
    LLMUnavailable,
    build_clients,
    extract_candidate_alternatives,
    extract_verdict_confidence,
    parse_json_response,
    verdict_schema,
)
from src.normalization.compound_span import strip_lab_value_suffix
from src.provenance import (
    candidates_hash,
    provenance_alter_statements,
    provenance_column_sql,
    provenance_placeholders,
    provenance_params,
)

# docs/MoLLM_Redesign_Proposal.md S4.3 -- floor below which an alternative
# verdict's probability is treated as "rejected" and logged. CALIBRATION-
# PENDING like every other threshold in this file's ensemble math (see
# AUTO_VALIDATE_THRESHOLD/MOLLM_RESOLVE_THRESHOLD below); a placeholder, not a
# measured value, until there is a labeled sample to check it against.
REJECTED_CANDIDATE_FLOOR = 0.05

warnings.filterwarnings("ignore")

# Verdict vocabularies, closed per mode.
CONTRADICTION_VERDICTS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}
RESOLUTION_VERDICTS_BASE = {"NONE_CORRECT", "INSUFFICIENT_EVIDENCE"}

# Routing thresholds on composite_confidence. Calibration targets against the
# validation slice (docs/Evaluation_Criteria.md), NOT settled values -- they are
# named here so there is one place to change them after the ECE analysis.
AUTO_VALIDATE_THRESHOLD = 0.85
MOLLM_RESOLVE_THRESHOLD = 0.60

# ==========================================================================
# ASYMMETRIC OVERRIDE GATE (2026-08-13,
# docs/2026-08-13_Code_Improvement_Proposals.md P2.1)
# ==========================================================================
#
# THE MEASUREMENT THIS ANSWERS. evaluation/stage2b_cal_eval.py's cross-tab
# over the 31-note corpus (2026-08-13 report S6):
#
#     CAUGHT_AND_FIXED   13  ( 9.7%)  Stage 2b wrong, MoLLM fixed it
#     INTRODUCED_ERROR   30  (22.4%)  Stage 2b RIGHT, MoLLM broke it
#     NET VALUE         -17
#
# and a fresh 3-note replication, immune to the candidate-drift confound,
# pointing the same way (0 fixed, 1 introduced).
#
# WHY A THRESHOLD CHANGE WOULD NOT FIX THIS. route() currently treats
# "confirm Stage 2b's top-1" and "override it" as the same decision, gated on
# the same number. They are not the same decision. Confirming a correct link
# costs nothing when wrong (it was already right); overriding a correct link
# DESTROYS it. The measured override is wrong more than twice as often as it
# is right, so the two directions cannot share an acceptance bar.
#
# THE GATE. An override -- any verdict RESOLVED_TO_CANDIDATE_n where n is not
# Stage 2b's own top-1 -- must clear ALL THREE of:
#   * a calibrator score >= OVERRIDE_TOP1_THRESHOLD (raw composite_confidence
#     is explicitly not sufficient: report S5.4 measured it at 14.29% accuracy
#     across the whole threshold sweep, so it cannot carry this decision),
#   * grounding_basis == "guideline_rule" -- the model must have had real
#     retrieved evidence, not its own terminology intuitions, and
#   * citation_verified -- that evidence must actually check out.
# Anything less routes to a human under `unsupported_top1_override`.
#
# WHY THIS IS SAFE RATHER THAN MERELY CONSERVATIVE. It cannot lose a
# CAUGHT_AND_FIXED case outright -- a blocked override becomes HITL, where a
# human can still make the correction. It can only convert silent corrections
# into reviewed ones. INTRODUCED_ERROR, by construction, goes toward zero.
#
# DEFAULT ON, unlike P1.1's ranker, because unlike the ranker this changes no
# stored Stage 2b value and invalidates nothing already measured -- it only
# redirects decisions that the same report showed to be net-harmful.
OVERRIDE_TOP1_THRESHOLD = 0.90
OVERRIDE_GATE_ENABLED = os.environ.get(
    "CNSP_OVERRIDE_GATE", "1").strip().lower() not in ("0", "false", "no")

# Applied to composite_confidence when the two models agree on the verdict but
# differ substantially in how confident they were. Disagreement in degree is
# weaker evidence than agreement in degree, and averaging alone would hide it.
CONFIDENCE_SPREAD_PENALTY = 0.85
CONFIDENCE_SPREAD_TRIGGER = 0.30

# Containment threshold for strict citation verification. Deliberately the same
# metric and cutoff scripts/clean_local_triplets.py used to ASSIGN
# citation_type in the first place, so one methodology both builds the ground
# truth and audits the model against it.
CITATION_CONTAINMENT_THRESHOLD = 0.8

# ==========================================================================
# REASONING/VERDICT CONSISTENCY CHECK
# ==========================================================================
#
# Found by hand-tracing evaluation/stage2b_cal_eval.py's INTRODUCED_ERROR rows
# (docs/2026-08-13_Implementation_Verification.md follow-up): for the
# 'aortic valve leaflets' entity (note 11838076-DS-20), BOTH ensemble models
# named candidate [2] ("Structure of cusp of aortic valve") verbatim in their
# free-text `reasoning` field, then independently voted
# RESOLVED_TO_CANDIDATE_3 -- a DIFFERENT candidate. Traced to
# _format_candidates() below: candidate [3]'s own is-a hint line repeats
# candidate [2]'s name verbatim, and a guided-JSON decode can lock the
# structured `verdict` field onto the wrong bracket independently of what the
# free-text `reasoning` field (generated from the same pass) actually argued
# for. Same containment metric and cutoff as citation verification, reused
# rather than re-invented, so a candidate name that reasoning only PARAPHRASES
# does not false-positive -- only a near-verbatim mention counts.
REASONING_MENTION_THRESHOLD = 0.8

# Evidence cap after an expansion request. Higher than the first-round 5, but
# still bounded -- the 8,192-token window is the constraint, not willingness to
# show more.
MAX_RULES_AFTER_EXPANSION = 15

# Matches build_prompt()'s RESOLVED_TO_CANDIDATE_{i} verdict vocabulary. Used
# by _is_top1_override() to tell a confirmation of Stage 2b's own pick from a
# substitution of a different concept for it.
RESOLVED_RE = re.compile(r"^RESOLVED_TO_CANDIDATE_(\d+)$")

INGESTION_AUTO = "AUTO_VALIDATED"
INGESTION_RESOLVED = "MOLLM_RESOLVED"
INGESTION_HITL = "HITL_REQUIRED"

# RULES harmonized with src/mollm_review.py's 2026-08-15 anchoring-bias fix
# (see docs/2026-08-15_Contradiction_Detection_Analysis.md S6-S7): the
# contradiction-detection matrix measured this module's own tier-1/exact-match
# recall at 21.3%, worse than Objective 3's, and the root cause traced back to
# a conflicting pair of instructions in THIS prompt -- the old "PROVENANCE
# OVERRIDES SPELLING ... Trust these relationships completely" bullet sat
# directly above the old "SCORE WARNING" bullet, one saying trust exact_text
# completely and the other saying a high score proves nothing. You cannot ask
# a model to be a skeptical auditor and mandate blind trust in the same
# breath. Fixed by splitting PROVENANCE OVERRIDES SPELLING into two claims
# that were being conflated -- "the string-level LINK is real" (keep trusting
# this) vs. "the reading is CONTEXTUALLY correct" (never implied by the
# former, and measured at ~52% for exact_text specifically via
# evaluation/stage2b_cal_eval.py, cited inline below) -- then adding the same
# DEVIL'S ADVOCATE + CONCEPTUAL FIREWALL framing already validated in
# mollm_review.py. NOTE: this override is safe to ship even mid-uncertainty
# because of the ASYMMETRIC OVERRIDE GATE above (OVERRIDE_TOP1_THRESHOLD +
# citation_verified) -- a more skeptical model can only convert a silent
# confirmation into a routed HITL case, never silently corrupt a write, and
# every decision from this module is ALSO currently gated behind blanket
# human review regardless of tier (docs/Implementation_Checklist.md Stage 4).
# NOT YET RE-VALIDATED against a real A/B batch the way mollm_review.py's
# equivalent fix was -- see scripts/analysis/prompt_ab_test.py for the
# pattern to follow before trusting this in a full-corpus run.
SYSTEM_PROMPT = """You are a clinical terminology validator in an auditable pipeline.

You will be given one entity extracted from a clinical note, the concept(s) a
terminology service mapped it to, and any clinical guideline evidence retrieved
from a curated knowledge graph.

Your task is LABELING, not clinical decision-making: you are judging whether a
concept is the CORRECT LABEL for a text span, not whether a patient's care is
clinically appropriate, not a diagnosis, and not a treatment recommendation.
Keep every judgment scoped to that question.

RULES:
- When guideline evidence IS shown, judge only what is in the EVIDENCE block.
  Do not use outside knowledge to override or supplement it.
- When the EVIDENCE block says no guideline evidence was retrieved, you may
  use your knowledge of medical terminology and clinical concepts to judge
  whether the candidate concept is the correct label for the entity shown,
  given its context. This is a labeling judgment -- whether this text
  correctly refers to this concept -- not a judgment about whether the
  patient's care is clinically appropriate. If you use this basis, set
  "reasoning_basis" to "medical_terminology_knowledge"; if your judgment
  rests only on the ontology facts you were shown (concept name, semantic
  tag, parents) with no additional knowledge needed, set it to
  "ontology_only". Leave "reasoning_basis" unset when guideline evidence was
  shown and used.
- You may cite evidence ONLY by the rule_id values in the EVIDENCE block. Never
  invent a rule_id, and never cite text that is not shown to you. This applies
  regardless of which basis you reasoned from -- you must never attribute a
  terminology-knowledge judgment to a guideline.
- If the EVIDENCE block says no guideline evidence was retrieved, you cannot
  return CONTRADICTED. Return SUPPORTED if the mapping is a correct label,
  otherwise INSUFFICIENT_EVIDENCE.
- Respect the stated assertion status. A finding marked ABSENT was explicitly
  negated in the note; do not treat it as present.
- PROVENANCE VERIFIES THE LINK, NOT THE CONTEXT: a candidate with basis
  verified_brand_alias or exact_text has a mathematically real string-level
  relationship to the entity text -- do not reject it merely because its
  spelling differs from what you expected. But a verified LINK is not the
  same claim as a verified FIT: this corpus's own calibration measured plain
  exact_text matches at only ~52% accuracy, a coin flip, so exact_text alone
  is NOT evidence the reading is contextually correct here. Judge every
  candidate, regardless of basis, on whether it fits the surrounding clinical
  context -- never on the strength of its match_basis or score alone.
- IGNORE THE SCORE WHILE JUDGING FIT: a high similarity score (e.g. above
  0.80) or a perfect string match is NOT reliable evidence of contextual
  correctness on its own. Evaluate the candidate's semantic and clinical fit
  as if the score and match_basis were hidden from you; use them only as a
  tie-breaker between readings you already judge equally plausible on
  context, never as the reason to accept a reading.
- BE THE DEVIL'S ADVOCATE: treat every candidate as unproven until context
  confirms it, not as confirmed until context contradicts it. Before
  accepting a candidate, actively look for the reason it might be wrong -- a
  wrong abbreviation expansion, a string match with the wrong clinical
  meaning, a lab-test-vs-diagnosis mismatch.
- CONCEPTUAL FIREWALL: "after abbreviation expansion" in the ENTITY block
  below is Stage 1's own guess at what the TEXT means, not a fact about any
  candidate. When judging what a candidate concept means, use ONLY its own
  name/vocabulary identity shown in CANDIDATE CONCEPTS -- never let Stage 1's
  expansion stand in for it. A candidate can share zero words with a correct
  expansion and still be the wrong concept.
- SYNONYM TOLERANCE: do not reject valid clinical paraphrases (e.g. rejecting
  "Right atrial structure" for the entity "right atrium") purely due to minor
  wording differences. You are matching semantic concepts, not exact strings.

The EVIDENCE block is a ranked selection, not everything that was found. If it
is insufficient and more of the retrieved record would change your answer, you
may ask for it ONCE by setting "request" to one of:
  MORE_RULES        - lower-ranked rules that were retrieved but not shown
  SUPPRESSED_RULES  - rules found under this concept's code but withheld
                      because the guideline node's name disagreed with the
                      entity. These are UNTRUSTED and may describe a different
                      clinical concept entirely.
  CANDIDATE_DETAIL  - fuller detail on the alternative candidate concepts
Only ask if it would genuinely change your verdict; otherwise answer now.

Reply with a single JSON object and nothing else:
{"verdict": "<one of the allowed verdicts>",
 "reasoning": "<complete these steps in order: (1) what the extracted text means from the surrounding sentence alone; (2) what the candidate concept means, using ONLY its own name/vocabulary identity -- never Stage 1's abbreviation expansion; (3) whether those two meanings actually agree, ignoring the match score/basis while you decide. Four sentences maximum.>",
 "cited_evidence": [{"rule_id": "<id from EVIDENCE>", "quote": "<exact text from that rule>"}],
 "confidence": "<HIGH|MEDIUM|LOW>",
 "request": "<MORE_RULES|SUPPRESSED_RULES|CANDIDATE_DETAIL|NONE>",
 "reasoning_basis": "<ontology_only|medical_terminology_knowledge, ONLY when no guideline evidence was retrieved>"}
Use an empty cited_evidence list if you cited nothing, "NONE" if you need no
further information, and omit reasoning_basis entirely when guideline
evidence was shown and used."""


# ==========================================================================
# Prompt assembly
# ==========================================================================

def _lab_value_note(entity_text: str) -> str:
    """2026-08-14 (Option 1.5, lab-value display cleanup). Candidates for a
    flowsheet-style entity like "WBC-13.0" are ALREADY correctly resolved --
    src/normalization/orchestrator.py's LAB VALUE SUFFIX FALLBACK already
    strips the value and re-normalizes before storing normalized_entities.
    candidates (confirmed empirically: raw "WBC-13.0" fails Tier 3 entirely,
    below TIER3_SIMILARITY_FLOOR; the stored 0.892-score "Leucocyte count"
    candidate comes from the cleaned "WBC" embedding, correctly adopted by
    the tier-rank comparison in that fallback). What's NOT clean is what the
    MoLLM prompt itself shows: the entity fields still carry the raw value,
    so the model has to notice on its own that "WBC-13.0" ~ "Leucocyte
    count" is a valid match despite the numeric suffix -- exactly the
    lexical-greediness inconsistency that produces 2-1 splits on these
    entities. This is display-only: the caller's own text fields are
    untouched (still "WBC-13.0" everywhere else -- DB writes, grading,
    entity_id), so it can't affect gold-span scoring. Returns "" when
    strip_lab_value_suffix() finds no lab-value shape, so ordinary entities
    get no extra line.
    """
    candidates = strip_lab_value_suffix(entity_text or "")
    if not candidates:
        return ""
    return (f'\n  (this looks like a lab/flowsheet result -- "{candidates[0]}" '
            f'is the test name, the trailing number is its measured VALUE, not '
            f'part of the concept. Match on "{candidates[0]}", not the number.)')


def _format_candidates(record, retrieval) -> str:
    cands = record.get("candidates") or []
    if not cands:
        return "CANDIDATE CONCEPTS: none — Stage 2 could not map this entity.\n"
    ctxs = retrieval.get("candidate_contexts") or []
    # 2026-08-13 (reasoning_verdict_mismatch's motivating case): a candidate's
    # is-a PARENT can happen to be another candidate's own concept name (e.g.
    # "Entire cusp of aortic valve" is-a "Structure of cusp of aortic valve",
    # which is ALSO candidate [2] here). Read flat, that line is easy to
    # misattribute to the sibling candidate rather than to the parent
    # relationship it actually states -- both ensemble models did exactly
    # that, naming [2] in their reasoning while voting for [3]. Marking the
    # collision explicitly removes the ambiguity instead of relying on the
    # model to track bracket numbers across two separate mentions of the same
    # name.
    name_to_index = {
        (c.get("concept_name") or "").strip().lower(): i
        for i, c in enumerate(cands, 1) if c.get("concept_name")
    }
    lines = ["CANDIDATE CONCEPTS:"]
    for i, c in enumerate(cands, 1):
        ctx = ctxs[i - 1] if i - 1 < len(ctxs) else {}
        line = (f"  [{i}] {c.get('concept_name')} (OMOP {c.get('omop_concept_id')}, "
                f"{c.get('vocabulary_id')}/{c.get('domain_id')}, "
                f"match tier {c.get('match_tier')}, score {c.get('similarity_score')}, "
                f"basis {c.get('match_basis', 'semantic_similarity')})")
        parents = ctx.get("parents") or []
        if parents:
            parts = []
            for p in parents[:3]:
                pname = p["name"]
                sibling_idx = name_to_index.get(pname.strip().lower())
                if sibling_idx and sibling_idx != i:
                    parts.append(f"{pname} (this IS candidate [{sibling_idx}] above -- "
                                 f"a PARENT of [{i}], not a name for [{i}] itself)")
                else:
                    parts.append(pname)
            line += "\n      is-a: " + "; ".join(parts)
        lines.append(line)
    return "\n".join(lines) + "\n"


def _format_evidence(retrieval) -> str:
    """Renders the evidence block, including WHY each rule was considered
    relevant.

    match_channel and match_confidence are shown to the model deliberately: a
    3-hop hierarchy generalisation is weaker evidence than a direct code match,
    and flattening both into an undifferentiated list would launder that
    difference away. The model is told which it is looking at.
    """
    if retrieval.get("retrieval_skipped_reason"):
        return (f"EVIDENCE: guideline retrieval was deliberately skipped.\n"
                f"  Reason: {retrieval['retrieval_skipped_reason']}\n")

    sup = retrieval.get("suppression") or {}
    rejected = sup.get("suppressed_name_disagreement", 0)
    unverified = sup.get("suppressed_unverified_code_assertion", 0)

    rules = retrieval.get("rules") or []
    if not rules:
        # "Nothing exists" and "candidates existed but none were trustworthy"
        # are different facts and the model should not have to guess which it
        # is looking at. The second means the concept IS covered by the
        # guidelines, but under a code assertion the symbolic layer could not
        # verify -- so INSUFFICIENT_EVIDENCE is the honest verdict, not an
        # assumption that the guidelines are silent.
        note = ""
        if rejected or unverified:
            note = (f"  ({rejected + unverified} candidate rule(s) WERE found under this "
                    f"concept's code but were suppressed because the guideline node's name "
                    f"did not agree with this entity, or its code assertion was unverified. "
                    f"The guidelines may cover this concept; the link to it could not be "
                    f"trusted.)\n")
        return ("EVIDENCE: NO GUIDELINE EVIDENCE RETRIEVED.\n"
                + note
                + "  You cannot return CONTRADICTED for this record.\n")

    lines = ["EVIDENCE (guideline rules retrieved from the knowledge graph):"]
    for r in rules:
        how = f"channel {r['match_channel']}, relevance {r['match_confidence']}"
        if r.get("hierarchy_hops"):
            how += f", matched {r['hierarchy_hops']} level(s) above this concept"
        if r.get("name_agreement") == "weak":
            how += ", WEAK name agreement — treat with caution"
        if r.get("type_agreement") == "mismatch":
            how += (f", node is typed '{r.get('matched_node_type')}' but the extractor "
                    f"labelled this entity differently — consider whether the label is wrong")
        lines.append(f"  rule_id: {r['rule_id']}")
        # Node @types are shown inline. They let the model judge something the
        # symbolic layer cannot: whether the EXTRACTION LABEL itself is wrong.
        # An entity GLiNER called a Medication matching a guideline node typed
        # Intervention is a real signal, not noise -- and a wrong label is a
        # finding worth surfacing rather than quietly downweighting.
        src_t = f" ({r['matched_node_type']})" if r.get("direction") == "outgoing" and r.get("matched_node_type") else ""
        tgt_t = f" ({r['other_node_type']})" if r.get("direction") == "outgoing" and r.get("other_node_type") else ""
        lines.append(f"    statement: {r['source_name']}{src_t} —[{r['predicate']}]→ "
                     f"{r['target_name']}{tgt_t}")
        if r.get("rationale"):
            lines.append(f"    rationale: {r['rationale']}")
        citable = _citable_text(r)
        if citable:
            label = "quotable source" if r["citation_type"] in (
                "verbatim", "paraphrase_with_recovered_excerpt") else "paraphrase (NOT a direct quote)"
            lines.append(f"    {label}: \"{citable}\"")
        lines.append(f"    source: {r.get('source_document')} — {r.get('section_title')}")
        lines.append(f"    retrieved via: {how}")

    # One compact line, not the full accounting: the evidence shown is a
    # ranked subset, and a model told it has seen everything will reason as
    # though absence of a rule means absence of guidance. The full breakdown
    # is in the decision artifact for an auditor.
    dropped = sup.get("dropped_by_cap", 0)
    footnotes = []
    if dropped:
        footnotes.append(f"{dropped} further rule(s) ranked lower and were not shown")
    if rejected or unverified:
        footnotes.append(f"{rejected + unverified} suppressed as untrustworthy links")
    if footnotes:
        lines.append(f"  [{'; '.join(footnotes)} — this is a ranked selection, "
                     f"not the complete set]")
    return "\n".join(lines) + "\n"


def _citable_text(rule) -> str:
    """The text the model is permitted to quote, chosen by citation_type.

    pointer_unverifiable rules and rules with no citation offer nothing
    quotable: there is no source text to check a quote against, so offering one
    would invite a citation that cannot be verified either way.
    """
    ctype = rule.get("citation_type")
    if ctype in ("verbatim", "paraphrase_with_recovered_excerpt"):
        return rule.get("citation_verbatim_excerpt") or rule.get("citation") or ""
    if ctype == "paraphrase":
        return rule.get("citation") or ""
    return ""


def build_prompt(record: dict, retrieval: dict) -> tuple:
    """Returns (user_prompt, mode, allowed_verdicts)."""
    tier = record.get("confidence_tier_in", "HIGH")
    skipped = bool(retrieval.get("retrieval_skipped_reason"))
    has_evidence = bool(retrieval.get("rules"))

    assertion_line = (
        f"  assertion: {record.get('assertion_status', 'PRESENT')} / "
        f"experiencer: {record.get('experiencer', 'PATIENT')} / "
        f"temporality: {record.get('temporality', 'CURRENT')}"
    )
    if record.get("assertion_cue"):
        assertion_line += f"  (cue: \"{record['assertion_cue']}\")"

    parts = [
        "ENTITY:",
        f"  text as written: {record.get('original_text')!r}",
        f"  after abbreviation expansion: {record.get('expanded_text')!r}"
        f"{_lab_value_note(record.get('expanded_text'))}",
        f"  extractor label: {record.get('gliner_label')} "
        f"(confidence {record.get('gliner_confidence')})",
        assertion_line,
    ]
    if record.get("expansion_ambiguous"):
        parts.append(
            f"  NOTE: this abbreviation has multiple known expansions "
            f"{record.get('candidate_expansions')}. The one above was chosen "
            f"without context. Consider whether it is the right reading."
        )
    parts.append("")
    parts.append(f"SECTION: {record.get('section_name') or 'unknown'}")
    parts.append(f"CONTEXT: ...{record.get('local_context', '')}...")
    parts.append("")
    parts.append(_format_candidates(record, retrieval))

    rels = record.get("relations") or []
    if rels:
        parts.append("RELATIONS involving this entity:")
        for r in rels[:5]:
            status = "" if r.get("entity_link_status") == "linked" else " [endpoint unlinked]"
            parts.append(f"  {r.get('relation_label')}: {r.get('other_endpoint_text')}"
                         f" (confidence {r.get('relation_confidence')}){status}")
        parts.append("")

    parts.append(_format_evidence(retrieval))
    suppressed_block = _format_suppressed(retrieval)
    if suppressed_block:
        parts.append(suppressed_block)
    if retrieval.get("expansion_applied"):
        parts.append(f"[This is a FOLLOW-UP round. {retrieval['expansion_applied']}. "
                     f"Answer now — no further requests will be served.]")

    if tier == "LOW" and len(record.get("candidates") or []) > 1:
        mode = "resolution"
        allowed = RESOLUTION_VERDICTS_BASE | {
            f"RESOLVED_TO_CANDIDATE_{i}" for i in range(1, len(record["candidates"]) + 1)
        }
        # docs/Stage3_Open_Issues.md Issue 3 (spirnolactone -> SPIRAPRILAT,
        # 2026-08-10 run): NONE_CORRECT was legal and correct but unused. The
        # candidates here came from a lexical/semantic fallback (Tier 3), not
        # a guarantee the true concept is even on the list -- the previous
        # wording implied picking one of them was always the task. Both models
        # picked the closest-SPELLED candidate over the closest-MEANING one;
        # BioMistral's own reasoning named "Spirinolactone" correctly and then
        # selected SPIRAPRILAT anyway. The instruction now names that failure
        # mode explicitly rather than leaving NONE_CORRECT to be inferred from
        # the verdict enum.
        parts.append(
            "TASK: Stage 2 could not decide between the candidate concepts above. "
            "Using the context, section and any evidence, decide which candidate the "
            "entity refers to.\n"
            "These candidates were surfaced by lexical/semantic similarity, not "
            "guaranteed correctness -- the true concept may not be among them at all, "
            "particularly for misspellings, where the nearest-SPELLED match and the "
            "nearest-MEANING match can be different concepts. If none of the candidates "
            "is the concept this entity actually refers to, return NONE_CORRECT rather "
            "than choosing the closest-looking one.\n"
            f"Allowed verdicts: {', '.join(sorted(allowed))}"
        )
    elif skipped or record.get("assertion_status") == "ABSENT" or \
            record.get("experiencer", "PATIENT") != "PATIENT":
        mode = "non_asserted_check"
        allowed = {"SUPPORTED", "INSUFFICIENT_EVIDENCE", "NONE_CORRECT"}
        parts.append(
            "TASK: This mention is not an asserted current finding about the patient. "
            "Do NOT assess guideline compliance. Judge only whether the concept mapping "
            "is correct for the text as written.\n"
            f"Allowed verdicts: {', '.join(sorted(allowed))}"
        )
    else:
        mode = "contradiction"
        allowed = set(CONTRADICTION_VERDICTS)
        if not has_evidence:
            allowed.discard("CONTRADICTED")
        parts.append(
            "TASK: Decide whether this concept assignment is supported by, or contradicted "
            "by, the evidence above.\n"
            f"Allowed verdicts: {', '.join(sorted(allowed))}"
        )

    return "\n".join(parts), mode, allowed


# ==========================================================================
# Citation verification
# ==========================================================================

def _containment(needle: str, haystack: str) -> float:
    """Longest-common-substring containment, matching clean_local_triplets.py."""
    import difflib
    if not needle or not haystack:
        return 0.0
    sm = difflib.SequenceMatcher(None, haystack, needle, autojunk=False)
    match = sm.find_longest_match(0, len(haystack), 0, len(needle))
    return match.size / max(1, len(needle))


def verify_citations(model_output: dict, retrieval: dict) -> dict:
    """Checks every cited_evidence entry against what the model was shown.

    Two tiers, matching what is actually checkable:
      * STRICT for rules whose source was a real quote (verbatim /
        paraphrase_with_recovered_excerpt) -- the quoted text must genuinely
        appear in the offered source.
      * LOOSE for paraphrase-sourced rules -- containment cannot be required of
        text that was never a verbatim quote, so this degrades to verifying the
        cited rule_id was actually shown, i.e. detecting fabricated attribution.

    A fabricated rule_id fails outright: that is a citation to evidence that
    does not exist, which is the failure mode this whole mechanism exists for.
    """
    rules_by_id = {r["rule_id"]: r for r in (retrieval.get("rules") or [])}
    # Released suppressed rules ARE citable once shown -- refusing to verify
    # them would fail every verdict that used the evidence we deliberately
    # handed over. Their use is flagged instead of forbidden.
    suppressed_ids = set()
    for r in (retrieval.get("suppressed_rules_released") or []):
        rules_by_id[r["rule_id"]] = r
        suppressed_ids.add(r["rule_id"])
    citations = model_output.get("cited_evidence") or []

    results = []
    all_ok = True
    for c in citations:
        if not isinstance(c, dict):
            all_ok = False
            results.append({"rule_id": None, "verified": False, "reason": "malformed_citation"})
            continue

        rule_id = c.get("rule_id")
        quote = (c.get("quote") or "").strip()
        rule = rules_by_id.get(rule_id)

        if rule is None:
            all_ok = False
            results.append({"rule_id": rule_id, "verified": False,
                            "reason": "fabricated_rule_id_not_in_evidence"})
            continue

        ctype = rule.get("citation_type")
        source_text = _citable_text(rule)

        if ctype in ("verbatim", "paraphrase_with_recovered_excerpt"):
            score = _containment(quote, source_text)
            ok = score >= CITATION_CONTAINMENT_THRESHOLD
            results.append({"rule_id": rule_id, "verified": ok, "mode": "strict",
                            "containment": round(score, 4),
                            "reason": None if ok else "quote_not_found_in_source"})
        elif ctype == "paraphrase":
            results.append({"rule_id": rule_id, "verified": True, "mode": "loose",
                            "reason": "rule_id_present; source is paraphrase, quote not checkable"})
            ok = True
        else:
            ok = False
            results.append({"rule_id": rule_id, "verified": False, "mode": "loose",
                            "reason": "rule_not_citable (pointer_unverifiable or no citation)"})
        all_ok = all_ok and ok

    cited_suppressed = [c.get("rule_id") for c in citations
                        if isinstance(c, dict) and c.get("rule_id") in suppressed_ids]
    return {"citation_verified": all_ok, "citation_checks": results,
            "citations_made": len(citations),
            # Surfaced, not blocked: a verdict resting on evidence the symbolic
            # guard rejected is one a human should look at, and the artifact
            # has to say so plainly.
            "cited_suppressed_rules": cited_suppressed}


# ==========================================================================
# Ensemble
# ==========================================================================

VALID_REQUESTS = {"MORE_RULES", "SUPPRESSED_RULES", "CANDIDATE_DETAIL"}
# One follow-up round only. Bounded deliberately: an unbounded ask-for-more
# loop would make cost per entity unpredictable and, worse, non-deterministic
# in depth -- two runs of the same record could take different numbers of
# rounds and produce different evidence. One round covers the real case (the
# top-5 were not enough) without either problem.
MAX_EXPANSION_ROUNDS = 1


def expand_evidence(retrieval: dict, request: str) -> dict:
    """Returns a retrieval dict widened per the model's request.

    Served entirely from the ALREADY-STORED record -- all_ranked_rules and
    suppression.suppressed_rules were persisted by retrieval.retrieve() for
    exactly this purpose. Nothing is re-queried: re-running retrieval to answer
    a follow-up would risk returning different evidence than the first round
    saw (guard bands, KG contents and hierarchy can all change between runs),
    which would make the two rounds incomparable and the audit trail wrong.

    SUPPRESSED_RULES is the sensitive one. Those rules were withheld because
    the name-agreement guard judged the code assertion untrustworthy -- the
    NSTEMI/STEMI class of collision. They are released only on explicit
    request, individually re-labelled UNTRUSTED in the prompt, and any verdict
    that cites one is flagged in the artifact so a reviewer can see the model
    leaned on evidence the symbolic layer had rejected.
    """
    expanded = dict(retrieval)
    sup = retrieval.get("suppression") or {}

    if request == "MORE_RULES":
        all_ranked = retrieval.get("all_ranked_rules") or []
        expanded["rules"] = all_ranked[:MAX_RULES_AFTER_EXPANSION]
        expanded["expansion_applied"] = (
            f"MORE_RULES: showing {len(expanded['rules'])} of "
            f"{len(all_ranked)} retrieved rules")

    elif request == "SUPPRESSED_RULES":
        suppressed = list(sup.get("suppressed_rules") or [])[:MAX_RULES_AFTER_EXPANSION]
        for r in suppressed:
            r = r.setdefault("name_agreement", "reject")
        expanded["suppressed_rules_released"] = suppressed
        expanded["expansion_applied"] = (
            f"SUPPRESSED_RULES: releasing {len(suppressed)} withheld rule(s), "
            f"flagged UNTRUSTED")

    elif request == "CANDIDATE_DETAIL":
        expanded["show_candidate_detail"] = True
        expanded["expansion_applied"] = "CANDIDATE_DETAIL"

    return expanded


def _format_suppressed(retrieval: dict) -> str:
    released = retrieval.get("suppressed_rules_released") or []
    if not released:
        return ""
    lines = ["", "WITHHELD EVIDENCE (released on request) — READ THE WARNING:",
             "  These rules were found under this concept's SNOMED code, but the",
             "  guideline node's name did not agree with this entity. In this corpus a",
             "  single code often carries clinically unrelated concepts, so these may",
             "  describe something else entirely. Treat as UNTRUSTED. If you cite one,",
             "  say explicitly why you believe it applies to THIS entity."]
    for r in released:
        lines.append(f"  rule_id: {r['rule_id']}   [UNTRUSTED — name disagreement]")
        lines.append(f"    guideline node: {r.get('matched_node_name')!r} "
                     f"({r.get('matched_node_type')})")
        lines.append(f"    statement: {r['source_name']} —[{r['predicate']}]→ {r['target_name']}")
        if r.get("rationale"):
            lines.append(f"    rationale: {r['rationale']}")
    return "\n".join(lines) + "\n"


def _query_one(client, system_prompt, user_prompt, allowed_verdicts,
                require_citation: bool = True) -> dict:
    # Guided decoding constrains the verdict to allowed_verdicts, so the
    # out-of-vocabulary branch below should become unreachable -- but it is
    # kept, because llm_client falls back to unguided json_object mode if the
    # server rejects the guided request, and a silent fallback must not turn
    # into a silent coercion.
    #
    # require_citation=False (docs/Stage3_Open_Issues.md Issue 2 experiment):
    # caller sets this when retrieval found zero rules, testing whether an
    # unconditionally-required cited_evidence array was itself inviting
    # fabrication under guided decoding. verify_citations() still runs
    # either way -- this changes what the model is pressured to emit, not
    # what gets checked.
    raw = client.complete(system_prompt, user_prompt,
                          schema=verdict_schema(allowed_verdicts, require_citation=require_citation))
    parsed = parse_json_response(raw["text"])

    verdict = str(parsed.get("verdict", "")).strip().upper()
    if verdict not in allowed_verdicts:
        # An out-of-vocabulary verdict is treated as an abstention rather than
        # coerced to the nearest allowed value -- guessing what the model meant
        # would fabricate a clinical decision it did not make.
        parsed["_verdict_out_of_vocabulary"] = verdict or None
        verdict = "INSUFFICIENT_EVIDENCE"

    reasoning_basis = parsed.get("reasoning_basis")
    if reasoning_basis not in ("ontology_only", "medical_terminology_knowledge"):
        reasoning_basis = None

    return {
        "model": raw["model"],
        "verdict": verdict,
        "reasoning": parsed.get("reasoning"),
        "cited_evidence": parsed.get("cited_evidence") or [],
        "raw_confidence_label": parsed.get("confidence"),
        "request": (str(parsed.get("request", "")).strip().upper()
                    if str(parsed.get("request", "")).strip().upper() in VALID_REQUESTS
                    else None),
        "logprob_confidence": extract_verdict_confidence(raw["tokens"], verdict),
        # docs/MoLLM_Redesign_Proposal.md S9.5 -- self-reported grounding basis,
        # only meaningful (and only requested by the prompt) when no guideline
        # evidence was retrieved. None when unset, mis-set, or not applicable;
        # never coerced to a default, same discipline as logprob_confidence.
        "reasoning_basis": reasoning_basis,
        # docs/MoLLM_Redesign_Proposal.md S4.3 -- probability the model
        # assigned to verdicts it did NOT choose, read off the logprob
        # alternates already being requested (TOP_LOGPROBS) but discarded
        # before 2026-08-11. {} when unavailable or not applicable (e.g. a
        # contradiction-mode verdict with no sibling candidates to compare
        # against), never omitted -- an empty dict is distinguishable from
        # "not computed" by callers that check for the key.
        "candidate_alternatives": extract_candidate_alternatives(
            raw["tokens"], verdict, allowed_verdicts),
        "verdict_out_of_vocabulary": parsed.get("_verdict_out_of_vocabulary"),
        "finish_reason": raw.get("finish_reason"),
        # Guided vs unguided changes the logprob distribution, so a calibration
        # set must not mix the two. Recorded per call rather than assumed.
        "decoding_mode": raw.get("decoding_mode"),
        # 2026-08-13 (P4). Carried through from src/llm_client.py so combine()
        # can exclude this verdict from the agreement test and so the per-model
        # rate survives into the models JSON column -- which is what makes the
        # FREQUENCY_PENALTY fix measurable rather than merely asserted.
        "degenerate_generation": raw.get("degenerate_generation", False),
        "degenerate_retried": raw.get("degenerate_retried", False),
        "degenerate_detail": raw.get("degenerate_detail"),
    }


def combine(model_results: list) -> dict:
    """ensemble_agreement + composite_confidence.

    Mean of logprob_confidence when the models agree. When they disagree the
    number is not discounted-and-used, it is marked and the caller forces HITL
    (safety rule 2) -- a discounted average of two opposite clinical verdicts
    is not a meaningful quantity and should not be allowed to look like one.

    Returns composite_confidence None when no model reported a logprob, so
    "unmeasured" stays distinguishable from "measured as low".
    """
    # 2026-08-13 (docs/2026-08-13_Code_Improvement_Proposals.md P4).
    # DEGENERATE GENERATIONS ARE NOT VOTES.
    #
    # The 2026-08-13 report S4.2 measured 475 verdicts -- 23.8% of every
    # BioMistral verdict this project has ever produced -- whose `reasoning`
    # was a repetition loop, always defaulting to INSUFFICIENT_EVIDENCE. Those
    # were counted here as genuine disagreement, which forced HITL via safety
    # rule 1. On the 3-note sample the report inspected by hand, 4 of 8
    # "disagreements" were this: OpenBioLLM answering correctly and
    # confidently about an entity stated plainly in the note, against a
    # BioMistral output that was not a second opinion but a stuck decoder.
    #
    # A model that could not stop talking did not disagree. Its verdict is
    # excluded from the agreement test, and the exclusion is RECORDED (never
    # silent) so the decision is auditable and the rate is measurable.
    #
    # THE ONE CASE WHERE THIS MUST NOT APPLY: if EVERY model degenerated there
    # is no surviving opinion, and dropping them all would leave an empty
    # verdict set that len(set()) == 1 would score as False -- accidentally
    # correct routing for entirely the wrong reason. That case is handled
    # explicitly below and routed on its own queue_reason.
    usable = [m for m in model_results if not m.get("degenerate_generation")]
    n_degenerate = len(model_results) - len(usable)
    all_degenerate = bool(model_results) and not usable

    scored = model_results if all_degenerate else usable
    verdicts = [m["verdict"] for m in scored]
    agreement = len(set(verdicts)) == 1

    # Confidence is likewise averaged over the USABLE models only: a
    # degenerate generation's logprob describes how sure the model was about
    # repeating itself, which is not a confidence in the verdict.
    confs = [m["logprob_confidence"] for m in scored if m["logprob_confidence"] is not None]

    degeneracy = {
        "n_degenerate_models": n_degenerate,
        "all_models_degenerate": all_degenerate,
        "degenerate_models": [m.get("model") for m in model_results
                              if m.get("degenerate_generation")],
        "n_models_scored": len(scored),
    }

    if not confs:
        return {"ensemble_agreement": agreement, "composite_confidence": None,
                "confidence_basis": "no_logprobs_available",
                "confidence_spread": None, **degeneracy}

    mean = sum(confs) / len(confs)
    spread = (max(confs) - min(confs)) if len(confs) > 1 else 0.0

    if not agreement:
        return {"ensemble_agreement": False, "composite_confidence": round(mean, 6),
                "confidence_basis": "verdicts_disagree_composite_not_used_for_routing",
                "confidence_spread": round(spread, 6), **degeneracy}

    composite = mean
    basis = "mean_logprob_agreeing_verdicts"
    if spread >= CONFIDENCE_SPREAD_TRIGGER:
        composite *= CONFIDENCE_SPREAD_PENALTY
        basis += "+spread_penalty"

    if n_degenerate and not all_degenerate:
        # Agreement reached only because a degenerate voter was set aside --
        # a single surviving opinion is unanimous by construction. Recorded in
        # the basis string so a reader of routing_basis can see that this was
        # a one-model decision, not a two-model consensus.
        basis += f"+{n_degenerate}_degenerate_excluded"

    return {"ensemble_agreement": True, "composite_confidence": round(composite, 6),
            "confidence_basis": basis, "confidence_spread": round(spread, 6),
            **degeneracy}


def _is_top1_override(verdict: str, candidate_permutation=None) -> bool:
    """True when `verdict` picks a candidate OTHER than Stage 2b's own top-1.

    Candidate numbering in the prompt is 1-based and follows the order the
    candidates were PRESENTED in (build_prompt()'s RESOLVED_TO_CANDIDATE_{i}
    for i in 1..n).

    THE PERMUTATION MATTERS AND IS EASY TO MISS. Under
    --shuffle-candidates (P2.2), presented position 1 is NOT Stage 2b's top-1
    -- it is whichever candidate the shuffle put first. Comparing the verdict
    index against 1 without consulting the permutation would make the override
    gate fire on confirmations and wave through genuine overrides, i.e. exactly
    backwards, in precisely the experimental run whose numbers are meant to be
    compared against the unshuffled baseline. candidate_permutation[i] is the
    original index of the candidate shown at position i, so original index 0 is
    Stage 2b's top-1.

    NONE_CORRECT is deliberately NOT an override for this purpose. It does not
    substitute a different concept for a correct one; it declines to link,
    which routes onward on its own existing path and cannot silently install a
    wrong concept_id. The failure mode P2.1 targets is specifically "replaced
    something right with something wrong".
    """
    m = RESOLVED_RE.match(verdict or "")
    if not m:
        return False
    presented_idx = int(m.group(1)) - 1  # 1-based verdict -> 0-based position
    if candidate_permutation:
        try:
            original_idx = candidate_permutation[presented_idx]
        except (IndexError, TypeError):
            return True  # unmappable verdict: treat as an override, i.e. gate it
    else:
        original_idx = presented_idx
    return original_idx != 0


def reasoning_verdict_mismatch(verdict: str, reasoning: str, candidates: list) -> dict:
    """Detects a model's free-text `reasoning` naming a DIFFERENT candidate
    than the one its structured `verdict` field points to. See
    REASONING_MENTION_THRESHOLD above for the motivating case.

    Only meaningful for RESOLVED_TO_CANDIDATE_n verdicts against >= 2
    candidates -- NONE_CORRECT, INSUFFICIENT_EVIDENCE and contradiction-mode
    verdicts (SUPPORTED/CONTRADICTED) have no sibling candidate name a
    mismatch could point to, so those return checked=False rather than a
    forced non-mismatch, keeping "not applicable" distinguishable from
    "checked and consistent".

    A mismatch requires the best-matching OTHER candidate's containment score
    to both clear REASONING_MENTION_THRESHOLD and exceed the chosen
    candidate's own score -- so a reasoning text that happens to mention
    multiple candidate names (e.g. to rule them out) does not false-positive
    as long as it argues most strongly for the one actually chosen.
    """
    m = RESOLVED_RE.match(verdict or "")
    if not m or not reasoning or len(candidates) < 2:
        return {"checked": False, "mismatch": False}

    chosen_idx = int(m.group(1)) - 1
    if chosen_idx < 0 or chosen_idx >= len(candidates):
        return {"checked": False, "mismatch": False}

    reasoning_lower = reasoning.lower()
    scores = []
    for c in candidates:
        name = (c.get("concept_name") or "").strip().lower()
        scores.append(_containment(name, reasoning_lower) if name else 0.0)

    chosen_score = scores[chosen_idx]
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_score = scores[best_idx]

    if (best_idx != chosen_idx
            and best_score >= REASONING_MENTION_THRESHOLD
            and best_score > chosen_score):
        return {"checked": True, "mismatch": True,
                "reasoning_names_candidate": best_idx + 1,
                "reasoning_names_score": round(best_score, 4),
                "verdict_candidate": chosen_idx + 1,
                "verdict_candidate_score": round(chosen_score, 4)}
    return {"checked": True, "mismatch": False}


def route(ensemble: dict, citation: dict, model_results: list,
          calibrator_score: float = None, grounding_basis: str = None,
          candidate_permutation=None, normalized_from: str = None) -> dict:
    """Final routing. The four safety rules are checked BEFORE the thresholds
    and each short-circuits, so no confidence score -- calibrated or not --
    can override them.

    `calibrator_score` (docs/MoLLM_Redesign_Proposal.md S4.1, "MoLLM-Cal"):
    when provided, SUBSTITUTES for ensemble["composite_confidence"] in the two
    threshold comparisons below, and ONLY there. It is never consulted for the
    three safety-rule checks above it, and it cannot promote a CONTRADICTED,
    INSUFFICIENT_EVIDENCE or NONE_CORRECT verdict past those either -- a
    trained calibrator changes what counts as "confident enough", not which
    checks a decision must clear first. None (the default) reproduces exactly
    today's behavior; callers only pass a value once a calibrator has actually
    been fit on real data (mollm_calibrator.MoLLMCalibrator.score() itself
    returns None until trained, so an untrained calibrator is indistinguishable
    from no calibrator at the call site).

    `normalized_from` (2026-08-14, Fragile Concept Gate, user proposal):
    normalized_entities.normalized_from, passed straight through by the
    caller. When it starts with "value_stripped_from_", Stage 2b's own
    winning concept only exists because
    src/normalization/orchestrator.py's LAB VALUE SUFFIX FALLBACK had to
    salvage it after Tier 1/2 failed on the raw text -- see the
    AUTO_VALIDATE_THRESHOLD branch below for what this changes. Unlike the
    four hard safety rules above, this is not a short-circuit to HITL: it
    caps how far a fragile-foundation decision can rise, it does not veto
    it outright. A None default (no normalized_from passed, or the field
    was never populated) reproduces exactly today's behavior.
    """
    # 2026-08-13 (P4). Every model's generation degenerated into a repetition
    # loop, so there is no opinion here at all -- not a disagreement, not an
    # abstention, just no usable output. Routed HITL like any other unresolved
    # case, but under its own queue_reason so the 2026-08-13 report's 69.8%
    # model_disagreement figure DECOMPOSES on the next run instead of silently
    # absorbing a decoding bug. This check precedes the agreement rule because
    # combine() scores the degenerate verdicts in this case (see its comment on
    # all_degenerate) and they would otherwise be read as a real vote.
    if ensemble.get("all_models_degenerate"):
        return {"mollm_routing_decision": INGESTION_HITL,
                "queue_reason": "degenerate_generation",
                "routing_basis": "safety_rule: every validator's output was a "
                                 "repetition loop, not a verdict "
                                 "(src/llm_client.py is_degenerate)"}

    if not ensemble["ensemble_agreement"]:
        return {"mollm_routing_decision": INGESTION_HITL,
                "queue_reason": "model_disagreement",
                "routing_basis": "safety_rule: independent validators returned different verdicts"}

    if not citation["citation_verified"]:
        return {"mollm_routing_decision": INGESTION_HITL,
                "queue_reason": "citation_verification_failed",
                "routing_basis": "safety_rule: model cited evidence not present in what it was shown"}

    # 2026-08-13: a model's free-text reasoning naming a different candidate
    # than its structured verdict field is evidence the verdict was not
    # actually produced by the reasoning shown alongside it -- see
    # reasoning_verdict_mismatch() above. Checked regardless of agreement or
    # confidence, same as the other three: a self-inconsistent verdict is not
    # made trustworthy by a second model reaching it independently, or by a
    # high logprob on the (inconsistent) token that was sampled.
    mismatched = [m for m in model_results
                  if m.get("reasoning_verdict_check", {}).get("mismatch")]
    if mismatched:
        detail = mismatched[0]["reasoning_verdict_check"]
        return {"mollm_routing_decision": INGESTION_HITL,
                "queue_reason": "reasoning_verdict_mismatch",
                "routing_basis": (
                    f"safety_rule: {mismatched[0].get('model')}'s reasoning named "
                    f"candidate {detail['reasoning_names_candidate']} "
                    f"(containment {detail['reasoning_names_score']}) but its verdict "
                    f"selected candidate {detail['verdict_candidate']} "
                    f"(containment {detail['verdict_candidate_score']})")}

    composite = ensemble["composite_confidence"]
    if composite is None:
        return {"mollm_routing_decision": INGESTION_HITL,
                "queue_reason": "confidence_unmeasurable",
                "routing_basis": "no logprobs returned; cannot calibrate a routing decision"}

    verdict = model_results[0]["verdict"]
    if verdict == "CONTRADICTED":
        return {"mollm_routing_decision": INGESTION_HITL,
                "queue_reason": "guideline_contradiction",
                # docs/MoLLM_Redesign_Proposal.md S9.6: this flags a label
                # that conflicts with retrieved guideline evidence about the
                # CONCEPT -- an entity-linking inconsistency for a human to
                # adjudicate, not a judgment about the patient's care.
                "routing_basis": "a flagged label/evidence inconsistency for a human to "
                                 "adjudicate, not something to auto-resolve"}

    if verdict == "INSUFFICIENT_EVIDENCE" or verdict == "NONE_CORRECT":
        return {"mollm_routing_decision": INGESTION_HITL,
                "queue_reason": f"verdict_{verdict.lower()}",
                "routing_basis": "no usable resolution produced"}

    used_calibrator = calibrator_score is not None
    routed_on = calibrator_score if used_calibrator else composite
    basis_label = "calibrator_score" if used_calibrator else "composite_confidence"

    # 2026-08-13 (P2.1). ASYMMETRIC OVERRIDE GATE -- see OVERRIDE_TOP1_THRESHOLD
    # above for the -17 NET VALUE measurement this exists to stop. Placed here,
    # AFTER the hard safety rules and BEFORE the thresholds, because it is
    # neither: it does not override a safety rule (those already returned), and
    # it is not a confidence cutoff (it adds requirements a confidence score
    # alone cannot satisfy). Applies only to overrides -- a verdict confirming
    # Stage 2b's own top-1, or NONE_CORRECT, falls straight through to the
    # unchanged threshold logic below.
    if OVERRIDE_GATE_ENABLED and _is_top1_override(verdict, candidate_permutation):
        blockers = []
        if calibrator_score is None:
            blockers.append("no_calibrator")
        elif calibrator_score < OVERRIDE_TOP1_THRESHOLD:
            blockers.append(f"calibrator_score {calibrator_score} < "
                            f"{OVERRIDE_TOP1_THRESHOLD}")
        if grounding_basis != "guideline_rule":
            blockers.append(f"grounding_basis={grounding_basis!r} "
                            f"(guideline_rule required)")
        if not citation.get("citation_verified"):
            blockers.append("citation_unverified")
        if blockers:
            return {"mollm_routing_decision": INGESTION_HITL,
                    "queue_reason": "unsupported_top1_override",
                    "routing_basis":
                        "override of Stage 2b's top-1 requires calibrator >= "
                        f"{OVERRIDE_TOP1_THRESHOLD} AND verified guideline "
                        f"evidence; blocked by: {'; '.join(blockers)}",
                    "confidence_basis_for_routing": basis_label}

    if routed_on >= AUTO_VALIDATE_THRESHOLD:
        # 2026-08-14 (Fragile Concept Gate). Measured on the 16393593-DS-5
        # smoke test: giving the models a clean lab-value display made
        # WBC-13/RBC-3/PTT-114/AlkPhos-255/Phos-9.4 unanimously WRONG
        # instead of visibly split -- ensemble consensus went up, precision
        # did not, and the only tier that skips human review is exactly
        # where that gap must not land silently. A fragile-foundation
        # decision is capped at MOLLM_RESOLVED rather than promoted to
        # AUTO_VALIDATED, same as a 2-1 split gets today -- not routed all
        # the way to HITL, since the confidence signal itself is still
        # intact and usable, just not license to skip review.
        if normalized_from and normalized_from.startswith("value_stripped_from_"):
            return {"mollm_routing_decision": INGESTION_RESOLVED, "queue_reason": None,
                    "routing_basis": f"{basis_label} {routed_on} >= {AUTO_VALIDATE_THRESHOLD}, "
                                     f"but capped at MOLLM_RESOLVED: winning concept only "
                                     f"resolved via lab-value-suffix fallback "
                                     f"({normalized_from}) -- too fragile a foundation for a "
                                     f"human-review skip regardless of ensemble agreement",
                    "confidence_basis_for_routing": basis_label}
        return {"mollm_routing_decision": INGESTION_AUTO, "queue_reason": None,
                "routing_basis": f"{basis_label} {routed_on} >= {AUTO_VALIDATE_THRESHOLD}",
                "confidence_basis_for_routing": basis_label}
    if routed_on >= MOLLM_RESOLVE_THRESHOLD:
        return {"mollm_routing_decision": INGESTION_RESOLVED, "queue_reason": None,
                "routing_basis": f"{basis_label} {routed_on} in "
                                 f"[{MOLLM_RESOLVE_THRESHOLD}, {AUTO_VALIDATE_THRESHOLD})",
                "confidence_basis_for_routing": basis_label}
    return {"mollm_routing_decision": INGESTION_HITL,
            "queue_reason": "below_confidence_threshold",
            "routing_basis": f"{basis_label} {routed_on} < {MOLLM_RESOLVE_THRESHOLD}",
            "confidence_basis_for_routing": basis_label}


def _compute_grounding_basis(retrieval: dict, model_results: list) -> tuple:
    """(grounding_basis, self_reports) per docs/MoLLM_Redesign_Proposal.md S9.5.

    Mechanical when evidence exists: "guideline_rule" whenever Channels A/B/D/E
    retrieved anything, regardless of whether the model actually cited it --
    this tier answers "was checkable evidence available", not "was it used".

    When no evidence was retrieved, falls back to the models' own
    self-reported `reasoning_basis` (S9.2's prompt addition). If EITHER model
    reports "medical_terminology_knowledge", that is the tier recorded -- the
    less-checkable basis is the more informative one to surface, and hiding
    it behind an "ontology_only" default because the OTHER model reported
    (or omitted) something more conservative would understate how much of
    the auto-validated set actually rests on the least-checkable tier. Only
    when NEITHER model self-reports does this default to "ontology_only",
    on the mechanical grounds that Channel C's structured facts are always
    present regardless (S9.1) -- the raw self-reports are still returned
    alongside so this default is never the only record of what happened.
    """
    self_reports = [m.get("reasoning_basis") for m in model_results]
    if retrieval.get("rules"):
        return "guideline_rule", self_reports
    if "medical_terminology_knowledge" in self_reports:
        return "model_terminology_knowledge", self_reports
    if "ontology_only" in self_reports:
        return "ontology_only", self_reports
    return "ontology_only", self_reports


def _compute_annotation_discrepancy(verdict: str, mode: str):
    """d_anno (docs/MoLLM_Redesign_Proposal.md S4.2/S9.3): does MoLLM's
    verdict CONFIRM or OVERRIDE Stage 2b's own top-ranked candidate?

    resolution mode: RESOLVED_TO_CANDIDATE_1 confirms Stage 2b's rank-1 pick
    (False = no discrepancy); any other candidate index, or NONE_CORRECT,
    overrides it (True). contradiction mode has only one candidate, so
    SUPPORTED plays the same "confirms" role and CONTRADICTED the same
    "overrides" role. INSUFFICIENT_EVIDENCE is neither a confirmation nor an
    override -- returns None, same "not measured" discipline as
    logprob_confidence, rather than being coerced into either bucket.
    """
    if mode == "resolution":
        if verdict == "RESOLVED_TO_CANDIDATE_1":
            return False
        if verdict.startswith("RESOLVED_TO_CANDIDATE_") or verdict == "NONE_CORRECT":
            return True
        return None
    if mode == "contradiction":
        if verdict == "SUPPORTED":
            return False
        if verdict == "CONTRADICTED":
            return True
        return None
    return None


def _compute_rejected_candidates(model_results: list) -> list:
    """Merges candidate_alternatives across models (docs/MoLLM_Redesign_Proposal.md
    S4.3), keeping only alternates below REJECTED_CANDIDATE_FLOOR. Pure
    bookkeeping -- no training, no gradient, just a record of which verdicts
    the ensemble considered and dismissed, for later Stage 2b/KG curation
    review (S9.4) or as a MoLLM-Cal feature (n_rejected_candidates)."""
    by_verdict = {}
    for m in model_results:
        for alt_verdict, prob in (m.get("candidate_alternatives") or {}).items():
            if prob is not None and prob < REJECTED_CANDIDATE_FLOOR:
                by_verdict.setdefault(alt_verdict, []).append(
                    {"model": m.get("model"), "probability": prob})
    return [{"verdict": v, "observations": obs} for v, obs in by_verdict.items()]


def _shuffled_candidates(record: dict, seed_key: str):
    """Returns (shuffled_record, permutation) or (record, None).

    2026-08-13 (docs/2026-08-13_Code_Improvement_Proposals.md P2.2). WHY THIS
    EXPERIMENT EXISTS. Stage 2b hands candidates to the prompt in
    concept_id-ascending order, and the report's INTRODUCED_ERROR examples
    cluster in one family of near-identical lab strings (AST-956, AST-909,
    AST-16, CK(CPK)). Position bias in list-selection prompts is well
    documented, and a block of near-identical options is exactly where it
    shows. If it is real, then a chunk of the 22.4% INTRODUCED_ERROR rate is a
    PROMPT bug being read as a model-quality finding -- a different problem
    with a different fix. Running the same slice with and without shuffling
    settles it.

    THE PERMUTATION IS RECORDED, NOT JUST APPLIED. `permutation[i]` is the
    ORIGINAL index of the candidate presented in position i, so a verdict of
    RESOLVED_TO_CANDIDATE_{i+1} maps back to original candidate
    permutation[i]. Without that mapping the run would be unanalysable, and
    _is_top1_override() would compare against the wrong candidate.

    SEEDED PER ENTITY, not globally: the shuffle must be reproducible on a
    re-run (this project's whole determinism argument) but must not apply the
    same permutation to every entity, which would just be a second fixed
    ordering rather than a randomisation.
    """
    import random

    cands = record.get("candidates") or []
    if len(cands) < 2:
        return record, None
    order = list(range(len(cands)))
    random.Random(seed_key).shuffle(order)
    if order == sorted(order):
        return record, None
    shuffled = dict(record)
    shuffled["candidates"] = [cands[i] for i in order]
    return shuffled, order


def validate_record(record: dict, retriever, clients: dict = None,
                    calibrator=None, shuffle_candidates: bool = False) -> dict:
    """Runs Stage 3 for one entity and returns the decision artifact.

    Artifact field names follow docs/Provenance_Schema.md's Stage 3 spec so it
    can be written straight into the KG3 :MoLLMDecision node without a
    translation layer that could drift.

    `calibrator` (docs/MoLLM_Redesign_Proposal.md S4.1): an optional
    mollm_calibrator.MoLLMCalibrator. None (the default) reproduces exactly
    today's threshold-only routing -- existing callers (scripts/test_stage3_live.py,
    scripts/run_stage3_batch.py) need no changes to keep working. Passing a
    loaded calibrator only changes behavior once it has actually been fit
    (score() returns None otherwise, see route()'s calibrator_score docstring).

    `shuffle_candidates` (2026-08-13, P2.2): permutes the candidate list before
    building the prompt, recording the permutation in the artifact so verdicts
    map back to original indices. An EXPERIMENT, default False -- see
    _shuffled_candidates() for the position-bias hypothesis it tests.
    """
    clients = clients if clients is not None else build_clients()

    candidate_permutation = None
    if shuffle_candidates:
        record, candidate_permutation = _shuffled_candidates(
            record, seed_key=str(record.get("entity_id") or record.get("note_id")))

    retrieval = retriever.retrieve(record)
    user_prompt, mode, allowed = build_prompt(record, retrieval)

    artifact = {
        "mollm_call_id": str(uuid.uuid4()),
        "entity_id": record.get("entity_id"),
        "note_id": record.get("note_id"),
        "confidence_tier_in": record.get("confidence_tier_in"),
        "mode": mode,
        "allowed_verdicts": sorted(allowed),
        # 2026-08-13 (report S6): hashed HERE, at decision time, because this
        # is the only point where the candidate list this verdict was made
        # against is in scope. store_decision() writes the 16 characters, not
        # the list. Without it, "was this decision graded against the same
        # candidates it was made against" is unanswerable after any Stage 2b
        # rerun -- the exact confound that caveats the NET VALUE = -17 result.
        "candidates_hash": candidates_hash(record.get("candidates")),
        # NULL means presented in Stage 2b's own order. When set,
        # candidate_permutation[i] is the ORIGINAL index of the candidate
        # shown in position i -- the mapping any grader needs to interpret a
        # RESOLVED_TO_CANDIDATE_n verdict from a shuffled run.
        "candidate_permutation": candidate_permutation,
        "retrieved_context": {
            "rules": retrieval.get("rules"),
            "channels_run": retrieval.get("channels_run"),
            "snomed_code": retrieval.get("snomed_code"),
            "rules_pooled_before_cap": retrieval.get("rules_pooled_before_cap"),
            "retrieval_skipped_reason": retrieval.get("retrieval_skipped_reason"),
            "hierarchy_available": retrieval.get("hierarchy_available"),
            "concept_context": retrieval.get("concept_context"),
            # Everything found-then-discarded, with reasons, plus the channels
            # that never ran and why. The prompt gets a one-line summary; the
            # artifact keeps the full record so a decision can be re-audited
            # against what was actually available at the time, not just what
            # was shown.
            "suppression": retrieval.get("suppression"),
            "channels_skipped": retrieval.get("channels_skipped"),
            "candidate_contexts": retrieval.get("candidate_contexts"),
            # The COMPLETE ranked evidence set, not just the five shown. This
            # is what makes a follow-up request answerable from the record and
            # what lets a re-audit ask "what else was available?" without
            # re-running retrieval against a KG that may have changed.
            "all_ranked_rules": retrieval.get("all_ranked_rules"),
            "all_ranked_rules_truncated_from": retrieval.get("all_ranked_rules_truncated_from"),
        },
        "prompt": user_prompt,
    }

    require_citation = bool(retrieval.get("rules"))
    model_results = []
    for name, client in clients.items():
        try:
            model_results.append(_query_one(client, SYSTEM_PROMPT, user_prompt, allowed,
                                              require_citation=require_citation))
        except (LLMUnavailable, ValueError) as exc:
            # An unavailable or unparseable model invalidates the ensemble.
            # Recorded and routed to a human rather than proceeding on one
            # opinion -- a single-model verdict is not the consensus check the
            # thesis claims, and silently degrading to it would misrepresent
            # what actually happened.
            artifact["models"] = model_results
            artifact["error"] = f"{name}: {exc}"
            artifact.update({
                "ensemble_agreement": False, "composite_confidence": None,
                "citation_verified": False, "citation_checks": [],
                "mollm_routing_decision": INGESTION_HITL,
                "queue_reason": "model_unavailable_or_invalid_output",
                "routing_basis": "ensemble incomplete; cannot claim consensus validation",
            })
            return artifact

    # ---- optional single expansion round -------------------------------
    # Honoured only if BOTH models asked, or if one asked and the other
    # returned INSUFFICIENT_EVIDENCE. A single model wanting more while the
    # other answered confidently is not a reason to widen the evidence for
    # both -- that would let one model's uncertainty rewrite the other's
    # input, and the two verdicts would no longer be independent.
    requests = [m.get("request") for m in model_results]
    verdicts = [m["verdict"] for m in model_results]
    wants = [r for r in requests if r]
    should_expand = bool(wants) and (
        len(wants) == len(model_results)
        or any(v == "INSUFFICIENT_EVIDENCE" for v in verdicts))

    if should_expand:
        request = wants[0]
        expanded = expand_evidence(retrieval, request)
        exp_prompt, exp_mode, exp_allowed = build_prompt(record, expanded)
        artifact["expansion"] = {
            "requested_by": [n for n, m in zip(clients.keys(), model_results) if m.get("request")],
            "request": request,
            "applied": expanded.get("expansion_applied"),
            "round_1_models": model_results,
            "round_1_prompt": user_prompt,
        }
        exp_require_citation = bool(expanded.get("rules"))
        second = []
        for name, client in clients.items():
            try:
                second.append(_query_one(client, SYSTEM_PROMPT, exp_prompt, exp_allowed,
                                          require_citation=exp_require_citation))
            except (LLMUnavailable, ValueError) as exc:
                artifact["expansion"]["error"] = f"{name}: {exc}"
                second = []
                break
        if second:
            model_results = second
            retrieval = expanded
            user_prompt = exp_prompt
            artifact["prompt"] = exp_prompt
            artifact["retrieved_context"]["expansion_applied"] = expanded.get("expansion_applied")

    # 2026-08-13: computed on the FINAL model_results (post-expansion, if any)
    # against the candidates actually shown -- record["candidates"] is the
    # presented order even under --shuffle-candidates (_shuffled_candidates()
    # reassigns record["candidates"] itself), so no separate permutation
    # handling is needed here the way _is_top1_override() needs one.
    for m in model_results:
        m["reasoning_verdict_check"] = reasoning_verdict_mismatch(
            m["verdict"], m.get("reasoning"), record.get("candidates") or [])

    citation = verify_citations(model_results[0], retrieval)
    for m in model_results[1:]:
        other = verify_citations(m, retrieval)
        citation["citation_verified"] = citation["citation_verified"] and other["citation_verified"]
        citation["citation_checks"].extend(other["citation_checks"])

    ensemble = combine(model_results)

    # docs/MoLLM_Redesign_Proposal.md S4.2/S4.3/S9.5 -- computed from the FINAL
    # model_results/retrieval (post-expansion, if any), before route() runs.
    grounding_basis, reasoning_bases = _compute_grounding_basis(retrieval, model_results)
    annotation_discrepancy = _compute_annotation_discrepancy(
        model_results[0]["verdict"], mode)
    rejected_candidates = _compute_rejected_candidates(model_results)

    calibrator_score = None
    if calibrator is not None:
        feature_ctx = {
            "confidence_tier_in": record.get("confidence_tier_in"),
            "mode": mode,
            "ensemble": ensemble,
            "retrieval": retrieval,
            "model_results": model_results,
            "grounding_basis": grounding_basis,
            "annotation_discrepancy": annotation_discrepancy,
            "rejected_candidates": rejected_candidates,
        }
        calibrator_score = calibrator.score(feature_ctx)

    routing = route(ensemble, citation, model_results,
                    calibrator_score=calibrator_score,
                    # 2026-08-13 (P2.1): the override gate needs to know
                    # whether the model had real retrieved guideline evidence
                    # to reason from, which is exactly what grounding_basis
                    # already records. Computed above, before route(), so this
                    # is a pass-through rather than a new derivation.
                    grounding_basis=grounding_basis,
                    candidate_permutation=candidate_permutation,
                    normalized_from=record.get("normalized_from"))

    modes = {m.get("decoding_mode") for m in model_results}
    artifact["decoding_modes"] = sorted(x for x in modes if x)
    if len(modes) > 1:
        # One model guided and the other not means their confidences are not
        # on the same scale, which silently breaks ensemble_agreement's
        # premise. Flagged rather than averaged over.
        artifact["decoding_mode_mismatch"] = True

    artifact["models"] = model_results
    artifact["grounding_basis"] = grounding_basis
    artifact["reasoning_bases"] = reasoning_bases
    artifact["annotation_discrepancy"] = annotation_discrepancy
    artifact["rejected_candidates"] = rejected_candidates
    artifact["calibrator_score"] = calibrator_score
    # 2026-08-13 (P4). Row-level flag: did ANY model in this decision's
    # ensemble degenerate. Persisted to mollm_decisions.degenerate_generation
    # so `SELECT count(*) FILTER (WHERE degenerate_generation)` before and
    # after the FREQUENCY_PENALTY change is a one-line query rather than a
    # re-parse of the models JSON. The per-model detail stays in that JSON.
    artifact["degenerate_generation"] = any(
        m.get("degenerate_generation") for m in model_results)
    artifact.update(ensemble)
    artifact.update(citation)
    artifact.update(routing)
    return artifact


def load_validation_records(conn, note_id: str, limit: int = None,
                            tier: str = None, include_subthreshold: bool = False) -> list:
    """Reads Stage 2 output back into the ValidationRecord shape Stage 3 expects.

    This is the Stage 2 -> Stage 3 boundary made concrete. It reads from the
    tables rather than taking run_pipeline()'s in-memory return value, because
    Stage 3 must be runnable independently: over a note processed hours ago,
    re-runnable after a threshold change without re-extracting, and restartable
    mid-batch. Coupling it to a live pipeline object would make all three
    impossible.

    The join is on entity_id -- the identifier minted in Stage 2a precisely so
    this join would have something stable to use. Before it existed,
    extracted_entities and normalized_entities were unique on two DIFFERENT
    composite keys and could not be reliably joined at all.

    include_subthreshold defaults False. Spans between SUBTHRESHOLD_FLOOR and
    EXTRACTION_THRESHOLD are retained for analysis but have NOT passed the
    extraction gate; feeding them to Stage 3 by default would quietly change
    what the pipeline claims to validate. Set True only when deliberately
    measuring how many of them Stage 3 would recover.

    2026-08-14: filters out superseded_by_split/superseded_by_growth rows.
    src/clinical_pipeline.py's split_compound_entities()/grow_entity_spans()
    keep a superseded parent's extracted_entities row for audit rather than
    deleting it (this codebase's flag-don't-delete policy), which meant this
    function was joining that stale parent back in ALONGSIDE its own correct
    split children/grown replacement -- e.g. "Gunshot wound to abdomen
    \nPneumonia" (17751158-DS-19) reaching Stage 3 as one composite entity
    even though its three split children ("Gunshot wound", "abdomen",
    "Pneumonia") were already sitting in normalized_entities too. Confirmed
    87 such stale rows in normalized_entities project-wide. Same fix
    evaluation/stage2b_cal_eval.py's query already applies.
    """
    where = ["e.note_id = ?",
             "(e.superseded_by_split IS NULL OR e.superseded_by_split = FALSE)",
             "(e.superseded_by_growth IS NULL OR e.superseded_by_growth = FALSE)"]
    params = [note_id]
    if not include_subthreshold:
        where.append("(e.below_threshold IS NULL OR e.below_threshold = FALSE)")
    if tier:
        where.append("n.confidence_tier_in = ?")
        params.append(tier)

    rows = conn.sql(f"""
        SELECT e.entity_id, e.note_id, e.original_text, e.expanded_text,
               e.entity_label, e.confidence, e.orig_start, e.orig_end,
               e.local_context, e.section_name, e.assertion_status, e.experiencer,
               e.temporality, e.assertion_cue, e.expansion_ambiguous,
               e.candidate_expansions,
               n.candidates, n.confidence_tier_in, n.is_ambiguous,
               n.ambiguity_reason, n.match_tier, n.matched, n.normalized_from
        FROM extracted_entities e
        JOIN normalized_entities n ON n.entity_id = e.entity_id
        WHERE {' AND '.join(where)}
        ORDER BY e.orig_start ASC
        {f'LIMIT {int(limit)}' if limit else ''}
    """, params=params).fetchall()

    def _json(v, default):
        if not v:
            return default
        try:
            return json.loads(v) if isinstance(v, str) else v
        except (ValueError, TypeError):
            return default

    records = []
    for r in rows:
        entity_id = r[0]
        # Relations touching this entity, from EITHER endpoint. A relation is
        # equally informative whichever side the entity sits on.
        rels = []
        try:
            for rel in conn.sql("""
                SELECT relation_id, relation_label, head_entity_id, tail_entity_id,
                       head_entity_text, tail_entity_text, relation_confidence,
                       head_link_status, tail_link_status
                FROM extracted_relations
                WHERE note_id = ? AND (head_entity_id = ? OR tail_entity_id = ?)
            """, params=[note_id, entity_id, entity_id]).fetchall():
                is_head = rel[2] == entity_id
                rels.append({
                    "relation_id": rel[0],
                    "relation_label": rel[1],
                    "other_endpoint_text": rel[5] if is_head else rel[4],
                    "other_entity_id": rel[3] if is_head else rel[2],
                    "relation_confidence": rel[6],
                    "entity_link_status": rel[8] if is_head else rel[7],
                })
        except Exception:
            pass

        records.append({
            "entity_id": entity_id, "note_id": r[1],
            "original_text": r[2], "expanded_text": r[3],
            "gliner_label": r[4], "gliner_confidence": r[5],
            "orig_start": r[6], "orig_end": r[7],
            "local_context": r[8] or "", "section_name": r[9],
            "assertion_status": r[10] or "PRESENT",
            "experiencer": r[11] or "PATIENT",
            "temporality": r[12] or "CURRENT",
            "assertion_cue": r[13],
            "expansion_ambiguous": bool(r[14]),
            "candidate_expansions": _json(r[15], None),
            "candidates": _json(r[16], []),
            "confidence_tier_in": r[17] or "HIGH",
            "is_ambiguous": bool(r[18]),
            "ambiguity_reason": r[19],
            "match_tier": r[20],
            "matched": r[21],
            "normalized_from": r[22],
            "relations": rels,
        })
    return records


def store_decision(artifact: dict, conn, is_test: bool = False):
    """Persists the decision artifact to DuckDB.

    Written to DuckDB rather than straight to the graph because KG3 ingestion
    is Stage 4's job (docs/Databases.md's Stage-to-DB matrix) and does not
    exist yet. Keeping the artifact here means Stage 3 can be run, measured and
    calibrated before Stage 4 lands, instead of being blocked on it. The full
    prompt is stored deliberately: without it, a decision cannot be re-audited,
    since the verdict is only interpretable against the exact evidence shown.
    """
    conn.sql("""
    CREATE TABLE IF NOT EXISTS mollm_decisions (
        mollm_call_id VARCHAR PRIMARY KEY,
        entity_id VARCHAR,
        note_id VARCHAR,
        confidence_tier_in VARCHAR,
        mode VARCHAR,
        ensemble_agreement BOOLEAN,
        composite_confidence DOUBLE,
        confidence_spread DOUBLE,
        confidence_basis VARCHAR,
        citation_verified BOOLEAN,
        cited_suppressed_rules JSON,
        expansion JSON,
        decoding_modes JSON,
        mollm_routing_decision VARCHAR,
        queue_reason VARCHAR,
        routing_basis VARCHAR,
        models JSON,
        retrieved_context JSON,
        citation_checks JSON,
        prompt VARCHAR,
        error VARCHAR,
        is_test BOOLEAN DEFAULT FALSE
    );
    """)
    # Additive migration for tables created before docs/MoLLM_Redesign_Proposal.md's
    # implementation (2026-08-11) -- CREATE TABLE IF NOT EXISTS above is a no-op
    # against an EC2 table that already exists from before these columns existed,
    # same pattern src/extraction.py's extracted_relations table already uses.
    for stmt in [
        "ALTER TABLE mollm_decisions ADD COLUMN IF NOT EXISTS grounding_basis VARCHAR;",
        "ALTER TABLE mollm_decisions ADD COLUMN IF NOT EXISTS reasoning_bases JSON;",
        "ALTER TABLE mollm_decisions ADD COLUMN IF NOT EXISTS annotation_discrepancy BOOLEAN;",
        "ALTER TABLE mollm_decisions ADD COLUMN IF NOT EXISTS rejected_candidates JSON;",
        "ALTER TABLE mollm_decisions ADD COLUMN IF NOT EXISTS calibrator_score DOUBLE;",
        "ALTER TABLE mollm_decisions ADD COLUMN IF NOT EXISTS confidence_basis_for_routing VARCHAR;",
        # confidence_spread was computed by combine() and merged into every
        # artifact via artifact.update(ensemble) from day one, but was never
        # actually listed in this table's column/INSERT list below -- silently
        # dropped on every write until 2026-08-11. Backfilling the column here;
        # existing rows will have NULL until re-run, which is honest (the value
        # was genuinely never persisted) rather than reconstructed after the fact.
        "ALTER TABLE mollm_decisions ADD COLUMN IF NOT EXISTS confidence_spread DOUBLE;",
        # 2026-08-13: degenerate-generation flag (report S4.2). Persisted so
        # the FREQUENCY_PENALTY fix's effect is MEASURABLE on a later run
        # rather than asserted -- S9.5 lists "not yet re-validated against a
        # live batch" as outstanding precisely because nothing recorded it.
        "ALTER TABLE mollm_decisions ADD COLUMN IF NOT EXISTS degenerate_generation BOOLEAN;",
        # 2026-08-13: the permutation candidates were PRESENTED in, when
        # scripts/run_stage3_batch.py --shuffle-candidates is used. NULL means
        # presented in Stage 2b's own order. See report S6's position-bias
        # hypothesis for why the ordering has to be recoverable per decision.
        "ALTER TABLE mollm_decisions ADD COLUMN IF NOT EXISTS candidate_permutation JSON;",
    ] + provenance_alter_statements("mollm_decisions"):
        conn.sql(stmt)
    conn.sql(f"""
    INSERT INTO mollm_decisions
    (mollm_call_id, entity_id, note_id, confidence_tier_in, mode, ensemble_agreement,
     composite_confidence, confidence_basis, citation_verified, cited_suppressed_rules,
     expansion, decoding_modes, mollm_routing_decision, queue_reason,
     routing_basis, models, retrieved_context, citation_checks, prompt,
     error, is_test, grounding_basis, reasoning_bases, annotation_discrepancy,
     rejected_candidates, calibrator_score, confidence_basis_for_routing,
     confidence_spread, degenerate_generation, candidate_permutation,
     {provenance_column_sql()})
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            {provenance_placeholders()})
    ON CONFLICT (mollm_call_id) DO NOTHING;
    """, params=[
        artifact["mollm_call_id"], artifact.get("entity_id"), artifact.get("note_id"),
        artifact.get("confidence_tier_in"), artifact.get("mode"),
        artifact.get("ensemble_agreement"), artifact.get("composite_confidence"),
        artifact.get("confidence_basis"), artifact.get("citation_verified"),
        json.dumps(artifact.get("cited_suppressed_rules"), default=str),
        json.dumps(artifact.get("expansion"), default=str) if artifact.get("expansion") else None,
        json.dumps(artifact.get("decoding_modes")),
        artifact.get("mollm_routing_decision"), artifact.get("queue_reason"),
        artifact.get("routing_basis"), json.dumps(artifact.get("models"), default=str),
        json.dumps(artifact.get("retrieved_context"), default=str),
        json.dumps(artifact.get("citation_checks"), default=str),
        artifact.get("prompt"), artifact.get("error"), is_test,
        artifact.get("grounding_basis"), json.dumps(artifact.get("reasoning_bases"), default=str),
        artifact.get("annotation_discrepancy"),
        json.dumps(artifact.get("rejected_candidates"), default=str),
        artifact.get("calibrator_score"), artifact.get("confidence_basis_for_routing"),
        artifact.get("confidence_spread"),
        artifact.get("degenerate_generation"),
        json.dumps(artifact.get("candidate_permutation")) if artifact.get("candidate_permutation") else None,
        # candidates_hash was computed in validate_record() where the candidate
        # list was actually in scope; pass it through rather than re-deriving
        # it from an artifact that does not carry the list.
        *provenance_params(precomputed_hash=artifact.get("candidates_hash")),
    ])
