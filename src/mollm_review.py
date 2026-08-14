"""
src/mollm_review.py — Objective 3 (Active Learning / Feedback Loop) reviewer.

WHY THIS IS A SEPARATE FILE FROM src/mollm_ensemble.py, NOT A MODIFICATION OF IT.
mollm_ensemble.py implements Objective 2 ("Neuro-Symbolic Prompting Strategy" —
deterministic KG context injection, evidence-only reasoning, hard citation/
contradiction safety rules) exactly as written in the COM748 proposal, and it
is this project's primary novelty claim. It stays exactly as it is.

This module implements Objective 3 instead: "Engineer a semi-automatic
labelling pipeline utilizing uncertainty sampling. Low confidence predictions
will be routed to a human subject matter expert interface, while high
confidence annotations will be systematically written back to the KG..."
(Project Proposal Form, Project Objectives). The proposal's own methodology
section describes the mechanism this way: "A voting mechanism then evaluates
the confidence of the proposed labels. If the ensemble's consensus falls
below a predefined confidence threshold, the labels alongside its retrieved
contextual facts will be routed to a human subject matter expert (the oracle)
for manual annotations. High confidence labels are automatically accepted."
That is a description of a CONFIDENCE-DRIVEN ROUTING POLICY over an ensemble's
own judgment — not a description of evidence-only, citation-gated validation.
Grounding (Objective 2) is instrumental to that judgment, not the thing being
routed on.

WHAT IS DIFFERENT FROM mollm_ensemble.py, DELIBERATELY.
  1. NOT EVIDENCE-RESTRICTED. mollm_ensemble.py's SYSTEM_PROMPT forbids using
     outside medical knowledge whenever guideline evidence was retrieved, and
     forbids CONTRADICTED whenever it wasn't. This module's prompt explicitly
     asks the model to reason as a clinical/terminology expert using BOTH
     whatever KG evidence and prior pipeline decisions are shown AND its own
     medical knowledge, on every call, regardless of whether evidence exists.
  2. REVIEWS ACROSS ALL THREE CONFIDENCE TIERS. mollm_ensemble.py's resolution
     mode only ever runs on LOW-tier entities in practice (per
     scripts/run_stage3_batch.py's --tier default). This module calls
     load_validation_records() with no tier filter, so HIGH/MEDIUM/LOW-tier
     entities are all in scope for review — literally "all three tiers" of
     confidence_tier_in.
  3. CAN PROPOSE A CORRECTION TO EITHER STAGE 1 (the extractor's entity label)
     OR STAGE 2 (the normalizer's concept mapping), not just judge Stage 2's
     concept assignment against guideline text.
  4. ROUTING IS NOT GATED ON CITATION VERIFICATION OR A CONTRADICTED VERDICT.
     mollm_ensemble.py's route() has three hard safety rules that bypass the
     confidence math entirely (model disagreement, failed citation,
     CONTRADICTED). This module keeps exactly one of those in the same
     spirit — model disagreement still forces HITL, because two independent
     reviewers giving opposite answers IS the uncertainty signal Objective 3's
     "voting mechanism...confidence threshold" language describes — but does
     NOT hard-block on citation containment. Citations are still recorded and
     softly checked (evidence_alignment_notes) for the dissertation's
     reliability analysis, but a model that reasoned correctly from general
     knowledge without a KG citation is not penalized for it here, matching
     this module's explicit brief.

WHAT IS THE SAME, ON PURPOSE. Same two local models (src/llm_client.py),
same KG evidence retrieval (src/retrieval.py) and formatting
(_format_candidates/_format_evidence, imported from mollm_ensemble.py rather
than re-implemented, so both modules show the SAME evidence for the SAME
entity — the only thing that differs between them is policy, not input).
Kept as a 2-model ensemble rather than collapsed to one: disagreement between
two independent reviewers is itself a cheap, meaningful uncertainty signal for
the routing decision, and discarding it would throw away exactly the kind of
signal an active-learning querying strategy is supposed to use.

WHAT THIS MODULE DELIBERATELY DOES NOT DO.
  * Does not touch extracted_entities or normalized_entities. A proposed
    correction is a recommendation for a human reviewer's queue, never an
    automatic edit — Objective 3's "written back to the KG" step is KG3
    ingestion (Stage 4), which docs/Proposal_Alignment_Review.md §6 confirms
    does not exist yet ("triplets first... write-back should be the last
    step in the pipeline"). This module produces the CANDIDATE rows that a
    future Stage 4 job would consume; it does not perform the write-back
    itself.
  * Does not modify src/mollm_ensemble.py or its `mollm_decisions` table.
    Writes to a new `mollm_review_decisions` table instead, joinable on
    entity_id/note_id, so the two decision styles (grounded/citation-gated
    vs. confidence-driven/expert-judgment) can later be compared against each
    other and against gold — the natural next step this module is shaped to
    support, but not itself.
"""

import json
import os
import sys
import uuid
import warnings

# Bootstrap sys.path BEFORE the src.* imports below, not inside main().
# This file lives inside src/ itself, so `from src.llm_client import ...`
# only resolves if PROJECT_DIR (the parent of src/) is already importable --
# true automatically when a caller that already set up sys.path imports this
# module (e.g. a future scripts/run_review_batch.py, exactly how
# src/mollm_ensemble.py is normally used), but NOT true if this file is run
# directly as `python3 src/mollm_review.py`, since Python then puts this
# file's own directory (.../src) on sys.path[0], not its parent. Matches the
# pattern evaluation/mollm_trust_tiers.py already uses (sys.path.insert at
# true top-of-file, before its own src.* imports) rather than the
# PROJECT_DIR/sys.path setup mollm_ensemble.py's CALLERS do internally to
# main() -- that placement only works there because those callers (e.g.
# scripts/run_stage3_batch.py) are not themselves inside src/.
PROJECT_DIR = os.environ.get(
    "MOLLM_PROJECT_DIR", "/home/ec2-user/clinical_neuro_symbolic_pipeline")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.llm_client import (
    LLMUnavailable,
    build_clients,
    extract_verdict_confidence,
    parse_json_response,
)
from src.mollm_ensemble import (
    CITATION_CONTAINMENT_THRESHOLD,
    _citable_text,
    _containment,
    _format_candidates,
    _format_evidence,
    load_validation_records,  # reused as-is; the Stage2->Stage3 boundary
                               # query doesn't change just because the policy
                               # consuming it does.
)
from src.provenance import (
    provenance_alter_statements,
    provenance_column_sql,
    provenance_params,
    provenance_placeholders,
)

warnings.filterwarnings("ignore")

ASSESSMENT_VALUES = {
    "CORRECT",
    "ENTITY_LABEL_INCORRECT",
    "CONCEPT_MAPPING_INCORRECT",
    "BOTH_INCORRECT",
    "UNCERTAIN",
}
CORRECTION_ASSESSMENTS = {
    "ENTITY_LABEL_INCORRECT", "CONCEPT_MAPPING_INCORRECT", "BOTH_INCORRECT",
}

# KNOWN LIMITATION (observed 2026-08-13, note 10000032-DS-21, 27 freshly
# reviewed entities post-wording-fix): BioMistral still selects
# ENTITY_LABEL_INCORRECT for complaints whose reasoning is entirely about the
# concept NAME mapping (e.g. edema, Neck, HCV Cirrhosis, food poisoning,
# flank dullness, Paracentesis, Bipolar affective disorder, LAD -- 8 of the
# entities that round), despite SYSTEM_PROMPT explicitly distinguishing
# "entity TYPE" from "concept NAME mapping" in both the task description and
# the assessment-value definitions themselves. The wording fix was tested
# and did not resolve it, so this is being treated as a capability limit of
# the smaller/quantized (AWQ 4-bit) 7B model on this specific distinction,
# not a fixable prompt-clarity bug -- see docs/Proposal_Alignment_Review.md's
# note on model choice being narrow on a single 15.36GB T4. NOT chasing
# further prompt iterations on it: `assessment` alone is unreliable evidence
# of WHICH stage BioMistral actually means, but `reasoning` (free text) still
# carries the real signal, `proposed_concept_name`/`proposed_entity_label`
# still carry the real proposed fix, and ensemble disagreement still routes
# these to a human either way (route_review() does not depend on the
# category being right, only on the two models agreeing on one). A reviewer
# reading mollm_review_decisions.models[].reasoning is not misled by this;
# only an analysis that trusts `assessment` categories in isolation would be.


# Routing thresholds on composite_confidence. CALIBRATION-PENDING, same
# discipline as mollm_ensemble.py's AUTO_VALIDATE_THRESHOLD/
# MOLLM_RESOLVE_THRESHOLD: placeholders naming where to plug in a measured
# value once there is a labeled sample to check them against, not settled
# numbers. Deliberately a SEPARATE pair of constants from mollm_ensemble.py's
# rather than imported ones — this prompt's logprob distribution over a
# 5-way assessment enum is not the same distribution as that module's
# per-mode verdict enum, so there is no reason to expect the same cutoff
# values would mean the same thing here.
AL_AUTO_ACCEPT_THRESHOLD = 0.85
AL_PROVISIONAL_THRESHOLD = 0.60

AL_ACCEPTED = "AL_ACCEPTED"
AL_ACCEPTED_PROVISIONAL = "AL_ACCEPTED_PROVISIONAL"
AL_HITL_REQUIRED = "AL_HITL_REQUIRED"

SYSTEM_PROMPT = """You are a clinical terminology reviewer in a semi-automatic, \
active-learning labeling pipeline. Your review sits downstream of an automatic \
extractor and normalizer; your job is uncertainty-sampling triage — decide \
whether their output is correct, propose a fix if not, and say how confident \
you are, so the pipeline can auto-accept what you're sure of and route the \
rest to a human subject matter expert.

Your task is LABELING, not clinical decision-making: judge whether the extractor's \
entity TYPE and/or concept NAME mapping already chosen are the CORRECT selections for the \
text span shown, not whether a patient's care is clinically appropriate, not \
a diagnosis, and not a treatment recommendation.

You will be shown: the entity as extracted (Stage 1), the concept(s) it was \
mapped to (Stage 2), any KG guideline evidence retrieved for it, and — when \
available — a PRIOR AUTOMATED DECISION already made about this same entity \
by a separate, stricter validator. Use all of this as context, but you are \
NOT confined to it: you are expected to reason as a clinical/medical-\
terminology expert using your own knowledge as well, especially when the KG \
has nothing relevant (a knowledge graph is never assumed complete). When you \
lean on a specific fact you were shown, name it in "basis_notes"; when you \
lean on your own knowledge instead, say so there rather than implying it came \
from the evidence.

Assessment values:
  CORRECT                    - both the extractor's entity TYPE and concept NAME mapping are right
  ENTITY_LABEL_INCORRECT     - the Stage 1 extractor's entity TYPE is wrong; concept NAME mapping may or may not follow
  CONCEPT_MAPPING_INCORRECT  - the Stage 2 concept NAME mapping is wrong; extractor's entity TYPE is fine
  BOTH_INCORRECT             - both are wrong
  UNCERTAIN                  - you cannot defensibly call this either way; do not guess

UNCERTAIN IS NOT A DEFAULT FOR INCOMPLETE INFORMATION. A human reviewer never \
has perfect information either — the note is genuinely all there is. Missing \
context, an ambiguous abbreviation, or a differential of plausible readings is \
the NORMAL case, and a working clinician commits to their best-supported \
reading anyway, then flags remaining doubt through "confidence": "LOW" rather \
than refusing to answer. Reserve UNCERTAIN for when you truly cannot construct \
ANY defensible reading from what's shown — not when you merely lack \
certainty. If you can articulate even one clinically reasonable answer, give \
it as CORRECT or the relevant INCORRECT category, at LOW confidence if needed \
— do not retreat to UNCERTAIN instead.

If you choose anything other than CORRECT or UNCERTAIN, propose your best \
replacement in "proposed_entity_label" and/or "proposed_concept_name" — plain \
text, your own words, not required to match any candidate list. This is a \
recommendation for a human reviewer's queue, not an automatic edit, so give \
your honest best answer even under some uncertainty; reserve UNCERTAIN for \
when you genuinely have no defensible call to make.

Respect the stated assertion status: a finding marked ABSENT was explicitly \
negated in the note and should not be judged as though it were a present, \
current finding.

Reply with a single JSON object and nothing else:
{"assessment": "<one of the five values above>",
 "reasoning": "<two sentences maximum>",
 "proposed_entity_label": "<text, or empty string if not proposing one>",
 "proposed_concept_name": "<text, or empty string if not proposing one>",
 "basis_notes": "<what you leaned on: a specific shown fact, or 'own medical knowledge'>",
 "cited_evidence": [{"rule_id": "<id from EVIDENCE, if any>", "quote": "<exact text you're citing>"}],
 "confidence": "<HIGH|MEDIUM|LOW>"}
Use an empty cited_evidence list if you cited nothing from the EVIDENCE block."""


def review_schema() -> dict:
    """JSON Schema for guided decoding, same rationale as
    mollm_ensemble.verdict_schema(): makes an out-of-vocabulary assessment
    structurally impossible and puts the logprob mass on `assessment` behind
    the enum rather than the whole-vocabulary distribution.

    `cited_evidence` deliberately unconstrained to known rule_ids, same as
    mollm_ensemble.py — a fabricated rule_id is itself a finding worth being
    able to observe, not something to make impossible.
    """
    return {
        "type": "object",
        "properties": {
            "assessment": {"type": "string", "enum": sorted(ASSESSMENT_VALUES)},
            "reasoning": {"type": "string"},
            "proposed_entity_label": {"type": "string"},
            "proposed_concept_name": {"type": "string"},
            "basis_notes": {"type": "string"},
            "cited_evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_id": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ["rule_id", "quote"],
                },
            },
            "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
        },
        "required": ["assessment", "reasoning", "confidence"],
    }


# ==========================================================================
# Prompt assembly
# ==========================================================================

def _format_prior_decision(conn, entity_id: str) -> str:
    """Renders the PRIOR AUTOMATED DECISION block from mollm_decisions, if
    one exists for this entity.

    This is the "logs" half of the brief — feeding this reviewer awareness of
    what the stricter, evidence-gated validator already concluded, WITHOUT
    treating it as ground truth (labeled explicitly as such below). Read-only:
    this function never writes to mollm_decisions, and its absence (no prior
    Stage 3 run for this entity yet) is a normal, common case, not an error —
    this reviewer is meant to run standalone across all three confidence
    tiers, most of which mollm_ensemble.py's batch runner never even reaches
    (--tier defaults to LOW only).
    """
    try:
        rows = conn.execute("""
            SELECT mollm_call_id, mollm_routing_decision, ensemble_agreement,
                   composite_confidence, citation_verified, mode
            FROM mollm_decisions
            WHERE entity_id = ? AND error IS NULL
            ORDER BY mollm_call_id
        """, [entity_id]).fetchall()
    except Exception:
        # Table may not exist yet in a fresh DB (mollm_ensemble.py's batch
        # runner hasn't been run) -- absence of prior context, not a fault.
        return "", None

    if not rows:
        return "", None

    call_id, routing, agreement, composite, citation_verified, mode = rows[-1]
    block = (
        "\nPRIOR AUTOMATED DECISION (for context only — a separate, stricter, "
        "evidence-only validator already reviewed this entity; treat this as "
        "informative, not authoritative):\n"
        f"  mode: {mode}\n"
        f"  routing: {routing}\n"
        f"  models agreed: {agreement}\n"
        f"  composite_confidence: {composite}\n"
        f"  citation_verified: {citation_verified}\n"
    )
    return block, {
        "mollm_call_id": call_id, "mollm_routing_decision": routing,
        "ensemble_agreement": agreement, "composite_confidence": composite,
        "citation_verified": citation_verified, "mode": mode,
    }


def build_review_prompt(record: dict, retrieval: dict, prior_block: str) -> str:
    """Returns the user prompt. Unlike mollm_ensemble.build_prompt(), there is
    no mode branching -- the task ("is this correct, across all three
    confidence tiers, propose a fix if not") is the same shape regardless of
    which tier or assertion status the entity carries. Tier and assertion
    status are still shown, as context the model should weigh, not as a
    switch that changes which verdicts are legal.
    """
    assertion_line = (
        f"  assertion: {record.get('assertion_status', 'PRESENT')} / "
        f"experiencer: {record.get('experiencer', 'PATIENT')} / "
        f"temporality: {record.get('temporality', 'CURRENT')}"
    )
    if record.get("assertion_cue"):
        assertion_line += f"  (cue: \"{record['assertion_cue']}\")"

    parts = [
        "ENTITY (Stage 1 extraction):",
        f"  text as written: {record.get('original_text')!r}",
        f"  after abbreviation expansion: {record.get('expanded_text')!r}",
        f"  extractor label: {record.get('gliner_label')} "
        f"(confidence {record.get('gliner_confidence')})",
        f"  confidence tier assigned by the pipeline: {record.get('confidence_tier_in')}",
        assertion_line,
    ]
    if record.get("expansion_ambiguous"):
        parts.append(
            f"  NOTE: this abbreviation has multiple known expansions "
            f"{record.get('candidate_expansions')}. The one above was chosen "
            f"without context."
        )
    parts.append("")
    parts.append(f"SECTION: {record.get('section_name') or 'unknown'}")
    parts.append(f"CONTEXT: ...{record.get('local_context', '')}...")
    parts.append("")
    parts.append("STAGE 2 (normalization) mapping under review:")
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
    if prior_block:
        parts.append(prior_block)

    parts.append(
        "\nTASK: Using everything above AND your own clinical/terminology "
        "knowledge, decide whether the Stage 1 entity label and Stage 2 "
        "concept mapping are correct for this text. Propose a correction if "
        "either is wrong."
    )
    return "\n".join(parts)


# ==========================================================================
# Per-model query
# ==========================================================================

def _query_one_review(client, user_prompt: str) -> dict:
    raw = client.complete(SYSTEM_PROMPT, user_prompt, schema=review_schema())
    parsed = parse_json_response(raw["text"])

    assessment = str(parsed.get("assessment", "")).strip().upper()
    if assessment not in ASSESSMENT_VALUES:
        # Same discipline as mollm_ensemble._query_one(): an out-of-vocabulary
        # value is treated as an abstention, not coerced -- guessing what the
        # model meant would fabricate a review it did not actually give.
        assessment_out_of_vocab = assessment or None
        assessment = "UNCERTAIN"
    else:
        assessment_out_of_vocab = None

    return {
        "model": raw["model"],
        "assessment": assessment,
        "assessment_out_of_vocabulary": assessment_out_of_vocab,
        "reasoning": parsed.get("reasoning"),
        "proposed_entity_label": (parsed.get("proposed_entity_label") or "").strip() or None,
        "proposed_concept_name": (parsed.get("proposed_concept_name") or "").strip() or None,
        "basis_notes": parsed.get("basis_notes"),
        "cited_evidence": parsed.get("cited_evidence") or [],
        "raw_confidence_label": parsed.get("confidence"),
        "logprob_confidence": extract_verdict_confidence(raw["tokens"], assessment),
        "finish_reason": raw.get("finish_reason"),
        "decoding_mode": raw.get("decoding_mode"),
    }


def _soft_evidence_check(model_output: dict, retrieval: dict) -> dict:
    """Informational only -- unlike mollm_ensemble.verify_citations(), this
    NEVER gates routing (see module docstring, difference #4). Recorded so
    the dissertation can still report how often this reviewer's citations
    would have passed the strict check, without making that check a
    precondition for accepting this reviewer's judgment.
    """
    rules_by_id = {r["rule_id"]: r for r in (retrieval.get("rules") or [])}
    checks = []
    for c in (model_output.get("cited_evidence") or []):
        if not isinstance(c, dict):
            checks.append({"rule_id": None, "aligned": False, "reason": "malformed_citation"})
            continue
        rule = rules_by_id.get(c.get("rule_id"))
        if rule is None:
            checks.append({"rule_id": c.get("rule_id"), "aligned": False,
                           "reason": "rule_id_not_in_shown_evidence"})
            continue
        source_text = _citable_text(rule)
        score = _containment((c.get("quote") or "").strip(), source_text) if source_text else None
        checks.append({"rule_id": c.get("rule_id"),
                       "aligned": score is not None and score >= CITATION_CONTAINMENT_THRESHOLD,
                       "containment": round(score, 4) if score is not None else None})
    return {"evidence_alignment_checks": checks}


# ==========================================================================
# Ensemble combination + routing
# ==========================================================================

def _normalize_correction(text) -> str:
    return (text or "").strip().lower()


def combine_review(model_results: list) -> dict:
    """ensemble_agreement + composite_confidence, analogous to
    mollm_ensemble.combine() but with a different agreement rule: when both
    models propose a CORRECTION (not just CORRECT/UNCERTAIN), agreement also
    requires their proposed text to match. Two models both saying
    "CONCEPT_MAPPING_INCORRECT" while proposing two different replacement
    concepts is not consensus on what the pipeline should do next, even
    though they agree the current mapping is wrong -- collapsing that into
    "agreement=True" would silently pick one model's correction over the
    other's with no signal that they disagreed on the answer, not just the
    verdict category.
    """
    assessments = [m["assessment"] for m in model_results]
    same_assessment = len(set(assessments)) == 1

    agreement = same_assessment
    if same_assessment and assessments[0] in CORRECTION_ASSESSMENTS:
        labels = {_normalize_correction(m.get("proposed_entity_label")) for m in model_results}
        concepts = {_normalize_correction(m.get("proposed_concept_name")) for m in model_results}
        # Only compare the field(s) relevant to what was actually flagged
        # wrong; a field neither model was asked to fix (both left it blank)
        # trivially matches and should not count against agreement.
        if assessments[0] in ("ENTITY_LABEL_INCORRECT", "BOTH_INCORRECT") and len(labels) > 1:
            agreement = False
        if assessments[0] in ("CONCEPT_MAPPING_INCORRECT", "BOTH_INCORRECT") and len(concepts) > 1:
            agreement = False

    confs = [m["logprob_confidence"] for m in model_results if m["logprob_confidence"] is not None]
    if not confs:
        return {"ensemble_agreement": agreement, "composite_confidence": None,
                "confidence_basis": "no_logprobs_available", "confidence_spread": None}

    mean = sum(confs) / len(confs)
    spread = (max(confs) - min(confs)) if len(confs) > 1 else 0.0

    if not agreement:
        return {"ensemble_agreement": False, "composite_confidence": round(mean, 6),
                "confidence_basis": "assessments_disagree_composite_not_used_for_routing",
                "confidence_spread": round(spread, 6)}

    return {"ensemble_agreement": True, "composite_confidence": round(mean, 6),
            "confidence_basis": "mean_logprob_agreeing_assessment",
            "confidence_spread": round(spread, 6)}


def route_review(ensemble: dict, model_results: list) -> dict:
    """Final AL routing. Confidence-driven per Objective 3's own description
    ("a voting mechanism...evaluates the confidence of the proposed labels").
    Exactly one hard rule ahead of the thresholds: model disagreement, which
    is itself the strongest uncertainty signal available and is the same
    signal the proposal's own "consensus falls below threshold" language is
    describing at its most extreme (zero consensus). UNCERTAIN from either
    model forces HITL too -- a model that says "I can't tell" is not
    something a confidence score should be allowed to override, regardless
    of how confident it sounds saying it.

    Deliberately does NOT check citation_verified/evidence_alignment here --
    see module docstring difference #4.
    """
    if not ensemble["ensemble_agreement"]:
        return {"al_routing_decision": AL_HITL_REQUIRED,
                "queue_reason": "reviewer_disagreement",
                "routing_basis": "independent reviewers gave different assessments "
                                 "or different proposed corrections"}

    assessment = model_results[0]["assessment"]
    if assessment == "UNCERTAIN":
        return {"al_routing_decision": AL_HITL_REQUIRED,
                "queue_reason": "reviewers_uncertain",
                "routing_basis": "both reviewers declined to make a defensible call"}

    composite = ensemble["composite_confidence"]
    if composite is None:
        return {"al_routing_decision": AL_HITL_REQUIRED,
                "queue_reason": "confidence_unmeasurable",
                "routing_basis": "no logprobs returned; cannot compute a routing confidence"}

    if composite >= AL_AUTO_ACCEPT_THRESHOLD:
        return {"al_routing_decision": AL_ACCEPTED, "queue_reason": None,
                "routing_basis": f"composite_confidence {composite} >= {AL_AUTO_ACCEPT_THRESHOLD}"}
    if composite >= AL_PROVISIONAL_THRESHOLD:
        return {"al_routing_decision": AL_ACCEPTED_PROVISIONAL, "queue_reason": None,
                "routing_basis": f"composite_confidence {composite} in "
                                 f"[{AL_PROVISIONAL_THRESHOLD}, {AL_AUTO_ACCEPT_THRESHOLD})"}
    return {"al_routing_decision": AL_HITL_REQUIRED,
            "queue_reason": "below_confidence_threshold",
            "routing_basis": f"composite_confidence {composite} < {AL_PROVISIONAL_THRESHOLD}"}


# ==========================================================================
# Entry point
# ==========================================================================

def review_record(record: dict, retriever, conn, clients: dict = None) -> dict:
    """Runs the Objective-3 review for one entity and returns the decision
    artifact. Mirrors mollm_ensemble.validate_record()'s shape (same
    retriever, same per-model try/except-on-unavailable pattern, same
    "ensemble incomplete -> HITL, never proceed on one opinion" discipline)
    so the two artifact styles stay comparable, but calls a completely
    different prompt/schema/routing policy -- see module docstring.

    `conn` is required (unlike validate_record()) because this function reads
    mollm_decisions for the prior-decision context block. It is not written
    to here; only mollm_review_decisions is (via store_review_decision()).
    """
    clients = clients if clients is not None else build_clients()
    retrieval = retriever.retrieve(record)
    prior_block, prior_meta = _format_prior_decision(conn, record.get("entity_id"))
    user_prompt = build_review_prompt(record, retrieval, prior_block)

    artifact = {
        "review_call_id": str(uuid.uuid4()),
        "entity_id": record.get("entity_id"),
        "note_id": record.get("note_id"),
        "confidence_tier_in": record.get("confidence_tier_in"),
        "original_text": record.get("original_text"),
        "prior_stage3_decision": prior_meta,
        "retrieved_context": {
            "rules": retrieval.get("rules"),
            "channels_run": retrieval.get("channels_run"),
            "retrieval_skipped_reason": retrieval.get("retrieval_skipped_reason"),
            "candidate_contexts": retrieval.get("candidate_contexts"),
        },
        "prompt": user_prompt,
    }

    model_results = []
    for name, client in clients.items():
        try:
            model_results.append(_query_one_review(client, user_prompt))
        except (LLMUnavailable, ValueError) as exc:
            # Same rule as mollm_ensemble.py: an unavailable/unparseable
            # model invalidates the ensemble rather than silently degrading
            # to a single opinion.
            artifact["models"] = model_results
            artifact["error"] = f"{name}: {exc}"
            artifact.update({
                "ensemble_agreement": False, "composite_confidence": None,
                "al_routing_decision": AL_HITL_REQUIRED,
                "queue_reason": "model_unavailable_or_invalid_output",
                "routing_basis": "ensemble incomplete; cannot claim consensus review",
            })
            return artifact

    for m in model_results:
        m["evidence_alignment"] = _soft_evidence_check(m, retrieval)

    ensemble = combine_review(model_results)
    routing = route_review(ensemble, model_results)

    modes = {m.get("decoding_mode") for m in model_results}
    artifact["decoding_modes"] = sorted(x for x in modes if x)
    if len(modes) > 1:
        artifact["decoding_mode_mismatch"] = True

    artifact["models"] = model_results
    artifact["assessment"] = model_results[0]["assessment"] if ensemble["ensemble_agreement"] else None
    artifact["proposed_entity_label"] = (
        model_results[0]["proposed_entity_label"] if ensemble["ensemble_agreement"] else None)
    artifact["proposed_concept_name"] = (
        model_results[0]["proposed_concept_name"] if ensemble["ensemble_agreement"] else None)
    artifact.update(ensemble)
    artifact.update(routing)
    return artifact


def _ensure_review_table(conn):
    """CREATE TABLE IF NOT EXISTS for mollm_review_decisions, factored out of
    store_review_decision() so main()'s resume-check (a SELECT, run before
    any row has necessarily ever been stored) can call it too. On a fresh DB
    the table has never been created, and DuckDB's SELECT ... FROM a
    not-yet-existing table is a CatalogException, not an empty result set --
    unlike _format_prior_decision()'s read of mollm_decisions (which can
    genuinely be a normal, pre-Stage-3 state for a given DB), a fresh
    mollm_review_decisions table is not an error state for THIS module either,
    so the fix is to guarantee the table exists rather than to catch and
    swallow the exception -- the caller wants the resume-check's absence of
    rows to be meaningful ("nothing reviewed yet"), not silently coincide with
    "the table isn't there".
    """
    conn.sql("""
    CREATE TABLE IF NOT EXISTS mollm_review_decisions (
        review_call_id VARCHAR PRIMARY KEY,
        entity_id VARCHAR,
        note_id VARCHAR,
        confidence_tier_in VARCHAR,
        original_text VARCHAR,
        prior_stage3_call_id VARCHAR,
        prior_stage3_routing VARCHAR,
        assessment VARCHAR,
        ensemble_agreement BOOLEAN,
        composite_confidence DOUBLE,
        confidence_spread DOUBLE,
        confidence_basis VARCHAR,
        proposed_entity_label VARCHAR,
        proposed_concept_name VARCHAR,
        al_routing_decision VARCHAR,
        queue_reason VARCHAR,
        routing_basis VARCHAR,
        decoding_modes JSON,
        models JSON,
        retrieved_context JSON,
        prompt VARCHAR,
        error VARCHAR,
        is_test BOOLEAN DEFAULT FALSE
    );
    """)
    # 2026-08-13 (report S6 / P2.3): same four provenance columns every other
    # decision-bearing table gets. Added here even though this table is new
    # and currently uncontaminated -- mollm_decisions was new once too, and
    # the cost of retrofitting provenance after rows accumulate is that the
    # rows already written can never have it.
    for stmt in provenance_alter_statements("mollm_review_decisions"):
        conn.sql(stmt)


def store_review_decision(artifact: dict, conn, is_test: bool = False):
    """Persists to `mollm_review_decisions` -- a NEW table, entirely separate
    from mollm_ensemble.store_decision()'s `mollm_decisions`. Joinable on
    entity_id/note_id for a later side-by-side comparison; this function does
    not perform that comparison itself.
    """
    _ensure_review_table(conn)
    prior = artifact.get("prior_stage3_decision") or {}
    conn.sql(f"""
    INSERT INTO mollm_review_decisions
    (review_call_id, entity_id, note_id, confidence_tier_in, original_text,
     prior_stage3_call_id, prior_stage3_routing, assessment, ensemble_agreement,
     composite_confidence, confidence_spread, confidence_basis,
     proposed_entity_label, proposed_concept_name, al_routing_decision,
     queue_reason, routing_basis, decoding_modes, models, retrieved_context,
     prompt, error, is_test, {provenance_column_sql()})
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            {provenance_placeholders()})
    ON CONFLICT (review_call_id) DO NOTHING;
    """, params=[
        artifact["review_call_id"], artifact.get("entity_id"), artifact.get("note_id"),
        artifact.get("confidence_tier_in"), artifact.get("original_text"),
        prior.get("mollm_call_id"), prior.get("mollm_routing_decision"),
        artifact.get("assessment"), artifact.get("ensemble_agreement"),
        artifact.get("composite_confidence"), artifact.get("confidence_spread"),
        artifact.get("confidence_basis"), artifact.get("proposed_entity_label"),
        artifact.get("proposed_concept_name"), artifact.get("al_routing_decision"),
        artifact.get("queue_reason"), artifact.get("routing_basis"),
        json.dumps(artifact.get("decoding_modes")),
        json.dumps(artifact.get("models"), default=str),
        json.dumps(artifact.get("retrieved_context"), default=str),
        artifact.get("prompt"), artifact.get("error"), is_test,
        # Review mode shows the full candidate list rather than Stage 3's
        # filtered one (this module's whole premise), so hash whatever this
        # artifact was actually built against.
        *provenance_params(candidates=artifact.get("candidates")),
    ])


# ==========================================================================
# CLI batch runner -- mirrors scripts/run_stage3_batch.py's shape so this is
# directly runnable rather than import-only. Kept in this file rather than a
# separate scripts/run_review_batch.py because, unlike Stage 3, this policy
# IS the whole module (no separate retrieval-design doc to keep in sync
# against) -- evaluation/mollm_trust_tiers.py sets the same precedent of a
# self-contained argparse-driven module in this codebase.
# ==========================================================================

def main():
    import argparse
    import collections
    import time
    import traceback

    import duckdb

    # PROJECT_DIR/sys.path already set up at module import time (see top of
    # file) -- not repeated here.
    DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

    TRIPLETS_CANDIDATES = [
        os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded_rules_added"),
        os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded"),
        os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned"),
    ]

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None,
                     help="Comma-separated note_ids. Default: every note_id with "
                          "is_test=TRUE rows in extracted_entities.")
    ap.add_argument("--tier", default=None, choices=["HIGH", "MEDIUM", "LOW"],
                     help="Restrict to one confidence_tier_in. Default: none -- "
                          "reviews all three tiers, per this module's brief.")
    ap.add_argument("--limit-per-note", type=int, default=None)
    ap.add_argument("--max-consecutive-failures", type=int, default=5)
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    print("=" * 78)
    print("MoLLM REVIEW BATCH RUN (Objective 3 -- active-learning reviewer)")
    print("=" * 78)

    from scripts.test_stage3_live import preflight
    from src.llm_client import MODEL_NAMES

    if not preflight(MODEL_NAMES):
        print("\nAborting -- pull the missing model(s) with `ollama pull <model>` first.")
        return 1

    from src.retrieval import DuckDBHierarchy, GroundingRetriever, GuidelineIndex, VocabularyRetriever

    triplets = next((p for p in TRIPLETS_CANDIDATES if os.path.exists(p)), None)
    if not triplets:
        print(f"\nNo guideline corpus found. Tried: {TRIPLETS_CANDIDATES}")
        return 1

    conn = duckdb.connect(args.db, read_only=False)
    try:
        if args.note_ids:
            note_ids = [n.strip() for n in args.note_ids.split(",") if n.strip()]
        else:
            note_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test = TRUE"
            ).fetchall()]
        if not note_ids:
            raise SystemExit("No is_test=TRUE rows in extracted_entities. "
                             "Run scripts/test_pipeline_e2e.py first.")

        print(f"db:    {args.db}")
        print(f"notes: {len(note_ids)}")
        print(f"tier:  {args.tier or 'ALL (HIGH+MEDIUM+LOW)'}")

        index = GuidelineIndex(triplets)
        vocab = VocabularyRetriever(conn)
        retriever = GroundingRetriever(index, vocab, hierarchy=DuckDBHierarchy(conn))
        clients = build_clients()
        print(f"guideline KG: {index.stats['nodes']} nodes, {index.stats['rules']} rules")

        _ensure_review_table(conn)  # first run on a fresh DB: table must
                                     # exist before the resume-check SELECT below.
        already_done = {r[0] for r in conn.execute("""
            SELECT DISTINCT entity_id FROM mollm_review_decisions
            WHERE is_test = TRUE AND error IS NULL AND note_id IN ({})
        """.format(",".join("?" * len(note_ids))), note_ids).fetchall()} if note_ids else set()
        print(f"resume check: {len(already_done)} entity_id(s) already reviewed\n")

        routing_totals = collections.Counter()
        assessment_totals = collections.Counter()
        n_processed = n_skipped = n_errors = 0
        consecutive_failures = 0
        start_time = time.time()

        for note_idx, note_id in enumerate(note_ids, 1):
            records = load_validation_records(conn, note_id, tier=args.tier)
            if args.limit_per_note:
                records = records[:args.limit_per_note]
            todo = [r for r in records if r["entity_id"] not in already_done]
            n_skipped += len(records) - len(todo)

            print(f"[note {note_idx}/{len(note_ids)}] {note_id}: {len(records)} "
                  f"record(s), {len(todo)} remaining after resume-skip")

            for rec_idx, rec in enumerate(todo, 1):
                elapsed_min = (time.time() - start_time) / 60
                print(f"  [{rec_idx}/{len(todo)}] {rec['original_text']!r} "
                      f"({rec['gliner_label']}, tier={rec['confidence_tier_in']})  "
                      f"[elapsed {elapsed_min:.1f}m | done {n_processed} | "
                      f"skipped {n_skipped} | errors {n_errors}]")
                try:
                    artifact = review_record(rec, retriever, conn, clients=clients)
                except Exception as exc:
                    n_errors += 1
                    consecutive_failures += 1
                    print(f"    ERROR: {exc.__class__.__name__}: {exc}")
                    traceback.print_exc()
                    if consecutive_failures >= args.max_consecutive_failures:
                        print(f"\n{consecutive_failures} consecutive failures -- stopping early.")
                        return 1
                    continue

                consecutive_failures = 0
                n_processed += 1
                already_done.add(rec["entity_id"])
                routing_totals[artifact.get("al_routing_decision")] += 1
                assessment_totals[artifact.get("assessment")] += 1

                if artifact.get("error"):
                    print(f"    artifact error: {artifact['error']} "
                          f"-> {artifact.get('al_routing_decision')}")
                else:
                    print(f"    -> {artifact.get('al_routing_decision')} "
                          f"({artifact.get('queue_reason') or 'ok'}), "
                          f"assessment={artifact.get('assessment')}")

                store_review_decision(artifact, conn, is_test=True)

        elapsed_min = (time.time() - start_time) / 60
        print("\n" + "=" * 78)
        print("REVIEW BATCH COMPLETE")
        print("=" * 78)
        print(f"processed: {n_processed}   skipped(resume): {n_skipped}   errors: {n_errors}")
        print(f"elapsed: {elapsed_min:.1f} minutes")
        print(f"\nrouting:    {dict(routing_totals)}")
        print(f"assessment: {dict(assessment_totals)}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
