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
import re
import uuid
import warnings

from src.llm_client import (
    LLMUnavailable,
    build_clients,
    extract_verdict_confidence,
    parse_json_response,
    verdict_schema,
)
from src.retrieval import CITABLE_TYPES

warnings.filterwarnings("ignore")

# Verdict vocabularies, closed per mode.
CONTRADICTION_VERDICTS = {"SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}
RESOLUTION_VERDICTS_BASE = {"NONE_CORRECT", "INSUFFICIENT_EVIDENCE"}

# Routing thresholds on composite_confidence. Calibration targets against the
# validation slice (docs/Evaluation_Criteria.md), NOT settled values -- they are
# named here so there is one place to change them after the ECE analysis.
AUTO_VALIDATE_THRESHOLD = 0.85
MOLLM_RESOLVE_THRESHOLD = 0.60

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

# Evidence cap after an expansion request. Higher than the first-round 5, but
# still bounded -- the 8,192-token window is the constraint, not willingness to
# show more.
MAX_RULES_AFTER_EXPANSION = 15

INGESTION_AUTO = "AUTO_VALIDATED"
INGESTION_RESOLVED = "MOLLM_RESOLVED"
INGESTION_HITL = "HITL_REQUIRED"

SYSTEM_PROMPT = """You are a clinical terminology validator in an auditable pipeline.

You will be given one entity extracted from a clinical note, the concept(s) a
terminology service mapped it to, and any clinical guideline evidence retrieved
from a curated knowledge graph.

RULES:
- Judge ONLY what you are shown. Do not use outside clinical knowledge as
  evidence for a verdict.
- You may cite evidence ONLY by the rule_id values in the EVIDENCE block. Never
  invent a rule_id, and never cite text that is not shown to you.
- If the EVIDENCE block says no guideline evidence was retrieved, you cannot
  return CONTRADICTED. Return SUPPORTED if the mapping is ontologically sound,
  otherwise INSUFFICIENT_EVIDENCE.
- Respect the stated assertion status. A finding marked ABSENT was explicitly
  negated in the note; do not treat it as present.

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
 "reasoning": "<two sentences maximum>",
 "cited_evidence": [{"rule_id": "<id from EVIDENCE>", "quote": "<exact text from that rule>"}],
 "confidence": "<HIGH|MEDIUM|LOW>",
 "request": "<MORE_RULES|SUPPRESSED_RULES|CANDIDATE_DETAIL|NONE>"}
Use an empty cited_evidence list if you cited nothing, and "NONE" if you need
no further information."""


# ==========================================================================
# Prompt assembly
# ==========================================================================

def _format_candidates(record, retrieval) -> str:
    cands = record.get("candidates") or []
    if not cands:
        return "CANDIDATE CONCEPTS: none — Stage 2 could not map this entity.\n"
    ctxs = retrieval.get("candidate_contexts") or []
    lines = ["CANDIDATE CONCEPTS:"]
    for i, c in enumerate(cands, 1):
        ctx = ctxs[i - 1] if i - 1 < len(ctxs) else {}
        line = (f"  [{i}] {c.get('concept_name')} (OMOP {c.get('omop_concept_id')}, "
                f"{c.get('vocabulary_id')}/{c.get('domain_id')}, "
                f"match tier {c.get('match_tier')}, score {c.get('similarity_score')})")
        parents = ctx.get("parents") or []
        if parents:
            line += "\n      is-a: " + "; ".join(p["name"] for p in parents[:3])
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
        f"  after abbreviation expansion: {record.get('expanded_text')!r}",
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
        parts.append(
            "TASK: Stage 2 could not decide between the candidate concepts above. "
            "Using the context, section and any evidence, decide which candidate the "
            "entity refers to.\n"
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


def _query_one(client, system_prompt, user_prompt, allowed_verdicts) -> dict:
    # Guided decoding constrains the verdict to allowed_verdicts, so the
    # out-of-vocabulary branch below should become unreachable -- but it is
    # kept, because llm_client falls back to unguided json_object mode if the
    # server rejects the guided request, and a silent fallback must not turn
    # into a silent coercion.
    raw = client.complete(system_prompt, user_prompt,
                          schema=verdict_schema(allowed_verdicts))
    parsed = parse_json_response(raw["text"])

    verdict = str(parsed.get("verdict", "")).strip().upper()
    if verdict not in allowed_verdicts:
        # An out-of-vocabulary verdict is treated as an abstention rather than
        # coerced to the nearest allowed value -- guessing what the model meant
        # would fabricate a clinical decision it did not make.
        parsed["_verdict_out_of_vocabulary"] = verdict or None
        verdict = "INSUFFICIENT_EVIDENCE"

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
        "verdict_out_of_vocabulary": parsed.get("_verdict_out_of_vocabulary"),
        "finish_reason": raw.get("finish_reason"),
        # Guided vs unguided changes the logprob distribution, so a calibration
        # set must not mix the two. Recorded per call rather than assumed.
        "decoding_mode": raw.get("decoding_mode"),
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
    verdicts = [m["verdict"] for m in model_results]
    agreement = len(set(verdicts)) == 1

    confs = [m["logprob_confidence"] for m in model_results if m["logprob_confidence"] is not None]
    if not confs:
        return {"ensemble_agreement": agreement, "composite_confidence": None,
                "confidence_basis": "no_logprobs_available", "confidence_spread": None}

    mean = sum(confs) / len(confs)
    spread = (max(confs) - min(confs)) if len(confs) > 1 else 0.0

    if not agreement:
        return {"ensemble_agreement": False, "composite_confidence": round(mean, 6),
                "confidence_basis": "verdicts_disagree_composite_not_used_for_routing",
                "confidence_spread": round(spread, 6)}

    composite = mean
    basis = "mean_logprob_agreeing_verdicts"
    if spread >= CONFIDENCE_SPREAD_TRIGGER:
        composite *= CONFIDENCE_SPREAD_PENALTY
        basis += "+spread_penalty"

    return {"ensemble_agreement": True, "composite_confidence": round(composite, 6),
            "confidence_basis": basis, "confidence_spread": round(spread, 6)}


def route(ensemble: dict, citation: dict, model_results: list) -> dict:
    """Final routing. The three safety rules are checked BEFORE the thresholds
    and each short-circuits, so no confidence score can override them."""
    if not ensemble["ensemble_agreement"]:
        return {"mollm_routing_decision": INGESTION_HITL,
                "queue_reason": "model_disagreement",
                "routing_basis": "safety_rule: independent validators returned different verdicts"}

    if not citation["citation_verified"]:
        return {"mollm_routing_decision": INGESTION_HITL,
                "queue_reason": "citation_verification_failed",
                "routing_basis": "safety_rule: model cited evidence not present in what it was shown"}

    composite = ensemble["composite_confidence"]
    if composite is None:
        return {"mollm_routing_decision": INGESTION_HITL,
                "queue_reason": "confidence_unmeasurable",
                "routing_basis": "no logprobs returned; cannot calibrate a routing decision"}

    verdict = model_results[0]["verdict"]
    if verdict == "CONTRADICTED":
        return {"mollm_routing_decision": INGESTION_HITL,
                "queue_reason": "guideline_contradiction",
                "routing_basis": "a flagged contradiction is a clinical finding for a human, "
                                 "not something to auto-resolve"}

    if verdict == "INSUFFICIENT_EVIDENCE" or verdict == "NONE_CORRECT":
        return {"mollm_routing_decision": INGESTION_HITL,
                "queue_reason": f"verdict_{verdict.lower()}",
                "routing_basis": "no usable resolution produced"}

    if composite >= AUTO_VALIDATE_THRESHOLD:
        return {"mollm_routing_decision": INGESTION_AUTO, "queue_reason": None,
                "routing_basis": f"composite_confidence {composite} >= {AUTO_VALIDATE_THRESHOLD}"}
    if composite >= MOLLM_RESOLVE_THRESHOLD:
        return {"mollm_routing_decision": INGESTION_RESOLVED, "queue_reason": None,
                "routing_basis": f"composite_confidence {composite} in "
                                 f"[{MOLLM_RESOLVE_THRESHOLD}, {AUTO_VALIDATE_THRESHOLD})"}
    return {"mollm_routing_decision": INGESTION_HITL,
            "queue_reason": "below_confidence_threshold",
            "routing_basis": f"composite_confidence {composite} < {MOLLM_RESOLVE_THRESHOLD}"}


def validate_record(record: dict, retriever, clients: dict = None) -> dict:
    """Runs Stage 3 for one entity and returns the decision artifact.

    Artifact field names follow docs/Provenance_Schema.md's Stage 3 spec so it
    can be written straight into the KG3 :MoLLMDecision node without a
    translation layer that could drift.
    """
    clients = clients if clients is not None else build_clients()
    retrieval = retriever.retrieve(record)
    user_prompt, mode, allowed = build_prompt(record, retrieval)

    artifact = {
        "mollm_call_id": str(uuid.uuid4()),
        "entity_id": record.get("entity_id"),
        "note_id": record.get("note_id"),
        "confidence_tier_in": record.get("confidence_tier_in"),
        "mode": mode,
        "allowed_verdicts": sorted(allowed),
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

    model_results = []
    for name, client in clients.items():
        try:
            model_results.append(_query_one(client, SYSTEM_PROMPT, user_prompt, allowed))
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
        second = []
        for name, client in clients.items():
            try:
                second.append(_query_one(client, SYSTEM_PROMPT, exp_prompt, exp_allowed))
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

    citation = verify_citations(model_results[0], retrieval)
    for m in model_results[1:]:
        other = verify_citations(m, retrieval)
        citation["citation_verified"] = citation["citation_verified"] and other["citation_verified"]
        citation["citation_checks"].extend(other["citation_checks"])

    ensemble = combine(model_results)
    routing = route(ensemble, citation, model_results)

    modes = {m.get("decoding_mode") for m in model_results}
    artifact["decoding_modes"] = sorted(x for x in modes if x)
    if len(modes) > 1:
        # One model guided and the other not means their confidences are not
        # on the same scale, which silently breaks ensemble_agreement's
        # premise. Flagged rather than averaged over.
        artifact["decoding_mode_mismatch"] = True

    artifact["models"] = model_results
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
    """
    where = ["e.note_id = ?"]
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
               n.ambiguity_reason, n.match_tier, n.matched
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
    conn.sql("""
    INSERT INTO mollm_decisions
    (mollm_call_id, entity_id, note_id, confidence_tier_in, mode, ensemble_agreement,
     composite_confidence, confidence_basis, citation_verified, cited_suppressed_rules,
     expansion, decoding_modes, mollm_routing_decision, queue_reason,
     routing_basis, models, retrieved_context, citation_checks, prompt,
     error, is_test)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    ])
