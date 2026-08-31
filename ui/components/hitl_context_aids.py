"""ui/components/hitl_context_aids.py -- 2026-08-31: pure, testable
context-aid computations for the HITL review page -- factored out of the
page itself so they're unit-testable without a live Streamlit run.

Two real aids:
  agreement_summary() -- a one-line "3/3 agree" vs "2-1 split" read,
    computed the exact same way route_tier()'s own plurality logic does
    (reuses evaluation.tier_gate_grading.plurality_candidate_index()
    directly, not a re-derived copy), so a reviewer can triage at a
    glance which cases need real scrutiny vs. a quick approve.
  known_risk_flags() -- surfaces the SAME hard-trap/near-duplicate
    patterns this project's own pipeline code already checks for
    (src.mollm_tier_gate's coronary-segment trap, short-alphanumeric-code
    trap) plus a domain-conflict check between the top two candidates --
    real, documented failure patterns this session found repeatedly
    (Procedure-vs-Observation near-duplicates), not a guessed list.
"""


def agreement_summary(models: list) -> dict:
    """{"total_usable": int, "top_verdict": str|None, "top_count": int,
    "unanimous": bool, "vote_counts": dict, "label": str} -- `label` is
    the ready-to-display one-liner."""
    from evaluation.tier_gate_grading import plurality_candidate_index

    _, top_verdict, vote_counts = plurality_candidate_index(models or [])
    vote_counts = dict(vote_counts or {})
    total_usable = sum(vote_counts.values())
    total_models = len(models or [])
    top_count = vote_counts.get(top_verdict, 0) if top_verdict else 0
    unanimous = total_usable > 0 and top_count == total_usable

    if total_usable == 0:
        label = f"⚠️ No usable votes ({total_models} model(s), all degenerate/error)"
    elif unanimous:
        label = f"✅ {top_count}/{top_count} agree: {top_verdict}"
    else:
        parts = ", ".join(f"{v}×{n}" for v, n in sorted(vote_counts.items(), key=lambda kv: -kv[1]))
        label = f"⚠️ Split ({total_usable} usable): {parts}"

    return {"total_usable": total_usable, "top_verdict": top_verdict, "top_count": top_count,
           "unanimous": unanimous, "vote_counts": vote_counts, "label": label}


def known_risk_flags(original_text: str, candidates: list, suggested_omop_concept_id) -> list:
    """List of human-readable warning strings for known, documented
    failure patterns -- empty list if none apply. Every check here mirrors
    a REAL check already live in src.mollm_tier_gate's production gate,
    or a real domain-conflict pattern this project measured repeatedly
    this session -- not a speculative list."""
    from src.mollm_tier_gate import _is_coronary_segment_trap, _is_short_alphanumeric_code

    flags = []
    entity = {"original_text": original_text or ""}
    candidate_index = None
    for i, c in enumerate(candidates or [], 1):
        if c.get("omop_concept_id") == suggested_omop_concept_id:
            candidate_index = i
            break

    if _is_short_alphanumeric_code(entity):
        flags.append("🔤 Short alphanumeric/shorthand mention (e.g. `S2`, `LAD`) — this exact "
                     "shape has been a real, repeated source of wrong auto-approvals this "
                     "project measured (coronary-segment and lab-code collisions).")
    if _is_coronary_segment_trap(entity, candidate_index, candidates or []):
        flags.append("❤️ Coronary-artery-segment abbreviation pattern — this project found "
                     "these resolve to an overly generic parent concept ('Coronary artery "
                     "structure') instead of the specific named branch.")

    if candidates and len(candidates) >= 2:
        top_domain = candidates[0].get("domain_id")
        second_domain = candidates[1].get("domain_id")
        if top_domain and second_domain and top_domain != second_domain:
            flags.append(f"🔀 Domain conflict between top 2 candidates ({top_domain} vs. "
                         f"{second_domain}) — this project repeatedly found near-duplicate "
                         f"SNOMED concept pairs that differ only by domain/class (e.g. "
                         f"'Allergy to X' vs. 'Allergic reaction caused by X', a lab test's "
                         f"Procedure vs. Observable-Entity form). Worth checking closely.")

    return flags
