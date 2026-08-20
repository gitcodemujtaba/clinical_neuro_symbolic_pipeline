"""src/guideline_evidence.py — 2026-08-20: wires the guideline-derived KG
(src.retrieval.GuidelineIndex, the 76-file curated triplet corpus reviewed
in docs/Guideline_Triplets_KG_Review.md) into the PRODUCTION tier gate's
tiebreak prompt for the first time.

WHY THIS EXISTS. Proposal alignment gap analysis (2026-08-20,
docs/2026-08-20_Session_Results_And_Status.md §9) found the production
gate (src.mollm_tier_gate) never actually consumed the guideline-derived
KG the project proposal specifically describes ("deterministic context
injection... from established medical guidelines") -- GuidelineIndex is
real, working, already-built code (Channels A/D of the SUPERSEDED old
ensemble, src.mollm_ensemble), just never reached by route_tier(). This
module is the bridge: a thin, additive evidence formatter, not a new KG
mechanism.

MATCHED BY NAME + TYPE, NOT SNOMED CODE (2026-08-20, deliberate design
choice). The triplet corpus's own SNOMED codes carry a documented
quality caveat -- several nodes are flagged quality_flag=
"same_snomed_type_mismatch_not_merged" in the source JSON-LD, meaning
the corpus's own curators already found cases where trusting the code
alone would wrongly conflate distinct concepts. Matching on the node's
`name` (case-insensitive exact match, via GuidelineIndex.nodes_for_name())
plus a soft compatibility check against the node's own `@type` (Finding/
Condition/Intervention/Medication/Acuity/...) is more conservative and
more directly tied to what a guideline document actually says, rather
than trusting a crosswalk this corpus's own quality flags already
call into question.

REAL, MEASURED COVERAGE, NOT ASSUMED. GuidelineIndex covers 1,286
distinct node names across 76 files (1,700 nodes, 1,162 rules). Checked
directly against every candidate concept currently in the DB: 67 of
7,151 distinct candidate concept names (0.9%) have an exact
case-insensitive name match in the guideline corpus. Modest, not
transformative -- reported honestly rather than oversold. The point of
this module is closing a real architectural gap against the project
proposal, not a precision play.

SAME "EVIDENCE TO WEIGH, NEVER A DETERMINISTIC OVERRIDE" DISCIPLINE
already established for src.tier4_kg_escalation's KG evidence and
CONDITION_VS_OBSERVATION_PRIOR's corpus-convention prior -- this only
ever ADDS a factual evidence block to the tiebreak prompt; it never
short-circuits or overrides model reasoning.

OFF BY DEFAULT (GUIDELINE_EVIDENCE_ENABLED / CNSP_GUIDELINE_EVIDENCE),
same pattern as CNSP_HYBRID_RETRIEVAL / ACRONYM_ESCALATION_ENABLED --
this touches an already-tuned, live prompt (the tiebreak path, which
directly decides TIER_2/TIER_4 split-vote resolution), so it needs its
own validation batch before flipping on, not a silent behavior change.
"""
import os

GUIDELINE_EVIDENCE_ENABLED = os.environ.get(
    "CNSP_GUIDELINE_EVIDENCE", "").strip().lower() in ("1", "true", "yes")

_INDEX = None


def get_guideline_index():
    """Lazily-loaded, process-wide singleton -- GuidelineIndex is
    file-backed, read-only, and loads in well under a second (confirmed:
    0.06s for the full 76-file corpus), so one load per process is cheap
    and there is no mutable state to worry about sharing across calls."""
    global _INDEX
    if _INDEX is None:
        from src.retrieval import GuidelineIndex
        _INDEX = GuidelineIndex()
    return _INDEX


# guideline node @type -> OMOP domain_id values that type is compatible
# with, a SOFT filter (not a hard requirement -- a name match with no
# type entry here, or a candidate with no domain_id, still passes; this
# only EXCLUDES a name match when both sides have a type/domain and they
# are clearly incompatible, e.g. a "Medication" guideline node matching a
# Procedure-domain candidate by name coincidence).
_TYPE_TO_COMPATIBLE_DOMAINS = {
    "Condition": {"Condition"},
    "Finding": {"Condition", "Observation"},
    "Medication": {"Drug"},
    "Intervention": {"Procedure"},
}


def _type_compatible(node_type: str, domain_id: str) -> bool:
    compatible = _TYPE_TO_COMPATIBLE_DOMAINS.get(node_type)
    if compatible is None or not domain_id:
        return True  # no known constraint either side -- don't false-exclude
    return domain_id in compatible


def guideline_evidence_for_candidates(guideline_index, candidates: list, vocab=None) -> str:
    """Real guideline-derived evidence (node names + the rules touching
    them, with rationale/citation) for whichever of `candidates` have a
    name the guideline corpus actually covers, filtered by a soft
    type/domain compatibility check (see module docstring for why name+
    type, not SNOMED code). `candidates` here is _tiebreak_prompt()'s own
    `accepted` list -- each item a dict with an `index` and a `candidate`
    (the full candidate dict). `vocab` is accepted for call-site
    compatibility but unused by name-based matching.

    Returns "" (not None) when no candidate has any guideline coverage --
    the caller can just check truthiness, no special-casing needed.
    """
    blocks = []
    for item in candidates:
        cand = item["candidate"]
        concept_name = cand.get("concept_name")
        if not concept_name:
            continue
        nodes = guideline_index.nodes_for_name(concept_name)
        if not nodes:
            continue
        nodes = [n for n in nodes if _type_compatible(n.get("node_type"), cand.get("domain_id"))]
        for node in nodes:
            rules = guideline_index.rules_touching(node["uid"])
            if not rules:
                continue
            rule_lines = []
            for r in rules[:3]:  # cap per node -- a hub concept can have many
                rationale = r.get("rationale") or "(no rationale recorded)"
                citation = r.get("citation")
                direction = "->" if r["direction"] == "outgoing" else "<-"
                rule_lines.append(
                    f"      {direction} {r.get('predicate')}: {rationale}"
                    + (f" [source: {citation}]" if citation else ""))
            if rule_lines:
                blocks.append(
                    f"  candidate [{item['index']}] ({cand.get('concept_name')}, "
                    f"guideline type: {node.get('node_type') or 'unknown'}) -- "
                    f"guideline evidence from "
                    f"{node.get('source_document') or 'unknown source'}:\n"
                    + "\n".join(rule_lines))
    if not blocks:
        return ""
    return (
        "OFFICIAL GUIDELINE EVIDENCE (from curated clinical-guideline triplets, "
        "not a similarity guess -- weigh this as real evidence, it does not "
        "decide the answer for you):\n" + "\n".join(blocks)
    )
