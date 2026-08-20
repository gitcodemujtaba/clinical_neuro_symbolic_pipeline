"""src/tier4_kg_escalation.py -- 2026-08-20 experiment: an 8B-model escalation
path, AIDED by real KG evidence, for entities the 3-model 3B ensemble could
not resolve (TIER_4_ENSEMBLE_SPLIT/ensemble_split, TIER_5_TRUE_AMBIGUITY/
verdict_none_correct, TIER_5_TRUE_AMBIGUITY/below_similarity_floor, low-
confidence unanimous SUPPORTED_1, TIER_2_AUTO_RESOLVED-pending-revalidation).

DESIGN, per explicit 2026-08-20 direction: KG facts are EVIDENCE the model
weighs alongside the entity's own context, never a deterministic override
that bypasses the model call. Earlier draft of this module had a
"apply_deterministic_kg_rule() short-circuits and skips the LLM entirely"
path -- removed. Every entity in scope gets exactly one 8B-model call; the
KG grounding changes what's IN that call's prompt, not whether the call
happens.

KG search is by BOTH SNOMED code and name, not just a narrow direct-
relationship-id lookup: for each candidate, pulls its own SNOMED
concept_code, ALL relationship types linking it to any other candidate
(not just the "Asso morph of" pattern), its closest standard ancestor, and
a same-name cross-check (does any OTHER concept in the vocabulary share
this exact name, and if so what domain/class is it -- surfaces a
duplicate-concept situation as a fact for the model to weigh, not a
pre-decided answer).
"""
from src.llm_client import LLMUnavailable, parse_json_response


def _kg_relationships_between(conn, id_a: int, id_b: int) -> list:
    """ALL SNOMED relationship types between two concepts, either direction
    -- not filtered to any specific pattern. Empty list if none exists."""
    try:
        rows = conn.execute("""
            SELECT relationship_id FROM athena_concept_relationship
            WHERE ((concept_id_1 = ? AND concept_id_2 = ?)
                OR (concept_id_1 = ? AND concept_id_2 = ?))
            AND invalid_reason IS NULL
        """, [id_a, id_b, id_b, id_a]).fetchall()
        return sorted({r[0] for r in rows})
    except Exception:
        return []


def _kg_closest_ancestor(conn, concept_id: int):
    """Closest standard ancestor concept (real parent-of relationship, not a
    guess) -- gives the model a genuine hierarchy anchor."""
    try:
        row = conn.execute("""
            SELECT c.concept_name, a.min_levels_of_separation
            FROM athena_concept_ancestor a
            JOIN athena_concept c ON c.concept_id = a.ancestor_concept_id
            WHERE a.descendant_concept_id = ? AND a.min_levels_of_separation > 0
            AND c.standard_concept = 'S'
            ORDER BY a.min_levels_of_separation ASC LIMIT 1
        """, [concept_id]).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _kg_concept_code(conn, concept_id: int):
    try:
        row = conn.execute(
            "SELECT concept_code, vocabulary_id FROM athena_concept WHERE concept_id = ?",
            [concept_id]).fetchone()
        return (row[0], row[1]) if row else (None, None)
    except Exception:
        return (None, None)


def _kg_name_collision_check(conn, concept_id: int, concept_name: str):
    """Searches the vocabulary BY NAME (not just this candidate's own
    concept_id neighborhood) for any OTHER standard concept sharing the
    identical name -- surfaces a real duplicate-concept situation as a fact
    ('this name also exists under concept_id X, domain Y') rather than the
    model only ever seeing the single candidate it was handed. This is the
    'search KG via snomed code and name' requirement: code-based lookups
    (relationship/ancestor above) alone would miss a duplicate that has NO
    direct graph edge to this candidate at all."""
    try:
        rows = conn.execute("""
            SELECT concept_id, domain_id, concept_class_id, concept_code
            FROM athena_concept
            WHERE lower(concept_name) = lower(?) AND standard_concept = 'S'
            AND concept_id != ?
            LIMIT 5
        """, [concept_name, concept_id]).fetchall()
        return [{"concept_id": r[0], "domain_id": r[1], "concept_class_id": r[2],
                "concept_code": r[3]} for r in rows]
    except Exception:
        return []


def build_kg_context(conn, candidates: list) -> list:
    """One KG-evidence dict per candidate: own SNOMED code/vocab, closest
    ancestor, relationships to every OTHER candidate in this same set, and
    any name-collision siblings found elsewhere in the vocabulary. Pure
    data -- no verdict, no ranking, just facts for the prompt builder and
    the human reviewing this run to both see identically."""
    context = []
    for i, c in enumerate(candidates):
        cid = c.get("omop_concept_id")
        code, vocab = _kg_concept_code(conn, cid)
        ancestor = _kg_closest_ancestor(conn, cid)
        rels_to_others = {}
        for j, other in enumerate(candidates):
            if i == j:
                continue
            rels = _kg_relationships_between(conn, cid, other.get("omop_concept_id"))
            if rels:
                rels_to_others[j + 1] = rels
        collisions = _kg_name_collision_check(conn, cid, c.get("concept_name") or "")
        context.append({
            "index": i + 1,
            "concept_id": cid,
            "concept_code": code,
            "vocabulary": vocab,
            "closest_ancestor": ancestor,
            "relationships_to_other_candidates": rels_to_others,
            "name_collisions_elsewhere": collisions,
        })
    return context


# 2026-08-20, added per explicit direction: a single real, gold-verified
# worked example, not a synthetic/made-up one -- the exact wound-dehiscence
# case investigated earlier this session (note 13538696-DS-11), with its
# REAL KG evidence (confirmed live via athena_concept_relationship) and
# gold's REAL answer (410723003, the Observation/Morph-Abnormality reading).
# Shows the model the expected reasoning shape: cite the specific evidence
# used, don't just restate the candidate name back.
_WORKED_EXAMPLE = """EXAMPLE (for format and reasoning style only -- the actual case below is different):

ENTITY TEXT AS WRITTEN: 'wound dehiscence'
SECTION: Discharge Diagnosis
CONTEXT: ...Facility:\\n___\\n \\nDischarge Diagnosis:\\nwound dehiscence...
ASSERTION: PRESENT

CANDIDATES (with knowledge-graph evidence for each):

[1] name: Wound dehiscence
    domain: Condition
    SNOMED code: 225553008 (SNOMED)
    closest parent concept: Disorder of skin
    direct KG relationship to: candidate [2] via ['Asso morph of']
    direct KG relationship to other candidates: none found
    NOTE: this exact name also exists as a separate concept (id=4253777, domain=Observation, class=Morph Abnormality) -- SNOMED sometimes files the same clinical idea twice under different domain/class framings; this is informational, weigh it against the note's own context.

[2] name: Wound dehiscence
    domain: Observation
    SNOMED code: 410723003 (SNOMED)
    closest parent concept: Morphologically abnormal structure
    direct KG relationship to: candidate [1] via ['Asso morph of']

CORRECT ANSWER: {"best_index": "2", "reasoning": "Both candidates are the identical clinical finding (confirmed by the direct 'Asso morph of' KG relationship linking them), but this note documents wound dehiscence as an observed finding under 'Discharge Diagnosis', not as a distinct disorder being separately diagnosed -- SNOMED's Observation/Morphologic-Abnormality framing (candidate 2) is the standard reading for a documented finding of this kind, and this corpus's own convention favors it in this exact duplicate pattern."}

Note how the reasoning cites the SPECIFIC KG evidence (the relationship linking the two candidates, the domain/class distinction) and the entity's own documentation context together -- not just one or the other.
"""

_ESCALATION_SYSTEM_PROMPT = (
    "You are a senior clinical terminology reviewer resolving a case a "
    "smaller model ensemble could not agree on. You are given real "
    "knowledge-graph facts about each candidate concept -- its SNOMED code, "
    "closest parent concept, any direct graph relationship to the other "
    "candidates, and whether its exact name also exists elsewhere in the "
    "vocabulary under a different concept. Treat these as EVIDENCE to weigh "
    "alongside the entity's own text and clinical context -- they are "
    "supporting facts, not a predetermined answer. Your own clinical "
    "judgment of what the entity actually means still governs the final "
    "choice.\n\n" + _WORKED_EXAMPLE
)


def _format_kg_evidence(kg_ctx: dict) -> str:
    lines = [f"    SNOMED code: {kg_ctx['concept_code'] or 'unknown'} ({kg_ctx['vocabulary'] or 'unknown'})"]
    lines.append(f"    closest parent concept: {kg_ctx['closest_ancestor'] or 'unknown'}")
    if kg_ctx["relationships_to_other_candidates"]:
        rel_str = "; ".join(f"candidate [{idx}] via {rels}"
                            for idx, rels in kg_ctx["relationships_to_other_candidates"].items())
        lines.append(f"    direct KG relationship to: {rel_str}")
    else:
        lines.append("    direct KG relationship to other candidates: none found")
    if kg_ctx["name_collisions_elsewhere"]:
        coll = kg_ctx["name_collisions_elsewhere"][0]
        lines.append(f"    NOTE: this exact name also exists as a separate concept "
                     f"(id={coll['concept_id']}, domain={coll['domain_id']}, "
                     f"class={coll['concept_class_id']}) -- SNOMED sometimes files the "
                     f"same clinical idea twice under different domain/class framings; "
                     f"this is informational, weigh it against the note's own context.")
    return "\n".join(lines)


def build_escalation_prompt(entity: dict, candidates: list, kg_context: list) -> str:
    blocks = []
    for c, kg in zip(candidates, kg_context):
        blocks.append(
            f"[{kg['index']}] name: {c.get('concept_name')}\n"
            f"    domain: {c.get('domain_id')}\n"
            f"{_format_kg_evidence(kg)}"
        )
    indices = [kg["index"] for kg in kg_context]
    return (
        f"ENTITY TEXT AS WRITTEN: {entity.get('original_text')!r}\n"
        f"SECTION: {entity.get('section_name') or 'unknown'}\n"
        f"CONTEXT: ...{entity.get('local_context') or ''}...\n"
        f"ASSERTION: {entity.get('assertion_status', 'PRESENT')}\n\n"
        f"CANDIDATES (with knowledge-graph evidence for each):\n\n"
        + "\n\n".join(blocks) + "\n\n"
        f"Pick the single best-matching candidate using the entity's own "
        f"context AND the knowledge-graph evidence above -- the KG facts "
        f"support your reasoning, they do not replace it. Valid indices: {indices}.\n"
        'Reply with JSON: {"best_index": "<one of the valid indices, as a '
        'string>", "reasoning": "<one sentence citing the specific evidence '
        'used>"}'
    )


def _escalation_schema(indices: list) -> dict:
    return {
        "type": "object",
        "properties": {
            "best_index": {"type": "string", "enum": [str(i) for i in indices]},
            "reasoning": {"type": "string"},
        },
        "required": ["best_index", "reasoning"],
    }


def escalate_to_8b(client, conn, entity: dict, candidates: list) -> dict:
    """One KG-grounded call to the 8B model. Returns
    {"index": int|None, "reasoning": str|None, "prompt": str, "kg_context": list,
     "error": str|None} -- always includes the full prompt and KG context used,
    for transparency/audit, regardless of success or failure."""
    kg_context = build_kg_context(conn, candidates)
    prompt = build_escalation_prompt(entity, candidates, kg_context)
    indices = [kg["index"] for kg in kg_context]
    try:
        raw = client.complete(_ESCALATION_SYSTEM_PROMPT, prompt, schema=_escalation_schema(indices))
        parsed = parse_json_response(raw["text"])
        idx = int(parsed.get("best_index"))
        if idx not in indices:
            return {"index": None, "reasoning": parsed.get("reasoning"), "prompt": prompt,
                    "kg_context": kg_context, "error": "invalid_index"}
        return {"index": idx, "reasoning": parsed.get("reasoning"), "prompt": prompt,
                "kg_context": kg_context, "error": None}
    except (LLMUnavailable, ValueError, TypeError, KeyError) as exc:
        return {"index": None, "reasoning": None, "prompt": prompt, "kg_context": kg_context,
                "error": f"{type(exc).__name__}: {exc}"}
