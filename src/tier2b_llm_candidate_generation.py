"""src/tier2b_llm_candidate_generation.py -- 2026-08-20 experiment: use an
8B model to GENERATE a candidate clinical concept name (AND independently
judge its concept type) from its own knowledge, then VERIFY that name
against the real vocabulary before it's ever trusted -- a hallucinated
suggestion can never enter the candidate pool, only a real,
vocabulary-confirmed concept can.

Built after the Stage 3 escalation smoke test (src.tier4_kg_escalation)
showed the real bottleneck for many "hard cases" isn't Stage 3 reasoning --
it's that Stage 2b's own Tier 1-3 retrieval sometimes hands over exactly
ONE (wrong) candidate, or a candidate pool that never contained the
correct concept at all.

Two-step, generate-then-verify:

  1. GENERATE (LLM). Given the entity's raw text and local context, asks
     the model for the plain-language standard clinical name AND its own
     independent judgment of what TYPE of clinical concept this is
     (Condition/Medication/Procedure/Lab Test/Anatomy/Symptom) -- NOT
     required to agree with GLiNER's own Stage-1 label. 2026-08-20 fix:
     the first version of this module trusted the Stage-1 label
     unconditionally for the verification domain, which meant a
     mislabeled entity (confirmed live: 'heart' extracted as Lab Test)
     anchored the model's own generation toward that wrong framing and
     then verification could never even SEE a Condition-domain concept
     to confirm it against, regardless of what the model actually
     suspected. The model now states its own type judgment, and
     verification searches that domain -- decoupled from Stage 1's guess
     rather than constrained by it.

  2. VERIFY (KG/vocabulary). Searches athena_concept/athena_concept_synonym
     for the generated name, using the domain implied by the MODEL's own
     type judgment (falling back to GLiNER's label only if the model's
     judgment is missing/unparseable) -- exact/synonym match first (the
     same _lookup_tier12 machinery Stage 2b's own lab-suffix fallback
     already uses), then a SapBERT semantic search as a fallback. ONLY a
     real, found concept_id is ever added to the candidate pool.

  3. KG ENRICHMENT (audit/evidence, not a gate). Once a real concept_id is
     verified, reuses src.tier4_kg_escalation's KG-lookup tools (closest
     ancestor, relationships to any of the entity's EXISTING candidates,
     name-collision check) to attach real graph evidence to the result --
     the same "KG rules aid the LLM, never a bypass" discipline established
     for the Stage 3 escalation module, applied here for reporting/audit
     rather than a second model call (kept to one LLM call per entity for
     cost, since this is the generation step's own confidence check, not
     a comparison needing a second opinion).

The new candidate is ADDED to the existing pool (tagged match_basis=
"llm_generated_verified"), never a silent replacement.
"""
from src.llm_client import LLMUnavailable, parse_json_response
from src.normalization.compound_span import _lookup_tier12
from src.normalization.constants import (
    GLINER_LABEL_TO_DOMAIN, VOCAB_BY_LABEL, DEFAULT_VOCAB, TIER3_SIMILARITY_FLOOR)
from src.normalization.sapbert_model import get_sapbert_embedding
from src.normalization.tier_retrieval import _tier3_semantic_rows, _candidate
from src.tier4_kg_escalation import (
    _kg_closest_ancestor, _kg_relationships_between, _kg_name_collision_check)

_VALID_TYPES = ["Condition", "Medication", "Procedure", "Lab Test", "Anatomy", "Symptom"]

# 2026-08-20, added per explicit direction (same discipline as the Stage 3
# escalation module's worked example): a single REAL, gold-verified case,
# not synthetic. This is the 'WBC-8' entity investigated earlier this
# session -- Stage 2b's own retrieval had force-included only "CD18 count"
# (an unrelated immunology test) as its sole candidate; this exact
# generate-then-verify call recovered "White blood cell count"
# (concept_id 4298431), confirmed live to be the EXACT gold-correct answer
# (SNOMED 767002).
_WORKED_EXAMPLE = """EXAMPLE (for format and reasoning style only -- the actual case below is different):

TEXT AS WRITTEN: 'WBC-8'
AUTOMATED SYSTEM'S TYPE GUESS (may be wrong): Lab Test
SECTION: ADMISSION LABS
CONTEXT: ...WBC-8.7 RBC-3.25* HGB-9.8* HCT-28.1* MCV-87\\nMCH-30.2 MCHC-34.9 RDW-15.2...

CORRECT ANSWER: {"standard_name": "White blood cell count", "concept_type": "Lab Test", "reasoning": "'WBC' is the standard flowsheet abbreviation for white blood cell count, and the surrounding CBC panel context (RBC, HGB, HCT, MCV -- all other blood count measurements) confirms this reading; the automated type guess (Lab Test) is correct here."}

(This one confirmed the automated guess was right -- but you should disagree with it just as readily when the content indicates otherwise, as in the type-independence instruction below.)
"""

_GENERATE_SYSTEM_PROMPT = (
    "You are a senior clinical terminology reviewer. Given a short span of "
    "text from a clinical note and its surrounding context, state the "
    "standard clinical name of what this term actually refers to -- the "
    "way it would be named in a formal medical terminology (SNOMED CT), "
    "not a paraphrase of the raw text. If the text is an abbreviation, "
    "expand it to its most likely meaning given the context. Be specific: "
    "for a lab test, name the actual measurement (e.g. 'White blood cell "
    "count', not 'blood test'); for a condition, name the specific "
    "diagnosis, not a general category.\n\n"
    "Also state what TYPE of clinical concept this actually is. An "
    "automated system already guessed a type when it first found this "
    "span (shown below) -- but that guess can be wrong, especially for "
    "abbreviations or short spans lifted out of context. Judge the type "
    "independently from the entity's own meaning and the surrounding "
    "text; do not simply repeat the automated guess if the content "
    "suggests otherwise (e.g. a word extracted as a lab test that is "
    "actually describing a diagnosis should be typed as a Condition, not "
    "Lab Test).\n\n" + _WORKED_EXAMPLE
)


def _generate_prompt(entity: dict) -> str:
    return (
        f"TEXT AS WRITTEN: {entity.get('original_text')!r}\n"
        f"AUTOMATED SYSTEM'S TYPE GUESS (may be wrong): {entity.get('gliner_label')}\n"
        f"SECTION: {entity.get('section_name') or 'unknown'}\n"
        f"CONTEXT: ...{entity.get('local_context') or ''}...\n\n"
        'Reply with JSON: {"standard_name": "<the standard clinical/SNOMED '
        'name for this>", "concept_type": "<your own independent judgment, '
        f'one of {_VALID_TYPES}>", "reasoning": "<one sentence, note '
        'explicitly if you disagree with the automated type guess>"}'
    )


def _generate_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "standard_name": {"type": "string"},
            "concept_type": {"type": "string", "enum": _VALID_TYPES},
            "reasoning": {"type": "string"},
        },
        "required": ["standard_name", "concept_type", "reasoning"],
    }


def generate_candidate_name(client, entity: dict) -> dict:
    """One LLM call, no vocabulary access -- returns
    {"standard_name": str|None, "concept_type": str|None, "reasoning": str|None,
     "error": str|None}."""
    try:
        raw = client.complete(_GENERATE_SYSTEM_PROMPT, _generate_prompt(entity),
                              schema=_generate_schema())
        parsed = parse_json_response(raw["text"])
        name = (parsed.get("standard_name") or "").strip()
        if not name:
            return {"standard_name": None, "concept_type": None, "reasoning": None,
                    "error": "empty_generation"}
        ctype = parsed.get("concept_type")
        if ctype not in _VALID_TYPES:
            ctype = None
        return {"standard_name": name, "concept_type": ctype,
                "reasoning": parsed.get("reasoning"), "error": None}
    except (LLMUnavailable, ValueError, TypeError, KeyError) as exc:
        return {"standard_name": None, "concept_type": None, "reasoning": None,
                "error": f"{type(exc).__name__}: {exc}"}


def verify_against_vocabulary(conn, generated_name: str, concept_type: str) -> dict:
    """Real vocabulary lookup for `generated_name`, searching the domain
    implied by `concept_type` (the model's OWN judgment -- see module
    docstring for why this is no longer tied to GLiNER's Stage-1 label).
    Exact/synonym first, SapBERT semantic fallback second. Returns
    {"concept_id": int|None, "concept_name": str|None, "domain_id": str|None,
     "vocabulary_id": str|None, "match_tier": str|None, "similarity_score": float|None}
    -- concept_id is None if nothing verifiable was found."""
    vocabs = VOCAB_BY_LABEL.get(concept_type, DEFAULT_VOCAB)
    domains = GLINER_LABEL_TO_DOMAIN.get(concept_type)

    exact = _lookup_tier12(conn, generated_name, vocabs, domains=domains, gliner_label=concept_type)
    if exact:
        return {"concept_id": exact["omop_concept_id"], "concept_name": exact["concept_name"],
                "domain_id": exact["domain_id"], "vocabulary_id": exact.get("vocabulary_id"),
                "match_tier": exact["match_tier"], "similarity_score": 1.0}

    vector = get_sapbert_embedding(generated_name)
    rows = _tier3_semantic_rows(conn, vector, vocabs, domains)
    if rows and rows[0][4] >= TIER3_SIMILARITY_FLOOR:
        r = rows[0]
        return {"concept_id": r[0], "concept_name": r[1], "domain_id": r[2],
                "vocabulary_id": r[3], "match_tier": "3 (Semantic)",
                "similarity_score": round(r[4], 4)}

    return {"concept_id": None, "concept_name": None, "domain_id": None,
            "vocabulary_id": None, "match_tier": None, "similarity_score": None}


def _kg_enrichment(conn, concept_id: int, concept_name: str, existing_candidates: list) -> dict:
    """Real KG evidence for the newly-verified concept -- reuses
    src.tier4_kg_escalation's tools (same functions the Stage 3 escalation
    module uses), applied here as audit/evidence attached to the result,
    not a gate on whether the candidate gets added."""
    rels_to_existing = {}
    for i, c in enumerate(existing_candidates, 1):
        rels = _kg_relationships_between(conn, concept_id, c.get("omop_concept_id"))
        if rels:
            rels_to_existing[i] = rels
    return {
        "closest_ancestor": _kg_closest_ancestor(conn, concept_id),
        "relationships_to_existing_candidates": rels_to_existing,
        "name_collisions_elsewhere": _kg_name_collision_check(conn, concept_id, concept_name or ""),
    }


def augment_candidates_with_llm(client, conn, entity: dict, existing_candidates: list) -> dict:
    """Top-level entry point. Returns
    {"generated": dict, "verified": dict, "kg_evidence": dict|None,
     "new_candidate": dict|None, "already_present": bool} -- new_candidate is
     a full _candidate()-shaped dict ready to prepend/append to
     entity["candidates"], or None if generation failed or verification
     found nothing real."""
    generated = generate_candidate_name(client, entity)
    if not generated["standard_name"]:
        return {"generated": generated, "verified": None, "kg_evidence": None,
                "new_candidate": None, "already_present": False}

    # Model's own type judgment first; only fall back to GLiNER's original
    # label if the model didn't give a usable one.
    concept_type = generated["concept_type"] or entity.get("gliner_label")
    verified = verify_against_vocabulary(conn, generated["standard_name"], concept_type)

    if verified["concept_id"] is None:
        return {"generated": generated, "verified": verified, "kg_evidence": None,
                "new_candidate": None, "already_present": False}

    kg_evidence = _kg_enrichment(conn, verified["concept_id"], verified["concept_name"], existing_candidates)

    existing_ids = {c.get("omop_concept_id") for c in existing_candidates}
    already_present = verified["concept_id"] in existing_ids
    new_candidate = None
    if not already_present:
        row = (verified["concept_id"], verified["concept_name"], verified["domain_id"], verified["vocabulary_id"])
        new_candidate = _candidate(row, verified["match_tier"], verified["similarity_score"],
                                   match_basis="llm_generated_verified")
    return {"generated": generated, "verified": verified, "kg_evidence": kg_evidence,
            "new_candidate": new_candidate, "already_present": already_present}
