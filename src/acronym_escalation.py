"""
src/acronym_escalation.py -- Pass 1: MoLLM acronym escalation for
already-flagged ambiguous abbreviations.

TARGETS entities with expansion_ambiguous=TRUE (src/preprocessing.py's
expand_text_and_track_offsets() already computed and stored their
candidate_expansions list -- this module resolves ambiguity, it does not
discover it). See docs/2026-08-16_Shadow_Run_Precision_At_Scale.md and the
project plan's Phase 4 section for the full design rationale: local Ollama
(not an external API), domain classification via
src/preprocessing.py's _omop_domain_for_meaning() rather than an LLM guess,
and interception in src/normalization/orchestrator.py's
process_and_normalize_entities() (reads `mollm_resolved_expansion` off each
entity dict, the same upstream-attaches-a-field pattern already used for
assertion_status/is_allergy_context -- and critically feeds a search-only
variable there, never the stored expanded_text column, per the duplicate-row
bug found and fixed while building step 1).

BUILD-ORDER STEP 2 (current state): resolve_ambiguous_acronyms() now makes a
real local-Ollama call per ambiguous entity, replacing step 1's
MOCK_RESOLUTIONS fixture. ONE CALL PER ENTITY, not batched across a note's
several ambiguous entities in one completion, despite
src/mollm_wholenote_ensemble.py's chunk-of-15 pattern being the nearer
analog in spirit (full note context, one call covers several entities).
Deliberate deviation: that pattern works because every array item there
shares ONE fixed assessment enum (CORRECT/INCORRECT/.../UNCERTAIN,
mollm_wholenote_ensemble.SYSTEM_PROMPT's ASSESSMENT_VALUES). Here every
entity has a DIFFERENT enum -- its own candidate_expansions -- and JSON
Schema cannot express a different enum per array item under one guided-
decoding schema. Per-entity single-object calls (the same shape
src/mollm_tier_gate.py's Step A/B already use, proven in production) keep
the "hallucinated expansion structurally impossible" guarantee the plan
calls for; batching efficiency is what build-order step 3's acronym_priors
cache is actually for; not this.

EMPIRICAL FINDINGS FROM BUILDING THIS STEP (2026-08-16):
- Found and fixed a real prompt-quality bug before measuring anything:
  the first version's local context came from ent.get("local_context"),
  which src/preprocessing.py computes from Stage 1's EXPANDED text. For an
  ambiguous entity that means it already spells out Stage 1's OWN guess as
  if it were the note's literal wording (e.g. "...midLAD 100%, patent
  ductus arteriosus 80%..." instead of the note's actual "...midLAD 100%,
  PDA 80%..."), which is circular -- it looks like corroborating evidence
  for the very guess being reconsidered. Replaced with raw_local_context(),
  sliced directly from the unexpanded note around the entity's own offsets.
  See that function's own docstring for the full story.
- Measured 15/15 on a broader, non-cherry-picked sample (first 15
  expansion_ambiguous=TRUE rows in note 10000032-DS-21) -- all resolved to
  contextually plausible expansions (ABD->abdomen not albumin-binding
  domain, ED->emergency department not Ectodermal Dysplasia, LAD->left
  anterior descending artery not a dermatology term, CN II-XII->cranial
  nerve, etc.).
- But NOT a universal fix: note 11134545-DS-21's "PDA" (patent ductus
  arteriosus vs. posterior descending artery, in a 3-vessel coronary artery
  disease stenosis-percentage list -- LAD/PDA/RCA) resolved WRONG across
  all three available models, even after the local_context fix, even with
  an explicit anti-prior-bias instruction sentence added to the prompt.
  qwen2.5:3b's reasoning under the anti-bias prompt literally said the
  coronary-stenosis context "rules out congenital heart defects" and then
  still output "patent ductus arteriosus" as chosen_expansion anyway -- a
  genuine reasoning/verdict mismatch, the same failure class
  src/mollm_ensemble.py's reasoning_verdict_mismatch() already exists to
  catch for the Tier 1-5 gate, not yet guarded against here. A real,
  documented model-capability limit (a 3B model's prior toward the more
  textbook-famous expansion overriding correct in-context reasoning), not
  a wiring bug -- logged honestly rather than papered over by only
  reporting the 15/15 success.

Still no cache (step 3) and not yet wired into run_pipeline() (step 4).
"""

from src.llm_client import LLMUnavailable, build_clients, parse_json_response
from src.preprocessing import _omop_domain_for_meaning

# Single model, not a 3-model ensemble like the Tier 1-5 gate -- this is a
# narrow, enum-constrained multiple-choice pick (which of N known dictionary
# meanings fits this context), not a high-stakes unreviewed-write decision.
# qwen2.5:3b chosen as the fastest of the three models measured this session
# (docs/2026-08-16_Shadow_Run_Precision_At_Scale.md's Stage 3 timing data).
ESCALATION_MODEL = "qwen2.5:3b"

SYSTEM_PROMPT = (
    "You are a clinical terminology expert reading a clinical note. You are "
    "shown the note's raw text and ONE abbreviation from it that has "
    "multiple possible expansions in a controlled clinical dictionary. Using "
    "ONLY the note's own context, pick which ONE of the given candidate "
    "expansions is the correct reading for THIS SPECIFIC mention -- do not "
    "invent an expansion that is not in the candidate list."
)


def build_escalation_prompt(raw_text: str, abbreviation: str, candidate_expansions: list,
                            local_context: str = None, section_name: str = None) -> str:
    lines = [
        "FULL CLINICAL NOTE:",
        "-" * 40,
        raw_text or "",
        "-" * 40,
        "",
        f"ABBREVIATION TO RESOLVE: {abbreviation!r}",
        f"SECTION: {section_name or 'unknown'}",
        f"LOCAL CONTEXT: ...{local_context or ''}...",
        "",
        "CANDIDATE EXPANSIONS (pick exactly one, copy it EXACTLY):",
    ]
    for m in candidate_expansions:
        lines.append(f"  - {m}")
    lines.append("")
    lines.append(
        'Reply with JSON: {"chosen_expansion": "<one candidate above, copied '
        'exactly>", "reasoning": "<one sentence citing what in the note '
        'supports this reading>"}')
    return "\n".join(lines)


def build_escalation_schema(candidate_expansions: list) -> dict:
    """enum-constrained to THIS entity's own candidate_expansions -- same
    "hallucination structurally impossible, not just detectable" discipline
    every other guided-decoding schema in this codebase already uses (see
    e.g. src/mollm_wholenote_ensemble.py's build_chunk_schema() docstring)."""
    return {
        "type": "object",
        "properties": {
            "chosen_expansion": {"type": "string", "enum": list(candidate_expansions)},
            "reasoning": {"type": "string"},
        },
        "required": ["chosen_expansion", "reasoning"],
    }


LOCAL_CONTEXT_WINDOW = 300  # characters each side of the abbreviation's own raw-text span


def raw_local_context(raw_text: str, orig_start: int, orig_end: int,
                      window: int = LOCAL_CONTEXT_WINDOW) -> str:
    """A context window sliced directly from raw_text (the UNEXPANDED note)
    around the entity's own character offsets -- NOT
    extracted_entities.local_context.

    Why this matters: that stored column is computed from Stage 1's
    EXPANDED text (src/preprocessing.py's process_and_store_note() ->
    _local_context_window() runs after expand_text_and_track_offsets()).
    For an ambiguous entity, that means it already shows Stage 1's OWN
    alphabetical-default guess spelled out as if it were the note's literal
    wording (e.g. "...midLAD 100%, patent ductus arteriosus 80% diffusely
    diseased..." instead of the note's actual "...midLAD 100%, PDA 80%
    diffusely diseased..."). Feeding that into an escalation prompt asking
    the model to independently judge the abbreviation is circular -- it
    looks like corroborating textual evidence for the very guess being
    reconsidered. Confirmed empirically this session
    (docs/2026-08-16_Shadow_Run_Precision_At_Scale.md): even after fixing
    this, the affected case (note 11134545-DS-21's "PDA") still resolved
    wrong across all three models -- a separate, real model-capability
    limitation (strong prior toward the more textbook-famous expansion,
    even reasoning correctly then naming the WRONG one in `chosen_expansion`
    anyway) -- but this fix is still correct and necessary regardless: the
    model must see the note's actual wording, not Stage 1's own guess
    presented as fact.
    """
    if raw_text is None or orig_start is None or orig_end is None:
        return None
    start = max(0, orig_start - window)
    end = min(len(raw_text), orig_end + window)
    return raw_text[start:end]


def escalate_one_entity(client, raw_text: str, ent: dict) -> dict:
    """Runs one guided-decoding call for one ambiguous entity. Returns
    {"chosen_expansion": str, "reasoning": str} on success, or None on any
    failure (model unavailable, unparseable response, or -- belt and braces,
    since guided decoding is not blindly trusted anywhere else in this
    codebase either -- a response outside the given candidate list).
    Never raises: a failed escalation should fall through to today's Stage 1
    alphabetical-default behavior unchanged, not abort a batch.
    """
    candidate_expansions = ent.get("candidate_expansions") or []
    if len(candidate_expansions) < 2:
        return None  # nothing to disambiguate

    local_context = raw_local_context(raw_text, ent.get("orig_start"), ent.get("orig_end"))
    prompt = build_escalation_prompt(
        raw_text, ent.get("original_text"), candidate_expansions,
        local_context=local_context, section_name=ent.get("section_name"))
    schema = build_escalation_schema(candidate_expansions)

    try:
        raw = client.complete(SYSTEM_PROMPT, prompt, schema=schema)
        parsed = parse_json_response(raw["text"])
    except (LLMUnavailable, ValueError):
        return None

    chosen = parsed.get("chosen_expansion")
    if chosen not in candidate_expansions:
        return None

    return {"chosen_expansion": chosen, "reasoning": parsed.get("reasoning")}


def resolve_ambiguous_acronyms(entities: list, raw_text: str, note_id: str, conn,
                               client=None) -> dict:
    """Returns {entity_id: {"expansion": str, "omop_domain": str|None,
    "source": "mollm"}} for every expansion_ambiguous=TRUE entity in
    `entities` that a real MoLLM call successfully resolved. An entity that
    is not ambiguous, has fewer than 2 candidates, or whose escalation call
    fails for any reason is simply absent from the returned dict -- it falls
    through to today's Stage 1 alphabetical-default expansion unchanged,
    same fallback contract as build-order step 1's mock had.

    `client`: an already-built src.llm_client.LLMClient (reuse one across a
    batch rather than reconnecting per note/entity). None (default) builds
    ESCALATION_MODEL's own client for this call only -- convenient for a
    single ad hoc call, wasteful for a real batch; batch callers (run_pipeline()
    once step 4 wires this in) should build once and pass it in.
    """
    resolved = {}
    own_client = client is None
    if own_client:
        clients = build_clients()
        client = clients.get(ESCALATION_MODEL)
        if client is None:
            return resolved  # ESCALATION_MODEL not available in this environment

    for ent in entities:
        if not ent.get("expansion_ambiguous"):
            continue
        entity_id = ent.get("entity_id")
        if entity_id is None:
            continue

        result = escalate_one_entity(client, raw_text, ent)
        if result is None:
            continue

        chosen = result["chosen_expansion"]
        omop_domain = _omop_domain_for_meaning(conn, chosen) if conn is not None else None
        resolved[entity_id] = {
            "expansion": chosen,
            "omop_domain": omop_domain,
            "source": "mollm",
        }

    return resolved
