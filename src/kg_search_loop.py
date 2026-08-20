"""src/kg_search_loop.py -- 2026-08-20: a genuine multi-round KG SEARCH loop
for TIER_4_ENSEMBLE_SPLIT / TIER_5_TRUE_AMBIGUITY entities, built per
explicit direction ("for tier 4 and 5 create a loop where mollm search the
KG rules and find relevant rules to improve their decisions").

WHY A LOOP, NOT ANOTHER ONE-SHOT PRE-FETCH. src.tier4_kg_escalation's
build_kg_context() already pre-fetches ALL evidence for ALL candidates in
one pass -- fast and complete, but it decides what's relevant FOR the
model rather than letting the model decide what it actually needs, and
dumps every candidate's full evidence block into one prompt regardless of
whether it's useful for THIS entity. This module inverts that: the model
sees only the entity and each candidate's name/domain first, then
explicitly requests whichever searches it wants (relationships between
specific candidates, a candidate's closest ancestor, a name-collision
check, or an open-ended text search against the vocabulary for a concept
NOT already in the candidate pool) before committing to a verdict.
Bounded at MAX_SEARCH_ROUNDS rounds so cost stays predictable; the final
round forces a verdict with no further searches allowed.

TOOL ACCESS SCOPING NOTE (same as src.tier4_arbiter_8b and
src.tier4_kg_escalation): src.llm_client.LLMClient supports single-shot
structured completions, not an interactive tool-calling API. "The model
searches the KG" is implemented here as a genuine multi-turn PROMPT loop
-- each round is its own complete LLM call, with the previous round's
search results appended as prompt context for the next -- rather than a
mid-generation function-call/tool-result exchange within one call. This
is a real difference from OpenAI-style function calling, flagged
explicitly rather than silently: the model cannot inspect a search
result and immediately continue the SAME reasoning chain, only start a
fresh one that includes the result as given context.

REUSES, DOES NOT DUPLICATE: _kg_relationships_between/_kg_closest_ancestor/
_kg_name_collision_check are imported directly from
src.tier4_kg_escalation, the same functions that module's one-shot
pre-fetch uses -- only the ORCHESTRATION (what gets called, when, and how
results are fed back) is new here, not the underlying KG queries.

VERDICT (2026-08-20, smoke-tested against 9 real TIER_4_ENSEMBLE_SPLIT/
TIER_5_TRUE_AMBIGUITY entities, scripts/smoke_test_kg_search_loop.py):
NOT ADOPTED, NOT WIRED IN, and no further batch validation is planned.
A real bug was found and fixed live during the smoke test (the first
schema let the model answer action="search" with zero queries attached
even on the forced-final round, producing an unusable None verdict every
time) -- but even after that fix, 7 of 9 sampled entities still declined
to commit to a verdict at all ("I need more evidence, but I'm not sure
what specific search would be helpful"), despite 2-4 search rounds
already having run. The architectural read: letting a 3-8B-class model
drive its own multi-round search strategy asks for planning/executive
function these models don't reliably have -- they can request searches,
but often can't tell when they've gathered enough to decide, unlike
src.tier4_kg_escalation's one-shot pre-fetch-everything design, which
forces a single decision against a complete, static evidence block and
has no "not sure, need more" escape hatch to fall back on. That module
(and specifically src.tier4_arbiter_8b, the strongest of the three
pre-fetch-style architectures at 51.0% precision) remains the reference
design for future escalation-tier work; this ReAct-style loop does not
supersede it. Kept in the repo as a real, working implementation of the
pattern (in case a genuinely more capable model is ever substituted in),
not as active or planned production code.
"""
from src.llm_client import LLMUnavailable, parse_json_response
from src.tier4_kg_escalation import (
    _kg_closest_ancestor, _kg_name_collision_check, _kg_relationships_between)

MAX_SEARCH_ROUNDS = 3

_SYSTEM_PROMPT = (
    "You are a senior clinical terminology reviewer resolving a case where "
    "an ensemble of smaller models could not agree (or found no candidate "
    "acceptable) on the correct SNOMED CT concept for a clinical mention. "
    "You have real, on-demand access to the knowledge graph: you may "
    "request specific facts (how two candidates relate, a candidate's "
    "parent concept, whether another concept elsewhere in the vocabulary "
    "shares a candidate's exact name, or a free-text search for a concept "
    "not currently in your candidate list) before deciding. Use this "
    "access when it would actually change your answer -- don't request "
    "searches you don't need, and don't guess when a real search could "
    "settle the question. KG facts are evidence to weigh, not a rule that "
    "decides for you: a candidate can have a clean KG relationship and "
    "still be the wrong clinical answer for this specific mention."
)


def _kg_text_search(conn, term: str, limit: int = 5) -> list:
    """Open-ended vocabulary search by name, NOT tied to an existing
    candidate -- the one search type build_kg_context() has no equivalent
    for, since it only ever enriches candidates already in the pool. Plain
    substring match (ILIKE), not semantic -- deterministic and cheap,
    appropriate for a model-issued lookup of a specific term it names."""
    try:
        rows = conn.execute("""
            SELECT concept_id, concept_name, domain_id, concept_code
            FROM athena_concept
            WHERE concept_name ILIKE ? AND standard_concept = 'S'
            ORDER BY length(concept_name) ASC LIMIT ?
        """, [f"%{term}%", limit]).fetchall()
        return [{"concept_id": r[0], "concept_name": r[1], "domain_id": r[2],
                "concept_code": r[3]} for r in rows]
    except Exception:
        return []


def dispatch_search(conn, query: dict, candidates: list) -> dict:
    """Executes one model-requested search query. `query` shape:
    {"type": "relationships"|"ancestor"|"name_collision"|"text_search",
     "candidate_index": int (1-based, for relationships/ancestor/name_collision),
     "other_index": int (1-based, for relationships only),
     "term": str (for text_search only)}.
    Returns {"query": query, "result": <search-specific payload>} -- always
    a dict, never raises, so a malformed request degrades to an empty
    result rather than aborting the whole round."""
    qtype = query.get("type")
    idx = query.get("candidate_index")
    cand = candidates[idx - 1] if isinstance(idx, int) and 0 < idx <= len(candidates) else None

    if qtype == "text_search":
        term = (query.get("term") or "").strip()
        return {"query": query, "result": _kg_text_search(conn, term) if term else []}

    if cand is None:
        return {"query": query, "result": None, "error": "invalid or missing candidate_index"}
    cid = cand.get("omop_concept_id")
    if cid is None:
        return {"query": query, "result": None, "error": "candidate has no omop_concept_id"}

    if qtype == "ancestor":
        return {"query": query, "result": _kg_closest_ancestor(conn, cid)}

    if qtype == "name_collision":
        return {"query": query, "result": _kg_name_collision_check(
            conn, cid, cand.get("concept_name") or "")}

    if qtype == "relationships":
        other_idx = query.get("other_index")
        other = candidates[other_idx - 1] if isinstance(other_idx, int) \
            and 0 < other_idx <= len(candidates) else None
        if other is None or other.get("omop_concept_id") is None:
            return {"query": query, "result": None, "error": "invalid other_index"}
        return {"query": query, "result": _kg_relationships_between(conn, cid, other.get("omop_concept_id"))}

    return {"query": query, "result": None, "error": f"unknown search type {qtype!r}"}


def _format_search_result(entry: dict) -> str:
    q, r = entry["query"], entry.get("result")
    qtype = q.get("type")
    if entry.get("error"):
        return f"  [{qtype}] ERROR: {entry['error']}"
    if qtype == "relationships":
        return (f"  [relationships between candidate {q.get('candidate_index')} and "
                f"{q.get('other_index')}]: {r or 'none found'}")
    if qtype == "ancestor":
        return f"  [closest ancestor of candidate {q.get('candidate_index')}]: {r or 'none found'}"
    if qtype == "name_collision":
        return (f"  [other concepts sharing candidate {q.get('candidate_index')}'s exact name]: "
                f"{r or 'none found'}")
    if qtype == "text_search":
        return f"  [vocabulary search for {q.get('term')!r}]: {r or 'no matches'}"
    return f"  [{qtype}]: {r}"


def build_round_prompt(entity: dict, candidates: list, search_history: list,
                       round_num: int, max_rounds: int) -> str:
    cand_blocks = [f"  [{i}] {c.get('concept_name')} (domain: {c.get('domain_id')})"
                   for i, c in enumerate(candidates, 1)]
    evidence_block = ""
    if search_history:
        lines = []
        for round_entries in search_history:
            for entry in round_entries:
                lines.append(_format_search_result(entry))
        evidence_block = "KG SEARCH RESULTS SO FAR:\n" + "\n".join(lines) + "\n\n"

    rounds_left = max_rounds - round_num
    if rounds_left <= 0:
        action_instruction = (
            "You have used all your search rounds. You MUST give a final verdict now "
            '-- reply with {"action": "verdict", "best_index": "<index or null if none '
            'correct>", "reasoning": "<one sentence>"}.')
    else:
        action_instruction = (
            f"You have {rounds_left} more search round(s) available. Reply with JSON, "
            'either: {"action": "search", "search_queries": [{"type": "relationships"|'
            '"ancestor"|"name_collision"|"text_search", "candidate_index": <int, for '
            'relationships/ancestor/name_collision>, "other_index": <int, relationships '
            'only>, "term": "<string, text_search only>"}], "reasoning": "<why you need '
            'this>"} to request more evidence, OR {"action": "verdict", "best_index": '
            '"<index or null if none correct>", "reasoning": "<one sentence>"} to decide now.')

    return (
        f"ENTITY TEXT AS WRITTEN: {entity.get('original_text')!r}\n"
        f"SECTION: {entity.get('section_name') or 'unknown'}\n"
        f"CONTEXT: ...{entity.get('local_context') or ''}...\n"
        f"ASSERTION: {entity.get('assertion_status', 'PRESENT')}\n\n"
        f"CANDIDATES:\n" + "\n".join(cand_blocks) + "\n\n"
        f"{evidence_block}"
        f"{action_instruction}"
    )


def _round_schema(n_candidates: int, is_final: bool = False) -> dict:
    """2026-08-20 fix: the first version allowed action="search" (with an
    empty search_queries list and no best_index) on the FINAL round too --
    confirmed live in the first smoke-test pass that the model actually
    does this ("I'd like to see the relationships..." with zero queries
    attached), producing an unusable None verdict despite the prompt's own
    text instruction to decide now. `is_final=True` removes "search" from
    the action enum entirely and requires best_index, so a final-round
    non-decision is structurally impossible rather than merely
    discouraged."""
    indices = [str(i) for i in range(1, n_candidates + 1)] if n_candidates > 0 else ["1"]

    if is_final:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["verdict"]},
                "best_index": {"type": ["string", "null"], "enum": indices + [None]},
                "reasoning": {"type": "string"},
            },
            "required": ["action", "best_index", "reasoning"],
        }

    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "verdict"]},
            "search_queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string",
                                "enum": ["relationships", "ancestor", "name_collision", "text_search"]},
                        "candidate_index": {"type": "string", "enum": indices},
                        "other_index": {"type": "string", "enum": indices},
                        "term": {"type": "string"},
                    },
                    "required": ["type"],
                },
            },
            "best_index": {"type": ["string", "null"], "enum": indices + [None]},
            "reasoning": {"type": "string"},
        },
        "required": ["action", "reasoning"],
    }


def run_kg_search_loop(client, conn, entity: dict, candidates: list,
                       max_rounds: int = MAX_SEARCH_ROUNDS) -> dict:
    """Runs the multi-round search loop for one entity. Returns
    {"index": int|None, "reasoning": str|None, "n_rounds": int,
    "n_searches": int, "trace": list, "error": str|None} -- trace is a list
    of {"round": int, "prompt": str, "raw_response": dict|None,
    "searches_requested": list, "searches_executed": list} for full audit/
    transparency, win or lose. Never raises -- an LLM failure at any round
    ends the loop with error set and index=None, same discipline as every
    other escalation module this session."""
    search_history = []
    trace = []
    n_searches = 0

    for round_num in range(max_rounds):
        is_final = round_num == max_rounds - 1
        schema = _round_schema(len(candidates), is_final=is_final)
        prompt = build_round_prompt(entity, candidates, search_history, round_num, max_rounds)
        entry = {"round": round_num, "prompt": prompt, "raw_response": None,
                 "searches_requested": [], "searches_executed": []}
        try:
            raw = client.complete(_SYSTEM_PROMPT, prompt, schema=schema)
            parsed = parse_json_response(raw["text"])
        except (LLMUnavailable, ValueError, TypeError, KeyError) as exc:
            trace.append(entry)
            return {"index": None, "reasoning": None, "n_rounds": round_num + 1,
                    "n_searches": n_searches, "trace": trace,
                    "error": f"{type(exc).__name__}: {exc}"}

        entry["raw_response"] = parsed
        action = parsed.get("action")

        if action == "verdict" or round_num == max_rounds - 1:
            best = parsed.get("best_index")
            idx = int(best) if best not in (None, "null", "") else None
            if idx is not None and not (0 < idx <= len(candidates)):
                idx = None
            trace.append(entry)
            return {"index": idx, "reasoning": parsed.get("reasoning"),
                    "n_rounds": round_num + 1, "n_searches": n_searches,
                    "trace": trace, "error": None}

        queries = parsed.get("search_queries") or []
        # Coerce string indices (schema requires strings for enum
        # constraints) back to ints for dispatch_search().
        normalized_queries = []
        for q in queries:
            nq = dict(q)
            for key in ("candidate_index", "other_index"):
                if key in nq and nq[key] is not None:
                    try:
                        nq[key] = int(nq[key])
                    except (TypeError, ValueError):
                        nq[key] = None
            normalized_queries.append(nq)
        entry["searches_requested"] = normalized_queries

        results = [dispatch_search(conn, q, candidates) for q in normalized_queries]
        entry["searches_executed"] = results
        n_searches += len(results)
        search_history.append(results)
        trace.append(entry)

    # Defensive fallback -- the max_rounds-1 branch above should always
    # return before this point, but keep the loop honest if max_rounds<=0.
    return {"index": None, "reasoning": None, "n_rounds": len(trace),
            "n_searches": n_searches, "trace": trace, "error": "loop exited without a verdict"}
