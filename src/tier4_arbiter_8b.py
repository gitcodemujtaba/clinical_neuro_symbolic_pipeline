"""src/tier4_arbiter_8b.py -- 2026-08-20 experiment: an 8B-model ARBITER
placed at the end of the pipeline, after the 3-model 3B MoLLM ensemble has
already voted. Unlike src.tier4_kg_escalation (which re-evaluates candidates
from scratch), this module shows the 8B model the 3B ensemble's OWN
verdicts and reasoning for each model, plus KG/vocabulary evidence, and
asks it to arbitrate the disagreement directly -- targeting TIER_4_
ENSEMBLE_SPLIT specifically (both 1:1:1 and 2:1 split shapes).

Why this is a different bet than the from-scratch escalation module: the
3B models' own reasoning is itself informative -- the wound-dehiscence and
MCH/MCHC investigations earlier this session both showed the SAME
anchoring-on-surface-text-similarity failure mode across all 3 models'
independent reasoning. An arbiter that can SEE that pattern in the actual
arguments (not just re-derive candidates blind) has a better shot at
catching it than another cold re-evaluation would.

TOOL ACCESS SCOPING NOTE: the current LLMClient/Ollama integration in this
codebase (src.llm_client) supports single-shot structured completions, not
an interactive tool-calling loop (model requests a lookup mid-reasoning,
gets a result, continues). Implementing genuine tool use would need new
client-side plumbing this experiment doesn't have yet. "Access to KG and
vocabulary" is implemented here as PRE-FETCHED evidence included in the
one prompt (same real athena_concept_relationship/athena_concept_ancestor
queries as src.tier4_kg_escalation, reused directly, not re-derived) --
the model sees the same facts it would get from a search, just already
retrieved rather than requested live. Flagging this scoping choice
explicitly rather than silently.
"""
import json

from src.llm_client import LLMUnavailable, parse_json_response
from src.tier4_kg_escalation import build_kg_context, _format_kg_evidence

_ARBITER_SYSTEM_PROMPT = (
    "You are a senior clinical terminology reviewer arbitrating a case "
    "where a 3-model ensemble of smaller language models could not agree. "
    "You are shown each model's own verdict and reasoning, plus real "
    "knowledge-graph evidence about each candidate concept. Your job is "
    "NOT to just pick the majority vote -- weigh the actual REASONING each "
    "model gave, not just how many models said it. A unanimous-sounding "
    "argument can still be wrong if all three models made the same "
    "mistake (e.g. all anchoring on which candidate's name sounds most "
    "literally similar to the raw text, rather than which one the KG "
    "evidence and clinical context actually support)."
)


def _format_model_verdicts(model_results: list) -> str:
    blocks = []
    for m in model_results:
        verdict = m.get("verdict", "ERROR")
        reasoning = None
        trail = m.get("eval_trail") or []
        if trail:
            # Prefer the tiebreak reasoning if present (the model's OWN
            # final comparative judgment), else the last real trail entry.
            tiebreak = next((t for t in trail if t.get("tiebreak")), None)
            reasoning = (tiebreak or trail[-1]).get("reasoning")
        blocks.append(f"  model: {m.get('model')}\n"
                      f"    verdict: {verdict}\n"
                      f"    reasoning: {reasoning or '(none recorded)'}")
    return "\n\n".join(blocks)


def build_arbiter_prompt(entity: dict, candidates: list, kg_context: list, model_results: list) -> str:
    cand_blocks = []
    for c, kg in zip(candidates, kg_context):
        cand_blocks.append(
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
        f"THE 3-MODEL ENSEMBLE'S OWN VERDICTS AND REASONING:\n\n"
        f"{_format_model_verdicts(model_results)}\n\n"
        f"CANDIDATES (with knowledge-graph evidence for each):\n\n"
        + "\n\n".join(cand_blocks) + "\n\n"
        f"Decide the single best-matching candidate. Weigh the ensemble's "
        f"reasoning AND the KG evidence together -- do not simply default "
        f"to whichever candidate got the most votes if the reasoning "
        f"behind those votes looks weak (e.g. surface-text similarity "
        f"rather than genuine clinical/KG support). Valid indices: {indices}.\n"
        'Reply with JSON: {"best_index": "<one of the valid indices, as a '
        'string>", "agrees_with_majority": <true or false>, '
        '"reasoning": "<one sentence citing the specific ensemble argument '
        'or KG evidence that decided it>"}'
    )


def _arbiter_schema(indices: list) -> dict:
    return {
        "type": "object",
        "properties": {
            "best_index": {"type": "string", "enum": [str(i) for i in indices]},
            "agrees_with_majority": {"type": "boolean"},
            "reasoning": {"type": "string"},
        },
        "required": ["best_index", "agrees_with_majority", "reasoning"],
    }


def arbitrate(client, conn, entity: dict, candidates: list, model_results: list) -> dict:
    """One 8B call, shown the 3B ensemble's own verdicts/reasoning plus
    pre-fetched KG evidence. Returns
    {"index": int|None, "agrees_with_majority": bool|None, "reasoning": str|None,
     "prompt": str, "kg_context": list, "error": str|None} -- always includes
    the full prompt/KG context for transparency, win or lose."""
    kg_context = build_kg_context(conn, candidates)
    prompt = build_arbiter_prompt(entity, candidates, kg_context, model_results)
    indices = [kg["index"] for kg in kg_context]
    try:
        raw = client.complete(_ARBITER_SYSTEM_PROMPT, prompt, schema=_arbiter_schema(indices))
        parsed = parse_json_response(raw["text"])
        idx = int(parsed.get("best_index"))
        if idx not in indices:
            return {"index": None, "agrees_with_majority": None, "reasoning": parsed.get("reasoning"),
                    "prompt": prompt, "kg_context": kg_context, "error": "invalid_index"}
        return {"index": idx, "agrees_with_majority": parsed.get("agrees_with_majority"),
                "reasoning": parsed.get("reasoning"), "prompt": prompt,
                "kg_context": kg_context, "error": None}
    except (LLMUnavailable, ValueError, TypeError, KeyError) as exc:
        return {"index": None, "agrees_with_majority": None, "reasoning": None,
                "prompt": prompt, "kg_context": kg_context, "error": f"{type(exc).__name__}: {exc}"}
