"""
scripts/experiment_3b_voting.py — EXPERIMENTAL, standalone. NOT wired into the
production Stage 3 pipeline (src/mollm_ensemble.py, src/llm_client.py).

WHAT THIS IS. A mini-test comparing a 3-model majority-vote ensemble
(qwen2.5:3b, llama3.2:3b, phi4-mini, served locally via Ollama) against the
production BioMistral+OpenBioLLM setup, on the same real candidate lists
already sitting in normalized_entities. Motivated by three entities found by
hand-tracing model_disagreement cases in the live DB (docs from this session):

    'lasix'     -- candidates [lasalocid, LASCUFLOXACIN, lascufloxacin
                   hydrochloride, latex]. The true concept (furosemide) is not
                   in the list at all (Tier 1/2 both 0 hits -- a vocabulary
                   synonym gap, not a resolution problem). Correct verdict:
                   NONE_CORRECT.
    'hemetesis' -- candidates [Hemerocampa, Haemadipsa, Haematoxenus] (moth /
                   leech / parasite genus names -- SapBERT matched on
                   lexical/phonetic proximity to a misspelling, not meaning).
                   Correct verdict: NONE_CORRECT.
    'edema'     -- candidates [Edema, Edema] (a literal duplicate -- a small
                   separate Stage 2b bug). Correct verdict: candidate 1 (or 2,
                   they're identical).

WHY NOT JUST REUSE src/mollm_ensemble.py's build_prompt(). That function
renders assertion status, guideline evidence, is-a hierarchy hints, and
citation instructions the production ensemble depends on. This experiment
uses a deliberately simpler prompt (given by the user) to test whether
smaller general models handle the CORE task -- "is any of these candidates
actually this entity" -- differently than the specialized biomedical models,
before deciding whether to port any of the production safety machinery over.

VOTING RULE: 3-0 or 2-1 majority -> AUTO_VALIDATED (verdict = majority
answer). 1-1-1 (three-way split, or three-way including ERROR) -> HITL.
Grading against gold reuses the SAME crosswalk logic evaluation/cal_eval.py
uses (VocabularyRetriever.snomed_code_for_concept), so AUTO_VALIDATED
precision here is directly comparable to the P2.1 override-gate replay
numbers already measured for the production pipeline.

Run:  python3 scripts/experiment_3b_voting.py --note-ids 10000032-DS-21 --limit-per-note 30
      python3 scripts/experiment_3b_voting.py --diagnostic-only   # just the 3 traced cases
"""

import argparse
import collections
import concurrent.futures
import json
import os
import re
import sys
import time

import duckdb
import ollama

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
sys.path.insert(0, PROJECT_DIR)

from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from scripts.score_gold_recall import load_gold, overlaps  # noqa: E402
from src.retrieval import (  # noqa: E402
    DuckDBHierarchy, GroundingRetriever, GuidelineIndex, VocabularyRetriever,
)
from src.mollm_ensemble import (  # noqa: E402
    _format_evidence, _format_suppressed, _lab_value_note, verify_citations,
)
from src.llm_client import verdict_schema  # noqa: E402
from scripts.test_stage3_live import TRIPLETS_CANDIDATES  # noqa: E402

MODELS = ["qwen2.5:3b", "llama3.2:3b", "phi4-mini"]

# 2026-08-14 (cascade architecture -- user's dynamic routing funnel proposal,
# built on the live finding that a topically-irrelevant single retrieved rule
# got cited by name anyway when it was the only evidence shown). Shared by
# BOTH prompt variants below because these six apply regardless of whether
# an EVIDENCE block is present -- PROVENANCE/SCORE bear on the CANDIDATE
# list's own Basis/Score fields, not on retrieved guideline evidence, so the
# clean Stage 2 baseline pass needs them just as much as the Stage 3 rescue
# does. Only the citation rule is genuinely evidence-specific, and it lives
# solely in PROMPT_TEMPLATE_GROUNDED below -- the baseline prompt doesn't
# mention citation at all, not even to say "there is none", so a model never
# has "cite something" primed into its context when there is nothing to cite.
_BASE_INSTRUCTIONS = """INSTRUCTIONS:
1. PROVENANCE OVERRIDES SPELLING: a candidate with "Basis: verified_brand_alias" is a mathematically verified database link. Trust it over your own lexical matching; do not reject it just because the spelling differs from the entity.
2. SCORE WARNING: a high "Score" is not reliable evidence on its own. Near-identical spelling frequently points to completely unrelated concepts. Confirm clinical meaning first.
3. SYNONYM TOLERANCE: do not reject valid clinical paraphrases purely due to minor wording differences. You are matching semantic concepts, not exact strings.
4. SAFETY FIRST: while you should tolerate synonyms, NEVER force a match if the candidate is clinically unrelated to the entity (e.g., mapping a symptom to a biological genus). If no candidate is a true clinical match, you MUST return "NONE_CORRECT".
5. IGNORE NEGATION: you are mapping semantic concepts, not diagnosing. If an entity is NEGATED (e.g. 'denies fever'), you must still map it to the 'Fever' concept.
6. Most candidate lists are AI-generated and may contain NO correct answers. If none are a perfect clinical match, you MUST output "NONE_CORRECT".
7. COMPOSITE ENTITY REJECTION: if the EXTRACTED ENTITY contains multiple distinct clinical measurements, medications, or findings run together (e.g. "AlkPhos-84 Amylase-149" is two separate lab values, "ASA-NEG Ethanol-28" is a medication status AND a lab value), you MUST vote "NONE_CORRECT" unless a candidate explicitly represents a multi-panel or combined test. Do not force a match onto just one of the components, and do not force a match onto a "ratio" concept unless the entity text itself names a ratio -- two lab values reported side by side are not a ratio between them."""

PROMPT_TEMPLATE_BASELINE = """You are a strict clinical data validator. Match the extracted entity to the single most clinically accurate candidate concept using the provided context and provenance.

EXTRACTED ENTITY: "{entity_text}"{lab_value_note}
ENTITY TYPE: "{entity_label}"
SECTION: "{section_name}"
ASSERTION STATUS: "{assertion_status}"
LOCAL CONTEXT: "...{local_context}..."

CANDIDATE CONCEPTS:
{candidates_text}

""" + _BASE_INSTRUCTIONS + """

Return JSON with keys "reasoning" (1-2 sentences) and "verdict" ("1", "2", "3", etc., or "NONE_CORRECT")."""

PROMPT_TEMPLATE_GROUNDED = """You are a strict clinical data validator. Match the extracted entity to the single most clinically accurate candidate concept using the provided context, provenance, and guideline evidence.

EXTRACTED ENTITY: "{entity_text}"{lab_value_note}
ENTITY TYPE: "{entity_label}"
SECTION: "{section_name}"
ASSERTION STATUS: "{assertion_status}"
LOCAL CONTEXT: "...{local_context}..."

CANDIDATE CONCEPTS:
{candidates_text}

{evidence_text}
""" + _BASE_INSTRUCTIONS + """
8. CITE YOUR EVIDENCE: the EVIDENCE block above (if non-empty) lists guideline rules, each starting with "rule_id: REF-X" where X is a short letter (REF-A, REF-B, ...). If one of these rules supports your verdict, cite it by that exact short REF-X code and quote the exact text you are citing. It is NEVER a plain number 1-8. Never cite a REF-X code that is not shown inside the EVIDENCE block, and never cite one of these numbered instructions. Leave cited_evidence empty if the EVIDENCE block is empty or none of its rules are relevant to your verdict.

Return JSON with keys "reasoning" (1-2 sentences), "verdict" ("1", "2", "3", etc., or "NONE_CORRECT"), and "cited_evidence" (array of {{"rule_id": "REF-X", "quote": "..."}} using ONLY REF-X codes from the EVIDENCE block, possibly empty)."""


def load_entities(conn, note_ids, limit_per_note=None):
    """2026-08-14 (GroundingRetriever bridge): now pulls entity_id,
    expanded_text, entity_label (gliner_label) and experiencer too --
    GroundingRetriever.retrieve() needs all of these to run its four
    evidence channels, plus the entity's linked relations (fetched below,
    same query mollm_ensemble.py's load_records_for_stage3() uses) for
    Channel E. Without these fields retrieval degrades silently to
    "no evidence found" rather than actually running -- this is what makes
    the difference between grounding actually happening and merely looking
    like it does.
    """
    rows = conn.execute("""
        SELECT n.note_id, n.original_text, n.candidates,
               e.entity_id, e.expanded_text, e.entity_label, e.experiencer,
               e.local_context, e.section_name, e.assertion_status,
               e.orig_start, e.orig_end, n.normalized_from
        FROM normalized_entities n
        JOIN extracted_entities e
          ON e.note_id = n.note_id AND e.original_text = n.original_text
         AND e.expanded_text = n.expanded_text AND e.entity_label = n.gliner_label
        WHERE n.is_test = TRUE AND n.confidence_tier_in = 'LOW'
          AND (e.superseded_by_split IS NULL OR e.superseded_by_split = FALSE)
          AND (e.superseded_by_growth IS NULL OR e.superseded_by_growth = FALSE)
          AND n.note_id IN ({})
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    out = []
    per_note = collections.Counter()
    for (note_id, text, cands_json, entity_id, expanded_text, entity_label, experiencer,
         context, section, assertion_status, start, end, normalized_from) in rows:
        cands = json.loads(cands_json) if isinstance(cands_json, str) else (cands_json or [])
        if len(cands) < 2:
            continue
        if limit_per_note and per_note[note_id] >= limit_per_note:
            continue
        per_note[note_id] += 1

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

        out.append({"note_id": note_id, "text": text, "candidates": cands,
                    "entity_id": entity_id, "expanded_text": expanded_text or text,
                    "gliner_label": entity_label, "experiencer": experiencer or "PATIENT",
                    "confidence_tier_in": "LOW",  # the WHERE clause above already guarantees this
                    "original_text": text, "relations": rels,
                    "context": context or "", "section": section or "",
                    "assertion_status": assertion_status or "PRESENT",
                    "orig_start": start, "orig_end": end,
                    "normalized_from": normalized_from})
    return out


_HALO_BASES = ("verified_brand_alias", "exact_text", "synonym")


def _format_candidates(cands):
    """Multi-line, visually-isolated candidate block. A dense single-line
    '(Domain: X, Vocab: Y, Score: Z, Basis: W)' parenthetical measurably lets
    3B models detach the Basis tag from its own [i] index and misattribute it
    to whichever candidate has the highest score instead (2026-08-14, live
    diagnostic regression: two 'lasix' cases both moved to candidate [2]
    LASCUFLOXACIN -- the highest-scored one, not [4] furosemide, the one
    actually tagged verified_brand_alias). Putting Basis on its own line, in
    a format the bracket-tracking attention only has to resolve once per
    candidate rather than mid-clause, is the fix being tested here.
    """
    lines = []
    for i, c in enumerate(cands, 1):
        basis = c.get("match_basis", "semantic_similarity")
        score = c.get("similarity_score", "N/A")
        if isinstance(score, float):
            score = round(score, 4)
        basis_str = f">>> {basis.upper()} <<<" if basis in _HALO_BASES else basis
        lines.append(
            f"[{i}] {c.get('concept_name')}\n"
            f"    Basis: {basis_str}\n"
            f"    Details: Domain: {c.get('domain_id')} | Vocab: {c.get('vocabulary_id')} | Score: {score}"
        )
    return "\n\n".join(lines)


def _alias_rule_ids(retrieval):
    """Replaces every real rule_id (a ~100-200 char compound string like
    'filename.json::PREDICATE::urn:uuid:...->urn:uuid:...') with a short
    REF-A/REF-B/... handle before the prompt is built, and returns the
    mapping needed to translate a citation back afterward.

    WHY THIS EXISTS (2026-08-14, live diagnostic finding). qwen2.5:3b could
    reproduce a full real rule_id verbatim and get verified; llama3.2:3b and
    phi4-mini, given the identical retrieved evidence, consistently
    mangled it -- citing just the trailing UUID fragment, or dropping the
    filename prefix -- while still quoting the RIGHT text. That is a
    reproduction-fidelity failure (a long, token-dense, unusually-shaped
    string is hard for a 3B model to copy exactly), not a fabrication, but
    verify_citations() cannot tell the difference from a real fabrication
    without exact match, and loosening verify_citations() itself would weaken
    a deliberate patient-safety check shared with the live Stage 3 pipeline.
    Shortening the ID the model has to reproduce removes the friction at the
    source instead: verify_citations() still runs against the REAL rule_id
    (mapped back in get_vote() before it's called), so its strictness is
    completely unaffected -- this only changes what makes a citation easy to
    get right, not what counts as verified.
    """
    rules = retrieval.get("rules") or []
    released = retrieval.get("suppressed_rules_released") or []
    if not rules and not released:
        return retrieval, {}

    def _short_id(n):
        return f"REF-{chr(65 + n)}" if n < 26 else f"REF-{n}"

    short_to_real = {}
    aliased_rules = []
    for i, r in enumerate(rules):
        sid = _short_id(i)
        short_to_real[sid] = r["rule_id"]
        aliased_rules.append({**r, "rule_id": sid})

    aliased_released = []
    for j, r in enumerate(released, start=len(rules)):
        sid = _short_id(j)
        short_to_real[sid] = r["rule_id"]
        aliased_released.append({**r, "rule_id": sid})

    aliased = dict(retrieval)
    aliased["rules"] = aliased_rules
    if released:
        aliased["suppressed_rules_released"] = aliased_released
    return aliased, short_to_real


def _baseline_verdict_schema(allowed_verdicts):
    """2026-08-14 (cascade architecture): the genuinely clean Stage 2 schema
    -- no cited_evidence property at all, not merely an empty/optional one.
    Deliberately NOT built from src.llm_client.verdict_schema() (which always
    includes cited_evidence/request/reasoning_basis, since those exist for
    production's guideline-citation and follow-up-request workflow) -- none
    of that applies to a pass that was never shown any evidence to begin
    with, and a present-but-unused property is exactly the kind of thing a
    3B model on greedy decoding will occasionally fill in anyway.
    """
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": sorted(allowed_verdicts)},
            "reasoning": {"type": "string"},
            "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
        },
        "required": ["verdict", "reasoning", "confidence"],
    }


def _grounded_verdict_schema(allowed_verdicts, short_ids, require_citation=True):
    """verdict_schema() (src/llm_client.py, shared with production) plus one
    local tightening: cited_evidence[].rule_id is constrained to an ENUM of
    the short IDs actually offered in this prompt, not a free string. Under
    guided decoding this makes citing a nonexistent rule_id structurally
    impossible, the same way the verdict field itself is already
    structurally impossible to set to an out-of-vocabulary value. Built by
    extending verdict_schema()'s own output rather than duplicating its
    structure, so the two schemas cannot drift apart on everything else.
    """
    schema = verdict_schema(allowed_verdicts, require_citation=require_citation)
    if short_ids:
        schema["properties"]["cited_evidence"]["items"]["properties"]["rule_id"] = {
            "type": "string", "enum": short_ids,
        }
    return schema


def build_prompt(entity, retrieval=None):
    """2026-08-14 (cascade architecture): `retrieval=None` selects
    PROMPT_TEMPLATE_BASELINE -- no EVIDENCE section, no citation instruction,
    no cited_evidence key requested at all, the genuinely "clean" Stage 2
    prompt. A retrieval dict (already rule_id-aliased by the caller via
    _alias_rule_ids(), so the EVIDENCE block carries short REF-A/REF-B ids,
    never the real ones) selects PROMPT_TEMPLATE_GROUNDED, the Stage 3 rescue
    prompt. Evidence rendering reuses src/mollm_ensemble.py's
    _format_evidence()/_format_suppressed() rather than reimplementing it --
    the same one production's build_prompt() uses.
    """
    lab_value_note = _lab_value_note(entity["text"])
    if retrieval is None:
        return PROMPT_TEMPLATE_BASELINE.format(
            entity_text=entity["text"], lab_value_note=lab_value_note,
            entity_label=entity.get("gliner_label") or "unknown",
            section_name=entity["section"] or "unknown",
            assertion_status=entity["assertion_status"], local_context=entity["context"][:800],
            candidates_text=_format_candidates(entity["candidates"]))
    evidence_text = _format_evidence(retrieval) + _format_suppressed(retrieval)
    return PROMPT_TEMPLATE_GROUNDED.format(
        entity_text=entity["text"], lab_value_note=lab_value_note,
        entity_label=entity.get("gliner_label") or "unknown",
        section_name=entity["section"] or "unknown",
        assertion_status=entity["assertion_status"], local_context=entity["context"][:800],
        candidates_text=_format_candidates(entity["candidates"]),
        evidence_text=evidence_text)


_DIGIT_RE = re.compile(r"\b([1-9])\b")


def normalize_verdict(raw_verdict, candidates):
    """Maps a raw model 'verdict' string to a canonical '1'..'N' / NONE_CORRECT
    / UNPARSEABLE label.

    Without ollama's format='json' enforcing an enum (unlike production's
    vLLM guided_json_schema), a 3B model frequently answers with the
    candidate's NAME, an index buried in other text (e.g. '[2] Intermaxillary
    fixation...'), or an index embedded in a section header it echoed back
    ('Cleaned Candidates[3]') instead of a bare integer. Counter(raw_verdicts)
    then treats 'Construction of tracheostomy' and '1' as different votes even
    when they name the same candidate, manufacturing 1-1-1 splits (-> HITL)
    out of real 2-1 majorities.

    ORDER MATTERS. Name-matching runs BEFORE digit-extraction, not after --
    verified against a real case that gets this wrong the other way round:
    verdict text 'First heart sound S-1' against candidates
    ['Serum tumor marker stage S1', '1/36', 'First heart sound S-1'].
    Digit-first extraction finds the '1' in 'S-1' (word-bounded) and would
    resolve to candidate 1 -- WRONG, that's a different concept the '1' just
    happens to appear next to. The verdict text is an exact (case-insensitive)
    match for candidate 3's own name; checking names first resolves correctly
    and digit-extraction is only a fallback for text with no matching name.

    EXACT match beats SUBSTRING match beats digit extraction, and among
    substring matches the LONGEST candidate name wins -- otherwise a short
    generic name (e.g. 'Sweating', a substring of 'Excessive sweating') could
    shadow the correct longer, more specific candidate purely by iteration
    order.

    Returns 'UNPARSEABLE' (never 'NONE_CORRECT') when nothing matches, kept
    distinct so a genuinely garbled response ('CALCULATED_AS_CORRECT') cannot
    manufacture a false NONE_CORRECT consensus with a real NONE_CORRECT vote.
    """
    if not raw_verdict:
        return "NONE_CORRECT"
    raw_str = str(raw_verdict).strip()
    if not raw_str:
        return "NONE_CORRECT"
    if "NONE" in raw_str.upper():
        return "NONE_CORRECT"

    raw_lower = raw_str.lower()
    names = [(i, (c.get("concept_name") or "").strip().lower())
             for i, c in enumerate(candidates, 1)]

    for idx, name in names:
        if name and name == raw_lower:
            return str(idx)

    substring_matches = [(idx, name) for idx, name in names
                         if name and (name in raw_lower or raw_lower in name)]
    if substring_matches:
        idx, _ = max(substring_matches, key=lambda pair: len(pair[1]))
        return str(idx)

    digit_match = _DIGIT_RE.search(raw_str)
    if digit_match:
        cand_idx = int(digit_match.group(1))
        if 1 <= cand_idx <= len(candidates):
            return str(cand_idx)

    return "UNPARSEABLE"


def get_vote(model_name, prompt, candidates, retrieval=None, short_to_real=None):
    """2026-08-14 (GroundingRetriever bridge): now requests schema-constrained
    decoding (src.llm_client.verdict_schema(), the same one production uses)
    instead of bare format="json" -- the verdict is now structurally
    constrained to a real candidate index or NONE_CORRECT, and cited_evidence
    is a real (rule_id, quote) array the model fills in rather than free
    text. normalize_verdict()'s name/digit-matching fallback is kept as a
    safety net (schema requests can still fall back to unguided mode), but
    should mostly be a no-op now.

    `short_to_real` (from _alias_rule_ids()) tightens the schema further --
    cited_evidence[].rule_id becomes an ENUM of just the short REF-X codes
    shown in this prompt (via _grounded_verdict_schema()), so citing anything
    else is structurally impossible, not merely checked after the fact.

    citation_verified/citation_checks are computed here (via
    src/mollm_ensemble.verify_citations(), reused rather than
    reimplemented) whenever `retrieval` is given. Verification always runs
    against the REAL rule_ids (retrieval is never aliased) -- a cited REF-X
    is mapped back to its real rule_id first, so verify_citations()'s
    strictness against the actual evidence is completely unaffected by the
    aliasing; `cited_evidence` itself is stored with the short codes still in
    it, more compact for a human reading the report than the ~200-char
    original would be.
    """
    allowed = {str(i) for i in range(1, len(candidates) + 1)} | {"NONE_CORRECT"}
    if retrieval is None:
        # Genuinely no cited_evidence property in the schema at all -- not
        # merely optional. A model that was never shown any evidence and is
        # still asked "did you cite anything?" has an empty box to fill, and
        # a 3B model on greedy decoding will sometimes fill it anyway (the
        # exact 'Rule1: Leucocyte count basis - verified_brand_alias' style
        # hallucination measured before this schema existed). Removing the
        # property removes the invitation.
        schema = _baseline_verdict_schema(allowed)
    else:
        require_citation = bool(retrieval.get("rules"))
        short_ids = list((short_to_real or {}).keys())
        schema = _grounded_verdict_schema(allowed, short_ids, require_citation=require_citation)
    try:
        response = ollama.generate(model=model_name, prompt=prompt, format=schema,
                                   options={"temperature": 0.0})
        result = json.loads(response["response"])
        raw_verdict = result.get("verdict", "ERROR")
        reasoning = result.get("reasoning", "")
        cited_evidence = result.get("cited_evidence") or []
        verdict = normalize_verdict(raw_verdict, candidates)
        vote = {"model": model_name, "verdict": verdict, "raw_verdict": str(raw_verdict).strip(),
               "reasoning": reasoning, "cited_evidence": cited_evidence}
        if retrieval is not None:
            resolved = [{**c, "rule_id": (short_to_real or {}).get(c.get("rule_id"), c.get("rule_id"))}
                       for c in cited_evidence if isinstance(c, dict)]
            vote.update(verify_citations({"cited_evidence": resolved}, retrieval))
        return vote
    except Exception as exc:
        return {"model": model_name, "verdict": "ERROR", "raw_verdict": "ERROR",
               "reasoning": f"{type(exc).__name__}: {exc}", "cited_evidence": [],
               "citation_verified": None, "citation_checks": [], "citations_made": 0}


BINARY_PROMPT_TEMPLATE = """You are a clinical data validator.

EXTRACTED ENTITY: "{entity_text}"
SECTION: "{section_name}"
ASSERTION STATUS: "{assertion_status}"
LOCAL CONTEXT: "...{local_context}..."

CANDIDATE CONCEPT:
Name: {concept_name}
Domain: {domain_id}
Basis: {basis}

RULES:
1. Ignore negation status when mapping the concept -- you are labeling which concept the text refers to, not diagnosing. A NEGATED entity (e.g. 'denies fever') still maps to its concept ('Fever') if the name matches.
2. If Basis is verified_brand_alias, it is a mathematically verified terminology-database link -- accept it even if the spelling looks unlike the entity.
3. If this candidate is a distinct or clinically unrelated concept (e.g. mapping a symptom to a biological genus), reject it. Do not force a match.

Is this candidate concept a valid match for the extracted entity? Return JSON with keys "reasoning" (1 sentence) and "match" (true or false)."""


def build_binary_prompt(entity, candidate):
    basis = candidate.get("match_basis", "semantic_similarity")
    return BINARY_PROMPT_TEMPLATE.format(
        entity_text=entity["text"], section_name=entity["section"] or "unknown",
        assertion_status=entity["assertion_status"], local_context=entity["context"][:800],
        concept_name=candidate.get("concept_name"), domain_id=candidate.get("domain_id"),
        basis=basis)


def get_binary_vote(model_name, prompt):
    try:
        response = ollama.generate(model=model_name, prompt=prompt, format="json",
                                   options={"temperature": 0.0})
        result = json.loads(response["response"])
        match = result.get("match")
        if not isinstance(match, bool):
            # Ollama's basic format="json" mode has no boolean-typed schema
            # enforcement (unlike vLLM's guided_json in production) -- a 3B
            # model sometimes answers "true"/"yes" as a string instead of a
            # JSON bool. Best-effort coercion rather than a hard failure.
            match = str(match).strip().lower() in ("true", "yes")
        return {"match": match, "reasoning": result.get("reasoning", ""), "error": None}
    except Exception as exc:
        return {"match": False, "reasoning": "", "error": f"{type(exc).__name__}: {exc}"}


def evaluate_candidates_sequentially(model_name, entity):
    """1-to-1 binary evaluation loop (2026-08-14, "attention dilution"
    follow-up to the multi-line formatting fix). The 1-to-N multiple-choice
    prompt measurably let small models detach a candidate's Basis tag from
    its own [i] index and misattribute it to whichever candidate scored
    highest instead -- even with the tag visually isolated on its own line.
    Asking about exactly ONE candidate per call removes bracket-tracking
    entirely: there is no index to hallucinate. Stops at the first accepted
    candidate (matching the original list's rank order, i.e. candidates most
    likely correct by Stage 2b's own signal are checked first).

    Costs len(candidates) calls per model in the worst case (all rejected)
    instead of 1 -- a real, multiplicative increase in inference cost this
    trades for eliminating the index-hallucination failure mode.
    """
    trail = []
    any_success = False
    for i, cand in enumerate(entity["candidates"], 1):
        prompt = build_binary_prompt(entity, cand)
        result = get_binary_vote(model_name, prompt)
        trail.append({"candidate_index": i, "concept_name": cand.get("concept_name"), **result})
        if result["error"] is None:
            any_success = True
        if result["match"]:
            return {"model": model_name, "verdict": str(i), "raw_verdict": str(i),
                   "reasoning": result["reasoning"], "eval_trail": trail}
    if not any_success:
        return {"model": model_name, "verdict": "ERROR", "raw_verdict": "ERROR",
               "reasoning": "all candidate calls errored", "eval_trail": trail}
    return {"model": model_name, "verdict": "NONE_CORRECT", "raw_verdict": "NONE_CORRECT",
           "reasoning": trail[-1]["reasoning"] if trail else "", "eval_trail": trail}


def run_vote_sequential(entity):
    """Same aggregation/voting rule as run_vote(), but each model reaches its
    verdict via evaluate_candidates_sequentially() instead of one 1-to-N call."""
    votes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(evaluate_candidates_sequentially, m, entity): m for m in MODELS}
        for future in concurrent.futures.as_completed(futures):
            votes.append(future.result())

    vote_counts = collections.Counter(v["verdict"] for v in votes)
    top_verdict, top_count = vote_counts.most_common(1)[0]

    if top_count == 3 and top_verdict not in ("ERROR", "UNPARSEABLE"):
        status = "AUTO_VALIDATED"
        confidence = "HIGH"
    elif top_count == 2 and top_verdict not in ("ERROR", "UNPARSEABLE"):
        status = "MOLLM_RESOLVED"
        confidence = "MEDIUM"
    else:
        status = "HITL_REQUIRED"
        confidence = None

    return {"entity": entity, "votes": votes, "vote_counts": dict(vote_counts),
           "status": status, "confidence": confidence, "final_verdict": top_verdict}


def check_deterministic_bypass(entity):
    """Fast-path: skip the MoLLM ensemble entirely when Stage 2's OMOP graph
    traversal already proved the answer, rather than asking a 3B model for
    permission to trust a fact the knowledge graph already confirmed.

    ONLY verified_brand_alias qualifies -- NOT exact_text. Measured this
    corpus's own stage2b_cal_eval.py numbers before adding this: Tier 1
    "1 (Exact)" accuracy is 52.48% (402/766), barely better than chance, so
    "the search text happens to equal some concept's name" is nowhere near a
    safe auto-validate criterion -- it says nothing about which DOMAIN or
    SENSE of an ambiguous abbreviation was actually meant. verified_brand_alias
    is categorically different: it means a specific 3-hop KG relationship
    (Brand name of -> Tradename of -> RxNorm has ing) was walked and confirmed
    to exist, not that a string happened to match.

    ONLY fires when there is EXACTLY ONE such candidate. A combination brand
    (e.g. "Tylenol") can legitimately produce several verified_brand_alias
    candidates, one per active ingredient -- which one THIS mention means is
    real ambiguity the graph does not resolve, so multiple hits fall through
    to the ensemble instead of arbitrarily picking by list order.
    """
    alias_hits = [(i, c) for i, c in enumerate(entity.get("candidates", []), 1)
                 if c.get("match_basis") == "verified_brand_alias"]
    if len(alias_hits) != 1:
        return None
    i, c = alias_hits[0]
    return {
        "entity": entity, "status": "AUTO_VALIDATED", "confidence": "HIGH",
        "final_verdict": str(i), "vote_counts": {str(i): 3},
        "votes": [{
            "model": "deterministic_graph_engine", "verdict": str(i), "raw_verdict": str(i),
            "reasoning": (f"Bypassed LLM ensemble: candidate [{i}] ({c.get('concept_name')}) "
                         f"is a graph-verified brand alias, not a probabilistic match."),
        }],
    }


def run_vote(entity, retrieval=None):
    aliased_retrieval, short_to_real = _alias_rule_ids(retrieval or {})
    prompt = build_prompt(entity, aliased_retrieval)
    votes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(get_vote, m, prompt, entity["candidates"],
                                   retrieval, short_to_real): m
                  for m in MODELS}
        for future in concurrent.futures.as_completed(futures):
            votes.append(future.result())

    vote_counts = collections.Counter(v["verdict"] for v in votes)
    top_verdict, top_count = vote_counts.most_common(1)[0]

    # 2026-08-13 (user decision after the first 75-entity mini-test measured
    # 2-1 "MEDIUM" precision at 50.0% -- a coin flip -- against 3-0 "HIGH" at
    # 72.7%). 2-1 no longer auto-validates; it gets its own MOLLM_RESOLVED
    # tier (mirroring src/mollm_ensemble.py's INGESTION_RESOLVED) so a
    # majority verdict is still recorded and usable, just not silently
    # shipped without review.
    # UNPARSEABLE excluded from winning a majority for the same reason ERROR
    # is: a garbled, unmappable response is not a vote for anything, and must
    # not silently count toward consensus just because two of them happened
    # to produce the identical UNPARSEABLE sentinel.
    #
    # 2026-08-14 (Fragile Concept Gate, user proposal). A 3-0 consensus is
    # only as trustworthy as the candidate concept it agreed on. When the
    # winning candidate only exists in normalized_entities because
    # src/normalization/orchestrator.py's LAB VALUE SUFFIX FALLBACK had to
    # salvage it (Tier 1/2 failed on the raw text; the value-stripped retry
    # is what actually resolved) -- normalized_from starts with
    # "value_stripped_from_" -- three models agreeing is still built on one
    # fragile retrieval step, not independent confirmation of a solid match.
    # Measured on the 16393593-DS-5 smoke test: giving the models a clean
    # display of the entity (Option 1.5's lab-value note) made WBC-13/RBC-3/
    # PTT-114/AlkPhos-255/Phos-9.4 unanimously WRONG instead of visibly
    # split -- consensus went up, precision did not, and AUTO_VALIDATED (the
    # tier that skips human review) is exactly where that must not happen.
    # Gated on this PRECISE provenance field, not "gliner_label == 'Lab Test'
    # and match_tier == 3 (Semantic)" -- that broader heuristic was checked
    # empirically first and would have wrongly caught 145/362 (40%) clean,
    # legitimate Tier-3 resolutions (e.g. "ALT", "hemoglobin") that never
    # went through this fallback at all.
    fragile_concept = bool(entity.get("normalized_from")) and \
        entity["normalized_from"].startswith("value_stripped_from_")

    if top_count == 3 and top_verdict not in ("ERROR", "UNPARSEABLE") and not fragile_concept:
        status = "AUTO_VALIDATED"
        confidence = "HIGH"
    elif top_count == 3 and top_verdict not in ("ERROR", "UNPARSEABLE") and fragile_concept:
        status = "MOLLM_RESOLVED"
        confidence = "MEDIUM"
    elif top_count == 2 and top_verdict not in ("ERROR", "UNPARSEABLE"):
        status = "MOLLM_RESOLVED"
        confidence = "MEDIUM"
    else:
        status = "HITL_REQUIRED"
        confidence = None

    # 2026-08-14 (GroundingRetriever bridge): among the votes that actually
    # agreed on top_verdict, did any of them back it with a citation that
    # verify_citations() confirmed genuinely holds up (not just "the model
    # tried to cite something")? This is what "how many 2-1 splits are
    # grounded" actually means -- citations_made > 0 alone would also count
    # a fabricated one.
    top_verdict_votes = [v for v in votes if v.get("verdict") == top_verdict]
    grounded = any(v.get("citation_verified") and v.get("citations_made", 0) > 0
                  for v in top_verdict_votes)

    return {"entity": entity, "votes": votes, "vote_counts": dict(vote_counts),
           "status": status, "confidence": confidence, "final_verdict": top_verdict,
           "top_verdict_grounded": grounded}


def run_cascade(entity, retriever):
    """2026-08-14 (dynamic routing funnel -- user's cascade proposal, direct
    response to the live finding that feeding every entity guideline evidence
    it doesn't need is pure noise/hallucination risk, not a free win).

    Stage 1: the existing deterministic bypass (verified_brand_alias /
    exact_text) -- unchanged, still the fastest and safest path when it
    fires.

    Stage 2: zero-shot ensemble, NO EVIDENCE block (run_vote(entity,
    retrieval=None), which now selects the genuinely evidence-free
    PROMPT_TEMPLATE_BASELINE / _baseline_verdict_schema()). ANY clean 3-0
    consensus -- including NONE_CORRECT -- is trusted and returned
    immediately: an independent, unprompted 3-0 agreement that nothing fits
    is as strong a signal as a 3-0 agreement on a specific candidate, and
    most entities are a straightforward semantic match in the first place.
    Showing every entity guideline text it doesn't need only gives a small
    model something irrelevant to latch onto (measured directly:
    'UreaN-13 Creat-1.2' had a single retrieved rule about sepsis/antibiotic
    duration, topically unrelated to a renal lab ratio, and got cited by
    name anyway before this schema existed).

    Stage 3: only entities Stage 2 could NOT resolve cleanly -- a genuine
    2-1 or 1-1-1 split -- pay for GroundingRetriever.retrieve() and, if it
    actually found evidence, re-run WITH the EVIDENCE block as a rescue pass.
    If retrieval comes back with zero rules there is nothing to rescue with,
    so Stage 2's split is returned as final rather than paying for a second
    ensemble call that would see an empty EVIDENCE block and reduce to a
    (non-deterministic-looking but actually redundant) repeat of Stage 2.
    Both stages' outcomes are kept on the returned record (stage2_result)
    whenever Stage 3 actually runs, so the rescue's effect stays auditable
    rather than only the final answer surviving.

    retriever=None (e.g. --no-grounding) disables Stage 3 entirely --
    Stage 2's result is returned as final for anything the bypass didn't
    catch, giving a pure zero-shot baseline run.
    """
    bypassed = check_deterministic_bypass(entity)
    if bypassed:
        bypassed["cascade_stage"] = "1_deterministic_bypass"
        return bypassed

    stage2 = run_vote(entity, retrieval=None)
    stage2["cascade_stage"] = "2_zero_shot"
    if stage2["status"] == "AUTO_VALIDATED":
        return stage2
    if retriever is None:
        return stage2

    retrieval = retriever.retrieve(entity)
    if not retrieval.get("rules"):
        stage2["cascade_stage"] = "2_zero_shot_unresolved_no_evidence"
        return stage2

    stage2_summary = {"status": stage2["status"], "final_verdict": stage2["final_verdict"],
                      "vote_counts": stage2["vote_counts"]}
    stage3 = run_vote(entity, retrieval)
    stage3["cascade_stage"] = "3_kg_grounded_rescue"
    stage3["stage2_result"] = stage2_summary
    return stage3


def grade(entity, verdict, gold_by_note, vocab):
    """Same crosswalk logic as evaluation/cal_eval.py grade(): does the
    verdict's chosen candidate's SNOMED code match an overlapping gold span.
    Returns 'correct' / 'incorrect' / None (ungradable)."""
    gold = gold_by_note.get(entity["note_id"], [])
    overlapping = [g for g in gold
                   if overlaps(entity["orig_start"], entity["orig_end"], g["start"], g["end"])]
    if not overlapping:
        return None
    gold_codes = {g["concept_id"] for g in overlapping}

    if verdict == "NONE_CORRECT":
        # Correct iff NO candidate in the list crosswalks to gold -- same
        # definition cal_eval.py's grade() uses for NONE_CORRECT.
        candidate_codes = [vocab.snomed_code_for_concept(c.get("omop_concept_id"))
                           for c in entity["candidates"]]
        any_correct = any(code in gold_codes for code in candidate_codes if code)
        return "correct" if not any_correct else "incorrect"

    try:
        idx = int(verdict) - 1
    except (TypeError, ValueError):
        return None
    if idx < 0 or idx >= len(entity["candidates"]):
        return None
    concept_id = entity["candidates"][idx].get("omop_concept_id")
    code = vocab.snomed_code_for_concept(concept_id) if concept_id is not None else None
    if code is None:
        return None
    return "correct" if code in gold_codes else "incorrect"


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None, help="comma-separated note_ids")
    ap.add_argument("--limit-per-note", type=int, default=None)
    ap.add_argument("--diagnostic-only", action="store_true",
                    help="only run the 3 hand-traced cases: lasix, hemetesis, edema "
                         "(note 10000032-DS-21)")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--sequential", action="store_true",
                    help="use the 1-to-1 binary evaluation loop "
                         "(evaluate_candidates_sequentially/run_vote_sequential) "
                         "instead of the 1-to-N multiple-choice prompt. Costs up to "
                         "len(candidates)x more calls per model; trades that for "
                         "eliminating bracket/index-tracking failures on 3B models. "
                         "NOT wired to GroundingRetriever (see --no-grounding) -- "
                         "grounding was only bridged into the default 1-to-N path, so "
                         "the baseline this flag is compared against stays isolated.")
    ap.add_argument("--no-grounding", action="store_true",
                    help="2026-08-14: skip GroundingRetriever entirely and render an "
                         "empty EVIDENCE block, reproducing the pre-bridge prompt "
                         "exactly. Escape hatch for a quick A/B or if retrieval setup "
                         "fails. Has no effect with --sequential, which never grounds.")
    args = ap.parse_args()
    vote_fn = run_vote_sequential if args.sequential else run_vote

    conn = duckdb.connect(args.db, read_only=True)

    if args.diagnostic_only:
        entities = load_entities(conn, ["10000032-DS-21"])
        entities = [e for e in entities if e["text"].lower() in ("lasix", "hemetesis", "edema")]
    else:
        note_ids = [n.strip() for n in args.note_ids.split(",")] if args.note_ids else \
            ["10000032-DS-21"]
        entities = load_entities(conn, note_ids, limit_per_note=args.limit_per_note)

    print(f"entities to test: {len(entities)}")
    for m in MODELS:
        print(f"  model available: {m}")
    print()

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    all_notes = sorted(set(e["note_id"] for e in entities))
    gold_rows = load_gold(gold_path, all_notes)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)
    vocab = VocabularyRetriever(conn)

    # 2026-08-14 (GroundingRetriever bridge). Built once, same construction
    # pattern as scripts/test_stage3_live.py, reusing this run's own `vocab`
    # rather than a second VocabularyRetriever instance. --sequential never
    # grounds (see --sequential's help text); --no-grounding skips it too.
    retriever = None
    if not args.sequential and not args.no_grounding:
        triplets = next((p for p in TRIPLETS_CANDIDATES if os.path.exists(p)), None)
        if not triplets:
            print(f"\nNo guideline corpus found. Tried: {TRIPLETS_CANDIDATES}")
            print("Re-run with --no-grounding to proceed without KG evidence.")
            return 1
        index = GuidelineIndex(triplets)
        retriever = GroundingRetriever(index, vocab, hierarchy=DuckDBHierarchy(conn))
        print(f"guideline KG: {index.stats['nodes']} nodes, {index.stats['rules']} rules")
        print("hierarchy: DuckDBHierarchy (athena_concept_ancestor)\n")

    results = []
    t0 = time.time()
    for i, entity in enumerate(entities, 1):
        if args.sequential:
            r = check_deterministic_bypass(entity) or vote_fn(entity)
        else:
            # run_cascade handles the deterministic bypass itself (Stage 1),
            # then zero-shot (Stage 2), then -- only for what Stage 2
            # couldn't resolve cleanly -- the KG-grounded rescue (Stage 3).
            # retriever=None (--no-grounding) makes Stage 3 a no-op, so this
            # degrades to a pure zero-shot run without a separate code path.
            r = run_cascade(entity, retriever)
        outcome = grade(entity, r["final_verdict"], gold_by_note, vocab)
        r["outcome"] = outcome
        results.append(r)
        elapsed = time.time() - t0
        flag = "  <-- DIAGNOSTIC CASE" if entity["text"].lower() in ("lasix", "hemetesis", "edema") else ""
        grounded_flag = "  [grounded]" if r.get("top_verdict_grounded") else ""
        stage_flag = f"  [{r['cascade_stage']}]" if r.get("cascade_stage") else ""
        print(f"[{i}/{len(entities)}] [{elapsed:.0f}s] {entity['text']!r} "
              f"({len(entity['candidates'])} cands) -> {r['status']} "
              f"verdict={r['final_verdict']} votes={r['vote_counts']} "
              f"outcome={outcome}{grounded_flag}{stage_flag}{flag}")

    print()
    print("=" * 78)
    print("DIAGNOSTIC CASES")
    print("=" * 78)
    for r in results:
        if r["entity"]["text"].lower() in ("lasix", "hemetesis", "edema"):
            print(f"\n{r['entity']['text']!r}: expected NONE_CORRECT (lasix/hemetesis) "
                  f"or a match (edema)")
            print(f"  final: {r['status']} / {r['final_verdict']} votes={r['vote_counts']}")
            for v in r["votes"]:
                print(f"    {v['model']:<12} verdict={v['verdict']:<15} "
                      f"reasoning={v['reasoning'][:150]!r}")

    print()
    print("=" * 78)
    print("AGGREGATE")
    print("=" * 78)
    status_counts = collections.Counter(r["status"] for r in results)
    print(f"routing: {dict(status_counts)}")

    for tier in ("AUTO_VALIDATED", "MOLLM_RESOLVED", "HITL_REQUIRED"):
        rows = [r for r in results if r["status"] == tier]
        graded = [r for r in rows if r["outcome"] in ("correct", "incorrect")]
        correct = sum(1 for r in graded if r["outcome"] == "correct")
        prec = f"{correct/len(graded)*100:.1f}%" if graded else "n/a"
        print(f"\n{tier}: {len(rows)} total, {len(graded)} gradable, "
             f"{correct} correct -- precision {prec}")

    coverage = len([r for r in results if r["status"] == "AUTO_VALIDATED"]) / len(results) * 100
    print(f"\nAUTO_VALIDATED coverage (fraction skipping human review): {coverage:.1f}%")

    if retriever is not None:
        print("\n" + "-" * 78)
        print("CASCADE (2026-08-14 dynamic routing funnel)")
        print("-" * 78)
        stage_counts = collections.Counter(r.get("cascade_stage") for r in results)
        for stage in ("1_deterministic_bypass", "2_zero_shot",
                     "2_zero_shot_unresolved_no_evidence", "3_kg_grounded_rescue"):
            n = stage_counts.get(stage, 0)
            print(f"  {stage}: {n} ({n/len(results)*100:.1f}%)")

        # Among entities escalated to Stage 3: did grounding actually change
        # the answer relative to what the ungrounded Stage 2 pass alone would
        # have shipped, and was the change an improvement?
        escalated = [r for r in results if r.get("cascade_stage") == "3_kg_grounded_rescue"]
        if escalated:
            changed = sum(1 for r in escalated
                         if r["stage2_result"]["final_verdict"] != r["final_verdict"])
            print(f"\n  of {len(escalated)} escalated to Stage 3, grounding changed the "
                 f"verdict in {changed} ({changed/len(escalated)*100:.1f}%)")
            stage2_outcomes = [grade(r["entity"], r["stage2_result"]["final_verdict"],
                                     gold_by_note, vocab) for r in escalated]
            stage2_graded = [o for o in stage2_outcomes if o in ("correct", "incorrect")]
            stage2_correct = sum(1 for o in stage2_graded if o == "correct")
            stage3_graded = [r["outcome"] for r in escalated if r["outcome"] in ("correct", "incorrect")]
            stage3_correct = sum(1 for o in stage3_graded if o == "correct")
            if stage2_graded:
                print(f"  Stage 2 (ungrounded) precision on this same escalated subset: "
                     f"{stage2_correct}/{len(stage2_graded)} = {stage2_correct/len(stage2_graded)*100:.1f}%")
            if stage3_graded:
                print(f"  Stage 3 (KG-grounded) precision on this same escalated subset: "
                     f"{stage3_correct}/{len(stage3_graded)} = {stage3_correct/len(stage3_graded)*100:.1f}%")

        print("\n" + "-" * 78)
        print("GROUNDING (2026-08-14 GroundingRetriever bridge)")
        print("-" * 78)
        for tier in ("AUTO_VALIDATED", "MOLLM_RESOLVED", "HITL_REQUIRED"):
            rows = [r for r in results if r["status"] == tier and r["votes"]
                   and r["votes"][0].get("model") != "deterministic_graph_engine"]
            grounded = sum(1 for r in rows if r.get("top_verdict_grounded"))
            print(f"{tier}: {grounded}/{len(rows)} top-verdict decisions backed by a "
                 f"verified citation" + (f" ({grounded/len(rows)*100:.1f}%)" if rows else ""))
        total_citations = sum(v.get("citations_made", 0) for r in results for v in r["votes"])
        verified_citations = sum(
            1 for r in results for v in r["votes"] for c in (v.get("citation_checks") or [])
            if c.get("verified"))
        print(f"\ncitations attempted: {total_citations}   verified: {verified_citations}"
             + (f"  ({verified_citations/total_citations*100:.1f}%)" if total_citations else ""))

    none_correct_auto = sum(1 for r in results if r["status"] == "AUTO_VALIDATED"
                            and r["final_verdict"] == "NONE_CORRECT")
    print(f"AUTO_VALIDATED as NONE_CORRECT: {none_correct_auto}")

    error_votes = sum(1 for r in results for v in r["votes"] if v["verdict"] == "ERROR")
    print(f"\nmodel call errors (bad JSON / timeout / exception): {error_votes} "
         f"of {len(results)*3} total calls")
    unparseable_votes = sum(1 for r in results for v in r["votes"] if v["verdict"] == "UNPARSEABLE")
    print(f"unparseable verdicts (valid JSON, but verdict text matched no candidate "
         f"name/index): {unparseable_votes} of {len(results)*3} total calls")
    renamed_votes = sum(1 for r in results for v in r["votes"]
                        if v["verdict"] not in ("ERROR", "UNPARSEABLE")
                        and v["verdict"] != v.get("raw_verdict"))
    print(f"votes normalized from a name/index string to a canonical index "
         f"(the bug this run fixes): {renamed_votes} of {len(results)*3} total calls")

    out_name = "3b_voting_results_grounded.json" if retriever is not None else "3b_voting_results.json"
    with open("/tmp/claude-1000/-home-ec2-user-clinical-neuro-symbolic-pipeline/"
             f"7444c2b9-cd38-4c07-8fda-852a885403b4/scratchpad/{out_name}",
             "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "entity"} |
                  {"note_id": r["entity"]["note_id"], "text": r["entity"]["text"],
                   "candidates": r["entity"]["candidates"]}
                  for r in results], f, indent=2, default=str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
