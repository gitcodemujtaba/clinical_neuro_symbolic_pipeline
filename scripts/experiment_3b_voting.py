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
from src.retrieval import VocabularyRetriever  # noqa: E402

MODELS = ["qwen2.5:3b", "llama3.2:3b", "phi4-mini"]

PROMPT_TEMPLATE = """You are a strict clinical data validator. Match the extracted entity to the single most clinically accurate candidate concept using the provided context and provenance.

EXTRACTED ENTITY: "{entity_text}"
SECTION: "{section_name}"
ASSERTION STATUS: "{assertion_status}"
LOCAL CONTEXT: "...{local_context}..."

CANDIDATE CONCEPTS:
{candidates_text}

RULES:
1. PROVENANCE OVERRIDES SPELLING: a candidate with "Basis: verified_brand_alias" is a mathematically verified database link. Trust it over your own lexical matching; do not reject it just because the spelling differs from the entity.
2. SCORE WARNING: a high "Score" is not reliable evidence on its own. Near-identical spelling frequently points to completely unrelated concepts. Confirm clinical meaning first.
3. SYNONYM TOLERANCE: do not reject valid clinical paraphrases purely due to minor wording differences. You are matching semantic concepts, not exact strings.
4. SAFETY FIRST: while you should tolerate synonyms, NEVER force a match if the candidate is clinically unrelated to the entity (e.g., mapping a symptom to a biological genus). If no candidate is a true clinical match, you MUST return "NONE_CORRECT".
5. IGNORE NEGATION: you are mapping semantic concepts, not diagnosing. If an entity is NEGATED (e.g. 'denies fever'), you must still map it to the 'Fever' concept.
6. Most candidate lists are AI-generated and may contain NO correct answers. If none are a perfect clinical match, you MUST output "NONE_CORRECT".

Return JSON with keys "reasoning" (1-2 sentences) and "verdict" ("1", "2", "3", etc., or "NONE_CORRECT")."""


def load_entities(conn, note_ids, limit_per_note=None):
    rows = conn.execute("""
        SELECT n.note_id, n.original_text, n.candidates,
               e.local_context, e.section_name, e.assertion_status,
               e.orig_start, e.orig_end
        FROM normalized_entities n
        JOIN extracted_entities e
          ON e.note_id = n.note_id AND e.original_text = n.original_text
         AND e.expanded_text = n.expanded_text AND e.entity_label = n.gliner_label
        WHERE n.is_test = TRUE AND n.confidence_tier_in = 'LOW'
          AND n.note_id IN ({})
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    out = []
    per_note = collections.Counter()
    for note_id, text, cands_json, context, section, assertion_status, start, end in rows:
        cands = json.loads(cands_json) if isinstance(cands_json, str) else (cands_json or [])
        if len(cands) < 2:
            continue
        if limit_per_note and per_note[note_id] >= limit_per_note:
            continue
        per_note[note_id] += 1
        out.append({"note_id": note_id, "text": text, "candidates": cands,
                    "context": context or "", "section": section or "",
                    "assertion_status": assertion_status or "PRESENT",
                    "orig_start": start, "orig_end": end})
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


def build_prompt(entity):
    return PROMPT_TEMPLATE.format(
        entity_text=entity["text"], section_name=entity["section"] or "unknown",
        assertion_status=entity["assertion_status"], local_context=entity["context"][:800],
        candidates_text=_format_candidates(entity["candidates"]))


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


def get_vote(model_name, prompt, candidates):
    try:
        response = ollama.generate(model=model_name, prompt=prompt, format="json",
                                   options={"temperature": 0.0})
        result = json.loads(response["response"])
        raw_verdict = result.get("verdict", "ERROR")
        reasoning = result.get("reasoning", "")
        verdict = normalize_verdict(raw_verdict, candidates)
        return {"model": model_name, "verdict": verdict, "raw_verdict": str(raw_verdict).strip(),
               "reasoning": reasoning}
    except Exception as exc:
        return {"model": model_name, "verdict": "ERROR", "raw_verdict": "ERROR",
               "reasoning": f"{type(exc).__name__}: {exc}"}


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


def run_vote(entity):
    prompt = build_prompt(entity)
    votes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(get_vote, m, prompt, entity["candidates"]): m for m in MODELS}
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
                         "eliminating bracket/index-tracking failures on 3B models.")
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

    results = []
    t0 = time.time()
    for i, entity in enumerate(entities, 1):
        r = check_deterministic_bypass(entity) or vote_fn(entity)
        outcome = grade(entity, r["final_verdict"], gold_by_note, vocab)
        r["outcome"] = outcome
        results.append(r)
        elapsed = time.time() - t0
        flag = "  <-- DIAGNOSTIC CASE" if entity["text"].lower() in ("lasix", "hemetesis", "edema") else ""
        print(f"[{i}/{len(entities)}] [{elapsed:.0f}s] {entity['text']!r} "
              f"({len(entity['candidates'])} cands) -> {r['status']} "
              f"verdict={r['final_verdict']} votes={r['vote_counts']} "
              f"outcome={outcome}{flag}")

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

    with open("/tmp/claude-1000/-home-ec2-user-clinical-neuro-symbolic-pipeline/"
             "7444c2b9-cd38-4c07-8fda-852a885403b4/scratchpad/3b_voting_results.json",
             "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "entity"} |
                  {"note_id": r["entity"]["note_id"], "text": r["entity"]["text"],
                   "candidates": r["entity"]["candidates"]}
                  for r in results], f, indent=2, default=str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
