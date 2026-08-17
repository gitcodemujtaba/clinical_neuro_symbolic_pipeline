"""
src/mollm_wholenote_ensemble.py -- EXPERIMENTAL. Not Objective 2, not
Objective 3, not wired into src/hitl_queue.py or any production write path.

WHAT THIS TESTS. User proposal (2026-08-15, same day as the anchoring-bias
prompt fixes in mollm_ensemble.py/mollm_review.py): instead of validating one
entity at a time against a sentence-bounded local_context window, show each
model the ENTIRE raw clinical note plus a batch of that note's extracted
entities (current Stage 1 label + Stage 2 concept mapping) at once, so the
model can cross-reference OTHER mentions/sections in the same note to
disambiguate (the exact failure mode the anchoring-bias work kept finding --
"S2" resolved to a sacral vertebra when the rest of the note is a cardiac
exam, "AVL" resolved to a laterality qualifier when the rest of the note is
an EKG lead list -- information a single-sentence window structurally cannot
supply but a whole note usually can).

WHY THIS IS A SEPARATE FILE ON A SEPARATE BRANCH, NOT A MODIFICATION OF
EITHER EXISTING MODULE. This is a materially different call shape (N entities
per completion, not 1) with real, unresolved architectural costs that the
existing modules deliberately avoid:
  1. GUIDED-DECODING ARRAY RISK. mollm_ensemble.py/mollm_review.py constrain
     generation to ONE verdict object; this constrains it to an ARRAY of
     15-20. Longer structured generations degrade more on 3B local models
     ("lost in the middle"), and a bad completion here loses an entire
     chunk's verdicts, not one entity's.
  2. NO PER-ENTITY EVIDENCE CITATION. Objective 2's citation-gated,
     evidence-only design is this project's stated primary novelty claim.
     Retrieving and including guideline evidence for every entity in a chunk,
     on top of the full note text, would blow prompt size past what's
     practical for local 3B models at this corpus's entity density (this
     corpus averages 107 entities/note, up to 169 -- see
     docs/2026-08-15_Contradiction_Detection_Analysis.md's investigation of
     this proposal for the numbers). Scoped out deliberately for this
     experiment: this module tests ONLY "does full-note context beat
     sentence-window context", not a citation-gated replacement for
     Objective 2. Do not wire this into KG3 write-back or treat its verdicts
     as evidence-grounded -- they are terminology-knowledge judgments only,
     same basis as mollm_review.py's no-evidence branch.
  3. NOT PER-ENTITY ROBUST. One malformed generation now costs a whole chunk
     (CHUNK_SIZE entities), not one. _query_one_chunk() below does its own
     salvage (per-object regex recovery from a truncated array) rather than
     reusing llm_client.parse_json_response(), which is written for a single
     top-level object and would not help here.

Entities within a note are chunked to CHUNK_SIZE (not all ~100+ at once) --
see the module docstring above and the design discussion in
docs/2026-08-15_Contradiction_Detection_Analysis.md for why an ungated
single-call-per-note design was rejected as too high-risk for local 3B
models before ever building it.
"""
import csv
import json
import os
import re
import sys
import uuid

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.llm_client import LLMUnavailable, build_clients

CHUNK_SIZE = 15
# ~250 output tokens/entity (reasoning + fields) * chunk size + buffer.
# mollm_review.py/mollm_ensemble.py's MAX_OUTPUT_TOKENS=800 is sized for one
# verdict object; this call produces up to CHUNK_SIZE of them in one array.
MAX_TOKENS_PER_CHUNK = 250 * CHUNK_SIZE + 500
CLIENT_TIMEOUT = 300.0  # full note + chunk of entities is a much bigger prompt/completion than either existing module sends

GOLD_NOTES_CSV = os.path.join(PROJECT_DIR, "data", "raw_notes", "gold_notes.csv")

ASSESSMENT_VALUES = {
    "CORRECT",
    "ENTITY_LABEL_INCORRECT",
    "CONCEPT_MAPPING_INCORRECT",
    "BOTH_INCORRECT",
    "UNCERTAIN",
}

SYSTEM_PROMPT = """You are a clinical terminology auditor reviewing a batch of entities \
extracted from ONE clinical note by an automatic pipeline. You are shown the ENTIRE note, \
not just the sentence each entity came from -- USE the rest of the note to disambiguate. \
An abbreviation or finding that looks ambiguous in isolation is often resolved unambiguously \
by another mention, a section header, or the overall clinical picture elsewhere in the SAME \
note (e.g. a cardiac-exam section elsewhere confirms "S2" means a heart sound, not a sacral \
vertebra; an EKG/cath-lab section confirms "LCx" means an artery, not a laterality qualifier).

Your task is LABELING, not clinical decision-making: for each entity, judge whether the \
extractor's entity TYPE and the terminology mapping's concept NAME are the CORRECT selections \
for that text span in THIS note's context -- not whether the patient's care is appropriate, \
not a diagnosis, not a treatment recommendation.

=== CRITICAL AUDIT INSTRUCTIONS ===
1. IGNORE THE MATCH BASIS/SCORE shown for each entity's current mapping. A high similarity \
score or an exact-text match does NOT mean the mapping is contextually correct -- evaluate as \
if it were hidden.
2. BE THE DEVIL'S ADVOCATE: treat every current mapping as unproven until the note confirms \
it, not as confirmed until the note contradicts it. Actively look for the reason it might be \
wrong -- a wrong abbreviation expansion, a string match with the wrong clinical meaning, a \
lab-test-vs-diagnosis mismatch.
3. CONCEPTUAL FIREWALL: judge what the CURRENT MAPPING actually denotes using ONLY its own \
name -- never let a superficially similar phrase elsewhere in the note stand in for it.
4. Respect assertion status shown for each entity: a finding marked ABSENT was explicitly \
negated in the note and should not be judged as though present.
5. UNCERTAIN IS NOT A DEFAULT FOR INCOMPLETE INFORMATION. If you can construct even one \
clinically reasonable reading from the note, give CORRECT or the relevant INCORRECT category \
at LOW confidence rather than retreating to UNCERTAIN.

Assessment values:
  CORRECT                    - both the entity TYPE and concept NAME mapping are right
  ENTITY_LABEL_INCORRECT     - the entity TYPE is wrong; concept mapping may or may not follow
  CONCEPT_MAPPING_INCORRECT  - the concept NAME mapping is wrong; entity TYPE is fine
  BOTH_INCORRECT             - both are wrong
  UNCERTAIN                  - you genuinely cannot construct any defensible reading

Reply with a JSON ARRAY, one object per entity shown, in any order, using EXACTLY the \
entity_id values given -- no extras, no omissions:
[{"entity_id": "<must match one shown below>",
  "assessment": "<one of the five values above>",
  "reasoning": "<one sentence: what does the rest of the note tell you about this entity that its own sentence alone would not?>",
  "proposed_entity_label": "<text, or empty string if not proposing one>",
  "proposed_concept_name": "<text, or empty string if not proposing one>",
  "confidence": "<HIGH|MEDIUM|LOW>"}, ...]"""


_notes_cache = None


def load_raw_note_text(note_id: str) -> str:
    """Reads data/raw_notes/gold_notes.csv once, cached module-wide -- 272
    rows, small enough to hold entirely in memory rather than re-parsing the
    CSV per note.
    """
    global _notes_cache
    if _notes_cache is None:
        _notes_cache = {}
        with open(GOLD_NOTES_CSV, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                _notes_cache[row["note_id"]] = row["text"]
    text = _notes_cache.get(note_id)
    if text is None:
        raise KeyError(f"note_id {note_id!r} not found in {GOLD_NOTES_CSV}")
    return text


def chunk_entities(records: list, chunk_size: int = CHUNK_SIZE) -> list:
    """Splits records (already ordered by orig_start -- see
    load_validation_records()) into reading-order chunks. Reading order,
    not random/hash order, so a chunk boundary rarely splits two entities
    that were about to disambiguate each other anyway (imperfect, but a
    cheap improvement over an arbitrary split)."""
    return [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]


def _format_entity_line(rec: dict) -> str:
    cands = rec.get("candidates") or []
    top1 = cands[0] if cands else None
    if top1:
        mapping = (f'{top1.get("concept_name")} (OMOP {top1.get("omop_concept_id")}, '
                   f'{top1.get("vocabulary_id")}/{top1.get("domain_id")})')
    else:
        mapping = "NONE -- Stage 2 could not map this entity"
    assertion = rec.get("assertion_status") or "PRESENT"
    return (f'  [{rec["entity_id"]}] text: {rec.get("original_text")!r}  '
            f'section: {rec.get("section_name") or "unknown"}  '
            f'assertion: {assertion}\n'
            f'      extractor label: {rec.get("gliner_label") or rec.get("entity_label")}  '
            f'currently mapped to: {mapping}')


def build_chunk_prompt(raw_text: str, chunk_records: list) -> str:
    lines = [
        "FULL CLINICAL NOTE:",
        "-" * 40,
        raw_text,
        "-" * 40,
        "",
        f"ENTITIES TO REVIEW ({len(chunk_records)} of this note's total, extracted by an "
        f"automatic pipeline -- review each using the WHOLE note above):",
    ]
    for rec in chunk_records:
        lines.append(_format_entity_line(rec))
    lines.append("")
    lines.append("Reply with the JSON array now, one object per entity_id listed above.")
    return "\n".join(lines)


def build_chunk_schema(entity_ids: list) -> dict:
    """entity_id constrained to an enum of the EXACT ids in this chunk --
    same rationale as verdict_schema()'s enum: makes a hallucinated id
    structurally impossible rather than merely detectable. minItems/maxItems
    matching chunk size is advisory (guided decoding can still under/over-
    produce or duplicate an id -- validated post-hoc in review_note(), not
    trusted blindly)."""
    return {
        "type": "array",
        "minItems": len(entity_ids),
        "maxItems": len(entity_ids),
        "items": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "enum": entity_ids},
                "assessment": {"type": "string", "enum": sorted(ASSESSMENT_VALUES)},
                "reasoning": {"type": "string"},
                "proposed_entity_label": {"type": "string"},
                "proposed_concept_name": {"type": "string"},
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
            },
            "required": ["entity_id", "assessment", "reasoning", "confidence"],
        },
    }


def _salvage_array(text: str) -> list:
    """Best-effort recovery when the array didn't parse as whole JSON --
    llm_client.parse_json_response()'s salvage logic is written for ONE
    top-level object (looks for a single "verdict" key) and doesn't apply
    here. Each array element is independently useful, so this extracts
    complete {...} objects one at a time via a brace-matching scan and skips
    anything truncated mid-object, rather than discarding the whole chunk's
    response for one bad/cut-off tail element."""
    objs = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        objs.append(json.loads(text[start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
    return objs


def _parse_chunk_response(text: str) -> list:
    if not text or not text.strip():
        raise ValueError("empty response")
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # Some models wrap the array in {"entities": [...]} despite the
            # schema -- recover it rather than treat as unparseable.
            for v in parsed.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass
    salvaged = _salvage_array(cleaned)
    if salvaged:
        return salvaged
    raise ValueError(f"could not parse a JSON array from response: {cleaned[:200]!r}")


def _query_one_chunk(client, user_prompt: str, entity_ids: list) -> dict:
    """Returns {entity_id: model_output_dict}, plus '_meta' for ids that
    were missing/duplicated/unrecognized -- reported, never silently
    dropped or guessed."""
    schema = build_chunk_schema(entity_ids)
    raw = client.complete(SYSTEM_PROMPT, user_prompt, schema=schema,
                           max_tokens=MAX_TOKENS_PER_CHUNK)
    items = _parse_chunk_response(raw["text"])

    by_id = {}
    unknown_ids = []
    duplicate_ids = []
    for item in items:
        eid = item.get("entity_id")
        if eid not in entity_ids:
            unknown_ids.append(eid)
            continue
        if eid in by_id:
            duplicate_ids.append(eid)
            continue  # first one wins; duplicate reported, not silently overwritten
        assessment = str(item.get("assessment", "")).strip().upper()
        assessment_out_of_vocab = None
        if assessment not in ASSESSMENT_VALUES:
            assessment_out_of_vocab = assessment or None
            assessment = "UNCERTAIN"
        by_id[eid] = {
            "model": raw["model"],
            "assessment": assessment,
            "assessment_out_of_vocabulary": assessment_out_of_vocab,
            "reasoning": item.get("reasoning"),
            "proposed_entity_label": (item.get("proposed_entity_label") or "").strip() or None,
            "proposed_concept_name": (item.get("proposed_concept_name") or "").strip() or None,
            "raw_confidence_label": item.get("confidence"),
            "finish_reason": raw.get("finish_reason"),
            "decoding_mode": raw.get("decoding_mode"),
        }
    missing_ids = [eid for eid in entity_ids if eid not in by_id]
    return {
        "by_id": by_id,
        "missing_ids": missing_ids,
        "unknown_ids": unknown_ids,
        "duplicate_ids": duplicate_ids,
    }


def review_note(note_id: str, conn, clients: dict = None, chunk_size: int = CHUNK_SIZE,
                 progress_cb=None) -> dict:
    """Runs the whole-note review for every entity in `note_id` (ALL tiers --
    same "all tiers" scope as mollm_review.py, since this experiment's whole
    point is holistic review, not LOW-tier-only triage) and returns
    {entity_id: {..., 'models': [...]}} plus a run-level 'issues' list for
    any missing/duplicate/unknown ids across all chunks.
    """
    from src.mollm_ensemble import load_validation_records

    clients = clients if clients is not None else build_clients(timeout=CLIENT_TIMEOUT)
    records = load_validation_records(conn, note_id)
    raw_text = load_raw_note_text(note_id)
    chunks = chunk_entities(records, chunk_size)

    results = {}
    issues = []
    for chunk_idx, chunk in enumerate(chunks):
        entity_ids = [r["entity_id"] for r in chunk]
        user_prompt = build_chunk_prompt(raw_text, chunk)
        if progress_cb:
            progress_cb(note_id, chunk_idx, len(chunks), entity_ids)

        chunk_models = {eid: [] for eid in entity_ids}
        for name, client in clients.items():
            try:
                out = _query_one_chunk(client, user_prompt, entity_ids)
            except (LLMUnavailable, ValueError) as exc:
                issues.append({"note_id": note_id, "chunk_idx": chunk_idx,
                                "model": name, "error": str(exc)})
                continue
            for eid, model_out in out["by_id"].items():
                chunk_models[eid].append(model_out)
            if out["missing_ids"] or out["unknown_ids"] or out["duplicate_ids"]:
                issues.append({
                    "note_id": note_id, "chunk_idx": chunk_idx, "model": name,
                    "missing_ids": out["missing_ids"], "unknown_ids": out["unknown_ids"],
                    "duplicate_ids": out["duplicate_ids"],
                })

        for rec in chunk:
            eid = rec["entity_id"]
            results[eid] = {
                "entity_id": eid, "note_id": note_id, "chunk_idx": chunk_idx,
                "original_text": rec.get("original_text"),
                "entity_label": rec.get("gliner_label") or rec.get("entity_label"),
                "candidates": rec.get("candidates"),
                "models": chunk_models[eid],
            }

    return {"note_id": note_id, "entities": results, "issues": issues}


def main():
    import argparse
    import time

    import duckdb

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", required=True,
                     help="Comma-separated note_ids to run the whole-note review over.")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--db", default=os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb"))
    ap.add_argument("--out-dir", default=os.path.join(
        PROJECT_DIR, "reports", "contradiction_detection", "wholenote_results"))
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    note_ids = [n.strip() for n in args.note_ids.split(",") if n.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    conn = duckdb.connect(args.db, read_only=True)
    clients = build_clients(timeout=CLIENT_TIMEOUT)

    def progress(note_id, chunk_idx, n_chunks, entity_ids):
        print(f"  [{note_id}] chunk {chunk_idx + 1}/{n_chunks} "
              f"({len(entity_ids)} entities) -- querying {len(clients)} models...")

    for note_id in note_ids:
        start = time.time()
        print(f"=== {note_id} ===")
        result = review_note(note_id, conn, clients=clients,
                             chunk_size=args.chunk_size, progress_cb=progress)
        elapsed = time.time() - start
        n_issues = len(result["issues"])
        print(f"  done in {elapsed/60:.1f}m -- {len(result['entities'])} entities, "
              f"{n_issues} issue(s) logged")
        if n_issues:
            for issue in result["issues"]:
                print(f"    ISSUE: {issue}")
        out_path = os.path.join(args.out_dir, f"{note_id}.json")
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"  wrote {out_path}")

    conn.close()


if __name__ == "__main__":
    main()
