"""src/narrative_state_word_coldstart.py — "cold start" direct-injection for
common single-word clinical state descriptors (alert/improved/baseline/
warm/clinic) GLiNER never proposes as entities at any confidence -- the
second population identified in the same corpus-wide extraction-recall
miss analysis that produced src.lab_abbrev_coldstart.

WHY THIS EXISTS, AND WHY IT'S SMALLER THAN THE LAB-ABBREVIATION FIX. The
top-repeated missed texts included several common narrative words
(pain, alert, improved, stable, negative, baseline, warm, tender, masses,
wound, procedure, support) alongside the lab abbreviations. Unlike bare
lab-panel abbreviations, these are genuinely polysemous -- a bare
"stable" can mean vital-signs-stable, condition-stable, or something else
entirely depending on context, and blindly cold-starting an ambiguous
term would inject wrong concepts confidently, the opposite of what a
"verified" mapping is supposed to be.

EVIDENCE-GROUNDED SCREEN, CASE-MERGED. Every candidate term was checked
against train_annotations.csv with case variants MERGED into one
consistency count (an earlier case-SEPARATED count understated some
terms' real consistency and overstated others' -- e.g. 'alert'/'Alert'
each looked ~80-100% consistent split apart but are 100% consistent to
the SAME concept once merged). Only terms clearing >=95% single-concept
consistency, same bar as src.physexam_shorthand and src.lab_abbrev_
coldstart, are included here:
  alert     100.0% (377/377)  -> 'Mentally alert'
  improved  100.0% (270/270)  -> "Patient's condition improved"
  baseline  100.0% (163/163)  -> 'Baseline state'
  warm       98.9% (174/176)  -> 'Warm skin'
  clinic    100.0% ( 34/34 )  -> 'Outpatient care management'

DELIBERATELY EXCLUDED, checked and found NOT safe: pain (76.7%, splits
across 4 distinct concepts by body site/type), stable (88.4%, splits
across 4 concepts), negative (64.5%, splits ~65/28 across two genuinely
different concepts), procedure (80.0%), support (44.4%), tender (84.0%),
masses (44.7%), wound (80.0%). These stay GLiNER's job (or a future,
context-aware fix) rather than being force-injected on a false-confidence
majority.

Also deliberately excluded despite clearing the bar: 'edema' (98.2%,
267038008) -- src.physexam_shorthand already has its own 'edema' entry
mapped to a DIFFERENT concept (433595), evidently a real context-
dependent difference (physexam's is scoped to physical-exam-section
occurrences specifically) rather than a bug in either mapping. Keeping
this module's responsibility off any term physexam_shorthand already
owns avoids the two modules silently disagreeing on the same token, same
call made for 'CV' in src.lab_abbrev_coldstart.

'alert' is ALSO in src.physexam_shorthand's own dictionary (same concept,
4086843/'Mentally alert') -- not a conflict, since that module only
scans WITHIN physical-exam-family sections; this module scans the WHOLE
note (including the lowercase 'alert' case-variant that module's dict
doesn't have), so the two are additive, not competing, and the standard
skip-anything-already-extracted check prevents any double-injection.

NO SECTION GATING, unlike physexam_shorthand -- these words plausibly
appear anywhere in a note's narrative, not just within a recognized
physical-exam section, so (unlike src.lab_abbrev_coldstart too) this is
a whole-note word-boundary scan.

Reuses src.normalization.orchestrator's _cold_start_mapping() (the same
generalized bypass src.physexam_shorthand uses) via the
`narrative_coldstart` marker field -- these concepts skip Tier 1-3
search entirely rather than relying on ordinary semantic search to
rediscover a common-English-word's one gold-verified sense.
"""
import re

# text (exact case, as it appears in gold) ->
# (omop_concept_id, concept_name, omop_domain, gliner_label)
NARRATIVE_STATE_WORD_TERMS = {
    "alert": (4086843, "Mentally alert", "Observation", "Symptom"),
    "Alert": (4086843, "Mentally alert", "Observation", "Symptom"),
    "improved": (4149524, "Patient's condition improved", "Observation", "Symptom"),
    "Improved": (4149524, "Patient's condition improved", "Observation", "Symptom"),
    "baseline": (4029350, "Baseline state", "Observation", "Symptom"),
    "Baseline": (4029350, "Baseline state", "Observation", "Symptom"),
    "warm": (4010974, "Warm skin", "Observation", "Symptom"),
    "Warm": (4010974, "Warm skin", "Observation", "Symptom"),
    "clinic": (42537845, "Outpatient care management", "Procedure", "Procedure"),
}

NARRATIVE_STATE_WORD_MATCH_BASIS = "verified_narrative_state_word"

_TERM_RE_CACHE = {}


def _term_re(term: str):
    pat = _TERM_RE_CACHE.get(term)
    if pat is None:
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])")
        _TERM_RE_CACHE[term] = pat
    return pat


def find_narrative_state_word_spans(raw_text: str) -> list:
    """Whole-note word-boundary scan for NARRATIVE_STATE_WORD_TERMS keys,
    exact case. Returns {start, end, text, omop_concept_id, concept_name,
    omop_domain, gliner_label}, offsets into raw_text, sorted by start."""
    out = []
    for term, (concept_id, concept_name, domain, label) in NARRATIVE_STATE_WORD_TERMS.items():
        for m in _term_re(term).finditer(raw_text):
            out.append({
                "start": m.start(), "end": m.end(), "text": m.group(0),
                "omop_concept_id": concept_id, "concept_name": concept_name,
                "omop_domain": domain, "gliner_label": label,
            })
    out.sort(key=lambda d: d["start"])
    return out


_LOCAL_CONTEXT_WINDOW = 150


def build_narrative_state_word_entities(raw_text: str, note_id: str,
                                        existing_entities: list) -> list:
    """Full extracted_entities-shaped dicts for narrative-state-word spans
    not already covered by an existing entity, ready to merge into
    src.clinical_pipeline.run_pipeline()'s `accepted` list. Same shape/
    contract as src.physexam_shorthand.build_physexam_shorthand_entities()
    and src.lab_abbrev_coldstart.build_lab_abbrev_coldstart_entities().
    """
    from src.entity_extraction import make_entity_id
    from src.assertion import annotate_assertions

    existing_spans = [(e["orig_start"], e["orig_end"]) for e in existing_entities]

    def _overlaps_existing(start, end):
        return any(not (end <= es or start >= ee) for es, ee in existing_spans)

    candidates = [c for c in find_narrative_state_word_spans(raw_text)
                 if not _overlaps_existing(c["start"], c["end"])]
    if not candidates:
        return []

    # Batched per line, same negation-scope discipline as
    # src.lab_abbrev_coldstart / src.physexam_shorthand -- these words are
    # genuinely negatable in a way bare lab abbreviations rarely are
    # ("not alert", "no improvement"), so real per-line assertion
    # detection matters here more than it did for that module.
    assertions = [None] * len(candidates)
    by_line = {}
    for idx, c in enumerate(candidates):
        line_start = raw_text.rfind("\n", 0, c["start"]) + 1
        line_end = raw_text.find("\n", c["end"])
        if line_end == -1:
            line_end = len(raw_text)
        by_line.setdefault((line_start, line_end), []).append(idx)

    for (line_start, line_end), idxs in by_line.items():
        line_text = raw_text[line_start:line_end]
        line_spans = [(candidates[i]["start"] - line_start,
                      candidates[i]["end"] - line_start, candidates[i]["gliner_label"])
                     for i in idxs]
        line_assertions = annotate_assertions(line_text, line_spans)
        for i, a in zip(idxs, line_assertions):
            assertions[i] = a

    out = []
    for c, assertion in zip(candidates, assertions):
        start, end = c["start"], c["end"]
        ctx_start = max(0, start - _LOCAL_CONTEXT_WINDOW)
        ctx_end = min(len(raw_text), end + _LOCAL_CONTEXT_WINDOW)
        entity_id = make_entity_id(note_id, start, end, c["gliner_label"])
        out.append({
            "entity_id": entity_id,
            "note_id": note_id,
            "entity_label": c["gliner_label"],
            "expanded_text": c["text"],
            "original_text": c["text"],
            "confidence": 1.0,
            "orig_start": start,
            "orig_end": end,
            "exp_start": start,
            "exp_end": end,
            "assertion_status": assertion["assertion_status"],
            "experiencer": assertion["experiencer"],
            "temporality": assertion["temporality"],
            "assertion_cue": assertion["assertion_cue"],
            "assertion_cue_start": assertion.get("assertion_cue_start"),
            "assertion_cue_end": assertion.get("assertion_cue_end"),
            "assertion_cue_category": assertion.get("assertion_cue_category"),
            "assertion_engine": assertion["assertion_engine"],
            "section_name": None,
            "sentence_id": None,
            "local_context": raw_text[ctx_start:ctx_end],
            "local_context_basis": "narrative_state_word_coldstart_raw_window",
            "expansion_ambiguous": False,
            "candidate_expansions": None,
            "selection_basis": None,
            "gliner_model_version": "narrative_state_word_coldstart",
            "extraction_threshold": None,
            "below_threshold": False,
            "flat_ner": None,
            "crosses_sentence_boundary": False,
            "sentence_ids_spanned": [],
            "compound_split_of": None,
            "superseded_by_split": False,
            "grown_from": None,
            "superseded_by_growth": False,
            "possibly_truncated": False,
            "gliner_input_token_count": None,
            # Consumed by src.normalization.orchestrator.
            # process_and_normalize_entities() to bypass Tier 1-3 search.
            "narrative_coldstart": {
                "omop_concept_id": c["omop_concept_id"],
                "concept_name": c["concept_name"],
                "omop_domain": c["omop_domain"],
            },
        })
    return out
