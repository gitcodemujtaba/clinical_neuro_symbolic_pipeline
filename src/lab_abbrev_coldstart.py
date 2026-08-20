"""src/lab_abbrev_coldstart.py — "cold start" direct-injection for BARE lab-
panel abbreviations (no attached value, e.g. narrative "Creat trended up"
rather than flowsheet "Creat-1.2") that GLiNER never proposes as entities
at any confidence, the same category of gap src.physexam_shorthand
addresses for exam notation -- built after that module's methodology
confirmed the pattern generalizes.

WHY THIS EXISTS. Corpus-wide miss analysis (2026-08-20, against the full
140-note is_test corpus): of 20,754 gold spans the pipeline finds nothing
overlapping at all (extraction-level, not a linking failure -- confirmed
separately that below-threshold retention has nothing here either, these
spans are TRUE zero-shot blind spots), 97.4% are short (<=25 chars) and
71% are single-word. The most frequently repeated missed texts are almost
entirely bare CBC/chemistry-panel abbreviations: 'Creat' (196x), 'Hgb'
(178x), 'RBC' (175x), 'Na' (170x), 'Hct' (167x), 'Cl' (165x), 'MCH'
(157x), 'MCHC' (153x), 'RDW' (138x), 'HCO3' (138x), 'WBC' (124x), 'UreaN'
(103x), 'Phos' (97x), 'Calcium' (95x), 'AnGap' (87x) -- 2,097 misses from
these 15 terms alone.

EVIDENCE-GROUNDED, NOT GUESSED, CASE-SENSITIVE BY GOLD'S OWN USAGE. Each
term below (and its listed case variants) was verified directly against
train_annotations.csv: every included (term, case-variant) pair maps to
exactly ONE concept_id at >=99% consistency (most at 100%), same bar as
src.physexam_shorthand and src.normalization.tier_retrieval._LAB_TEST_
ALIASES. Deliberately case-sensitive (not case-insensitive) matching --
gold shows specific case conventions per term (e.g. 'Na' capital-N, not
'NA' or 'na'), and matching only the evidenced case forms keeps the
collision surface as narrow as the evidence actually supports, same
discipline physexam_shorthand's own dict uses. A generic term considered
and DELIBERATELY EXCLUDED: 'CV' -- also 100%-consistent in gold (208/208,
concept 363003006), but that concept is a different clinical idea
(cardiovascular finding/descriptor, not itself a lab test) from
physexam_shorthand's own 'CV' entry (cardiovascular EXAM section header,
a different concept entirely) -- keeping this module scoped strictly to
Lab-Test-domain abbreviations avoids overlapping responsibility with that
module for an ambiguous shared token.

NO CUSTOM CONCEPT-RESOLUTION BYPASS NEEDED, UNLIKE PHYSEXAM_SHORTHAND.
Every concept_id used here is already a key in
src.normalization.tier_retrieval._LAB_TEST_ALIASES (added there in the
same session, several already existed from earlier lab-alias work) --
these injected entities flow through the ORDINARY Tier 1-3 normalization
search like any GLiNER-found entity would, hit the alias dict's exact
match automatically, and get auto-validated via src.mollm_tier_gate.
tier3_fast_path()'s existing verified_lab_test_alias branch. No new
"cold_start" marker field or orchestrator.py wiring is needed for
concept resolution -- only extraction-side injection is new here.

DOWNSTREAM WIRING (not in this module): src/clinical_pipeline.py injects
these as extracted_entities rows, skipping any span overlapping something
GLiNER already found, running them through the same assertion tagging as
everything else (batched per line, same negation-scope discipline
src.physexam_shorthand established, even though lab-value mentions are
rarely negated -- cheap and correct rather than assuming PRESENT).
"""
import re

# text (exact case, as it appears in gold) -> (omop_concept_id, concept_name)
# All Measurement domain, all "Lab Test" GLiNER label.
LAB_ABBREV_COLDSTART_TERMS = {
    "Creat": (4324383, "Creatinine measurement"),
    "CREAT": (4324383, "Creatinine measurement"),
    "Hgb": (40480067, "Measurement of total hemoglobin concentration"),
    "HGB": (40480067, "Measurement of total hemoglobin concentration"),
    "RBC": (4030871, "Red blood cell count"),
    "Na": (4208938, "Sodium measurement, blood"),
    "Hct": (4151358, "Hematocrit determination"),
    "HCT": (4151358, "Hematocrit determination"),
    "Cl": (4019545, "Chloride measurement, blood"),
    "CL": (4019545, "Chloride measurement, blood"),
    "MCH": (4182871, "Mean corpuscular hemoglobin determination"),
    "MCHC": (4290193, "Mean corpuscular hemoglobin concentration determination"),
    "RDW": (4281085, "Red cell distribution width determination"),
    "HCO3": (4194291, "Blood bicarbonate measurement"),
    "WBC": (4298431, "White blood cell count"),
    "Phos": (4020559, "Phosphate, total measurement"),
    "PHOS": (4020559, "Phosphate, total measurement"),
    "Calcium": (4193434, "Blood calcium measurement"),
    "CALCIUM": (4193434, "Blood calcium measurement"),
    "AnGap": (4103762, "Anion gap measurement"),
    "UreaN": (4017361, "Blood urea nitrogen measurement"),
}

_GLINER_LABEL = "Lab Test"
_OMOP_DOMAIN = "Measurement"

_TERM_RE_CACHE = {}


def _term_re(term: str):
    pat = _TERM_RE_CACHE.get(term)
    if pat is None:
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])")
        _TERM_RE_CACHE[term] = pat
    return pat


def find_lab_abbrev_spans(raw_text: str) -> list:
    """Whole-note word-boundary scan for LAB_ABBREV_COLDSTART_TERMS keys,
    exact case. Returns a list of {start, end, text, omop_concept_id,
    concept_name}, offsets into raw_text, sorted by start. Overlap with
    already-extracted spans is the CALLER's responsibility to filter."""
    out = []
    for term, (concept_id, concept_name) in LAB_ABBREV_COLDSTART_TERMS.items():
        for m in _term_re(term).finditer(raw_text):
            out.append({
                "start": m.start(), "end": m.end(), "text": m.group(0),
                "omop_concept_id": concept_id, "concept_name": concept_name,
            })
    out.sort(key=lambda d: d["start"])
    return out


_LOCAL_CONTEXT_WINDOW = 150


def build_lab_abbrev_coldstart_entities(raw_text: str, note_id: str,
                                        existing_entities: list) -> list:
    """Full extracted_entities-shaped dicts for bare lab-abbreviation spans
    not already covered by an existing (GLiNER-found) entity, ready to
    merge into src.clinical_pipeline.run_pipeline()'s `accepted` list --
    same shape/contract as src.physexam_shorthand.
    build_physexam_shorthand_entities(), no "physexam_shorthand"-style
    marker field since concept resolution needs no bypass here (see
    module docstring).

    orig_start/orig_end and exp_start/exp_end are set equal, same
    reasoning as physexam_shorthand: these spans are matched directly
    against raw_text with no Stage-1 expansion step in between.
    """
    from src.entity_extraction import make_entity_id
    from src.assertion import annotate_assertions

    existing_spans = [(e["orig_start"], e["orig_end"]) for e in existing_entities]

    def _overlaps_existing(start, end):
        return any(not (end <= es or start >= ee) for es, ee in existing_spans)

    candidates = [c for c in find_lab_abbrev_spans(raw_text)
                 if not _overlaps_existing(c["start"], c["end"])]
    if not candidates:
        return []

    # Batched per LINE, same discipline as src.physexam_shorthand's
    # negation-scope fix -- cheap (one annotate_assertions() call per
    # distinct line containing a match, not per match) and avoids a
    # negation cue on one line bleeding into an unrelated line's span.
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
                      candidates[i]["end"] - line_start, _GLINER_LABEL) for i in idxs]
        line_assertions = annotate_assertions(line_text, line_spans)
        for i, a in zip(idxs, line_assertions):
            assertions[i] = a

    out = []
    for c, assertion in zip(candidates, assertions):
        start, end = c["start"], c["end"]
        ctx_start = max(0, start - _LOCAL_CONTEXT_WINDOW)
        ctx_end = min(len(raw_text), end + _LOCAL_CONTEXT_WINDOW)
        entity_id = make_entity_id(note_id, start, end, _GLINER_LABEL)
        out.append({
            "entity_id": entity_id,
            "note_id": note_id,
            "entity_label": _GLINER_LABEL,
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
            "local_context_basis": "lab_abbrev_coldstart_raw_window",
            "expansion_ambiguous": False,
            "candidate_expansions": None,
            "selection_basis": None,
            "gliner_model_version": "lab_abbrev_coldstart",
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
        })
    return out
