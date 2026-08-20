"""src/physexam_shorthand.py — "cold start" direct-injection for physical-exam
telegraphic shorthand GLiNER never proposes as entities at all, at any
confidence.

WHY THIS EXISTS. Corpus-wide sizing (2026-08-18) against the 272-note gold
corpus, two passes:
  1. Gold annotations sitting inside "Physical Exam"-named sections: 1,788
     (2.4% of the whole 75,491-annotation corpus), 98.9% of them short
     (<=25 char) abbreviation-shaped spans (VS, RA, BP, NAD, HEENT, NT, ND,
     WWP, MMM, JVD, ...).
  2. A second, LARGER pass specifically on gold spans matching a section
     HEADER name that src.preprocessing.segment_sections() also detects as
     its own section (its flat header regex treats "Gen:"/"Abd:"/"Ext:"
     sub-headers as PARALLEL top-level sections, not nested under
     "Physical Exam:" -- confirmed live, not a hypothetical): HEENT alone
     is annotated 254 times corpus-wide, Mental Status 211, Activity
     Status 140, and every spelling/case variant of General/Abdomen/
     Extremities/Neuro/Lungs/Skin/Cardiovascular runs into the dozens to
     hundreds each. This is the dominant share of the whole pattern, not a
     side note.

Two manually-inspected notes this session showed 100% miss on this exact
category: GLiNER's training distribution evidently doesn't cover this
compressed, slash-delimited clinical shorthand ("Abd: S/NT/ND" = Soft/
Non-Tender/Non-Distended in one 8-char token) or the bare 2-4-letter
section-header abbreviations, at ANY confidence -- unlike the separate
below-threshold-recovery fix (EXTRACTION_THRESHOLD 0.5->0.35), lowering
the threshold further doesn't touch this, since these spans were never
proposed as candidates in the first place.

EVIDENCE-GROUNDED, NOT GUESSED. Every mapping below was built by mining
train_annotations.csv directly for the (text -> concept_id) mapping gold
actually uses, keeping only >=95%-consistent terms (in practice every
included term is 100% consistent). Concept IDs and domains are OMOP
concept_id/domain_id, resolved once from gold's own SNOMED code via
athena_concept -- same pattern as _LAB_TEST_ALIASES /
CORONARY_SEGMENT_TRAP_ABBREVIATIONS.

TWO DICTIONARIES, TWO DIFFERENT MATCH SHAPES.
  PHYSEXAM_HEADER_TERMS -- matched against a SECTION'S OWN NAME (exact,
    case-sensitive, from segment_sections()'s `name` field) -- these are
    the section/subsection labels themselves (Gen, Abd, HEENT, Mental
    Status, ...), annotated as entities in their own right by gold.
  PHYSEXAM_FINDING_TERMS -- matched via word-boundary scan WITHIN the BODY
    of a recognized physical-exam-family section (NAD, WNL, 'distress'/
    'edema'/'soft'/'S2'/..., the compressed NT/ND pair) -- free-text
    findings, not section labels, so scanning the header name alone would
    never find these; they need the section's body text.
  A section counts as "physical-exam-family" if its name is a key in
  PHYSEXAM_HEADER_TERMS, or its name_norm contains 'exam' (covers the
  un-subdivided case: one flat "Physical Exam:" section with no
  Gen:/Abd:/Ext: sub-headers at all).

INHERENTLY-NEGATED FINDING TERMS. 'NT' (non-tender) and 'ND' (non-
distended) code to their POSITIVE-finding concept (Abdominal tenderness /
Swollen abdomen) by gold's own convention -- SNOMED has no separate
"absence of X" concept for either, so the corpus relies on assertion
status to carry the negation. The generic cue-based assertion detector
(src.assertion) has no signal to find here: unlike "no distress" (an
explicit "no" cue word sitting right next to the span), the negation is
baked into the abbreviation's own letters with nothing nearby to detect.
Marked inherently_negated=True; src.clinical_pipeline force-sets
assertion_status=ABSENT for these AFTER the normal assertion pass runs,
overriding whatever (wrong, PRESENT-default) result the cue detector
produced, rather than leaving it to chance.

DOWNSTREAM WIRING (not in this module):
  src/clinical_pipeline.py  -- injects these as extracted_entities rows
    (skipping any span overlapping something GLiNER already found),
    running them through the same assertion tagging as everything else,
    then forcing ABSENT for inherently_negated terms.
  src/normalization/orchestrator.py -- recognizes entities carrying the
    `physexam_shorthand` marker this module's injected dicts set, and
    writes their normalized_entities row DIRECTLY from the pre-verified
    concept (match_tier "1 (Exact)", match_basis
    PHYSEXAM_SHORTHAND_MATCH_BASIS), skipping the Tier 1-3 search entirely
    -- these were selected by exact gold-mined text match, not similarity.
  src/mollm_tier_gate.py -- tier3_fast_path() recognizes
    PHYSEXAM_SHORTHAND_MATCH_BASIS and auto-validates without spending any
    model calls, the same bypass verified_brand_alias already gets.
"""
import re

# gliner_label deliberately avoids "Qualifier" -- src.mollm_tier_gate's
# qualifier_fragment_precheck() short-circuits any Qualifier-labeled entity
# straight to TIER_5 HITL with no model calls, which would silently defeat
# this module's whole point (getting these auto-validated via the Tier 3
# fast path).

# text (as it appears as a segment_sections() section NAME) ->
# (omop_concept_id, concept_name, omop_domain, gliner_label)
PHYSEXAM_HEADER_TERMS = {
    "HEENT": (4240345, "Physical examination", "Procedure", "Procedure"),
    "Physical Exam": (4240345, "Physical examination", "Procedure", "Procedure"),
    "Exam": (4240345, "Physical examination", "Procedure", "Procedure"),
    "Mental Status": (4190990, "Neurological mental status determination", "Procedure", "Procedure"),
    "Activity Status": (4030753, "Physical functional dependency", "Observation", "Symptom"),
    "Gen": (4036803, "General examination of patient", "Procedure", "Procedure"),
    "GEN": (4036803, "General examination of patient", "Procedure", "Procedure"),
    "General": (4036803, "General examination of patient", "Procedure", "Procedure"),
    "GENERAL": (4036803, "General examination of patient", "Procedure", "Procedure"),
    "Abd": (4075966, "Examination of abdomen", "Procedure", "Procedure"),
    "ABD": (4075966, "Examination of abdomen", "Procedure", "Procedure"),
    "Abdomen": (4075966, "Examination of abdomen", "Procedure", "Procedure"),
    "ABDOMEN": (4075966, "Examination of abdomen", "Procedure", "Procedure"),
    "Ext": (4121315, "Examination of limb", "Procedure", "Procedure"),
    "EXT": (4121315, "Examination of limb", "Procedure", "Procedure"),
    "Extremities": (4121315, "Examination of limb", "Procedure", "Procedure"),
    "EXTREMITIES": (4121315, "Examination of limb", "Procedure", "Procedure"),
    "Extrem": (4121315, "Examination of limb", "Procedure", "Procedure"),
    "EXTREM": (4121315, "Examination of limb", "Procedure", "Procedure"),
    "Neck": (4240345, "Physical examination", "Procedure", "Procedure"),
    "NECK": (4240345, "Physical examination", "Procedure", "Procedure"),
    "Neuro": (4225119, "Neurological examination", "Procedure", "Procedure"),
    "NEURO": (4225119, "Neurological examination", "Procedure", "Procedure"),
    "Lungs": (4149530, "Examination of respiratory system", "Procedure", "Procedure"),
    "LUNGS": (4149530, "Examination of respiratory system", "Procedure", "Procedure"),
    "Pulm": (4149530, "Examination of respiratory system", "Procedure", "Procedure"),
    "PULM": (4149530, "Examination of respiratory system", "Procedure", "Procedure"),
    "Resp": (4149530, "Examination of respiratory system", "Procedure", "Procedure"),
    "RESP": (4149530, "Examination of respiratory system", "Procedure", "Procedure"),
    "Pulmonary": (4149530, "Examination of respiratory system", "Procedure", "Procedure"),
    "Skin": (4150633, "Examination of skin", "Procedure", "Procedure"),
    "SKIN": (4150633, "Examination of skin", "Procedure", "Procedure"),
    "Cardiovascular": (4178669, "Cardiovascular physical examination", "Procedure", "Procedure"),
    "CV": (4178669, "Cardiovascular physical examination", "Procedure", "Procedure"),
    "CARDIAC": (4178669, "Cardiovascular physical examination", "Procedure", "Procedure"),
    "HEART": (4178669, "Cardiovascular physical examination", "Procedure", "Procedure"),
    "COR": (4178669, "Cardiovascular physical examination", "Procedure", "Procedure"),
    "Vitals": (4042138, "Vital signs finding", "Condition", "Symptom"),
    "VITALS": (4042138, "Vital signs finding", "Condition", "Symptom"),
    "Chest": (4149530, "Examination of respiratory system", "Procedure", "Procedure"),
}

# text (word-boundary scanned within a physical-exam-family section's BODY)
# -> (omop_concept_id, concept_name, omop_domain, gliner_label, inherently_negated)
PHYSEXAM_FINDING_TERMS = {
    "VS": (4042138, "Vital signs finding", "Condition", "Symptom", False),
    "RA": (36716647, "Breathing room air", "Condition", "Symptom", False),
    "BP": (4214962, "Blood pressure finding", "Condition", "Symptom", False),
    "HR": (4103189, "Finding of heart rate", "Condition", "Symptom", False),
    "RR": (4117286, "Finding of respiratory rate", "Condition", "Symptom", False),
    "NAD": (4085245, "No abnormality detected", "Condition", "Symptom", False),
    "WNL": (4085245, "No abnormality detected", "Condition", "Symptom", False),
    "soft": (4096862, "Abdomen soft", "Condition", "Symptom", False),
    "RRR": (4297303, "Normal heart rate", "Condition", "Symptom", False),
    "Temp": (4022230, "Body temperature finding", "Condition", "Symptom", False),
    "T": (4022230, "Body temperature finding", "Condition", "Symptom", False),
    "distress": (4239819, "Distress", "Condition", "Symptom", False),
    "edema": (433595, "Edema", "Condition", "Condition", False),
    "supple": (4179457, "Normal movement of neck", "Condition", "Symptom", False),
    "S2": (4008858, "Normal second heart sound S-2", "Condition", "Symptom", False),
    "O2 sat": (4020553, "Oxygen saturation measurement", "Measurement", "Lab Test", False),
    "Alert": (4086843, "Mentally alert", "Observation", "Symptom", False),
    "rebound": (4149024, "Rebound tenderness", "Condition", "Symptom", False),
    "guarding": (4091049, "Abdominal guarding", "Condition", "Symptom", False),
    "WWP": (605777, "Normal tissue perfusion", "Observation", "Symptom", False),
    "ND": (442597, "Swollen abdomen", "Condition", "Symptom", True),
    "MMM": (4170952, "Moist oral mucosa", "Condition", "Symptom", False),
    "JVD": (4154791, "Jugular venous engorgement", "Condition", "Symptom", False),
    "NT": (197981, "Abdominal tenderness", "Condition", "Symptom", True),
    "rashes": (140214, "Eruption", "Condition", "Condition", False),
    "oriented": (4092101, "Orientated", "Observation", "Symptom", False),
    "unremarkable": (4085245, "No abnormality detected", "Condition", "Symptom", False),
    "Unremarkable": (4085245, "No abnormality detected", "Condition", "Symptom", False),
    # 2026-08-18, round 2: 'general'/'room air' found as BODY TEXT, not a
    # segment_sections()-detected header -- confirmed live on note
    # 19895550-DS-7 ("general: ___ YO F in NAD"), where the lowercase
    # header doesn't match SECTION_HEADER_RE (needs a capitalized first
    # letter) and so never becomes its own section at all, leaving
    # 'general'/'Room air' sitting as plain text inside "Physical Exam"'s
    # own body. Added here (finding terms, body-scanned) rather than only
    # in PHYSEXAM_HEADER_TERMS, so it's caught either way regardless of
    # whether segment_sections() happens to detect it as a header in a
    # given note.
    "general": (4036803, "General examination of patient", "Procedure", "Procedure", False),
    "General": (4036803, "General examination of patient", "Procedure", "Procedure", False),
    "GENERAL": (4036803, "General examination of patient", "Procedure", "Procedure", False),
    "room air": (36716647, "Breathing room air", "Condition", "Symptom", False),
    "Room air": (36716647, "Breathing room air", "Condition", "Symptom", False),
    "rrr": (4297303, "Normal heart rate", "Condition", "Symptom", False),
    # 2026-08-18: verified against train_annotations.csv gold spans (33/33 =
    # 100%) -- "BS" (as in "+BS", positive bowel sounds on abdominal exam)
    # codes to SNOMED 61539000 specifically, not a generic bowel/breath
    # sounds concept. Word-boundary matching already handles the "+BS" shape
    # unchanged (the '+' is a non-word character, so \b still anchors before
    # "BS"). Raised during a review of dense-retrieval acronym failures;
    # confirmed via direct normalize_entity('BS', 'Symptom') call that bare
    # Tier 1-3 search (i.e. what runs when this cold-start dict doesn't
    # catch it) lands on 'BSG syndrome' -- a real, separate hallucination
    # this entry now bypasses entirely for physexam-family sections.
    "BS": (4263560, "Normal bowel sounds", "Condition", "Symptom", False),
}

# Same match_basis naming convention as verified_brand_alias /
# verified_lab_test_alias (src.normalization.tier_retrieval) -- a curated,
# pre-verified lookup, not a semantic-similarity guess.
PHYSEXAM_SHORTHAND_MATCH_BASIS = "verified_physexam_shorthand"

_TERM_RE_CACHE = {}


def _term_re(term: str):
    pat = _TERM_RE_CACHE.get(term)
    if pat is None:
        pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])")
        _TERM_RE_CACHE[term] = pat
    return pat


def is_physexam_family_section(section: dict) -> bool:
    if not section:
        return False
    if section["name"] in PHYSEXAM_HEADER_TERMS:
        return True
    return "exam" in section["name_norm"]


def find_physexam_shorthand_spans(raw_text: str, sections: list) -> list:
    """Scans every physical-exam-family section (see module docstring) for:
      (a) its own header name, if that name is in PHYSEXAM_HEADER_TERMS
          (matched against [header_start, start), the header text itself,
          not the body);
      (b) exact, case-sensitive, whole-token matches of
          PHYSEXAM_FINDING_TERMS within the section's BODY.

    Returns a list of dicts: {start, end, text, omop_concept_id,
    concept_name, omop_domain, gliner_label, inherently_negated}, offsets
    into raw_text. Overlap with already-extracted (e.g. GLiNER) spans is
    the CALLER's responsibility to filter -- this function only knows
    about the dictionary, not what else has already been found.
    """
    out = []
    for section in sections:
        if not is_physexam_family_section(section):
            continue

        header_info = PHYSEXAM_HEADER_TERMS.get(section["name"])
        if header_info:
            header_text = raw_text[section["header_start"]:section["start"]]
            m = _term_re(section["name"]).search(header_text)
            if m:
                concept_id, concept_name, domain, label = header_info
                out.append({
                    "start": section["header_start"] + m.start(),
                    "end": section["header_start"] + m.end(),
                    "text": m.group(0),
                    "omop_concept_id": concept_id,
                    "concept_name": concept_name,
                    "omop_domain": domain,
                    "gliner_label": label,
                    "inherently_negated": False,
                    "section_name": section["name"],
                })

        segment = raw_text[section["start"]:section["end"]]
        for term, (concept_id, concept_name, domain, label, negated) in PHYSEXAM_FINDING_TERMS.items():
            for m in _term_re(term).finditer(segment):
                out.append({
                    "start": section["start"] + m.start(),
                    "end": section["start"] + m.end(),
                    "text": m.group(0),
                    "omop_concept_id": concept_id,
                    "concept_name": concept_name,
                    "omop_domain": domain,
                    "gliner_label": label,
                    "inherently_negated": negated,
                    "section_name": section["name"],
                })
    out.sort(key=lambda d: d["start"])
    return out


_LOCAL_CONTEXT_WINDOW = 150


def build_physexam_shorthand_entities(raw_text: str, sections: list, note_id: str,
                                      existing_entities: list) -> list:
    """Full extracted_entities-shaped dicts for physical-exam shorthand spans
    not already covered by an existing (GLiNER-found) entity, ready to merge
    into src.clinical_pipeline.run_pipeline()'s `accepted` list.

    Deliberately does NOT reuse src.entity_extraction's expanded-text-
    coordinate helpers (find_sentence/build_local_context/
    sentence_ids_spanned) -- those operate on Stage 1's abbreviation-
    EXPANDED text offsets, and these spans are found directly against
    raw_text with no expansion involved (matched terms like 'NAD'/'HEENT'
    are already their own final form, nothing to expand). orig_start/
    orig_end and exp_start/exp_end are set equal for these entities --
    correct here specifically because there is no expansion step in
    between, not a general shortcut. local_context is a plain raw-text
    window rather than the sentence-bounded window Stage 1 builds: these
    entities are fast-pathed straight to AUTO_VALIDATED
    (src.mollm_tier_gate.tier3_fast_path) and never reach an LLM prompt,
    so nothing actually reads local_context's precise boundaries for them
    -- it exists here only for UI/audit display.

    Assertion status is computed via the SAME src.assertion.
    annotate_assertions() call GLiNER-extracted entities go through
    (called separately here, on raw_text + these spans specifically,
    since assertion detection is coordinate-agnostic), then
    inherently_negated terms (NT, ND) are force-overridden to ABSENT --
    see module docstring for why the generic cue detector can't find
    that negation on its own.
    """
    from src.entity_extraction import make_entity_id
    from src.assertion import annotate_assertions

    existing_spans = [(e["orig_start"], e["orig_end"]) for e in existing_entities]

    def _overlaps_existing(start, end):
        return any(not (end <= es or start >= ee) for es, ee in existing_spans)

    candidates = [c for c in find_physexam_shorthand_spans(raw_text, sections)
                 if not _overlaps_existing(c["start"], c["end"])]
    if not candidates:
        return []

    # 2026-08-18 (round 2 negation-scope fix). Calling annotate_assertions()
    # on the WHOLE raw_text let spaCy's sentence segmenter treat multiple
    # unrelated colon/newline-delimited telegraphic clauses as one giant
    # run-on "sentence" -- confirmed live on note 19895550-DS-7's
    # "COR: RRR S1, S2\nabd: soft, NT, ND, +BS\nextrem: no edema": the "no"
    # belonging to "no edema" bled BACKWARD across two unrelated lines,
    # marking COR/RRR/S2/soft ABSENT (wrong) while 'edema' itself -- the
    # term actually next to "no" -- came back PRESENT (also wrong). Scoping
    # each call to a single LINE contains negation scope to where it
    # actually applies in this notation style, matching how a human reader
    # parses "Abd: S/NT/ND" line by line, not as continuous prose.
    # Default PRESENT/PATIENT/CURRENT -- covers header-token entities that
    # land before the stripped colon (see the "safe" filter below) and are
    # never negated anyway, without ever leaving a None to crash on.
    _default_assertion = {
        "assertion_status": "PRESENT", "experiencer": "PATIENT", "temporality": "CURRENT",
        "assertion_cue": None, "assertion_cue_start": None, "assertion_cue_end": None,
        "assertion_cue_category": None, "assertion_engine": "physexam_shorthand_header_default",
    }
    assertions = [_default_assertion] * len(candidates)
    by_line = {}
    for idx, c in enumerate(candidates):
        line_start = raw_text.rfind("\n", 0, c["start"]) + 1
        line_end = raw_text.find("\n", c["end"])
        if line_end == -1:
            line_end = len(raw_text)
        by_line.setdefault((line_start, line_end), []).append(idx)

    for (line_start, line_end), idxs in by_line.items():
        line_text = raw_text[line_start:line_end]

        # A colon confirmed to break medspacy/pyConText's negation-trigger
        # matching entirely -- "extrem: no edema" -> PRESENT (wrong),
        # "no edema" (colon-and-header stripped) -> ABSENT (right), tested
        # directly against this exact pipeline. Telegraphic physical-exam
        # notation is essentially ALL "Header: findings" shaped, so this
        # isn't a rare edge case here -- strip up to and including the
        # FIRST colon (if any) before running assertion detection, and
        # shift span offsets to match. Only the assertion-detection call
        # uses this stripped text; entity offsets/local_context still use
        # the real raw_text coordinates throughout.
        colon_idx = line_text.find(":")
        strip_offset = colon_idx + 1 if colon_idx != -1 else 0
        detect_text = line_text[strip_offset:]

        line_spans = [(candidates[i]["start"] - line_start - strip_offset,
                      candidates[i]["end"] - line_start - strip_offset,
                      candidates[i]["gliner_label"]) for i in idxs]
        # A span landing BEFORE the stripped prefix (the header token
        # itself, e.g. 'Ext' in "Ext: WNL") can't be assertion-checked
        # against the stripped text -- skip those (default PRESENT is
        # correct for header entities regardless, they're never negated).
        safe = [(i, s) for i, s in zip(idxs, line_spans) if s[0] >= 0]
        if not safe:
            continue
        line_assertions = annotate_assertions(detect_text, [s for _, s in safe])
        for (i, _), a in zip(safe, line_assertions):
            assertions[i] = a

    out = []
    for c, assertion in zip(candidates, assertions):
        start, end = c["start"], c["end"]
        ctx_start = max(0, start - _LOCAL_CONTEXT_WINDOW)
        ctx_end = min(len(raw_text), end + _LOCAL_CONTEXT_WINDOW)

        if c["inherently_negated"]:
            assertion_status = "ABSENT"
            assertion_cue, assertion_engine = None, "physexam_shorthand_inherent_negation"
        else:
            assertion_status = assertion["assertion_status"]
            assertion_cue, assertion_engine = assertion["assertion_cue"], assertion["assertion_engine"]

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
            "assertion_status": assertion_status,
            "experiencer": assertion["experiencer"],
            "temporality": assertion["temporality"],
            "assertion_cue": assertion_cue,
            "assertion_cue_start": assertion.get("assertion_cue_start"),
            "assertion_cue_end": assertion.get("assertion_cue_end"),
            "assertion_cue_category": assertion.get("assertion_cue_category"),
            "assertion_engine": assertion_engine,
            "section_name": c.get("section_name"),
            "sentence_id": None,
            "local_context": raw_text[ctx_start:ctx_end],
            "local_context_basis": "physexam_shorthand_raw_window",
            "expansion_ambiguous": False,
            "candidate_expansions": None,
            "selection_basis": None,
            "gliner_model_version": "physexam_shorthand_cold_start",
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
            # Consumed by src.normalization.orchestrator.process_and_normalize_entities()
            # to bypass the Tier 1-3 search entirely for this entity.
            "physexam_shorthand": {
                "omop_concept_id": c["omop_concept_id"],
                "concept_name": c["concept_name"],
                "omop_domain": c["omop_domain"],
            },
        })
    return out
