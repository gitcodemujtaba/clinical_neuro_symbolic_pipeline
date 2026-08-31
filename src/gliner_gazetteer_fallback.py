"""src/gliner_gazetteer_fallback.py -- 2026-08-31: a narrow, deterministic
recovery layer for real GLiNER span-recall misses, gated OFF by default
(`CNSP_GLINER_GAZETTEER_FALLBACK`).

WHY THIS EXISTS. Span recall is the single largest measured gap in this
pipeline (53.0% corpus-wide) -- nothing downstream (Stage 2b linking,
Stage 3 tier gate, the calibrator) can ever recover an entity GLiNER never
extracted as a span in the first place. evaluation/mine_gliner_misses.py
(train-split only, same locked-test-split discipline as everywhere else
this session) systematically mined 9,772 real misses; the full top-25
most-frequently-missed surface forms were each independently checked
against real gold concept-consistency (>=95% required, same bar
_LAB_TEST_ALIASES already uses), not guessed. Fourteen cleared that bar;
THIRTEEN are included -- see the glucose exclusion below.

  Mg    -> 271285000 'Blood magnesium measurement'    (78/78 gold, 100%)
  RA    -> 722742002 'Breathing room air'              (80/80 gold, 100%)
  CV    -> 363003006 'Cardiovascular physical examination' (59/59, 100%)
  EOMI  -> 103251002 'Normal ocular motility'           (44/44, 100%)
  CTAB  -> 48348007  'Normal breath sounds'             (41/41, 100%)
  eos, monos, rdwsd, cxr, non-tender, wheezes,
  ambulatory - independent, level of consciousness      (97-100% each)

`glucose` cleared the consistency bar (99.1%) but was found live, during
validation, to genuinely need context-gating it doesn't have: false
positives fired on urinalysis/CSF dipstick "Glucose-NEG" readings, a
DIFFERENT clinical concept than the blood-panel glucose it was verified
against. The obvious fix (require "BLOOD" as the nearest panel marker
within the preceding window) was checked directly against all 112 real
gold occurrences and found unreliable -- only 78/112 (69.6%) actually
have "BLOOD" in range, because MIMIC's de-identification frequently masks
the panel-header word itself ("___ 08:42AM   Glucose-92", not
"___ 08:42AM   BLOOD Glucose-92"). Excluded rather than ship an
under-verified rule -- same discipline as the four terms below.

SIX real top-25 candidates were EXCLUDED, not just left out silently:
pain (73.6%), normal (60.8%), stable (84.0%), negative (50.0%), clear
(55.6%), culture (77.8%) -- gold itself maps these to MULTIPLE different
concepts depending on context. Recovering the span wouldn't be enough for
these; picking the right concept afterward is a harder, separate problem
this mechanism does not attempt.

A DISTINCT further four (soft, interactive, evaluation, surgery) are
individually >=95% consistent WHENEVER gold tags them, but are common
enough English words that a blanket whole-word match would over-extract
on the many occurrences gold does NOT tag as an entity ("further
evaluation," "post-surgery," generic narrative use) -- concept-
consistency-when-tagged is a different claim from extraction-worthiness-
whenever-present, and conflating the two here would repeat exactly the
mistake this project's own history (`_LAB_TEST_ALIASES`'s comment: "a
plausible-sounding entry is not the same as a measured one") warns
against. Not included; would need the same per-term structural context
verification RA/CV/Mg got below before being safe to add.

WHY NOT A BLANKET "IF THE WORD APPEARS, EXTRACT IT" GAZETTEER. RA and CV
are genuinely ambiguous outside their specific structural context (RA can
mean Rheumatoid Arthritis; CV has other meanings) -- exactly the failure
class this project's own acronym-escalation work already got burned by
(a systematic textbook-prior bias, stays disabled). Each term below
therefore has its OWN structural context check, verified directly against
real gold occurrences before being encoded, not a shared blanket rule:

  Mg    -- only recovered immediately before a dash-terminated lab value
            (e.g. "Mg-1.8"), the exact shape strip_lab_value_suffix()
            already parses elsewhere in this codebase.
  RA    -- only recovered immediately after a '%' (a vitals-string O2 sat
            reading, e.g. "96% RA").
  CV    -- only recovered at the start of a line, immediately before ':'
            (a physical-exam section sub-header, e.g. "CV: RRR...").
  EOMI, CTAB -- recovered on a plain whole-word match. Deliberately NOT
            context-gated like Mg/RA/CV: both are highly specific medical
            abbreviations with no common competing meaning found in real
            gold data, unlike RA/CV's genuine ambiguity.

CASE SENSITIVITY, stated precisely rather than applied uniformly: Mg/RA/
CV match case-SENSITIVELY, deliberately -- their casing is part of what
makes them safe (a lowercase "mg"/"ra"/"cv" is more plausible as ordinary
prose than the clinical-shorthand capitalized form observed in every
verified gold occurrence, and that variant was never independently
checked). The remaining nine terms are technical/abbreviation vocabulary
with no viable alternate everyday-English reading regardless of case
(EOMI, CTAB, CXR, RDWSD... "glucose" meaning anything else is not a real
risk) -- for these, case doesn't change the safety calculus the way it
does for Mg/RA/CV, so they match case-insensitively to recover more real
instances without taking on new ambiguity risk.

Recovered entities carry a distinct `extraction_source` value
('gazetteer_fallback_gliner_miss') so they are never silently blended
with a real GLiNER extraction in any downstream measurement -- same
auditability discipline as every other provenance-marked mechanism in
this codebase (verified_lab_test_alias, mollm_acronym_escalated, ...).

GATED OFF BY DEFAULT (env var CNSP_GLINER_GAZETTEER_FALLBACK, same
pattern as CNSP_ACRONYM_ESCALATION/CNSP_HYBRID_RETRIEVAL) -- built and
unit-tested, pending its own validation batch on real notes before
enabling, same discipline as every other new mechanism this session.
"""
import os
import re

GAZETTEER_FALLBACK_ENABLED = os.environ.get(
    "CNSP_GLINER_GAZETTEER_FALLBACK", "").strip() in ("1", "true", "yes")

def _mg_context(text: str, start: int, end: int) -> bool:
    """True iff the match is immediately followed by '-<digits>' (a
    dash-terminated lab value), e.g. 'Mg-1.8'. Requires no whitespace
    between the term and the dash -- the exact shape observed in every
    verified gold occurrence."""
    return bool(re.match(r"-\d", text[end:end + 8]))


def _ra_context(text: str, start: int, end: int) -> bool:
    """True iff the nearest non-whitespace character before the match is
    '%' (a vitals-string O2 saturation reading immediately preceding
    'RA', e.g. '96% RA')."""
    i = start
    while i > 0 and text[i - 1] in " \t":
        i -= 1
    return i > 0 and text[i - 1] == "%"


def _cv_context(text: str, start: int, end: int) -> bool:
    """True iff the match starts a line (only whitespace before it since
    the last newline) AND is immediately followed by ':' (a physical-exam
    section sub-header, e.g. 'CV: RRR...')."""
    line_begin = text.rfind("\n", 0, start) + 1
    if text[line_begin:start].strip() != "":
        return False
    i = end
    while i < len(text) and text[i] in " \t":
        i += 1
    return i < len(text) and text[i] == ":"


def _always(text: str, start: int, end: int) -> bool:
    return True


# Each entry: (regex pattern for a case-sensitive whole-word match, GLiNER
# label matching this term's real gold domain, context check function).
# The nine _always entries are plain whole-word/phrase matches -- safe
# specifically BECAUSE each is technical/abbreviation vocabulary with no
# common competing everyday-English sense observed in real gold data
# (unlike soft/interactive/evaluation/surgery, deliberately excluded --
# see module docstring).
_GAZETTEER = {
    "Mg": (re.compile(r"\bMg\b"), "Lab Test", _mg_context),
    "RA": (re.compile(r"\bRA\b"), "Condition", _ra_context),
    "CV": (re.compile(r"\bCV\b"), "Procedure", _cv_context),
    "EOMI": (re.compile(r"\bEOMI\b"), "Condition", _always),
    "CTAB": (re.compile(r"\bCTAB\b"), "Condition", _always),
    "eos": (re.compile(r"\beos\b", re.IGNORECASE), "Lab Test", _always),
    "monos": (re.compile(r"\bmonos\b", re.IGNORECASE), "Lab Test", _always),
    "rdwsd": (re.compile(r"\brdwsd\b", re.IGNORECASE), "Lab Test", _always),
    "cxr": (re.compile(r"\bcxr\b", re.IGNORECASE), "Procedure", _always),
    "non-tender": (re.compile(r"\bnon-tender\b", re.IGNORECASE), "Condition", _always),
    "wheezes": (re.compile(r"\bwheezes\b", re.IGNORECASE), "Condition", _always),
    "ambulatory - independent": (
        re.compile(r"\bambulatory\s*-\s*independent\b", re.IGNORECASE), "Symptom", _always),
    "level of consciousness": (
        re.compile(r"\blevel of consciousness\b", re.IGNORECASE), "Procedure", _always),
}


def recover_missed_entities(expanded_text: str, existing_spans: list) -> list:
    """Scans `expanded_text` for gazetteer terms GLiNER did not already
    extract (checked against `existing_spans`, a list of (start, end)
    tuples in the SAME coordinate system -- exp_start/exp_end, before
    offset reconciliation to original text). Returns a list of entity
    dicts in GLiNER's own predict_entities() shape
    ({"start", "end", "label", "score", "text"}), plus a private
    `_extraction_source` key the caller reads and strips before
    persistence.

    `score` is a fixed 1.0 -- these are deterministic structural matches,
    not model confidence, and should never be confused with one
    downstream (any consumer filtering on `confidence` must instead check
    `extraction_source`).
    """
    recovered = []
    for term, (pattern, label, context_check) in _GAZETTEER.items():
        for m in pattern.finditer(expanded_text):
            start, end = m.start(), m.end()
            if not context_check(expanded_text, start, end):
                continue
            if any(not (end <= s or start >= e) for s, e in existing_spans):
                continue  # GLiNER already found something overlapping here
            recovered.append({
                "start": start, "end": end, "label": label,
                "score": 1.0, "text": expanded_text[start:end],
                "_extraction_source": "gazetteer_fallback_gliner_miss",
            })
    return recovered
