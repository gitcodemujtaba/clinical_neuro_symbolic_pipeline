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

2026-09-01 EXTENSION -- eleven more terms, ranks 26-50 of the same
mining (`evaluation/mine_gliner_misses.py --top-n 50`, train-split only,
`logs/gliner_miss_report_top50.json`). Ranks 1-25 were already fully
triaged when this module was first built; this extension re-applies BOTH
required bars to every new candidate, not just the consistency-when-
tagged one:

  1. Gold-concept-consistency-when-tagged >=95% (same bar as the
     original 13).
  2. TAG RATE (extraction-worthiness-whenever-present): of every real,
     case-insensitive whole-word/phrase occurrence of the term in the
     actual train-split note text, what fraction does gold ALSO tag as
     this entity? A term can pass bar 1 while still being a common
     narrative word gold rarely tags (exactly the `interactive`/
     `evaluation`/`surgery`/`soft` trap already documented above) --
     bar 2 catches that empirically instead of by guessing.

Eleven cleared both bars (tag rate / consistency, all measured against
real train-split occurrences):

  totbili           -> 359986008 'Bilirubin, total measurement'   (100%/100%, n=67)
  perrl             -> 386666001 'Pupils equal and reacting to light' (100%/100%, n=56)
  rubs              -> 7036007   'Pericardial friction rub'       (98.1%/100%, n=53)
  well perfused     -> 1137685003 'Normal tissue perfusion'       (100%/100%, n=51)
  sclera anicteric  -> 427801009 'White sclera'                   (100%/100%, n=49)
  gallops           -> 2170000   'Gallop rhythm'                  (98.0%/100%, n=50)
  crackles          -> 48409008  'Respiratory crackles'           (100%/100%, n=49)
  rhonchi           -> 24612001  'Wheeze - rhonchi'                (100%/100%, n=46)
  cyanosis          -> 3415004   'Cyanosis'                        (100%/100%, n=43)
  vitals            -> 118227000 'Vital signs finding'            (93.2%/99.1%, n=118)
  ph                -> 81065003  'pH measurement'                 (90.8%/100%, n=65)

SIX of the 18 candidates checked at ranks 26-50 FAILED and were excluded,
same discipline as ranks 1-25's exclusions -- not silently dropped:

  k          -- tag rate 21.4% (248 real occurrences, only 53 are the
                entity) -- a single letter is inherently overloaded;
                worst tag rate of any candidate checked.
  mcv        -- tag rate 35.7% (280 occurrences, 100 tagged) -- collides
                with other real uses far more often than it's the lab
                value.
  urine      -- consistency only 39.2% -- gold maps it to too many
                different concepts depending on context, same failure
                shape as the original `pain`/`normal`/`stable` exclusions.
  infection  -- tag rate 95.1% (extraction-worthy) but consistency only
                49.6% -- genuinely multi-concept even when it IS the
                entity gold wanted.
  bleeding   -- tag rate 99.0% but consistency only 57.8% -- same
                multi-concept problem as `infection`.
  erythema   -- consistency 86.4%, just under the 95% bar.

`ph` carries one honest caveat worth flagging even though it cleared
both bars: it's a 2-character token, and the WORD BOUNDARY regex
(`\bph\b`) is what keeps it from matching inside "morph"/"graph" -- but
"PH"/"pH" has real-world alternate readings (pulmonary hypertension,
public health) this specific train-split corpus's measured 90.8%/100%
happened not to surface. Included on the strength of the actual
measurement (same standard `glucose` was excluded by, just the other
direction), not asserted risk-free by inspection alone -- worth
re-checking if a live validation run ever shows it misfiring.

STANDING SELECTION CRITERIA -- for gold today, for KG3 later. Every term
in this file, old and new, was vetted the same way: gold's own
annotations stand in for what a mature KG3 (real confirmed clinician
review) would eventually supply, since KG3 today holds zero real review
volume (see docs/KG3_Implementation_And_Feedback_Loop_Technical_
Reference.md). The two-bar rule above (>=95% consistency-when-tagged,
AND an empirically-measured extraction-worthiness/tag-rate check, not a
plausibility guess) is deliberately written as a METHOD, not a
gold-specific procedure, so the exact same two checks apply unchanged
the day KG3 has enough real, human-confirmed volume to mine from
instead of gold. One addition required specifically for a KG3-sourced
run, not needed for this gold-based one: a minimum REOCCURRENCE-COUNT
gate before trusting a mined term at all (the same discipline already
proven for `acronym_priors`' `MIN_CACHE_HIT_COUNT=2` and the abbreviation
flywheel's `VERIFIED_ALLOW_LIST`) -- gold is a fixed, already-adjudicated
dataset where a single measured tag-rate is trustworthy on its own; a
live KG3 mining pass would be reading the pipeline's OWN accumulating
decisions, which can self-reinforce a wrong pattern the same way
`compute_frequency_priority()`'s first, block-list-based version did
(7/7 wrong on first real-data check) before that mechanism was inverted
to an allow-list. A future KG3-sourced version of this vetting should
therefore require BOTH bars above AND independent reconfirmation across
multiple distinct real review events before promoting any candidate,
not a single mining pass's own numbers.
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
    # 2026-09-01 extension (11 terms, ranks 26-50 of the same mining --
    # see module docstring's "2026-09-01 EXTENSION" section for the real
    # tag-rate/consistency numbers each one cleared). All plain whole-
    # word/phrase, case-insensitive, no context gate needed -- same
    # "technical vocabulary with no viable everyday-English reading"
    # reasoning as eos/monos/rdwsd/cxr above, empirically confirmed via
    # each term's own measured tag rate rather than assumed by analogy.
    "totbili": (re.compile(r"\btotbili\b", re.IGNORECASE), "Lab Test", _always),
    "perrl": (re.compile(r"\bperrl\b", re.IGNORECASE), "Condition", _always),
    "rubs": (re.compile(r"\brubs\b", re.IGNORECASE), "Condition", _always),
    "well perfused": (
        re.compile(r"\bwell\s+perfused\b", re.IGNORECASE), "Symptom", _always),
    "sclera anicteric": (
        re.compile(r"\bsclera\s+anicteric\b", re.IGNORECASE), "Condition", _always),
    "gallops": (re.compile(r"\bgallops\b", re.IGNORECASE), "Condition", _always),
    "crackles": (re.compile(r"\bcrackles\b", re.IGNORECASE), "Condition", _always),
    "rhonchi": (re.compile(r"\brhonchi\b", re.IGNORECASE), "Condition", _always),
    "cyanosis": (re.compile(r"\bcyanosis\b", re.IGNORECASE), "Condition", _always),
    "vitals": (re.compile(r"\bvitals\b", re.IGNORECASE), "Condition", _always),
    # See the docstring's honest caveat on this one -- included on the
    # strength of its own 90.8%/100% measurement, not risk-free by
    # inspection (a 2-char token, real-world alternate readings exist).
    "ph": (re.compile(r"\bph\b", re.IGNORECASE), "Lab Test", _always),
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
