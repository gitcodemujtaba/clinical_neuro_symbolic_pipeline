"""
src/assertion.py — clinical assertion detection (Stage 2a sub-component).

NEW 2026-08-08. Closes the severity-1 gap in
docs/Stage1_2_Completeness_Audit.md: the pipeline had no notion of whether an
extracted entity was actually asserted about the patient.

WHY THIS EXISTS. Measured over the 75,491 gold annotations in the DrivenData
set, ~20.9% of annotated spans sit in a non-assertive context -- 14.2% behind
a negation cue (`denies`, `no`, `without`, `ruled out`), 4.3% historical, 2.0%
hypothetical, 1.1% about a family member. Without this module, a note reading
"denies chest pain" produces an entity `chest pain` -> OMOP 29857009, and
Stage 3 is then asked, in complete sincerity, whether the patient's chest pain
contradicts the ACS guidelines. Nothing in the schema could have told it
otherwise. The same defect would write negated and family-history findings
into KG 3 as asserted patient facts, which then feed the Stage 5 active
learning loop as pseudo-labels.

WHY RULE-BASED, NOT THE LLM. Negation inverts clinical meaning, so a missed
negation does not degrade a Stage 3 verdict gracefully -- it flips it. Leaving
that to LLM inference over the context window is precisely the black-box
judgement this pipeline exists to replace with deterministic symbolic signal,
and it would make a safety-critical property unauditable. medspaCy's ConText
implementation is rule-based and deterministic, and it reports the cue that
fired, so every assertion decision is reviewable by a human in the HITL queue.

EVALUATION CAVEAT worth carrying into the dissertation: the DrivenData gold
set annotates spans irrespective of assertion status, so span-level and
concept-level F1 are BLIND to errors made here. This component cannot be
validated by the main evaluation metric and needs its own manually-reviewed
sample. That the headline metric cannot see this error class is a reason to
take it more seriously, not less.

API STABILITY NOTE: medspaCy's ConText component has been exposed under
several names across releases (`ConTextComponent`, `ConText`, and the
registered pipe name `medspacy_context`). _build_pipeline() probes the known
names in order and, if none resolve, reports which context-like factories ARE
registered instead of letting spaCy raise a bare E002 that lists only its own
built-ins.

VERIFIED on the EC2 host 2026-08-08 (medspacy 1.3.1, spacy 3.7.5, Python 3.9).
Installation does not disturb scispacy. Two things had to be fixed to get
there, both recorded because they are the kind that recur:
  1. `add_pipe("medspacy_context")` fails with E002 unless medspacy is
     imported first -- spaCy registers factories as an import side effect.
  2. ConTextModifier has no `.span`; cues come from `.modifier_span` token
     indices plus `.category` (see _modifier_details).

Confirmed behaviour on a synthetic multi-case sentence:
    "denies any chest pain"  -> ABSENT  / cue "denies" / NEGATED_EXISTENCE
    "Mother had a stroke"    -> FAMILY  / cue "Mother" / FAMILY
    "History of MI in 2019"  -> HISTORICAL / cue "History" / HISTORICAL
    "Patient has COPD"       -> PRESENT / no cue (correctly, no modifier)

Still unmeasured: accuracy on real discharge notes at scale. Check the
`assertion_engine` column after a pipeline run -- `unavailable_default` means
this module never ran and every entity was silently marked PRESENT.
"""

import re
import warnings

warnings.filterwarnings("ignore")

# Assertion vocabulary. Deliberately mirrors the ConText problem framing
# (negation / temporality / experiencer / certainty) rather than inventing a
# scheme, so the fields map onto published clinical NLP conventions and are
# recognisable to an examiner.
STATUS_PRESENT = "PRESENT"
STATUS_ABSENT = "ABSENT"
STATUS_POSSIBLE = "POSSIBLE"
STATUS_CONDITIONAL = "CONDITIONAL"

EXPERIENCER_PATIENT = "PATIENT"
EXPERIENCER_FAMILY = "FAMILY"
EXPERIENCER_OTHER = "OTHER"

TEMPORALITY_CURRENT = "CURRENT"
TEMPORALITY_HISTORICAL = "HISTORICAL"

_nlp = None
_context = None
_segmenter_used = None



def _silence_pyrush():
    """Turns off PyRuSH's loguru DEBUG stream.

    PyRuSH logs a line PER TOKEN at DEBUG level ("Token 523 marked as sentence
    start", "GAP DETECTED ..."). On a 10k-character note that is thousands of
    lines to stdout, which buries the pipeline's own output and slows the run
    for no diagnostic benefit. loguru is configured independently of the
    `warnings` filter and of Python's logging module, so neither of the
    suppressions already in this file affects it -- it needs disabling by name.
    """
    try:
        from loguru import logger
        logger.disable("PyRuSH")
    except Exception:
        pass


def _build_pipeline():
    """Lazily builds a minimal spaCy pipeline carrying medspaCy's ConText.

    Kept minimal on purpose: a blank English pipeline plus a sentencizer plus
    ConText. ConText scopes its cues to sentence boundaries, so the sentencizer
    is required -- without it a negation in one sentence would leak across into
    the next, which is a classic and quiet source of false negations.

    We do NOT use medspacy.load()'s full default pipeline because that includes
    a TargetMatcher intended to find entities itself; here the entities are
    already known (GLiNER found them) and we only want the modifier logic
    applied to those spans.
    """
    global _nlp, _context
    if _context is not None:
        return _nlp, _context

    import spacy

    # THIS IMPORT IS LOAD-BEARING, not a formality. spaCy factories are
    # registered as a side effect of importing the module that declares them
    # (@Language.factory runs at import time), so `import spacy` alone leaves
    # medspaCy's components invisible and add_pipe("medspacy_context") fails
    # with E002 "Can't find factory". Confirmed on the EC2 host 2026-08-08:
    # medspacy 1.3.1 installed correctly, and the factory was still missing
    # until medspacy itself was imported. An earlier version of this function
    # relied on catching that failure and importing medspacy in the except
    # branch, which happened to work but made a required import look optional.
    try:
        import medspacy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "medspaCy is not installed. Install with:\n"
            "    pip install \"medspacy>=1.0.0\" \"spacy>=3.7.0,<3.8.0\"\n"
            "(constrain spacy in the same command -- scispacy requires <3.8.0 and "
            "pip will otherwise resolve a spacy that breaks it)."
        ) from exc

    _nlp = spacy.blank("en")

    # PyRuSH, NOT spaCy's sentencizer. ConText scopes its cues to sentence
    # boundaries, so sentence segmentation directly determines how far a
    # negation reaches -- and the plain sentencizer splits only on `.!?`, which
    # clinical notes barely use.
    #
    # MEASURED FAILURE on note 10000032-DS-21 (2026-08-08): with the
    # sentencizer, the physical-exam and laboratory block collapsed into ONE
    # 1,908-character sentence. A single "no" from "no murmurs, rubs, gallops"
    # then scoped over every lab value in it, marking GLUCOSE-109, UREA N-25,
    # CREAT-0.3, HGB-14 and MCV-99 as ABSENT -- measured results reported as
    # negated findings. That is a systematic false-negation across an entire
    # category of entity, not an edge case.
    #
    # PyRuSH (medspaCy's own segmenter, installed as one of its dependencies)
    # is built for clinical layout: it treats newlines, section headers and
    # list structure as boundaries rather than requiring terminal punctuation.
    # Falls back to the sentencizer if unavailable, because no segmentation at
    # all would be worse -- but the fallback is recorded on every record via
    # assertion_engine so a run using it is identifiable rather than silently
    # degraded.
    global _segmenter_used
    _silence_pyrush()
    try:
        _nlp.add_pipe("medspacy_pyrush")
        _segmenter_used = "pyrush"
    except Exception:
        _nlp.add_pipe("sentencizer")
        _segmenter_used = "sentencizer_fallback"

    # Tried in order of preference. medspaCy has exposed ConText under more
    # than one factory name across releases, so this probes rather than
    # assuming, and reports what IS available if none match -- an E002 listing
    # only spaCy's built-ins gives no clue that the real problem is a renamed
    # third-party factory.
    last_error = None
    for factory_name in ("medspacy_context", "context", "medspacy_pyrush_context"):
        try:
            _context = _nlp.add_pipe(factory_name)
            break
        except Exception as exc:
            last_error = exc
    else:
        try:
            from spacy.util import registry
            available = sorted(
                n for n in registry.factories.get_all() if "context" in n.lower()
            )
        except Exception:
            available = ["<could not enumerate>"]
        raise RuntimeError(
            "medspaCy is installed but no known ConText factory name resolved. "
            f"Context-like factories currently registered: {available}. "
            "Add the correct name to the tuple in _build_pipeline(). "
            f"Last error: {last_error}"
        )

    return _nlp, _context


def _flag(span, name: str) -> bool:
    """Reads a medspaCy Span extension defensively.

    Returns False when the extension is absent rather than raising, so a
    medspaCy version that drops or renames one flag degrades that single signal
    instead of breaking the whole pipeline. Absence is logged by the caller via
    `assertion_engine`, so a silently-missing flag is still visible in
    provenance rather than being indistinguishable from a real negative.
    """
    try:
        return bool(getattr(span._, name))
    except (AttributeError, KeyError):
        return False


# Structured laboratory-result patterns. See is_structured_result().
_LAB_NAME_VALUE = re.compile(r"^[A-Za-z][A-Za-z0-9 /%()-]*?-[<>]?\d", re.I)
_LAB_PANEL_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9]{1,14}-[<>]?\d+(\.\d+)?\*?")

# 2026-08-11: delimiter-less variants of the two patterns above. Measured via
# scripts/measure_heuristic_and_boundary.py's alnum_mix check (5.04% of the
# 25-note test split): vitals/labs like "RR18" and heart-sound findings like
# "S1S2" glue name and value directly together with NO hyphen at all, which
# _LAB_NAME_VALUE's required "-" cannot match -- confirmed these fall
# through both existing regexes on note 10043750-DS-6 and 10848570-DS-12.
#
# Bounded to a 1-4 letter prefix per repetition, requiring the string to END
# in a digit run. That bound alone is not enough to rule out every
# non-lab shorthand -- "AOx3" ("alert and oriented x3") also matches this
# shape ("AOx" + "3") despite not being a lab value. This is the same
# ambiguity _LAB_NAME_VALUE already has with "COVID-19", and it is solved
# the same way: is_structured_result() only applies signal (1) when
# gliner_label == "Lab Test", so an "AOx3" entity extracted under any other
# label is untouched by this pattern, exactly as "denies COVID-19" is
# untouched by _LAB_NAME_VALUE today.
_LAB_NAME_VALUE_NO_DELIM = re.compile(r"^(?:[A-Za-z]{1,4}\d+(?:\.\d+)?)+$")
_LAB_PANEL_TOKEN_NO_DELIM = re.compile(r"\b[A-Za-z]{1,4}\d+(?:\.\d+)?\*?\b")


def is_structured_result(entity_text: str, context_text: str,
                         gliner_label: str = None) -> bool:
    """True when the entity is part of a structured lab-result line.

    WHY THIS EXISTS. ConText is a model of clinical NARRATIVE: it finds a cue
    word and propagates its effect across the surrounding prose. Laboratory
    panels in MIMIC are not prose -- they are tables rendered as text:

        ___ 10:25PM   GLUCOSE-109* UREA N-25* CREAT-0.3* SODIUM-138
        POTASSIUM-3.4 CHLORIDE-105 TOTAL carbon dioxide-27 ANION GAP-9

    There is no assertion to detect here. Every value listed was measured; none
    is negated, hypothetical or about a relative. Running negation detection
    over it can only produce false negatives, and measured on note
    10000032-DS-21 (2026-08-08) it did exactly that: a ': No' cue elsewhere in
    the same PyRuSH sentence marked GLUCOSE-109, UREA N-25, CREAT-0.3, HGB-14,
    PTT-30, SGPT and SGOT as ABSENT -- 15+ measured results reported as negated
    findings.

    Better sentence segmentation (PyRuSH) reduced this but could not remove it,
    because the problem is not sentence length: it is that assertion logic does
    not apply to tabular data at all. So these entities are marked PRESENT with
    a distinct assertion_engine value rather than being run through ConText and
    then trusted.

    Two independent signals, either sufficient:
      1. The entity itself is a NAME-VALUE token, hyphenated (`CREAT-0.3`,
         `WBC-5`) or delimiter-less (`RR18`, `S1S2` -- 2026-08-11, see
         _LAB_NAME_VALUE_NO_DELIM above).
      2. Its surrounding line contains three or more such tokens (either
         form) -- which catches entities like `SGPT` or `ANION GAP` that
         carry no value in their own span but sit inside an obvious panel.

    Signal (1) is GATED ON THE `Lab Test` LABEL, for BOTH the hyphenated and
    delimiter-less forms. Without that gate the hyphenated pattern fires on
    "COVID-19" and the delimiter-less pattern fires on "AOx3" -- neither is a
    lab value, and forcing either to PRESENT would break "denies COVID-19" /
    "not AOx3", turning a correct negation into a false positive assertion.
    That is the direction of error that reaches KG3, so the gate matters for
    both forms equally. Signal (2) needs no gate: three or more NAME-VALUE
    tokens in one line is a panel regardless of how any one entity in it was
    labelled, which is what catches `SGPT` and `ANION GAP` sitting inside a
    panel without values of their own -- and now also catches delimiter-less
    panels like "RR18  SpO2  BP110/70".
    """
    if entity_text and gliner_label == "Lab Test":
        stripped = entity_text.strip()
        if _LAB_NAME_VALUE.match(stripped) or _LAB_NAME_VALUE_NO_DELIM.match(stripped):
            return True
    if context_text:
        panel_hits = (len(_LAB_PANEL_TOKEN.findall(context_text))
                     + len(_LAB_PANEL_TOKEN_NO_DELIM.findall(context_text)))
        if panel_hits >= 3:
            return True
    return False


def _default_assertion() -> dict:
    return {
        "assertion_status": STATUS_PRESENT,
        "experiencer": EXPERIENCER_PATIENT,
        "temporality": TEMPORALITY_CURRENT,
        "assertion_cue": None,
        "assertion_cue_start": None,
        "assertion_cue_end": None,
        "assertion_cue_category": None,
        "assertion_engine": "medspacy_context",
    }


def _modifier_details(doc, modifier) -> dict:
    """Extracts the triggering cue's text, character offsets and rule category.

    VERIFIED against medspacy 1.3.1 on the EC2 host, 2026-08-08. An earlier
    version of this code read `modifier.span` -- which does NOT exist on
    ConTextModifier and raised AttributeError, caught by a broad except that
    left every cue silently None. The status was still correct, but the
    EVIDENCE for it was missing, which is the part a HITL reviewer actually
    needs: "ABSENT" with no visible cue is an unauditable decision, and this
    pipeline's whole claim is that its decisions are auditable.

    ConTextModifier exposes `modifier_span` as a (start_token, end_token)
    tuple and `category` as the rule type (NEGATED_EXISTENCE, FAMILY,
    HISTORICAL, ...). Several access patterns are tried in order rather than
    assuming one, because this attribute has already changed shape once; each
    is a positive check, and failing all of them yields Nones rather than an
    exception.
    """
    details = {"text": None, "start": None, "end": None, "category": None}

    category = getattr(modifier, "category", None)
    if category is not None:
        details["category"] = str(category)

    # 1. Older API: a real spaCy Span.
    span = getattr(modifier, "span", None)
    if span is not None and hasattr(span, "start_char"):
        details.update({"text": span.text, "start": span.start_char, "end": span.end_char})
        return details

    # 2. medspacy 1.3.x: (start_token, end_token) into the parent doc.
    token_span = getattr(modifier, "modifier_span", None)
    if token_span is not None:
        try:
            start_tok, end_tok = int(token_span[0]), int(token_span[1])
            # The tuple's end has been inclusive in some versions; widening by
            # one token when the slice comes back empty is safer than assuming
            # either convention.
            sl = doc[start_tok:end_tok] or doc[start_tok:end_tok + 1]
            if len(sl):
                details.update({"text": sl.text, "start": sl.start_char, "end": sl.end_char})
                return details
        except (TypeError, ValueError, IndexError):
            pass

    # 3. Last resort: the literal text the matching rule was written against.
    #    No offsets available this way, so the reviewer gets the cue word but
    #    not its position -- recorded rather than pretended otherwise.
    rule = getattr(modifier, "rule", None)
    literal = getattr(rule, "literal", None) or getattr(modifier, "literal", None)
    if literal:
        details["text"] = str(literal)

    return details


def annotate_assertions(text: str, spans: list) -> list:
    """Assigns assertion status to each span.

    text  -- the text the span offsets refer to (Stage 2a passes expanded_text,
             the same coordinate system GLiNER produced the spans in, so no
             offset mapping happens here and there is nowhere for it to go
             wrong).
    spans -- list of (start, end, label) character offsets into `text`.

    Returns a list of assertion dicts positionally aligned with `spans`. Spans
    that cannot be converted to a valid token-aligned span (a boundary falling
    mid-token) get the default PRESENT/PATIENT/CURRENT record with
    `assertion_engine` set to `unaligned_default`, so "we defaulted because
    alignment failed" is distinguishable in provenance from "the detector
    genuinely found no modifier". Conflating those two would let a systematic
    alignment failure masquerade as a clean result.
    """
    if not spans:
        return []

    nlp, context = _build_pipeline()

    doc = nlp(text)

    ents = []
    span_index_map = {}
    for i, (start, end, label) in enumerate(spans):
        span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span is None:
            continue
        span_index_map[(span.start, span.end, span.label)] = i
        ents.append(span)

    # spaCy rejects overlapping entities. GLiNER-BioMed can return nested or
    # overlapping spans, so drop overlaps for the purposes of THIS pass only
    # (longest span wins) -- the dropped spans still get a default record
    # below, and the canonical extracted_entities table is unaffected.
    ents.sort(key=lambda s: (s.start, -(s.end - s.start)))
    non_overlapping = []
    last_end = -1
    for span in ents:
        if span.start >= last_end:
            non_overlapping.append(span)
            last_end = span.end

    try:
        doc.ents = non_overlapping
    except Exception:
        doc.ents = []

    doc = context(doc)

    results = [None] * len(spans)
    for span in doc.ents:
        idx = span_index_map.get((span.start, span.end, span.label))
        if idx is None:
            continue

        negated = _flag(span, "is_negated")
        hypothetical = _flag(span, "is_hypothetical")
        uncertain = _flag(span, "is_uncertain")
        family = _flag(span, "is_family")
        historical = _flag(span, "is_historical")

        # Precedence is deliberate: negation dominates uncertainty. "denies
        # possible chest pain" is an absent finding, not an uncertain one --
        # treating it as POSSIBLE would let it through the Stage 3 gate as a
        # real-but-unconfirmed patient fact.
        if negated:
            status = STATUS_ABSENT
        elif hypothetical:
            status = STATUS_CONDITIONAL
        elif uncertain:
            status = STATUS_POSSIBLE
        else:
            status = STATUS_PRESENT

        cue = {"text": None, "start": None, "end": None, "category": None}
        try:
            modifiers = list(span._.modifiers)
            if modifiers:
                # When several modifiers apply, report the one that determined
                # the status rather than simply the first in the list -- a
                # reviewer checking an ABSENT verdict needs the negation cue,
                # not an unrelated temporality cue that happened to sort first.
                wanted = {
                    STATUS_ABSENT: "NEGATED",
                    STATUS_CONDITIONAL: "HYPOTHETICAL",
                    STATUS_POSSIBLE: "UNCERTAIN",
                }.get(status)
                chosen = None
                if wanted:
                    chosen = next(
                        (mo for mo in modifiers
                         if wanted in str(getattr(mo, "category", "")).upper()),
                        None,
                    )
                cue = _modifier_details(doc, chosen or modifiers[0])
        except (AttributeError, KeyError, IndexError, TypeError):
            pass

        results[idx] = {
            "assertion_status": status,
            "experiencer": EXPERIENCER_FAMILY if family else EXPERIENCER_PATIENT,
            "temporality": TEMPORALITY_HISTORICAL if historical else TEMPORALITY_CURRENT,
            "assertion_cue": cue["text"],
            "assertion_cue_start": cue["start"],
            "assertion_cue_end": cue["end"],
            "assertion_cue_category": cue["category"],
            "assertion_engine": f"medspacy_context/{_segmenter_used}",
        }

    for i, r in enumerate(results):
        if r is None:
            d = _default_assertion()
            d["assertion_engine"] = "unaligned_default"
            results[i] = d

    return results


def apply_section_priors(assertion: dict, section_name_norm: str,
                         experiencer_overrides: dict, temporality_overrides: dict) -> dict:
    """Applies section-header priors on top of cue-based detection.

    A section header is stronger evidence than a nearby cue word: a condition
    listed under `Family History` is about a relative whether or not the word
    "mother" appears anywhere near it, and ConText will find no cue at all in
    a bare list of conditions under that header. So section priors OVERRIDE
    rather than merely supplement, and the override is recorded in
    `assertion_engine` so a reviewer can see which signal decided it.

    Deliberately one-directional: a section prior can move PATIENT -> FAMILY or
    CURRENT -> HISTORICAL, but never the reverse. A cue-detected negation
    inside `Past Medical History` stays negated -- the section says when, not
    whether.
    """
    if not section_name_norm:
        return assertion

    changed = []

    exp = experiencer_overrides.get(section_name_norm)
    if exp and assertion["experiencer"] == EXPERIENCER_PATIENT:
        assertion["experiencer"] = exp
        changed.append("experiencer")

    temp = temporality_overrides.get(section_name_norm)
    if temp and assertion["temporality"] == TEMPORALITY_CURRENT:
        assertion["temporality"] = temp
        changed.append("temporality")

    if changed:
        assertion["assertion_engine"] = (
            f"{assertion['assertion_engine']}+section_prior({','.join(changed)})"
        )
    return assertion
