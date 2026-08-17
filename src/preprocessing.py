"""
src/preprocessing.py — Stage 1.

2026-08-08 rewrite. Adds the Stage-1 items from
docs/Stage1_2_Completeness_Audit.md that Stage 3 depends on. What changed and
why, in the order the audit ranked them:

SEVERITY 3 — ambiguous abbreviations were being silently collapsed.
    load_abbreviations_dict() previously built
    `{abbr.lower(): meaning for abbr, meaning in rows}`. A dict comprehension
    keeps the LAST value for a duplicated key, so for an abbreviation with
    several expansions in kg2a_abbreviations (MS -> multiple sclerosis /
    mitral stenosis / morphine sulfate; PT -> prothrombin time / physical
    therapy / patient; DC -> discontinue / discharge) every expansion but one
    was discarded at load time, the survivor was chosen by whatever order the
    rows came back in, and it was then applied unconditionally to every
    occurrence in the note. A wrong expansion lands in expanded_text, which is
    what Stage 2a extracts from and Stage 2b normalizes against, so one bad
    expansion silently poisons the entity AND its concept mapping with nothing
    downstream able to notice. It is also why docs/Provenance_Schema.md's
    `ambiguous` flag wasn't merely unimplemented but uncomputable -- by the
    time expansion ran, the alternatives no longer existed in memory.
    Now: the dictionary keeps every meaning, ambiguous expansions are flagged
    and carry their full candidate list, and Stage 3 gets to disambiguate them
    from local context + KG evidence (a better demonstration of Objective 2
    than concept disambiguation alone).

SEVERITY 2 — note section structure was discarded.
    MIMIC-IV discharge notes are strongly sectioned and it is measurable: over
    the 272 gold notes, `Past Medical History` occurs 274 times, `History of
    Present Illness` 272, `Family History` 271, `Discharge Medications` 269.
    Section membership is a free, deterministic prior on exactly the questions
    assertion detection asks -- everything under `Family History` is about a
    relative regardless of whether a cue word appears, `Past Medical History`
    implies historical, `Allergies` drug mentions are emphatically NOT
    administered medications. Stage 1 already read the whole note and stored
    none of this.

SEVERITY 6 — sentence boundaries were computed and thrown away.
    nlp(text) was called and only its tokens used; doc.sents is exactly what
    Stage 2a's local_context window needs, so it was being re-derived from
    scratch downstream.

Also adds `abbreviation_dict_version` (a docs/Provenance_Schema.md Stage 1
field that was never written) and note-level metadata.

TWO spaCy PASSES, DELIBERATELY:
    The first pass runs on the ORIGINAL text because abbreviation expansion
    needs original-coordinate token offsets. The second runs on the EXPANDED
    text because sentence boundaries are consumed by Stage 2a, whose entity
    offsets are in expanded coordinates -- deriving expanded-text sentences by
    shifting original-text sentence offsets through the expansion map would be
    a second, redundant implementation of the offset arithmetic in
    entity_extraction.map_offsets_to_original(), and a second place for it to
    go wrong. Sections, by contrast, are stored in ORIGINAL coordinates: they
    are found by line-anchored header matching, which is unaffected by
    expansion, and every entity already carries orig_start for the lookup.
"""

import os
import re
import json
import hashlib
import duckdb
import spacy
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    nlp = spacy.load("en_core_sci_sm")
except Exception:
    import en_core_sci_sm
    nlp = en_core_sci_sm.load()

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

# Line-anchored section header matcher. Deliberately generic (any
# Capitalised-or-ALLCAPS short phrase at the start of a line, followed by a
# colon) rather than a fixed whitelist of MIMIC section names: a whitelist
# would silently drop sections in notes that phrase a header slightly
# differently, and the failure mode would be invisible. The length bound and
# the line anchor are what keep it from firing on mid-sentence colons
# ("Dr. Smith: ...", "Impression: stable" inline).
#
# Known limitation, worth stating rather than hiding: this is a heuristic over
# note formatting, not a parse. Notes with unusual layout may yield a
# <no section> stretch, which downstream code must treat as "unknown", NOT as
# a specific section.
SECTION_HEADER_RE = re.compile(r"^[ \t]*([A-Z][A-Za-z0-9 /\-'&\.]{2,45}?):", re.MULTILINE)

# Sections whose contents are, by the structure of the document itself, not
# asserted current findings about the patient. Stage 2a uses these as a prior
# that OVERRIDES cue-based detection, because a section header is stronger
# evidence than a nearby word: a condition listed under "Family History" is
# about a relative whether or not the word "mother" appears near it.
SECTION_EXPERIENCER_OVERRIDE = {
    "family history": "FAMILY",
}
SECTION_TEMPORALITY_OVERRIDE = {
    "past medical history": "HISTORICAL",
    "past surgical history": "HISTORICAL",
    "medications on admission": "HISTORICAL",
    "major surgical or invasive procedure": "HISTORICAL",
}


def _dict_version(rows) -> str:
    """Content hash of the abbreviation dictionary, so a provenance record can
    identify exactly which dictionary produced an expansion.

    Hashes the sorted (abbrev, meaning) pairs rather than using a row count or
    a load timestamp: the count can stay identical while contents change, and
    a timestamp says when the table was read, not what was in it. Sorting
    makes the hash independent of row order, which DuckDB does not guarantee.
    """
    h = hashlib.sha256()
    for abbr, meaning in sorted(rows):
        h.update(f"{abbr}\t{meaning}\n".encode("utf-8"))
    return f"sha256:{h.hexdigest()[:16]}/n={len(rows)}"


def load_abbreviations_dict(conn) -> tuple:
    """Returns (dict[abbrev_lower -> [meanings]], dict_version).

    Every meaning is preserved -- see this module's docstring for what the old
    single-meaning dict was silently doing. Meanings are sorted and
    deduplicated so that (a) the choice of primary expansion is deterministic
    across runs rather than dependent on DuckDB's row order, and (b) the
    ambiguity flag reflects genuinely distinct meanings rather than duplicate
    rows for the same expansion.

    2026-08-13 CASE-INSENSITIVE SORT FIX. Python's default sorted() is
    case-sensitive (all capitals sort before all lowercase in ASCII), and the
    source CSVs mix casing inconsistently -- proper nouns and acronym
    expansions capitalized ("American Society of Anesthesiologists",
    "Creatinine"), common terms lowercase ("acetylsalicylic acid (aspirin)",
    "clinical response"). Measured directly on the 272-note corpus: this
    silently made ASA default to "American Society of Anesthesiologists"
    instead of "aspirin" in every medication-dosage context, purely because
    of capitalization, not true a-z order. key=str.lower makes the sort (and
    therefore the deterministic default pick) reflect actual alphabetical
    order regardless of source casing.
    """
    tables = [t[0] for t in conn.sql("SHOW TABLES").fetchall()]
    if "kg2a_abbreviations" not in tables:
        return {}, "absent"

    rows = conn.sql("SELECT abbreviation, meaning FROM kg2a_abbreviations").fetchall()
    rows = [(a, m) for a, m in rows if a and m]

    grouped = {}
    for abbr, meaning in rows:
        grouped.setdefault(abbr.lower(), set()).add(meaning)

    return {a: sorted(m, key=str.lower) for a, m in grouped.items()}, _dict_version(rows)


def segment_sections(text: str) -> list:
    """Finds section header spans in the ORIGINAL note text.

    Returns [{"name", "name_norm", "header_start", "start", "end"}], where
    `start` is the first character AFTER the header (i.e. the section body)
    and `end` runs to the next header or end of note.
    """
    matches = list(SECTION_HEADER_RE.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({
            "name": name,
            "name_norm": name.lower(),
            "header_start": m.start(),
            "start": body_start,
            "end": body_end,
        })
    return sections


def section_for_offset(sections: list, offset: int):
    """Which section contains a given ORIGINAL-text character offset.

    Returns None for text before the first header (typically the note's
    administrative preamble) rather than guessing -- callers must treat None
    as "unknown section", not as any particular section.
    """
    hit = None
    for s in sections:
        if s["start"] <= offset < s["end"]:
            hit = s
        elif s["start"] > offset:
            break
    return hit


# 2026-08-13 NUMERIC-CONTEXT-AWARE MEANING SELECTION. Two shapes, matching a
# pattern measured directly in this project's data (SBP<100, HR<60 in note
# 10097089-DS-8; 12.5 mg dosage lines throughout):
#   "lab_value"  -- the abbreviation is immediately followed by a comparator
#                   (< > = -) then a number, e.g. "HR<60", "SBP <100". Prefer
#                   whichever candidate meaning resolves to OMOP's
#                   Measurement/Observation domain.
#   "unit_value" -- a bare number sits directly adjacent (either side, no
#                   comparator), e.g. "12.5 mg". Prefer whichever candidate
#                   resolves to OMOP's Unit domain.
# Classification is ontology-arbitrated (an athena_concept domain_id lookup),
# not a hand-maintained keyword list -- same philosophy as
# strip_lab_value_suffix()/find_compound_split() in src/normalization.py:
# "ontology arbitrates, not guesses". If no candidate resolves to the target
# domain (or conn is unavailable), falls back to the pre-existing
# alphabetical-first pick unchanged. Every candidate is still preserved in
# candidate_expansions regardless of which one is selected -- this only
# changes which one is applied to expanded_text, never what is logged.
_LAB_VALUE_CONTEXT_RE = re.compile(r'^\s*[<>=\-]\s*-?\d')
_UNIT_VALUE_AFTER_RE = re.compile(r'^\s*-?\d')
_UNIT_VALUE_BEFORE_RE = re.compile(r'-?\d\s*$')

_UNIT_DOMAINS = {"Unit"}
_LAB_VALUE_DOMAINS = {"Measurement", "Observation"}

_domain_lookup_cache = {}


def _numeric_context_kind(text: str, orig_start: int, orig_end: int):
    """Classifies the immediate character context around a token as
    'lab_value', 'unit_value', or None. Looks at the ORIGINAL (unexpanded)
    text using original-coordinate offsets, so a prior expansion earlier in
    the same note cannot shift what this token sees.
    """
    after = text[orig_end:orig_end + 12]
    before = text[max(0, orig_start - 12):orig_start]

    if _LAB_VALUE_CONTEXT_RE.match(after):
        return "lab_value"
    if _UNIT_VALUE_AFTER_RE.match(after) or _UNIT_VALUE_BEFORE_RE.search(before):
        return "unit_value"
    return None


def _omop_domain_for_meaning(conn, meaning: str):
    """OMOP domain_id for a candidate meaning's Tier-1/2-shaped exact or
    synonym match -- same SQL shape as normalization.py's _lookup_tier12(),
    deliberately Tier 1/2 only (no semantic guessing) since this decides
    WHICH expansion becomes the note's text, a higher-stakes choice than a
    single entity's candidate ranking. Cached per meaning per process:
    the same handful of words (milligram, heart rate, ...) recur across
    every note, so this avoids re-querying identical lookups thousands of
    times over a batch. Returns None (not a guess) on no hit or any DB
    error, so the caller always has a safe fallback.
    """
    key = meaning.lower()
    if key in _domain_lookup_cache:
        return _domain_lookup_cache[key]

    domain = None
    try:
        row = conn.execute("""
            SELECT domain_id FROM athena_concept
            WHERE lower(concept_name) = ? AND standard_concept = 'S'
            ORDER BY concept_id ASC LIMIT 1
        """, [key]).fetchone()
        if row is None:
            row = conn.execute("""
                SELECT c.domain_id FROM athena_concept_synonym s
                JOIN athena_concept c ON s.concept_id = c.concept_id
                WHERE lower(s.concept_synonym_name) = ? AND c.standard_concept = 'S'
                ORDER BY c.concept_id ASC LIMIT 1
            """, [key]).fetchone()
        if row is not None:
            domain = row[0]
    except duckdb.Error:
        domain = None

    _domain_lookup_cache[key] = domain
    return domain


def _select_by_numeric_context(conn, meanings: list, kind: str):
    """Returns whichever meaning (in the existing sorted order, so ties still
    resolve deterministically) resolves to the domain implied by `kind`, or
    None if nothing qualifies -- the caller keeps the alphabetical default
    in that case rather than forcing a pick.
    """
    if conn is None:
        return None
    target_domains = _UNIT_DOMAINS if kind == "unit_value" else _LAB_VALUE_DOMAINS
    for m in meanings:
        if _omop_domain_for_meaning(conn, m) in target_domains:
            return m
    return None


def _select_by_groundability(conn, meanings: list):
    """Fallback tiebreak for ambiguous abbreviations with NO numeric-context
    trigger (_numeric_context_kind returned None, or the numeric-context pick
    itself found nothing): prefer a candidate meaning that resolves to SOME
    real OMOP concept over one that does not, when exactly one candidate
    does.

    2026-08-13, found while diagnosing why 'L \\nclavicular fx' still wasn't
    compound-splitting after the laterality-floor fix (_SHORT_LATERALITY_
    TOKENS, src/normalization.py): kg2a_abbreviations has 'fx' -> {'fracture',
    'fractions'} (case-insensitive-merged, per this module's 2026-08-13 sort
    fix -- that fix is working correctly). Plain alphabetical order has no
    relationship to clinical likelihood, though: comparing the two strings
    character by character, 'fractions' sorts before 'fracture' purely
    because 'i' < 'u' at the first differing character, so the alphabetical
    default picked 'fractions' -- not a real OMOP concept at all in this
    corpus -- over 'fracture', the standard orthopedic/radiology reading and
    an actual SNOMED concept. Same "let the ontology arbitrate instead of
    guessing" principle _select_by_numeric_context/find_compound_split()/
    strip_lab_value_suffix() (src/normalization.py) already use elsewhere in
    this pipeline, generalized to the no-numeric-context case rather than
    left to fall through to alphabetical order unchallenged.

    Deliberately conservative: if MULTIPLE candidates resolve to a real
    concept (a genuine cross-concept ambiguity -- two distinct real
    meanings) or NONE do (neither is a known OMOP concept), this returns
    None and the caller keeps the alphabetical default -- forcing a pick in
    either of those cases would be a guess, not an arbitration, the same
    reasoning find_compound_split()'s REVERT NOTICE gives for requiring
    EVERY part to resolve rather than "at least one."
    """
    if conn is None:
        return None
    grounded = [m for m in meanings if _omop_domain_for_meaning(conn, m) is not None]
    if len(grounded) == 1:
        return grounded[0]
    return None


def expand_text_and_track_offsets(text: str, abbrev_dict: dict, conn=None):
    """Expands known abbreviations, tracking original<->expanded offsets.

    abbrev_dict maps abbreviation -> LIST of meanings. When an abbreviation
    has more than one known meaning, five tiebreaks are tried in order,
    each only consulted when every one before it declined to pick (returned
    None): (1) a mined context-pattern rule from real reviewer-confirmed
    data (src.abbreviation_flywheel.select_by_context_pattern() -- 2026-08-17,
    empty/no-op until real HITL/SME review data exists to mine from), (2)
    numeric context around the token (_numeric_context_kind/
    _select_by_numeric_context above), (3) the pipeline's own observed-
    frequency prior, excluding abbreviations with known systematic bias
    (src.abbreviation_flywheel.compute_frequency_priority(), 2026-08-17), (4)
    OMOP groundability when nothing above resolved anything
    (_select_by_groundability above -- prefers a candidate that resolves to
    a real OMOP concept over one that doesn't, when exactly one does), (5)
    if nothing resolves anything, the first (sorted) meaning is applied so
    the expansion is still deterministic. Either way, the expansion record
    carries `ambiguous: True` plus the full `candidate_expansions` list so
    Stage 3 can reconsider the choice with context the dictionary lookup
    never had. This is the point: Stage 1 should not be silently resolving
    a genuine clinical ambiguity by row order (or by any of these five
    tiebreaks either -- every one of them is a prior, not a verdict, except
    (1) which is grounded in actual confirmed outcomes).

    conn is optional (default None): passing it enables the numeric-context
    OMOP lookup and the two flywheel tiebreaks; omitting it preserves the
    pre-2026-08-13 pure-alphabetical behavior unchanged, so any other caller
    that hasn't been updated keeps working exactly as before.
    """
    doc = nlp(text)
    expanded_text = text
    expansions_log = []
    offset_shift = 0

    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue

        token_lower = token.text.lower()
        meanings = abbrev_dict.get(token_lower)
        if not meanings:
            continue

        is_ambiguous = len(meanings) > 1
        expansion = meanings[0]
        selection_basis = "alphabetical_default"

        orig_start = token.idx
        orig_end = token.idx + len(token.text)

        if is_ambiguous:
            # 2026-08-17 (abbreviation flywheel, checked FIRST): a
            # deterministic pre-filter from real reviewer-confirmed context
            # patterns outranks every heuristic below it, including
            # numeric-context -- it's the only one of these five actually
            # grounded in confirmed outcomes for THIS specific occurrence's
            # wording, not a general-purpose guess. Returns None (falls
            # through unchanged) until real HITL/SME review data exists to
            # mine rules from -- see src.abbreviation_flywheel's module
            # docstring for why this one, unlike the frequency-priority
            # check below, is intentionally allowed to apply even to
            # abbreviations known to have systematic model bias.
            from src.abbreviation_flywheel import (
                compute_frequency_priority, select_by_context_pattern)
            preferred = select_by_context_pattern(
                conn, meanings, token_lower, text, orig_start, orig_end)
            if preferred is not None:
                expansion = preferred
                selection_basis = "context_pattern_rule"

            if selection_basis == "alphabetical_default":
                kind = _numeric_context_kind(text, orig_start, orig_end)
                if kind is not None:
                    preferred = _select_by_numeric_context(conn, meanings, kind)
                    if preferred is not None:
                        expansion = preferred
                        selection_basis = f"numeric_context:{kind}"
            if selection_basis == "alphabetical_default":
                # 2026-08-17 (abbreviation flywheel): the pipeline's own
                # accumulated observed-frequency prior, EXCLUDING
                # abbreviations already known to have systematic bias (see
                # src.abbreviation_flywheel.compute_frequency_priority()'s
                # own docstring) -- checked before groundability since a
                # data-driven empirical preference is stronger evidence
                # than "resolves to some real concept or doesn't."
                preferred = compute_frequency_priority(conn, token_lower, meanings)
                if preferred is not None:
                    expansion = preferred
                    selection_basis = "observed_frequency_priority"
            if selection_basis == "alphabetical_default":
                preferred = _select_by_groundability(conn, meanings)
                if preferred is not None:
                    expansion = preferred
                    selection_basis = "omop_groundability"

        exp_start = orig_start + offset_shift
        exp_end = exp_start + len(expansion)

        record = {
            "abbrev": token.text,
            "expansion": expansion,
            "orig_start": orig_start,
            "orig_end": orig_end,
            "exp_start": exp_start,
            "exp_end": exp_end,
            "ambiguous": is_ambiguous,
        }
        if is_ambiguous:
            record["candidate_expansions"] = meanings
            record["selection_basis"] = selection_basis
        expansions_log.append(record)

        expanded_text = (
            expanded_text[:exp_start]
            + expansion
            + expanded_text[exp_start + len(token.text):]
        )
        offset_shift += (len(expansion) - len(token.text))

    return expanded_text, expansions_log


_sentencizer = None



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


def _clinical_sentencizer():
    """PyRuSH-based segmenter for clinical layout, built once and reused.

    scispaCy's parser-based sentence splitting has the same weakness as
    spaCy's rule sentencizer on this corpus: MIMIC notes separate content with
    NEWLINES and `Header:` lines rather than terminal punctuation. Measured on
    note 10000032-DS-21 (2026-08-08), that produced a single 1,908-character
    "sentence" spanning the entire physical exam and laboratory block -- which
    both corrupted assertion scope (see src/assertion.py) and made
    local_context windows far larger than the entity's actual context.

    Returns None if PyRuSH is unavailable, and the caller falls back to the
    scispaCy doc rather than failing.
    """
    global _sentencizer
    if _sentencizer is not None:
        return _sentencizer
    try:
        import spacy
        import medspacy  # noqa: F401  -- registers the factory
        _silence_pyrush()
        pipe = spacy.blank("en")
        pipe.add_pipe("medspacy_pyrush")
        _sentencizer = pipe
    except Exception:
        _sentencizer = False
    return _sentencizer or None


def sentence_spans(expanded_text: str) -> list:
    """Sentence boundaries over the EXPANDED text, in expanded coordinates.

    Consumed by Stage 2a to build each entity's local_context window without
    cutting mid-sentence (which would strip exactly the negation/qualifier
    cues that give the window its value), and -- indirectly, by keeping the
    two consistent -- by assertion scope.
    """
    pipe = _clinical_sentencizer()
    doc = pipe(expanded_text) if pipe is not None else nlp(expanded_text)
    return [
        {"sentence_id": i, "start": sent.start_char, "end": sent.end_char}
        for i, sent in enumerate(doc.sents)
    ]


def process_and_store_note(note_id: str, raw_text: str, conn, is_test: bool = False):
    """Runs Stage 1 and persists the provenance record to DuckDB.

    Returns (expanded_text, provenance_data). provenance_data now carries
    `sections` and `sentences` alongside `expansions`, so Stage 2a can stamp
    each entity with its section and build a sentence-bounded context window
    without recomputing either.

    is_test flags this row as coming from a test/smoke-test run rather than a
    real production ingest, so test rows can be purged from note_expansions
    later without touching real note data. Defaults to False.
    """
    abbrev_dict, dict_version = load_abbreviations_dict(conn)
    expanded_text, expansions_log = expand_text_and_track_offsets(raw_text, abbrev_dict, conn=conn)
    sections = segment_sections(raw_text)
    sentences = sentence_spans(expanded_text)

    provenance_data = {
        "note_id": note_id,
        "abbreviation_dict_version": dict_version,
        "expansions_count": len(expansions_log),
        "ambiguous_expansions_count": sum(1 for e in expansions_log if e.get("ambiguous")),
        "expansions": expansions_log,
        "sections": sections,
        "sentences": sentences,
        "note_metadata": {
            "raw_char_length": len(raw_text),
            "expanded_char_length": len(expanded_text),
            "sentence_count": len(sentences),
            "section_count": len(sections),
        },
    }

    conn.sql("""
    CREATE TABLE IF NOT EXISTS note_expansions (
        note_id VARCHAR PRIMARY KEY,
        provenance JSON,
        is_test BOOLEAN DEFAULT FALSE
    );
    """)
    conn.sql("ALTER TABLE note_expansions ADD COLUMN IF NOT EXISTS is_test BOOLEAN DEFAULT FALSE;")

    conn.sql("""
    INSERT INTO note_expansions (note_id, provenance, is_test)
    VALUES (?, ?, ?)
    ON CONFLICT (note_id) DO UPDATE SET provenance = EXCLUDED.provenance, is_test = EXCLUDED.is_test;
    """, params=[note_id, json.dumps(provenance_data), is_test])

    return expanded_text, provenance_data
