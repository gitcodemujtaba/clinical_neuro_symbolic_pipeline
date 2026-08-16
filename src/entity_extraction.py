"""
src/entity_extraction.py — Stage 2a entity extraction.

2026-08-07: swapped from urchade/gliner_medium-v2.1 (general-domain) to
GLiNER-BioMed, a suite of GLiNER checkpoints trained specifically on
biomedical NER benchmarks (including clinical narratives) by DS4DH/Univ. of
Geneva. Closed the gap docs/Implementation_Checklist.md and
docs/Proposal_Alignment_Review.md S3.7 flagged: a general-domain model would
understate the architecture's real accuracy against the Clinical-T5 baseline
for Objective 5's benchmark numbers.

2026-08-08: adds everything Stage 3 needs that this stage was previously
dropping on the floor (docs/Stage1_2_Completeness_Audit.md, severity 1/2/6 and
the hard prerequisites):

  * entity_id -- a stable identifier, minted here and carried through every
    later stage. Previously extracted_entities had no key at all and
    normalized_entities joined back to it on a DIFFERENT composite key
    (note_id, original_text, expanded_text, gliner_label) than the one
    extracted_entities was unique on (note_id, orig_start, orig_end,
    entity_label). Stage 3 writes one decision artifact per entity and Stage 4
    hangs a provenance chain off it, so both need something stable to point
    at. Format matches docs/Databases.md's own PatientObservation.entity_id
    spec: <note_id>-e<hash>. The hash is over (orig_start, orig_end,
    entity_label) -- the same fields the table's UNIQUE constraint already
    uses -- so re-running a note reproduces identical IDs rather than
    generating a second set of rows for the same spans.

  * assertion_status / experiencer / temporality -- see src/assertion.py for
    the full reasoning and the measured 20.9% figure. This is the single most
    important addition in this change.

  * section_name, sentence_id, local_context -- Stage 3 runs per-entity, not
    per-note, because the ensemble's shared prompt budget (src/llm_client.py's
    PROMPT_BUDGET_TOKENS) is far smaller than notes run (2,374-24,858
    characters). A sentence-bounded window keeps every prompt's size
    independent of note length.

  * gliner_model_version / extraction_threshold -- docs/Provenance_Schema.md
    fields that were never written. Not bookkeeping: the checkpoint changed on
    2026-08-07, and without a version stamp there is no way to tell which rows
    in extracted_entities came from which model.

RETURN TYPE CHANGED: this function now returns a list of dicts rather than a
list of positional tuples. The tuple form was already being indexed by
position in two other modules (normalization.py's ent[0]..ent[4],
clinical_pipeline.py's e[3]) and this change adds nine more fields -- extending
a positional tuple to fourteen elements would have made every consumer a
silent breakage risk the next time a field was inserted. Callers updated in
the same change: src/normalization.py, src/clinical_pipeline.py. The
scripts/test_*.py smoke tests still index tuples and need updating.
"""

import os
import json
import hashlib
import re
import duckdb
from gliner import GLiNER
import warnings

from src.assertion import (
    annotate_assertions,
    apply_section_priors,
    apply_allergy_context_override,
    is_structured_result,
    _default_assertion,
)
from src.preprocessing import (
    section_for_offset,
    SECTION_EXPERIENCER_OVERRIDE,
    SECTION_TEMPORALITY_OVERRIDE,
)

warnings.filterwarnings("ignore")

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

GLINER_MODEL_NAME = "Ihor/gliner-biomed-large-v1.0"
EXTRACTION_THRESHOLD = 0.5

# SUB-THRESHOLD RETENTION. Extraction runs at SUBTHRESHOLD_FLOOR and everything
# between it and EXTRACTION_THRESHOLD is STORED but flagged `below_threshold`,
# excluded from normalisation and from every downstream stage.
#
# Why keep them at all: spans scoring 0.35-0.50 are exactly the population where
# a KG-grounded second opinion has the most to contribute, and discarding them
# at extraction time makes one of Objective 2's most direct claims impossible to
# evidence -- "Stage 3 recovered N entities the extractor nearly discarded" is
# only measurable if the near-misses were kept. It also gives an empirical basis
# for the 0.5 threshold itself, which is currently an unexamined default.
#
# Why flag rather than promote: these have NOT passed the extraction gate. They
# must never reach normalisation, KG3 or the evaluation counts by default, or
# precision collapses. The flag is the whole mechanism -- src/clinical_pipeline.py
# filters on it before calling Stage 2b.
RETAIN_SUBTHRESHOLD = True
SUBTHRESHOLD_FLOOR = 0.35

# FLAT vs NESTED spans, made explicit and SHARED with src/extraction.py.
# Previously entity_extraction.py used predict_entities()' default (flat) while
# extraction.py passed flat_ner=False (nested) -- the two Stage 2a models
# disagreed about whether overlapping spans were allowed, which made their
# outputs harder to reconcile and was never a deliberate choice.
#
# Set to flat. extracted_entities is the canonical table: Stage 4 writes one
# :PatientObservation per entity, so nested spans ('kidney' inside 'acute kidney
# injury') would create two observations for one clinical fact, and the
# offset-overlap relation linking in extraction.py would have several equally
# valid targets. Nested extraction is defensible on its own terms, but it needs
# a deduplication policy that does not exist yet, and adopting it silently
# through a library default is not the way to get one.
FLAT_NER = True

print("Loading GLiNER-BioMed model... (this may take a moment on the first run)")
model = GLiNER.from_pretrained(GLINER_MODEL_NAME)

CLINICAL_LABELS = [
    "Condition",
    "Symptom",
    "Medication",
    "Procedure",
    "Anatomy",
    "Lab Test",
]

# Target size for the per-entity context window handed to Stage 3. Sized
# against the token budget in docs/MoLLM_Stage3_Retrieval_Design.md S6.5: at
# roughly 4 chars/token this is ~200 tokens, which leaves room for the
# candidate list and up to five guideline rules inside the ensemble's shared
# 8,192-token prompt budget (src/llm_client.py's CONTEXT_WINDOW_TOKENS) with
# headroom to spare.
LOCAL_CONTEXT_MAX_CHARS = 800
LOCAL_CONTEXT_MIN_CHARS = 300


def map_offsets_to_original(exp_start_idx, exp_end_idx, expansions):
    """
    The 'Time Machine': Maps character offsets from the expanded text back
    to the exact character offsets in the original raw clinical note.
    """
    orig_start = exp_start_idx
    orig_end = exp_end_idx

    for exp in expansions:
        shift = (exp["exp_end"] - exp["exp_start"]) - (exp["orig_end"] - exp["orig_start"])

        # Adjust start index
        if exp["exp_end"] <= exp_start_idx:
            orig_start -= shift
        elif exp["exp_start"] <= exp_start_idx < exp["exp_end"]:
            orig_start = exp["orig_start"]

        # Adjust end index
        if exp["exp_end"] <= exp_end_idx:
            orig_end -= shift
        elif exp["exp_start"] < exp_end_idx <= exp["exp_end"]:
            orig_end = exp["orig_end"]

    return orig_start, orig_end


def make_entity_id(note_id: str, orig_start: int, orig_end: int, label: str) -> str:
    """Stable per-span identifier, per docs/Databases.md's <note_id>-e<hash>.

    Hashed over exactly the fields extracted_entities is UNIQUE on, so the ID
    is a deterministic function of the span rather than of insertion order --
    re-running a note yields the same IDs and updates rows in place instead of
    orphaning every Stage 3 decision that referenced the previous run.
    """
    digest = hashlib.sha1(f"{orig_start}:{orig_end}:{label}".encode("utf-8")).hexdigest()
    return f"{note_id}-e{digest[:10]}"


def find_sentence(sentences: list, exp_start: int):
    """Returns the sentence record containing an expanded-text offset."""
    for s in sentences:
        if s["start"] <= exp_start < s["end"]:
            return s
    return None


def sentence_ids_spanned(sentences: list, exp_start: int, exp_end: int) -> list:
    """Sentence IDs whose span overlaps [exp_start, exp_end) in expanded-text
    coordinates.

    2026-08-10, added after scripts/score_gold_recall.py's compound-span
    detection measured a concrete case on 17751158-DS-19: GLiNER extracted
    "Gunshot wound to abdomen\\nPneumonia" as ONE entity, merging a gunshot
    wound diagnosis with an unrelated line from what is almost certainly a
    discharge-diagnosis list -- 34 characters, one predicted concept, and it
    can never correctly link to either gold annotation it overlaps.

    WHY SENTENCE BOUNDARIES AND NOT JUST "\\n" IN THE SPAN. A naive
    "reject any span containing a newline" rule was considered and rejected:
    the SAME note's gold set links "left \\nbase" (60859004) as ONE
    annotation, because that newline is mid-statement line-wrap
    ("...decreased breath sounds at left base"), not a real boundary. This
    module's own docstring (see _clinical_sentencizer) already establishes
    that MIMIC notes use newlines/list breaks as REAL structural boundaries
    more often than punctuation, which is why PyRuSH sentence segmentation
    was adopted for local_context in the first place -- but nothing
    previously checked an extracted span itself against those boundaries. A
    span crossing >1 PyRuSH sentence is the signal that generalizes: a
    line-wrapped continuation of one clinical statement stays inside a single
    sentence, while two distinct diagnosis-list lines do not.

    Returns a list (not a bool) so the caller/reviewer can see exactly which
    sentences got merged, not just that a merge happened.
    """
    return [s["sentence_id"] for s in sentences
            if not (s["end"] <= exp_start or s["start"] >= exp_end)]


def build_local_context(expanded_text: str, sentences: list, exp_start: int, exp_end: int) -> dict:
    """Builds the sentence-bounded context window Stage 3 reasons over.

    Sentence-bounded rather than a raw character window because the whole
    point of the window is to carry negation and qualifier context, and those
    cues cluster at sentence starts ("She denies ...", "No evidence of ...").
    A fixed character window can cut immediately before the cue and produce a
    window that looks complete while having removed the one word that changes
    the meaning.

    Expands to neighbouring sentences when the containing sentence is short
    (clinical notes are full of terse fragments and bulleted lists where the
    governing statement sits in the previous line), then hard-caps at
    LOCAL_CONTEXT_MAX_CHARS, trimming from whichever side is further from the
    entity so the entity itself is never cut.

    Falls back to a plain character window if sentence data is unavailable, so
    a note whose sentence segmentation failed still yields usable context.
    """
    if not sentences:
        start = max(0, exp_start - LOCAL_CONTEXT_MAX_CHARS // 2)
        end = min(len(expanded_text), exp_end + LOCAL_CONTEXT_MAX_CHARS // 2)
        return {
            "text": expanded_text[start:end],
            "start": start,
            "end": end,
            "sentence_id": None,
            "basis": "char_window_no_sentences",
        }

    sent = find_sentence(sentences, exp_start)
    if sent is None:
        sent = min(sentences, key=lambda s: abs(s["start"] - exp_start))

    start, end = sent["start"], sent["end"]
    idx = sent["sentence_id"]
    basis = "sentence"

    if (end - start) < LOCAL_CONTEXT_MIN_CHARS:
        prev_s = sentences[idx - 1] if idx - 1 >= 0 else None
        next_s = sentences[idx + 1] if idx + 1 < len(sentences) else None
        if prev_s:
            start = prev_s["start"]
        if next_s:
            end = next_s["end"]
        basis = "sentence+neighbours"

    if (end - start) > LOCAL_CONTEXT_MAX_CHARS:
        overflow = (end - start) - LOCAL_CONTEXT_MAX_CHARS
        room_left = max(0, exp_start - start)
        room_right = max(0, end - exp_end)
        trim_left = min(room_left, overflow * room_left // max(1, room_left + room_right))
        start += trim_left
        end -= min(room_right, overflow - trim_left)
        basis += "+trimmed"

    return {
        "text": expanded_text[start:end],
        "start": start,
        "end": end,
        "sentence_id": sent["sentence_id"],
        "basis": basis,
    }


def ensure_extracted_entities_table(conn):
    """Creates/migrates extracted_entities. Idempotent.

    Factored out of extract_and_store_entities() (2026-08-10) so
    src/clinical_pipeline.py's compound-span splitter (see
    src/normalization.find_compound_split()) can call store_entities() below
    against the SAME schema definition, rather than maintaining a second DDL
    block that could silently drift from this one -- exactly the kind of
    two-copies-of-the-same-thing risk this codebase's other docstrings
    repeatedly flag (see e.g. src/normalization.py's _tier_queries()).
    """
    conn.sql("""
    CREATE TABLE IF NOT EXISTS extracted_entities (
        note_id VARCHAR,
        entity_label VARCHAR,
        expanded_text VARCHAR,
        original_text VARCHAR,
        confidence FLOAT,
        orig_start INT,
        orig_end INT,
        exp_start INT,
        exp_end INT,
        is_test BOOLEAN DEFAULT FALSE,
        UNIQUE(note_id, orig_start, orig_end, entity_label)
    );
    """)
    for ddl in [
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS is_test BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS entity_id VARCHAR;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS assertion_status VARCHAR;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS experiencer VARCHAR;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS temporality VARCHAR;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS assertion_cue VARCHAR;",
        # Cue offsets, not just the cue text. The HITL reviewer needs to see
        # WHERE the negation fired, not merely that a word matched -- with the
        # text alone, a note containing "no" five times gives a reviewer no way
        # to check the detector picked the right one. Stored in expanded-text
        # coordinates, the same system assertion detection ran in.
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS assertion_cue_start INT;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS assertion_cue_end INT;",
        # The ConText rule category that fired (NEGATED_EXISTENCE, FAMILY,
        # HISTORICAL, ...). More diagnostic than the cue text alone: "denies"
        # and "no" both produce ABSENT, but the category tells a reviewer which
        # rule class made the call, which is what you need to spot a systematic
        # detector failure rather than a one-off.
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS assertion_cue_category VARCHAR;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS assertion_engine VARCHAR;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS section_name VARCHAR;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS sentence_id INT;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS local_context VARCHAR;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS expansion_ambiguous BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS candidate_expansions JSON;",
        # 2026-08-13: see store_entities()'s docstring / the comment at this
        # field's computation site above -- which tiebreak resolved an
        # ambiguous abbreviation, so Stage 1's disambiguation accuracy can be
        # graded per method rather than only "ambiguous vs not".
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS selection_basis VARCHAR;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS gliner_model_version VARCHAR;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS extraction_threshold FLOAT;",
        # TRUE for spans that scored between SUBTHRESHOLD_FLOOR and
        # EXTRACTION_THRESHOLD. Retained for analysis only -- every consumer
        # MUST filter these out unless deliberately studying them.
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS below_threshold BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS flat_ner BOOLEAN;",
        # See sentence_ids_spanned()'s docstring above. TRUE means this span
        # was very likely two unrelated mentions GLiNER merged into one --
        # flagged, not dropped, same policy as expansion_ambiguous and
        # domain_conflict: a reviewer/Stage 3 gets to see it rather than
        # having it silently discarded or silently trusted.
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS crosses_sentence_boundary BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS sentence_ids_spanned JSON;",
        # 2026-08-10, compound-span splitting (src/clinical_pipeline.py,
        # src/normalization.find_compound_split()). compound_split_of is the
        # entity_id of the ORIGINAL merged entity a split-half was derived
        # from (NULL for every entity that was never split). superseded_by_split
        # marks that original entity's own row TRUE once it has been replaced
        # by its split halves -- the row is kept, not deleted (this codebase's
        # consistent policy: is_test/flags over destructive deletes), so the
        # merged extraction remains visible and auditable, just excluded from
        # normalization and downstream stages by the same accepted-list
        # filtering that below_threshold already uses.
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS compound_split_of VARCHAR;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS superseded_by_split BOOLEAN DEFAULT FALSE;",
        # 2026-08-10, span growth (src/clinical_pipeline.py,
        # src/normalization.find_span_growth()) -- the mirror image of the
        # compound-split columns above. grown_from is the entity_id of the
        # pre-growth (narrower) entity a widened entity was derived from
        # (NULL for every entity that was never grown). superseded_by_growth
        # marks that narrower entity's own row TRUE once replaced -- same
        # keep-not-delete policy as superseded_by_split.
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS grown_from VARCHAR;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS superseded_by_growth BOOLEAN DEFAULT FALSE;",
        # 2026-08-15 (Implementation_Checklist.md Stage 1/2a item: "silent
        # truncation on long notes"). GLiNER's own word-level tokenizer
        # silently truncates input over model.config.max_len (2048 for this
        # checkpoint) via a UserWarning that this module's blanket
        # warnings.filterwarnings("ignore") (top of file) was swallowing
        # entirely -- truncation was happening with no record of it anywhere.
        # extract_and_store_entities() now captures that warning explicitly
        # per note (see the predict_entities() call site) rather than relying
        # on it surfacing to a human. NOTE-level facts, written identically
        # onto every entity row for that note -- same redundant-but-simple
        # convention gliner_model_version/extraction_threshold already use,
        # not a new note-level table.
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS possibly_truncated BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS gliner_input_token_count INT;",
    ]:
        conn.sql(ddl)


def store_entities(conn, processed_entities: list, is_test: bool = False):
    """Writes processed-entity dicts (the shape built in
    extract_and_store_entities()'s loop, and by
    src/clinical_pipeline.py's compound-span splitter and span-growth step)
    to extracted_entities.

    Every dict must carry compound_split_of/superseded_by_split and
    grown_from/superseded_by_growth (default None / False for ordinary
    Stage 2a output) alongside the fields extract_and_store_entities() has
    always produced.
    """
    ensure_extracted_entities_table(conn)
    if not processed_entities:
        return

    rows = [(
        e["note_id"], e["entity_label"], e["expanded_text"], e["original_text"],
        e["confidence"], e["orig_start"], e["orig_end"], e["exp_start"], e["exp_end"],
        is_test, e["entity_id"], e["assertion_status"], e["experiencer"],
        e["temporality"], e["assertion_cue"], e["assertion_engine"],
        e["assertion_cue_start"], e["assertion_cue_end"], e["assertion_cue_category"],
        e["section_name"], e["sentence_id"], e["local_context"],
        e["expansion_ambiguous"],
        json.dumps(e["candidate_expansions"]) if e["candidate_expansions"] else None,
        e.get("selection_basis"),
        e["gliner_model_version"], e["extraction_threshold"],
        e["below_threshold"], e["flat_ner"],
        e["crosses_sentence_boundary"], json.dumps(e["sentence_ids_spanned"]),
        e.get("compound_split_of"), e.get("superseded_by_split", False),
        e.get("grown_from"), e.get("superseded_by_growth", False),
        e.get("possibly_truncated", False), e.get("gliner_input_token_count"),
    ) for e in processed_entities]

    # DO UPDATE rather than DO NOTHING: re-running a note after a model or
    # assertion-logic change must refresh the row, not silently preserve
    # whatever the first run computed. This is also how the splitter marks an
    # already-persisted parent row superseded_by_split=TRUE -- it resubmits
    # that row's own (note_id, orig_start, orig_end, entity_label) key with
    # the flag set, landing on this same UPDATE branch rather than a second
    # SQL statement.
    conn.executemany("""
    INSERT INTO extracted_entities
    (note_id, entity_label, expanded_text, original_text, confidence,
     orig_start, orig_end, exp_start, exp_end, is_test, entity_id,
     assertion_status, experiencer, temporality, assertion_cue, assertion_engine,
     assertion_cue_start, assertion_cue_end, assertion_cue_category,
     section_name, sentence_id, local_context, expansion_ambiguous,
     candidate_expansions, selection_basis, gliner_model_version, extraction_threshold,
     below_threshold, flat_ner, crosses_sentence_boundary, sentence_ids_spanned,
     compound_split_of, superseded_by_split, grown_from, superseded_by_growth,
     possibly_truncated, gliner_input_token_count)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT (note_id, orig_start, orig_end, entity_label) DO UPDATE SET
        expanded_text = EXCLUDED.expanded_text,
        original_text = EXCLUDED.original_text,
        confidence = EXCLUDED.confidence,
        exp_start = EXCLUDED.exp_start,
        exp_end = EXCLUDED.exp_end,
        is_test = EXCLUDED.is_test,
        entity_id = EXCLUDED.entity_id,
        assertion_status = EXCLUDED.assertion_status,
        experiencer = EXCLUDED.experiencer,
        temporality = EXCLUDED.temporality,
        assertion_cue = EXCLUDED.assertion_cue,
        assertion_cue_start = EXCLUDED.assertion_cue_start,
        assertion_cue_end = EXCLUDED.assertion_cue_end,
        assertion_cue_category = EXCLUDED.assertion_cue_category,
        assertion_engine = EXCLUDED.assertion_engine,
        section_name = EXCLUDED.section_name,
        sentence_id = EXCLUDED.sentence_id,
        local_context = EXCLUDED.local_context,
        expansion_ambiguous = EXCLUDED.expansion_ambiguous,
        candidate_expansions = EXCLUDED.candidate_expansions,
        selection_basis = EXCLUDED.selection_basis,
        gliner_model_version = EXCLUDED.gliner_model_version,
        extraction_threshold = EXCLUDED.extraction_threshold,
        below_threshold = EXCLUDED.below_threshold,
        flat_ner = EXCLUDED.flat_ner,
        crosses_sentence_boundary = EXCLUDED.crosses_sentence_boundary,
        sentence_ids_spanned = EXCLUDED.sentence_ids_spanned,
        compound_split_of = EXCLUDED.compound_split_of,
        superseded_by_split = EXCLUDED.superseded_by_split,
        grown_from = EXCLUDED.grown_from,
        superseded_by_growth = EXCLUDED.superseded_by_growth,
        possibly_truncated = EXCLUDED.possibly_truncated,
        gliner_input_token_count = EXCLUDED.gliner_input_token_count;
    """, rows)


def extract_and_store_entities(note_id: str, expanded_text: str, raw_text: str,
                               provenance: dict, conn, is_test: bool = False):
    """Runs GLiNER on expanded text, reconciles offsets, attaches assertion /
    section / context metadata, and stores in DuckDB.

    Returns a list of dicts (see this module's docstring for why this is no
    longer a list of tuples).

    is_test flags these rows as coming from a test/smoke-test run rather than
    production ingest, so extracted_entities rows can be traced back to (and
    purged for) the same test note without touching real data.

    `provenance` (2026-08-15, Implementation_Checklist.md's Stage 2a
    pure-function item): the caller's own Stage 1 output, passed in rather
    than re-queried from `note_expansions` here. `run_pipeline()`
    (src/clinical_pipeline.py) already holds this from
    `process_and_store_note()` moments earlier -- re-fetching an identical
    copy from the DB was a redundant read with no purpose beyond convenience,
    and it was the one remaining DB touch keeping this function from being a
    pure compute step ahead of the explicit `store_entities()` persist call
    below. Every caller must now have already run Stage 1 and be holding its
    provenance dict; there is no fallback DB read.
    """
    expansions = provenance.get("expansions", [])
    sections = provenance.get("sections", [])
    sentences = provenance.get("sentences", [])

    # Expansion lookup keyed by expanded-text span, so an entity that sits on
    # an expanded abbreviation can inherit that expansion's ambiguity flag --
    # this is what lets an ambiguous "MS" reach Stage 3 marked as ambiguous
    # rather than as a settled fact (Completeness Audit sev 3).
    ambiguous_expansions = [e for e in expansions if e.get("ambiguous")]

    # 2. Run Zero-Shot Extraction
    # Extracted at the LOWER floor; the split into accepted vs below-threshold
    # happens per-entity below. One model call, not two.
    floor = SUBTHRESHOLD_FLOOR if RETAIN_SUBTHRESHOLD else EXTRACTION_THRESHOLD
    # 2026-08-15: catch GLiNER's own truncation UserWarning
    # ("Sentence of length {num_tokens} has been truncated to {max_len}",
    # gliner/data_processing/processor.py) explicitly, scoped to just this
    # call -- NOT by touching the module-level warnings.filterwarnings(
    # "ignore") above, which suppresses other, unrelated library warnings
    # this module has no reason to newly surface. Long notes (mean ~10,257
    # chars, max 24,858) can exceed model.config.max_len=2048 word-tokens for
    # this checkpoint and were being silently truncated with zero record of
    # it. This does not fix truncation (that needs chunk-and-merge extraction,
    # a bigger feature deliberately not built here -- see
    # Implementation_Checklist.md) -- it makes it measurable.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw_entities = model.predict_entities(
            expanded_text, CLINICAL_LABELS, threshold=floor, flat_ner=FLAT_NER
        )
    possibly_truncated = False
    gliner_input_token_count = None
    for w in caught:
        m = re.search(r"Sentence of length (\d+) has been truncated to (\d+)", str(w.message))
        if m:
            possibly_truncated = True
            gliner_input_token_count = int(m.group(1))
            break

    # 3. Assertion detection over the SAME text and coordinate system GLiNER
    #    produced the spans in -- no offset mapping happens inside
    #    src/assertion.py, so there is nowhere for it to go wrong.
    spans = [(e["start"], e["end"], e["label"]) for e in raw_entities]
    try:
        assertions = annotate_assertions(expanded_text, spans)
    except Exception as exc:
        # A missing/incompatible medspacy must not take the whole pipeline
        # down, but it must also not be silently indistinguishable from "no
        # modifiers found" -- assertion_engine records the failure per row.
        print(f"⚠️ Assertion detection unavailable ({exc}); defaulting all entities to PRESENT.")
        assertions = []
        for _ in raw_entities:
            d = _default_assertion()
            d["assertion_engine"] = "unavailable_default"
            assertions.append(d)

    # 4. Create the extracted entities table if it doesn't exist
    ensure_extracted_entities_table(conn)

    processed_entities = []

    # 5. Reconcile offsets, attach metadata, and store
    for i, ent in enumerate(raw_entities):
        exp_start = ent["start"]
        exp_end = ent["end"]
        label = ent["label"]
        confidence = ent["score"]
        exp_text = ent["text"]

        orig_start, orig_end = map_offsets_to_original(exp_start, exp_end, expansions)
        orig_text = raw_text[orig_start:orig_end]

        entity_id = make_entity_id(note_id, orig_start, orig_end, label)

        section = section_for_offset(sections, orig_start)
        section_name = section["name"] if section else None
        section_norm = section["name_norm"] if section else None

        assertion = assertions[i] if i < len(assertions) else _default_assertion()

        assertion = apply_section_priors(
            assertion, section_norm,
            SECTION_EXPERIENCER_OVERRIDE, SECTION_TEMPORALITY_OVERRIDE,
        )
        assertion = apply_allergy_context_override(assertion, section_norm, label)

        context = build_local_context(expanded_text, sentences, exp_start, exp_end)

        spanned_sentence_ids = sentence_ids_spanned(sentences, exp_start, exp_end)
        crosses_sentence_boundary = len(spanned_sentence_ids) > 1

        # Structured lab panels are tabular data, not narrative -- ConText's
        # cue propagation does not apply and can only produce false negations
        # there (see src/assertion.is_structured_result).
        #
        # Applied LAST, after both ConText and the section priors, and it
        # supersedes both. That ordering is deliberate: how a value is
        # RECORDED (a result table) is a stronger signal than any cue found
        # near it or any section it falls under. The override is a separate
        # step rather than a skip so the record still shows the pipeline ran
        # detection, with assertion_engine naming what decided the outcome.
        if is_structured_result(orig_text, context["text"], gliner_label=label):
            assertion = dict(assertion)
            assertion.update({
                "assertion_status": "PRESENT",
                "experiencer": "PATIENT",
                "temporality": "CURRENT",
                "assertion_cue": None,
                "assertion_cue_start": None,
                "assertion_cue_end": None,
                "assertion_cue_category": "STRUCTURED_RESULT_NOT_APPLICABLE",
                "assertion_engine": "structured_result_override",
            })

        # Does this entity's span overlap an expansion that had more than one
        # known meaning? If so the surface form it was extracted from may be
        # the wrong reading of the abbreviation entirely, which Stage 3 needs
        # told explicitly rather than left to infer.
        overlapping = [
            e for e in ambiguous_expansions
            if not (e["exp_end"] <= exp_start or e["exp_start"] >= exp_end)
        ]
        expansion_ambiguous = bool(overlapping)
        candidate_expansions = overlapping[0].get("candidate_expansions") if overlapping else None
        # 2026-08-13: which of preprocessing.py's three tiebreaks resolved this
        # ambiguous abbreviation -- "numeric_context:<kind>", "omop_groundability",
        # or "alphabetical_default" (src/preprocessing.py's
        # expand_text_and_track_offsets() docstring). Computed there since
        # 2026-08-13 (the fx/fractions groundability fix) but never threaded
        # through to persistence until now -- there was no way to grade "did
        # Stage 1's disambiguation choice hold up" per tiebreak method without
        # it. None for every non-ambiguous entity, same convention as
        # candidate_expansions above.
        selection_basis = overlapping[0].get("selection_basis") if overlapping else None

        processed_entities.append({
            "entity_id": entity_id,
            "note_id": note_id,
            "entity_label": label,
            "expanded_text": exp_text,
            "original_text": orig_text,
            "confidence": confidence,
            "orig_start": orig_start,
            "orig_end": orig_end,
            "exp_start": exp_start,
            "exp_end": exp_end,
            "assertion_status": assertion["assertion_status"],
            "experiencer": assertion["experiencer"],
            "temporality": assertion["temporality"],
            "assertion_cue": assertion["assertion_cue"],
            "assertion_cue_start": assertion.get("assertion_cue_start"),
            "assertion_cue_end": assertion.get("assertion_cue_end"),
            "assertion_cue_category": assertion.get("assertion_cue_category"),
            "assertion_engine": assertion["assertion_engine"],
            "section_name": section_name,
            "sentence_id": context["sentence_id"],
            "local_context": context["text"],
            "local_context_basis": context["basis"],
            "expansion_ambiguous": expansion_ambiguous,
            "candidate_expansions": candidate_expansions,
            "selection_basis": selection_basis,
            "gliner_model_version": GLINER_MODEL_NAME,
            "extraction_threshold": EXTRACTION_THRESHOLD,
            "below_threshold": confidence < EXTRACTION_THRESHOLD,
            "flat_ner": FLAT_NER,
            "crosses_sentence_boundary": crosses_sentence_boundary,
            "sentence_ids_spanned": spanned_sentence_ids,
            # Ordinary Stage 2a output -- never a split or grown product,
            # never yet superseded. src/clinical_pipeline.py's compound-span
            # splitter and span-growth step set these explicitly on the
            # entities THEY build, and mutate the superseded_* field TRUE on
            # a parent's own dict when replacing it.
            "compound_split_of": None,
            "superseded_by_split": False,
            "grown_from": None,
            "superseded_by_growth": False,
            "possibly_truncated": possibly_truncated,
            "gliner_input_token_count": gliner_input_token_count,
        })

    store_entities(conn, processed_entities, is_test)
    return processed_entities
