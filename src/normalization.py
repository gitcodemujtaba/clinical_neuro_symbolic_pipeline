"""
src/normalization.py — Stage 2b grounding & normalization.

2026-08-07 added the domain/vocabulary restrictions and the Tier-3 similarity
floor documented below.

2026-08-08 adds the Stage 3 prerequisites from
docs/MoLLM_Stage3_Retrieval_Design.md S3 and
docs/Stage1_2_Completeness_Audit.md:

  * TOP-3 CANDIDATES INSTEAD OF LIMIT 1. Stage 3 does two different jobs
    depending on how confident Stage 2b was. When the match is clean it only
    needs to answer "does this contradict a guideline?", and extra candidates
    are wasted context. When the match is ambiguous it is genuinely
    disambiguating, and it cannot do that if Stage 2b already discarded the
    alternatives. So every tier now retrieves 3 and the caller forwards 1 or
    all 3 depending on the ambiguity gate below.

  * DETERMINISM FIX (pre-existing correctness bug, docs/Proposal_Alignment_Review.md
    S7). No tier had an ORDER BY, just LIMIT 1. When multiple OMOP rows share a
    lowercased concept_name -- routine in a multi-vocabulary Athena dump --
    DuckDB's row order under parallel/vectorized execution is not stable, so
    two runs over the same note could return different concepts (confirmed:
    `ED` resolved to `Ed District` on one run and `Erectile dysfunction` on the
    next). That directly contradicts the proposal's deterministic/traceable
    claim, since the same note could produce two different provenance chains
    depending on when it was run. Every tier now orders explicitly and breaks
    ties on the lowest concept_id.

  * AMBIGUITY DETECTION. The old LIMIT 1 didn't just pick arbitrarily, it
    concealed that there was anything to pick between. A Tier-1 lookup
    returning three distinct standard concepts for the same string is exactly
    the case Stage 3 exists to resolve, and it was previously indistinguishable
    from a clean single match.

  * confidence_tier_in. Computed here because this is the first point where
    all three uncertainty signals are available at once.
"""

import os
import re
import json
import itertools
import duckdb
import torch
from transformers import AutoTokenizer, AutoModel
import warnings

warnings.filterwarnings("ignore")

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

print("Loading SapBERT model for vector normalization... (this may take a moment)")
MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
SAPBERT_POOLING = "cls"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sapbert = AutoModel.from_pretrained(MODEL_NAME)

# DOMAIN FILTERING: docs/Databases.md's "Domain Filtering" open item calls for
# vector-search fallbacks to be domain-filtered by mapping GLiNER labels to
# OMOP domain IDs, to reduce cross-domain mismatches. Applied to ALL THREE
# tiers, not just Tier 3 -- the `ED` -> `Ed District` collision that motivated
# this was an exact-SYNONYM match at Tier 2, not a vector fallback, so gating
# only Tier 3 would have left the observed failure in place. A label with no
# entry gets no domain filter (safe default: matches pre-fix behavior).
GLINER_LABEL_TO_DOMAIN = {
    "Condition": ["Condition"],
    "Symptom": ["Condition", "Observation"],
    "Medication": ["Drug"],
    "Procedure": ["Procedure"],
    "Anatomy": ["Spec Anatomic Site"],
    "Lab Test": ["Measurement"],
    # "Qualifier" is never produced by GLiNER itself (not in CLINICAL_LABELS
    # in src/entity_extraction.py) -- it exists only as a relabeling target
    # for compound-split/span-growth halves whose resolved concept is a
    # standalone modifier (laterality words like "Left"/"Right" being the
    # concrete case that motivated this; see docs/Stage2_Compound_And_
    # Qualifier_Gaps.md gap 3). Both listed domain strings are best-effort
    # guesses at where SNOMED qualifier-value concepts land in this OMOP
    # dump -- NOT verified against the real vocabulary, since this sandbox
    # has no DB access. Getting the guess wrong here is harmless: callers
    # that already know the correct domain (the split/growth detectors) pass
    # it explicitly via normalize_entity()'s domain_override instead of
    # relying on this dict -- see that parameter's docstring for why this
    # entry existing or not doesn't change correctness, only the label's
    # display friendliness on a fallback path.
    "Qualifier": ["Qualifier Value", "Meas Value"],
}

# VOCABULARY RESTRICTION: previously none of the queries filtered on
# vocabulary_id, so an entity could match a concept from ANY vocabulary in
# athena_concept (ICD10CM, LOINC, RxNorm...) rather than SNOMED, which this KG
# is otherwise built around. OMOP codes medications via RxNorm rather than
# SNOMED's Substance hierarchy, so Medication is the one exception.
VOCAB_BY_LABEL = {
    "Medication": ["RxNorm", "RxNorm Extension"],
}
DEFAULT_VOCAB = ["SNOMED"]

# Reverse of GLINER_LABEL_TO_DOMAIN (defined above), used by
# src/clinical_pipeline.py's compound-span splitter (and now span-growth) to
# relabel a split/grown entity for DISPLAY/storage purposes -- e.g. "abdomen"
# split off "gunshot wound to abdomen" (parent label Condition) resolves to
# a Spec Anatomic Site concept, so it is stored as entity_label="Anatomy"
# rather than the misleading inherited "Condition". Not a semantic
# classifier, just "which label is honest about this domain".
#
# NOTE ON CORRECTNESS vs DISPLAY (2026-08-10, gap 3 fix). This dict used to
# ALSO be the mechanism that kept Stage 2b's domain restriction from
# excluding the very candidate the split detector already confirmed --
# relabeling "Left" as "Qualifier" only helped if "Qualifier" then had a
# correct, verified domain entry, which it doesn't (see the "Qualifier"
# comment on GLINER_LABEL_TO_DOMAIN above). That correctness job now belongs
# to normalize_entity()'s domain_override parameter, which
# src/clinical_pipeline.py populates directly from the domain the detector
# found -- no dictionary lookup, no guessing. This dict is now cosmetic: a
# domain with no entry here just keeps the parent's original label (as
# before), which is a harmless, purely-informational fallback rather than a
# correctness bug, because domain_override no longer depends on it.
DOMAIN_TO_GLINER_LABEL = {
    "Condition": "Condition",
    "Procedure": "Procedure",
    "Spec Anatomic Site": "Anatomy",
    "Measurement": "Lab Test",
    "Observation": "Symptom",
    "Drug": "Medication",
    "Drug Exposure": "Medication",
    "Qualifier Value": "Qualifier",
    "Meas Value": "Qualifier",
}

# Documented in docs/Implementation_Methodology.md but previously not
# implemented at all -- normalize_entity() always returned top-1 regardless of
# score, so "0 (Failed)" was effectively unreachable and confidently wrong
# matches (`lasix` -> `Laslades`, `bioplar` -> `Bourgvilain`) flowed through as
# if fully resolved.
TIER3_SIMILARITY_FLOOR = 0.72

# Stage 3 routing parameters. All three are calibration targets against the
# validation slice (docs/Evaluation_Criteria.md) rather than settled values --
# see docs/MoLLM_Stage3_Retrieval_Design.md S8.
CANDIDATE_LIMIT = 3
TIER3_AMBIGUITY_MARGIN = 0.05
GLINER_CONFIDENCE_FLOOR = 0.60


_VOCAB_RELEASE = None


def get_athena_vocabulary_release(conn) -> str:
    """Identifies which Athena vocabulary release produced a mapping.

    docs/Provenance_Schema.md Stage 2b specifies this field and it was never
    written. It is not bookkeeping: SNOMED concept IDs are stable across
    releases but concept NAMES, standard_concept flags and the ancestor closure
    are not, so a mapping produced against one release is not guaranteed
    reproducible against another. Without the stamp there is no way to tell,
    after the fact, whether two runs disagreed because the code changed or
    because the vocabulary did -- which is exactly the ambiguity that made the
    Stage 2b non-determinism bug hard to characterise.

    Prefers a real `vocabulary` table if the Athena dump included one. Falls
    back to a CONTENT SIGNATURE (row count + latest valid_start_date over
    SNOMED) rather than a load timestamp: a timestamp records when the file was
    read, not what was in it, and would make two different vocabularies look
    identical if loaded at the same moment.
    """
    global _VOCAB_RELEASE
    if _VOCAB_RELEASE is not None:
        return _VOCAB_RELEASE

    release = "unknown"
    try:
        tables = {t[0].lower() for t in conn.sql("SHOW TABLES").fetchall()}
        for candidate in ("athena_vocabulary", "vocabulary"):
            if candidate in tables:
                row = conn.sql(
                    f"SELECT vocabulary_version FROM {candidate} "
                    "WHERE vocabulary_id = 'SNOMED' LIMIT 1"
                ).fetchone()
                if row and row[0]:
                    _VOCAB_RELEASE = str(row[0])
                    return _VOCAB_RELEASE
    except Exception:
        pass

    try:
        n, latest = conn.sql("""
            SELECT count(*), max(valid_start_date)
            FROM athena_concept WHERE vocabulary_id = 'SNOMED'
        """).fetchone()
        release = f"signature:snomed_n={n},latest_valid_start={latest}"
    except Exception:
        pass

    _VOCAB_RELEASE = release
    return release


def _in_clause(values):
    return ",".join(["?"] * len(values))


# ==========================================================================
# COMPOUND-SPAN SPLITTING
#
# 2026-08-10, built after scripts/score_gold_recall.py's compound-span
# detection measured this concretely on 17751158-DS-19: GLiNER extracts
# "gunshot wound to abdomen" as ONE Procedure/Condition entity, but the gold
# set links "gunshot wound" (56768003) and "abdomen" (818983003) as TWO
# separate SNOMED annotations. A one-entity-one-concept design can only ever
# satisfy one of them, no matter how good Stage 2b/3 get -- this is the
# fix: give Stage 2a a way to emit two atomic entities instead of one
# compound one, when there's good evidence the text actually carries two
# concepts.
#
# find_compound_split() is the detector; src/clinical_pipeline.py owns
# turning a positive result into new entity_id-bearing rows (offset
# recomputation needs map_offsets_to_original() and the note's abbreviation
# expansion log, neither of which this module has -- see clinical_pipeline's
# docstring for that step).
# ==========================================================================

# Stripped from the ends of a candidate split before it is looked up, so a
# split like "gunshot wound" + "to abdomen" resolves as "abdomen" (a real
# SNOMED body-structure name) rather than failing on the bare preposition.
# Deliberately NOT stripped from the middle -- "left renal" and "left lobe"
# must survive intact, and no connector word sits inside either.
_CONNECTOR_WORDS = {
    "to", "of", "in", "at", "on", "or", "and", "with", "without",
    "the", "a", "an",
}
_TOKEN_RE = re.compile(r"\S+")

# A split part must be at least this long, so a stray single letter or a bare
# punctuation fragment left over after connector-trimming can never itself
# become a "concept" -- there is no legitimate 1-2 character SNOMED FSN this
# would need to match.
_MIN_SPLIT_HALF_CHARS = 3

# 2026-08-10, gap 1 (docs/Stage2_Compound_And_Qualifier_Gaps.md): the large
# gold note's laterality+device+action procedures ("right EVD placement" ->
# 'right' + 'EVD' + 'placement') need a 3-way split, which the original
# binary-only splitter structurally cannot produce. 4 is headroom past the
# largest real case measured so far (3), not an invitation to explore --
# see _partition_token_ranges()'s docstring for why this bound also keeps
# the search cheap.
_MAX_SPLIT_PARTS = 4


def _tokens_with_offsets(text: str) -> list:
    """[(token_text, start, end), ...] for whitespace-delimited tokens,
    offsets relative to `text`."""
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _trim_connectors(tokens: list) -> list:
    """Drops leading/trailing connector-word tokens (see _CONNECTOR_WORDS)
    from a token list. Compares the token stripped of trailing punctuation
    so "vomiting." or "abdomen," still trim correctly at a sentence edge."""
    start, end = 0, len(tokens)
    while start < end and tokens[start][0].strip(".,;:").lower() in _CONNECTOR_WORDS:
        start += 1
    while end > start and tokens[end - 1][0].strip(".,;:").lower() in _CONNECTOR_WORDS:
        end -= 1
    return tokens[start:end]


def _lookup_tier12(conn, text: str, vocabs: list, domains: list = None,
                   include_domains: bool = False):
    """Tier-1 (exact) then Tier-2 (synonym) lookup ONLY -- deliberately no
    Tier-3 SapBERT fallback and no cosine-similarity guessing.

    WHY NO TIER 3 HERE. A split/growth candidate must resolve with the SAME
    confidence bar as a clean whole-phrase match, or this would be doing
    exactly the thing this entire investigation flagged as the problem: a
    shaky semantic "match" on an arbitrary fragment of a phrase (recall the
    Zosyn->Zodasiran, tylenol->tylosin, SGPT->AST cases). Tier 1/2 only
    keeps false-split/false-growth risk low and keeps this cheap enough to
    try at every candidate boundary (no embedding call per attempt).

    Whitespace (including embedded newlines, e.g. "left \\nbase") is
    collapsed before lookup, since SNOMED concept names never contain one --
    the ORIGINAL text with its real whitespace is preserved separately by
    the caller for offset purposes.

    include_domains=True (2026-08-10, docs/Stage2_Compound_And_Qualifier_
    Gaps.md gap 3 refinement) additionally attaches "domain_ids" to the
    returned candidate: EVERY distinct domain_id among ALL Tier 1/2 matches
    for this text, not just the winning row's single domain_id. WHY THIS
    MATTERS: the top-1 candidate this function normally returns is picked
    via `ORDER BY concept_id ASC LIMIT 1`, which silently collapses
    multiple cross-domain matches into one arbitrary winner -- fine for
    "does this part resolve at all", wrong for domain_override (see
    normalize_entity()'s docstring). Measured concretely: "Left" has
    multiple distinct exact SNOMED concepts across different qualifier
    categories; collapsing to the lowest concept_id's domain meant Stage
    2b's restricted re-lookup only ever saw ONE of them and reported a
    confident-but-frequently-wrong HIGH-tier match instead of flagging the
    genuine ambiguity for Stage 3 to resolve. Only computed when requested
    (one extra query) since callers that just need a yes/no ("is this a
    real concept") -- the whole-phrase guard, part-validation during
    partition search -- don't need it and shouldn't pay for it.
    """
    search_text = re.sub(r"\s+", " ", text).strip().lower()
    if not search_text:
        return None

    domain_clause = f" AND domain_id IN ({_in_clause(domains)})" if domains else ""
    row = conn.sql(f"""
        SELECT concept_id, concept_name, domain_id, vocabulary_id
        FROM athena_concept
        WHERE lower(concept_name) = ? AND standard_concept = 'S'
        AND vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause}
        ORDER BY concept_id ASC LIMIT 1;
    """, params=[search_text, *vocabs, *(domains or [])]).fetchone()
    tier = None
    if row:
        tier = "1 (Exact)"
    else:
        domain_clause2 = f" AND c.domain_id IN ({_in_clause(domains)})" if domains else ""
        row = conn.sql(f"""
            SELECT DISTINCT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id
            FROM athena_concept_synonym s
            JOIN athena_concept c ON s.concept_id = c.concept_id
            WHERE lower(s.concept_synonym_name) = ? AND c.standard_concept = 'S'
            AND c.vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause2}
            ORDER BY c.concept_id ASC LIMIT 1;
        """, params=[search_text, *vocabs, *(domains or [])]).fetchone()
        if row:
            tier = "2 (Synonym)"

    if not row:
        return None

    candidate = _candidate(row, tier, 1.0)
    if include_domains:
        candidate["domain_ids"] = _tier12_domains(conn, search_text, vocabs)
    return candidate


def _tier12_domains(conn, normalized_text: str, vocabs: list) -> list:
    """All DISTINCT domain_ids across EVERY Tier-1 (exact) and Tier-2
    (synonym) match for `normalized_text` (already lowercased/whitespace-
    collapsed by the caller), unrestricted by domain. See
    _lookup_tier12()'s include_domains docstring for why this exists as a
    separate query rather than folded into the top-1 lookup above.
    """
    rows = conn.sql(f"""
        SELECT DISTINCT domain_id FROM athena_concept
        WHERE lower(concept_name) = ? AND standard_concept = 'S'
        AND vocabulary_id IN ({_in_clause(vocabs)});
    """, params=[normalized_text, *vocabs]).fetchall()
    domains = {r[0] for r in rows}
    if not domains:
        rows = conn.sql(f"""
            SELECT DISTINCT c.domain_id
            FROM athena_concept_synonym s
            JOIN athena_concept c ON s.concept_id = c.concept_id
            WHERE lower(s.concept_synonym_name) = ? AND c.standard_concept = 'S'
            AND c.vocabulary_id IN ({_in_clause(vocabs)});
        """, params=[normalized_text, *vocabs]).fetchall()
        domains = {r[0] for r in rows}
    return sorted(domains)


def _partition_token_ranges(n_tokens: int, max_parts: int):
    """Yields every way to cut `n_tokens` tokens into 2..max_parts contiguous,
    non-empty groups, as lists of (start_idx, end_idx) token-index ranges
    (end exclusive).

    ORDERED BY ASCENDING PART COUNT FIRST (all 2-way partitions, then all
    3-way, then all 4-way). find_compound_split() returns the first
    acceptable partition it finds, so this ordering is what guarantees it
    always prefers the SMALLEST number of concepts the evidence supports --
    the n-way generalization of the same principle the whole-phrase guard
    already enforces at k=0.

    Bounded by _MAX_SPLIT_PARTS, so this is at most a few dozen combinations
    for the short phrases (2-6 tokens) real entities are.
    """
    for k in range(1, min(max_parts, n_tokens)):
        for cuts in itertools.combinations(range(1, n_tokens), k):
            bounds = (0,) + cuts + (n_tokens,)
            yield [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def find_compound_split(conn, text: str, gliner_label: str) -> dict:
    """Tries every way of cutting `text` into 2..`_MAX_SPLIT_PARTS`
    contiguous word groups and returns the FIRST partition where EVERY part
    independently resolves at Tier 1/2 (after trimming leading/trailing
    connector words from each part). Tries 2-way partitions before 3-way
    before 4-way and, within a given part count, tries them in
    left-to-right cut-position order. Returns None if no such partition
    exists -- the common case. Most multi-word entities are genuinely one
    concept ("Exploratory laparotomy", "Aspiration pneumonia") and this
    must not fire on those; requiring EVERY part to independently clear
    Tier 1/2 is what keeps it from firing on an arbitrary phrase that just
    happens to have a plausible-looking cut point.

    2026-08-10 REVERT NOTICE. A same-day "allow one deferred (unresolved)
    part per partition" relaxation was tried here and reverted after a
    real, measured regression: on the large gold note it split "chronic
    systolic heart failure"-type phrases using "heart" as a resolved
    anchor -- "heart" turned out to have a spurious exact Tier-1 match
    against an unrelated SNOMED concept (concept_name "Initial", almost
    certainly a vocabulary data-quality artifact, not a real match) --
    which peeled a real anchor word away from its qualifiers and left the
    remainder ("systolic heart", missing "chronic"/"failure") to Tier 3
    with LESS context than the original merged phrase had. Combined
    linked_recall dropped 10.78% -> 9.72% (median 13.01% -> 11.15%, large
    8.35% -> 7.42%). The relaxation WAS intended to fix a real problem (see
    docs/Stage2_Compound_And_Qualifier_Gaps.md gap 1 refinement history for
    the "Right chest tube placement" over-atomization case it targeted),
    but "at least one part resolves" turned out to be too weak an anchor
    requirement -- this vocabulary has enough single-token exact-string
    collisions that one resolved token is not reliable evidence a split
    boundary is real, and there is no DB access in this development
    environment to characterize which collisions are trustworthy. Requiring
    EVERY part to resolve is a strictly stronger, previously-verified-safe
    bar (measured net-positive across all three gold notes) and is
    reinstated here; the over-atomization case it does not solve is a
    smaller, unconfirmed-impact issue left as a known, documented
    limitation rather than trading a measured win for a measured loss.

    WHOLE-PHRASE GUARD (2026-08-10, added after an earlier measured
    regression of its own). Before trying any partition, the WHOLE text is
    looked up at Tier 1/2 first. If it resolves on its own, this returns
    None immediately: the text is a legitimate standalone compound concept,
    and must not be demoted to its generic parts just because those parts
    also happen to be independently valid concepts. That is not
    hypothetical -- it is exactly what happened on the first real run of
    this function, before the guard existed: "aspiration pneumonia"
    (already correctly resolved as its own Tier-3 concept, matching gold)
    was split into "aspiration" + "pneumonia", and "retroperitoneal
    hematoma" (already correct) was split into "retroperitoneal" +
    "hematoma" -- both previously-correct links turned wrong, and
    note-level linked_recall measurably dropped (15/144 -> 13/144) despite
    the split also fixing two other cases.

    Measured against the real cases that motivated this (see this module's
    "COMPOUND-SPAN SPLITTING" section header): "gunshot wound to abdomen"
    -> "gunshot wound" + "abdomen", "left renal pseudoaneurysm" -> "left
    renal" + "pseudoaneurysm", "left lobe consolidation" -> "left lobe" +
    "consolidation", "nausea or vomiting" -> "nausea" + "vomiting" (all
    2-way). None of these has a Tier 1/2 match for the WHOLE phrase, which
    is why the guard above does not block them.

    Returns {"parts": [{"text", "start", "end", "candidate"}, ...]}, offsets
    relative to the start of `text`, parts in left-to-right order, or None.
    """
    vocabs = VOCAB_BY_LABEL.get(gliner_label, DEFAULT_VOCAB)

    if _lookup_tier12(conn, text, vocabs):
        return None

    tokens = _tokens_with_offsets(text)
    if len(tokens) < 2:
        return None

    for ranges in _partition_token_ranges(len(tokens), _MAX_SPLIT_PARTS):
        parts = []
        for (i, j) in ranges:
            group_tokens = _trim_connectors(tokens[i:j])
            if not group_tokens:
                parts = None
                break
            g_start, g_end = group_tokens[0][1], group_tokens[-1][2]
            g_text = text[g_start:g_end]
            if len(g_text) < _MIN_SPLIT_HALF_CHARS:
                parts = None
                break
            hit = _lookup_tier12(conn, g_text, vocabs, include_domains=True)
            if not hit:
                parts = None
                break
            parts.append({"text": g_text, "start": g_start, "end": g_end, "candidate": hit})
        if parts:
            return {"parts": parts}
    return None


# ==========================================================================
# SPAN GROWTH -- the mirror image of compound-span splitting
#
# 2026-08-10, docs/Stage2_Compound_And_Qualifier_Gaps.md gap 2, measured on
# 19442119-DS-15 (the median gold note): gold repeatedly annotates a LONGER,
# QUALIFIED phrase as one specific concept, while Stage 2a's predicted span
# is the SHORTER, unqualified sub-phrase, which then resolves to a more
# generic -- and WRONG -- concept. "congestive heart failure" (42343007) is
# not the same disease-concept as "heart failure" (84114007); "pulmonary
# hypertension" (70995007) is a different disease entirely from systemic
# "hypertension", not a narrower version of it. Compound-splitting cannot
# fix this -- there is nothing to split, the predicted span is already too
# SHORT, not compound. This needs the opposite operation: try WIDENING the
# span by absorbing adjacent words, and prefer the wider phrase whenever it
# resolves at the same Tier 1/2 confidence bar find_compound_split()
# requires.
#
# find_span_growth() is the detector, symmetric in spirit to
# find_compound_split() in the same module; src/clinical_pipeline.py owns
# turning a positive result into a replacement entity (offset recomputation
# needs the same map_offsets_to_original() machinery the splitter uses).
# ==========================================================================

# Real cases measured (docs/Stage2_Compound_And_Qualifier_Gaps.md gap 2) are
# all SINGLE-SIDED (a qualifier immediately to the LEFT of the extracted
# span): "congestive"+heart failure, "acute"+pulmonary edema,
# "valvular"+heart disease, "Mid"+LAD, "occlusion of the"+LAD (3 words),
# "stenosis of the"+midcircumflex (3 words), "acute on chronic
# systolic"+heart failure (4 words). 5 gives headroom past the longest of
# these without inviting a much larger search.
_MAX_GROW_TOKENS = 5


def find_span_growth(conn, entity_text: str, left_context: str, right_context: str,
                     gliner_label: str) -> dict:
    """Tries absorbing 1..`_MAX_GROW_TOKENS` adjacent words from the LEFT or
    RIGHT of `entity_text` (never both sides at once -- see below) and
    returns the FIRST growth whose combined text resolves at Tier 1/2,
    smallest absorption first, left before right.

    `left_context`/`right_context` MUST already be trimmed to the entity's
    own sentence by the caller (mirroring how src/entity_extraction.py's
    build_local_context() is sentence-bounded) -- this function has no
    sentence awareness of its own, so a caller that hands it a whole-note
    context risks absorbing words from an unrelated adjacent statement.

    WHY NO TIER 3 HERE. Same reasoning as find_compound_split(): a grown
    candidate must clear the SAME confidence bar as a clean match, or this
    would be trading a narrow-but-real match for a shaky semantic guess on
    a WIDER phrase -- worse than doing nothing.

    WHY THIS RUNS EVEN WHEN THE UN-GROWN TEXT ALREADY RESOLVES. Unlike
    find_compound_split()'s whole-phrase guard (which exists precisely to
    STOP it from overriding an already-correct match), growth exists BECAUSE
    a narrow match succeeding at Tier 1/2 does not mean it is the BEST
    match -- "heart failure" resolves cleanly on its own, and is still
    wrong when gold wants "congestive heart failure". A grown match is
    preferred over the narrow one whenever it ALSO resolves, since a longer
    phrase matching a real, specific SNOMED concept name is virtually
    always more clinically precise than a shorter substring's generic one.
    The collision risk (a coincidental unrelated concept happening to share
    the grown phrase's exact name) is the same low-probability class of
    risk find_compound_split() already accepts for requiring an exact
    string match, not a new one.

    WHY NOT BOTH SIDES AT ONCE. Every real case measured so far is single-
    sided. Trying combined left+right growth would multiply the search
    (a two-sided pass adds up to _MAX_GROW_TOKENS^2 extra lookups) for a
    pattern not yet observed -- left out deliberately as scope control, not
    an oversight; worth adding if a two-sided case is ever measured.

    Returns {"text", "grow_left_chars", "grow_right_chars", "candidate"}
    (char counts absorbed from left_context's END / right_context's START,
    letting the caller compute absolute offsets without re-tokenizing), or
    None.
    """
    vocabs = VOCAB_BY_LABEL.get(gliner_label, DEFAULT_VOCAB)

    left_tokens = _tokens_with_offsets(left_context)[-_MAX_GROW_TOKENS:]
    right_tokens = _tokens_with_offsets(right_context)[:_MAX_GROW_TOKENS]

    trials = []  # (grown_text, n_words_absorbed, grow_left_chars, grow_right_chars)
    for n in range(1, len(left_tokens) + 1):
        chosen = left_tokens[-n:]
        grow_chars = len(left_context) - chosen[0][1]
        grown = left_context[chosen[0][1]:] + entity_text
        trials.append((grown, n, grow_chars, 0))
    for n in range(1, len(right_tokens) + 1):
        chosen = right_tokens[:n]
        grow_chars = chosen[-1][2]
        grown = entity_text + right_context[:grow_chars]
        trials.append((grown, n, 0, grow_chars))

    trials.sort(key=lambda t: t[1])  # fewest absorbed words first

    for grown_text, _n, gl_chars, gr_chars in trials:
        hit = _lookup_tier12(conn, grown_text, vocabs, include_domains=True)
        if hit:
            return {
                "text": grown_text, "grow_left_chars": gl_chars,
                "grow_right_chars": gr_chars, "candidate": hit,
            }
    return None


# ==========================================================================
# LAB VALUE SUFFIX STRIPPING
#
# 2026-08-10, docs/Stage2_Compound_And_Qualifier_Gaps.md. MIMIC discharge
# summaries report labs in compact flowsheet notation -- "WBC-13.0",
# "Glucose-117", "UREA N-25" -- test name and numeric result glued together
# by a hyphen with no space. GLiNER extracts the WHOLE hyphenated token as
# one Lab Test span (there is no whitespace boundary to stop at), so
# normalize_entity() tries to match "wbc-13.0" against SNOMED/LOINC concept
# names -- which obviously never include the numeric result -- and fails at
# every tier. Measured directly: WBC-13.0, PTT-29.0 and Glucose-117 all "0
# (Failed)" on 17751158-DS-19; ALT-736/AST-956 the same way on
# 19442119-DS-15.
#
# This exact pattern was already known on the ASSERTION side -- see
# src/assertion.py's is_structured_result() docstring, which names the same
# GLUCOSE-109/UREA N-25/CREAT-0.3 examples as a reason lab-panel lines must
# not be treated as narrative prose -- but nothing closed the loop on
# normalization actually being able to link the test name.
# ==========================================================================

_LAB_VALUE_SUFFIX_RE = re.compile(r"^(?P<name>.+?)\s*-\s*(?P<value>-?\d+(?:\.\d+)?)\s*$")


def strip_lab_value_suffix(text: str) -> str:
    """Returns the test-name portion of a "TestName-Value" flowsheet-style
    lab result (e.g. "WBC-13.0" -> "WBC", "UREA N-25" -> "UREA N"), or None
    if `text` doesn't match that pattern or the remaining name is too short
    to be meaningful (reuses _MIN_SPLIT_HALF_CHARS's 3-character floor --
    same rationale as the compound splitter: no legitimate SNOMED/LOINC
    name is 1-2 characters).

    The numeric value is discarded, not carried forward -- it is a
    measurement RESULT, not part of the concept name, and this codebase has
    no lab-value-fact table yet to hand it to. That is a real, separate gap
    (the value itself is clinically meaningful and currently just vanishes
    from the entity once stripped), not something this function's caller
    should be expected to solve.
    """
    m = _LAB_VALUE_SUFFIX_RE.match(text)
    if not m:
        return None
    name = m.group("name").strip()
    return name if len(name) >= _MIN_SPLIT_HALF_CHARS else None


def get_sapbert_embedding(text: str) -> list:
    """Generates a 768-dimensional SapBERT vector for a given text."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = sapbert(**tokens)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return embedding


def _candidate(row, tier, score):
    return {
        "omop_concept_id": row[0],
        "concept_name": row[1],
        "domain_id": row[2],
        "vocabulary_id": row[3],
        "match_tier": tier,
        "similarity_score": score,
    }


def _tier_queries(conn, search_text, vocabs, domains, entity_text, vector=None):
    """Runs tiers 1-3 under a given domain restriction. Returns
    (candidates, tier_name, best_tier3_score) or (None, None, score).

    Factored out of normalize_entity() so the exact same three queries can be
    re-run with the domain filter lifted (see _detect_domain_conflict) without
    duplicating the SQL -- two copies of these queries drifting apart would be
    a silent correctness bug of the same family as the LIMIT-1 issue.
    """
    domain_clause = f" AND domain_id IN ({_in_clause(domains)})" if domains else ""
    domain_clause2 = f" AND c.domain_id IN ({_in_clause(domains)})" if domains else ""

    rows = conn.sql(f"""
        SELECT concept_id, concept_name, domain_id, vocabulary_id
        FROM athena_concept
        WHERE lower(concept_name) = ? AND standard_concept = 'S'
        AND vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause}
        ORDER BY concept_id ASC LIMIT {CANDIDATE_LIMIT};
    """, params=[search_text, *vocabs, *(domains or [])]).fetchall()
    if rows:
        return [_candidate(r, "1 (Exact)", 1.0) for r in rows], "1", None

    rows = conn.sql(f"""
        SELECT DISTINCT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id
        FROM athena_concept_synonym s
        JOIN athena_concept c ON s.concept_id = c.concept_id
        WHERE lower(s.concept_synonym_name) = ? AND c.standard_concept = 'S'
        AND c.vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause2}
        ORDER BY c.concept_id ASC LIMIT {CANDIDATE_LIMIT};
    """, params=[search_text, *vocabs, *(domains or [])]).fetchall()
    if rows:
        return [_candidate(r, "2 (Synonym)", 1.0) for r in rows], "2", None

    if vector is None:
        vector = get_sapbert_embedding(entity_text)
    rows = conn.sql(f"""
        SELECT concept_id, concept_name, domain_id, vocabulary_id,
               list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
        FROM athena_concept
        WHERE embedding IS NOT NULL AND standard_concept = 'S'
        AND vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause}
        ORDER BY similarity DESC, concept_id ASC LIMIT {CANDIDATE_LIMIT};
    """, params=[vector, *vocabs, *(domains or [])]).fetchall()
    if not rows:
        return None, None, 0.0
    cands = [_candidate(r, "3 (Semantic)", round(r[4], 4)) for r in rows]
    return cands, "3", cands[0]["similarity_score"]


def _detect_domain_conflict(conn, search_text, vocabs, domains, entity_text, vector):
    """Re-runs the tiers WITHOUT the domain restriction after a filtered miss.

    WHY THIS EXISTS. The domain filter is a hard WHERE clause on every tier, so
    when GLiNER labels a span `Medication` but the correct concept sits in the
    Condition domain, the query does not return that concept flagged as
    mismatched -- it returns nothing, and the entity is reported "0 (Failed)".
    That silently conflates two situations that need completely different
    handling downstream:

        (a) the concept genuinely is not in the vocabulary, and
        (b) the concept IS in the vocabulary, under a different domain than the
            extraction label predicted -- i.e. GLiNER probably mislabelled it.

    (b) is recoverable and is exactly the disambiguation Stage 3 exists to do,
    but the filter was destroying the evidence for it. It also inflates the
    apparent failure rate: a correctly-extracted-but-mislabelled entity was
    indistinguishable from an unmappable one in any evaluation.

    Only runs after a filtered miss, so it costs nothing on the common path.
    The SapBERT vector is passed in rather than recomputed for the same reason.
    """
    if not domains:
        return None
    cands, tier, _ = _tier_queries(conn, search_text, vocabs, None, entity_text, vector)
    if not cands:
        return None
    if tier == "3" and cands[0]["similarity_score"] < TIER3_SIMILARITY_FLOOR:
        return None
    return {
        "found_domain": cands[0]["domain_id"],
        "expected_domains": domains,
        "tier": tier,
        "candidates": cands,
    }


def normalize_entity(entity_text: str, conn, gliner_label: str = None,
                     domain_override: list = None) -> dict:
    """Maps an entity to OMOP concepts using the 3-tier approach.

    Returns the top candidate's fields (backwards-compatible keys) plus:
      candidates[]      -- up to CANDIDATE_LIMIT ranked candidates
      ambiguous         -- whether Stage 3 should be asked to disambiguate
      ambiguity_reason  -- why, so a reviewer never has to guess

    gliner_label drives both the vocabulary restriction and the (default)
    domain restriction applied to every tier. Passing None applies neither,
    i.e. the pre-2026-08-07 behavior -- callers should pass the real label
    whenever they have one.

    domain_override (2026-08-10, docs/Stage2_Compound_And_Qualifier_Gaps.md
    gap 3), if given, is used INSTEAD of GLINER_LABEL_TO_DOMAIN.get(gliner_label)
    for the domain restriction. WHY THIS EXISTS: src/clinical_pipeline.py's
    compound-split/span-growth detectors run their OWN unrestricted Tier 1/2
    lookup (src/normalization._lookup_tier12()) to confirm a split/grown
    part resolves at all, and that lookup already knows which OMOP domain
    the match actually lives in. Re-deriving a domain from the (possibly
    now-wrong, e.g. a bare qualifier word relabeled from its Procedure
    parent) gliner_label and hoping GLINER_LABEL_TO_DOMAIN happens to have a
    correct entry for it silently throws that evidence away -- concretely,
    "Left" split off "Left craniectomy" resolves during detection, but
    Stage 2b's un-overridden domain restriction (Procedure) excludes
    whatever domain the qualifier-value concept actually lives in, so the
    split's own confirmed match was previously lost on the very next call.
    Passing the detector's own domain straight through is self-consistent
    BY CONSTRUCTION -- it never requires knowing or guessing the correct
    OMOP domain_id string.

    CAN BE MULTIPLE DOMAINS (2026-08-10 refinement, gap 3 follow-up). The
    detector passes EVERY distinct domain_id its own unrestricted lookup
    found across ALL matching rows (_lookup_tier12()'s include_domains
    option), not just its single top-ranked candidate's domain. This
    matters because a real ambiguity was being silently hidden: "Left" has
    multiple distinct exact SNOMED concepts spread across different
    qualifier categories, and collapsing to just the lowest concept_id's
    domain meant this function's tier-1 query only ever saw ONE of them --
    reporting a confident HIGH-tier match that frequently disagreed with
    gold, instead of the len(candidates)>1 ambiguous=True path that already
    exists a few lines below for exactly this situation. Passing the full
    domain set lets that existing ambiguity-detection machinery do its job
    instead of being starved of the evidence it needs.
    """
    search_text = entity_text.lower().strip()
    vocabs = VOCAB_BY_LABEL.get(gliner_label, DEFAULT_VOCAB)
    domains = domain_override if domain_override is not None else GLINER_LABEL_TO_DOMAIN.get(gliner_label)
    domain_clause = f" AND domain_id IN ({_in_clause(domains)})" if domains else ""

    # Records which tiers were attempted and what each returned. Previously
    # only the WINNING tier survived in match_tier, so "Tier 1 and 2 found
    # nothing and Tier 3 scored 0.71" was indistinguishable from "Tier 3 scored
    # 0.71" with the earlier tiers never tried. Stage 3's deeper-resolution
    # mode is supposed to reason over exactly this search path, and it was
    # being thrown away before it got there.
    trace = []

    # ==========================================
    # TIER 1: Exact Lexical Match
    # ==========================================
    tier1_query = f"""
    SELECT concept_id, concept_name, domain_id, vocabulary_id
    FROM athena_concept
    WHERE lower(concept_name) = ?
    AND standard_concept = 'S'
    AND vocabulary_id IN ({_in_clause(vocabs)})
    {domain_clause}
    ORDER BY concept_id ASC
    LIMIT {CANDIDATE_LIMIT};
    """
    rows = conn.sql(tier1_query, params=[search_text, *vocabs, *(domains or [])]).fetchall()
    trace.append({"tier": "1 (Exact)", "attempted": True, "hits": len(rows)})
    if rows:
        cands = [_candidate(r, "1 (Exact)", 1.0) for r in rows]
        return _result(
            cands,
            ambiguous=len(cands) > 1,
            reason="multiple_exact_concept_name_matches" if len(cands) > 1 else None,
            trace=trace,
        )

    # ==========================================
    # TIER 2: Synonym Lexical Match
    # ==========================================
    domain_clause2 = f" AND c.domain_id IN ({_in_clause(domains)})" if domains else ""
    tier2_query = f"""
    SELECT DISTINCT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id
    FROM athena_concept_synonym s
    JOIN athena_concept c ON s.concept_id = c.concept_id
    WHERE lower(s.concept_synonym_name) = ?
    AND c.standard_concept = 'S'
    AND c.vocabulary_id IN ({_in_clause(vocabs)})
    {domain_clause2}
    ORDER BY c.concept_id ASC
    LIMIT {CANDIDATE_LIMIT};
    """
    rows = conn.sql(tier2_query, params=[search_text, *vocabs, *(domains or [])]).fetchall()
    trace.append({"tier": "2 (Synonym)", "attempted": True, "hits": len(rows)})
    if rows:
        cands = [_candidate(r, "2 (Synonym)", 1.0) for r in rows]
        return _result(
            cands,
            ambiguous=len(cands) > 1,
            reason="multiple_exact_synonym_matches" if len(cands) > 1 else None,
            trace=trace,
        )

    # ==========================================
    # TIER 3: Semantic Vector Match (SapBERT)
    # ==========================================
    vector = get_sapbert_embedding(entity_text)
    tier3_query = f"""
    SELECT concept_id, concept_name, domain_id, vocabulary_id,
           list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
    FROM athena_concept
    WHERE embedding IS NOT NULL
    AND standard_concept = 'S'
    AND vocabulary_id IN ({_in_clause(vocabs)})
    {domain_clause}
    ORDER BY similarity DESC, concept_id ASC
    LIMIT {CANDIDATE_LIMIT};
    """
    rows = conn.sql(tier3_query, params=[vector, *vocabs, *(domains or [])]).fetchall()
    trace.append({
        "tier": "3 (Semantic)", "attempted": True, "hits": len(rows),
        "top_score": round(rows[0][4], 4) if rows else None,
        "runner_up_score": round(rows[1][4], 4) if len(rows) > 1 else None,
        "floor": TIER3_SIMILARITY_FLOOR,
    })

    if not rows:
        conflict = _detect_domain_conflict(conn, search_text, vocabs, domains, entity_text, vector)
        if conflict:
            return _result(conflict["candidates"], ambiguous=True,
                           reason="label_domain_conflict", failed=True, conflict=conflict,
                           trace=trace)
        return _result([], ambiguous=True, reason="no_candidates_at_any_tier", failed=True,
                       trace=trace)

    cands = [_candidate(r, "3 (Semantic)", round(r[4], 4)) for r in rows]
    top = cands[0]["similarity_score"]

    # Below the floor, the top hit is reported as a FAILED match but the
    # candidates are still forwarded: "close but not close enough" is exactly
    # the case Stage 3's deeper-resolution mode exists for, and discarding the
    # near-misses would leave it nothing to reason about.
    if top < TIER3_SIMILARITY_FLOOR:
        # A weak in-domain match may still be beaten by a strong out-of-domain
        # one, which is itself the signal that the label was wrong -- so the
        # conflict check runs here too, not only on a total miss.
        conflict = _detect_domain_conflict(conn, search_text, vocabs, domains, entity_text, vector)
        if conflict:
            return _result(conflict["candidates"], ambiguous=True,
                           reason="label_domain_conflict", failed=True, conflict=conflict,
                           trace=trace)
        return _result(cands, ambiguous=True, reason="tier3_below_similarity_floor",
                       failed=True, trace=trace)

    if len(cands) > 1 and (top - cands[1]["similarity_score"]) < TIER3_AMBIGUITY_MARGIN:
        return _result(cands, ambiguous=True, reason="tier3_top2_margin_below_threshold",
                       trace=trace)

    return _result(cands, ambiguous=False, reason=None, trace=trace)


def _result(candidates: list, ambiguous: bool, reason, failed: bool = False,
            conflict: dict = None, trace: list = None) -> dict:
    """Shapes the return value, keeping the pre-existing top-level keys so
    existing consumers keep working while `candidates` carries the new
    information."""
    if not candidates or failed:
        top = {
            "match_tier": "0 (Failed)", "concept_id": None, "concept_name": "Unmapped",
            "domain_id": None, "vocab": None,
            "score": candidates[0]["similarity_score"] if candidates else 0.0,
        }
    else:
        c = candidates[0]
        top = {
            "match_tier": c["match_tier"], "concept_id": c["omop_concept_id"],
            "concept_name": c["concept_name"], "domain_id": c["domain_id"],
            "vocab": c["vocabulary_id"], "score": c["similarity_score"],
        }
    top["candidates"] = candidates
    top["ambiguous"] = ambiguous
    top["ambiguity_reason"] = reason
    top["domain_conflict"] = conflict
    top["tier_trace"] = trace
    return top


def compute_confidence_tier(gliner_confidence: float, normalization_ambiguous: bool,
                            expansion_ambiguous: bool, domain_conflict: bool = False,
                            crosses_sentence_boundary: bool = False) -> tuple:
    """Stage 3's routing tier, and the reasons behind it.

    LOW if ANY of five independent signals is weak. They capture genuinely
    different failure modes and none subsumes the others:
      * gliner_confidence         -- "is this even an entity"
      * normalization             -- "which concept is it"
      * expansion                 -- "was the abbreviation read correctly at all"
      * domain_conflict           -- "was the extraction LABEL even right"
      * crosses_sentence_boundary -- "is this even ONE entity"
    An entity can have a single clean, unambiguous OMOP match and still deserve
    deeper resolution because the span itself was shaky, or because it came
    from an abbreviation with three plausible readings.
    """
    reasons = []
    if gliner_confidence is not None and gliner_confidence < GLINER_CONFIDENCE_FLOOR:
        reasons.append("low_gliner_confidence")
    if normalization_ambiguous:
        reasons.append("normalization_ambiguous")
    if expansion_ambiguous:
        reasons.append("expansion_ambiguous")
    # A cross-domain match means the extraction LABEL is in question, not just
    # the concept -- always worth Stage 3's deeper-resolution path, since a
    # mislabelled entity that silently resolves would carry a wrong domain into
    # KG3 and into every guideline lookup made against it thereafter.
    if domain_conflict:
        reasons.append("label_domain_conflict")
    # 2026-08-10: measured concretely on 17751158-DS-19 via
    # scripts/score_gold_recall.py's compound-span detection -- GLiNER
    # extracted "Gunshot wound to abdomen\nPneumonia" as ONE entity spanning
    # two PyRuSH sentences, merging an unrelated diagnosis-list line into a
    # single concept that can never link correctly to either gold annotation
    # it overlaps. See src/entity_extraction.py's sentence_ids_spanned() for
    # why sentence boundaries (not raw "\n") are the right signal -- a
    # line-wrapped continuation of one statement stays inside one sentence,
    # so this does not fire on those.
    if crosses_sentence_boundary:
        reasons.append("entity_span_crosses_sentence_boundary")
    return ("LOW" if reasons else "HIGH"), reasons


def process_and_normalize_entities(extracted_entities: list, conn, is_test: bool = False) -> list:
    """Normalizes Stage 2a entities against OMOP and persists results.

    Takes the list of DICTS that entity_extraction.extract_and_store_entities()
    now returns (it previously took positional tuples -- see that module's
    docstring for why the shape changed).

    DEDUP FAN-OUT: normalization is computed once per distinct
    (expanded_text, gliner_label) pair, then that result is written out for
    EVERY entity_id sharing it. The old orchestrator deduplicated entities
    before this function ever saw them, which saved the redundant SapBERT calls
    but also meant only one of several identical mentions got a normalization
    row at all -- so mentions 2..n had no concept mapping, and once Stage 3
    runs per entity_id they would have had nothing to validate. Computing once
    and fanning out keeps the saving without the data loss.
    """
    conn.sql("""
    CREATE TABLE IF NOT EXISTS normalized_entities (
        note_id VARCHAR,
        original_text VARCHAR,
        expanded_text VARCHAR,
        gliner_label VARCHAR,
        gliner_confidence FLOAT,
        omop_concept_id BIGINT,
        omop_concept_name VARCHAR,
        omop_domain VARCHAR,
        omop_vocab VARCHAR,
        match_tier VARCHAR,
        similarity_score FLOAT,
        is_test BOOLEAN DEFAULT FALSE,
        UNIQUE(note_id, original_text, expanded_text, gliner_label)
    );
    """)
    for ddl in [
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS is_test BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS entity_id VARCHAR;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS candidates JSON;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS is_ambiguous BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS ambiguity_reason VARCHAR;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS confidence_tier_in VARCHAR;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS tier_reasons JSON;",
        # Records that a good concept was found under a DIFFERENT OMOP domain
        # than the GLiNER label predicted. Lets the evaluation decompose
        # "0 (Failed)" into genuinely-unmappable vs mislabelled-but-mappable,
        # which were previously indistinguishable.
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS domain_conflict JSON;",
        # Which tiers ran and what each returned, including the Tier-3
        # runner-up score. Lets an auditor see the search path, not just its
        # outcome.
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS tier_trace JSON;",
        # docs/Provenance_Schema.md Stage 2b fields that were never written.
        # `matched` is redundant with match_tier != '0 (Failed)' but is what the
        # schema specifies and what downstream filters will actually read --
        # a boolean is far harder to get wrong than string-matching a tier label.
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS matched BOOLEAN;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS normalized_from VARCHAR;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS athena_vocabulary_release VARCHAR;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS domain_id_queried JSON;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS vocab_queried JSON;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS sapbert_pooling_method VARCHAR;",
    ]:
        conn.sql(ddl)

    vocab_release = get_athena_vocabulary_release(conn)
    cache = {}
    normalized_results = []
    rows_to_insert = []

    for ent in extracted_entities:
        note_id = ent["note_id"]
        orig_text = ent["original_text"]
        expanded_text = ent["expanded_text"]
        label = ent["entity_label"]
        # See normalize_entity()'s domain_override docstring -- set only on
        # compound-split/span-growth-produced entities
        # (src/clinical_pipeline.py); ordinary Stage 2a entities have no
        # such key, so .get() returns None and behavior is unchanged.
        domain_override = ent.get("domain_override")

        # Normalization is a pure function of (text, label, domain_override)
        # within a note, so the cache is the fan-out mechanism: identical
        # mentions share one computation but each still gets its own row
        # keyed by entity_id. domain_override is part of the key because two
        # entities can share (text, label) yet carry different overrides --
        # e.g. the SAME bare qualifier word split off two different parent
        # labels in one note.
        cache_key = (expanded_text, label,
                    tuple(domain_override) if domain_override else None)
        if cache_key not in cache:
            mapping = normalize_entity(expanded_text, conn, gliner_label=label,
                                       domain_override=domain_override)
            normalized_from = "expanded"

            # ORIGINAL-FORM FALLBACK. Normalisation runs on expanded_text, so a
            # wrong abbreviation expansion poisons it with no way to recover --
            # and wrong expansions are real: measured on note 10000032-DS-21,
            # `NAD` expanded to "nicotinamide adenine dinucleotide" (the
            # alphabetically-first meaning) rather than "no acute distress",
            # and `TTP` to "Thrombotic Thrombocytopenic Purpura" rather than
            # "tenderness to palpation". When the expanded form fails to map,
            # the raw abbreviation is often the better key -- `SOB`, `EGD` and
            # `INR` all resolve directly.
            #
            # Only fires on failure, and only when the two forms actually
            # differ, so it costs nothing on the common path. `normalized_from`
            # records which one produced the result, so a reviewer can see that
            # the expansion was bypassed rather than having to infer it.
            if (mapping["match_tier"] == "0 (Failed)"
                    and orig_text and orig_text.strip().lower() != expanded_text.strip().lower()):
                retry = normalize_entity(orig_text, conn, gliner_label=label,
                                         domain_override=domain_override)
                if retry["match_tier"] != "0 (Failed)":
                    mapping = retry
                    normalized_from = "original_after_expanded_failed"

            # LAB VALUE SUFFIX FALLBACK (2026-08-10, see
            # strip_lab_value_suffix()'s docstring above for the full
            # rationale). Only fires on a still-failed Lab Test entity, so
            # it costs nothing on the common path and never touches labels
            # where a "-NUMBER" suffix wouldn't mean the same thing. Tries
            # whichever text form(s) actually carry the pattern -- expanded
            # first (matching the two fallbacks above), then original, in
            # case abbreviation expansion altered the hyphen/spacing.
            if mapping["match_tier"] == "0 (Failed)" and label == "Lab Test":
                for candidate_text, source_name in (
                        (expanded_text, "expanded"), (orig_text, "original")):
                    stripped = strip_lab_value_suffix(candidate_text or "")
                    if not stripped:
                        continue
                    retry = normalize_entity(stripped, conn, gliner_label=label,
                                             domain_override=domain_override)
                    if retry["match_tier"] != "0 (Failed)":
                        mapping = retry
                        normalized_from = f"value_stripped_from_{source_name}"
                        break

            mapping = dict(mapping)
            mapping["normalized_from"] = normalized_from
            cache[cache_key] = mapping
        mapping = cache[cache_key]

        conflict = mapping.get("domain_conflict")
        tier, tier_reasons = compute_confidence_tier(
            ent.get("confidence"),
            mapping["ambiguous"],
            ent.get("expansion_ambiguous", False),
            domain_conflict=bool(conflict),
            crosses_sentence_boundary=ent.get("crosses_sentence_boundary", False),
        )

        # Only forward alternatives when there is something to choose between;
        # a confident, unambiguous match ships a single candidate so Stage 3's
        # prompt doesn't spend tokens on options nobody is choosing between.
        forwarded = mapping["candidates"] if mapping["ambiguous"] else mapping["candidates"][:1]

        record = {
            "entity_id": ent["entity_id"],
            "note_id": note_id,
            "original_text": orig_text,
            "expanded_text": expanded_text,
            "gliner_label": label,
            "gliner_confidence": round(ent["confidence"], 4) if ent.get("confidence") is not None else None,
            "omop_concept_id": mapping["concept_id"],
            "omop_concept_name": mapping["concept_name"],
            "omop_domain": mapping["domain_id"],
            "omop_vocab": mapping["vocab"],
            "match_tier": mapping["match_tier"],
            "similarity_score": mapping["score"],
            "candidates": forwarded,
            "is_ambiguous": mapping["ambiguous"],
            "ambiguity_reason": mapping["ambiguity_reason"],
            "confidence_tier_in": tier,
            "tier_reasons": tier_reasons,
            "domain_conflict": conflict,
            "tier_trace": mapping.get("tier_trace"),
            "matched": mapping["match_tier"] != "0 (Failed)",
            "normalized_from": mapping.get("normalized_from", "expanded"),
        }
        normalized_results.append(record)

        rows_to_insert.append((
            note_id, orig_text, expanded_text, label, record["gliner_confidence"],
            mapping["concept_id"], mapping["concept_name"], mapping["domain_id"],
            mapping["vocab"], mapping["match_tier"], mapping["score"], is_test,
            ent["entity_id"], json.dumps(forwarded), mapping["ambiguous"],
            mapping["ambiguity_reason"], tier, json.dumps(tier_reasons),
            json.dumps(conflict) if conflict else None,
            json.dumps(mapping.get("tier_trace")),
            mapping["match_tier"] != "0 (Failed)",
            mapping.get("normalized_from", "expanded"),
            vocab_release,
            json.dumps(GLINER_LABEL_TO_DOMAIN.get(label)),
            json.dumps(VOCAB_BY_LABEL.get(label, DEFAULT_VOCAB)),
            SAPBERT_POOLING,
        ))

    if rows_to_insert:
        conn.executemany("""
        INSERT INTO normalized_entities
        (note_id, original_text, expanded_text, gliner_label, gliner_confidence,
         omop_concept_id, omop_concept_name, omop_domain, omop_vocab,
         match_tier, similarity_score, is_test, entity_id, candidates,
         is_ambiguous, ambiguity_reason, confidence_tier_in, tier_reasons,
         domain_conflict, tier_trace, matched, normalized_from,
         athena_vocabulary_release, domain_id_queried, vocab_queried,
         sapbert_pooling_method)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (note_id, original_text, expanded_text, gliner_label) DO UPDATE SET
            gliner_confidence = EXCLUDED.gliner_confidence,
            omop_concept_id = EXCLUDED.omop_concept_id,
            omop_concept_name = EXCLUDED.omop_concept_name,
            omop_domain = EXCLUDED.omop_domain,
            omop_vocab = EXCLUDED.omop_vocab,
            match_tier = EXCLUDED.match_tier,
            similarity_score = EXCLUDED.similarity_score,
            is_test = EXCLUDED.is_test,
            entity_id = EXCLUDED.entity_id,
            candidates = EXCLUDED.candidates,
            is_ambiguous = EXCLUDED.is_ambiguous,
            ambiguity_reason = EXCLUDED.ambiguity_reason,
            confidence_tier_in = EXCLUDED.confidence_tier_in,
            tier_reasons = EXCLUDED.tier_reasons,
            domain_conflict = EXCLUDED.domain_conflict,
            tier_trace = EXCLUDED.tier_trace,
            matched = EXCLUDED.matched,
            normalized_from = EXCLUDED.normalized_from,
            athena_vocabulary_release = EXCLUDED.athena_vocabulary_release,
            domain_id_queried = EXCLUDED.domain_id_queried,
            vocab_queried = EXCLUDED.vocab_queried,
            sapbert_pooling_method = EXCLUDED.sapbert_pooling_method;
        """, rows_to_insert)

    return normalized_results
