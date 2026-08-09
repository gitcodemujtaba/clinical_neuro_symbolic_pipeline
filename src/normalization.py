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
import json
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


def normalize_entity(entity_text: str, conn, gliner_label: str = None) -> dict:
    """Maps an entity to OMOP concepts using the 3-tier approach.

    Returns the top candidate's fields (backwards-compatible keys) plus:
      candidates[]      -- up to CANDIDATE_LIMIT ranked candidates
      ambiguous         -- whether Stage 3 should be asked to disambiguate
      ambiguity_reason  -- why, so a reviewer never has to guess

    gliner_label drives both the vocabulary restriction and the domain
    restriction applied to every tier. Passing None applies neither, i.e. the
    pre-2026-08-07 behavior -- callers should pass the real label whenever they
    have one.
    """
    search_text = entity_text.lower().strip()
    vocabs = VOCAB_BY_LABEL.get(gliner_label, DEFAULT_VOCAB)
    domains = GLINER_LABEL_TO_DOMAIN.get(gliner_label)
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
                            expansion_ambiguous: bool, domain_conflict: bool = False) -> tuple:
    """Stage 3's routing tier, and the reasons behind it.

    LOW if ANY of three independent signals is weak. They capture genuinely
    different failure modes and none subsumes the others:
      * gliner_confidence  -- "is this even an entity"
      * normalization      -- "which concept is it"
      * expansion          -- "was the abbreviation read correctly at all"
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

        # Normalization is a pure function of (text, label) within a note, so
        # the cache is the fan-out mechanism: identical mentions share one
        # computation but each still gets its own row keyed by entity_id.
        cache_key = (expanded_text, label)
        if cache_key not in cache:
            mapping = normalize_entity(expanded_text, conn, gliner_label=label)
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
                retry = normalize_entity(orig_text, conn, gliner_label=label)
                if retry["match_tier"] != "0 (Failed)":
                    mapping = retry
                    normalized_from = "original_after_expanded_failed"

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
