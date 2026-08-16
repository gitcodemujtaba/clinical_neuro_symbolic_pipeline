"""
src/normalization/bm25_index.py — Pass 3 (plan Phase 3): DuckDB FTS sparse
search over athena_concept.concept_name, the "BM25" half of the hybrid
dense+sparse+prior Reciprocal Rank Fusion retrieval
src/normalization/tier_retrieval.py's _tier3_hybrid_rows() combines.

WHY DUCKDB'S NATIVE FTS RATHER THAN A PYTHON BM25 LIBRARY (e.g. rank_bm25).
This codebase's own convention throughout tier_retrieval.py is "the database
does the work" -- every existing tier query is plain SQL, no candidate list
is ever pulled into Python memory to be re-ranked there. athena_concept has
6.6M rows; loading concept_name strings for all of them into an in-process
rank_bm25 index would be a real memory/startup cost this module avoids
entirely by building the index once, in the database, and querying it like
any other tier.

INDEX SCOPE: concept_name only, not athena_concept_synonym. The existing
Tier 2 exact-synonym lookup (_tier_queries() in tier_retrieval.py) already
covers exact synonym matches; extending BM25 to synonyms too is a real,
separate scope increase (2.8M more rows, a second index, a UNION query) left
for a follow-up once concept_name-only coverage is measured -- not silently
assumed to be needed.

THE QUERY PATTERN MATTERS FOR PERFORMANCE -- MEASURED, NOT ASSUMED. The
naive, textbook-looking query (wrap match_bm25 in a subquery, then JOIN back
to athena_concept for concept_name/domain_id/vocabulary_id) measured 109s
per query against this table -- unusable even at the plan's generous 2-5
min/note budget. The fix, confirmed empirically: athena_concept already
carries every column the caller needs (concept_name, domain_id,
vocabulary_id) alongside concept_id, so the self-join is pure waste; a
single flat query against the base table with match_bm25() in the SELECT
list and `WHERE score IS NOT NULL` measured 0.2-1.6s with realistic
vocab/domain filters. query_bm25() below uses only the fast form -- this is
not a stylistic preference, the slow form is a real performance bug if
reintroduced.
"""
import duckdb

from .text_utils import _in_clause

FTS_INDEXED_TABLE = "athena_concept"
FTS_SCHEMA_NAME = f"fts_main_{FTS_INDEXED_TABLE}"


def fts_index_exists(conn) -> bool:
    rows = conn.execute(
        "SELECT 1 FROM duckdb_schemas() WHERE schema_name = ?", [FTS_SCHEMA_NAME]
    ).fetchall()
    return bool(rows)


def build_bm25_index(conn, overwrite: bool = False) -> bool:
    """Builds (or rebuilds) the FTS index over athena_concept.concept_name.

    Idempotent by default: does nothing and returns False if the index
    already exists, since PRAGMA create_fts_index with overwrite=1 always
    does a full rebuild (measured ~10s warm, but real work every call) --
    every pipeline start should not pay that cost. Pass overwrite=True only
    when athena_concept's contents have actually changed (e.g. after a fresh
    Athena vocabulary import via scripts/import_athena.py) and the index
    needs to reflect that.

    stemmer='none'/stopwords='none': clinical concept names are short,
    technical noun phrases (e.g. "Fracture of clavicle"), not prose -- an
    English stemmer/stopword list is tuned for natural-language documents and
    risks collapsing clinically distinct terms (e.g. stemming could conflate
    unrelated word forms) for no measured benefit here. ignore=non-alphanumeric
    strips punctuation without attempting linguistic normalization.
    """
    if not overwrite and fts_index_exists(conn):
        return False
    conn.execute("INSTALL fts")
    conn.execute("LOAD fts")
    conn.execute(f"""
        PRAGMA create_fts_index(
            '{FTS_INDEXED_TABLE}', 'concept_id', 'concept_name',
            stemmer='none', stopwords='none', ignore='(\\.|[^a-z0-9])+',
            strip_accents=1, lower=1, overwrite=1
        )
    """)
    return True


def query_bm25(conn, search_text: str, vocabs=None, domains=None, limit: int = 20) -> list:
    """Returns up to `limit` rows (concept_id, concept_name, domain_id,
    vocabulary_id, bm25_score), ranked by BM25 score descending, or []
    if the FTS index has not been built yet (caller's responsibility to
    call build_bm25_index() first -- this function does not build it
    implicitly, matching _tier3_semantic_rows()'s own "vector must already
    be computed" contract rather than hiding a slow first-call cost inside
    what looks like a cheap lookup).

    standard_concept = 'S' is always applied (matches every other tier
    query in this package -- non-standard concepts are never valid
    normalization targets here).
    """
    conn.execute("LOAD fts")
    if not fts_index_exists(conn):
        return []
    vocab_clause = f" AND vocabulary_id IN ({_in_clause(vocabs)})" if vocabs else ""
    domain_clause = f" AND domain_id IN ({_in_clause(domains)})" if domains else ""
    try:
        rows = conn.execute(f"""
            SELECT concept_id, concept_name, domain_id, vocabulary_id,
                   {FTS_SCHEMA_NAME}.match_bm25(concept_id, ?) AS bm25_score
            FROM {FTS_INDEXED_TABLE}
            WHERE bm25_score IS NOT NULL AND standard_concept = 'S'
            {vocab_clause} {domain_clause}
            ORDER BY bm25_score DESC, concept_id ASC LIMIT {int(limit)}
        """, [search_text, *(vocabs or []), *(domains or [])]).fetchall()
    except duckdb.Error:
        return []
    return rows
