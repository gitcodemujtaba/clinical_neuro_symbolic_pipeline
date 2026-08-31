"""ui/components/concept_search.py -- 2026-08-31: live OMOP/SNOMED concept
search by name, for the HITL review page's "correct to a concept" tool.

WHY THIS EXISTS. Before this, correcting to a concept NOT already in
Stage 2b's own candidate list meant a reviewer had to already know the
exact numeric OMOP concept_id and type it into a bare text box -- the
real, practical blocker on "can a human actually fix a retrieval miss"
(the single most common real failure mode this whole project measures
against). This is a plain name search over athena_concept, nothing more
-- no embedding/SapBERT call, so it stays fast enough for live UI use on
every keystroke-triggered rerun.
"""


def search_concepts(conn, query: str, limit: int = 15) -> list:
    """Real, live search -- concept_name ILIKE, standard concepts (S) first,
    exact-match-prefix ranked ahead of a bare substring hit. Returns
    [{"concept_id", "concept_name", "domain_id", "vocabulary_id",
    "concept_class_id", "standard_concept"}], empty list for a blank/too-
    short query (avoids a full-table substring scan on every keystroke of
    a 1-2 character query).
    """
    query = (query or "").strip()
    if len(query) < 3:
        return []
    like_pattern = f"%{query}%"
    prefix_pattern = f"{query}%"
    rows = conn.execute("""
        SELECT concept_id, concept_name, domain_id, vocabulary_id,
               concept_class_id, standard_concept
        FROM athena_concept
        WHERE concept_name ILIKE ?
          AND invalid_reason IS NULL
        ORDER BY
            standard_concept = 'S' DESC,
            concept_name ILIKE ? DESC,
            length(concept_name) ASC
        LIMIT ?
    """, [like_pattern, prefix_pattern, limit]).fetchall()
    return [
        {"concept_id": r[0], "concept_name": r[1], "domain_id": r[2],
         "vocabulary_id": r[3], "concept_class_id": r[4], "standard_concept": r[5]}
        for r in rows
    ]


def format_concept_option(c: dict) -> str:
    """One-line label for a search-result dropdown entry."""
    std = "" if c.get("standard_concept") == "S" else "  ⚠️ non-standard"
    return (f"{c['concept_name']}  —  {c['domain_id']}/{c['concept_class_id']} "
           f"({c['vocabulary_id']}, id={c['concept_id']}){std}")
