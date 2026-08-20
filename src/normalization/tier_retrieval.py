"""src/normalization/tier_retrieval.py — Tier 1-4 candidate retrieval, ranking, alias expansion, hierarchy collapse (split from src/normalization.py, 2026-08-14)."""
import os
import re
import duckdb

from .constants import *  # noqa: F401,F403
from .text_utils import _in_clause
from .sapbert_model import _cosine


def get_sapbert_embedding(text):
    """Late-bound passthrough to src.normalization.get_sapbert_embedding
    (2026-08-14 package split). A plain `from .sapbert_model import
    get_sapbert_embedding` would bind THIS module's own independent
    reference to the original function at import time; a monkey-patch
    applied afterward to the package's re-export (`src.normalization
    .get_sapbert_embedding = fake`, as tests/test_tier12_ranking.py does)
    would then silently miss every call made from in here. Importing the
    package lazily, inside the call, keeps this a live lookup instead."""
    import src.normalization as _pkg
    return _pkg.get_sapbert_embedding(text)


_TIER_DEFAULT_MATCH_BASIS = {
    "1 (Exact)": "exact_text",
    "2 (Synonym)": "synonym",
    "3 (Semantic)": "semantic_similarity",
    "4 (Fuzzy)": "fuzzy_edit_distance",
}




def _candidate(row, tier, score, match_basis=None):
    """match_basis (2026-08-14, Stage 3 "Lasix problem" follow-up) records HOW
    a candidate was found, not just how well it scored. Without it, a
    KG-verified brand-alias hit (furosemide for "Lasix", similarity 0.57) and
    a coincidental spelling match (lasalocid, similarity 0.68) are
    STRUCTURALLY IDENTICAL dicts to Stage 3 -- nothing distinguishes "this is
    a certain fact from the terminology graph" from "this merely sounds
    alike", so a small model has no way to weigh the former over the
    latter's higher raw score. Defaults to the tier's usual basis; callers
    that inject alias-expansion candidates (see _tier3_semantic_rows) override
    it to "verified_brand_alias" for exactly those rows.
    """
    return {
        "omop_concept_id": row[0],
        "concept_name": row[1],
        "domain_id": row[2],
        "vocabulary_id": row[3],
        "match_tier": tier,
        "similarity_score": score,
        "match_basis": match_basis or _TIER_DEFAULT_MATCH_BASIS.get(tier, "unknown"),
    }




def _concept_class_map(conn, concept_ids):
    """{concept_id: concept_class_id} for the given ids.

    A separate lookup rather than an extra column on every Tier 1/2 SELECT,
    for two reasons: the tuple shape those queries return feeds _candidate()
    and three other call sites that would all need changing, and this query
    only runs when ranking is enabled AND there is more than one candidate to
    rank -- i.e. never on the common single-hit path. Failure returns an empty
    map, which makes every candidate score equally on class and lets the
    remaining criteria decide, rather than breaking the lookup.
    """
    if not concept_ids:
        return {}
    try:
        rows = conn.sql(
            f"SELECT concept_id, concept_class_id FROM athena_concept "
            f"WHERE concept_id IN ({','.join('?' * len(concept_ids))});",
            params=list(concept_ids)).fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}




def _class_rank(concept_class_id, gliner_label):
    """Lower is better. Preferred classes score 0..n by their position in
    TIER12_CLASS_PREFERENCE; unlisted classes score 90; demoted classes 99.
    """
    if concept_class_id is None:
        return 90
    preferred = TIER12_CLASS_PREFERENCE.get(gliner_label) or []
    if concept_class_id in preferred:
        return preferred.index(concept_class_id)
    if concept_class_id in TIER12_CLASS_DEMOTED:
        return 99
    return 90




def _specificity(conn, concept_ids):
    """{concept_id: n_ancestors}. More ancestors = deeper in the SNOMED
    hierarchy = more specific.

    Used only as a late tiebreak, and deliberately preferring the MORE
    specific concept: the DrivenData gold annotations name specific clinical
    findings ("Fracture of clavicle") far more often than their generic
    parents ("Fracture"), so when class, domain and semantic similarity have
    all failed to separate two candidates, the specific one is the better bet.
    Best-effort -- athena_concept_ancestor is 78.4M rows but this queries a
    handful of ids at a time, and any failure degrades to "no specificity
    signal" rather than raising.
    """
    if not concept_ids:
        return {}
    try:
        rows = conn.sql(
            f"SELECT descendant_concept_id, COUNT(*) FROM athena_concept_ancestor "
            f"WHERE descendant_concept_id IN ({','.join('?' * len(concept_ids))}) "
            f"GROUP BY descendant_concept_id;",
            params=list(concept_ids)).fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}




def _rank_tier12_candidates(conn, rows, gliner_label, entity_text, domains=None,
                            vector=None, use_semantic=None):
    """Orders Tier 1/2 result rows by clinical plausibility instead of by
    concept_id, and reports whether the ranking actually separated them.

    Returns (ordered_rows, basis, unresolved_tie).
      ordered_rows    -- the same tuples, reordered
      basis           -- "concept_id_asc" or "ranked_v1", persisted to
                         normalized_entities.tier12_rank_basis so an A/B is
                         gradable per row rather than per run
      unresolved_tie  -- True when the top two could not be separated; the
                         caller turns this into is_ambiguous so Stage 3 sees
                         a real question instead of an arbitrary answer

    RANKING KEY, most significant first. Each criterion is cheap and only
    consulted when the ones above it tie:
      1. concept_class_id preference for this GLiNER label. One SQL lookup,
         no model call, and the strongest signal available -- a "Condition"
         mention resolving to a Qualifier Value concept is almost always the
         collision case, not the intended reading.
      2. Domain agreement with GLINER_LABEL_TO_DOMAIN. Usually already
         enforced by the query's domain filter, but NOT when domain_override
         is in play (compound-split entities) or when the label has no domain
         mapping, so it still discriminates in exactly the cases that are
         hardest.
      3. SapBERT cosine between the ENTITY TEXT and each CONCEPT NAME. Reuses
         the embedding machinery Tier 3 already depends on. Costs at most
         CANDIDATE_LIMIT (3) embeddings, and only for multi-hit Tier 1/2
         lookups, which are the minority.
      4. Specificity (ancestor depth), preferring the more specific concept.
      5. concept_id ASC -- retained as the FINAL tiebreak so the function stays
         deterministic and reproducible. The complaint was never that
         concept_id ordering is unstable; it is that it was the FIRST
         criterion rather than the last.

    Never raises. Any failure (no class map, no embeddings, DB error) degrades
    to the legacy ordering with basis "concept_id_asc", so a ranking problem
    can never be worse than today's behaviour.
    """
    if not rows or len(rows) == 1:
        return rows, "concept_id_asc", False

    try:
        ids = [r[0] for r in rows]
        classes = _concept_class_map(conn, ids)
        depths = _specificity(conn, ids)

        # Criterion 3 is opt-in (TIER12_RANK_SEMANTIC). When off, sims stays
        # empty and key() reads a constant 0.0 for every candidate, so the
        # criterion contributes nothing to the ordering and no embedding is
        # requested -- not for the entity, and not for any concept name.
        semantic = TIER12_RANK_SEMANTIC if use_semantic is None else use_semantic
        sims = {}
        if semantic:
            if vector is None:
                vector = get_sapbert_embedding(entity_text)
            for r in rows:
                try:
                    sims[r[0]] = _cosine(vector, get_sapbert_embedding(r[1]))
                except Exception:
                    sims[r[0]] = 0.0

        wanted_domains = set(domains or GLINER_LABEL_TO_DOMAIN.get(gliner_label) or [])

        def key(r):
            cid, domain_id = r[0], r[2]
            return (
                _class_rank(classes.get(cid), gliner_label),      # asc, lower better
                0 if (not wanted_domains or domain_id in wanted_domains) else 1,
                -sims.get(cid, 0.0),                              # desc
                -depths.get(cid, 0),                              # desc (more specific)
                cid,                                              # asc, final tiebreak
            )

        ordered = sorted(rows, key=key)

        # BASIS IS HONEST ABOUT DEGRADED INPUTS. _concept_class_map() swallows
        # its own errors and returns {} so a vocabulary without concept_class_id
        # cannot break normalization -- but a ranking made without the single
        # strongest criterion is not the same ranking, and labelling it
        # "ranked_v1" would make the A/B compare two different things under one
        # name. Recorded separately so those rows can be excluded from (or
        # examined in) the comparison.
        basis = "ranked_v1" if classes else "ranked_v1_no_class"
        if semantic:
            basis += "_semantic"

        # A tie that survives class, domain and semantics is a real clinical
        # ambiguity, not something an ordering rule should pretend to settle.
        k0, k1 = key(ordered[0]), key(ordered[1])
        unresolved = (k0[0] == k1[0] and k0[1] == k1[1]
                      and abs(k0[2] - k1[2]) < TIER12_TIE_EPSILON)
        return ordered, basis, unresolved
    except Exception:
        return rows, "concept_id_asc", False




def _fuzzy_typo_candidates(conn, search_text, vocabs, domains, exclude_ids=None):
    """Edit-distance supplement to an already-uncertain Tier 3 result. See the
    FUZZY_MAX_EDIT_DISTANCE comment above for why this exists and why it is
    additive rather than its own tier.

    Best-effort: DuckDB's levenshtein() is a core scalar function as of the
    duckdb>=0.9.0 pinned in requirements.txt, but this was written and is not
    exercised against a live database in this environment -- a missing or
    renamed function must degrade to "no fuzzy candidates found" rather than
    breaking normalize_entity() for every call, so any duckdb error here is
    swallowed rather than raised.
    """
    if len(search_text) < FUZZY_MIN_TEXT_LENGTH:
        return []
    domain_clause = f" AND domain_id IN ({_in_clause(domains)})" if domains else ""
    lo, hi = len(search_text) - FUZZY_MAX_EDIT_DISTANCE, len(search_text) + FUZZY_MAX_EDIT_DISTANCE
    try:
        rows = conn.sql(f"""
            WITH scored AS (
                SELECT concept_id, concept_name, domain_id, vocabulary_id,
                       levenshtein(lower(concept_name), ?) AS edit_distance
                FROM athena_concept
                WHERE standard_concept = 'S'
                AND vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause}
                AND length(concept_name) BETWEEN ? AND ?
            )
            SELECT concept_id, concept_name, domain_id, vocabulary_id, edit_distance
            FROM scored
            WHERE edit_distance <= ?
            ORDER BY edit_distance ASC, concept_id ASC
            LIMIT {CANDIDATE_LIMIT};
        """, params=[search_text, *vocabs, *(domains or []), lo, hi,
                     FUZZY_MAX_EDIT_DISTANCE]).fetchall()
    except duckdb.Error:
        return []
    exclude_ids = exclude_ids or set()
    out = []
    for concept_id, name, domain_id, vocab_id, edit_distance in rows:
        if concept_id in exclude_ids:
            continue
        # Score on the same 0-1 scale as Tier 3's cosine similarity so it can
        # sit in the same candidate list, without claiming false precision --
        # this is "how much of the string matched", not a calibrated quantity.
        score = round(1 - (edit_distance / max(len(search_text), len(name))), 4)
        out.append(_candidate((concept_id, name, domain_id, vocab_id), "4 (Fuzzy)", score))
    return out




def _alias_expand_brand_to_generic(conn, search_text):
    """Brand-name -> generic-drug alias resolution (2026-08-13, "Lasix
    problem": SapBERT's embedding space does not reliably place a brand name
    close to its active ingredient, so a brand-only mention can silently drop
    out of Tier 3's top-K even though every other signal -- MoLLM reasoning,
    RxNorm itself -- agrees on the underlying drug).

    There is no direct 1-hop CONCEPT_RELATIONSHIP edge from a Brand Name
    concept to its generic ingredient (verified empirically against this DB:
    a 'Maps to'/'Tradename of' query straight from the brand concept returns
    nothing). The real path is three hops:
        brand -[Brand name of]-> branded product (Branded Drug Comp)
              -[Tradename of]-> generic Clinical Drug Comp (standard, RxNorm)
              -[RxNorm has ing]-> Ingredient (standard, RxNorm)

    Only the final Ingredient concept_ids are returned, not the intermediate
    dose-specific Clinical Drug Comp ones -- a well-established brand like
    Lasix has hundreds of branded SKUs (every dose/form/pack-size
    combination), each fanning out to its own generic Clinical Drug Comp, so
    keeping those would balloon the candidate list back into the exact
    crowding problem Fix 2 exists to prevent. The active ingredient(s) are
    what a bare brand mention ("Lasix") actually needs to resolve to, and
    every branded SKU collapses to the same small handful of ingredient
    concept_ids (usually one; combination drugs have a few).

    Returns a (possibly empty) set of standard concept_ids. Empty whenever
    search_text isn't an exact RxNorm Brand Name -- i.e. free on every
    non-brand entity, which is almost all of them.
    """
    try:
        rows = conn.sql("""
            WITH brand AS (
                SELECT concept_id FROM athena_concept
                WHERE lower(concept_name) = ?
                AND vocabulary_id = 'RxNorm' AND concept_class_id = 'Brand Name'
            ),
            branded_product AS (
                SELECT DISTINCT r.concept_id_2 AS concept_id
                FROM athena_concept_relationship r
                JOIN brand b ON r.concept_id_1 = b.concept_id
                WHERE r.relationship_id = 'Brand name of' AND r.invalid_reason IS NULL
            ),
            generic AS (
                SELECT DISTINCT r.concept_id_2 AS concept_id
                FROM athena_concept_relationship r
                JOIN branded_product bp ON r.concept_id_1 = bp.concept_id
                WHERE r.relationship_id = 'Tradename of' AND r.invalid_reason IS NULL
            ),
            ingredient AS (
                SELECT DISTINCT r.concept_id_2 AS concept_id
                FROM athena_concept_relationship r
                JOIN generic g ON r.concept_id_1 = g.concept_id
                WHERE r.relationship_id = 'RxNorm has ing' AND r.invalid_reason IS NULL
            )
            SELECT DISTINCT c.concept_id
            FROM athena_concept c
            WHERE c.concept_id IN (SELECT concept_id FROM ingredient)
            AND c.standard_concept = 'S';
        """, params=[search_text]).fetchall()
        return {r[0] for r in rows}
    except duckdb.Error:
        return set()


# 2026-08-18, "CHEM-7" investigation, GENERALIZED to "SGPT" (same session,
# same mechanism). Confirmed directly against athena_concept_synonym for
# both: neither "CHEM-7"/"Chem 7"/"chem7" nor "SGPT" appear as a clean,
# standalone-matchable synonym string anywhere in this dump. "SGPT" DOES
# appear, but only inside LOINC's packed multi-synonym cells that mix
# several distinct-test abbreviations together in one string (e.g. one real
# row's synonym cell is literally "...ALT; ...AST; ...SGOT; SGPT; ..." all
# together, for a concept that isn't even the plain ALT concept) -- Tier
# 2's exact-match requires the WHOLE cell to equal the search text, so a
# packed cell can never match "sgpt" alone. This is a genuine vocabulary
# gap, not a normalization/lookup bug -- Tier 1/2 structurally cannot find
# something that isn't cleanly there, no matter how the search text is
# normalized. Same fix shape as _alias_expand_brand_to_generic()
# (force-include a verified id into Tier 3 regardless of cosine rank), but
# a flat curated dict instead of a live KG walk, since there is no ontology
# relationship to traverse for informal/historical lab shorthand -- this IS
# the ontology-external knowledge.
#
# Confirmed root cause for SGPT specifically: it's a genuine Tier 3
# semantic-drift case, same family as tylenol->tylosin/coumadin->coumaran/
# metop->METOPON this session -- SapBERT ranks "Aspartate transaminase
# activity measurement" (WRONG; that's AST, i.e. SGOT's meaning) at 0.739
# vs. the correct "Alanine transaminase activity measurement" (SGPT's
# actual meaning) at 0.7239, a narrow but wrong-direction margin. SGPT
# (Serum Glutamic-Pyruvic Transaminase) is unambiguously ALT's old name;
# SGOT (Serum Glutamic-Oxaloacetic Transaminase) is unambiguously AST's --
# standard, fixed clinical terminology, not context-dependent, so a curated
# alias is safe here the same way it's safe for a brand name. SGOT is
# included for symmetry (same fixed-terminology basis) even though it
# already resolves correctly today -- insurance against Tier 3 ranking
# noise, not a fix for an independently measured wrong case the way SGPT is.
#
# Deliberately narrow otherwise: only the two cases with a specific,
# verified root cause. Other lab shorthand (CHEM-20, BMP, CMP, CBC, LFTs)
# was raised as plausible analogues in this session but NOT individually
# verified against real data the way these were -- adding them on
# guesswork would repeat the exact mistake this project's own history
# already learned from (the abbreviation flywheel's block-list-to-allow-
# list reversal, 2026-08-17): a plausible-sounding entry is not the same as
# a measured one. Extend this dict only after the same direct verification
# these entries got.
_LAB_TEST_ALIASES = {
    "chem-7": 3041230, "chem 7": 3041230, "chem7": 3041230,
    # -> 'Basic metabolic panel, Blood' (LOINC, Measurement/Lab Test) --
    # the plain/unqualified panel concept, not one of its dated (1998/2000/
    # 2008) or add-on (with Hgb/Hct, with albumin, with ionized calcium)
    # variants, since bare "CHEM-7" implies none of those extra qualifiers.
    "sgpt": 44810789,   # -> 'Alanine transaminase activity measurement' (= ALT)
    "sgot": 44810795,   # -> 'Aspartate transaminase activity measurement' (= AST)
    # 2026-08-18, verified against train_annotations.csv gold spans (same
    # discipline as the entries above -- direct corpus measurement, not a
    # plausible-sounding guess): bare "HCT" -> SNOMED 28317006 in 568/569
    # (99.8%) of gold annotations; bare "MCH" -> SNOMED 54706004 in 408/408
    # (100%). Both are flowsheet-style CBC differential abbreviations that
    # strip_lab_value_suffix() reduces "HCT-32"/"MCH-28" to -- SapBERT alone
    # was measured to place the bare abbreviation below TIER3_SIMILARITY_FLOOR
    # against their full determination-concept names (confirmed live on note
    # 13538696-DS-11: HCT-32 and the merged MCV-90 MCH-28 span both landed at
    # 0 (Failed) / tier3_below_similarity_floor_rejected).
    "hct": 4151358,   # -> 'Hematocrit determination'
    "mch": 4182871,   # -> 'Mean corpuscular hemoglobin determination'
    # 2026-08-18, verified against train_annotations.csv gold spans: bare
    # "CXR" -> SNOMED 399208008 in 232/232 (100%) of gold annotations, the
    # largest single-abbreviation sample verified for this dict so far.
    # Confirmed live that bare Tier 1-3 search returns "0 (Failed)" for
    # "CXR" today (below-floor semantic drift, not a domain/vocab issue).
    "cxr": 4163872,   # -> 'Plain X-ray of chest'
    # 2026-08-20, verified against train_annotations.csv gold spans, same
    # discipline as the entries above -- root-caused from the fresh25
    # validation batch's Tier 1 Lab Test finding (64/94 = 68% wrong,
    # evaluation/grade_fresh25_by_tier.py). NOT a lab_procedure_preferred
    # regression (only 1/120 Lab Test Tier-1 candidates used that basis) --
    # plain SapBERT semantic_similarity was landing on a plausible but
    # not-exact SNOMED near-duplicate (e.g. generic "Calcium measurement"
    # instead of gold's "Blood calcium measurement"; a UK-SNOMED-extension
    # code instead of the US-core concept for ALT). Each entry below is
    # 100% consistent in gold, sample sizes 41-491 (bare abbreviation,
    # value-suffix already stripped by strip_lab_value_suffix()):
    "calcium": 4193434,  # -> 'Blood calcium measurement' (n=274/274)
    "alt": 4146380,       # -> 'Alanine aminotransferase measurement' (n=208/208)
    "na": 4208938,        # -> 'Sodium measurement, blood' (n=407/407)
    "urean": 4017361,     # -> 'Blood urea nitrogen measurement' (n=368/368)
    "ph": 4215028,        # -> 'pH measurement' (n=103/103)
    "creat": 4324383,     # -> 'Creatinine measurement' (n=474/474)
    "phos": 4020559,      # -> 'Phosphate, total measurement' (n=260/260)
    "inr": 4261078,       # -> 'Calculation of international normalized ratio' (n=41/41)
    "rdw": 4281085,       # -> 'Red cell distribution width determination' (n=491/491)
    "mcv": 4016239,       # -> 'Erythrocyte mean corpuscular volume determination' (n=192/192)
    "mchc": 4290193,      # -> 'Mean corpuscular hemoglobin concentration determination' (n=490/490)
    "total co2": 4193415,  # -> 'Blood total carbon dioxide (calculated)' (n=88/88)
}


def _lab_test_alias(search_text: str) -> set:
    concept_id = _LAB_TEST_ALIASES.get(search_text.strip().lower())
    return {concept_id} if concept_id else set()


_COMPOUND_NAME_RE = re.compile(r"\b(and|&)\b", re.IGNORECASE)


def _is_compound_concept_name(name: str) -> bool:
    """True when `name` reads as a combined/multi-part concept ("X and Y"),
    per the coordinating-conjunction signal.

    2026-08-14, Intermaxillary-fixation regression (found via a fresh
    GOLD_MISSING diagnostic on the 27-note corpus, note 14490470-DS-11).
    "Intermaxillary fixation of mandible AND maxilla" (gold, 302474008)
    Subsumes "...of mandible" alone -- a real SNOMED edge, but NOT the
    "same clinical idea at different specificity" pattern
    _collapse_hierarchy_duplicates() was built for (see its own docstring's
    ALT-measurement case, where the three trio members are near-perfect
    synonyms for ONE lab test regardless of which is kept). Here the parent
    and child describe genuinely DIFFERENT clinical facts -- which bones
    were fixed -- so silently discarding the compound parent in favor of a
    higher-scored single-component child (0.9407 vs 0.9282, kept "mandible"
    alone, discarded "mandible and maxilla") loses real information gold
    depends on. A bare conjunction check is a narrow, explainable signal
    directly targeting this exact failure shape, matching this codebase's
    existing "and"/"or" compound-phrase handling in find_compound_split()'s
    whole-phrase guard -- not a general semantic-similarity judgment call.
    """
    return bool(_COMPOUND_NAME_RE.search(name or ""))


def _collapse_hierarchy_duplicates(conn, cands):
    """Collapses candidates connected by a direct SNOMED 'Is a'/'Subsumes'
    edge into a single entry, keeping whichever has the higher similarity
    score.

    WHY THIS EXISTS (2026-08-13, ALT-measurement example from
    docs/Stage3_Open_Issues.md): Tier 3 can retrieve several *distinct*
    standard concepts that are not accidental duplicates but real SNOMED
    parent/child concepts describing the same clinical idea at slightly
    different specificity (e.g. "ALT - blood measurement" is a parent of both
    "Alanine transaminase activity measurement" and "Alanine aminotransferase
    measurement"). Presenting all of them to MoLLM voting fractures what
    should be one answer into an artificial 2-1 or 1-1-1 split. A plain
    GROUP BY concept_id can't catch this -- the ids are genuinely different --
    so this walks the relationship graph among just the candidate set and
    merges connected components instead.

    Only direct edges within the candidate set are considered (no external
    hierarchy lookups), so this stays cheap: at most CANDIDATE_LIMIT-ish rows
    are ever passed in. Never raises -- any DB error degrades to returning
    cands unchanged.

    2026-08-14: a candidate whose name reads as a compound/combined concept
    (_is_compound_concept_name()) is EXCLUDED from collapsing regardless of
    its similarity score, and survives as its own entry -- see that
    function's docstring for why "highest similarity wins" is the wrong rule
    specifically for this shape of parent/child pair.
    """
    if len(cands) < 2:
        return cands
    ids = [c["omop_concept_id"] for c in cands]
    placeholders = ",".join("?" * len(ids))
    try:
        rows = conn.sql(f"""
            SELECT DISTINCT concept_id_1, concept_id_2
            FROM athena_concept_relationship
            WHERE concept_id_1 IN ({placeholders}) AND concept_id_2 IN ({placeholders})
            AND relationship_id IN ('Is a', 'Subsumes') AND invalid_reason IS NULL
            AND concept_id_1 != concept_id_2;
        """, params=[*ids, *ids]).fetchall()
    except duckdb.Error:
        return cands
    if not rows:
        return cands

    parent = {cid: cid for cid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in rows:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 2026-08-14: compound-named candidates never enter the reduction below --
    # they are always kept, and never used to evict (or get evicted by) a
    # sibling. best_for_root is built from SIMPLE candidates only, so a
    # compound candidate sharing a root with a simple one does not change
    # which simple candidate wins that root -- the two collapses are
    # independent, matching the union-find's own per-root grouping.
    best_for_root = {}
    for c in cands:
        if _is_compound_concept_name(c.get("concept_name")):
            continue
        root = find(c["omop_concept_id"])
        current = best_for_root.get(root)
        if current is None or c["similarity_score"] > current["similarity_score"]:
            best_for_root[root] = c

    seen_roots = set()
    out = []
    for c in cands:
        if _is_compound_concept_name(c.get("concept_name")):
            out.append(c)
            continue
        root = find(c["omop_concept_id"])
        if root in seen_roots:
            continue
        seen_roots.add(root)
        out.append(best_for_root[root])
    return out


# concept_class_id penalty applied to an "Observable Entity" candidate when a
# "Procedure" sibling is also present, for Lab-Test-labeled entities only --
# see _prefer_lab_procedure_over_observable()'s docstring. Large enough to
# reliably flip every score gap actually observed on the 27-note corpus
# (max ~0.06, e.g. "MCHC" candidates), without being so large it could ever
# plausibly matter against a genuinely unrelated third candidate.
_LAB_PROCEDURE_CLASS_BONUS = 0.1


def _prefer_lab_procedure_over_observable(conn, cands, gliner_label):
    """For Lab-Test-labeled entities, re-ranks a "Procedure"-class candidate
    ahead of an "Observable Entity"-class sibling describing the same test.

    2026-08-14, WBC-13 investigation (docs from this session): SapBERT
    consistently scores the Observable-Entity-class concept for a lab test
    (e.g. "Leucocyte count", the abstract property measured) higher than the
    Procedure-class concept for the SAME test (e.g. "White blood cell
    count", the act of measuring it) -- 0.892 vs 0.8694 for WBC. Measured
    directly against the SNOMED CT Entity Linking Challenge's own gold
    annotations across every Lab-Test entity in the 27-note corpus with BOTH
    classes present among its candidates: the Procedure-class concept is
    gold-correct in 78/78 cases where either is correct at all (0
    counterexamples, 23 "neither" cases where gold is a different concept
    entirely). This is a real, exceptionless annotation convention in this
    specific gold set (the measurement ACT, not the abstract observable
    property), not a heuristic guess -- the 3B ensemble was unanimously
    picking the higher-SCORED, gold-WRONG concept every time this pattern
    occurred, exactly the mechanism the earlier "ontology drift" discussion
    this session identified but initially mischaracterized as a coverage
    gap (the correct concept was always present, just outranked).

    Implemented as a rank-only penalty (subtracted from a private sort key,
    never written back to the candidate's own displayed similarity_score),
    so the MoLLM prompt still shows the model's true, unmodified SapBERT
    score -- only the ORDER changes, not the evidence.
    """
    if gliner_label != "Lab Test" or len(cands) < 2:
        return cands
    ids = [c["omop_concept_id"] for c in cands]
    placeholders = ",".join("?" * len(ids))
    try:
        classes = dict(conn.sql(
            f"SELECT concept_id, concept_class_id FROM athena_concept "
            f"WHERE concept_id IN ({placeholders})", params=ids).fetchall())
    except duckdb.Error:
        return cands
    has_observable = any(classes.get(c["omop_concept_id"]) == "Observable Entity" for c in cands)
    has_procedure = any(classes.get(c["omop_concept_id"]) == "Procedure" for c in cands)
    if not (has_observable and has_procedure):
        return cands

    def _sort_key(c):
        penalty = _LAB_PROCEDURE_CLASS_BONUS \
            if classes.get(c["omop_concept_id"]) == "Observable Entity" else 0.0
        return c["similarity_score"] - penalty

    ranked = sorted(cands, key=_sort_key, reverse=True)

    # 2026-08-19. A rank-only nudge wasn't enough on its own -- corpus-scale
    # grading (evaluation/tier_gate_grading.py) found the 3-model ensemble
    # unanimously RE-RANKING AWAY from this exact winner regardless (e.g.
    # "RDW-13": correct "...determination" concept placed at #1 by this very
    # function, all 3 models still rejected it for the Observable-Entity
    # sibling at #2). Tried fixing via a prompt clause first -- proved
    # unreliable (one model actively got WORSE, re-ranking to an unrelated
    # 3rd candidate) because _binary_match_prompt() judges one candidate per
    # call with zero visibility into any other candidate's text, so a
    # clause phrased "even when a later candidate looks closer" asks the
    # model to reason about something outside its own context window.
    # Reverted that attempt. This tag instead lets tier3_fast_path()
    # (src.mollm_tier_gate) skip the ensemble ENTIRELY for this pattern --
    # same "curated, pre-verified, not a similarity guess" trust tier as
    # verified_brand_alias/verified_lab_test_alias/physexam_shorthand,
    # justified by the same 78/78-exceptionless corpus evidence this
    # function's own docstring already cites for the ranking bonus above.
    # Only tags the winner when its OWN match_basis is still the generic
    # Tier 3 default -- never overwrites a candidate that already carries a
    # more specific verified-alias basis.
    if ranked and classes.get(ranked[0]["omop_concept_id"]) == "Procedure" \
            and ranked[0].get("match_basis") in (None, "semantic_similarity"):
        ranked[0]["match_basis"] = "lab_procedure_preferred"

    return ranked


def _lab_procedure_sibling_check(conn, cands, gliner_label):
    """True when `cands` (a Tier 1/2 exact/synonym result) needs the
    supplementary Procedure-class semantic lookup below -- i.e. every
    candidate is Observable-Entity-class, for a Lab-Test-labeled entity.

    2026-08-19, corpus-scale grading finding. _prefer_lab_procedure_over_
    observable() above (78/78-exceptionless evidence for this exact
    Observable-Entity-vs-Procedure duplicate pattern) can only ever fire
    when BOTH classes are already present in `cands` -- true for Tier 3's
    top-K semantic search, never true for Tier 1/2's literal exact/synonym
    match, since the Procedure-class sibling has entirely different text
    ("MCV - Mean corpuscular volume" [Observable Entity, exact hit] vs
    "Erythrocyte mean corpuscular volume determination" [Procedure, gold's
    actual target] -- no shared substring, not even case-insensitively, and
    British/American spelling differences like "haemoglobin"/"hemoglobin"
    rule out a plain LIKE-based rescue too). Confirmed live via
    evaluation/tier_gate_grading.py against the corpus so far: RDW/MCV/MCHC
    all land on the Observable-Entity concept at TIER_2_AUTO_RESOLVED with
    zero exceptions across every occurrence graded, and no direct SNOMED
    relationship exists between the two concepts in either case (checked
    athena_concept_relationship directly) -- confirming this needs a
    semantic lookup, not a graph traversal, the same way
    _alias_expand_brand_to_generic's KG walk wasn't available for "Lasix"
    either.
    """
    if gliner_label != "Lab Test" or not cands:
        return False
    ids = [c["omop_concept_id"] for c in cands]
    placeholders = ",".join("?" * len(ids))
    try:
        classes = dict(conn.sql(
            f"SELECT concept_id, concept_class_id FROM athena_concept "
            f"WHERE concept_id IN ({placeholders})", params=ids).fetchall())
    except duckdb.Error:
        return False
    return all(classes.get(cid) == "Observable Entity" for cid in ids)


def _lab_procedure_sibling(conn, vector, exclude_ids):
    """Semantic search for the best Procedure-class SNOMED Measurement
    concept for `vector` (the entity's own SapBERT embedding) -- the
    supplementary lookup _lab_procedure_sibling_check() gates. Scoped
    tightly (domain_id='Measurement', concept_class_id='Procedure',
    vocabulary_id='SNOMED', standard_concept='S') since this is a narrow
    rescue for one specific, evidence-verified duplicate pattern, not a
    general Tier 3 fallback -- a low-confidence match here would silently
    override an already-confident Tier 1/2 exact hit, so the caller applies
    its own similarity floor before trusting the result.
    """
    exclude_ids = set(exclude_ids or [])
    try:
        rows = conn.sql("""
            SELECT concept_id, concept_name, domain_id, vocabulary_id,
                   list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
            FROM athena_concept
            WHERE embedding IS NOT NULL AND standard_concept = 'S'
            AND vocabulary_id = 'SNOMED' AND domain_id = 'Measurement'
            AND concept_class_id = 'Procedure'
            ORDER BY similarity DESC LIMIT 1;
        """, params=[vector]).fetchall()
    except duckdb.Error:
        return None
    if not rows or rows[0][0] in exclude_ids:
        return None
    return rows[0]


def _tier3_semantic_rows(conn, vector, vocabs, domains, alias_ids=None, limit=None):
    """The Tier 3 SapBERT top-K query, plus force-including any alias_ids
    (see _alias_expand_brand_to_generic) regardless of where they land in the
    similarity ranking. Without this, a real cosine-similarity gap between a
    brand name and its own generic ingredient silently drops the correct
    concept out of the top CANDIDATE_LIMIT before Stage 3 ever sees it.

    alias_ids are scored by their own cosine similarity (not pinned to 1.0),
    so Stage 3 still sees the true semantic distance -- only their presence
    in the candidate list is guaranteed, not their rank.

    `limit` (2026-08-16, Pass 3 hybrid retrieval): defaults to CANDIDATE_LIMIT
    for every existing caller, unchanged. _tier3_hybrid_rows() passes a wider
    pool size (RRF_POOL_SIZE) here, since RRF fusion needs enough ranked
    candidates PER SIGNAL to be meaningful before truncating to
    CANDIDATE_LIMIT after fusion, not before.
    """
    limit = limit or CANDIDATE_LIMIT
    domain_clause = f" AND domain_id IN ({_in_clause(domains)})" if domains else ""
    alias_ids = list(alias_ids or [])
    try:
        if alias_ids:
            placeholders = ",".join("?" * len(alias_ids))
            rows = conn.sql(f"""
                WITH topk AS (
                    SELECT concept_id, concept_name, domain_id, vocabulary_id,
                           list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
                    FROM athena_concept
                    WHERE embedding IS NOT NULL AND standard_concept = 'S'
                    AND vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause}
                    ORDER BY similarity DESC, concept_id ASC LIMIT {int(limit)}
                ),
                alias AS (
                    SELECT concept_id, concept_name, domain_id, vocabulary_id,
                           list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
                    FROM athena_concept
                    WHERE embedding IS NOT NULL AND standard_concept = 'S'
                    AND concept_id IN ({placeholders})
                )
                SELECT * FROM topk
                UNION
                SELECT * FROM alias
                ORDER BY similarity DESC, concept_id ASC;
            """, params=[vector, *vocabs, *(domains or []), vector, *alias_ids]).fetchall()
        else:
            rows = conn.sql(f"""
                SELECT concept_id, concept_name, domain_id, vocabulary_id,
                       list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
                FROM athena_concept
                WHERE embedding IS NOT NULL AND standard_concept = 'S'
                AND vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause}
                ORDER BY similarity DESC, concept_id ASC LIMIT {int(limit)};
            """, params=[vector, *vocabs, *(domains or [])]).fetchall()
    except duckdb.Error:
        return []
    return rows


# ==========================================================================
# HYBRID (BM25 + SapBERT + empirical prior) RETRIEVAL, RECIPROCAL RANK FUSION
# (2026-08-16, plan Phase 3). DEFAULT OFF -- see HYBRID_RETRIEVAL_ENABLED.
# ==========================================================================
#
# Score(c) = w_dense * RRF_dense(c) + w_sparse * RRF_sparse(c) + w_prior * P(c|Mention)
# RRF_x(c) = 1 / (RRF_K + rank_x(c))    -- standard Reciprocal Rank Fusion,
# rank-based rather than raw-score-based specifically because SapBERT cosine
# and BM25 scores do not live on comparable scales (cosine in [0,1]-ish;
# BM25 here measured in the 0-10 range with no fixed ceiling) -- averaging
# raw scores would let whichever signal happens to have the larger numeric
# range dominate for no principled reason.
#
# WEIGHTS ARE CALIBRATION TARGETS, NOT SETTLED VALUES -- named here so there
# is one place to change them after measuring against
# evaluation/stage2b_cal_eval.py's ranking harness, same discipline every
# other threshold in this codebase follows (see e.g.
# src/mollm_ensemble.py's AUTO_VALIDATE_THRESHOLD). Starting point: dense
# weighted highest since SapBERT is the currently-validated, working signal;
# prior weighted lowest since Phase 4's Empirical Prior Matrix (which alone
# populates P(c|Mention) with anything beyond 0) does not exist yet.
RRF_K = 60
RRF_WEIGHT_DENSE = 0.5
RRF_WEIGHT_SPARSE = 0.3
RRF_WEIGHT_PRIOR = 0.2

# Candidates considered PER SIGNAL before fusion and truncation to
# CANDIDATE_LIMIT -- must exceed CANDIDATE_LIMIT for RRF to have room to
# reorder based on the OTHER signal (a dense-only top-CANDIDATE_LIMIT pool
# fused with a sparse-only top-CANDIDATE_LIMIT pool would just be the same
# two short lists interleaved, not a real fusion).
RRF_POOL_SIZE = 20

HYBRID_RETRIEVAL_ENABLED = os.environ.get(
    "CNSP_HYBRID_RETRIEVAL", "").strip() in ("1", "true", "yes")


def _rrf_scores(ranked_ids: list) -> dict:
    return {cid: 1.0 / (RRF_K + rank) for rank, cid in enumerate(ranked_ids, 1)}


def _tier3_hybrid_rows(conn, entity_text, vector, vocabs, domains, prior_lookup=None,
                       alias_ids=None):
    """Additive alternative to _tier3_semantic_rows() -- ADDITIVE, not a
    replacement: _tier_queries() still calls the dense-only path by default
    (see that function), and this is wired in only where a caller explicitly
    opts in behind HYBRID_RETRIEVAL_ENABLED. Fuses:
      - dense: SapBERT cosine ranking (reuses _tier3_semantic_rows() itself,
        just with a wider pool -- no separate dense-query implementation).
      - sparse: BM25 ranking (src.normalization.bm25_index.query_bm25()).
      - prior: `prior_lookup`, a caller-supplied {concept_id: P(c|Mention)}
        dict. None (the default) means every prior term is 0 -- an explicit
        code path via .get(cid, 0.0), not a silent division-by-zero risk --
        since Phase 4's Empirical Prior Matrix, the only thing that would
        ever populate this meaningfully, does not exist yet.

    TIER3_SIMILARITY_FLOOR stays anchored to the DENSE cosine specifically
    (`similarity_score` in the returned candidate dicts), not the fused RRF
    score, which lives on a different, unbounded scale -- per the plan's own
    design note. A concept found only via BM25 (no dense score in the
    RRF_POOL_SIZE dense pool) gets its cosine looked up in a small, targeted
    follow-up query rather than left None, so the floor check downstream
    never silently skips a sparse-only hit for lack of a comparable score.
    """
    from .bm25_index import query_bm25  # local import: avoids a hard,
    # always-paid dependency on the bm25_index module (and its FTS LOAD) for
    # every caller of this file that never uses the hybrid path.

    dense_rows = _tier3_semantic_rows(conn, vector, vocabs, domains, limit=RRF_POOL_SIZE)
    sparse_rows = query_bm25(conn, entity_text, vocabs=vocabs, domains=domains,
                             limit=RRF_POOL_SIZE)

    dense_by_id = {r[0]: r for r in dense_rows}
    sparse_by_id = {r[0]: r for r in sparse_rows}
    dense_rrf = _rrf_scores([r[0] for r in dense_rows])
    sparse_rrf = _rrf_scores([r[0] for r in sparse_rows])
    prior_lookup = prior_lookup or {}
    alias_ids = set(alias_ids or [])

    # alias_ids (verified_brand_alias, see _alias_expand_brand_to_generic)
    # are force-included the same way _tier3_semantic_rows() force-includes
    # them for the dense-only path -- a real, walked KG relationship should
    # not depend on making either ranked pool by chance.
    all_ids = set(dense_by_id) | set(sparse_by_id) | alias_ids
    missing_dense_ids = [cid for cid in all_ids if cid not in dense_by_id]
    dense_lookup = {}
    if missing_dense_ids and vector is not None:
        placeholders = ",".join("?" * len(missing_dense_ids))
        try:
            rows = conn.sql(f"""
                SELECT concept_id, concept_name, domain_id, vocabulary_id,
                       list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
                FROM athena_concept
                WHERE concept_id IN ({placeholders}) AND embedding IS NOT NULL
            """, params=[vector, *missing_dense_ids]).fetchall()
            dense_lookup = {r[0]: r for r in rows}
        except duckdb.Error:
            dense_lookup = {}

    fused = []
    for cid in all_ids:
        rrf = (RRF_WEIGHT_DENSE * dense_rrf.get(cid, 0.0)
               + RRF_WEIGHT_SPARSE * sparse_rrf.get(cid, 0.0)
               + RRF_WEIGHT_PRIOR * prior_lookup.get(cid, 0.0))
        row = dense_by_id.get(cid) or sparse_by_id.get(cid) or dense_lookup.get(cid)
        if row is None:
            continue  # alias id whose embedding lookup itself failed -- nothing to show
        dense_row = dense_by_id.get(cid) or dense_lookup.get(cid)
        dense_score = dense_row[4] if dense_row else None
        fused.append((cid, row, dense_score, rrf))

    fused.sort(key=lambda t: t[3], reverse=True)
    # alias_ids are GUARANTEED inclusion regardless of RRF rank -- same
    # guarantee _tier3_semantic_rows() gives them via its SQL UNION, kept
    # here rather than left to chance now that they're one signal among
    # three instead of unconditionally unioned in.
    alias_entries = [t for t in fused if t[0] in alias_ids]
    other_entries = [t for t in fused if t[0] not in alias_ids]
    slots_for_others = max(0, CANDIDATE_LIMIT - len(alias_entries))
    top = alias_entries + other_entries[:slots_for_others]
    top.sort(key=lambda t: t[3], reverse=True)

    out = []
    for cid, row, dense_score, rrf in top:
        # 0.0, never None: every existing consumer of similarity_score
        # (_collapse_hierarchy_duplicates, _prefer_lab_procedure_over_observable,
        # the TIER3_SIMILARITY_FLOOR check) does numeric comparisons/arithmetic
        # on this field and was written assuming a float is always present --
        # a rare missing-embedding edge case should read as "score unmeasurably
        # low", not crash those callers.
        c = _candidate(row, "3 (Hybrid)", round(dense_score, 4) if dense_score is not None else 0.0,
                       match_basis="verified_brand_alias" if cid in alias_ids else None)
        c["rrf_score"] = round(rrf, 6)
        c["retrieval_method"] = "hybrid_rrf"
        out.append(c)
    return out




def _tier_queries(conn, search_text, vocabs, domains, entity_text, vector=None,
                  gliner_label=None):
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
        # 2026-08-13 (P1.1): rank before returning, so candidates[0] -- the
        # pick every downstream consumer treats as Stage 2b's answer -- is
        # chosen on clinical plausibility rather than on integer id.
        rows, _basis, _tie = (_rank_tier12_candidates(
            conn, rows, gliner_label, entity_text, domains=domains, vector=vector)
            if RANKED_TIER12 else (rows, "concept_id_asc", False))
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
        rows, _basis, _tie = (_rank_tier12_candidates(
            conn, rows, gliner_label, entity_text, domains=domains, vector=vector)
            if RANKED_TIER12 else (rows, "concept_id_asc", False))
        return [_candidate(r, "2 (Synonym)", 1.0) for r in rows], "2", None

    if vector is None:
        vector = get_sapbert_embedding(entity_text)
    alias_ids = _alias_expand_brand_to_generic(conn, search_text)
    panel_ids = _lab_test_alias(search_text)
    force_include_ids = alias_ids | panel_ids
    if HYBRID_RETRIEVAL_ENABLED:
        cands = _tier3_hybrid_rows(conn, entity_text, vector, vocabs, domains,
                                   alias_ids=force_include_ids)
        if not cands:
            return None, None, 0.0
    else:
        rows = _tier3_semantic_rows(conn, vector, vocabs, domains, alias_ids=force_include_ids)
        if not rows:
            return None, None, 0.0
        def _match_basis(concept_id):
            if concept_id in alias_ids:
                return "verified_brand_alias"
            if concept_id in panel_ids:
                return "verified_lab_test_alias"
            return None
        cands = [_candidate(r, "3 (Semantic)", round(r[4], 4), match_basis=_match_basis(r[0]))
                for r in rows]
    cands = _collapse_hierarchy_duplicates(conn, cands)
    cands = _prefer_lab_procedure_over_observable(conn, cands, gliner_label)
    return cands, "3", cands[0]["similarity_score"]




# concept_class_id values that can never be a legitimate resolution for a
# PATIENT clinical mention -- see _detect_domain_conflict()'s 2026-08-14
# docstring paragraph for the "galea"/"Genus Galea" case that motivated this.
_NON_CLINICAL_CONCEPT_CLASSES = {"Organism"}


def _detect_domain_conflict(conn, search_text, vocabs, domains, entity_text, vector,
                            gliner_label=None):
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

    2026-08-14 (galea regression, GOLD_MISSING diagnostic on 14490470-DS-11):
    an unrestricted out-of-domain search can surface SNOMED's Organism
    hierarchy (biological genus/species taxonomy, concept_class_id
    "Organism") purely on embedding proximity to a short/polysemous English
    word -- "galea" (Latin for helmet) scored 0.90 against "Genus Galea" (a
    rodent genus) versus 0.61 for the correct anatomy concept "Structure of
    galea aponeurotica". This pipeline only ever extracts PATIENT-relevant
    clinical mentions (Condition/Procedure/Drug/Measurement/Anatomy, per
    GLINER_LABEL_TO_DOMAIN) -- an Organism-class concept is never a
    legitimate resolution for any of them, unlike a genuine cross-domain
    mislabeling (Medication text actually describing a Condition, say),
    which is what this function exists to catch. Filtered HERE rather than
    in _tier_queries()/_candidate() itself, since the exclusion is specific
    to the "did GLiNER mislabel this" question this function answers, not a
    general candidate-quality rule that should apply everywhere.
    """
    if not domains:
        return None

    def _try(vocabs_to_use):
        cands, tier, _ = _tier_queries(conn, search_text, vocabs_to_use, None, entity_text,
                                       vector, gliner_label=gliner_label)
        if not cands:
            return None
        ids = [c["omop_concept_id"] for c in cands]
        placeholders = ",".join("?" * len(ids))
        try:
            classes = dict(conn.sql(
                f"SELECT concept_id, concept_class_id FROM athena_concept "
                f"WHERE concept_id IN ({placeholders})", params=ids).fetchall())
        except duckdb.Error:
            classes = {}
        cands = [c for c in cands
                if classes.get(c["omop_concept_id"]) not in _NON_CLINICAL_CONCEPT_CLASSES]
        if not cands:
            return None
        if tier == "3" and cands[0]["similarity_score"] < TIER3_SIMILARITY_FLOOR:
            return None
        return cands, tier

    result = _try(vocabs)
    vocab_relaxed = False
    # 2026-08-14 (diuretics/diuresis regression, GOLD_MISSING diagnostic on
    # 16393593-DS-5 and others): relaxing domains alone is not enough for
    # "Medication"-labeled GENERIC drug-class mentions ("diuretics",
    # "antibiotics", as opposed to a specific named drug). Gold annotates
    # these with a SNOMED "X therapy" Procedure-domain concept (e.g.
    # "Diuretic therapy", 722048006) -- RxNorm/RxNorm Extension
    # (VOCAB_BY_LABEL["Medication"]) has no equivalent therapeutic-class-
    # level concept, only ingredients/products, so even a domain-unrestricted
    # retry that keeps the label's own vocab restriction can never find it.
    # Retried here with vocabs relaxed to DEFAULT_VOCAB (SNOMED) -- a no-op
    # for every other label, which already defaults to SNOMED -- rather than
    # widening VOCAB_BY_LABEL itself, since Medication's RxNorm restriction
    # is still correct and wanted for the common case of a specific,
    # correctly-named drug.
    if result is None and vocabs != DEFAULT_VOCAB:
        result = _try(DEFAULT_VOCAB)
        vocab_relaxed = True
    if result is None:
        return None
    cands, tier = result
    return {
        "found_domain": cands[0]["domain_id"],
        "expected_domains": domains,
        "tier": tier,
        "candidates": cands,
        "vocab_relaxed": vocab_relaxed,
    }



