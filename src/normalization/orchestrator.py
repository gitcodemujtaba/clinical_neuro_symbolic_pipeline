"""src/normalization/orchestrator.py — normalize_entity()/process_and_normalize_entities() top-level orchestration (split from src/normalization.py, 2026-08-14)."""
import re
import json

from src.provenance import (
    provenance_alter_statements,
    provenance_column_sql,
    provenance_params,
    provenance_placeholders,
)

from .constants import *  # noqa: F401,F403
from .text_utils import _in_clause
from .vocab_release import get_athena_vocabulary_release
from .sapbert_model import SAPBERT_POOLING
from .compound_span import _lookup_tier12, strip_lab_value_suffix, _LAB_TIER_RANK
from .tier_retrieval import (
    _candidate, _rank_tier12_candidates, _fuzzy_typo_candidates,
    _alias_expand_brand_to_generic, _collapse_hierarchy_duplicates,
    _prefer_lab_procedure_over_observable, _tier3_semantic_rows,
    _detect_domain_conflict,
)


def get_sapbert_embedding(text):
    """Late-bound passthrough to src.normalization.get_sapbert_embedding --
    see the identical function in tier_retrieval.py for why this indirection
    exists (2026-08-14 package split, preserves package-level monkey-patch
    semantics tests rely on)."""
    import src.normalization as _pkg
    return _pkg.get_sapbert_embedding(text)


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
        # 2026-08-13 (P1.1). Two changes, both gated on RANKED_TIER12:
        #
        #  (a) candidates[0] is chosen by _rank_tier12_candidates() rather than
        #      by lowest concept_id -- the fix for the 52.71% Tier 1 accuracy
        #      in the 2026-08-13 report S5.3.
        #
        #  (b) `ambiguous` now means "ranking could NOT separate the top two",
        #      not "there was more than one row". That distinction is the point:
        #      the old flag fired on every multi-match, so Stage 3 was handed a
        #      flood of cases the ranking can now settle, while the genuinely
        #      undecidable ones were silently resolved by integer id and never
        #      flagged at all. Exactly inverted from what is useful.
        rows, basis, unresolved = (
            _rank_tier12_candidates(conn, rows, gliner_label, entity_text,
                                    domains=domains)
            if RANKED_TIER12 else (rows, "concept_id_asc", len(rows) > 1))
        cands = [_candidate(r, "1 (Exact)", 1.0) for r in rows]
        ambiguous = unresolved if RANKED_TIER12 else len(cands) > 1
        return _result(
            cands,
            ambiguous=ambiguous,
            reason=("unresolved_exact_tie_after_ranking" if RANKED_TIER12
                    else "multiple_exact_concept_name_matches") if ambiguous else None,
            trace=trace,
            rank_basis=basis,
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
        rows, basis, unresolved = (
            _rank_tier12_candidates(conn, rows, gliner_label, entity_text,
                                    domains=domains)
            if RANKED_TIER12 else (rows, "concept_id_asc", len(rows) > 1))
        cands = [_candidate(r, "2 (Synonym)", 1.0) for r in rows]
        ambiguous = unresolved if RANKED_TIER12 else len(cands) > 1
        return _result(
            cands,
            ambiguous=ambiguous,
            reason=("unresolved_synonym_tie_after_ranking" if RANKED_TIER12
                    else "multiple_exact_synonym_matches") if ambiguous else None,
            trace=trace,
            rank_basis=basis,
        )

    # ==========================================
    # TIER 3: Semantic Vector Match (SapBERT)
    # ==========================================
    vector = get_sapbert_embedding(entity_text)
    alias_ids = _alias_expand_brand_to_generic(conn, search_text)
    rows = _tier3_semantic_rows(conn, vector, vocabs, domains, alias_ids=alias_ids)
    trace.append({
        "tier": "3 (Semantic)", "attempted": True, "hits": len(rows),
        "top_score": round(rows[0][4], 4) if rows else None,
        "runner_up_score": round(rows[1][4], 4) if len(rows) > 1 else None,
        "floor": TIER3_SIMILARITY_FLOOR,
        "alias_expanded": bool(alias_ids),
    })

    if not rows:
        conflict = _detect_domain_conflict(conn, search_text, vocabs, domains, entity_text,
                                           vector, gliner_label=gliner_label)
        if conflict:
            return _result(conflict["candidates"], ambiguous=True,
                           reason="label_domain_conflict", failed=True, conflict=conflict,
                           trace=trace)
        return _result([], ambiguous=True, reason="no_candidates_at_any_tier", failed=True,
                       trace=trace)

    cands = [_candidate(r, "3 (Semantic)", round(r[4], 4),
                       match_basis="verified_brand_alias" if r[0] in alias_ids else None)
            for r in rows]
    cands = _collapse_hierarchy_duplicates(conn, cands)
    cands = _prefer_lab_procedure_over_observable(conn, cands, gliner_label)
    top = cands[0]["similarity_score"]

    # HARD CUTOFF below TIER3_SIMILARITY_FLOOR (2026-08-15, user decision after
    # being shown the counter-evidence below -- see
    # docs/2026-08-15_Contradiction_Detection_Analysis.md for the full
    # discussion this originated from). Previously the below-floor top hit was
    # reported as a FAILED match but candidates were still forwarded, on the
    # theory that Stage 3's deeper-resolution mode could sort out a "close but
    # not close enough" match. Changed to a genuine drop: below the floor,
    # Stage 2b now reports NO_CANDIDATE (empty list) rather than forwarding
    # weak matches downstream.
    #
    # WHY THIS IS A DELIBERATE, INFORMED TRADE-OFF, NOT A CLEAN WIN. The
    # 2026-08-13 calibration session (docs/2026-08-13_Calibration_Diagnostics_And_Fixes.md
    # S5.3) measured Tier 3 accuracy by similarity bin on real gold data:
    # [0.7,0.8) 18.69%, [0.8,0.9) 29.80%, even [0.9,1.0) only 54.87% --
    # "TIER3_SIMILARITY_FLOOR=0.72 cannot be fixed by raising it; the signal
    # itself doesn't discriminate well enough at any point on this curve."
    # 0.72 is not a clean boundary between garbage and good matches -- some
    # genuinely correct low-score matches will now be lost as NO_CANDIDATE
    # instead of being forwarded to Stage 3 as a last resort (this WILL move
    # GOLD_MISSING upward; re-measure with scripts/score_gold_recall.py before
    # trusting this in production). The trade being made: fewer low-quality
    # candidates reach Stage 3/HITL where an anchoring-biased model or a
    # rushed reviewer might rubber-stamp one (e.g. galea->"Gale" at 0.7686,
    # this exact fix's motivating case -- note 0.7686 is ABOVE 0.72, so this
    # SPECIFIC case is not caught by this cutoff either; it was already an
    # argument the user weighed before choosing to proceed anyway).
    if top < TIER3_SIMILARITY_FLOOR:
        # A weak in-domain match may still be beaten by a strong out-of-domain
        # one, which is itself the signal that the label was wrong -- so the
        # conflict check runs here too, not only on a total miss. Kept even
        # under the hard cutoff: this is a different, independent signal
        # (GLiNER mislabeling) from Tier 3's own vector-similarity quality,
        # not a rescue of the low-score match itself.
        conflict = _detect_domain_conflict(conn, search_text, vocabs, domains, entity_text,
                                           vector, gliner_label=gliner_label)
        if conflict:
            return _result(conflict["candidates"], ambiguous=True,
                           reason="label_domain_conflict", failed=True, conflict=conflict,
                           trace=trace)
        # No fuzzy-merge rescue here (unlike the margin-ambiguity branch
        # below, which is unaffected by this change) -- adding a
        # lower-confidence supplemental candidate back in after a hard
        # cutoff would defeat the purpose of cutting at all.
        return _result([], ambiguous=True, reason="tier3_below_similarity_floor_rejected",
                       failed=True, trace=trace)

    if len(cands) > 1 and (top - cands[1]["similarity_score"]) < TIER3_AMBIGUITY_MARGIN:
        # Record 3, docs/Stage3_Open_Issues.md Issue 3: three semantically-close
        # misspelled-drug candidates (SPIRILENE/SPIRGETINE/SPIRAPRILAT) crowded
        # out the correctly-spelled concept, which never appeared in the Tier 3
        # list at all. This is exactly the situation the fuzzy supplement
        # targets -- Tier 3 is already telling us it's unsure between several
        # near-tied options, so a near-exact spelling match belongs in that
        # same conversation.
        cands, fuzzy_added = _merge_fuzzy(conn, search_text, vocabs, domains, cands, trace)
        reason = "tier3_top2_margin_below_threshold_fuzzy_added" if fuzzy_added else \
            "tier3_top2_margin_below_threshold"
        return _result(cands, ambiguous=True, reason=reason, trace=trace)

    # 2026-08-13 (Fix 1 follow-up, "Aldactone problem"). A KG-verified brand
    # alias can lose the raw cosine-similarity race to a false-friend string
    # match confident enough to clear BOTH the floor and the margin check
    # above (measured: "Aldactone" -> "propiolactone" at 0.83, margin 0.11 --
    # spironolactone sat at 0.56, present in cands but never reached Stage 3).
    # process_and_normalize_entities() only forwards the full candidate list
    # when ambiguous=True (mapping["candidates"][:1] otherwise, to keep
    # Stage 3's prompt small on genuinely settled matches) -- so an alias hit
    # that isn't candidates[0] must force ambiguous here, or the guarantee
    # Fix 1 exists to provide is silently undone one function later.
    if alias_ids and cands[0]["omop_concept_id"] not in alias_ids \
            and any(c["omop_concept_id"] in alias_ids for c in cands):
        return _result(cands, ambiguous=True, reason="alias_candidate_outranked", trace=trace)

    return _result(cands, ambiguous=False, reason=None, trace=trace)




def _merge_fuzzy(conn, search_text, vocabs, domains, cands, trace):
    """Appends any edit-distance candidates not already present, capped so an
    already-uncertain result cannot grow unboundedly. Mutates nothing;
    returns (merged_candidates, fuzzy_candidates_added)."""
    existing_ids = {c["omop_concept_id"] for c in cands}
    fuzzy = _fuzzy_typo_candidates(conn, search_text, vocabs, domains, exclude_ids=existing_ids)
    trace.append({"tier": "4 (Fuzzy)", "attempted": True, "hits": len(fuzzy)})
    if not fuzzy:
        return cands, []
    merged = (cands + fuzzy)[:CANDIDATE_LIMIT * 2]
    return merged, fuzzy




def _result(candidates: list, ambiguous: bool, reason, failed: bool = False,
            conflict: dict = None, trace: list = None,
            rank_basis: str = None) -> dict:
    """Shapes the return value, keeping the pre-existing top-level keys so
    existing consumers keep working while `candidates` carries the new
    information.

    rank_basis (2026-08-13) records WHICH tiebreak produced candidates[0] --
    "ranked_v1" or "concept_id_asc" -- and is persisted to
    normalized_entities.tier12_rank_basis. Without it the A/B between the two
    orderings would only be measurable per run; with it, a mixed database can
    still be split into the two populations and compared.
    """
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
    top["tier12_rank_basis"] = rank_basis
    return top




def compute_confidence_tier(gliner_confidence: float, normalization_ambiguous: bool,
                            expansion_ambiguous: bool, domain_conflict: bool = False,
                            crosses_sentence_boundary: bool = False,
                            short_token_text: str = None,
                            match_tier: str = None) -> tuple:
    """Stage 3's routing tier, and the reasons behind it.

    LOW if ANY of these independent signals fires. They capture genuinely
    different failure modes and none subsumes the others:
      * gliner_confidence (HIGH_GLINER_RISK_FLOOR) -- "the model was
        confident, and confidence is inversely related to accuracy here"
      * normalization             -- "which concept is it"
      * expansion                 -- "was the abbreviation read correctly at all"
      * domain_conflict           -- "was the extraction LABEL even right"
      * crosses_sentence_boundary -- "is this even ONE entity"
      * short_token_text          -- "is this a bare abbreviation Stage 2b's
        exact/synonym/semantic tiers are structurally unlikely to resolve
        correctly" (three sub-signals: short_token, isupper_abbreviation,
        alnum_mix -- see the constants above this function for what each
        catches and the measured hit rate that shaped its bound)
    An entity can have a single clean, unambiguous OMOP match and still deserve
    deeper resolution because the span itself was shaky, or because it came
    from an abbreviation with three plausible readings.

    short_token_text is expected to be the entity's ORIGINAL (pre-expansion)
    surface form -- the heuristics were measured against original_text in
    scripts/measure_heuristic_and_boundary.py, and process_and_normalize_
    entities() passes orig_text here for that reason. Passing None (the
    default) skips all three sub-signals, preserving prior behavior for any
    other caller that hasn't been updated to supply it.

    2026-08-11 TIER-GATED EXEMPTION (not label-gated -- see below). match_tier,
    when it is "1 (Exact)" or "2 (Synonym)", suppresses the short_token/
    isupper_abbreviation/alnum_mix sub-signals entirely. Rationale: a bare
    abbreviation like "WBC"/"RBC"/"MCV" that Stage 2b's own exact or synonym
    vocabulary already resolved cleanly doesn't need Stage 3's deeper
    resolution just for being short -- the match itself is high-precision.

    This is deliberately gated on match_tier, NOT on gliner_label ==
    "Lab Test" -- an earlier version of this fix proposed exempting the
    entire Lab Test label from the short-token penalty, which was rejected:
    it would also exempt genuinely risky short lab abbreviations (K, Na, Ca,
    Cl) that only resolved via a weaker Tier 3 (Semantic)/Tier 4 (Fuzzy)
    match, not just the safe ones that motivated the original proposal. A
    Tier 3/4 "K" match is exactly the kind of shaky resolution this whole
    heuristic exists to catch, so it must still fire. Only Tier 1/2 -- the
    tiers structurally unlikely to be wrong -- get the exemption. match_tier
    of None (unmeasured, or an older caller that hasn't been updated) does
    NOT qualify for the exemption, preserving the pre-existing conservative
    default.

    2026-08-13 SUPERSEDED -- ALL TIER-1/2 EXEMPTIONS REMOVED. The paragraph
    above describes behaviour this function no longer has. It is retained
    because the reasoning it records (why the exemption was gated on
    match_tier rather than on gliner_label == "Lab Test") is still sound and
    still worth not re-litigating; only its premise turned out to be false.

    The premise was that Tier 1/2 are "structurally unlikely to be wrong".
    evaluation/stage2b_cal_eval.py measured them at 52.71% (n=776) and 61.84%
    (n=532) against gold. On 2026-08-12 the weak_match_tier rule below dropped
    its own exemption in response; the other two were left in place and
    described as "harmless no-ops", on the grounds that weak_match_tier
    already forces LOW for the same entities.

    They were harmless for ROUTING and harmful for MEASUREMENT. Because the
    gates sat around the `reasons.append(...)` calls rather than around the
    tier decision, a Tier 1/2 entity's tier_reasons recorded only
    "weak_match_tier" no matter how many other signals were true of it. The
    next iteration -- deciding which of these signals actually predicts
    correctness, and therefore which exemption could be safely restored --
    needs precisely the data those gates were throwing away. Both are now
    removed. Routing behaviour is bit-identical (weak_match_tier already
    forces LOW for every known tier, and `reasons` is OR'd); what changes is
    that tier_reasons is now a complete record rather than a censored one.
    """
    reasons = []
    # 2026-08-11: polarity flipped from "low_gliner_confidence" (was: fire
    # below a floor) to "high_gliner_risk" (fires at/above a floor) -- see
    # HIGH_GLINER_RISK_FLOOR's definition above for the inverted-calibration
    # finding that motivated this.
    #
    # 2026-08-13 TIER-GATED EXEMPTION, same pattern and rationale as the
    # short_token/isupper_abbreviation/alnum_mix trio above: measured via
    # scripts/measure_gliner_risk_vs_match_tier.py over the 28-note re-
    # ingested batch. Within EITHER match_tier bucket, high_gliner_risk adds
    # no real discriminative signal -- confirmed-Tier-1/2 accuracy is 57.7%
    # whether high_gliner_risk fires or not (57.7% vs 56.2%), and in the
    # unconfirmed-Tier-3/4/0 bucket high_gliner_risk entities were actually
    # MORE accurate, not less (33.3% vs 27.5%) -- the opposite of what "risk"
    # should predict. match_tier is doing the real work; gate this signal on
    # it the same way, so a high GLiNER-confidence span that Stage 2b's own
    # exact/synonym vocabulary already confirmed doesn't get pushed to
    # Stage 3 purely for a confidence score that measurement shows isn't
    # predictive once match_tier is known. match_tier of None (unmeasured)
    # does NOT qualify, same conservative default as the trio's gate.
    #
    # 2026-08-13 EXEMPTION REMOVED (docs/2026-08-13_Code_Improvement_
    # Proposals.md P1.2). The Tier-1/2 gate that used to wrap this append is
    # gone. ROUTING IS UNCHANGED -- weak_match_tier below already forces LOW
    # for every known match_tier, and `reasons` is OR'd -- but the RECORD is
    # not: while the gate stood, high_gliner_risk was never appended for a
    # Tier 1/2 entity, so tier_reasons was systematically blank on exactly the
    # population you would have to study to justify ever restoring the
    # exemption. Measuring a signal requires observing it fire. The gate
    # suppressed the observation, not the risk.
    if gliner_confidence is not None and gliner_confidence >= HIGH_GLINER_RISK_FLOOR:
        reasons.append("high_gliner_risk")
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

    # 2026-08-11 short-token/abbreviation signals. Independent of each other
    # (an entity can trip more than one) and independent of gliner_confidence
    # -- a short abbreviation reported at high confidence is still a short
    # abbreviation.
    #
    # 2026-08-13: Tier-1/2 exemption removed here too, same reasoning as
    # high_gliner_risk above and with the same zero routing impact. These three
    # sub-signals are the ones with concrete measured hit rates behind them
    # (scripts/measure_heuristic_and_boundary.py); censoring them on Tier 1/2
    # made those hit rates unmeasurable on more than half the corpus.
    if short_token_text:
        t = short_token_text.strip()
        if t:
            if len(t) <= SHORT_TOKEN_MAX_LEN:
                reasons.append("short_token")
            if len(t) <= ISUPPER_MAX_LEN and t.isupper() and any(c.isalpha() for c in t):
                reasons.append("isupper_abbreviation")
            if _ALNUM_MIX_RE.search(t):
                reasons.append("alnum_mix")

    # 2026-08-12 weak_match_tier. Independent of every signal above --
    # measured concretely on a real re-ingestion run: Glucose-72 and
    # UreaN-15/UreaN-17 resolved via Tier 3 (SapBERT semantic similarity,
    # not an exact/synonym confirmation) and STILL landed HIGH tier, because
    # nothing above checks which tier actually produced the match -- only
    # whether it was ambiguous. A Tier 3/4 match can be perfectly
    # unambiguous (no close runner-up candidate) and still be wrong; that is
    # exactly what Tier 3/4 mean here (see _lookup_tier12()'s own docstring:
    # "Tier 1/2 only keeps false-split/false-growth risk low" -- the same
    # principle applies to ordinary normalization, not just compound-split
    # candidates). This closes that gap: any match_tier other than Exact/
    # Synonym routes LOW regardless of label, ambiguity, or every other
    # signal -- a semantic guess should never silently outrank Stage 3
    # review just because no second candidate happened to compete with it.
    #
    # 2026-08-13 CORRECTION -- Tier 1/2 exemption removed. This rule (and
    # the high_gliner_risk / short_token exemptions above it) exempted Tier
    # 1 "Exact" and Tier 2 "Synonym" on the assumption that those tiers are
    # "structurally unlikely to be wrong" (see the Tier-gated-exemption
    # comment above this function, 2026-08-11). evaluation/stage2b_cal_eval.py
    # run against the 30-note corpus directly measured that assumption as
    # false: Tier 1 is 52.71% accurate against gold (776 gradable), Tier 2
    # is 61.84% (532 gradable) -- neither clears any reasonable bar for
    # "high-precision, skip review." Dropping the tuple exclusion here means
    # every entity with a KNOWN match_tier (1, 2, 3, or 0) now routes LOW
    # regardless of tier -- there is currently no match_tier value this
    # pipeline's own measurement can call trustworthy enough to silently
    # skip Stage 3. The high_gliner_risk and short_token Tier-1/2
    # exemptions above are left as-is (harmless now: this rule already
    # forces LOW for the same entities via `reasons`, an OR'd list), rather
    # than also rewritten, to keep this a single, auditable line-level
    # change against a specific measurement rather than a broader rewrite
    # under time pressure.
    #
    # match_tier of None (unmeasured, or an older caller that hasn't been
    # updated) does NOT trigger this -- same conservative-default choice as
    # the short-token exemption above, for the same reason: "we don't know"
    # must not be silently treated as either "safe" or "risky".
    if match_tier is not None:
        reasons.append("weak_match_tier")

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
        # 2026-08-13: which Tier 1/2 ranking produced the top-1 pick. NULL on
        # rows written before RANKED_TIER12 existed, "concept_id_asc" when the
        # legacy arbitrary tiebreak ran, "ranked_v1" when the ranker did. This
        # is what makes the A/B in report S5.3 gradable per row rather than
        # per run -- see _rank_tier12_candidates().
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS tier12_rank_basis VARCHAR;",
    ] + provenance_alter_statements("normalized_entities"):
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
        #
        # CACHE-KEY CANONICALIZATION ONLY (2026-08-15, Implementation_Checklist.md
        # Stage 2b item: near-duplicate spans like "abd distension" /
        # "ABD distension" / "abd  distension" being independently normalized
        # and landing on different concepts -- a real bug, since the exact-string
        # cache_key treated them as three unrelated lookups). Whitespace is
        # collapsed and case is folded for the KEY ONLY -- the actual
        # normalize_entity() call two lines below is still passed the
        # UNMODIFIED expanded_text. Deliberately not touching what
        # normalize_entity()/Tier 1-2's SQL actually matches on: this
        # corpus's Tier 1/2 queries are case-sensitive by design, and
        # docs/2026-08-14_GOLD_MISSING_RootCause_Fixes.md already documents a
        # real case-sensitivity collision (CTA vs cTa) as a DELIBERATELY
        # DEFERRED trade-off precisely because folding case there risks
        # regressions without corpus-wide A/B data. Folding only the cache
        # key means near-duplicate mentions now share one computation (fixing
        # the inconsistency bug) without silently re-deciding that separate,
        # already-weighed trade-off.
        canonical_text = re.sub(r"\s+", " ", expanded_text).strip().lower()
        cache_key = (canonical_text, label,
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

            # LAB VALUE SUFFIX FALLBACK (2026-08-10, extended 2026-08-11 to
            # try MULTIPLE candidates -- see strip_lab_value_suffix()'s
            # docstring and this module's "2026-08-11 DELIMITER-LESS CASE"
            # comment above for the full rationale). Tries whichever text
            # form(s) actually carry the pattern -- expanded first (matching
            # the two fallbacks above), then original, in case abbreviation
            # expansion altered the hyphen/spacing -- and within each text
            # form, tries every candidate strip_lab_value_suffix() returns
            # in order.
            #
            # 2026-08-12 GATE WIDENED from "== 0 (Failed)" to "not in
            # (1, 2)": a Tier 1/2 exact/synonym match on the cleanly-stripped
            # name should always be preferred over ANY weaker result on the
            # raw noisy text, not only over a total failure.
            #
            # 2026-08-12 ACCEPTANCE RULE, ATTEMPT 1 AND WHY IT REGRESSED.
            # First tried "only adopt if the stripped candidate reaches Tier
            # 1/2" -- reasoning that a Tier 3/4 stripped guess is not a
            # verified improvement. Re-ingestion promptly disproved this:
            # WBC-7, RBC-3, Hgb-9.9, MCV-97, MCHC-32, RDW-12, Glucose-72 and
            # UreaN-15/17 all went from "Tier 3, some plausible concept" to
            # "0 (Failed), Unmapped". Root cause: the RAW text ("WBC-7") was
            # already below TIER3_SIMILARITY_FLOOR and reporting "0 (Failed)"
            # even before today -- the OLD fallback (gated the same way) was
            # ALREADY firing and adopting whatever the stripped text got
            # (its old acceptance bar was simply "not 0 (Failed)"), which is
            # how "WBC-7" ever showed "Leucocyte count" via Tier 3 on "WBC"
            # in the first place. Requiring Tier 1/2 specifically rejected
            # that stripped Tier-3 result and fell back to the ORIGINAL
            # raw-text mapping -- itself "0 (Failed)" -- discarding a useful
            # near-miss candidate instead of improving on it.
            #
            # FIXED via explicit tier-rank comparison: adopt the stripped
            # candidate whenever its tier is STRICTLY BETTER (lower rank)
            # than whatever mapping already has, not only when it reaches
            # Tier 1/2. This restores the old "surface the best available
            # near-miss to Stage 3" behavior for the Failed->Tier3 case,
            # while still fixing the original problem this gate exists for
            # (a Tier 1/2 match on "WBC" beating a Tier 3 guess on "WBC-7")
            # and never adopting a same-or-worse-tier stripped guess just
            # because it happens to be a different string. Stops searching
            # further candidates only once Tier 1/2 is reached -- nothing
            # can beat a confirmed match, so there is no reason to keep
            # trying once one is found; a Tier 3/4 improvement keeps the
            # best seen so far but keeps looking, in case a LATER candidate
            # in the list reaches the confirmed tier instead (see strip_lab_
            # value_suffix()'s own docstring: "a wrong candidate earlier in
            # the list can never win over a right one later in it" -- that
            # guarantee only holds if a merely-better-but-unconfirmed early
            # hit does not stop the search).
            if mapping["match_tier"] not in ("1 (Exact)", "2 (Synonym)") and label == "Lab Test":
                pre_strip_tier = mapping["match_tier"]
                best_mapping = mapping
                best_source = None
                confirmed = False
                for candidate_text, source_name in (
                        (expanded_text, "expanded"), (orig_text, "original")):
                    for stripped in strip_lab_value_suffix(candidate_text or ""):
                        retry = normalize_entity(stripped, conn, gliner_label=label,
                                                 domain_override=domain_override)
                        if _LAB_TIER_RANK.get(retry["match_tier"], 9) < \
                                _LAB_TIER_RANK.get(best_mapping["match_tier"], 9):
                            best_mapping = retry
                            best_source = f"value_stripped_from_{source_name}:{stripped}"
                        if retry["match_tier"] in ("1 (Exact)", "2 (Synonym)"):
                            confirmed = True
                            break
                    if confirmed:
                        break
                if best_source:
                    mapping = best_mapping
                    # Records the winning candidate string AND what tier it
                    # upgraded from, not just which text form it came from --
                    # with multiple candidates now possible per form,
                    # "expanded" alone no longer says whether "RR" or "RR1"
                    # was the one that won, and the upgrade note makes it
                    # visible in an audit whether this reached a confirmed
                    # Tier 1/2 match or only improved on a weaker guess.
                    normalized_from = f"{best_source} (upgraded_from_{pre_strip_tier})"

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
            short_token_text=orig_text,
            match_tier=mapping["match_tier"],
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
            mapping.get("tier12_rank_basis"),
            # Hashed over `forwarded` -- the list Stage 3 will actually be
            # shown -- so mollm_decisions.candidates_hash and this column are
            # directly comparable. A mismatch on the same entity_id is exactly
            # the candidate-list drift report S6 could not previously detect.
            *provenance_params(candidates=forwarded),
        ))

    if rows_to_insert:
        conn.executemany(f"""
        INSERT INTO normalized_entities
        (note_id, original_text, expanded_text, gliner_label, gliner_confidence,
         omop_concept_id, omop_concept_name, omop_domain, omop_vocab,
         match_tier, similarity_score, is_test, entity_id, candidates,
         is_ambiguous, ambiguity_reason, confidence_tier_in, tier_reasons,
         domain_conflict, tier_trace, matched, normalized_from,
         athena_vocabulary_release, domain_id_queried, vocab_queried,
         sapbert_pooling_method, tier12_rank_basis, {provenance_column_sql()})
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                {provenance_placeholders()})
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
            sapbert_pooling_method = EXCLUDED.sapbert_pooling_method,
            tier12_rank_basis = EXCLUDED.tier12_rank_basis,
            -- Provenance is OVERWRITTEN on conflict, deliberately: this row
            -- now describes the latest normalization run, so its created_at/
            -- run_id/code_version/candidates_hash must describe that run too.
            -- Leaving the original values would make the row claim provenance
            -- it no longer has -- worse than no provenance at all.
            created_at = EXCLUDED.created_at,
            run_id = EXCLUDED.run_id,
            code_version = EXCLUDED.code_version,
            candidates_hash = EXCLUDED.candidates_hash;
        """, rows_to_insert)

    return normalized_results

