"""
src/clinical_pipeline.py

Orchestrator for Stage 1 (preprocessing) -> Stage 2a (GLiNER-BioMed entity
extraction + assertion detection + GLiNER-relex relation extraction) ->
Stage 2b (OMOP normalization). Exists so "run the pipeline on a note" is one
function call with one obvious owner, rather than every caller (the Streamlit
runner page, eval scripts, the forthcoming Stage 3) re-deriving the stage
wiring itself.

2026-08-08 CHANGES

  * DEDUPLICATION MOVED INTO STAGE 2B, AND CHANGED FROM DROP TO FAN-OUT.
    This orchestrator used to collapse entities to one row per distinct
    original_text before calling normalization, which avoided re-querying OMOP
    for repeated surface forms but also meant only ONE of several identical
    mentions ever received a normalization result. That was tolerable while
    Stage 2b output was a console printout; it is not tolerable now that Stage
    3 runs per entity_id and Stage 4 hangs a provenance chain off each one --
    mentions 2..n would have had no concept mapping to validate and would have
    silently vanished from the audit trail. process_and_normalize_entities()
    now caches by (expanded_text, gliner_label) internally, so the redundant
    SapBERT/DuckDB work is still avoided while every entity_id gets its own
    row. The dedupe_entities parameter is therefore gone: there is no longer a
    correctness/cost trade-off for a caller to make.

  * RELATIONS NOW RECEIVE THE CANONICAL ENTITY LIST so their endpoints can be
    linked to entity_id by character-offset overlap (see src/extraction.py).

  * Stage 2a now returns dicts rather than positional tuples (see
    src/entity_extraction.py's docstring for why).

2026-08-09: severity 5-6 closed. Tier trace, athena_vocabulary_release,
matched, normalized_from, sub-threshold retention and flat_ner consistency are
all now implemented -- see each module and docs/Stage1_2_Completeness_Audit.md.
The sub-threshold filter below is the one with teeth: it is what keeps
retained-for-study spans out of every downstream stage.
"""

import os
import duckdb

from src.preprocessing import process_and_store_note, section_for_offset, segment_sections
from src.entity_extraction import (
    extract_and_store_entities,
    store_entities,
    map_offsets_to_original,
    make_entity_id,
    build_local_context,
    sentence_ids_spanned,
    find_sentence,
)
from src.extraction import extract_and_store_relations
from src.normalization import (
    process_and_normalize_entities,
    find_compound_split,
    find_span_growth,
    DOMAIN_TO_GLINER_LABEL,
)
from src.acronym_escalation import ACRONYM_ESCALATION_ENABLED, resolve_ambiguous_acronyms
from src.db_utils import connect_with_retry
from src.physexam_shorthand import build_physexam_shorthand_entities
from src.lab_abbrev_coldstart import build_lab_abbrev_coldstart_entities
from src.narrative_state_word_coldstart import build_narrative_state_word_entities

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")


def _build_split_or_grown_entity(ent, note_id, raw_text, expanded_text, expansions,
                                 sections, sentences, abs_exp_start, abs_exp_end,
                                 new_text, candidate, provenance_key, parent_id):
    """Shared entity-construction logic for both split halves/parts and a
    grown replacement -- both need the same offset-mapping, relabeling and
    section/sentence recomputation, just from different callers with
    different provenance-field names (compound_split_of vs grown_from).
    Factored out so those two paths cannot silently drift apart on any of
    these fields, the same anti-duplication reasoning as
    src/normalization.py's _tier_queries().

    domain_override is set directly from `candidate`'s own resolved
    domain(s) -- see normalize_entity()'s domain_override docstring
    (src/normalization.py) for why this, not a DOMAIN_TO_GLINER_LABEL
    lookup, is what keeps Stage 2b's re-normalization from excluding the
    very match this function's caller already confirmed exists, and for
    why it can be MULTIPLE domains (2026-08-10 gap 3 refinement) rather
    than the single top-ranked one -- so a genuinely ambiguous qualifier
    word (multiple distinct "Left" concepts, say) gets flagged ambiguous
    by Stage 2b's own machinery instead of silently resolved against
    whichever domain happened to rank first.

    candidate is defensively allowed to be None, though as of 2026-08-10
    neither caller currently produces that: find_compound_split() briefly
    (same day) had an "at-most-one-unresolved-part" relaxation that
    produced None-candidate ("deferred") parts, but it was reverted after a
    measured regression -- see find_compound_split()'s "REVERT NOTICE"
    docstring in src/normalization.py for why. Kept here rather than
    stripped out: if candidate is ever None again, the new entity falls
    back to the PARENT's own entity_label with no domain_override, i.e.
    Stage 2b normalizes it exactly like an ordinary never-split entity,
    which is the safe default regardless of which detector produced it.
    """
    orig_start, orig_end = map_offsets_to_original(abs_exp_start, abs_exp_end, expansions)
    orig_text = raw_text[orig_start:orig_end]

    if candidate is not None:
        label = DOMAIN_TO_GLINER_LABEL.get(candidate["domain_id"], ent["entity_label"])
        domain_override = candidate.get("domain_ids") or [candidate["domain_id"]]
    else:
        label = ent["entity_label"]
        domain_override = None

    new_id = make_entity_id(note_id, orig_start, orig_end, label)
    section = section_for_offset(sections, orig_start)
    context = build_local_context(expanded_text, sentences, abs_exp_start, abs_exp_end)
    spanned = sentence_ids_spanned(sentences, abs_exp_start, abs_exp_end)

    new_ent = dict(ent)
    new_ent.update({
        "entity_id": new_id,
        "entity_label": label,
        "expanded_text": new_text,
        "original_text": orig_text,
        "orig_start": orig_start,
        "orig_end": orig_end,
        "exp_start": abs_exp_start,
        "exp_end": abs_exp_end,
        "section_name": section["name"] if section else None,
        "sentence_id": context["sentence_id"],
        "local_context": context["text"],
        "local_context_basis": context["basis"],
        "crosses_sentence_boundary": len(spanned) > 1,
        "sentence_ids_spanned": spanned,
        "domain_override": domain_override,
    })
    new_ent[provenance_key] = parent_id
    return new_ent


def split_compound_entities(accepted: list, conn, note_id: str, raw_text: str,
                            expanded_text: str, stage1_provenance: dict,
                            is_test: bool) -> list:
    """Replaces any accepted entity whose text carries two OR MORE SNOMED
    concepts (per normalization.find_compound_split()) with that many atomic
    split-part entities, each independently normalized by Stage 2b. Returns
    the (possibly larger) entity list Stage 2b should actually normalize.

    2026-08-10, built directly from scripts/score_gold_recall.py's
    compound-span measurement on 17751158-DS-19: "gunshot wound to abdomen"
    -> gold wants "gunshot wound" (56768003) AND "abdomen" (818983003) as
    separate links; a one-entity-one-concept design can satisfy at most one.
    EXTENDED 2026-08-10 (docs/Stage2_Compound_And_Qualifier_Gaps.md gap 1)
    from binary-only to N-way after the large gold note showed gold
    3-way-decomposing laterality+device+action procedures ("right EVD
    placement" -> 'right' + 'EVD' + 'placement') that a 2-way splitter
    cannot address at all. See src/normalization.py's "COMPOUND-SPAN
    SPLITTING" section for the detector itself and why it is restricted to
    Tier 1/2 (no semantic guessing) and to the fewest parts the evidence
    supports.

    WHY HERE, NOT IN entity_extraction.py. Splitting needs a SNOMED
    vocabulary lookup, and entity_extraction.py deliberately has no DB-
    vocabulary dependency of its own. Doing it here keeps that boundary
    intact and keeps this orchestrator, which already owns "how Stage 2a's
    raw output becomes what Stage 2b consumes" (see the 2026-08-08
    DEDUPLICATION note above), as the one place that decision is made.

    OFFSET CORRECTNESS, RELABELING AND WHAT IS/ISN'T RECOMPUTED --
    see _build_split_or_grown_entity()'s docstring for the shared mechanics.
    assertion_status/experiencer/temporality/assertion_cue* and the GLiNER
    confidence score are INHERITED from the parent for every part --
    recomputing assertion would mean re-running ConText per part, which is
    separate, riskier work not done here. This is a documented
    approximation, not an oversight.

    WHAT IS NOT PRESERVED. The merged entity itself (e.g. the single
    "gunshot wound to abdomen" fact) is replaced, not kept alongside its
    parts -- it can never correctly carry more than one SNOMED concept, so
    keeping it would leave a permanently-wrong row in normalized_entities.
    Its extracted_entities row is kept, not deleted, with
    superseded_by_split=TRUE for audit (this codebase's consistent
    flag-don't-delete policy). Relations already resolved against its
    entity_id (extract_and_store_relations() runs BEFORE this function, on
    the pre-split entity list) still point at the superseded entity_id --
    re-resolving relation endpoints against split parts is separate work,
    not done here.
    """
    expansions = stage1_provenance.get("expansions", [])
    sections = stage1_provenance.get("sections", [])
    sentences = stage1_provenance.get("sentences", [])

    result = []
    to_persist = []

    for ent in accepted:
        split = find_compound_split(conn, ent["expanded_text"], ent["entity_label"])
        if not split:
            result.append(ent)
            continue

        parent_exp_start = ent["exp_start"]
        new_parts = []
        for part in split["parts"]:
            abs_exp_start = parent_exp_start + part["start"]
            abs_exp_end = parent_exp_start + part["end"]
            new_ent = _build_split_or_grown_entity(
                ent, note_id, raw_text, expanded_text, expansions, sections,
                sentences, abs_exp_start, abs_exp_end, part["text"],
                part["candidate"], "compound_split_of", ent["entity_id"])
            new_ent["superseded_by_split"] = False
            new_parts.append(new_ent)

        result.extend(new_parts)
        to_persist.extend(new_parts)

        # Mark the parent's OWN already-persisted row superseded, by
        # resubmitting its (note_id, orig_start, orig_end, entity_label) key
        # -- store_entities()'s ON CONFLICT DO UPDATE lands on the existing
        # row instead of inserting a new one.
        parent_updated = dict(ent)
        parent_updated["superseded_by_split"] = True
        to_persist.append(parent_updated)

    if to_persist:
        store_entities(conn, to_persist, is_test=is_test)

    return result


def grow_entity_spans(accepted: list, conn, note_id: str, raw_text: str,
                      expanded_text: str, stage1_provenance: dict,
                      is_test: bool) -> list:
    """Replaces any accepted entity whose span is missing a qualifier gold
    treats as part of the concept (per normalization.find_span_growth())
    with a single widened entity. Returns the entity list Stage 2b should
    actually normalize -- same length as `accepted` (growth replaces one
    entity with one wider entity, unlike splitting which replaces one with
    several).

    2026-08-10, docs/Stage2_Compound_And_Qualifier_Gaps.md gap 2, built from
    the median gold note (19442119-DS-15): gold annotates "congestive heart
    failure" (42343007) as one concept while Stage 2a extracted only "heart
    failure", which resolves to a DIFFERENT, more generic concept
    (84114007) -- span-compound-splitting cannot fix this, the predicted
    span is too SHORT, not compound. This is the mirror-image operation:
    widen instead of narrow.

    SENTENCE-BOUNDED CONTEXT. find_span_growth() has no sentence awareness
    of its own (see its docstring) -- this function is what enforces it,
    slicing left/right context strings from the entity's OWN sentence only
    (via find_sentence(), the same helper src/entity_extraction.py's
    build_local_context() uses), so growth can never absorb a word from an
    unrelated adjacent statement.

    OFFSET CORRECTNESS, RELABELING, WHAT IS/ISN'T RECOMPUTED -- see
    _build_split_or_grown_entity()'s docstring; identical reasoning to
    split_compound_entities() applies here (assertion/confidence inherited
    from the pre-growth entity, section/sentence/context recomputed).

    RUN ORDER. Called BEFORE split_compound_entities() in run_pipeline() --
    a grown entity is then handed straight into split-detection too, which
    is provably a no-op for it: find_compound_split()'s own whole-phrase
    guard re-checks Tier 1/2 on the (now-grown) text first, and
    find_span_growth() only just confirmed that exact text resolves, so the
    guard always fires and no further splitting is attempted. Growth and
    splitting have not been observed to both apply to the same entity in
    any measured case; this ordering is safe by construction if that ever
    changes.

    WHAT IS NOT PRESERVED. The pre-growth (narrower) entity is replaced, not
    kept alongside the grown one -- keeping both would double-count the
    same clinical mention. Its extracted_entities row is kept, not deleted,
    with superseded_by_growth=TRUE for audit, same policy as
    superseded_by_split.
    """
    expansions = stage1_provenance.get("expansions", [])
    sections = stage1_provenance.get("sections", [])
    sentences = stage1_provenance.get("sentences", [])

    result = []
    to_persist = []

    for ent in accepted:
        exp_start, exp_end = ent["exp_start"], ent["exp_end"]
        sent = find_sentence(sentences, exp_start) if sentences else None
        sent_start = sent["start"] if sent else max(0, exp_start - 200)
        sent_end = sent["end"] if sent else min(len(expanded_text), exp_end + 200)

        left_context = expanded_text[sent_start:exp_start]
        right_context = expanded_text[exp_end:sent_end]
        if not left_context and not right_context:
            result.append(ent)
            continue

        growth = find_span_growth(
            conn, ent["expanded_text"], left_context, right_context,
            ent["entity_label"])
        if not growth:
            result.append(ent)
            continue

        abs_exp_start = exp_start - growth["grow_left_chars"]
        abs_exp_end = exp_end + growth["grow_right_chars"]
        new_ent = _build_split_or_grown_entity(
            ent, note_id, raw_text, expanded_text, expansions, sections,
            sentences, abs_exp_start, abs_exp_end, growth["text"],
            growth["candidate"], "grown_from", ent["entity_id"])
        new_ent["superseded_by_growth"] = False

        result.append(new_ent)
        to_persist.append(new_ent)

        parent_updated = dict(ent)
        parent_updated["superseded_by_growth"] = True
        to_persist.append(parent_updated)

    if to_persist:
        store_entities(conn, to_persist, is_test=is_test)

    return result


def run_pipeline(note_id: str, raw_text: str, conn, is_test: bool = False) -> dict:
    """Runs Stage 1 -> Stage 2a -> Stage 2b on a single note over an
    already-open DuckDB connection, returning everything each stage produced so
    a caller can inspect intermediate output without re-running earlier stages.

    conn is accepted rather than opened here so callers that already hold a
    connection (the Streamlit UI, a batch loop over many notes) aren't forced
    to open a second one -- see run_pipeline_for_db_path() for the convenience
    case.

    is_test is forwarded to all stages so rows written by smoke tests are
    flagged and can be purged without touching production rows.
    """
    expanded_text, stage1_provenance = process_and_store_note(
        note_id, raw_text, conn, is_test=is_test
    )

    entities = extract_and_store_entities(
        note_id, expanded_text, raw_text, stage1_provenance, conn, is_test=is_test
    )

    # GLiNER-relex does its own internal entity detection and cannot accept
    # pre-extracted spans, so it runs on expanded_text rather than on
    # `entities`. The canonical entities are passed in only so its endpoints
    # can be resolved back to entity_id by offset overlap after the fact.
    relations = extract_and_store_relations(
        note_id, expanded_text, conn, entities=entities, is_test=is_test
    )

    # SUB-THRESHOLD FILTER. entity_extraction.py now extracts down to
    # SUBTHRESHOLD_FLOOR (0.35) and flags anything under EXTRACTION_THRESHOLD
    # (0.50) as below_threshold. Those spans are STORED for analysis but have
    # not passed the extraction gate, so they must not reach Stage 2b, Stage 3,
    # KG3 or any evaluation count -- normalising them would silently trade
    # precision for a recall number nobody asked for.
    #
    # This single line is the entire mechanism keeping retained-for-study
    # separate from accepted-as-real, which is why it is here in the
    # orchestrator rather than left to each caller to remember.
    accepted = [e for e in entities if not e.get("below_threshold")]
    subthreshold = [e for e in entities if e.get("below_threshold")]

    # PHYSICAL-EXAM SHORTHAND COLD START (2026-08-18). GLiNER never proposes
    # this compressed telegraphic notation (VS/BP/HEENT/NAD/S-NT-ND/...) as
    # entities at ANY confidence -- confirmed corpus-wide, see
    # src.physexam_shorthand's own module docstring for the sizing. Injected
    # here (after the sub-threshold filter, so these are never confused with
    # a low-confidence GLiNER guess; before growth/split, since these spans
    # are already atomic and complete) directly from raw_text, skipping any
    # span a GLiNER entity already covers.
    physexam_entities = build_physexam_shorthand_entities(
        raw_text, segment_sections(raw_text), note_id, entities)
    if physexam_entities:
        store_entities(conn, physexam_entities, is_test=is_test)
        accepted = accepted + physexam_entities

    # BARE LAB-ABBREVIATION COLD START (2026-08-20). Same category of gap
    # as the physexam cold start above, different population: bare CBC/
    # chemistry-panel abbreviations with no attached value ("Creat trended
    # up", not "Creat-1.2") -- confirmed corpus-wide (full 140-note
    # is_test corpus) that 100% of the 20,754 gold spans with zero
    # overlapping prediction are true zero-shot blind spots (GLiNER
    # proposes nothing at any confidence, not just below-threshold), and
    # the most-repeated missed texts are dominated by exactly this
    # pattern. See src.lab_abbrev_coldstart's own module docstring for the
    # gold-verification methodology. Passed the COMBINED accepted list so
    # far (not just `entities`) so it also skips anything the physexam
    # cold start just added, not only GLiNER's own output. Unlike
    # physexam_shorthand, these need no dedicated concept-resolution
    # bypass -- they flow through ordinary Tier 1-3 normalization and hit
    # _LAB_TEST_ALIASES/tier3_fast_path's existing verified_lab_test_alias
    # branch automatically.
    lab_abbrev_entities = build_lab_abbrev_coldstart_entities(raw_text, note_id, accepted)
    if lab_abbrev_entities:
        store_entities(conn, lab_abbrev_entities, is_test=is_test)
        accepted = accepted + lab_abbrev_entities

    # NARRATIVE-STATE-WORD COLD START (2026-08-20). Second population from
    # the same miss analysis -- a small, gold-consistency-screened set of
    # common single-word state descriptors (alert/improved/baseline/warm/
    # clinic) GLiNER never proposes. See src.narrative_state_word_coldstart
    # for the >=95%-consistency screening that kept several tempting but
    # genuinely polysemous candidates (pain/stable/negative/...) OUT of
    # this dict. Passed the combined accepted list so far, same reasoning
    # as the lab-abbreviation cold start above.
    narrative_entities = build_narrative_state_word_entities(raw_text, note_id, accepted)
    if narrative_entities:
        store_entities(conn, narrative_entities, is_test=is_test)
        accepted = accepted + narrative_entities

    # SPAN GROWTH, then COMPOUND-SPAN SPLIT. Both run after the sub-
    # threshold filter (only real, accepted entities are worth the
    # vocabulary lookups these do) and before Stage 2b (so the grown/split
    # result is what actually gets normalized, not the pre-fix entity).
    # Growth runs FIRST -- see grow_entity_spans()'s "RUN ORDER" docstring
    # for why handing its output into splitting next is safe (provably a
    # no-op for anything growth touched). See each function's docstring for
    # the full rationale and what these deliberately do not do (relation
    # re-resolution, assertion recomputation).
    accepted = grow_entity_spans(
        accepted, conn, note_id, raw_text, expanded_text, stage1_provenance,
        is_test=is_test,
    )
    accepted = split_compound_entities(
        accepted, conn, note_id, raw_text, expanded_text, stage1_provenance,
        is_test=is_test,
    )

    # Phase 4 (Pass 1 acronym escalation), build-order step 4, 2026-08-16.
    # Gated behind ACRONYM_ESCALATION_ENABLED -- see src/acronym_escalation.py's
    # own docstring for why this stays off by default (only measured on ~16
    # entities so far). raw_text here is this function's own parameter, the
    # actual unexpanded note -- resolve_ambiguous_acronyms() needs that, not
    # expanded_text, so the model reads the note's literal wording rather
    # than Stage 1's own (possibly wrong) guess. Attaches
    # `mollm_resolved_expansion` onto each entity dict BEFORE
    # process_and_normalize_entities() runs -- that function's own
    # interceptor (is_allergy_context's sibling) reads it from there; no
    # further plumbing needed.
    if ACRONYM_ESCALATION_ENABLED:
        resolved_acronyms = resolve_ambiguous_acronyms(accepted, raw_text, note_id, conn)
        for ent in accepted:
            resolution = resolved_acronyms.get(ent.get("entity_id"))
            if resolution is not None:
                ent["mollm_resolved_expansion"] = resolution

    normalized = process_and_normalize_entities(accepted, conn, is_test=is_test)

    return {
        "note_id": note_id,
        "expanded_text": expanded_text,
        "stage1_provenance": stage1_provenance,
        "entities": accepted,
        # Returned separately, never merged into `entities`: a caller that
        # wants them has to ask, and one that doesn't cannot get them by
        # accident.
        "subthreshold_entities": subthreshold,
        "relations": relations,
        "normalized": normalized,
    }


def run_pipeline_for_db_path(note_id: str, raw_text: str, db_path: str = DB_PATH,
                             is_test: bool = False) -> dict:
    """Convenience wrapper for callers without an open DuckDB connection --
    opens one, runs run_pipeline(), and always closes it, including on error.

    2026-08-18: uses src.db_utils.connect_with_retry() rather than a bare
    duckdb.connect() -- this is now the PER-NOTE connection-cycling point
    for scripts/test_pipeline_e2e.py's batch loop (was previously one
    connection held open across the whole multi-note run), so a brief
    collision with something else's own short-lived connection (Streamlit's
    UI pages, this same pattern in a concurrently-run script) is expected
    and worth waiting out rather than crashing the whole note on."""
    conn = connect_with_retry(db_path, read_only=False)
    try:
        return run_pipeline(note_id, raw_text, conn, is_test=is_test)
    finally:
        conn.close()
