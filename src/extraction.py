"""
src/extraction.py — Stage 2a relation extraction ("ReLEx"), run over
expanded_text alongside (but independently of) src/entity_extraction.py's
canonical GLiNER-BioMed entity pass.

2026-08-07: Clinical-T5 was originally slated for this role but was removed
from the live pipeline (pretrained on MIMIC-III/IV -- contamination risk
against the evaluation notes), leaving Stage 2a relation extraction unowned.
GLiREL (jackboyla/glirel-large-v0) was tried first and abandoned: it produced
near-zero scores even on its own README example (documented 0.9923, actual
0.0028), root-caused to transformers==4.57.6/torch==2.8.0 being far newer than
what that single unmaintained "v0" checkpoint was validated against, with
DeBERTa-v3 known to drift numerically across transformers versions. Replaced
rather than chasing a downgrade that risked breaking GLiNER-BioMed/SapBERT.

CURRENT: GLiNER-relex (knowledgator/gliner-relex-large-v1.0) -- a joint
zero-shot NER+RE model on the same `gliner` library entity_extraction.py
already uses, actively maintained.

MedGemma 4B was considered and deliberately NOT used here: it is already
committed to Stage 3's MoLLM ensemble, and a model that both generated a
relation in Stage 2a and voted on validating it in Stage 3 would not be an
independent check -- which is the whole basis of the consensus-validation
claim.

2026-08-08 — ENTITY LINKING NOW USES OFFSETS, NOT TEXT MATCHING.
This module previously stored only head/tail TEXT and LABEL, discarding
GLiNER-relex's character offsets AND the entity list its internal NER pass
returns (it was assigned to `_entities` and thrown away). The earlier docstring
described cross-model span alignment as "a bigger, separate piece of work" and
deferred it -- but per docs/Stage1_2_Completeness_Audit.md S4, that job was
only hard BECAUSE the offsets had been discarded. Both models run over the same
expanded_text, so their spans live in one coordinate system: linking a relation
endpoint to a canonical extracted_entities row is a character-overlap test, not
a fuzzy text-similarity heuristic. Overlap is deterministic, explainable, and
reviewable -- text matching is none of those, and would have silently failed
whenever the two models tokenized a span slightly differently, which is
precisely the case it was meant to handle.

The trade-off that remains genuine: GLiNER-relex still does its own
(general-domain, not clinically tuned) entity detection internally and cannot
accept externally supplied spans, so its head/tail typing may differ from
GLiNER-BioMed's. entity_extraction.py remains the single source of truth for
extracted_entities; when an endpoint can't be linked with sufficient overlap it
is recorded as `unresolved` rather than guessed at, and Stage 3 treats that
endpoint as ungrounded free text.

2026-08-09 — SHARED FLAT_NER, AND A PREDICTION THAT WAS WRONG.
This module previously passed flat_ner=False (nested) while
entity_extraction.py used predict_entities()' flat default. That divergence was
a library-default accident rather than a decision, so both now import a single
FLAT_NER constant.

The change was expected to REDUCE relation yield, on the reasoning that nested
spans give relex more candidate endpoints to relate. Measured on note
10000032-DS-21 it did the opposite:

    before (nested relex):  3 relations, 2 with an unlinked endpoint
    after  (flat, shared):  3 relations, 0 unlinked

Same relations found, and endpoint linking went from 33% to 100%. The reason is
the linking step, not the extraction step: when both models produce flat spans
they agree on boundaries, so _overlap_ratio() clears MIN_ENDPOINT_OVERLAP that
it previously missed. The two unresolved endpoints under the old setting were
both nested relex spans ("right upper quadrant") with no flat counterpart in
extracted_entities.

Recorded because the prediction was confidently wrong in a way only the data
caught -- and because it means consistency between the two models is worth
MORE than the extra candidate endpoints nesting provided, which is the opposite
of the trade-off assumed when this was written.
"""

import os
import json
import hashlib
import duckdb
import warnings

warnings.filterwarnings("ignore")

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

print("Loading GLiNER-relex model... (this may take a moment on the first run)")
from gliner import GLiNER
from src.entity_extraction import CLINICAL_LABELS, FLAT_NER, map_offsets_to_original

RELEX_MODEL_NAME = "knowledgator/gliner-relex-large-v1.0"
model = GLiNER.from_pretrained(RELEX_MODEL_NAME)

# Clinical relation vocabulary. GLiNER-relex is zero-shot -- these labels are
# passed at inference time, so this list can be extended without retraining.
# "treated with" head/tail direction matches the
# [Medication]-[:TREATED_WITH]->[Condition] example in
# docs/Implementation_Methodology.md's Stage 2a description, verbatim.
RELATION_LABELS = [
    "treated with",
    "indicates",
    "causes",
    "located in",
    "measured by",
]

# GLiNER-relex's model card documents no allowed_head/allowed_tail constraint
# mechanism, so plausibility is enforced here as a post-hoc filter instead --
# the same pattern Stage 2b uses for concept candidates, applied to relation
# pairs.
RELATION_CONSTRAINTS = {
    "treated with": {"allowed_head": ["Medication", "Procedure"], "allowed_tail": ["Condition", "Symptom"]},
    "indicates": {"allowed_head": ["Lab Test", "Symptom"], "allowed_tail": ["Condition"]},
    "causes": {"allowed_head": ["Medication", "Condition"], "allowed_tail": ["Symptom", "Condition"]},
    "located in": {"allowed_head": ["Condition", "Symptom", "Procedure"], "allowed_tail": ["Anatomy"]},
    "measured by": {"allowed_head": ["Condition", "Symptom"], "allowed_tail": ["Lab Test"]},
}

# Per knowledgator/gliner-relex-large-v1.0's model card recommendations.
ENTITY_CONFIDENCE_FLOOR = 0.5
ADJACENCY_THRESHOLD = 0.5
RELATION_CONFIDENCE_FLOOR = 0.7

# Minimum character overlap between a GLiNER-relex endpoint span and a
# canonical extracted_entities span for the two to be treated as the same
# mention. 0.5 of the shorter span tolerates the boundary disagreements the two
# models genuinely have ("acute kidney injury" vs "kidney injury") while
# refusing loose partial overlaps ("pain" vs "chest pain radiating to the arm").
# Calibration target, per docs/MoLLM_Stage3_Retrieval_Design.md S8.
MIN_ENDPOINT_OVERLAP = 0.50


def _passes_label_constraint(relation_label: str, head_label, tail_label) -> bool:
    """Rejects a prediction if either endpoint label is missing or outside the
    allowed set for that relation type. Unknown relation labels pass through
    unfiltered rather than being rejected, so extending RELATION_LABELS without
    a matching RELATION_CONSTRAINTS entry doesn't silently drop everything."""
    constraint = RELATION_CONSTRAINTS.get(relation_label)
    if constraint is None:
        return True
    if not head_label or not tail_label:
        return False
    return head_label in constraint["allowed_head"] and tail_label in constraint["allowed_tail"]


def _span_offsets(endpoint) -> tuple:
    """Pulls (start, end) off a GLiNER-relex endpoint.

    Defensive because the library has used more than one key naming across
    versions. Returns (None, None) when offsets genuinely aren't present, which
    the caller records as an unresolved link rather than silently treating as
    offset 0 -- a wrong offset would link the endpoint to whatever entity
    happens to sit at the start of the note.
    """
    if not isinstance(endpoint, dict):
        return None, None
    for s_key, e_key in (("start", "end"), ("start_idx", "end_idx"), ("char_start", "char_end")):
        if s_key in endpoint and e_key in endpoint:
            try:
                return int(endpoint[s_key]), int(endpoint[e_key])
            except (TypeError, ValueError):
                return None, None
    return None, None


def _overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    """Character overlap as a fraction of the SHORTER span.

    Shorter-span denominator on purpose: a canonical entity `chest pain` fully
    contained in a relex endpoint `severe chest pain` should score 1.0, because
    they refer to the same mention. Using the union (Jaccard) would penalise
    exactly the nesting the two models are expected to disagree about.
    """
    overlap = min(a_end, b_end) - max(a_start, b_start)
    if overlap <= 0:
        return 0.0
    return overlap / max(1, min(a_end - a_start, b_end - b_start))


def _link_endpoint(exp_start, exp_end, entities: list) -> tuple:
    """Resolves an endpoint span to a canonical entity_id by best overlap.

    Returns (entity_id | None, overlap_ratio, status). Ties are broken by the
    larger overlap then the lower entity_id, so linking is deterministic across
    runs -- the same property Stage 2b's ORDER BY fix exists to guarantee.
    """
    if exp_start is None or exp_end is None:
        return None, 0.0, "no_offsets"
    if not entities:
        return None, 0.0, "no_candidate_entities"

    best = None
    best_ratio = 0.0
    for ent in entities:
        ratio = _overlap_ratio(exp_start, exp_end, ent["exp_start"], ent["exp_end"])
        if ratio > best_ratio or (ratio == best_ratio and best and ent["entity_id"] < best["entity_id"]):
            best, best_ratio = ent, ratio

    if best and best_ratio >= MIN_ENDPOINT_OVERLAP:
        return best["entity_id"], round(best_ratio, 4), "linked"
    return None, round(best_ratio, 4), "unresolved_low_overlap"


def make_relation_id(note_id: str, head_start, head_end, relation_label: str,
                     tail_start, tail_end) -> str:
    """Stable per-relation identifier, built from spans rather than text so it
    survives a re-run unchanged (docs/Provenance_Schema.md's `relation_id`)."""
    key = f"{head_start}:{head_end}:{relation_label}:{tail_start}:{tail_end}"
    return f"{note_id}-r{hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]}"


def extract_and_store_relations(note_id: str, expanded_text: str, conn,
                                entities: list = None, is_test: bool = False) -> list:
    """Runs GLiNER-relex over expanded_text and stores predicted relations.

    entities -- the canonical Stage 2a entity dicts from
    extract_and_store_entities(). Supplied so endpoints can be linked to
    entity_id by offset overlap. Passing None still works and still stores
    relations; every endpoint is simply recorded as unresolved, which is the
    honest representation rather than a silent degradation.
    """
    conn.sql("""
    CREATE TABLE IF NOT EXISTS extracted_relations (
        note_id VARCHAR,
        head_entity_text VARCHAR,
        head_entity_label VARCHAR,
        relation_label VARCHAR,
        tail_entity_text VARCHAR,
        tail_entity_label VARCHAR,
        relation_confidence FLOAT,
        is_test BOOLEAN DEFAULT FALSE,
        UNIQUE(note_id, head_entity_text, relation_label, tail_entity_text)
    );
    """)
    for ddl in [
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS is_test BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS relation_id VARCHAR;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS head_entity_id VARCHAR;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS tail_entity_id VARCHAR;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS head_exp_start INT;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS head_exp_end INT;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS tail_exp_start INT;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS tail_exp_end INT;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS head_orig_start INT;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS head_orig_end INT;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS tail_orig_start INT;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS tail_orig_end INT;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS head_link_status VARCHAR;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS tail_link_status VARCHAR;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS head_overlap_ratio FLOAT;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS tail_overlap_ratio FLOAT;",
        "ALTER TABLE extracted_relations ADD COLUMN IF NOT EXISTS relex_model_version VARCHAR;",
    ]:
        conn.sql(ddl)

    # Stage 1 expansions, so endpoint offsets can be mapped back to the
    # original note through the same "time machine" entities use -- without
    # this, a relation could not be located in the text a clinician actually
    # wrote, which the HITL reviewer needs to see.
    prov_row = conn.sql(
        "SELECT provenance FROM note_expansions WHERE note_id = ?", params=[note_id]
    ).fetchone()
    expansions = json.loads(prov_row[0]).get("expansions", []) if prov_row else []

    _relex_entities, raw_relations = model.inference(
        texts=[expanded_text],
        labels=CLINICAL_LABELS,
        relations=RELATION_LABELS,
        threshold=ENTITY_CONFIDENCE_FLOOR,
        adjacency_threshold=ADJACENCY_THRESHOLD,
        relation_threshold=RELATION_CONFIDENCE_FLOOR,
        return_relations=True,
        # Shared with entity_extraction.py rather than set independently. The
        # two Stage 2a models previously disagreed (flat there, nested here),
        # which was a library-default accident rather than a decision -- see
        # FLAT_NER's comment in that module.
        flat_ner=FLAT_NER,
    )
    raw_relations = raw_relations[0]

    processed_relations = []
    rows_to_insert = []
    unresolved_count = 0

    for rel in raw_relations:
        head, tail = rel["head"], rel["tail"]
        head_text, head_label = head["text"], head["type"]
        tail_text, tail_label = tail["text"], tail["type"]
        relation_label = rel["relation"]

        if not _passes_label_constraint(relation_label, head_label, tail_label):
            continue

        h_start, h_end = _span_offsets(head)
        t_start, t_end = _span_offsets(tail)

        head_id, head_overlap, head_status = _link_endpoint(h_start, h_end, entities)
        tail_id, tail_overlap, tail_status = _link_endpoint(t_start, t_end, entities)
        if head_status != "linked" or tail_status != "linked":
            unresolved_count += 1

        h_orig = map_offsets_to_original(h_start, h_end, expansions) if h_start is not None else (None, None)
        t_orig = map_offsets_to_original(t_start, t_end, expansions) if t_start is not None else (None, None)

        relation_id = make_relation_id(note_id, h_start, h_end, relation_label, t_start, t_end)
        confidence = round(rel["score"], 4)

        record = {
            "relation_id": relation_id,
            "note_id": note_id,
            "head_entity_id": head_id,
            "head_entity_text": head_text,
            "head_entity_label": head_label,
            "head_link_status": head_status,
            "head_overlap_ratio": head_overlap,
            "relation_label": relation_label,
            "tail_entity_id": tail_id,
            "tail_entity_text": tail_text,
            "tail_entity_label": tail_label,
            "tail_link_status": tail_status,
            "tail_overlap_ratio": tail_overlap,
            "relation_confidence": confidence,
            "head_exp_start": h_start, "head_exp_end": h_end,
            "tail_exp_start": t_start, "tail_exp_end": t_end,
            "head_orig_start": h_orig[0], "head_orig_end": h_orig[1],
            "tail_orig_start": t_orig[0], "tail_orig_end": t_orig[1],
        }
        processed_relations.append(record)
        rows_to_insert.append((
            note_id, head_text, head_label, relation_label, tail_text, tail_label,
            confidence, is_test, relation_id, head_id, tail_id,
            h_start, h_end, t_start, t_end,
            h_orig[0], h_orig[1], t_orig[0], t_orig[1],
            head_status, tail_status, head_overlap, tail_overlap, RELEX_MODEL_NAME,
        ))

    if unresolved_count:
        # Surfaced rather than silent: a high unresolved rate means the two
        # models are disagreeing about span boundaries far more than expected
        # and the 0.50 overlap threshold needs revisiting -- invisible if this
        # were only recorded per-row.
        print(f"ℹ️ {unresolved_count}/{len(processed_relations)} relations in {note_id} "
              f"had at least one unlinked endpoint.")

    if rows_to_insert:
        conn.executemany("""
        INSERT INTO extracted_relations
        (note_id, head_entity_text, head_entity_label, relation_label,
         tail_entity_text, tail_entity_label, relation_confidence, is_test,
         relation_id, head_entity_id, tail_entity_id,
         head_exp_start, head_exp_end, tail_exp_start, tail_exp_end,
         head_orig_start, head_orig_end, tail_orig_start, tail_orig_end,
         head_link_status, tail_link_status, head_overlap_ratio, tail_overlap_ratio,
         relex_model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (note_id, head_entity_text, relation_label, tail_entity_text)
        DO UPDATE SET
            head_entity_label = EXCLUDED.head_entity_label,
            tail_entity_label = EXCLUDED.tail_entity_label,
            relation_confidence = EXCLUDED.relation_confidence,
            is_test = EXCLUDED.is_test,
            relation_id = EXCLUDED.relation_id,
            head_entity_id = EXCLUDED.head_entity_id,
            tail_entity_id = EXCLUDED.tail_entity_id,
            head_exp_start = EXCLUDED.head_exp_start,
            head_exp_end = EXCLUDED.head_exp_end,
            tail_exp_start = EXCLUDED.tail_exp_start,
            tail_exp_end = EXCLUDED.tail_exp_end,
            head_orig_start = EXCLUDED.head_orig_start,
            head_orig_end = EXCLUDED.head_orig_end,
            tail_orig_start = EXCLUDED.tail_orig_start,
            tail_orig_end = EXCLUDED.tail_orig_end,
            head_link_status = EXCLUDED.head_link_status,
            tail_link_status = EXCLUDED.tail_link_status,
            head_overlap_ratio = EXCLUDED.head_overlap_ratio,
            tail_overlap_ratio = EXCLUDED.tail_overlap_ratio,
            relex_model_version = EXCLUDED.relex_model_version;
        """, rows_to_insert)

    return processed_relations
