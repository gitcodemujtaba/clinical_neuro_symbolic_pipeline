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

from src.preprocessing import process_and_store_note
from src.entity_extraction import extract_and_store_entities
from src.extraction import extract_and_store_relations
from src.normalization import process_and_normalize_entities

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")


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
        note_id, expanded_text, raw_text, conn, is_test=is_test
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
    opens one, runs run_pipeline(), and always closes it, including on error."""
    conn = duckdb.connect(db_path)
    try:
        return run_pipeline(note_id, raw_text, conn, is_test=is_test)
    finally:
        conn.close()
