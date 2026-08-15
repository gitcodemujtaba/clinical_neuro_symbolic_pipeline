"""
src/kg3_ingestion.py — Stage 4 KG3 write-back (Memgraph).

WHY THIS ONLY EVER RUNS FOR HUMAN_VERIFIED CASES. docs/Implementation_Checklist.md:
"Gate KG 3 write-back behind Stage 3 + the calibrated confidence threshold,
not Stage 2b confidence alone... a pseudo-labeling feedback-loop risk."
Measured 2026-08-14: AUTO_VALIDATED precision 39.4% on the freshly
re-validated corpus -- not yet safe to write unreviewed. Every case this
module ingests has already passed through src/hitl_queue.py's review
workflow (reviewer_decision in APPROVED/CORRECTED, final_ingestion_path=
'HUMAN_VERIFIED'); this module has no code path that writes anything else,
by construction -- see ingest_reviewed_case()'s validation.

WHY MERGE, NOT CREATE, EVERYWHERE. Idempotency: scripts/run_kg3_ingestion.py
is designed to be safely re-runnable (it only re-attempts cases whose
hitl_review_queue.ingested_at is still NULL), but a transaction that fails
AFTER creating some nodes and BEFORE the caller marks it ingested would
otherwise duplicate those nodes on retry. MERGE keyed on each node's natural
id (entity_id / mollm_call_id / hitl_case_id / omop_concept_id) makes a
retry a no-op for whatever already landed, not a duplicate.

CONNECTION PATTERN matches scripts/profile_databases.py exactly (same env
vars, same neo4j Bolt driver -- Memgraph is Bolt-compatible) rather than
introducing a second convention for the same two databases.
"""
import os

from neo4j import GraphDatabase

MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7688")
MEMGRAPH_AUTH = (os.getenv("MEMGRAPH_USER", ""), os.getenv("MEMGRAPH_PASSWORD", ""))


def get_memgraph_driver():
    return GraphDatabase.driver(MEMGRAPH_URI, auth=MEMGRAPH_AUTH)


class UningestibleCase(Exception):
    """Raised when a case reaches ingest_reviewed_case() without a resolvable
    omop_concept_id -- e.g. an APPROVED case sourced from
    mollm_review_decisions, which never stores a concept_id (see
    src/hitl_queue.py's _presented_suggestion_from_review() docstring).
    Raised loudly rather than silently skipped or guessed from a concept
    NAME string, matching this codebase's established "report exclusions
    loudly" policy (evaluation/cal_eval.py's module docstring) -- a case
    silently dropped here would look identical to one that was never queued.
    """


def resolve_concept_id(case: dict) -> int:
    decision = case["reviewer_decision"]
    if decision == "CORRECTED":
        if case.get("corrected_concept_id") is None:
            raise UningestibleCase(
                f"{case['hitl_case_id']}: reviewer_decision=CORRECTED but "
                f"corrected_concept_id is NULL"
            )
        return case["corrected_concept_id"]
    if decision == "APPROVED":
        suggested = (case.get("presented_suggestion") or {}).get("suggested_omop_concept_id")
        if suggested is None:
            raise UningestibleCase(
                f"{case['hitl_case_id']}: reviewer_decision=APPROVED but no "
                f"suggested_omop_concept_id was recorded (source="
                f"{case.get('source_table')}) -- re-review as CORRECTED with "
                f"an explicit concept_id instead"
            )
        return suggested
    raise UningestibleCase(
        f"{case['hitl_case_id']}: reviewer_decision={decision!r} is not "
        f"ingestible (only APPROVED/CORRECTED ever reach this function -- "
        f"see src/hitl_queue.py's load_ingestible_cases())"
    )


def _ingest_tx(tx, params: dict):
    tx.run(
        """
        MERGE (concept:Concept {omop_concept_id: $omop_concept_id})
        ON CREATE SET concept.vocabulary_id = $vocabulary_id,
                      concept.domain_id = $domain_id,
                      concept.concept_name = $concept_name

        MERGE (obs:PatientObservation {entity_id: $entity_id})
        SET obs.note_id = $note_id,
            obs.raw_text = $raw_text,
            obs.label = $label,
            obs.orig_start = $orig_start,
            obs.orig_end = $orig_end,
            obs.confidence = $confidence,
            obs.matched = true,
            obs.omop_concept_id = $omop_concept_id,
            obs.vocabulary_id = $vocabulary_id,
            obs.timestamp = timestamp()
        MERGE (obs)-[:INSTANCE_OF]->(concept)

        MERGE (decision:MoLLMDecision {mollm_call_id: $source_call_id})
        SET decision.source_table = $source_table,
            decision.routing_decision = $routing_decision,
            decision.confidence_tier_in = $confidence_tier_in,
            decision.composite_confidence = $composite_confidence
        MERGE (obs)-[:VALIDATED_BY]->(decision)

        MERGE (review:HITLReview {hitl_case_id: $hitl_case_id})
        SET review.queue_reason = $queue_reason,
            review.final_decision_status = $reviewer_decision,
            review.corrected_concept_id = $corrected_concept_id,
            review.review_duration = $review_duration
        MERGE (decision)-[:REVIEWED_BY]->(review)
        """,
        **params,
    )


def ingest_reviewed_case(driver, case: dict, entity_fields: dict):
    """Writes one HUMAN_VERIFIED case as a single atomic Cypher transaction:
    :PatientObservation -[:INSTANCE_OF]-> :Concept, plus the
    :MoLLMDecision -[:REVIEWED_BY]-> :HITLReview provenance chain per
    docs/Databases.md §3, via one execute_write() call (neo4j driver wraps
    the whole function in one transaction -- either all five MERGEs commit
    or none do).

    `case` is one row from src/hitl_queue.py's load_ingestible_cases().
    `entity_fields` is the caller's own DuckDB lookup of
    extracted_entities.{orig_start,orig_end,confidence} for case['entity_id']
    -- kept as a separate parameter rather than a query this module runs
    itself, so this module's only external dependency is Memgraph, not also
    DuckDB (scripts/run_kg3_ingestion.py owns that join).

    Raises UningestibleCase (not silently) when the case has no resolvable
    concept_id -- see resolve_concept_id().
    """
    omop_concept_id = resolve_concept_id(case)
    suggestion = case.get("presented_suggestion") or {}
    params = {
        "omop_concept_id": omop_concept_id,
        "vocabulary_id": entity_fields.get("vocabulary_id"),
        "domain_id": entity_fields.get("domain_id"),
        "concept_name": entity_fields.get("concept_name"),
        "entity_id": case["entity_id"],
        "note_id": case["note_id"],
        "raw_text": suggestion.get("original_text"),
        "label": suggestion.get("entity_label"),
        "orig_start": entity_fields.get("orig_start"),
        "orig_end": entity_fields.get("orig_end"),
        "confidence": entity_fields.get("confidence"),
        "source_call_id": case["source_call_id"],
        "source_table": case["source_table"],
        "routing_decision": suggestion.get("routing_decision"),
        "confidence_tier_in": suggestion.get("confidence_tier_in"),
        "composite_confidence": suggestion.get("composite_confidence"),
        "hitl_case_id": case["hitl_case_id"],
        "queue_reason": case.get("queue_reason"),
        "reviewer_decision": case["reviewer_decision"],
        "corrected_concept_id": case.get("corrected_concept_id"),
        "review_duration": case.get("review_duration"),
    }
    with driver.session() as session:
        session.execute_write(_ingest_tx, params)
