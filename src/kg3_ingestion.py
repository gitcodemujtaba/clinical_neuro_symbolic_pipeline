"""
src/kg3_ingestion.py — Stage 4/Pass 4 KG3 write-back (Memgraph).

TWO WRITE PATHS, BOTH GATED, NEITHER OPTIONAL. ingest_reviewed_case() is the
original, human-reviewed path: docs/Implementation_Checklist.md's "gate KG3
write-back behind Stage 3 + the calibrated confidence threshold... a
pseudo-labeling feedback-loop risk" concern, and the measured 39.4%/52.6%
AUTO_VALIDATED precision under the OLD binary route() (src/mollm_ensemble.py)
this concern was raised about -- every case it ingests has already passed
through src/hitl_queue.py's review workflow (reviewer_decision in
APPROVED/CORRECTED), and it has no code path that writes anything else, by
construction (see its own validation).

ingest_auto_decision() (added for src/mollm_tier_gate.py's Tier 1-5 gate) is
the NEW, deliberately narrower unreviewed path -- it only accepts Tier 1/2/3
decisions, whose whole purpose is to be trustworthy enough to skip human
review, and defaults to dry_run=True so it writes nothing until that trust
has actually been validated on real data. See its own docstring for the
gating detail. The two paths write structurally identical graph shapes (a
:HITLReview node either way, distinguished by final_decision_status =
'APPROVED'/'CORRECTED' vs. 'AUTO') so a query walking the graph never needs a
special case for which path a given observation came through.

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


# ==========================================================================
# Pass 4 (src/mollm_tier_gate.py) -- Tier 1/2/3 direct write, unreviewed
# ==========================================================================

def _ingest_auto_tx(tx, params: dict):
    # Identical shape to _ingest_tx() above, on purpose -- a query walking
    # :MoLLMDecision -[:REVIEWED_BY]-> :HITLReview should not need a special
    # case for "this one was never reviewed by a human"; it should see
    # review.final_decision_status = 'AUTO' and know what that means. The
    # only structural difference is corrected_concept_id/review_duration are
    # always NULL here -- an AUTO decision was never corrected by anyone.
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
            decision.tier = $tier,
            decision.composite_confidence = $composite_confidence
        MERGE (obs)-[:VALIDATED_BY]->(decision)

        MERGE (review:HITLReview {hitl_case_id: $hitl_case_id})
        SET review.queue_reason = $queue_reason,
            review.final_decision_status = 'AUTO',
            review.corrected_concept_id = NULL,
            review.review_duration = NULL
        MERGE (decision)-[:REVIEWED_BY]->(review)
        """,
        **params,
    )


def ingest_auto_decision(driver, decision: dict, entity_fields: dict,
                         dry_run: bool = True) -> dict:
    """Writes one Tier 1/2/3 AUTO_VALIDATED/AUTO_RESOLVED decision
    (src/mollm_tier_gate.py's route_tier()) as a single atomic Cypher
    transaction -- the direct, unreviewed KG3 write-back path Pass 4 needs
    and that does not exist anywhere else in this codebase: this module's own
    docstring above records that NOTHING writes to KG3 today without first
    passing through src/hitl_queue.py's human review workflow.

    `dry_run=True` (the default): does not touch Memgraph at all. Returns the
    params dict that WOULD have been written, for logging/review. Per the
    plan's own risk note, direct KG3 write-back is new, higher-stakes code
    and should run feature-flagged/log-only against a held-out slice before
    any real write happens -- callers must pass dry_run=False explicitly once
    that validation has actually been done.

    `decision` is one src/mollm_tier_gate.route_tier() return dict, plus
    mollm_call_id/entity_id/note_id merged in by the caller (route_tier()
    itself does not know those IDs). `entity_fields` is the caller's own
    DuckDB lookup of extracted_entities.{orig_start,orig_end,confidence} and
    normalized_entities.candidates for this entity_id -- same separation of
    concerns as ingest_reviewed_case(): this module's only external
    dependency is Memgraph, not also DuckDB.

    Raises UningestibleCase (not silently) for anything that reaches this
    function without a resolvable concept -- a Tier 4/5 decision, a missing
    final_candidate_index, or an out-of-range one.
    """
    # 2026-08-17 fix: this used to be its own hardcoded ("TIER_1_AUTO_VALIDATED",
    # "TIER_2_AUTO_RESOLVED", "TIER_3_AUTO_VALIDATED") tuple, a second,
    # independent copy of route_tier()'s AUTO_TIERS set -- silently missed
    # TIER_1B_CALIBRATED_AUTO_VALIDATED when Phase 6 added it, so every
    # calibrator-promoted decision was rejected here as UningestibleCase even
    # though it's a genuine AUTO tier (caught live: 6/6 TIER_1B decisions on
    # the first fresh-note validation run were blocked this way). Imported
    # from src.mollm_tier_gate now so the two can never drift apart again.
    from src.mollm_tier_gate import AUTO_TIERS
    tier = decision.get("tier")
    if tier not in AUTO_TIERS:
        raise UningestibleCase(
            f"{decision.get('mollm_call_id', '?')}: tier={tier!r} is not an "
            f"auto-write tier -- only route_tier()'s AUTO_TIERS ever reach "
            f"this function; Tier 4/5 decisions belong in "
            f"src/hitl_queue.py's review workflow instead")

    final_candidate_index = decision.get("final_candidate_index")
    candidates = entity_fields.get("candidates") or []
    idx = (final_candidate_index - 1) if final_candidate_index else None
    if idx is None or idx < 0 or idx >= len(candidates):
        raise UningestibleCase(
            f"{decision.get('mollm_call_id', '?')}: final_candidate_index="
            f"{final_candidate_index!r} does not resolve to a real candidate "
            f"in this entity's {len(candidates)}-candidate list")
    chosen = candidates[idx]
    omop_concept_id = chosen.get("omop_concept_id")
    if omop_concept_id is None:
        raise UningestibleCase(
            f"{decision.get('mollm_call_id', '?')}: chosen candidate "
            f"{chosen.get('concept_name')!r} has no omop_concept_id")

    params = {
        "omop_concept_id": omop_concept_id,
        "vocabulary_id": chosen.get("vocabulary_id"),
        "domain_id": chosen.get("domain_id"),
        "concept_name": chosen.get("concept_name"),
        "entity_id": decision.get("entity_id"),
        "note_id": decision.get("note_id"),
        "raw_text": entity_fields.get("original_text"),
        "label": entity_fields.get("entity_label"),
        "orig_start": entity_fields.get("orig_start"),
        "orig_end": entity_fields.get("orig_end"),
        "confidence": entity_fields.get("confidence"),
        "source_call_id": decision.get("mollm_call_id"),
        "source_table": "mollm_tier_gate_decisions",
        "routing_decision": decision.get("mollm_routing_decision"),
        "tier": tier,
        "composite_confidence": decision.get("composite_confidence"),
        "hitl_case_id": f"auto_{decision.get('mollm_call_id')}",
        "queue_reason": decision.get("queue_reason"),
    }
    if dry_run:
        return {"dry_run": True, "params": params}
    with driver.session() as session:
        session.execute_write(_ingest_auto_tx, params)
    return {"dry_run": False, "params": params}
