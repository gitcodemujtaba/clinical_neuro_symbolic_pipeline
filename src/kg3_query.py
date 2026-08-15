"""
src/kg3_query.py — read interface into KG3 (Memgraph), for Stage 5.

WHY THIS EXISTS NOW, WHEN THE ACTIVE-LEARNING FEEDBACK LOOP ITSELF DOES NOT.
docs/Implementation_Checklist.md scopes Stage 5 as depending on Stage 3 +
Stage 4; the real feedback mechanisms (GLiNER prompt/search-space feedback,
CompGCN/TransE/RotatE re-ranking) need a meaningful volume of accumulated
HUMAN_VERIFIED corrections to mean anything, which does not exist yet on
2026-08-14 (Step 4 has only just started writing to KG3). This module is the
READ side those mechanisms will eventually consume -- built and tested
against whatever Step 4 actually ingests, so it is ready the moment there is
real data, rather than being designed blind alongside the write side.

Deliberately read-only and free of any ranking/scoring logic of its own --
this module answers "what is in KG3", not "what should Stage 2b do about
it". That question is Stage 5's actual feedback mechanism, intentionally
not built yet (see docs/Implementation_Checklist.md's own scoping).
"""
from src.kg3_ingestion import get_memgraph_driver  # re-exported for callers
    # that only need one driver for both read and write


def count_by_label(driver) -> dict:
    """Node counts grouped by :PatientObservation.label -- the cheapest
    possible signal of what KG3 currently covers, useful as a first check
    that ingestion actually did something before running anything heavier.
    """
    with driver.session() as session:
        result = session.run("""
            MATCH (obs:PatientObservation)
            RETURN coalesce(obs.label, 'unknown') AS label, count(obs) AS n
            ORDER BY n DESC
        """)
        return {r["label"]: r["n"] for r in result}


def get_observations_for_concept(driver, omop_concept_id: int) -> list:
    """Every :PatientObservation instance-of a given concept, with its full
    provenance chain (decision + review), for auditing "what does KG3 say
    about this concept" or for a future re-ranker asking "how often has a
    human confirmed this concept for entities like this one".
    """
    with driver.session() as session:
        result = session.run("""
            MATCH (obs:PatientObservation)-[:INSTANCE_OF]->(c:Concept {omop_concept_id: $cid})
            OPTIONAL MATCH (obs)-[:VALIDATED_BY]->(d:MoLLMDecision)
            OPTIONAL MATCH (d)-[:REVIEWED_BY]->(r:HITLReview)
            RETURN obs.entity_id AS entity_id, obs.note_id AS note_id,
                   obs.raw_text AS raw_text, obs.label AS label,
                   d.routing_decision AS routing_decision,
                   r.final_decision_status AS final_decision_status,
                   r.hitl_case_id AS hitl_case_id
        """, cid=omop_concept_id)
        return [dict(r) for r in result]


def get_accepted_triples(driver, domain_id: str = None) -> list:
    """Every fully-provenanced observation (has a :HITLReview with
    APPROVED/CORRECTED status) currently in KG3, optionally restricted to
    one OMOP domain. This is the population Stage 5's eventual re-ranker/
    prompt-feedback mechanism will train or query against -- returned as
    plain dicts, not graph objects, so a caller never needs a live driver
    just to iterate the result.
    """
    domain_clause = "WHERE c.domain_id = $domain_id" if domain_id else ""
    with driver.session() as session:
        result = session.run(f"""
            MATCH (obs:PatientObservation)-[:INSTANCE_OF]->(c:Concept)
            MATCH (obs)-[:VALIDATED_BY]->(d:MoLLMDecision)-[:REVIEWED_BY]->(r:HITLReview)
            {domain_clause}
            WHERE r.final_decision_status IN ['APPROVED', 'CORRECTED']
            RETURN obs.entity_id AS entity_id, obs.note_id AS note_id,
                   obs.raw_text AS raw_text, obs.label AS label,
                   c.omop_concept_id AS omop_concept_id, c.concept_name AS concept_name,
                   c.domain_id AS domain_id, r.final_decision_status AS final_decision_status
        """, domain_id=domain_id)
        return [dict(r) for r in result]
