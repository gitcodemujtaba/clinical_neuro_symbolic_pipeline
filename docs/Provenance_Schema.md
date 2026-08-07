# docs/Provenance_Schema.md

## Design Philosophy
* Provenance accumulates across the pipeline by appending fields to a record rather than replacing prior data[cite: 7]. 
* Any fact in KG 3 can be traced back to the exact evidence and reasoning that produced it[cite: 7].

## Schema by Stage
* **Stage 1 (Preprocessing):** Captures `note_id`, `abbreviation_dict_version`, and a list of expansions[cite: 7]. Each expansion includes the `abbrev`, `expansion`, original and expanded character offsets, and an `ambiguous` flag[cite: 7].
* **Stage 2a (Extraction):** 
  * Entity records append: `entity_id`, `text`, `label`, expanded-text offsets, offsets mapped back to the original note, `gliner_confidence`, `gliner_model_version`, and the `extraction_threshold`[cite: 7].
  * Relation records append: `relation_id`, `head_entity_id`, `tail_entity_id`, `relation_label`, and `relation_confidence`[cite: 7].
* **Stage 2b (Normalization):** Appends `matched`, `omop_concept_id`, `concept_name`, `vocabulary_id`, `domain_id_queried`, `match_method`, `similarity_score`, `sapbert_pooling_method`, and `athena_vocabulary_release` to the entity record[cite: 7].
* **Stage 3 (MoLLM Gate):** Captures the decision artifact including `mollm_call_id`, `confidence_tier_in`, and the `retrieved_context`[cite: 7]. Per-model data includes `reasoning`, `verdict`, `cited_evidence`, `citation_verified`, `raw_confidence_label`, and `logprob_confidence`[cite: 7]. System ensemble stats include `ensemble_agreement`, `composite_confidence`, and `mollm_routing_decision`[cite: 7].
* **Stage 4 (HITL Routing):** Captures `hitl_case_id`, `queue_reason`, `presented_suggestion`, `reviewer_decision`, `corrected_concept_id`, `rejection_reason`, `review_duration`, and sets `final_ingestion_path` to `HUMAN_VERIFIED`[cite: 7].

## Storage Mapping
* Stage 1's high-volume expansion maps are stored in DuckDB keyed by `note_id`[cite: 7].
* Stages 2b through 4 are stored natively in the graph database, connecting `:PatientObservation` to `:MoLLMDecision` to `:HITLReview`[cite: 7].