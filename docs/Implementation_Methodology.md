# docs/Implementation_Methodology.md

## Objective
* The central claim under test is that a neuro-symbolic pipeline combining zero-shot extraction, KG-grounded normalization, and LLM-based validation with a human-in-the-loop (HITL) active learning loop can match or exceed the performance of a pretrained generative baseline (Clinical-T5) on clinical entity/relation extraction from MIMIC-IV discharge notes[cite: 7].
* This approach aims to require substantially less human annotation effort over time as the system's knowledge base grows[cite: 7]. 
* Performance is evaluated on two distinct axes: concept-level accuracy against a gold standard, and effort reduction[cite: 7].

## Data Sources
* **MIMIC-IV-Note:** The primary input corpus consisting of structured discharge summaries[cite: 7].
* **imantsm/medical_abbreviations:** A static abbreviation to expansion dictionary split across 17 CSV files[cite: 7].
* **Athena/OHDSI OMOP:** Vocabulary tables ingested into DuckDB[cite: 7].
* **SNOMED CT US:** A parent-child (IS_A) subsumption hierarchy knowledge graph queried via Memgraph[cite: 7].
* **Guideline-derived triplets:** Triples curated from clinical guidelines (KDIGO, AHA/ACC, GOLD, SSC, NPSG/ESI) stored in the same graph as the SNOMED hierarchy[cite: 7].

## Pipeline Architecture
* **Stage 1 (Preprocessing):** scispaCy tokenizes the note, and known short forms are expanded using a word-boundary dictionary lookup before semantic processing occurs[cite: 7]. Character-offset mapping is tracked through this step so spans identified in the expanded text can be traced back to the original note[cite: 7].
* **Stage 2a (Extraction):** GLiNER-ReLEx performs zero-shot span and relation extraction directly on the expanded text without database or graph access[cite: 7].
* **Stage 2b (Grounding & Normalization):** SapBERT converts extracted spans into dense vectors, and DuckDB resolves them to an OMOP concept ID via a tiered lookup (exact concept name, exact synonym name, then vector similarity fallback)[cite: 7]. A global similarity threshold of 0.72 gates whether a vector match counts as resolved[cite: 7].
* **Stage 3 (MoLLM Confidence-Gated Validation):** An ensemble of MedGemma 4B and OpenBioLLM 8B routes extractions based on Stage 2 confidence[cite: 7]. High-confidence inputs undergo a guideline-contradiction check, while low-confidence inputs prompt a deeper resolution attempt using provenance, the SNOMED hierarchy, and guideline-triplet evidence[cite: 7].
* **Stage 4 (Routing & Ingestion):** Accepted triples are tagged with their provenance path as AUTO_VALIDATED, MOLLM_RESOLVED, or HUMAN_VERIFIED[cite: 7]. Patient-instance facts are stored as `:PatientObservation` nodes linked via `:INSTANCE_OF` to `:Concept` nodes, keeping patient data structurally distinct from the reference ontology[cite: 7].
* **Stage 5 (Active Learning Feedback Loop):** The dynamic, hospital-specific knowledge graph (KG 3) feeds back into GLiNER's prompt and search space over time[cite: 7]. Updates are checked against a held-out gold-evaluation set to catch regressions[cite: 7].

## Baseline Comparison
* Clinical-T5 is utilized as an independent parallel baseline run separately on the same notes[cite: 7].
* Clinical-T5's generated output passes through the same SapBERT and DuckDB grounding step as the main pipeline to ensure both are scored at the concept-ID level rather than on raw text overlap[cite: 7].
* Comparisons happen at three longitudinal checkpoints: KG 3 empty (T0), roughly half of MIMIC-IV processed (T1), and fully processed (T2)[cite: 7].