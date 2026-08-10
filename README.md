# Clinical Neuro-Symbolic Pipeline for Entity Extraction & Verification

This repository contains the code for a Master's thesis project: an autonomous, neuro-symbolic clinical NLP pipeline. It utilizes zero-shot entity extraction, sequence-to-sequence relation extraction, vector-based normalization, and a multi-agent MoLLM (Medical Large Language Model) ensemble to extract grounded medical concepts and relations from unstructured clinical notes while strictly preserving character offsets and enforcing clinical guideline citations.

## System Architecture

The pipeline processes clinical text through 5 distinct stages:

1. **Preprocessing (scispaCy):** Tokenization and abbreviation expansion utilizing strict word-boundary dictionary lookups, generating a 1-to-1 mathematically verifiable character offset map.
2. **Extraction & Relation Extraction (GLiNER + Clinical-T5):** 
   * **GLiNER:** Zero-shot Named Entity Recognition (NER) executed on expanded text, mapping coordinate bounds back to raw source notes.
   * **Clinical-T5:** Sequence-to-sequence Relation Extraction (ReLEx) fine-tuned to extract structured clinical relation triples (e.g., `[Medication] -[:TREATED_WITH]-> [Condition]`).
3. **Normalization (DuckDB + SapBERT):** Cascading normalization (Exact Name -> Exact Synonym -> SapBERT CLS-token cosine similarity) against the Athena OMOP vocabulary (`kg2_lexical_store.duckdb`).
4. **MoLLM Consensus Gate (BioMistral + OpenBioLLM):** Local LLMs evaluate extractions against SNOMED hierarchies and clinical guideline triplets. Includes automated hallucination detection to verify cited evidence.
5. **Graph Ingestion & HITL:** Atomic Cypher transactions log the full provenance subgraph (`PatientObservation` -> `MoLLMDecision` -> `Concept`) into Memgraph. Low-confidence or contradictory extractions are routed to a Human-in-the-Loop (HITL) Streamlit queue.

## System Databases & Graph Infrastructure

* **Lexical & OMOP Store (DuckDB):** Stored locally at `db/kg2_lexical_store.duckdb`. Hosts Athena OMOP vocabularies and 768-dimensional SapBERT embeddings for concept normalization.
* **Reference Lexicon (Neo4j - Port 7687):** Holds static `SnomedConcept` nodes and SNOMED CT `fullySpecifiedName` definitions.
* **Rules & Provenance Graph (Memgraph - Port 7688):** Stores patient observations, clinical relations, rules, and the `MoLLMDecision` audit ledger.

## Project Structure

* `src/`: Core runtime pipeline, text processing (`preprocessing.py`), GLiNER/Clinical-T5 wrappers (`extraction.py`), SapBERT normalization (`normalization.py`), and MoLLM ensemble logic (`mollm_ensemble.py`).
* `scripts/`: Offline data ingestion, database indexing, database profiling (`profile_databases.py`), and infrastructure boot scripts (`boot_infra.sh`).
* `ui/`: Multi-page Streamlit dashboard for live execution, pipeline observability, and the HITL active learning queue.
* `evaluation/`: Scripts for rigorous statistical validation (Note-level Bootstrapping, Confidence Intervals, Expected Calibration Error).
* `data/` & `db/`: Storage for raw vocabularies, MIMIC-IV text, and DuckDB stores (ignored in git for privacy/size).

## Setup & Initialization

### 1. Environment Variables
Create a `.env` file in the root directory with the following variables:
```env
# Database 1: Neo4j (SNOMED CT Lexicon)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secure_password_here

# Database 2: Memgraph (Clinical Rules & Patient Provenance Subgraph)
MEMGRAPH_URI=bolt://localhost:7688
MEMGRAPH_USER=
MEMGRAPH_PASSWORD=

# Model Endpoints & Paths
BIOMISTRAL_BASE_URL=http://localhost:8000/v1
OPENBIOLLM_URL=http://localhost:8001/v1
CLINICAL_T5_MODEL_NAME=luisaaguiar/Clinical-T5-Base
ABBREV_DIR=/home/ec2-user/clinical_neuro_symbolic_pipeline/data/medical_abbreviations
'''