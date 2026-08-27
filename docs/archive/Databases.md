# Database Architecture & Schema Specification
## Clinical Neuro-Symbolic Pipeline for Clinical Entity Linking

This document outlines the hybrid relational-vector-graph database architecture used in the Clinical Neuro-Symbolic Pipeline. Storage and query responsibilities are split across **DuckDB**, **Neo4j**, and **Memgraph** to optimize for high-volume lexical/vector queries, static reference ontologies, and dynamic patient-centric observational graphs respectively.

---

### 1. Architectural Overview & Split of Responsibilities

The system divides storage responsibilities across three key technologies:
*   **DuckDB**: Handles high-volume lexical lookups, vector similarity search, and Stage 1 text expansion/preprocessing logs.
*   **Neo4j**: Dedicated to housing the static SNOMED CT reference lexicon and its parent-child hierarchies.
*   **Memgraph**: Manages dynamic clinical observation instances, clinical rules, Mixture of LLMs (MoLLM) validation audit ledgers, and Human-in-the-Loop (HITL) review queues.

---

### 2. DuckDB (`db/kg2_lexical_store.duckdb`)

*   **Role**: Lexical & OMOP Vocabulary Store, Vector Search Engine, Stage 1 Provenance Store.
*   **Access Mode**: Embedded file database accessed via the Python `duckdb` driver.
*   **Pipeline Stages Involved**:
    *   **Stage 1 (Preprocessing)**: Writes JSON provenance records mapping abbreviation expansions (e.g., "SOB" → "shortness of breath") keyed by `note_id`. These high-volume expansion maps are rarely re-queried, making DuckDB a more appropriate fit than the graph database.
    *   **Stage 2b (Grounding & Normalization)**: Resolves GLiNER-extracted text spans to OMOP concept IDs using a tiered query structure and executes 768-dimensional SapBERT cosine similarity search natively in SQL.
*   **Vector Search & Optimization**:
    *   DuckDB executes vector similarity as a native SQL full-scan search.
    *   *Open Item*: Incorporating a Vector Search Extension (VSS) or HNSW index is recognized as a necessary upgrade before moving to production volumes.
    *   *Domain Filtering*: To improve grounding accuracy and reduce cross-domain mismatches, vector search fallbacks are domain-filtered by mapping GLiNER labels to specific OMOP domain IDs (e.g., Medication → Drug, Anatomy → Spec Anatomic Site).

#### Table Schemas
The DuckDB instance ingests tables from the Athena/OHDSI OMOP vocabulary download:

1.  **`athena_concept`**: Contains standard OMOP concept definitions imported from Athena CSV releases, augmented with a SapBERT vector column.
    ```sql
    CREATE TABLE athena_concept (
        concept_id          INTEGER PRIMARY KEY,
        concept_name        VARCHAR NOT NULL,
        domain_id           VARCHAR NOT NULL,
        vocabulary_id       VARCHAR NOT NULL,
        concept_class_id    VARCHAR,
        standard_concept    VARCHAR(1), -- 'S' for Standard
        concept_code        VARCHAR,
        valid_start_date    DATE,
        valid_end_date      DATE,
        invalid_reason      VARCHAR,
        embedding           FLOAT[]     -- 768-dim SapBERT CLS vector
    );
    ```

2.  **`athena_concept_synonym`**: Maps exact lexical synonyms to core concept IDs.
    ```sql
    CREATE TABLE athena_concept_synonym (
        concept_id           INTEGER REFERENCES athena_concept(concept_id),
        concept_synonym_name VARCHAR NOT NULL,
        language_concept_id  INTEGER
    );
    ```

3.  **`athena_concept_relationship`**: Defines mapping and subsumption relationships between concepts.
    ```sql
    CREATE TABLE athena_concept_relationship (
        concept_id_1     INTEGER,
        concept_id_2     INTEGER,
        relationship_id  VARCHAR NOT NULL,
        valid_start_date DATE,
        valid_end_date   DATE,
        invalid_reason   VARCHAR
    );
    ```

4.  **`athena_concept_ancestor`**: Captures hierarchical lineage over OMOP concepts.
    ```sql
    CREATE TABLE athena_concept_ancestor (
        ancestor_concept_id   INTEGER,
        descendant_concept_id INTEGER,
        min_levels_of_separation INTEGER,
        max_levels_of_separation INTEGER
    );
    ```

5.  **`note_expansions`**: Stores Stage 1 preprocessing provenance records as JSON string payloads.
    ```sql
    CREATE TABLE note_expansions (
        note_id     VARCHAR PRIMARY KEY,
        provenance  JSON -- Structured expansion metadata and offset tracking
    );
    ```

---

### 3. Graph Databases (Neo4j & Memgraph)

The graph-native components support reference terminology representation and record end-to-end clinical provenance.

*   **Access Protocol**: Accessed over the Bolt protocol via the `neo4j` Python driver (exploiting Memgraph's Bolt compatibility).
*   **Unified Graph Design**: Both the static SNOMED CT subsumption hierarchy (parent-child `IS_A` relations) and the curated guideline triplets (derived from KDIGO, AHA/ACC, GOLD, SSC, NPSG/ESI) are represented as two distinct relationship types over shared `:Concept` nodes. This allows a relation extracted at one granularity to traverse the hierarchy and match clinical evidence stated at another granularity.

#### Graph Schemas & Nodes

##### Reference Ontology: `:Concept`
*   **Role**: Shared nodes representing verified SNOMED CT and OMOP vocabulary concepts.
*   **Edges**:
    *   `-[:IS_A]->`: Parent-child subsumption hierarchy.
    *   `-[:GUIDELINE_RELATION]->` (or domain-specific guideline edges): Connects concepts using curated clinical triplets.
*   *Open Item*: Storing non-SNOMED concepts (e.g., RxNorm medications) as graph-native `:Concept` nodes is a work in progress; currently, medications are stored as flat properties on observation nodes without active graph edges.

##### Patient Instances: `:PatientObservation`
*   **Role**: Stores patient-specific clinical facts extracted from notes.
*   **Properties**: `entity_id` (unique identifier `<note_id>-e<hash>`), `note_id`, `raw_text`, `label` (e.g., `Medication`, `Condition`), `orig_start`, `orig_end`, `confidence`, `matched`, `omop_concept_id`, `vocabulary_id`, and `timestamp`.
*   **Edges**:
    *   `-[:INSTANCE_OF]->`: Connects the observation to the corresponding grounding `:Concept` node.
*   **Structural Separation**: Keeping patient observation instances structurally isolated from the reference ontology prevents patient data from mutating the shared, static reference graph.

##### Provenance Ledger: `:MoLLMDecision` and `:HITLReview`
To trace any fact in the graph back to its extraction and validation source, Stages 2b through 4 write a sequential chain of node paths:
1.  **`:PatientObservation`** (Stage 2)
2.  `-[:VALIDATED_BY]->` **`:MoLLMDecision`** (Stage 3): Captures `mollm_call_id`, `routing_decision`, `composite_confidence`, `confidence_tier_in`, retrieved context, ensemble agreement, reasoning, verdicts, and log-probability confidence scores.
3.  `-[:REVIEWED_BY]->` **`:HITLReview`** (Stage 4): Captures the Human-in-the-Loop review case ID, queue reason, suggested resolution presented to the auditor, final decision status (`PENDING`, `APPROVED`, `CORRECTED`, `REJECTED`), corrections made, and review duration.

*Benefit*: This design allows a re-audit query to traverse the entire provenance chain for any given triple in a single graph traversal rather than executing costly joins across separate systems.

**Implementation note (2026-08-15, `docs/2026-08-15_Stage4_Stage5_Build.md`)**: as built, the *pre-review* HITL queue state lives in a DuckDB table (`hitl_review_queue`, `src/hitl_queue.py`), not the graph — this deliberately narrows "Stages 2b through 4 are stored natively in the graph database" above. Only a case that has actually been reviewed (`reviewer_decision` in `APPROVED`/`CORRECTED`) gets written to Memgraph as the `:PatientObservation`→`:MoLLMDecision`→`:HITLReview` chain (`src/kg3_ingestion.py`); a `REJECTED` case is never written to the graph at all. Reasoning: the queue itself churns constantly (every Stage 3 decision is queued regardless of tier, per the pseudo-labeling risk noted in `docs/Implementation_Checklist.md`'s Stage 4 section) and reuses this codebase's already-proven DuckDB read/write conventions for that high-churn, pre-commit state; Memgraph is reserved for what it's actually good at and what this design doc already scopes it for — a durable, queryable provenance ledger of *finished* decisions.

---

### 4. Stage-to-Database Operation Matrix

| Pipeline Stage | Database Used | Operation Type | Data Artifact / Query Description |
| :--- | :--- | :--- | :--- |
| **Stage 1 (Preprocessing)** | DuckDB | `INSERT` | Appends JSON provenance map to `note_expansions` tracking offset shifts. |
| **Stage 2b (Normalization)** | DuckDB | `SELECT` | Tier 1 & 2: Exact string lookup against `athena_concept` & `athena_concept_synonym`. |
| **Stage 2b (Normalization)** | DuckDB | `SELECT` | Tier 3: Vector search using `list_cosine_similarity(embedding, ?::FLOAT[])`. |
| **Stage 3 (MoLLM Gate)** | Memgraph | `MATCH` | Retrieves guideline triplets and neighboring context for model prompt construction. |
| **Stage 3 (MoLLM Gate)** | Neo4j | `MATCH` | Fetches `SnomedConcept` FSNs and IS_A parent/child hierarchy for granularity checks. |
| **Stage 4 (Ingestion)** | Memgraph | `CREATE / MATCH` | Executes atomic Cypher transaction writing `:PatientObservation`, `:MoLLMDecision`, and optional `:HITLReview`. |
| **Stage 5 (Active Learning)**| Memgraph / DuckDB | `MATCH / SELECT` | Re-audits accepted triples and feeds human corrections back to model prompts. |

---

### 5. Known Configuration & Engineering Open Items

1.  **Memgraph Persistence**: Real-world HITL-verified data requires snapshotting and Write-Ahead Logging (WAL) to be fully configured and tested. Unlike the reference vocabularies, dynamic patient observations and manual corrections cannot be regenerated from static source files if lost.
2.  **Vocabulary Scope Alignment**: DuckDB's OMOP vocabulary scope currently remains multi-vocabulary (RxNorm, ICD10CM, etc.). A decision is pending on whether to keep it broad or narrow it down to match the graph's SNOMED-only scope.
3.  **Abbreviation CSV Verification**: The static abbreviation table loader assumes the first two columns of the GitHub alphabetically split CSVs are positionally structured as (abbreviation, expansion) without independent schema verification.
4.  **Vector Indexing**: Implementing HNSW or vector indexes in DuckDB remains an open task to replace SQL-native full scans for similarity search at scale.