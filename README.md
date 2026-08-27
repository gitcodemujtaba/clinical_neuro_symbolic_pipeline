# Clinical Neuro-Symbolic Pipeline for Entity Extraction & Verification

This repository contains the code for a Master's thesis project (COM748,
"Towards Trustworthy Clinical Named Entity Recognition: Knowledge
Graph-Enhanced Local LLMs with Active Learning"): a neuro-symbolic
clinical NLP pipeline that extracts clinical entities from free-text
discharge notes, grounds them to SNOMED CT via a tiered retrieval
cascade, verifies each grounding through a 3-model local-LLM ensemble
with a trained statistical calibrator, and routes every decision through
human review before any write to a knowledge graph — entirely on
self-hosted infrastructure, with zero third-party API calls on patient
text.

> **This README describes the system as actually built and measured.**
> An earlier version of this document (and of several files in `docs/`)
> described the original proposal architecture — Clinical-T5 for
> relation extraction, a BioMistral/OpenBioLLM vLLM ensemble, Neo4j for
> the SNOMED reference graph. None of that is what runs today; see
> `docs/Implementation_Methodology.md` for the full architecture history
> and why each substitution was made.

---

## Current headline result

**76.8% precision on autonomously-written decisions** (43/56 gradable),
**31.2% deflection rate** (entities requiring no human review),
**33.7% linked concept-level F1** — all measured on 10 notes from the
official locked test split, held out from every fix and every model
training step this project made. See
`docs/FINAL_RESULTS_Single_Source_Of_Truth.md` for the full results
table (including the honest comparison against a higher, but
development-set-inflated, corpus-wide figure) and
`docs/Implementation_Decisions_Log.md` for every implementation decision
behind these numbers, each with its measured evidence.

---

## System architecture

The pipeline processes clinical text through 5 stages:

1. **Preprocessing** (`src/preprocessing.py`) — scispaCy tokenization,
   word-boundary abbreviation expansion (multi-meaning aware, with a
   4-rung tiebreak ladder that declines to guess rather than pick
   arbitrarily), section segmentation, and rule-based clinical assertion
   detection (medspacy/ConText — negation, family history, hypothetical,
   allergy).
2. **Extraction** (`src/entity_extraction.py`, `src/extraction.py`) —
   **GLiNER-BioMed** for zero-shot clinical NER (with sliding-window
   chunking for notes exceeding its token ceiling) and **GLiNER-relex**
   for relation extraction, linked by character-offset overlap.
   Extraction-side cold-start injectors recover common lab abbreviations
   and narrative state words GLiNER never proposes at any confidence.
3. **Normalization** (`src/normalization/`) — a tiered retrieval cascade
   (exact name → exact synonym → SapBERT dense semantic similarity, plus
   curated brand/lab-alias lookups and a fuzzy-match fallback) against
   the Athena OMOP vocabulary in DuckDB, with SNOMED regional-extension
   namespace filtering and class-based near-duplicate preference rules.
4. **MoLLM Tier Gate** (`src/mollm_tier_gate.py`) — three local LLMs
   (**qwen2.5:3b, llama3.2:3b, phi4-mini**, served via Ollama) each run
   an isolated two-step chain-of-thought (define the clinical meaning
   first, with no candidate visible; then judge each candidate
   independently). A trained logistic-regression calibrator
   (`ConsensusCalibrator`, 16 features) grades non-unanimous votes that
   would otherwise be discarded to review. Hard, evidence-based safety
   traps guard known-fragile patterns regardless of vote confidence.
5. **HITL Routing & KG3 Ingestion** (`src/hitl_queue.py`,
   `src/kg3_ingestion.py`) — every decision, regardless of tier, is
   queued for human review (a deliberate, current policy — see the
   decisions log). Approved/corrected decisions write to Memgraph as a
   `:PatientObservation → :MoLLMDecision → :Concept` provenance chain.
   **All writes remain `dry_run=True`** — nothing reaches the live graph
   unreviewed.

A separate, evaluated-but-not-deployed component: a real TransE
knowledge-graph embedding (`src/kg_embedding.py`) trained on the SNOMED
reference graph, tested as a candidate-disambiguation signal and found
less safe than the existing hand-built rule for the pattern it was
compared against — kept as a documented, honest negative result, not
silently dropped. Full detail in
`docs/Entity_Journey_Plain_Language_Walkthrough.md` §8.

---

## Databases & knowledge graphs

* **Lexical & OMOP store (DuckDB)** — `db/kg2_lexical_store.duckdb`
  (~31GB, gitignored). Hosts the full Athena OMOP/SNOMED vocabulary
  (~1.09M concepts), SapBERT embeddings for semantic retrieval, and
  every table this pipeline writes (`extracted_entities`,
  `normalized_entities`, `mollm_tier_gate_decisions`,
  `hitl_review_queue`, etc.). **This project's own generated data**
  (not the imported reference vocabulary) is exported to
  `exports/*.parquet` for portability — see
  `scripts/export_pipeline_tables.py`.
* **Rules-based clinical guideline graph** — a file-backed index
  (`GuidelineIndex` in `src/retrieval.py`) over 76 source documents
  (~1,700 nodes, 1,162 extracted rules) built from `data/local_triplets_db2_v6_*`.
  Matched by concept name + type (not SNOMED code — see
  `docs/MoLLM_Prompts_And_Reasoning_Technical_Reference.md` §8.2 for
  why). Bridged into the MoLLM tiebreak prompt, currently gated behind
  `CNSP_GUIDELINE_EVIDENCE` (off by default, real coverage ~0.9% of
  candidate concepts, pending its own validation batch).
* **Patient-instance provenance graph (Memgraph, port 7688)** — the
  dynamic graph this pipeline's own decisions write to (dry-run only
  today). Kept structurally separate from the reference vocabulary, so
  patient data never mixes with the static SNOMED/guideline knowledge.

There is **no Neo4j deployment** in the current system — the original
proposal's Neo4j-hosted SNOMED reference lexicon was superseded by
querying the Athena OMOP vocabulary directly in DuckDB, which turned out
to be both simpler and fast enough for this project's actual query
patterns.

---

## Project structure

* `src/` — core runtime pipeline: `preprocessing.py` (Stage 1),
  `entity_extraction.py`/`extraction.py` (Stage 2a),
  `normalization/` (Stage 2b, split into a package),
  `mollm_tier_gate.py` (Stage 3, the current production gate —
  `mollm_ensemble.py` is the superseded predecessor, kept for reference/
  comparison), `mollm_tier_calibrator.py` (`ConsensusCalibrator`),
  `hitl_queue.py`/`kg3_ingestion.py`/`kg3_query.py` (Stage 4/5),
  `kg_embedding.py`/`kg_embedding_tiebreak.py` (the TransE work),
  `guideline_evidence.py` (the rules-KG bridge).
* `scripts/` — batch runners (`run_stage3_tier_gate.py`), one-off
  re-normalization/backfill scripts, data export
  (`export_pipeline_tables.py`), the KGE trainer
  (`build_kg_embeddings.py`).
* `ui/` — 4-page Streamlit dashboard: Pipeline Runner, HITL Review Queue
  (full-note side-by-side view + per-model reasoning trail), Troubleshooting
  (step-by-step gold-vs-prediction diff), Evaluation Metrics (live P/R/F1/
  IoU/deflection, including a consolidated "Overall" tab).
* `evaluation/` — grading and calibration scripts: gold-recall scoring,
  the DrivenData benchmark char-IoU metric, tier-gate grading, the KGE
  tiebreak validation harness, the exhaustive-candidate-eval impact
  assessment.
* `docs/` — see the index below.
* `data/` & `db/` — raw vocabularies and DuckDB stores. **MIMIC-IV and
  the SNOMED-CT entity-linking challenge note text are PhysioNet-restricted
  and are gitignored** (never redistributed via this public repo) — see
  `.gitignore` and `docs/Implementation_Decisions_Log.md` for the
  compliance note.

---

## Getting the data (not included in this repo)

Two of this project's data dependencies are gitignored on purpose — see
the note above — and need to be obtained independently by anyone
reproducing this work. **This project does not redistribute either.**

### Clinical notes & gold annotations (PhysioNet, credentialed access required)

* **MIMIC-IV-Note** (the raw discharge-summary corpus) —
  https://physionet.org/content/mimic-iv-note/ . Requires a free
  PhysioNet account, completing the required CITI human-subjects-research
  training, and signing the dataset's specific data use agreement before
  download access is granted.
* **SNOMED CT Entity Linking Challenge gold annotations** (the 272-note,
  75,491-annotation-row evaluation set this project's gold-recall/precision
  numbers are measured against) — start from the DrivenData benchmark
  page, https://www.drivendata.org/benchmarks/310/benchmark-snomed-ct/page/983/
  (this is also the source of the official char-IoU metric definition
  this project implements in `evaluation/iou_metrics.py`) — it links
  through to the underlying PhysioNet-hosted dataset, which is also
  credentialed-access, same PhysioNet account as above.

Once obtained, place them at `data/raw_notes/` (discharge note CSVs) and
`data/snomed-ct-entity-linking-challenge-1.2.0/` respectively — the exact
paths `.gitignore` already excludes and the `RAW_TEXT_CANDIDATES`-style
fallback lists in `scripts/` already expect.

### Athena/OHDSI OMOP vocabulary (free registration, not PhysioNet-restricted)

The SNOMED CT/RxNorm/OMOP vocabulary tables this pipeline's retrieval
stages query (`athena_concept`, `athena_concept_synonym`,
`athena_concept_relationship`, `athena_concept_ancestor`) come from the
OHDSI Athena vocabulary browser: https://athena.ohdsi.org/ . Free
registration (no PhysioNet credentialing needed), then select the
vocabularies you need (this project uses SNOMED and RxNorm) and download.
Load the resulting CSVs into `db/kg2_lexical_store.duckdb` — see
`scripts/` for the loading scripts this project used, and
`docs/Implementation_Decisions_Log.md` §7 for why SNOMED regional
extensions specifically need the namespace-pattern filter documented
there if you're building retrieval logic against this data yourself.

---

## Documentation index

Start here, depending on what you need:

| Document | What it's for |
|---|---|
| `docs/FINAL_RESULTS_Single_Source_Of_Truth.md` | **Start here for numbers.** Every measured result, one place, with sources. |
| `docs/Implementation_Methodology.md` | Full architecture narrative — every stage, every design choice, why. |
| `docs/Implementation_Decisions_Log.md` | 135 "why did we choose X" decisions across the whole project, each with its measured evidence. |
| `docs/Code_Flow.md` | Execution-order trace — what calls what, with real function names. |
| `docs/Code_Reference_Stages_And_Metrics.md` | Core code snippets per stage + every metric's exact formula. |
| `docs/Entity_Journey_Plain_Language_Walkthrough.md` | One real entity traced through every stage, written for a non-technical reader. |
| `docs/MoLLM_Prompts_And_Reasoning_Technical_Reference.md` | Every MoLLM prompt verbatim, how provenance selects prompt rules, the rules-KG bridge. |
| `docs/ConsensusCalibrator_Technical_Reference.md` | The 16-feature calibrator: exact features, training, thresholds, safety traps. |
| `docs/Provenance_Fields_Technical_Reference.md` | Every provenance field the pipeline writes, where computed, every downstream consumer. |
| `docs/Evaluation_Criteria.md` | The proposal's own evaluation-criteria spec, verbatim — what a claim needs to satisfy to count. |
| `docs/2026-08-20_Session_Results_And_Status.md` | Chronological session log — the narrative behind the numbers above. |

`docs/archive/` holds superseded design docs and dated investigation
logs (2026-08-13 through 2026-08-19) — every real finding in them has
been mined into `Implementation_Decisions_Log.md` above, kept via
`git mv` (full history, nothing deleted) rather than removed. See
`docs/archive/README.md` for what superseded what.

---

## Setup & initialization

### Environment variables

The pipeline itself needs very little configuration — most behavior is
controlled by feature flags with safe, validated defaults (see
`docs/Implementation_Decisions_Log.md` for why each stays off by
default where it does).

```env
# Local LLM serving (Ollama, self-hosted -- qwen2.5:3b, llama3.2:3b, phi4-mini)
OLLAMA_HOST=http://localhost:11434

# Patient-instance provenance graph (dry-run only in this deployment)
MEMGRAPH_URI=bolt://localhost:7688
MEMGRAPH_USER=
MEMGRAPH_PASSWORD=

# Optional: point at a non-default DuckDB file (tests, throwaway DBs)
CNSP_DB_PATH=db/kg2_lexical_store.duckdb

# Optional feature flags -- all default OFF or to a validated setting,
# see docs/Implementation_Decisions_Log.md for the measured reasoning
# behind each default:
# CNSP_HYBRID_RETRIEVAL       (BM25+dense retrieval -- validated OFF, dense-only wins)
# CNSP_ACRONYM_ESCALATION     (MoLLM-resolved ambiguous abbreviations -- validated OFF, 34-36% precision)
# CNSP_GUIDELINE_EVIDENCE     (rules-KG evidence in the tiebreak prompt -- OFF pending validation)
# CNSP_CONTEXTUAL_CANDIDATES  (exhaustive candidate evaluation -- ON by default)
```

### Running the pipeline

```bash
# Full test suite
pytest tests/

# Run Stage 1-2b on a note interactively
streamlit run ui/app.py   # then use the Pipeline Runner page

# Batch Stage 3 tier-gating over already-normalized notes
python3 scripts/run_stage3_tier_gate.py --note-ids <comma-separated-ids>
```
