# Implementation Checklist
## Clinical Neuro-Symbolic Pipeline — Coding Project

Last updated: 2026-08-07. Cross-referenced to the 5 proposal objectives and `Proposal_Alignment_Review.md`. Status reflects the actual code/data in the repo as of this date, not the design docs alone.

Legend: `[x]` done · `[~]` partially done / working but has known bugs · `[ ]` not started

---

## Stage 1 — Preprocessing (`src/preprocessing.py`)
Supports: Objective 1 (baseline architecture)

- [x] scispaCy tokenization + word-boundary abbreviation expansion
- [x] Character-offset tracking (original ↔ expanded text)
- [x] Provenance write to DuckDB `note_expansions`
- [ ] Add `abbreviation_dict_version` field to provenance record (specified in `Provenance_Schema.md`, missing from current output)
- [ ] Add `ambiguous` flag per expansion (specified in `Provenance_Schema.md`, missing from current output)
- [ ] Set explicit tokenizer `max_length` downstream in Stage 2a to stop silent truncation on long notes (notes run 2,374–24,858 chars, mean ~10,257 — confirmed via `evaluaiton-dataset/`)

## Stage 2a — Extraction (`src/entity_extraction.py`, `src/extraction.py`)
Supports: Objective 1

- [x] GLiNER zero-shot span extraction — **swapped 2026-08-07** from `urchade/gliner_medium-v2.1` (general-domain) to `Ihor/gliner-biomed-large-v1.0` (GLiNER-BioMed, DS4DH/Univ. of Geneva, trained on biomedical NER benchmarks incl. clinical narratives). Closes the item below.
- [x] Offset reconciliation back to original note text ("time machine")
- [x] **Relation-extraction approach decided and implemented (2026-08-07, revised same day): option (a), GLiNER-relex** (`knowledgator/gliner-relex-large-v1.0`) via `src/extraction.py`'s `extract_and_store_relations()`, wired into `src/clinical_pipeline.py`. Zero-shot relation classification using a constrained clinical relation vocabulary (`treated with`, `indicates`, `causes`, `located in`, `measured by`), with `allowed_head`/`allowed_tail` plausibility enforced as a post-hoc filter (not a model-native constraint — see below). MedGemma 4B was considered (already committed to Stage 3's MoLLM ensemble) but deliberately not reused here — doing so would mean Stage 3 partially validates its own Stage 2a output, undermining the ensemble's independence.
  - **First attempt was GLiREL** (`jackboyla/glirel-large-v0`), correctly wired (including catching a real bug: its README's `allowed_head`/`allowed_tail` constrained-label dict format only works through its separate spaCy pipeline component, not the raw `predict_relations()` API — passing it directly silently scored every pair against one bogus class, producing exactly 0 relations across all 10 real test notes), plus token-based chunking for its hard 512-token cap. After fixing the label-format bug, still produced near-zero scores — root-caused to `transformers==4.57.6`/`torch==2.8.0` (installed) being far newer than what this single, unmaintained "v0" checkpoint (trained ~Nov 2024, no newer release since) was validated against; DeBERTa-v3 (its backbone) is known to silently drift numerically across `transformers` versions. Confirmed via `strict=True` weight loading (clean, rules out a state_dict mismatch) and reproducing GLiREL's own README example verbatim (documented score 0.9923, actual: 0.0028). Replaced rather than chasing a transformers/torch downgrade that risked breaking GLiNER-BioMed/SapBERT elsewhere.
  - Results land in the `extracted_relations` DuckDB table; `relation_id`/`head_entity_id`/`tail_entity_id` fields from `Provenance_Schema.md` are not yet populated (currently keyed by head/tail text+label, not a stable entity_id FK) — tracked as follow-up.
  - **Known trade-off**: GLiNER-relex does its own internal (general-domain, not clinically-tuned) entity detection to anchor relations — it can't accept `entity_extraction.py`'s GLiNER-BioMed entities directly. `extracted_relations`' head/tail text+label may occasionally diverge slightly from the canonical `extracted_entities` table as a result. Not yet reconciled; would need cross-model span-alignment logic.
- [x] ~~Swap `gliner_medium-v2.1` (general-domain) for a clinical/biomedical GLiNER checkpoint before generating any benchmark numbers~~ — done above.
- [ ] Refactor so extraction is a pure function and DB persistence is a separate call (currently `entity_extraction.py` reads/writes DuckDB inline, contradicting `Implementation_Methodology.md`'s "no database access" description of this stage — fix either the code or the doc)

## Stage 2b — Normalization (`src/normalization.py`)
Supports: Objective 1

- [x] Tier 1 exact concept-name match
- [x] Tier 2 exact synonym match
- [x] Tier 3 SapBERT cosine-similarity fallback
- [x] **Non-determinism fixed (2026-08-08, see `normalization.py` module docstring).** Every tier query (`normalize_entity()` main queries, `_tier_queries()`, `_lookup_tier12()`) now has an explicit `ORDER BY concept_id ASC` (Tier 1/2) or `ORDER BY similarity DESC, concept_id ASC` (Tier 3) tiebreak before `LIMIT`. This item stayed open in the checklist for a day after the code fix landed; empirically re-confirmed 2026-08-11 by running `test_stage3_live.py --store` twice back-to-back on note `10000032-DS-21` after clearing stale `is_test` rows — every entity returned identical `snomed_code` and identical retrieved rule IDs across both runs (see `docs/Stage3_Issue1_Rule_Backfill.md`).
- [ ] Add the documented 0.72 global similarity threshold gate to Tier 3 (currently always accepts top-1 regardless of score — confirmed producing garbage matches: `lasix`→`Laslades`, `spirnolactone`→`SPIRILENE`, `bioplar`→`Bourgvilain`)
- [ ] Add OMOP `domain_id` filtering to Tier 3 (GLiNER label → OMOP domain, per `Databases.md`) — currently missing entirely
- [ ] Extend domain filtering to Tier 2 as well — confirmed collision (`ED` → `Ed District`) happened at the exact-synonym tier, not just the vector fallback
- [ ] Verify `athena_concept_synonym` brand-name coverage for common drugs (e.g. `lasix` failing Tier 1/2 despite being a standard RxNorm concept suggests possible gaps)
- [ ] Canonicalize whitespace/case before entity dedup keys upstream (near-duplicate spans like `abd distension` / `ABD distension` / `abd  distension` are each independently normalized and can land on different concepts)

## Stage 3 — MoLLM Consensus Gate (`src/mollm_ensemble.py`)
Supports: **Objective 2 in its entirety** — top priority, currently 0 lines

- [ ] Local LLM client(s) for MedGemma 4B / OpenBioLLM 8B (vLLM endpoints per `.env`)
- [ ] KG retrieval: guideline triplets (Memgraph) + SNOMED FSN/hierarchy (Neo4j) for prompt context
- [ ] Confidence-gated routing: high-confidence → contradiction check; low-confidence → deeper resolution using provenance + hierarchy + guideline evidence
- [ ] Ensemble voting / composite confidence calculation
- [ ] Citation verification (hallucination detection against cited evidence)
- [ ] Decision artifact matching `Provenance_Schema.md` Stage 3 schema (`mollm_call_id`, `confidence_tier_in`, `retrieved_context`, per-model `reasoning`/`verdict`/`cited_evidence`/`citation_verified`, ensemble stats)

## KG Infrastructure & Ingestion
Supports: Objective 1 (blocks Stage 3, which can't query an empty graph)

- [ ] `scripts/import_athena.py` — Athena/OMOP → DuckDB ingestion (0 lines)
- [ ] `scripts/build_concept_embeddings.py` — SapBERT embeddings for `athena_concept` (0 lines; downstream `merge_embeddings.py` already works and expects this table to exist)
- [ ] `scripts/init_memgraph_snomed.py` — SNOMED IS_A hierarchy → Neo4j (0 lines)
- [x] **Final KG source decision (2026-08-07): `data/local_triplets_db2_v6/`** (MedCAT pipeline), not `data/rules-llm/`. `rules-llm` had better raw grounding (96.5% vs 50.5%) but worse structural integrity (14.4% dangling rule targets vs. 0%, inconsistent `snomed`/`icd10` field typing, weaker chunk-level provenance — see `Rules_LLM_Triplets_Review.md`). Structural soundness was judged more important than starting grounding %, since the grounding gap is fixable (see below) and broken references/types are a worse foundation to build Stage 3 retrieval on.
- [x] **`scripts/clean_local_triplets.py` written and run against the full 76-file corpus (2026-08-07)** — fixes the structural weaknesses from `Guideline_Triplets_KG_Review.md` that don't need the populated DuckDB. Non-destructive: writes to `data/local_triplets_db2_v6_cleaned/`, originals untouched. Report: `data/cleaning_reports/clean_local_triplets_20260807_143942.json`. Results:
  - [x] **Node consolidation (§3.2)** — 272 duplicate nodes merged by matching `snomed` code, with all rule `target` references rewritten to the surviving canonical node and 33 resulting self-loop rules dropped. Worked example: the KDIGO AKI chunk went from 16 nodes / 34 rules to 4 nodes / 24 rules, with all 6 duplicate "Acute Kidney Injury" nodes correctly consolidated into 1. **Safety finding during development**: an unconditional merge-by-snomed-code first pass incorrectly merged "SBP ≥160 mmHg" and "SBP <90 mmHg" into one node because they share a generic parent SNOMED code — fixed by only auto-merging when all duplicate nodes agree on `@type`; type-mismatched groups (41 found, e.g. that BP pair, and "suspected X" vs "confirmed X" pairs) are left unmerged and tagged `quality_flag: same_snomed_type_mismatch_not_merged` for human review instead of guessed at.
  - [x] **Predicate/type canonicalization (§3.4)** — 38 predicates and 26 types canonicalized via a conservative, hand-reviewed mapping (`SUGGESTS NOT USING`/`RECOMMENDS NOT USING` → `NOT_RECOMMENDED_FOR`; `quantitative_threshold`/`REQUIRES_QUANTITATIVE_THRESHOLD`/`HAS_THRESHOLD` → `HAS_QUANTITATIVE_THRESHOLD`; `IS_DEFINED_BY` → `DEFINED_BY`; type `quantitative_threshold` → `Quantitative Threshold`). Direction-ambiguous pairs (`RECOMMENDS` vs `RECOMMENDED_FOR`, `IS_OUTCOME` vs `IS_OUTCOME_OF`) deliberately left alone rather than guessed.
  - [x] **Citation classification (§3.3)** — every citation re-checked against its source chunk and tagged `citation_type`: 492 `verbatim` (≥0.8 containment), 132 `paraphrase_with_recovered_excerpt` (a real verbatim excerpt was found in the source and attached alongside the original), 202 `paraphrase` (no strong excerpt recoverable), 213 `pointer_unverifiable` (<0.4 containment — Stage 3's `citation_verified` check should skip or specially handle these rather than fail on them).
  - [x] **Boilerplate flagging (§3.7)** — nodes/rules matching known non-clinical boilerplate patterns (journal running headers, literature-search-methodology paragraphs) tagged `quality_flag: likely_boilerplate` rather than deleted, so a human/filter step decides inclusion.
  - [x] **Referential integrity re-verified post-cleaning: 0 dangling rule targets**, confirmed programmatically after all rewrites (not just assumed).
  - [ ] **Not yet done**: the actual SNOMED/ICD10 grounding backfill for the remaining `"N/A"` nodes — `scripts/backfill_guideline_grounding.py` (written 2026-08-07) still needs to run against the populated `kg2_lexical_store.duckdb` on the EC2 box, which isn't available from this workspace. Points at `data/local_triplets_db2_v6_cleaned/` by default so grounding backfill builds on top of the structural cleanup rather than the raw corpus.
    - [x] **GLiNER compound-name pre-extraction added (2026-08-07)**. GLiNER itself can't produce a code (Stage 2a only does span-finding/label classification, no vocabulary lookup) — clarified this explicitly since it was worth being precise about. What it *can* do: many ungrounded node names are compound phrases ("Serum creatinine increase by ≥0.3 mg/dl within 48 hours") unlikely to hit an exact Tier 1/2 match as a whole string, previously forcing them straight to the noisier Tier 3 fallback. The script now runs GLiNER over the name first, tries Tier 1/2 exact/synonym matching on the extracted core span (e.g. "Serum creatinine") before falling back to Tier 3 on the full name, and records `resolved_via` (`full_name` vs `gliner_span`) plus `matched_text` on every result so a reviewer can confirm the extracted span still represents the right concept. Deliberately does not run Tier 3 on GLiNER spans — stacking two fuzzy-match steps compounds uncertainty rather than reducing it. `--no-gliner` reverts to the original full-name-only behavior. Still needs the same populated DuckDB to actually run.
- [ ] `MERGE` nodes by `snomed` code at Memgraph-ingestion time too, in case the same concept appears across different *files* (the cleaning pass above only deduped *within* each file)
- [ ] Preserve the real predicate vocabulary as distinct Memgraph edge types rather than collapsing to the generic `GUIDELINE_RELATION` placeholder in `Databases.md` — update `Databases.md` to reflect it
- [ ] Confirm SNOMED release used for ingestion matches whatever release the evaluation dataset's `concept_id`s were annotated against (see Evaluation section below)
- [x] Source data now lives under `data/` (moved 2026-08-07): `data/local_triplets_db2_v6/` (raw), `data/local_triplets_db2_v6_cleaned/` (post-cleaning, use this for ingestion), `data/rules-llm/` (reviewed, not selected — see `Rules_LLM_Triplets_Review.md`), `data/triplets-rules-backup-data/` (raw chunks/MedCAT entities/grounded chunks), `data/evaluaiton-dataset/`. `code/data/guidelines/`, `code/data/snomed_kg/`, `code/data/athena_omop/` still only contain `.gitkeep`.

## Stage 4 — Routing & Ingestion (HITL + KG write-back)
Supports: **Objective 3** — not started

- [ ] Provenance tagging: `AUTO_VALIDATED` / `MOLLM_RESOLVED` / `HUMAN_VERIFIED`
- [ ] Atomic Cypher transaction writing `:PatientObservation` → `:MoLLMDecision` → optional `:HITLReview`
- [ ] `ui/pages/2_🩺_HITL_Review_Queue.py` — reviewer interface (0 lines)
- [ ] HITL review schema per `Provenance_Schema.md` (`hitl_case_id`, `queue_reason`, `presented_suggestion`, `reviewer_decision`, `corrected_concept_id`, `rejection_reason`, `review_duration`)
- [ ] **Gate KG 3 write-back behind Stage 3 + the calibrated confidence threshold, not Stage 2b confidence alone.** Writing today's unfiltered high-confidence outputs (raw GLiNER softmax ≥0.5, ungated Tier-3 top-1) into KG 3 risks baking silent errors (e.g. `bioplar`→`Bourgvilain`) into the graph as "verified," which then get retrieved as grounding evidence for future extractions — a pseudo-labeling feedback-loop risk.

## Stage 5 — Active Learning Feedback Loop
Supports: **Objective 3 / Objective 4** — not started, depends on Stage 3 + Stage 4

- [ ] KG 3 → GLiNER prompt/search-space feedback mechanism
- [ ] Regression check against held-out gold-evaluation set on each KG 3 update
- [ ] **Triplets-first KG 3 design, KGE as a periodic derived layer** (not a replacement — see `Proposal_Alignment_Review.md` §6 for full reasoning): store extractions as actual graph triples for auditability/HITL/provenance; separately compute CompGCN/TransE/RotatE embeddings over the accumulated graph on a batch cadence, used to re-rank Tier-3 candidates and support cold-start bootstrapping for graph-connected but textually-novel entities.
- [ ] Deflection rate / false-deflection rate tracking (Objective 5 metrics, but the data only exists once this loop runs)

## Evaluation Suite (`evaluation/`)
Supports: **Objective 5** — dataset is now ready; code is not

- [x] **Evaluation dataset confirmed and verified** — `evaluaiton-dataset/snomed-ct-entity-linking-challenge-1.2.0/`: 272 notes, 75,491 annotations, 6,595 concepts, official `train`(204 notes)/`test`(68 notes) split preserved via `annotation_type` column, plus 1,065 supplementary `proposed_ACCEPTED` annotations across 233 notes.
- [ ] Data loader: cast `start`/`end` from float-strings to int before slicing note text
- [ ] Decide `proposed_ACCEPTED` handling (include, exclude, or report both ways) and document the choice
- [ ] Verify whether medication-related concept_ids in the dataset (confirmed present: `Aspirin`, `Warfarin`, `heparin drip`, etc.) resolve to Clinical Finding or Substance domain once SNOMED KG is loaded — determines whether `Medication` GLiNER outputs are scoreable against this dataset directly
- [ ] `eval_suite.py` — span-level char-IoU (for any DrivenData-comparable claims) + concept-level (span, concept_id) precision/recall/F1 (for internal reporting), broken down by entity type/vocabulary with confidence intervals (still 0 lines itself, but `scripts/score_gold_recall.py` already implements both the recall/precision breakdown and a faithful `official_character_iou()` re-implementation of the benchmark's own `scoring.py` — worth pointing `eval_suite.py` at that rather than rebuilding it, or renaming/moving it)
- [x] `cal_eval.py` — Expected Calibration Error / reliability diagram on the validation slice. **DONE 2026-08-11.** Grades resolution-mode `mollm_decisions` against SNOMED gold via candidate crosswalk; computes ECE, reliability table, threshold sweep, per-model `raw_confidence_label` breakdown. Deliberately does NOT grade contradiction/non_asserted_check verdicts (no gold label exists for guideline-compliance correctness — needs the Stage 5 human re-audit sample) and reports that exclusion explicitly rather than silently only covering part of Stage 3. See `docs/Stage3_Open_Issues.md` Issue 4 for the full scoping rationale and the sample-size caveat — this is tooling, not yet a real threshold fit (too little data run through Stage 3 so far).
- [ ] `ablations.py` (0 lines)
- [ ] Note-level bootstrap CIs, paired comparison design vs. Clinical-T5 baseline
- [ ] If claiming leaderboard-comparable results: implement scoring per the official `scoring.py` (char-IoU) separately from internal F1 reporting — the two metrics aren't interchangeable
- [ ] Explicitly state in the dissertation that evaluation uses the archived Challenge's own official train/test split (not a custom split, and not the separate live rolling Benchmark's hidden test set)

## UI (`ui/`)
Supports: Objective 3

- [ ] `ui/app.py` — Streamlit entrypoint (0 lines)
- [ ] `1_🚀_Pipeline_Runner.py`, `2_🩺_HITL_Review_Queue.py`, `3_🔍_Troubleshooting.py`, `4_📊_Evaluation_Metrics.py` — all 0 lines
- [ ] `ui/components/graph_visualizer.py`, `json_tree_view.py`, `offset_highlighter.py` — all 0 lines

## Housekeeping / Cross-Cutting
- [ ] Confirm whether inference (not just training) runs on the `/home/ec2-user/...` EC2 host, and whether that's consistent with the proposal's "local infrastructure only" privacy claim once real MIMIC-IV data is processed there — check PhysioNet's cloud-hosting attestation requirements if so
- [ ] Update `code/README.md` — still describes Clinical-T5 as part of Stage 2a; should reflect its baseline-only role
- [ ] Point `evaluation/eval_suite.py`'s data loader at `evaluaiton-dataset/`, kept structurally separate from `data/raw_notes/discharge.csv` (the bulk unannotated AL-stream corpus) so the 272 gold notes never leak into the AL stream
- [ ] Consider renaming `evaluaiton-dataset/` → `evaluation-dataset/` (typo, currently harmless but will get baked into paths/scripts as code is written against it)
- [ ] Tests: `tests/test_offset_mapping.py`, `tests/test_pipeline_integration.py` are 0 lines despite the offset-mapping logic being the most bug-sensitive part of the working code so far

---

## Suggested Build Order

Matches the proposal's own risk-mitigation priority ("build the core active learning loop before secondary optimization features"), which the current implementation state has inverted:

1. Fix remaining Stage 2b threshold + domain-filter bugs (determinism already fixed — see line 38; small, contained, unblocks trustworthy numbers for everything after)
2. `mollm_ensemble.py` (Stage 3) — the actual Objective 2 deliverable
3. KG ingestion scripts — Stage 3 needs a populated graph to query
4. Stage 4 routing/ingestion + HITL queue — Objective 3
5. Stage 5 AL feedback loop, KG 3 write-back gated behind Stage 3/4
6. `eval_suite.py` / `cal_eval.py` / `ablations.py` — dataset is ready, so this can start as soon as there's a pipeline to measure
