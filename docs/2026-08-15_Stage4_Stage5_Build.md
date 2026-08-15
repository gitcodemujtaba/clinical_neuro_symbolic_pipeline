# Stage 4 (Routing & Ingestion) + Stage 5 (Active Learning) Foundations — Session Report

**Date:** 2026-08-14 → 2026-08-15 (overnight, immediately following `docs/2026-08-14_GOLD_MISSING_RootCause_Fixes.md`)
**Scope:** `src/hitl_queue.py`, `ui/app.py` + `ui/pages/2_🩺_HITL_Review_Queue.py`, `src/kg3_ingestion.py` + `scripts/run_kg3_ingestion.py`, `src/kg3_query.py` + `scripts/regression_check_kg3.py`
**Status:** Stage 4 built and verified against real production data. Stage 5 deliberately scoped to foundations only.

---

## 1. What triggered this session

With Stage 3 substantially reworked and re-validated the same day (`docs/2026-08-14_GOLD_MISSING_RootCause_Fixes.md`), the natural next step was Stage 4 (routing/HITL/KG write-back) and Stage 5 (active-learning feedback). `docs/Implementation_Checklist.md`'s own Stage 4 entry already warned against writing unfiltered high-confidence Stage 3 output straight into KG3 ("a pseudo-labeling feedback-loop risk") — this session's own AUTO_VALIDATED precision measurement (39.4% on the experimental harness, later 52.6% on the real production path — see §4) confirmed that risk is live, not hypothetical, and shaped the whole design: **every** Stage 3/Objective-3 decision is queued for human review right now, regardless of its own routing tier, until a calibrated confidence threshold says otherwise.

## 2. Pre-build research findings

- **`ui/` was entirely empty** — 0 bytes, every file, since the initial commit. Not "one missing page among working siblings" as the checklist's phrasing implied; the HITL Review Queue page is the first real Streamlit code in this repo, with no prior convention to follow. `streamlit` was pinned in `requirements.txt` but not actually installed.
- **No Memgraph/Neo4j write code existed anywhere** — only a read/profile script (`scripts/profile_databases.py`, `neo4j` Bolt driver against both `NEO4J_URI`/`MEMGRAPH_URI`). Both ports (7687, 7688) confirmed reachable in this environment.
- **Two parallel decision tables** feed Stage 4: `mollm_decisions` (Objective 2, `src/mollm_ensemble.py`, citation-gated — what the same-day Stage 3 session validated) and `mollm_review_decisions` (Objective 3, `src/mollm_review.py`, confidence-driven, all-tier — its own docstring states it exists specifically to "produce the CANDIDATE rows a future Stage 4 job would consume"). Decision: read and unify both, tagged by `source_table`.
- **Today's earlier Stage 3 validation never touched either production table** — it ran through the experimental harness (`scripts/experiment_3b_voting.py`), which writes to a JSON file only. Both tables held only stale data (1870 rows from 2026-08-13, before the day's fixes) or none. Decision: run the real production paths first.
- `run_stage3_batch.py`/`mollm_review.py` process the **raw** tier population — 2253 LOW-tier + 50 HIGH-tier = 2303 entities for this corpus — not the `candidates>=2` "genuinely ambiguous" subset (~753-759) the experimental harness measures. The two harnesses' precision numbers are not directly comparable.

## 3. What was built

### Step 1 — Real production data
Cleared 1870 stale `mollm_decisions` rows for this corpus (backed up to JSON first; all confirmed stale via `code_version` — either `NULL` or predating the day's fixes, never the current commit). Ran `scripts/run_stage3_batch.py` (full LOW-tier scope) and `src/mollm_review.py`'s own `main()` (full HIGH+LOW scope) to completion: **2247 + 2297 processed, 0 errors, 0 degenerate generations**, ~4.6h and ~4.9h respectively.

### Step 2 — `src/hitl_queue.py`
`hitl_review_queue` DuckDB table, schema exactly per `docs/Provenance_Schema.md`'s Stage 4 field list. `enqueue_pending_cases()` unifies both source tables (idempotent — re-running adds only genuinely new rows). `submit_review()` sets `final_ingestion_path='HUMAN_VERIFIED'` only on APPROVED/CORRECTED, never REJECTED.

**Real design gap found and fixed here:** resolving "what concept does an Approve click confirm" needed parsing the actual production verdict format — `RESOLVED_TO_CANDIDATE_<N>` for resolution-mode majority vote, `SUPPORTED`/`CONTRADICTED`/`INSUFFICIENT_EVIDENCE` for contradiction-mode (defaulting to Stage 2b's own top-1, since Stage 3 in that mode validates the existing pick rather than choosing among several). `_suggested_omop_concept_id()` implements this, computed once at enqueue time so "Approve" has an unambiguous, fixed meaning rather than being re-derived at ingestion time. `mollm_review_decisions` rows have no concept_id anywhere (only a `proposed_concept_name` string) — `suggested_omop_concept_id` is always `None` for that source, and an APPROVED case from it raises `UningestibleCase` loudly at ingestion time rather than guessing from the name.

### Step 3 — `ui/app.py` + `ui/pages/2_🩺_HITL_Review_Queue.py`
Minimal landing page + the reviewer interface: filterable queue, one case at a time, full candidate/model-verdict display, Approve/Correct/Reject with a concept picker and rejection-reason field, session-tracked review duration. All three review paths verified via `streamlit.testing.v1.AppTest` (automated, no live browser needed) — including a real bug caught this way: `load_hitl_queue()` ran unconditionally on page load before `ensure_hitl_queue_table()` had ever been called, crashing on a fresh database. Fixed by calling table-creation unconditionally right after connecting.

### Step 4 — `src/kg3_ingestion.py` + `scripts/run_kg3_ingestion.py`
`ingest_reviewed_case()`: one atomic Cypher transaction (`session.execute_write()`) writing `MERGE`-keyed `:PatientObservation`→`[:INSTANCE_OF]`→`:Concept`, `:PatientObservation`→`[:VALIDATED_BY]`→`:MoLLMDecision`→`[:REVIEWED_BY]`→`:HITLReview`, matching `docs/Databases.md` §3 exactly. `MERGE` (not `CREATE`) throughout for idempotency — verified directly against the real running Memgraph instance: full chain readable via Cypher after write, node counts unchanged after a deliberate re-ingestion of the same case. Batch driver reads `hitl_review_queue` for un-ingested `HUMAN_VERIFIED` rows, resolves the concept via `resolve_concept_id()`, looks up `vocabulary_id`/`domain_id`/`concept_name` from `athena_concept`, writes, and stamps `ingested_at` so re-runs only pick up new cases. `--dry-run` resolves and prints without writing.

### Stage 5 foundations — `src/kg3_query.py` + `scripts/regression_check_kg3.py`
Read interface (`count_by_label()`, `get_observations_for_concept()`, `get_accepted_triples()`) built and tested against real Memgraph, ready for whenever the real feedback mechanism gets built. Regression-check tool reuses `scripts/score_gold_recall.py`'s scoring directly (not reimplemented), snapshots gold-recall + KG3 node counts to `reports/kg3_regression_snapshots/`, diffs against the previous snapshot and flags drops (never flags improvements, even if the raw number also moved). Verified against real data — baseline snapshot taken 2026-08-15.

**Deliberately not built:** the GLiNER prompt-feedback mechanism and CompGCN/TransE/RotatE re-ranking layer. Both need a meaningful volume of accumulated `HUMAN_VERIFIED` corrections to mean anything; as of this session's end, 0 cases have been human-reviewed (4,763 are queued and pending). Building the feedback mechanism now would mean building it against no data.

## 4. Verification performed

Every new write-path component (`hitl_queue.py`, the Streamlit page, `kg3_ingestion.py`) was built and unit/integration-tested against a synthetic temp DuckDB file first, since both Stage 3 batch jobs held DuckDB's file-level write lock (blocks ALL connections, not just to the tables being written) for their full ~4.5-5h runtime. Once released, every component was re-verified against the real, current data:
- `enqueue_pending_cases()`: 4,763 real cases enqueued from both source tables in ~33s.
- Spot-checked real `mollm_decisions`-sourced cases resolve `suggested_omop_concept_id` correctly (non-null, plausible values).
- `scripts/run_kg3_ingestion.py --dry-run`: correctly reports 0 ingestible cases (nothing reviewed yet).
- `scripts/regression_check_kg3.py`: full run against real data, baseline snapshot written, comparison-against-previous-snapshot path also verified on a second run.
- Full test suite (47 pytest-collected tests) passing throughout.

### Real production Stage 3 results (from the Step 1 batch runs)

Gold-crosswalk precision by tier, computed directly against `mollm_decisions` (1665 gradable of 2337 total decisions):

| Tier | Gradable | Correct | Precision |
|---|---|---|---|
| AUTO_VALIDATED | 532 | 280 | **52.6%** |
| MOLLM_RESOLVED | 100 | 35 | 35.0% |
| HITL_REQUIRED | 1033 | 423 | 40.9% |

Unlike the experimental-harness measurement the same day (where AUTO_VALIDATED trailed MOLLM_RESOLVED), production shows the *correct* tier ordering — AUTO_VALIDATED highest, consistent with its being 97.7% "contradiction mode" (confirming Stage 2b's existing pick against evidence, a genuinely easier judgment than picking among ambiguous candidates; resolution-mode entities essentially never reach AUTO_VALIDATED, only 1.9% of it). Still, 52.6% is far short of safe-to-skip-review — the empirical justification for queuing everything.

HITL_REQUIRED breakdown (1354 cases): `model_disagreement` alone drives 82.1% of all HITL_REQUIRED cases (1112) — the hard citation/reasoning/contradiction safety gates combined are only ~8% (181 cases). By GLiNER label, Lab Test dominates disagreement (23.7%) while being one of the lowest AUTO_VALIDATED-share labels (6.9%) — consistent with the same-day GOLD_MISSING session's finding that lab-value entities remain the hardest category even after targeted fixes.

## 5. Where to pick up next

1. **Review some of the 4,763 queued cases** through the UI — the only way Stage 5's real feedback loop gets real data to work with.
2. Once meaningful review volume accumulates, fit `src/mollm_calibrator.py` against real reviewed outcomes and replace the blanket "review everything" gate with an actual calibrated threshold.
3. Only then does building the real Stage 5 feedback mechanism (GLiNER prompt feedback, KGE re-ranking) become buildable against real data rather than a guess.
