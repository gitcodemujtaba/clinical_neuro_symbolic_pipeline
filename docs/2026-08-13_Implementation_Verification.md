# P0–P5 Implementation — Verification Record

**Date:** 2026-08-13
**Scope:** the eight items in `docs/2026-08-13_Code_Improvement_Proposals.md` §"Suggested sequencing".

---

## 1. Provenance note — who wrote what

Steps 1–7 and the library half of step 8 **were already on disk** when implementation was requested. File mtimes place them between 14:57 and 15:28, i.e. after the proposal document (14:52) and before the request to implement it. They are recorded here as **verified, not authored**, in this session:

| Step | Item | Files | Status |
|---|---|---|---|
| 1 | P2.3 provenance columns | `src/provenance.py` (new), migrations on `mollm_decisions` / `normalized_entities` / `mollm_review_decisions` | pre-existing, verified |
| 2 | P5 reconciliation | `evaluation/stage_calibration.py` | pre-existing, verified |
| 3 | P0.1/P0.2/P0.3 | `scripts/make_splits.py`, `evaluation/splits.py`, `evaluation/metrics.py` (new); 4 eval scripts retrofitted | pre-existing, verified + 1 gap fixed |
| 4 | P1.1 Tier 1/2 ranker | `src/normalization.py` | pre-existing, verified + 1 divergence corrected |
| 5 | P1.2 uncensor reasons | `src/normalization.py` | pre-existing, verified |
| 6 | P4 degeneracy detection | `src/llm_client.py`, `src/mollm_ensemble.py` | pre-existing, verified |
| 7 | P2.1/P2.2 override + shuffle | `src/mollm_ensemble.py`, `scripts/run_stage3_batch.py` | pre-existing, verified |
| 8 | P3 calibrator | `src/mollm_calibrator.py` | half pre-existing; **script half written this session** |

---

## 2. Work completed this session

### 2.1 P3 — the fitting script had not been touched (`scripts/fit_mollm_calibrator.py`)

`src/mollm_calibrator.py` was reworked at 15:28; the script that actually fits and saves the production model was last modified at 11:42 and still carried every defect P3 names. Fixed:

- **A live drift bug, not just a style issue.** The script hand-rolled `LogisticRegression(max_iter=1000, C=1.0)` under a comment promising it matched `fit()`. P3.3 had added `class_weight="balanced"` to `fit()` and not to the script — so the model written to `models/mollm_calibrator_v1.pkl` was a *different estimator* from the documented one, and nothing could detect it. Closed by adding `MoLLMCalibrator.fit_vectors()` and `_new_estimator()`: `fit()` now delegates, the script calls `fit_vectors()`, and there is exactly one construction site. `tests/test_calibrator_fit.py` asserts that by AST, so a second copy configured identically today still fails the test.
- **The leakage guard was inert.** `load()`'s refusal depends on `training_note_ids` recorded at save time; the script called `save(path)` with no provenance, so the guard would have printed "cannot be checked" forever while appearing to work. The script now passes `training_note_ids`, `training_split` and `code_version`. This required `cal_eval.py --emit-training-data` to emit `note_id`/`split` per row — it did not.
- **Accuracy demoted, discrimination promoted (P3.1).** AUROC and average precision are now the headline, with the null model's numbers printed in the same table and an explicit "reading" block that says so when the AUROC interval spans 0.5. The prior report's 85.71% held-out accuracy and 0.0299 ECE were both what a base-rate predictor produces on a 14.3% positive rate.
- **Stratified 5-fold out-of-fold CV (P3.2)** replaces the single unstratified 80/20 split, so evaluation covers all 140 rows rather than 28, with folds capped at the minority-class size.
- **Note-level bootstrap CIs (P0.3)** on AUROC and AP; falls back to row-level only with a printed warning.
- **`_ece()` deleted (P0.2)** — the third copy of the formula. Now imports `evaluation.metrics`.
- **`PROJECT_DIR` derived from `__file__`** (P3.5), overridable via `CNSP_PROJECT_DIR`. The closing NOTE claiming `run_stage3_batch.py` has no calibrator wiring was stale and has been replaced.

### 2.2 P0.1 — `evaluation/mollm_trust_tiers.py` was the last script defaulting to "every note in the DB"

Retrofitted with `add_split_args` / `resolve_note_ids` / `assert_no_contamination`. This matters more than the other four: its 47.6% `REJECTED_GOLD_MISMATCH` figure is the closest thing the project has to the false-deflection rate `docs/Evaluation_Criteria.md` names as a patient-safety metric, so it is precisely the number that must not be computed over an ad hoc pool.

### 2.3 P1.1 — semantic criterion gated off, per the metadata-only decision

The ranker as found computed SapBERT cosine for **every** candidate on **every** multi-hit Tier 1/2 lookup. That contradicts both the metadata-only choice made for this implementation and `_lookup_tier12()`'s own defended property — *"no embedding call per attempt"* — which matters because compound-split partition search calls Tier 1/2 once per candidate boundary, so the cost scales with boundaries tried, not entities.

Criterion 3 is now behind `TIER12_RANK_SEMANTIC` (`CNSP_TIER12_SEMANTIC`, default off) and an explicit `use_semantic=` parameter. Default ordering is `concept_class → domain → specificity → concept_id`: pure SQL and dict lookups, no latency change, every decision traceable to an inspectable vocabulary field. `tier12_rank_basis` records `ranked_v1` vs `ranked_v1_semantic` so the two are separable per row — semantics-on-top-of-metadata is a second A/B, and running it simultaneously with the ranker's own A/B would make both uninterpretable.

### 2.4 A whole test suite was silently red

`tests/test_offset_mapping.py` raised `NameError: _numeric_context_kind` on **every** test. Its AST-based loader extracts a named set of pure functions from `preprocessing.py`; the numeric-context and groundability fixes earlier the same day added helpers that `expand_text_and_track_offsets()` now calls without adding them to that set. This predates the P0–P5 work and was found by the verification pass, not reported by it. The 2026-08-13 report's "6 stub tests, all passed" for the groundability fix refers to separate isolated tests — this older suite was left broken. Fixed by extending the extraction set.

---

## 3. Verification results

`py_compile` clean across `src/`, `evaluation/`, `scripts/`, `tests/`.

| Suite | Result |
|---|---|
| `evaluation/metrics.py` self-test | 27 passed, 0 failed |
| `evaluation/splits.py` self-test | 15 passed, 0 failed |
| `tests/test_offset_mapping.py` | ALL PASS (was: NameError on every test) |
| `tests/test_tier12_ranking.py` | 33 passed, 0 failed (+6 for the semantic gate) |
| `tests/test_override_gate.py` | 33 passed, 0 failed |
| `tests/test_degenerate_generation.py` | 29 passed, 0 failed |
| `tests/test_confidence_tier_reasons.py` | 27 passed, 0 failed |
| `tests/test_calibrator_fit.py` (new) | 54 passed, 0 failed |

Split file checked directly: 272 notes — **70 test / 60 val / 142 train**, matching `docs/Evaluation_Criteria.md`'s ~70-locked design, with a SHA256 recorded in the file header.

**Not runnable in this environment** (no DuckDB, no scikit-learn, no network): anything requiring the live EC2 database, and `tests/test_stage3_safety_rules.py`, which needs `pytest`. `tests/test_calibrator_fit.py` injects a fake `sklearn` so the plumbing is still covered; it does not verify that logistic regression works, only that the right estimator is constructed once, provenance round-trips, the leakage guard fires, and every row gets an out-of-fold prediction.

---

## 4. Two things that need a decision before the next run

**The highest-value fix is not switched on.** `RANKED_TIER12` defaults to `False`. The reasoning in the code is sound — enabling it changes what Stage 2b returns and invalidates existing `mollm_decisions` rows — and the A/B procedure is documented at the constant. But until it is flipped, Tier 1's measured 52.71% is unchanged. The documented order is: run `stage2b_cal_eval.py --per-candidate --split val` first to read the ranking **headroom** (oracle minus top-1). If that number is near zero, the right concept is not in the candidate list at all, retrieval is the bug, and the ranker cannot help regardless.

**`models/mollm_calibrator_v1.pkl` is stale and should be refit.** It was produced by the pre-P3 script, so it (a) was fit without `class_weight="balanced"`, and (b) carries no `training_note_ids` — meaning `load()` cannot leakage-check it and will say so on every batch. Refit before the next Stage 3 run:

```
python3 evaluation/cal_eval.py --split val --emit-training-data mollm_cal_train.jsonl
python3 scripts/fit_mollm_calibrator.py --train-data mollm_cal_train.jsonl \
    --out models/mollm_calibrator_v1.pkl --report-json cal_fit_report.json
```

Expect the AUROC interval to span 0.5 on n=140 with 20 positives. If it does, the honest conclusion is that the calibrator does not discriminate yet and its low ECE is base-rate prediction — which is what the new report block is built to say out loud rather than leave to a footnote.
