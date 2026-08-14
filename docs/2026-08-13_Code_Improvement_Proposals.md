# Code Improvement Proposals — from the 2026-08-13 Calibration Report

**Input:** `docs/2026-08-13_Calibration_Diagnostics_And_Fixes.md`
**Method:** every proposal below is anchored to a specific finding in that report and a specific file/line in `code/`. Nothing here is speculative refactoring.
**Framing:** the report's own headline — two of three confidence signals don't predict correctness — is a *measurement* result. The right response is (a) make the measurement trustworthy, (b) fix the thing the measurement blames, (c) stop the pipeline from doing net harm while (a) and (b) land. Ordered accordingly.

---

## P0 — The measurement itself is not yet trustworthy

These block the value of every number in §5–§7. Until they land, the report's own caveat applies: *"development-time measurements, not proposal-compliant validation-slice numbers."*

### P0.1 Materialise the train/val/test split as data, not as discipline
**Finding:** §4.3 — no file anywhere enumerates the ~70-note locked test set. Every eval script defaults `--note-ids` to "every note_id in the database."
**Why it matters:** `docs/Evaluation_Criteria.md` makes the split a leakage-control requirement, and the report concedes today's ECE pool "likely overlaps with notes used to derive today's own fixes (e.g. the `fx` groundability heuristic)." A convention that lives only in a doc is not a control.

**Change:**
- New `scripts/make_splits.py` → writes `data/splits/note_splits.csv` (`note_id,split`) once, fixed seed, and prints a SHA256 of the file to be recorded in the doc. Regeneration must be idempotent.
- New `evaluation/splits.py`: `load_split(name) -> set[str]`, plus `add_split_args(parser)` giving every script a `--split {val,test,all}`.
- **Default `--split val`**, not `all`. `--split test` requires an explicit `--unlock-test` flag and prints a loud banner. `--note-ids` becomes mutually exclusive with `--split`.
- Retrofit: `cal_eval.py`, `stage2a_cal_eval.py`, `stage2b_cal_eval.py`, `stage1_disambiguation_eval.py`, `stage_calibration.py`, `fit_mollm_calibrator.py`.

This is the single highest-value change in this document — it costs a day and it is the difference between "we measured" and "we measured something we can cite."

### P0.2 One ECE implementation, and better than equal-width bins
**Finding:** §5 methodology names three separate implementations of the same formula (`cal_eval.compute_ece`, `fit_mollm_calibrator._ece`, "reimplemented identically" in `stage2b_cal_eval.py`).
**Why it matters:** the report's §3.3 `decoding_mode_mismatch` bug is exactly this failure class — two copies of one idea drifting. Also, equal-width ECE is close to meaningless on Stage 3's distribution: 128 of 140 points sit in a single bin, so ECE 0.773 is essentially `|0.918 − 0.125|` with extra steps.

**Change:** new pure-Python `evaluation/metrics.py` (no DB imports, so `fit_mollm_calibrator.py` can use it too), exporting:
- `compute_ece(pairs, n_bins=10, scheme="equal_width"|"equal_mass")` — add **equal-mass (quantile) binning** as the reported default for Stage 3, and report `n_nonempty_bins` alongside so a degenerate single-bin result is visible on its face.
- `brier_score()`, `max_calibration_error()`, `auroc()`, `average_precision()`.
- Delete the two duplicate implementations; import from here.

### P0.3 Bootstrap confidence intervals — required by the proposal, absent from the repo
**Finding:** `grep -rl bootstrap evaluation/ scripts/` returns nothing. `docs/Evaluation_Criteria.md` requires note-level bootstrap CIs and a paired comparison design, and requires CIs "alongside every subgroup."
**Why it matters:** every headline number in the report is a bare point estimate. 14.29% on n=140 carries roughly ±6pp; Tier 1's 52.71% vs Tier 2's 61.84% is presented as a real inversion but is never tested; **NET VALUE = −17 has no interval at all**, and the fresh 3-note replication (n=5) is correctly caveated but cannot be combined with the main result without one.

**Change:** `evaluation/metrics.py: bootstrap_ci(units, statistic_fn, n_resamples=2000, seed=...)`, **resampling at the note level, not the entity level** (entities within a note are not independent). Every accuracy/ECE/net-value table in the eval scripts prints `estimate [lo, hi]`. Add `paired_bootstrap_diff()` now, since the T0/T1/T2 Clinical-T5 comparison will need it and building it under deadline is how it gets built wrong.

---

## P1 — Fix what the measurement actually blames

### P1.1 Replace `ORDER BY concept_id ASC LIMIT 1` with a real ranker (highest accuracy lever in the pipeline)
**Finding:** §5.3 — Tier 1 "Exact" is **52.71%** accurate, worse than Tier 2's 61.84%, and the report already root-causes it: *"an arbitrary tie-break, not a clinical one."*
**Where:** `src/normalization.py` — `_lookup_tier12()` (L437, L450), `_tier_queries()` (L~990), `normalize_entity()`'s Tier 1/2 queries (L~1128).
**Why it matters:** lowest `concept_id` reflects Athena's ID assignment order and nothing else. This is not a calibration problem to route around; it is ~47% of exact-matched entities being handed the wrong concept before any confidence signal is even computed. Every downstream number in the report inherits it.

**Change:** keep the SQL returning up to `CANDIDATE_LIMIT` rows, then rank in Python with an explicit, documented, deterministic key:
1. `concept_class_id` preference conditioned on `gliner_label` (Condition → *Clinical Finding*/*Disorder* ahead of *Qualifier Value*/*Navigational Concept*/*Staging & Scales*; Procedure → *Procedure*; Anatomy → *Body Structure*).
2. Agreement with `GLINER_LABEL_TO_DOMAIN[gliner_label]`.
3. SapBERT cosine between the **entity's sentence context** and the concept name — the embedding path already exists for Tier 3; reuse it here rather than only as a fallback tier.
4. `concept_ancestor` descendant count as a specificity prior (prefer the more specific concept when both are plausible).
5. `concept_id ASC` as the final tiebreak, so behaviour stays deterministic and reproducible.

**And the part that matters as much as the ranking:** when the top two survive step 3 within an epsilon, set `ambiguous=True` / `reason="unresolved_exact_tie"` instead of returning top-1 silently. The ambiguity machinery already exists (`multiple_exact_concept_name_matches`) but currently fires on `len(cands) > 1` *before* any ranking, which is why Stage 3 is drowning in easy cases while the genuinely hard ties get resolved arbitrarily and never surface.

**Related, cheap, testable:** the `"Left"` / `"Initial"` collision family suggests a hard guard — a Tier-1 exact match whose `concept_class_id` is *Qualifier Value* and whose surface form is a bare positional/temporal/laterality word should be **rejected outright**, not linked at 100% "exact" confidence. Measure the hit rate first with a one-off script before wiring it in.

### P1.2 Stop `compute_confidence_tier()` from suppressing the signals you'll need to re-enable it
**Finding:** §3.2 — after removing the Tier 1/2 exemption from `weak_match_tier`, `match_tier is not None → LOW`. The report itself notes the other two exemptions "become harmless no-ops."
**Why it matters:** they are not harmless for *measurement*. The `high_gliner_risk` gate (L1348) and the `short_token`/`isupper`/`alnum_mix` gate (L1376) still skip *appending the reason* for Tier 1/2 entities. So the `reasons` list — the only record of which signals fired — is now systematically censored exactly on the population you would need to study to ever restore an exemption. The function has also become, in effect, a constant: it returns HIGH essentially never, which makes the whole tier column non-discriminative in future evals.

**Change:** remove the `match_tier not in (...)` conditions from both remaining gates so every signal *fires and is recorded*, while routing behaviour stays identical (it's an OR, and `weak_match_tier` already forces LOW). Zero behaviour change, full signal recovery. Then add a `reasons`-vs-gold breakdown to `stage2b_cal_eval.py` so the next iteration can say which signal actually predicts anything.

---

## P2 — Stop Stage 3 doing net harm while P1 lands

### P2.1 Require MoLLM to *beat* Stage 2b, not merely differ from it
**Finding:** §6 — INTRODUCED_ERROR 30 (22.4%) vs CAUGHT_AND_FIXED 13 (9.7%). **NET VALUE = −17**, replicated in direction on fresh notes.
**Where:** `src/mollm_ensemble.py: route()` (L~686+), constants at L72–73.
**Why it matters:** resolution mode currently treats "override Stage 2b's top-1" and "confirm it" as symmetric decisions gated by the same threshold. They are not symmetric: an override that is wrong destroys a correct link, and the measured override is wrong more than twice as often as it is right.

**Change:** asymmetric acceptance. When `verdict != RESOLVED_TO_CANDIDATE_1` (i.e. MoLLM is overriding Stage 2b's own top-1), require **all** of:
- `calibrator_score >= OVERRIDE_TOP1_THRESHOLD` (new, explicitly named constant — not a reuse of `AUTO_VALIDATE_THRESHOLD`),
- `grounding_basis == "guideline_rule"` with `citation_verified == True`.

Otherwise route `HITL_REQUIRED` with `queue_reason="unsupported_top1_override"`. By construction this drives INTRODUCED_ERROR toward zero while preserving evidence-backed CAUGHT_AND_FIXED cases. It also gives §6 a metric that can be re-run to prove the change worked.

### P2.2 Test for candidate-position bias before trusting any resolution number
**Finding:** §6 — INTRODUCED_ERROR examples cluster in one family (`AST-956`, `AST-909`, `AST-16`, `CK(CPK)`).
**Why it matters:** candidates are presented to the model in `concept_id ASC` order (P1.1). Position bias in list-selection prompts is well documented, and a cluster of near-identical lab strings is precisely where it would show. If it's real, it is a prompt bug being misread as a model-quality finding.

**Change:** `--shuffle-candidates` flag on `run_stage3_batch.py` that permutes the presented order, records the permutation in the decision artifact, and maps the verdict back. Run the same 3-note slice both ways and compare. Cheap; settles the question either way.

### P2.3 `mollm_decisions` has no provenance columns at all
**Finding:** §6 caveat — *"no timestamp column to distinguish which decisions were made against which version of the candidate list"*; §9.6 — 22 orphan decisions, uninvestigated.
**Why it matters:** this permanently contaminates every retrospective analysis of that table, and it is the cheapest fix in this document.

**Change:** additive ALTERs, same pattern the module already uses (L1084–1098):
```sql
ALTER TABLE mollm_decisions ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;
ALTER TABLE mollm_decisions ADD COLUMN IF NOT EXISTS run_id VARCHAR;
ALTER TABLE mollm_decisions ADD COLUMN IF NOT EXISTS code_version VARCHAR;   -- git rev-parse HEAD
ALTER TABLE mollm_decisions ADD COLUMN IF NOT EXISTS candidates_hash VARCHAR; -- hash of the exact list shown
```
`candidates_hash` is what actually kills the drift confound: it lets a later eval join on "was this decision made against the same candidate list I'm grading it against," which no timestamp can answer. Apply the same four columns to `normalized_entities` and `mollm_review_decisions` while you're in there.

---

## P3 — The calibrator is currently a base-rate predictor

**Finding:** §5.4 — held-out accuracy 85.71% "exactly matches the base rate," mean predicted P = 0.1274, flagged in the report as a likely majority-class artifact. Held-out ECE 0.0299 is presented with the right caveat but will be misread by anyone reading only the table.
**Where:** `src/mollm_calibrator.py`, `scripts/fit_mollm_calibrator.py: held_out_check()`.

**Changes, in order of importance:**

1. **Stop reporting accuracy as the headline.** On a 14.3% base rate, "85.71% accurate" is what you get by predicting `0` for everything. Report **AUROC and average precision** as the primary discrimination metrics, with accuracy demoted or removed. Print the **null-model baseline** (predict the base rate for every example) and its ECE next to the model's — that single row would have made the artifact self-evident rather than requiring a caveat.
2. **Stratified 5-fold CV, not one 80/20 shuffle.** `held_out_check()`'s docstring says stratification "doesn't matter much at n=140" — with only 20 positives, a random split can hand the test fold 1–2 positives, which is why n_test=28 produced exactly the base rate. Use `StratifiedKFold`, report mean ± sd across folds.
3. **13 features on 140 examples with a 14% positive rate is ~1.5 positives per feature.** Either set `class_weight="balanced"` and select `C` by nested CV, or cut to 4–5 features by univariate signal and bump `FEATURE_SET_VERSION` to 2. The module docstring's own reasoning ("fewer parameters than examples") is the right instinct applied at the wrong ratio — the binding count is *positives*, not rows.
4. **Persist training provenance in the pickle** — `note_ids`, `split_name`, `code_version` — and have `MoLLMCalibrator.load()` return untrained (its existing safe state) when the batch being scored intersects its own training notes. This puts P0.1's leakage control in code rather than in a policy sentence, which matters because §8's standing policy is currently enforced by whoever remembers to type `--calibrator`.
5. Minor: `PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"` is hardcoded at `fit_mollm_calibrator.py` L~54 — derive from `__file__` or read an env var, so the script runs off-EC2.

---

## P4 — Decoding pathology: detect it, don't just penalise it

**Finding:** §4.2 — 475/3990 verdicts (11.9% overall, **~23.8% of every BioMistral verdict ever produced**) degenerate into repetition, always with `raw_confidence_label = None`, always terminating exactly at `MAX_OUTPUT_TOKENS = 800`, always collapsing to `INSUFFICIENT_EVIDENCE`. Fix `FREQUENCY_PENALTY = 0.4` shipped but **unvalidated** (§9.5).
**Why it matters:** these degenerate outputs are currently indistinguishable from genuine `INSUFFICIENT_EVIDENCE`, and they feed the `model_disagreement` hard rule — the single largest HITL category at 69.8%. The report's own 8-case sample found 4/8 disagreements were content-free repetition. The system is routing to humans on the basis of a decoding bug and cannot currently tell how often.

**Change:**
- `src/llm_client.py`: `_is_degenerate(text, finish_reason)` — flag when `finish_reason == "length"` **and** the max repeated-trigram ratio exceeds a threshold. Cheap, no model call.
- On detection: one retry at a higher penalty, then set an explicit `degenerate_generation: True` on the model result and persist it (new `mollm_decisions` column, same additive pattern as P2.3).
- `combine()` / `route()`: a disagreement where one side is flagged degenerate is **not** `model_disagreement` — give it its own `queue_reason` (`degenerate_generation`) so the 69.8% figure decomposes and the fix becomes measurable rather than asserted.
- Add explicit `stop` sequences, and consider vLLM's `repetition_penalty` via `extra_body` (a different mechanism from `frequency_penalty`; both are deterministic, so the reproducibility argument in §3.3 holds for either).
- Re-run the 3-note slice before/after and report the degeneration rate as a number. This is §9.5's own stated next step and should not be skipped.

---

## P5 — Housekeeping with real risk attached

- **Reconcile `evaluation/stage_calibration.py` (2026-08-11) against today's four scripts** (§9.3). Two scripts computing the same quantity by different code is the exact bug class of §3.3's `decoding_mode_mismatch`. Pick one; make the other import from it or delete it with a docstring pointer. Do this *before* P0.2, or you'll unify three ECE implementations and leave a fourth.
- **Finish the `selection_basis` backfill** (§9.4, stopped at 8/32) — Stage 1 is currently the only stage with no accuracy number at all, and `stage1_disambiguation_eval.py` is already written and tested. This is the cheapest unblocked deliverable on the list.
- **Instrument false-deflection rate** — `docs/Evaluation_Criteria.md` names it as a patient-safety metric tracked at each checkpoint. §7's trust-tier table (47.6% REJECTED_GOLD_MISMATCH) is the closest thing in the repo but is a one-off manual eval, not a tracked metric with a pre-set bound. The proposal's success criteria depend on it existing at T0/T1/T2.

---

## Suggested sequencing

| Order | Item | Why first |
|---|---|---|
| 1 | P2.3 provenance columns | Minutes of work; every day without it contaminates more rows |
| 2 | P5 `stage_calibration.py` reconciliation | Must precede any ECE unification |
| 3 | P0.1 splits + P0.2 metrics module + P0.3 bootstrap | Makes every subsequent number citable |
| 4 | P1.1 Tier 1/2 ranker | Largest single accuracy lever; needs P0 to prove it worked |
| 5 | P1.2 uncensor `reasons` | Zero-risk, unblocks the next round of measurement |
| 6 | P4 degeneracy detection | Unblocks an honest read of the 69.8% HITL rate |
| 7 | P2.1 asymmetric override + P2.2 shuffle test | Turns NET VALUE −17 non-negative |
| 8 | P3 calibrator rework | Worth redoing only once the gradable sample grows past n=140 |

**One caveat on the whole list:** P1.1 changes what Stage 2b returns, which invalidates every Stage 3 decision already in `mollm_decisions`. Land P2.3 first so the invalidation is at least *detectable*, and plan for a clean re-run of the validation slice after P1.1 rather than grading old decisions against new candidate lists — which is precisely the confound §6 already flags.
