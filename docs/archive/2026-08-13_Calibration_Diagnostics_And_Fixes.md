# Stage 1–3 Calibration, Diagnostics, and Fixes — Technical Report

> **Model-choice note (2026-08-14):** this doc was written against the earlier two-model vLLM ensemble (BioMistral-7B-AWQ / OpenBioLLM-Llama3-8B-AWQ). The standing ensemble as of 2026-08-14 is qwen2.5:3b / llama3.2:3b / phi4-mini served via Ollama (`src/llm_client.py`) — see `docs/2026-08-14_Stage2_Alias_Fixes_And_Stage3_Provenance.md` §6 for why, and `docs/2026-08-14_Dead_Code_Audit.md` for what else changed alongside it. Design rationale below may still apply structurally; specific model names, base URLs, and context-window numbers do not.

**Date:** 2026-08-13
**Scope:** Clinical Neuro-Symbolic NER Pipeline — Stage 1 (preprocessing), Stage 2a (GLiNER-BioMed extraction), Stage 2b (OMOP concept-linking), Stage 3 (MoLLM ensemble validation)
**Gold standard:** DrivenData SNOMED CT Entity Linking Challenge, `train_annotations.csv` / `train_notes.csv` (272 MIMIC-IV-Note discharge notes, SNOMED-annotated). Confirmed byte-identical (SHA256) between the repo copy and the version supplied directly.

---

## 1. Executive Summary

Today's work had two parts: (1) build the missing per-stage calibration (Expected Calibration Error) infrastructure so the pipeline's accept/reject thresholds are set from measured data rather than hand-picked constants, and (2) act on what that measurement found. The headline result is that **two of the pipeline's three confidence signals do not reliably predict correctness in the range that matters**, and the routing logic has been changed today to reflect that rather than to continue trusting them:

| Stage | Signal | Gradable n | Accuracy | ECE | Verdict |
|---|---|---|---|---|---|
| 1 (disambiguation) | none (deterministic tiebreak) | — | — | N/A | No probability to calibrate — see §5.1 |
| 2a (GLiNER extraction) | `confidence` | 1944 | 89.87% | 0.1437 | Empirically supported — no change needed |
| 2b (OMOP linking, Tier 3) | SapBERT `similarity_score` | 751 | 37.15% (tier avg 37–62% across all tiers) | 0.5016 | Not supported at any threshold — code fixed today |
| 3 (MoLLM ensemble) | `composite_confidence` | 140 | 14.29% | 0.773 | Not supported at any threshold — calibrator required |

Consequently, `src/normalization.py`'s `compute_confidence_tier()` was changed today so that no `match_tier` value (Exact, Synonym, Semantic, or Failed) any longer exempts an entity from Stage 3 review, and the standing policy for any future Stage 3 batch run is that it must be invoked with `--calibrator models/mollm_calibrator_v1.pkl`, never on raw `composite_confidence`.

---

## 2. Pipeline Recap

```
Stage 0 (raw note ingestion)
  -> Stage 1  (src/preprocessing.py)       scispaCy tokenization, PyRuSH sentence
                                            segmentation, abbreviation expansion
  -> Stage 2a (src/entity_extraction.py)   GLiNER-BioMed extraction, assertion,
                                            relations
  -> Stage 2b (src/normalization.py)       4-tier OMOP concept-linking cascade
                                            (Exact / Synonym / Semantic-SapBERT / Fuzzy)
  -> Stage 3  (src/mollm_ensemble.py)      BioMistral + OpenBioLLM ensemble
                                            validation (contradiction-check /
                                            resolution / non-asserted-check modes)
```

Composite key used throughout for joining tables across stages: `(note_id, original_text, expanded_text, entity_label)` — not `entity_id` — because of a documented `normalized_entities` `UNIQUE` constraint fan-out (see `scripts/score_gold_recall.py` module docstring).

---

## 3. Code Changes Implemented Today

### 3.1 Stage 1 — `src/preprocessing.py`, `src/entity_extraction.py`

**`_select_by_groundability()` (new function, `src/preprocessing.py`).** Third-tier fallback tiebreak for ambiguous abbreviations, used only when numeric-context selection (`_select_by_numeric_context()`) does not apply or does not resolve. Prefers a candidate meaning that grounds to a real OMOP concept over one that does not, but only when *exactly one* candidate grounds — if zero or both ground, it declines and falls through to the alphabetical default, deliberately not resolving an ambiguity it cannot adjudicate.

- **Root cause fixed:** `"fx"` → `{"fracture", "fractions"}`, where `"fractions"` sorted before `"fracture"` alphabetically (`i` < `u`) despite being the clinically wrong reading — the prior alphabetical-default tiebreak was clinically arbitrary.
- **Verified:** 6 stub tests (isolated unit tests for grounded/ungrounded/both-grounded/`conn=None` cases, plus 2 end-to-end integration tests through `expand_text_and_track_offsets()` with a hand-written fake spaCy tokenizer). All passed.
- **Live diagnostic confirmation:** Both `"fracture"` (OMOP 45876626 / 4217808, exact) and `"fractions"` (OMOP 4081834, exact) are real, distinct OMOP concepts, so the function correctly declines to override — a genuine, unresolvable-by-this-heuristic ambiguity, not a bug in the fix. The two originally-flagged compound-split cases (`"L/clavicular fx"`, `"L L2/transverse process fx"`) still do not split, independently confirmed as a separate, pre-existing vocabulary-coverage gap: `"clavicular fracture"` has zero exact/synonym SNOMED matches at all, since real SNOMED entries are ICD10CM-flavored and heavily qualified (e.g. *"Fracture of unspecified part of left clavicle, initial encounter for closed fracture"*).
- **Baseline check before Stage 3 launch:** combined Stage 1/2 fixes produced a small, real IoU improvement with no regression — macro IoU 0.0816 vs. 0.079 baseline; weighted IoU 0.0986 vs. 0.095 baseline (via `scripts/score_gold_recall.py`).

**`selection_basis` persistence (new, `src/entity_extraction.py`).** `preprocessing.py`'s `expand_text_and_track_offsets()` has computed `selection_basis` (`"numeric_context:<kind>"` / `"omop_groundability"` / `"alphabetical_default"`) per ambiguous abbreviation since the fix above, but the value was discarded before reaching the database — no consumer ever persisted it. Patched:
- `ensure_extracted_entities_table()`: additive `ALTER TABLE extracted_entities ADD COLUMN IF NOT EXISTS selection_basis VARCHAR;`
- `store_entities()`: threaded `selection_basis` through the `processed_entities` dict, the `INSERT` column list, and the `ON CONFLICT ... DO UPDATE SET` clause (35 columns / 35 placeholders, verified by count).
- The read side (pulling `selection_basis` from the `ambiguous_expansions` overlap check) was added at the same call site that already threads `candidate_expansions` through.
- **Verified:** `py_compile` clean.
- **Status:** column exists and will populate on any future extraction run; a full-corpus backfill (re-running `test_pipeline_e2e.py` across all 32 already-processed notes) was started but interrupted by the user at note 8/32 for time — safe to stop mid-run (DuckDB autocommits each note's writes independently, no partial/corrupted rows; notes 1–7 are durably saved, 9–32 remain unbackfilled until resumed). **Not required for the Stage 2a/2b/3 ECE numbers below.**

### 3.2 Stage 2b — `src/normalization.py`

**`weak_match_tier` Tier 1/2 exemption removed (`compute_confidence_tier()`).** Previously, the `weak_match_tier` signal (added 2026-08-12) forced any entity whose Stage 2b `match_tier` was *not* `"1 (Exact)"` or `"2 (Synonym)"` to route to `LOW` confidence tier (i.e., to Stage 3 for review). Tier 1/2 were exempted on the explicit, documented assumption that they are "structurally unlikely to be wrong... the match itself is high-precision" (2026-08-11 tier-gated-exemption rationale).

- **Why changed:** `evaluation/stage2b_cal_eval.py`, run against the 30-note corpus, measured that assumption as false — see §5.3. Tier 1 is 52.71% accurate (776 gradable), Tier 2 is 61.84% (532 gradable). Neither clears any reasonable bar for "skip review."
- **Change:** `if match_tier is not None and match_tier not in ("1 (Exact)", "2 (Synonym)"):` → `if match_tier is not None:`. A single condition change; the two other, narrower Tier-1/2 exemptions in the same function (`high_gliner_risk`, `short_token`/`isupper_abbreviation`/`alnum_mix`) were deliberately left untouched, since `weak_match_tier` now fires unconditionally for any known tier and the tier decision is an OR across all `reasons` — those other exemptions become harmless no-ops rather than needing a parallel rewrite, keeping this a single, auditable, time-boxed change.
- **Verified:** `py_compile` clean. 6 stub tests confirmed: Tier 1/2/3/0 all now correctly fire `weak_match_tier` and route `LOW`; `match_tier=None` is unchanged (does not fire, preserving the existing conservative "we don't know" default); a fully clean entity with no known tier and high GLiNER confidence is now the only remaining path to a `HIGH` (Stage-3-skipping) tier.
- **Practical effect:** essentially no entity with a resolved OMOP link can skip Stage 3 review anymore, since no measured tier value earns that trust.

### 3.3 Stage 3 — `src/llm_client.py`, `src/mollm_ensemble.py` (imports only), `scripts/run_stage3_batch.py`

**`FREQUENCY_PENALTY = 0.4` added (`src/llm_client.py`).** Root-cause fix for a corpus-wide reliability bug (see §4.2). Applied to both models' `base_kwargs` alongside the existing `TEMPERATURE = 0.0`. `frequency_penalty` is a deterministic token-selection penalty, not a source of randomness, so the reproducibility guarantee that motivated greedy decoding (`TEMPERATURE=0.0`, "same entity + same evidence → same verdict on a re-run") is preserved.
- **Verified:** `py_compile` clean. Fix implemented but not yet re-validated against a fresh batch run at the time of writing (next step: clear the 3-note test slice's `mollm_decisions` rows and re-run to confirm the degenerate-generation rate drops).

**`decoding_mode_mismatch` bugfix (`evaluation/cal_eval.py`).** Pre-existing defect, found proactively while building today's evaluation scripts, not user-reported. The column was computed in-memory (`mollm_ensemble.py`, `combine()`) but never added to `mollm_decisions`'s schema (`CREATE TABLE` / `ALTER TABLE` / `INSERT` lists) — `cal_eval.py`'s original `load_gradable_decisions()` `SELECT` would have raised on any real database. Fixed by removing the column from the `SELECT`/`cols` list and defaulting `_decoding_purity()`'s `mismatch` parameter to `False`, relying solely on the `decoding_modes` JSON list check. Verified `py_compile` clean; confirmed working against the real EC2 database.

**`--calibrator` CLI argument (`scripts/run_stage3_batch.py`).** Loads a `MoLLMCalibrator` (`.pkl`), passes it into `validate_record()`, prints load success/failure loudly (never silently falls back), and tracks/prints `n_calibrator_scored` in the batch summary. Confirmed working live on EC2 (`calibrator: models/mollm_calibrator_v1.pkl (140 training examples, feature_set_version=1)`). An externally-proposed simpler wiring (hardcoded path, no failure visibility) was reviewed and explicitly declined in favor of this version — two concrete gaps identified: no CLI override path, and no visibility if `MoLLMCalibrator.load()`'s documented silent-failure behavior triggered.

### 3.4 Evaluation infrastructure (all new today)

| Script | Purpose | Verification |
|---|---|---|
| `evaluation/stage2a_cal_eval.py` | Stage 2a extraction-confidence ECE | `py_compile` + stub tests, all passed |
| `evaluation/mollm_trust_tiers.py` | Trust-tier classification (A/B/C/REJECTED) for accepted Stage 3 decisions | `py_compile` + 7 stub cases, all passed |
| `evaluation/stage2b_cal_eval.py` | Stage 2b discrete/continuous reliability + Stage2b-vs-Stage3 cross-tab | `py_compile` + 4 stub logic tests, all passed |
| `evaluation/stage1_disambiguation_eval.py` | Stage 1 tiebreak-method accuracy (post-`selection_basis`-persistence) | `py_compile` + 5 stub tests, all passed |
| `scripts/fit_mollm_calibrator.py` | Fits `MoLLMCalibrator` from `cal_eval.py --emit-training-data` output | `py_compile` + 5 stub tests incl. a hand-written `FakeLogisticRegression` save/load round-trip |

**Pre-existing script found late in the session:** `evaluation/stage_calibration.py` (dated 2026-08-11, predates this session) already covers substantially the same ground as `stage2a_cal_eval.py`/`stage2b_cal_eval.py` — same three-way "different correctness per stage" framing, same "Stage 1 deliberately excluded, no continuous confidence" reasoning. **Not yet reconciled against today's scripts** — flagged as outstanding (§9).

---

## 4. Diagnostic Findings (Root-Caused)

### 4.1 `fx` → `fractions` alphabetical-tiebreak bug
See §3.1. Root cause: alphabetical sort used as a clinical tiebreak is arbitrary and can select a real-but-wrong OMOP concept when multiple candidate meanings are each independently valid strings. Fixed via `_select_by_groundability()`, confirmed working as designed via live diagnostic queries.

### 4.2 BioMistral repetition-loop decoding degeneration
**Measured extent:** 475 / 3990 total per-model verdicts corpus-wide (11.9%) show `reasoning` degenerating into the same sentence repeated 10–15+ times (in one case, a multi-thousand-character garbage digit string inside `cited_evidence.rule_id`), always paired with `raw_confidence_label = None`, always defaulting to verdict `INSUFFICIENT_EVIDENCE`. **100% of these (475/475) are BioMistral; 0 are OpenBioLLM.** Since each decision produces exactly one BioMistral verdict, this is ~23.8% of every BioMistral verdict ever produced in this project.

**Root cause:** `src/llm_client.py` set `TEMPERATURE = 0.0` (greedy decoding, deliberately, for reproducibility) with no `frequency_penalty`/`repetition_penalty` anywhere — the textbook precondition for greedy-decoding repetition loops (Holtzman et al. 2019). The failure mode always terminates exactly at `MAX_OUTPUT_TOKENS = 800`, consistent with the model looping until hitting the output cap rather than reaching a natural stop token.

**Concrete example set (3 fresh, never-before-processed notes, 8 raw model_disagreement cases inspected):** of 8 disagreements, 4 (`biliary pancreatitis`, `nasojejunal feedings`, `GI tract`, `RA`) were this degenerate pattern — OpenBioLLM answered each correctly and confidently on entities explicitly stated in the note text, while BioMistral's "disagreement" was content-free repetition, not genuine second-guessing. The other 4 (`postnecrotic pseudocyst`, `CVA`, `MI`, `PULM`) were genuine substantive disagreements with coherent, non-degenerate reasoning on both sides.

**Fix:** `FREQUENCY_PENALTY = 0.4` added, §3.3. **Not yet re-validated on a fresh batch.**

**Practical significance:** on the same 3-note fresh batch (86 decisions), 60/86 (69.8%) were routed `HITL_REQUIRED` via the `model_disagreement` hard safety rule — the single largest blocking category, ahead of `citation_verification_failed` (15/86, 17.4%), `verdict_insufficient_evidence` (6/86, 7.0%), and `below_confidence_threshold` via the calibrator (5/86, 5.8%, correctly conservative given the calibrator's learned ~13% base rate). If a meaningful fraction of that 69.8% is decoding-pathology-driven rather than genuine disagreement, this fix has real potential to reduce spurious HITL routing — but this is a routing-volume effect, separate from and not a fix for the accuracy problem in §5.4/§6.

### 4.3 No enforced train/validation/test split
`docs/Evaluation_Criteria.md` specifies a three-way split of the 272-note corpus: ~70 notes locked for final T0/T1/T2 benchmark comparison against Clinical-T5, and a validation slice carved from the remaining ~200 used solely to calibrate MoLLM's confidence thresholds. **No file anywhere in the repository enumerates which notes belong to which bucket.** Every evaluation script — including `evaluation/stage_calibration.py` (pre-existing, 2026-08-11) and all four scripts written today — defaults `--note-ids` to "every note_id currently present in the database," with the burden of manually respecting the split left entirely to whoever runs the script. `stage_calibration.py`'s own docstring already flags this gap explicitly. **Practical consequence:** today's ECE numbers are computed over an ad hoc, opportunistically-grown note pool that likely overlaps with notes used to derive/validate today's own fixes (e.g. the `fx` groundability heuristic) — a leakage risk the proposal's split design specifically exists to prevent. **Not resolved today** (§9).

---

## 5. Calibration Results (ECE per Stage)

**Methodology (all stages, identical formula — Guo et al. 2017 equal-width-bin ECE):** 10 bins of width 0.1 over confidence ∈ [0, 1]. Per bin: `mean_confidence` (average stated confidence) and `accuracy` (fraction actually correct). `ECE = Σ (bin_size / N) × |accuracy − mean_confidence|`. 0 = perfectly calibrated; higher = confidence is systematically misleading. What differs between stages is only what feeds in as `confidence` and what counts as `correct` — detailed per stage below. Canonical implementation: `evaluation/cal_eval.py`'s `compute_ece()`, reused directly by `stage2a_cal_eval.py`, reimplemented identically (not reinvented) in `stage2b_cal_eval.py` and `fit_mollm_calibrator.py`.

**Corpus used for the numbers below:** 32 notes with `is_test=TRUE` rows in `extracted_entities` (one, `10000032-DS-21`, has zero gold annotations and is excluded from grading — 31 gold-comparable notes effective for Stage 2a/2b).

### 5.1 Stage 1 — No ECE (by design)

Abbreviation-expansion disambiguation (`src/preprocessing.py`) is deterministic rule-based tiebreak logic (numeric-context match → OMOP groundability → alphabetical default), not a probabilistic model — it never outputs a confidence score, so there is nothing to bin against accuracy. The analogous, correct question — does the tiebreak method that resolved an ambiguity actually predict downstream correctness — is answered by `evaluation/stage1_disambiguation_eval.py` (discrete accuracy per `selection_basis` value), but this requires the `selection_basis` backfill (§3.1) to complete, which was interrupted for time. **Not yet run to completion.**

### 5.2 Stage 2a — GLiNER-BioMed Extraction Confidence

- **Population:** 1944 in-scope entities (Anatomy/Condition/Procedure/Symptom — gold has no Medication/Lab Test annotations, so those are reported separately, never scored as "incorrect"). 1307 entities excluded as out-of-gold-scope; 125 excluded for notes with no gold annotations.
- **Accuracy:** 1747 / 1944 = **89.87%**
- **ECE = 0.1437**

| Confidence bin | n | Mean confidence | Accuracy |
|---|---|---|---|
| [0.5, 0.6) | 423 | 0.5468 | 85.11% |
| [0.6, 0.7) | 360 | 0.6496 | 89.44% |
| [0.7, 0.8) | 339 | 0.7489 | 88.50% |
| [0.8, 0.9) | 338 | 0.8521 | 89.94% |
| [0.9, 1.0) | 484 | 0.9518 | 95.25% |

(No entities below 0.5 — `EXTRACTION_THRESHOLD=0.5` already filters those out upstream; this is an enforced floor, not missing data.)

**Per-label breakdown:** Anatomy 90.3% (n=517), Condition 88.6% (n=771), Procedure 88.7% (n=222), Symptom 92.2% (n=434).

**Reading:** mildly *under*-confident throughout (accuracy consistently at or above stated confidence) — the safe direction to be miscalibrated in. `EXTRACTION_THRESHOLD=0.5` is empirically supported; no case to move it.

### 5.3 Stage 2b — OMOP Concept-Linking (Match Tier / SapBERT Similarity)

**Discrete reliability by match tier** (accuracy = crosswalked SNOMED code matches an overlapping gold annotation's `concept_id`):

| Tier | n | Correct | Accuracy |
|---|---|---|---|
| 1 (Exact) | 776 | 409 | **52.71%** |
| 2 (Synonym) | 532 | 329 | **61.84%** |
| 3 (Semantic) | 751 | 279 | **37.15%** |
| 0 (Failed) | 162 | 0 | 0.00% (definitional — unmapped can never match gold) |

**Note:** Tier 1 ("Exact") underperforming Tier 2 ("Synonym") is not a fluke of this sample — independently cross-validated against `score_gold_recall.py`'s own earlier per-note breakdown (e.g. note `10371195-DS-9`: Tier 1 was 9/28 = 32.1% correct, Tier 2 was 16/19 = 84.2% correct). Likely cause: Tier 1/2 lookups break ties among multiple exact-string-matching concepts via `ORDER BY concept_id ASC LIMIT 1` — an arbitrary tie-break, not a clinical one, when a short/common string legitimately maps to several distinct SNOMED concepts (same root-cause class as previously documented `"Left"`/`"Initial"` collision cases).

**Tier 3 continuous ECE = 0.5016** (n=751):

| Similarity bin | n | Mean similarity | Accuracy |
|---|---|---|---|
| [0.7, 0.8) | 198 | 0.7668 | 18.69% |
| [0.8, 0.9) | 245 | 0.8548 | 29.80% |
| [0.9, 1.0) | 308 | 0.9561 | 54.87% |

**Reading:** direction is correct (higher similarity → higher accuracy) but magnitude is badly overconfident — even similarity ≈ 0.96 corresponds to barely better than a coin flip. No similarity value in the observed range supports auto-acceptance. `TIER3_SIMILARITY_FLOOR=0.72` cannot be fixed by raising it; the signal itself doesn't discriminate well enough at any point on this curve. Combined with the Tier 1/2 finding above, **no Stage 2b tier currently clears a reasonable bar for skipping downstream review** — the basis for the `compute_confidence_tier()` fix in §3.2.

### 5.4 Stage 3 — MoLLM Ensemble (`composite_confidence`)

- **Resolution-mode decisions found:** 798. Gradable (verdict + candidates + overlapping gold span): **140**. Ungradable breakdown: `model_disagreement` 585, `no_overlapping_gold_span` 67, `chosen_candidate_uncrosswalked` 6.
- **Ungraded entirely** (no gold label exists for contradiction/non-asserted-check modes — needs a Stage 5 human re-audit sample): `non_asserted_check` 142, `contradiction` 981.
- **Decoding purity (gradable subset):** 140/140 used guided decoding — the numbers below are not contaminated by unguided fallback.
- **Accuracy: 20 / 140 = 14.29%**
- **ECE = 0.773**

| Confidence bin | n | Mean confidence | Accuracy |
|---|---|---|---|
| [0.8, 0.9) | 12 | 0.8892 | 33.33% |
| [0.9, 1.0) | 128 | 0.9184 | **12.50%** |

**Reading:** inverted — the higher-confidence bin is *less* accurate than the lower one, and virtually the entire gradable population (128/140) sits in the top bin. Threshold sweep is flat at 14.3% precision from t=0.05 through t=0.85 (i.e. `AUTO_VALIDATE_THRESHOLD=0.85` currently admits a stream that is ~86% wrong); only moves at t=0.95 (50.0% precision, but n=4, 2.9% coverage — not a usable operating point).

**Self-reported confidence vs. actual accuracy, per model:** BioMistral HIGH: 16/115 = 13.9% correct. OpenBioLLM HIGH: 19/136 = 14.0% correct; LOW: 1/4 = 25.0% correct. Self-reported confidence carries no discriminative signal either.

**Note:** this n=140 sample is identical across every run this session that measured it (no new resolution-mode decisions were generated between measurements) — not independently re-confirmed on fresh data, though the 3-fresh-note cross-tab in §6 is a separate, smaller, genuinely-new-data check that points the same direction.

**MoLLM-Cal calibrator fit** (`scripts/fit_mollm_calibrator.py`, `models/mollm_calibrator_v1.pkl`): label balance 20/140 correct (14.3%). Held-out check (80/20 split, n_test=28): accuracy 85.71%, but flagged as a likely majority-class-prediction artifact — mean predicted P(correct) = 0.1274, and 24/28 = 85.71% exactly matches the base rate. Held-out ECE = 0.0299 (large nominal improvement over raw 0.773, but not verified as genuine discrimination given the artifact flag above). Saved on 100% of the n=140 data. **Wired into `run_stage3_batch.py` via `--calibrator` and confirmed working live** — on the 3-fresh-note batch, the calibrator was consulted for all 86 decisions but only became the deciding routing factor for 5 (the rest were blocked earlier by hard safety rules); all 5 it decided were correctly routed to HITL, consistent with its learned conservative posture.

---

## 6. Stage 2b vs. Stage 3 Value-Add (Does MoLLM Improve on Stage 2b Alone?)

Cross-tabulates Stage 2b's own top-1 correctness against MoLLM's resolution-mode verdict on the same entity (composite key join).

**Full 31-note corpus** (134 entities gradable on both sides; 642 ungradable on at least one side; 22 MoLLM decisions with no matching Stage 2b row — flagged as unexpected, not yet investigated):

| Category | n | % | Meaning |
|---|---|---|---|
| CONFIRMED_CORRECT | 6 | 4.5% | Stage 2b already right, MoLLM agrees |
| CAUGHT_AND_FIXED | 13 | 9.7% | Stage 2b wrong, MoLLM fixed it — MoLLM's entire reason for existing |
| INTRODUCED_ERROR | 30 | 22.4% | Stage 2b right, MoLLM broke it — should be ~0 |
| MISSED_ERROR | 85 | 63.4% | Stage 2b wrong, MoLLM did not fix it |

**NET VALUE (CAUGHT_AND_FIXED − INTRODUCED_ERROR) = −17.** MoLLM's resolution-mode ensemble is currently net-harmful on the measured, gradable slice.

**Fresh 3-note replication** (never-before-processed notes, no candidate-list-drift confound possible): 5 gradable — 0 CONFIRMED_CORRECT, 0 CAUGHT_AND_FIXED, 1 INTRODUCED_ERROR (`CK(CPK)`, a lab-enzyme entity — same family as `AST-956`/`AST-909`/`AST-16` in the full-corpus INTRODUCED_ERROR examples), 4 MISSED_ERROR. NET VALUE = −1. Small n, cannot add statistical weight alone, but replicates the same direction and rules out the candidate-list-drift hypothesis as the explanation for the full-corpus result (this slice has none).

**Caveat on interpretation:** `mollm_decisions` has accumulated across multiple Stage 1/2 reruns with no timestamp column to distinguish which decisions were made against which version of the candidate list — some INTRODUCED_ERROR cases could in principle be grading artifacts from candidate-index drift rather than genuine model failures. The fresh-3-note replication (immune to this by construction) showing the same direction weakens, but does not fully eliminate, this caveat for the full-corpus number.

---

## 7. Trust-Tier Classification (Accepted Decisions Only)

`evaluation/mollm_trust_tiers.py`, run on the overnight batch (63 `AUTO_VALIDATED`/`MOLLM_RESOLVED` decisions, i.e. decisions that passed routing and would have been trusted without further review):

| Tier | n | % | Definition |
|---|---|---|---|
| A | 6 | 9.5% | Gold match or human-verified match |
| B | 3 | 4.8% | Cited guideline rule + verified citation |
| C | 24 | 38.1% | Model agreement only, no independent verification |
| REJECTED_GOLD_MISMATCH | 30 | 47.6% | Accepted by routing, but gold disagrees |

**Nearly half (47.6%) of decisions the system would have accepted without human review are gold-confirmed wrong.**

---

## 8. Routing Policy — Current State (End of Day)

| Stage | Threshold constant | Value | Status |
|---|---|---|---|
| 2a | `EXTRACTION_THRESHOLD` (`src/entity_extraction.py`) | 0.5 | Empirically supported — unchanged |
| 2b | `TIER3_SIMILARITY_FLOOR` (`src/normalization.py`) | 0.72 | Not supported at any value — bypassed today: no `match_tier` now exempts an entity from Stage 3 (§3.2) |
| 3 | `AUTO_VALIDATE_THRESHOLD` / `MOLLM_RESOLVE_THRESHOLD` (`src/mollm_ensemble.py`) | 0.85 / 0.60 | Not supported on raw `composite_confidence` — standing policy: every batch run must use `--calibrator models/mollm_calibrator_v1.pkl` |

Net effect of today's changes: nothing auto-approves on a signal that cannot currently back it up. The pipeline defaults to routing through Stage 3 and, from there, predominantly to human review (`HITL_REQUIRED`) — a deliberate, data-grounded posture, not a stalled or degraded one.

---

## 9. Outstanding / Not Completed Today

1. **Task #32** (pre-existing, unrelated to today): compare imported `kg2a_abbreviations` DB table against newly-placed CSVs.
2. **Train/validation/test split** (§4.3): no file exists enumerating the ~70-note locked test set vs. the validation slice `docs/Evaluation_Criteria.md` specifies. All ECE numbers in this report should be read as development-time measurements, not proposal-compliant validation-slice numbers, until this exists.
3. **`evaluation/stage_calibration.py` reconciliation**: pre-existing script (2026-08-11) substantially overlapping today's `stage2a_cal_eval.py`/`stage2b_cal_eval.py` — not yet diffed or reconciled.
4. **`selection_basis` backfill**: interrupted at note 8/32 of 32; Stage 1 disambiguation-accuracy numbers (§5.1) not yet produced.
5. **`FREQUENCY_PENALTY` fix** (§4.2 / §3.3): implemented and unit-verified, but not yet re-validated against a live batch run to confirm the BioMistral degeneration rate actually drops.
6. **22 MoLLM decisions with no matching Stage 2b row** (§6): flagged by `stage2b_cal_eval.py` as unexpected — not yet investigated.
7. **`src/mollm_expert_review.py`** (exploratory, not built): mid-session, an alternative Stage 3 design was discussed — a separate, parallel module where MoLLM reviews all three Stage 2b tiers (not just LOW) as an unconstrained "medical expert," without the three hard safety rules, logged to a new table for comparison against the guardrailed production path. Substantial groundwork was read (record shapes, prompt-assembly helpers, `normalize_entity()`'s full-candidate-list behavior) but no code was written before priorities shifted to the ECE work in this report. Not started as of this writing.
