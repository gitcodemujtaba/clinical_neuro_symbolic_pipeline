# 2026-08-17 — Crosswalk Fix, Corpus-Wide Status Snapshot, and Flywheel Production Run

Continuation of the same day's `2026-08-17_Phase5_Phase6_Closeout_And_Corpus_Validation.md` and `2026-08-17_Pipeline_Tier_By_Tier_Walkthrough.md`. Covers: the two remaining UI pages, a real test-isolation bug fix, `proof20`'s final validated numbers, a corpus-wide precision/recall snapshot, a SNOMED crosswalk bug found and fixed while investigating it, a significant annotation-schema finding that bug fix surfaced, and the abbreviation flywheel's first real production-data run (in progress as of this writing).

---

## 1. Test-isolation bug fix + two new UI pages (commits `b31e167`, then `4ad6a40`)

`ui/pages/1_🚀_Pipeline_Runner.py` and `ui/pages/4_📊_Evaluation_Metrics.py` were built (both previously 0 lines) — see `Implementation_Checklist.md`'s UI section for what each covers.

While finishing this work, `pytest tests/` failed only when run as a full-directory suite (never standalone): `tests/test_tier_gate_grading.py` hit `AttributeError: module 'duckdb' has no attribute 'connect'`. Root-caused by replaying pytest's actual import order manually (`importlib.import_module` per file, checking `sys.modules['duckdb']` after each) rather than guessing: `tests/test_confidence_tier_reasons.py` imports `_install_stubs()` from `tests/test_tier12_ranking.py` (which fakes `torch`/`transformers`/`duckdb` so importing `src.normalization` doesn't load a real SapBERT model) but never cleaned up afterward — leaving a fake `duckdb` module permanently in `sys.modules` for the rest of the pytest session. Two other files with the identical pattern had already been fixed to track-and-clean-up their own stubs earlier this session; this was a third, previously-missed source reached via an imported helper rather than a local copy. Fixed identically (track which modules were actually stubbed, pop them after the one import that needed them). Full suite: 60/60 passing.

`evaluation/tier_gate_grading.py` (new) factors out the general-purpose, `note_ids`-parameterized `grade_by_tier()` used by the new Evaluation Metrics page, so it isn't a third copy-paste of the per-tier SNOMED-crosswalk grading logic already duplicated across `evaluation/grade_overnight_corpus_run.py` and `evaluation/grade_fresh5_by_tier.py`.

---

## 2. `proof20` — final validated results (20 genuinely fresh notes, zero calibrator training overlap)

Confirmed via `ConsensusCalibrator.trained_on_any_of(note_ids)`: all 20 notes are outside the calibrator's training set.

| Tier | Decisions | Gradable | Correct | Precision |
|---|---|---|---|---|
| TIER_1_AUTO_VALIDATED | 347 | 189 | 166 | 87.8% |
| TIER_1B_CALIBRATED_AUTO_VALIDATED | 148 | 37 | 33 | 89.2% |
| TIER_2_AUTO_RESOLVED | 10 | 2 | 1 | 50.0% (n=2, not meaningful) |
| TIER_3_AUTO_VALIDATED | 40 | 2 | 0 | 0.0% (n=2, not meaningful) |
| TIER_4_ENSEMBLE_SPLIT (shadow) | 734 | 248 | 155 | 62.5% |

AUTO coverage on this batch: 545/1,678 = 32.5%. TIER_1B's 89.2% (37 gradable) is below the 98.0% measured on the calibrator's own held-out validation split during training, but not a clear regression given the sample size (95% CI roughly 79–99%) — a real, honest data point for the next retrain, not proof either way.

---

## 3. Corpus-wide precision/recall snapshot (all `is_test=TRUE` notes, ~57 notes at time of writing)

**Tier distribution / AUTO coverage:**

| Tier | Decisions | Share |
|---|---|---|
| TIER_4_ENSEMBLE_SPLIT | 2,724 | 47.5% (HITL) |
| TIER_5_TRUE_AMBIGUITY | 1,408 | 24.5% (HITL) |
| TIER_1_AUTO_VALIDATED | 1,176 | 20.5% (AUTO) |
| TIER_1B_CALIBRATED_AUTO_VALIDATED | 214 | 3.7% (AUTO) |
| None (unrouted) | 102 | 1.8% |
| TIER_3_AUTO_VALIDATED | 93 | 1.6% (AUTO) |
| TIER_2_AUTO_RESOLVED | 19 | 0.3% (AUTO) |

**AUTO coverage: 1,502/5,736 = 26.2%** — the real gap against the 90% target. TIER_4 + TIER_5 together are 72% of all decisions.

**Precision vs. gold:**

| Tier | Gradable | Correct | Precision |
|---|---|---|---|
| TIER_1_AUTO_VALIDATED | 699 | 596 | 85.3% |
| TIER_1B_CALIBRATED_AUTO_VALIDATED | 56 | 49 | 87.5% |
| TIER_2_AUTO_RESOLVED | 7 | 4 | 57.1% (n too small to trust) |
| TIER_3_AUTO_VALIDATED | 6 | 0 | 0.0% (see §4 — not a straightforward defect) |
| TIER_4_ENSEMBLE_SPLIT (shadow) | 943 | 642 | 68.1% |

**Recall / completeness (Stage 1/2):** 14,542 gold annotations corpus-wide. **Span recall 32.6%, linked recall 16.4%.** Missed spans cluster on physical-exam shorthand in template/list format (`Gen`, `Resp`, `CTAB`, `MMM`, `OP clear`) plus section headers — a specific, addressable Stage 1 (GLiNER) extraction pattern, not diffuse noise. This is the largest lever in the pipeline right now: everything downstream only ever gets a chance to run on the ~33% of gold entities Stage 1 actually finds.

**Ambiguous-abbreviation tiebreak accuracy (abbreviation flywheel motivation):** `omop_groundability` tiebreak wins are correct 53.1% of the time (160 cases); `alphabetical_default` wins are correct only 20.1% of the time (234 cases) — barely better than chance, and the single most actionable number behind Phase 7.

---

## 4. TIER_3's 0% precision — root-caused, not a simple pipeline bug

All 6 gradable TIER_3 cases trace to a **repeated discharge-instruction template** ("*AVOID any blood thinners such as ... Coumadin or Plavix ...*", a negated/hypothetical instruction appearing near-verbatim across ≥3 notes). The pipeline resolves "Coumadin" → RxNorm "warfarin" (the drug substance, the natural reading of the span) while gold codes the same span to `182764009` ("Anticoagulant therapy" — a class/therapy-level concept). This is a genuine annotation-philosophy difference for this specific repeated template, not a pipeline defect on its own — but investigating it surfaced a real, separate, much bigger bug (§5).

---

## 5. SNOMED crosswalk arbitrary-target-selection bug — found and fixed (commit `4ad6a40`)

`src/retrieval.py`'s `VocabularyRetriever.snomed_code_for_concept()` — the crosswalk used by every Medication-domain precision/recall measurement in this project (`evaluation/*.py`, `scripts/score_gold_recall.py`) — picked its SNOMED target via `ORDER BY concept_id ASC LIMIT 1` across every matching relationship row, with no preference for relationship type or domain. A single RxNorm concept can carry many legitimate relationship rows to SNOMED spanning different domains: RxNorm "warfarin" (concept_id 1310149) has 18, including a Drug-domain product concept, an Observation-domain allergy-history concept, and Procedure-domain "warfarin therapy"/"warfarin prophylaxis" concepts. The old logic landed on the Procedure concept instead of the drug product.

**Fixed** to prefer, in order: (1) the `RxNorm - SNOMED eq` relationship type (an explicit equivalence mapping) over the vaguer `Mapped from`/`Value mapped from`; (2) a SNOMED domain matching the source concept's own domain; (3) lowest `concept_id`, purely for determinism.

**Measured impact:** 153/314 (48.7%) of distinct Medication-domain concept_ids chosen across the full corpus now crosswalk to a different SNOMED code than before. New `tests/test_snomed_crosswalk.py` reproduces the real multi-relationship shape against a throwaway in-memory DuckDB connection. Full suite: 60/60 passing after this fix too.

**Important: this did NOT move aggregate Medication precision/recall numbers** — see §6.

---

## 6. Medication gold-annotation-schema mismatch — a real, structural finding (not fixed, documented)

Investigating why the crosswalk fix didn't move Medication precision revealed a systematic pattern: this gold corpus (SNOMED-CT entity linking challenge data) codes medication mentions to **Procedure-domain "Administration of X" / "X therapy" concepts**, not the drug substance/product. Sampled directly: 11/14 gold codes for gradable Medication entities were exactly this shape (`Administration of aspirin`, `Heparin therapy`, `Continuous infusion of heparin`, `Warfarin therapy`, etc.). The pipeline resolves medications to the drug substance (the natural reading of the span) — a genuine annotation-schema difference from gold's convention, not a pipeline defect. Corpus-wide gradable Medication precision measured 31.5% (17/54) as a direct consequence.

**Decision: do not chase this via further crosswalk tuning or by making the pipeline resolve to administration-event concepts instead of the drug substance** — that would be a large, clinically-questionable architecture change purely to match one dataset's annotation convention. If Medication-domain precision needs to look better for a specific audience, the honest fix is reporting it separately with this caveat stated, not patching the pipeline. Recorded as project memory (`medication-gold-annotation-mismatch-2026-08-17`, `snomed-crosswalk-arbitrary-target-fix-2026-08-17`) so this isn't re-discovered from scratch.

---

## 7. Abbreviation flywheel — first real production-data run (in progress)

Phase 7 (`src/abbreviation_flywheel.py`, commit `c33ab25`) landed while `proof20` already held the DB write lock running old code, so `abbreviation_observed_expansions` never received real production data. This session launched the first real population run:

- **Scope: 50 notes from the `train` split only** (`data/splits/note_splits.csv`), explicitly avoiding the `test` split (marked "LOCKED — final benchmark only" in that file's own header). 31 of the 57 already-processed `is_test=TRUE` notes turned out to be from the locked test split — populating a ledger that later *live-influences* pipeline decisions (via `compute_frequency_priority()`) using observations drawn from the benchmark set would be the same leakage risk `ConsensusCalibrator.assert_not_trained_on()` already exists to prevent for the calibrator. 124 unprocessed train-split notes were available; 50 were selected.
- **Stage 1→2b only, no Stage 3 ensemble** — `record_ambiguous_expansion_outcome()` fires during Stage 2b (`orchestrator.py`), gated on `match_tier != "0 (Failed)"`, with no dependency on Stage 3's routing at all. Running via `scripts/test_pipeline_e2e.py` (Stage 1→2a→2b only) is both sufficient and much cheaper than the full tier-gate ensemble.
- **Operational notes**: this worktree (`clinical_neuro_symbolic_pipeline_reorder`) never received the large untracked raw-note CSVs (`data/raw_notes/discharge.csv`, `gold_notes.csv` — both only exist in the sibling `clinical_neuro_symbolic_pipeline` worktree). Used `data/snomed-ct-entity-linking-challenge-1.2.0/train_notes.csv` instead (`--input` override), which has the identical `note_id`/`text` shape and covers the full 272-note gold population. Also hit and resolved a DB-lock conflict from the Streamlit UI's cached `st.cache_resource` read-only connection blocking a new writer — DuckDB's single-writer/multi-reader lock blocks a new writer even against an idle *read-only* connection; restarted the Streamlit process to release it.

**Locked post-run checklist** (to execute once the batch completes):
1. `SUM(hit_count)` aggregation query against `abbreviation_observed_expansions`, `HAVING` the real `MIN_FREQUENCY_PRIORITY_SUPPORT >= 3` threshold, checking the real `FREQUENCY_PRIORITY_MARGIN >= 0.20` acceptance rule from `compute_frequency_priority()`.
2. Gold spot-check the top-volume winners against `scripts/score_gold_recall.py`'s `load_gold()` for these 50 note_ids.
3. Confirm the "Bucket 1 vs. Bucket 2" prediction directly: dictionary-ambiguous tokens (`BP`, `RA`, `NS`, `LAD`, `DM`, `SGPT` — unverified assumption) should appear in the ledger; tokens that bypass `expansion_ambiguous` entirely because Stage 1 never flagged them as dictionary-ambiguous (`son`, `mild`, `air`, `tylenol` — all diagnosed as Stage 1 extraction/retrieval issues unrelated to abbreviation disambiguation, not flywheel-fixable) should be **absent**.
4. Extend `_ADDITIONAL_BIAS_ABBREVIATIONS` (`src/abbreviation_flywheel.py:62`) only if a high-volume token shows single-expansion ledger dominance but real context-variation in gold.
5. Dry-test `compute_frequency_priority()` in-process on one eligible and one excluded/low-volume token.
6. Grade the 50-note batch against gold; fold into the calibrator's *next* retrain as additive data only — note-disjoint split maintained, fresh held-out validation required before any threshold change (the `0.72` `CALIBRATED_AUTO_THRESHOLD` stays untouched until that validation happens for real; it was bumped from 0.65 specifically because 0.65 let real false positives through on fresh-note testing, not a placeholder waiting to be lowered).

**Explicitly corrected mid-conversation, worth keeping documented:** the flywheel only ever touches entities Stage 1 *already extracted and flagged* `expansion_ambiguous=TRUE` — even for genuinely dictionary-ambiguous tokens, it only affects *which expansion gets chosen*, never the post-expansion Stage 2b retrieval/ranking step. A wrong resolution downstream of a correct (or even non-ambiguous) expansion is a retrieval/embedding-collapse bug, not something either flywheel mechanism can fix — the same failure class already documented for LCX/LAD/short-alphanumeric-code collisions, just occurring at concept-matching rather than abbreviation-expansion.

**Batch completed: 50/50 notes, 8,062 entities, 0 errors, ~4h12m.** Also caught mid-run: the "Bucket 1 vs. Bucket 2" prediction from earlier in the conversation was 3/4 correct (`son`, `mild`, `tylenol` genuinely absent from the ledger as predicted) but wrong on `air` — it turned out `air` *is* a dictionary-ambiguous entry (10 ledger observations, 100% → "autoimmune retinopathy"), contradicting the earlier hypothesis that it bypassed expansion entirely. Worth keeping as a reminder that even a well-reasoned prediction needs the actual check.

**Post-run checklist results (steps 1-2), and the real finding that reshaped the rest of this work:**

The `SUM(hit_count)` ledger query returned 43 distinct abbreviations with ≥3 observations. Gold-checking the 7 highest-confidence, non-bias-excluded winners came back **7/7 wrong**:

| Abbreviation | Ledger's dominant winner | Gold says | Selection basis feeding it |
|---|---|---|---|
| `DM` | "deep masseter" (4/4) | Diabetes mellitus | `alphabetical_default` |
| `IVF` | "In Vitro Fertilization" (7/7) | Administration of IV fluids | `omop_groundability` |
| `air` | "autoimmune retinopathy" (10/10) | Breathing room air | `omop_groundability` |
| `CP` | "cerebral palsy" (6/6) | Chest pain | `alphabetical_default` |
| `SBP` | "spontaneous bacterial peritonitis" (5/6) | Blood pressure finding | `alphabetical_default` |
| `NC` | "neurogenic claudication" (5/5) | Nasal cannula (O2 admin) | `omop_groundability` |
| `ACS` | "Abdominal Compartment Syndrome" (3/3) | Acute coronary syndrome | `alphabetical_default` |

Critically, querying `selection_basis` directly showed `observed_frequency_priority` already appearing for several of these (`air`, `cp`, `dm`, `sbp`, `ivf`, `nc`) — the mechanism had started re-selecting its own earlier wrong guesses *within this same batch run*, since notes process sequentially and each note's outcome feeds the ledger the next note's expansion consults. This is the exact circularity failure mode the module's design was meant to guard against, just far more widespread than the ~20-abbreviation `_ADDITIONAL_BIAS_ABBREVIATIONS` block-list anticipated. A handful of other suspicious high-volume winners (`ED`→"Eating Disorder" 23/23, `NS`→"Noonan Syndrome" 8/8, `PMH`→"progressive macular hypomelanosis" 6/6) were flagged but deliberately **not** gold-checked — the 7/7 failure rate on real data was judged sufficient evidence without spending more verification time.

**Decision: inverted the gate from block-list to allow-list (commit pending).** `src/abbreviation_flywheel.py`'s `compute_frequency_priority()` now requires an abbreviation to be explicitly present in a new `VERIFIED_ALLOW_LIST` (starts **empty** — nothing pre-seeded, since nothing has actually been gold-verified *correct* yet) before it will ever return a ledger-derived answer, regardless of how dominant or high-volume the signal is. `_is_bias_excluded()`/`_ADDITIONAL_BIAS_ABBREVIATIONS` are kept as a redundant second safety layer, not removed. `record_ambiguous_expansion_outcome()` is untouched — the ledger keeps recording unconditionally, now serving purely as a passive diagnostic/audit table for later offline allow-list promotion, since the gate inversion means it can no longer influence any live decision by construction. Live-verified against the real 8,062-entity ledger: `DM`, `CAD`, `SBP` all correctly return `None` post-inversion, including the ones that would previously have fired. 60/60 tests passing (`tests/test_abbreviation_flywheel.py` updated to test both sides of the gate).

The corpus-scale data (CAD, MI, HD, CVA, HLD, "cn ii-xii", "nitro gtt", etc.) that *wasn't* gold-checked remains in the ledger, deliberately not deleted — it's exactly the candidate pool a human would review to populate `VERIFIED_ALLOW_LIST` offline, and once the gate is closed it carries zero live-decision risk either way.

## 8. Calibrator retrain attempt on the 51-note pool (overnight 31 + proof20 20) — a real ceiling found, not a win

Attempted to retrain `ConsensusCalibrator` on an expanded, still note-disjoint dataset: the existing 31-note overnight corpus (`evaluation/grade_overnight_corpus_run.NOTE_IDS`) plus `proof20`'s 20 notes, confirmed zero overlap, 51 distinct notes total. **First finding: this required no new compute** — the 50-note flywheel batch (§7) couldn't be reused for this, since it was deliberately Stage 1→2b only and has zero rows in `mollm_tier_gate_decisions`; the 51-note pool instead reuses two *already-Stage-3-processed* populations.

`evaluation/tier_gate_cal_eval.py`'s `build_labeled_examples()` was parameterized to accept an optional `note_ids` argument (defaults to the original 31, fully backward compatible) rather than duplicating its SQL/labeling logic a second time.

**Results, run as a diagnostic only — nothing saved over the production `.pkl`:**
- 916 labeled `TIER_4_ENSEMBLE_SPLIT` examples, split note-disjoint (39 train notes / 701 examples, 12 val notes / 215 examples).
- **val AUROC: 0.701** — *lower* than the existing 0.74 baseline. More volume did not improve separability.
- Threshold sweep, hard traps ON (production-equivalent routing): 0.60→86.7%, 0.65→88.5%, 0.68→90.0%, 0.70→91.7%, **0.72 (currently locked)→89.5%**, 0.75→88.9%. No band reaches the 98%+ precision that originally justified locking the threshold at 0.72.
- **The real headline finding: the currently-deployed 0.72 threshold itself only measures 89.5% precision on this larger, more diverse validation set** (vs. the ~100% measured on the original 126-example val set that justified it). That original number was very likely a small-sample artifact, not a robust floor.
- Investigated the 0.70→0.72 non-monotonicity (91.7%→89.5%) against a live hypothesis that hard-trap entities (`LCX`/`LAD`/`S2`-shaped) were leaking into the "trapped" sweep's denominator despite being hidden from the diagnostic printout. **Directly verified and disproven**: zero trapped entities appear in the promoted set at any threshold checked when `respect_hard_traps=True`. The real false positives at threshold≥0.70 (traps excluded) are only 3: `vancomycin`, `Hepatocellular Carcinoma`, `swelling` — all sharing a 2-1 `SUPPORTED_1`/`NONE_CORRECT` vote split, but no shared structural pattern the way the coronary-segment or short-alphanumeric-code traps did. With only 19-36 entities promoted per threshold band, the 91.7%→89.5% swing is 1-2 individual entities — real, but not statistically distinguishable from small-sample noise at this volume.

**Verdict: No-Go on lowering the threshold, per the pre-agreed criteria** — no band achieves 98%+ precision. Not deployed; the diagnostic calibrator was strictly worse by AUROC and was never saved over the production model. More importantly, this also means the production 0.72 threshold's safety margin is less certain than previously measured, not just "the experiment to lower it failed." Getting a statistically trustworthy read on the "confident hallucination" hypothesis (whether high-confidence 2-1 splits are systematically riskier) needs more Stage-3-processed volume than 51 notes provide — the same multi-hour compute tradeoff this session repeatedly chose not to spend, now the explicit blocker rather than an assumption.

## Pipeline status & diagnostic summary (closing entry for today)

**The precision ceiling (calibrator).** The `0.72` promotion threshold stays locked, but it is not the ~98% shield it was previously measured to be — on a more diverse, larger validation set it yields ~89.5% precision. The apparent non-monotonicity at the highest confidence bands (0.70→0.72) traces to small-sample noise at the distribution tail (33/36 vs. 17/19 correct), not a new, fixable, or structural failure mode — the coronary/short-code hard-trap leak hypothesis was checked directly and disproven. **Verdict: operating with a real, roughly ~10% imprecision rate at the current threshold on diverse data. Statistically trustworthy threshold tuning needs a genuinely larger Stage-3-processed volume than currently exists — a real compute cost, not a code fix.**

**The abbreviation flywheel (Phase 7).** The pre-existing, unsupervised tiebreakers (`alphabetical_default`, `omop_groundability`) are systematically biased toward rare/textbook expansions over common clinical readings — confirmed 7/7 on real gold-checked data (`DM`→"deep masseter" instead of diabetes mellitus, `IVF`→"In Vitro Fertilization" instead of IV fluids, and 5 more). Left unchecked, the frequency-priority mechanism was actively laundering this bias into its own ledger mid-batch, re-selecting its earlier wrong guesses as if they were independent confirmation. **Verdict: the gate is inverted. `compute_frequency_priority()` now requires explicit human-vetted `VERIFIED_ALLOW_LIST` membership (starts empty) before ever acting; the ledger remains active purely as a passive diagnostic tool for offline allow-list construction, and can no longer influence any live pipeline decision.**

**The recall bottleneck (Stage 1).** The real ceiling on autonomous coverage isn't precision tuning at all — it's Stage 1 entity extraction, measured at 32.6% span recall corpus-wide. Missed entities cluster heavily around non-prose physical-exam shorthand in template/list format (`CTAB`, `MMM`, `Gen`, `Resp`). **Verdict: the flywheel cannot address this by construction — it only ever acts on entities GLiNER already extracted. Unblocking the 32.6% ceiling needs a targeted Stage 1 intervention; what shape that intervention should take (a formatting-aware heuristic, a fine-tuning pass, something else) is an open question this session did not investigate, not a settled recommendation.**
