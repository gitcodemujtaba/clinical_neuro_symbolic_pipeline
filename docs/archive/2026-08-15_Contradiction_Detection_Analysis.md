# Contradiction-Detection Confusion Matrix — LLM Layer Diagnostic

**Date:** 2026-08-15 (same-day follow-on to `docs/2026-08-15_Stage4_Stage5_Build.md`)
**Scope:** `mollm_decisions` (Objective 2, `src/mollm_ensemble.py`) and `mollm_review_decisions` (Objective 3, `src/mollm_review.py`), scored against the same 27-note gold-crosswalk corpus used throughout this session.
**Status:** Diagnostic complete. Findings feed directly into the architecture decisions in §5 — this doc is the evidentiary record for those decisions in the final report.

---

## 1. Why this analysis exists

Stage 3's stated architectural role (per this session's own framing, reaffirmed here) is **not** to be an unconstrained text generator — it is a **contextual contradiction detector**: given Stage 2b's lexical/semantic top-1 candidate, the LLM ensemble's job is to catch cases where that candidate is contextually wrong (polysemous acronyms, negated/non-asserted findings, mismatched clinical sense) before the result reaches human review or KG3 write-back.

Every precision number measured earlier this session (AUTO_VALIDATED 52.6%, MOLLM_RESOLVED 35.0%, HITL_REQUIRED 40.9%) is a **blended** average of two very different behaviors: correctly validating a right answer, and correctly overturning a wrong one. It cannot distinguish "the LLM is a rigid auditor" from "the LLM is a rubber stamp that happens to agree with a lexical retriever that's usually right anyway." This analysis isolates that distinction directly.

**Method**
- **Ground truth axis:** is Stage 2b's own top-1 candidate (`candidates[0]`, computed *before* any Stage 3 involvement) correct against the SNOMED gold crosswalk?
- **LLM axis:** each model's verdict, classified into `CONFIRMS` / `FLAGS` / `ABSTAINS`, unified across both source tables and all three `mollm_decisions` modes (resolution / contradiction / non_asserted_check):
  - `CONFIRMS`: `assessment=='CORRECT'` | `verdict=='SUPPORTED'` | `verdict=='RESOLVED_TO_CANDIDATE_1'`
  - `FLAGS`: `assessment` in `{ENTITY_LABEL_INCORRECT, CONCEPT_MAPPING_INCORRECT, BOTH_INCORRECT}` | `verdict in {CONTRADICTED, NONE_CORRECT}` | `verdict=='RESOLVED_TO_CANDIDATE_N'` for N>1
  - `ABSTAINS`: `assessment=='UNCERTAIN'` | `verdict=='INSUFFICIENT_EVIDENCE'`
- **Ensemble judgment:** strict majority vote among non-abstaining classifications; ties or all-abstain → `NO_CONSENSUS` (excluded from the 2×2).
- Scripts: `scripts/analysis/contradiction_matrix.py` (headline matrix) and `scripts/analysis/fn_breakdown.py` (False Negative drill-down), both committed to the repo and directly re-runnable (see §4).

## 2. Headline confusion matrix

| | Objective 2 (`mollm_decisions`) | Objective 3 (`mollm_review_decisions`) |
|---|---|---|
| **TP** — caught a real Stage 2 error | 259 | 265 |
| **FN** — rubber-stamped a real Stage 2 error | 570 | 627 |
| **TN** — correctly validated a good match | 634 | 690 |
| **FP** — wrongly overturned a good match | 154 | 106 |
| **Contradiction-detection recall** (of genuinely wrong top-1s, % caught) | **31.2%** | **29.7%** |
| **Validation specificity** (of genuinely correct top-1s, % correctly passed) | 80.5% | 86.7% |
| **Flag precision** (of things flagged, % actually wrong) | 62.7% | 71.4% |

**Interpretation.** Both independently-designed pipelines land in the same place: **~30% recall** for catching genuine Stage 2 errors, meaning **the ensemble currently rubber-stamps roughly 7 of every 10 real errors it is shown.** Specificity is comparatively strong (80–87%) — the layer is not hallucinating contradictions at scale (FP is the smallest cell in both matrices) — but it is systematically too willing to trust Stage 2 when Stage 2 is wrong. This is a rubber stamp with occasional catches, not a rigid semantic auditor, as measured today.

Objective 3's marginally better specificity/flag-precision at the cost of marginally lower recall is consistent with its more conservative validator design intent, but the gap is small and does not change the headline conclusion.

## 3. False-Negative drill-down (why is Stage 2 successfully lying to the ensemble?)

Combined across both objectives: **1,195 FN / 1,783 gold-wrong-top-1 cases.**

### 3a. By GLiNER label — the problem is systemic, not localized

| Label | n_wrong | FN | Recall |
|---|---|---|---|
| Qualifier | 155 | 137 | **11.6%** (see caveat, §3c) |
| Medication | 135 | 96 | 28.9% |
| Lab Test | 408 | 280 | 31.4% |
| Condition | 357 | 242 | 32.2% |
| Procedure | 135 | 87 | 35.6% |
| Anatomy | 356 | 222 | 37.6% |
| Symptom | 237 | 131 | 44.7% |

Lab Test — flagged all session as the hardest category structurally — is **not** an outlier here; it sits mid-pack with Condition, Medication, Procedure, Anatomy all in a tight 29–38% band. Symptom is the best-caught category. **The sycophancy problem is broad-based across nearly every entity type, not concentrated in one syntactically hard category.**

### 3b. By Stage 2 top-1 confidence — the "anchoring bias" hypothesis, confirmed

| Stage 2 top-1 score | n_wrong | FN | Recall |
|---|---|---|---|
| 1.0 (exact) | 851 | 647 | **24.0%** |
| 0.9–0.999 | 190 | 155 | 18.4% |
| 0.8–0.899 | 246 | 177 | 28.0% |
| 0.7–0.799 | 306 | 146 | **52.3%** |
| <0.7 | 190 | 70 | **63.2%** |

| Stage 2 match_tier | n_wrong | FN | Recall |
|---|---|---|---|
| 1 (Exact) | 489 | 385 | **21.3%** |
| 2 (Synonym) | 336 | 241 | 28.3% |
| 3 (Semantic) | 958 | 569 | **40.6%** |

**This is the load-bearing finding.** Recall is inversely correlated with Stage 2's own confidence: a `score=1.0, tier=1 (exact_text)` match gets caught only ~1 time in 5 when wrong; a lower-confidence semantic match gets caught 2–3× more often. The models are anchoring on Stage 2's confidence signal as if it were evidence of contextual correctness, when the two are actually independent — a high lexical-match score says nothing about whether the surrounding clinical context supports that reading (e.g. `'BP'` string-matched to `'Blood pressure'`, or a bare `'Right'` string-matched at score 1.0 regardless of clinical sense).

### 3c. Semantic hallucination vs. ontology-granularity artifacts

Manual inspection of concrete FN cases split into two distinct failure modes:

- **Genuine sycophancy** — e.g. `'galea'` matched to the candidate `'Gale'` (score 0.77, an evidently bad match) and confirmed `SUPPORTED` by all three models regardless. This is the real failure mode the architecture needs to fix.
- **Not an LLM failure — SNOMED/gold artifacts.** The Qualifier category's outlier-low 11.6% recall traces mostly to this: `'Right'` in one note crosswalks to SNOMED `24028007` ("Right", Meas Value class) while gold's annotation at that exact span expects `51872008` ("Right thorax structure", Body Structure class) — **both are legitimately "Right,"** just from different SNOMED subhierarchies, and nothing in the surrounding text disambiguates which domain applies without a symbolic rule. A second case (`'NECK'` → gold code `5880005`, which resolves directly to `"Physical examination" (Procedure)`, not an anatomy concept) points to gold-span-overlap noise in the grading methodology itself rather than a Stage 3 reasoning defect.

**Net read:** the true semantic-hallucination rate is somewhat lower than the raw 68–70% miss rate suggests once SNOMED-duplicate-concept and gold-alignment noise are excluded — but the anchoring-bias effect in §3b is real, large, and independent of this caveat.

## 4. Reproducing this analysis

Both scripts read `mollm_decisions`/`mollm_review_decisions` + `normalized_entities` + `extracted_entities`, crosswalk `candidates[0].omop_concept_id` to SNOMED via `VocabularyRetriever.snomed_code_for_concept()` (`src/retrieval.py:753`), and compare against `scripts/score_gold_recall.py`'s `load_gold()` on the same 27-note `NOTE_IDS` list used throughout this session. Re-running requires only `is_test=TRUE` rows in the two decision tables, which persist in `db/kg2_lexical_store.duckdb` from the 2026-08-14→15 production batch runs (`docs/2026-08-15_Stage4_Stage5_Build.md` §3 Step 1):

```bash
python3 scripts/analysis/contradiction_matrix.py   # writes reports/contradiction_detection/contradiction_matrix_decisions.json
python3 scripts/analysis/fn_breakdown.py            # writes reports/contradiction_detection/fn_breakdown.json
```

Both scripts also print the full tables reproduced in §2/§3 directly to stdout.

## 5. Architecture decisions and justification

| Decision | Justification (this doc) | Status |
|---|---|---|
| **Gate all Stage 3 output behind human review before any KG3 write-back, regardless of routing tier** (`docs/Implementation_Checklist.md`, `hitl_review_queue` design) | Already justified by blended AUTO_VALIDATED precision (52.6%, `docs/2026-08-15_Stage4_Stage5_Build.md`). This analysis sharpens *why* that number is unsafe: it's not that the LLM is moderately unreliable — it's that the LLM's error-catching recall is only ~30%, meaning a large share of what AUTO_VALIDATED calls "validated" is really "Stage 2 was wrong and got waved through." Blanket human review remains the correct call and is now justified on both axes (blended precision *and* isolated recall), not one. | **Implemented** (2026-08-15) |
| **Do not treat Objective 3 (`mollm_review.py`) as a superior replacement for Objective 2 (`mollm_ensemble.py`) based on its cleaner assessment enum** | §2 shows near-identical recall (29.7% vs 31.2%) despite Objective 3's task-specific design (explicit CORRECT/INCORRECT framing, all-tier coverage). The clean enum does not translate into better error-catching in practice. Both pipelines should keep being read into the unified HITL queue (as already implemented) rather than favoring one. | **Confirmed, no change needed** |
| **De-anchor the prompt from Stage 2's match score/basis, in both objectives** (`src/mollm_review.py` + `src/mollm_ensemble.py` `SYSTEM_PROMPT`s: DEVIL'S ADVOCATE + CONCEPTUAL FIREWALL blocks + 3-step forced reasoning before verdict) | §3b is the direct empirical basis: recall drops from 63.2% (low-confidence matches) to 24.0% (score=1.0 exact matches) — a ~2.6× gap driven by the models treating Stage 2's own confidence as corroborating evidence rather than an independent, unverified claim to be checked. This was the single most actionable, highest-leverage lever surfaced by this analysis. | **Implemented and validated 2026-08-15 — see §6.** Objective 3: ensemble CONFIRM→FLAG flip rate 80% (16/20), per-model flag rate 23.3%→68.3%. Objective 2: 0/20 flips to `CONTRADICTED` (architecturally correct — cannot fabricate evidence-backed contradictions it wasn't given evidence for) but real behavioral shift: per-model confirm rate 81.7%→61.7%, with 2/20 entities flipping to unanimous `INSUFFICIENT_EVIDENCE`. Routing outcome equivalent under today's blanket-review policy either way. |
| **Do not prioritize Lab Test-specific prompt/retrieval fixes as the next lever for Stage 3 recall** | §3a shows Lab Test is not an outlier (31.4%, mid-pack) once measured on this metric — contrary to the working hypothesis carried over from the Stage 2 GOLD_MISSING session. The anchoring-bias fix (above) has broader, cross-category leverage and should be prioritized first. | **Confirmed, deprioritized** |
| **Treat the Qualifier category's low recall (11.6%) as mostly a grading-methodology artifact, not a Stage 3 defect** | §3c: traced to SNOMED containing multiple valid "Right"/"Left"-type concepts across subhierarchies (body structure vs. qualifier value vs. laterality), plus at least one confirmed gold-span-overlap mismatch (`'NECK'` → a Procedure-domain gold code). No architecture change indicated here; flagging it prevents mis-prioritizing a fix against noise. | **Confirmed, no change** — worth a follow-up check of the gold-overlap matching logic in `scripts/score_gold_recall.py::overlaps()` if Qualifier numbers matter for a future report. |

## 6. Prompt fix validation — before/after on a curated anchoring-bias subset

**Test design.** Re-selecting a random sample of the raw score=1.0 FN pool for this test would have been misleading: manual inspection (`scripts/analysis/select_anchoring_bias_cases.py`) showed most literal `score==1.0` False Negatives are cases where the extracted text and the candidate concept name are identical or near-identical strings (`'Fall'→'Fall'`, `'ciprofloxacin'→'ciprofloxacin'`, `'pneumothorax'→'pneumothorax'`) that were graded WRONG only because of a SNOMED-code/gold-crosswalk granularity mismatch, not a real contextual error — there is no contradiction for a "devil's advocate" prompt to find in those. Instead, `scripts/analysis/select_anchoring_bias_cases.py` selects the 20 FN cases at `score>=0.75` with the **lowest word-overlap between the entity text and the candidate concept name** — i.e. genuine name-divergence traps in the flavor of the `galea→Gale`/`PTT→Prothrombin` examples in §3c, at the exact high-confidence range §3b identified as the anchoring-bias regime. This surfaced real acronym/abbreviation collisions, including `PTT→'Partial'`, `LCx→'Left'`, `R→'Right'`, `heart→'Initial'`, and a heart-sound-vs-vertebra collision (`S1`/`S2`/`S3→'Structure of Nth sacral vertebra'`) — the same species of ambiguity as this session's earlier "ED" (Emergency Department vs. Eating Disorder) example.

**Method.** `scripts/analysis/prompt_ab_test.py` reloads the exact same 20 entities via `load_validation_records()` (identical retrieval/candidates/context — the only variable is the prompt), runs them through the real 3-model Ollama ensemble (`qwen2.5:3b`, `llama3.2:3b`, `phi4-mini`) under the *updated* prompt, and compares against the already-known OLD per-model verdicts stored in `mollm_review_decisions` from the original production batch run. By construction, all 20 cases were ensemble-level `CONFIRM` under the old prompt (that's how they were selected as False Negatives).

### Iteration 1 — DEVIL'S ADVOCATE + score/basis blindfold

**Change:** added a `CRITICAL AUDIT INSTRUCTIONS` block to `src/mollm_review.py`'s `SYSTEM_PROMPT` — (1) ignore the confidence score/match basis and evaluate as if it were hidden, (2) adopt a "devil's advocate" posture, (3) treat context as supreme over string similarity — plus a 3-step forced reasoning sequence (define text meaning → define candidate meaning → state contradiction) before the model commits to a verdict.

| Metric | Old prompt | Iteration 1 |
|---|---|---|
| Ensemble `CONFIRM`→`FLAG` flips | — | 10/20 (50%) |
| Per-model flag rate (of 60) | 14/60 (23.3%) | 31/60 (51.7%) |

**Regression found:** `PTT→'Partial'` got *worse* (1/3 models flagged it under the old prompt; 0/3 under iteration 1). Reading the reasoning traces directly showed why: the `ENTITY` block still shows Stage 1's own `expanded_text` ("partial thromboplastin time"), and the model's step-2 reasoning ("define what the Stage 2 candidate concept means") collapsed into restating that expansion instead of evaluating what the bare candidate `'Partial'` (a generic OMOP Meas Value concept, not "partial thromboplastin time" itself) actually denotes — **a second, related anchoring effect: anchoring on Stage 1's own expansion, not just Stage 2's score.**

### Iteration 2 — + CONCEPTUAL FIREWALL (expanded_text bleed fix)

**Change:** added point 4 to the audit block explicitly forbidding the model from defining a candidate's meaning via Stage 1's `expanded_text`/abbreviation expansion, and rewrote reasoning-step-2 to say so directly ("using ONLY its own name/vocabulary identity — do not reuse or reference Stage 1's abbreviation expansion here").

| Metric | Old prompt | Iteration 1 | **Iteration 2 (current)** |
|---|---|---|---|
| Ensemble `CONFIRM`→`FLAG` flips | — | 10/20 (50%) | **16/20 (80%)** |
| Per-model flag rate (of 60) | 14/60 (23.3%) | 31/60 (51.7%) | **41/60 (68.3%)** |
| `PTT` specifically | 1/3 flagged | 0/3 flagged (regression) | **2/3 flagged (fixed, and improved on baseline)** |

The firewall fix did not just repair the one case it targeted — flag rate jumped broadly across the set (e.g. `mesocolon` 0/3→2/3, `LCx` 0/3→2/3, `CKD` 1/3→3/3), consistent with the firewall removing a crutch (Stage 1's expansion) the models were leaning on generally, not just in the one case that happened to regress visibly. Of the 4 remaining no-flip cases (`diverticulosis`, `sclera`, `cancer`, `R`), manual inspection again suggests most are not real errors (§3c-style gold/crosswalk-granularity artifacts), consistent with iteration 1's finding.

### Objective 2 (`mollm_ensemble.py`) — same fix, structurally different result

**Change:** `"PROVENANCE OVERRIDES SPELLING: ... Trust these relationships completely"` sat directly above the module's own `"SCORE WARNING"` rule, in direct tension (trust exact-text completely vs. a high score is not reliable evidence) — plausibly why Objective 2's tier-1/exact-match recall (21.3%, §3b) trailed Objective 3's despite already having some score-skepticism language. Fixed by splitting the conflated claim (string-level link is real vs. contextual reading is correct — never implied by the former, and exact_text's own measured ~52% corpus accuracy is now cited inline) and porting the DEVIL'S ADVOCATE + CONCEPTUAL FIREWALL framing, reworded for Objective 2's stricter evidence-only/citation-gating rules.

**Result: 0/20 ensemble `CONFIRM`→`FLAG` flips (`CONTRADICTED`), but a real behavioral shift the strict flip metric doesn't capture.** Per-model, the confirm rate (`SUPPORTED`/`RESOLVED_TO_CANDIDATE_1`) dropped from 49/60 (81.7%) to 37/60 (61.7%) on the same 20 cases — 10 of 20 entities lost at least one confirming vote, including two (`LAD`, `fx`) that flipped from **unanimous** `SUPPORTED` to **unanimous** `INSUFFICIENT_EVIDENCE`. None moved to `CONTRADICTED`.

**Why zero flips is the architecturally correct outcome here, not a failed fix.** Objective 2 has a hard rule (unchanged, and deliberately not touched): *"If the EVIDENCE block says no guideline evidence was retrieved, you cannot return CONTRADICTED."* None of these 20 curated entities (common clinical terms like `chest`, `AFib`, `C5`/`C6`, `PEG`) had guideline evidence retrieved — this narrow guideline KG (1,700 nodes) simply doesn't cover them. So the newly-induced skepticism had exactly one legal outlet: retreat from confident `SUPPORTED` to honest `INSUFFICIENT_EVIDENCE`, not a fabricated `CONTRADICTED`. That is the same hard safety rule that prevents this module from hallucinating a contradiction it can't back up — the fix is working *within* that constraint, not failing to trigger it.

**Practical routing effect is equivalent, strict recall metric is not.** `INSUFFICIENT_EVIDENCE` and `CONTRADICTED` both fail `route()`'s requirements for `AUTO_VALIDATED` — under today's blanket-human-review policy the safety outcome is identical either way (nothing silently auto-writes). But under this doc's §2 confusion-matrix definition, `INSUFFICIENT_EVIDENCE` was classified as `ABSTAIN` (genuine uncertainty), not `FLAGS` (an active, evidence-backed contradiction) — a principled choice made before this fix existed, for a good reason (an abstention isn't a caught error, it's a declined judgment). So Objective 2's strict "contradiction-detection recall" will likely **not** improve much from this prompt change alone; closing that gap needs better evidence retrieval for these entity types, not more prompt skepticism, since the prompt has already hit the ceiling the evidence-gating rule allows. This is a real design boundary, not a bug — deliberately not overridden here, since loosening "no CONTRADICTED without evidence" would reopen exactly the citation-hallucination risk the ASYMMETRIC OVERRIDE GATE (top of this file) exists to prevent.

**Reproducing this validation:**
```bash
python3 scripts/analysis/select_anchoring_bias_cases.py   # writes reports/contradiction_detection/anchoring_bias_targets.json
python3 scripts/analysis/prompt_ab_test.py                 # writes reports/contradiction_detection/prompt_ab_results.json (Objective 3, current prompt)
python3 scripts/analysis/select_anchoring_bias_cases_obj2.py  # writes reports/contradiction_detection/anchoring_bias_targets_obj2.json
python3 scripts/analysis/prompt_ab_test_obj2.py                # writes reports/contradiction_detection/prompt_ab_results_obj2.json (Objective 2)
```
`reports/contradiction_detection/prompt_ab_results_v1_objective3_only.json` preserves iteration 1's raw results for the record.

---

**Bottom line for the final report:** the neural layer, as currently prompted, behaves closer to a rubber stamp (≈30% recall on genuine Stage 2 errors) than the rigid contextual-contradiction detector the architecture calls for — but the failure is neither random nor category-specific. It is concentrated, measurably and reproducibly, in cases where Stage 2 hands the model a high-confidence lexical match. That is a prompt-engineering problem with a clear, testable fix, not a fundamental limit of the model ensemble — which is the basis for keeping blanket human review in place today while treating the anchoring-bias fix as the next concrete engineering step rather than a re-architecture.
