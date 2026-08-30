# Final Results — Single Source of Truth for the Paper

Consolidated from `docs/2026-08-20_Session_Results_And_Status.md`
(chronological session log), `docs/Implementation_Methodology.md`
(architecture), `docs/Code_Reference_Stages_And_Metrics.md` (formulas),
and `docs/Code_Flow.md` (execution trace). This document is the one to
pull numbers from directly — every figure below states its source,
sample size, and date, and nothing here is reported without a verified
basis. Where a number could not be verified, that is stated explicitly
rather than omitted or estimated.

---

## 1. System Summary

A 5-stage neuro-symbolic clinical NER + entity-linking pipeline: GLiNER-BioMed
extraction → SapBERT/SNOMED tiered retrieval → a 3-model local-LLM
(qwen2.5:3b, llama3.2:3b, phi4-mini) tier-gate ensemble → HITL routing →
KG3 ingestion (dry-run only, no live unreviewed writes anywhere in the
system today). Full architecture: `docs/Implementation_Methodology.md`.
Full execution trace (what calls what, in order): `docs/Code_Flow.md`.

---

## 2. Headline Results — Corpus-Wide vs. Fresh-10 Held-Out

**This is the primary table for the paper.** Computed 2026-08-20 directly
against the live database, twice, deliberately: once over all 144
`is_test` notes (mixed development/held-out vintage) and once over 10
notes from the official locked test split
(`data/splits/note_splits.csv`) not used to debug this session's fixes
and (for 7 of 10) outside the tier-gate calibrator's training set.

| Metric | Corpus-wide (144 notes) | **Fresh-10 (held-out) — recommended headline** |
|---|---|---|
| Gold annotations | 39,403 | 1,497 |
| Span recall (Stage 1/2a) | 53.0% (20,873/39,403) | 49.5% (741/1,497) |
| Linked recall (Stage 1+2b) | 33.5% (13,208/39,403) | 26.8% (401/1,497) |
| Linked precision | 50.0% (13,197/26,382) | 45.3% (401/886) |
| **Linked F1** | 40.1% | **33.7%** |
| Benchmark char IoU (macro / weighted) | 0.1437 / 0.2824 | 0.1453 / 0.2425 |
| **AUTO-tier precision** | 86.9% (5,841/6,724 gradable) | **76.8% (43/56 gradable)** |
| **Deflection rate** (all `AUTO_TIERS` decisions / all Stage 3 decisions) | 57.0% (10,953/19,202) | **31.2% (78/250)** |

**Methodology note (load-bearing — read before citing)**: linked
precision is computed over the exact same population as linked recall
(every prediction with a resolved SNOMED code, across all tiers, checked
against gold for the same note set) — not AUTO-tier-only precision
against full-corpus recall, which would mix two different populations
and make F1 meaningless. Deflection rate uses ALL `AUTO_TIERS` decisions,
not just the clean-span-gradable subset — an earlier draft of this
comparison had a real bug (gradable-restricted numerator over an
unrestricted denominator) that undercounted deflection by ~20–26 points;
caught, reconciled against the raw tier distribution, and corrected
before this version.

### Why corpus-wide is higher on every metric — a root-caused finding, not a hand-wave

Broke the 25.8pp deflection gap down by tier:

| Tier | Corpus % | Fresh-10 % | Gap |
|---|---|---|---|
| `TIER_1_AUTO_VALIDATED` (unanimous 3-model ensemble) | 36.1% | 30.0% | +6.1pp |
| `TIER_4_ENSEMBLE_SPLIT` | 27.7% | 46.4% | −18.7pp |
| **`TIER_3_AUTO_VALIDATED`** (deterministic fast path, zero LLM calls) | **18.4%** | **1.2%** | **+17.2pp** |
| `TIER_5_TRUE_AMBIGUITY` | 13.1% | 19.6% | −6.5pp |
| `TIER_1B_CALIBRATED_AUTO_VALIDATED` | 3.0% | 0.0% | +3.0pp |
| `TIER_2_AUTO_RESOLVED` | 1.4% | 2.0% | −0.6pp |

**`TIER_3` alone accounts for two-thirds of the gap.** `TIER_3` is the
curated fast path (exact matches, the 26 `_LAB_TEST_ALIASES` entries,
brand aliases, cold-start dictionaries) — built by mining terms
encountered while debugging specific notes this session. It fires on
almost nothing in fresh-10 (1.2% vs. 18.4%). The actual 3-model LLM
ensemble (`TIER_1`) only differs by 6.1pp — it generalizes reasonably
well. **Conclusion for the paper**: the gap is specifically "curated
dictionaries don't yet cover fresh vocabulary," not "the pipeline doesn't
generalize" — a narrower, more actionable, more honest claim, and the
correct target for future work (widen the dictionaries) rather than
further ensemble/calibrator tuning.

---

## 3. Annotation Velocity & Cost-Effectiveness

**Manual baseline**, source-verified (Wei, Q., Franklin, A., Cohen, T., &
Xu, H. (2018). "Clinical text annotation — what factors are associated
with the cost of time?" *AMIA Annual Symposium Proceedings*): 9 clinician
annotators, 2010 i2b2/VA dataset, 6,663 sentences, problems/treatments/tests
categories (directly comparable to this project's Condition/Medication/
Procedure/Lab Test labels) — measured **40.47–92.22 words/minute**. This
project's corpus: 272 notes, mean 1,554.1 words/note.

| Configuration | Manual (range) | Pipeline | Ratio |
|---|---|---|---|
| Extraction + linking only (Stage 1→2b) | 16.85–38.41 min/note | 5.71 min/note | 2.95×–6.73× faster |
| Full pipeline incl. tier-gate ensemble (Stage 1→3) | 16.85–38.41 min/note | 21.33 min/note | 0.79×–1.80× — speed advantage largely disappears |

**Re-verified 2026-08-20 against real, directly-observed timing from this
session's own runs** (not just re-quoted): Stage 3 entities/note (mean
132.7, median 134, 144 notes) × real observed per-entity Stage 3 pace
(7.9–9.4 sec/entity, from two live 125-entity batches today) ≈ 17.6–21.0
min/note for Stage 3 alone; + Stage 1→2b (3.1–3.8 min observed on four
real fresh full-pipeline extractions today, shorter notes) ≈ **21–27
min/note full pipeline** — consistent with, or slightly above, the
original 21.33 figure. **The number is not an overstatement**; if
anything the honest re-verified range points higher.

**Honest framing for the paper**: the correct claim is NOT raw speed for
the full pipeline (that advantage largely disappears once the LLM
ensemble is counted — up to 18 LLM calls/entity is genuinely more
compute-intensive than a human reading and clicking once). The
supportable claim is **effort reduction**: 31.2–57.0% of entities never
require human review at all (see §2), cutting required expert review
volume substantially, even though the pipeline itself isn't "fast" once
verification is included.

**Gap, stated honestly**: no false-deflection rate exists — `hitl_review_queue`
has real cases now (19,103 total, 250 in the fresh-10 scope, populated
2026-08-20) but zero completed human reviews, so the patient-safety metric
the proposal names cannot be computed yet. No T0→T2 deflection-rate trend
exists — every number above is a single point-in-time read.

---

## 4. Benchmark Comparison (DrivenData SNOMED-CT Entity Linking Challenge)

Official metric definition, confirmed by reading the benchmark's own
metric section directly: "class" = SNOMED concept ID; a predicted span's
characters only count toward a class if its own resolved concept matches
exactly (relationships between concepts are not scored).

| | Macro char IoU | Weighted char IoU |
|---|---|---|
| Earlier this session (pre extraction-recall fix) | 0.1431 | 0.2400 |
| **Corpus-wide, current (144 notes)** | **0.1437** | **0.2824** |
| Fresh-10 held-out, current | 0.1453 | 0.2425 |

At the earlier measurement, this ranked **last (macro) and second-to-last
(weighted)** against the 5 public leaderboard entries available for
comparison. The weighted score has improved meaningfully since
(0.2400→0.2824); macro is essentially flat. **Not re-checked against the
current public leaderboard positions** — the ranking comparison above
reflects the earlier measurement's relative position, not a re-run
against updated competitor numbers.

Root cause of the low absolute score, corpus-wide miss analysis (140
notes, 38,689 gold annotations): **Stage 2a extraction recall was only
46.1%** at baseline — char IoU requires actual character overlap, so
roughly half of gold's characters were mathematically unreachable before
linking quality even entered the picture. 100% of misses were true
zero-shot blind spots (GLiNER proposed nothing at any confidence), ruling
out threshold tuning. Two extraction cold-start fixes
(`src/lab_abbrev_coldstart.py`, `src/narrative_state_word_coldstart.py`)
raised extraction recall **46.4% → 53.2% (+6.8 points)**, gold-verified,
21 lab-abbreviation + 9 narrative-state-word terms, each independently
checked for ≥95% single-concept consistency before inclusion.

---

## 5. SNOMED Near-Duplicate Retrieval Fix (2026-08-20)

**Problem**: MCHC/RDW (and other lab abbreviations) were losing Tier 2
ensemble votes to SNOMED regional-extension concepts instead of the
correct International Release concept.

**Root cause, verified against live data before fixing**: `vocabulary_id`
cannot discriminate International Release from regional extensions (only
one distinct value exists in the whole 1.09M-row `athena_concept` table).
`concept_class_id` alone is unreliable (23,842 extension concepts are
themselves `'Procedure'` class). The real signal is the SCTID's own
namespace-identifier block — extension concepts carry a reserved
`1000000` segment in `concept_code` that International Release concepts
never have. Confirmed: 98,487 of ~1.09M concepts (9%) match this pattern,
and **zero of 4,522 distinct gold-standard SNOMED codes** are among
them — excluding it cannot cost a true match.

**Fix**: `concept_code NOT LIKE '%1000000___'` in every open-ended Tier
1–4 retrieval query, plus extending the existing Procedure-class
preference rule to also cover `Qualifier Value` class (a second,
distinct near-duplicate pattern the namespace fix alone surfaced — 22
SNOMED "X calculation technique" concepts, also zero gold overlap).

**Result, gold-verified on MCHC + RDW (28 gradable entities)**: **100%
(28/28)** of post-fix candidate pools now surface the gold-correct
concept as the top SapBERT candidate — a complete fix at the retrieval
layer. Did not (yet) increase the auto-write rate for this specific
population, for an independent, pre-existing reason: the
`short_alphanumeric_code_trap` safety gate forces short lab codes through
the full ensemble regardless of retrieval quality. One case that did
reach unanimous ensemble agreement picked a **third**, previously
unexamined near-duplicate ("Mean cell hemoglobin concentration -
finding", Clinical Finding domain) — a real, open gap, not fixed this
session.

Full technical detail and code: `docs/Code_Reference_Stages_And_Metrics.md`
§ Stage 2b.

---

## 6. KG Embedding (TransE) Results

**Built**: real TransE (Bordes et al. 2013, plain PyTorch, no external
KGE library) on the SNOMED subgraph this pipeline's candidate pools
touch — 7,269 concepts, ~24,900 edges, 104 relation types.

**Intrinsic evaluation** (standard KGE protocol, RAW setting): **MRR
0.776, Hits@10 0.909** — 2,000 of 2,493 held-out test triples evaluated
(a random subsample, not the first N; `evaluate_link_prediction()` caps
at `max_eval=2000` since ranking each triple's true tail against the
full 7,269-entity set is the expensive part of the intrinsic protocol).
Verified directly against `logs/kg_embedding_results.json`
(`n_test: 2493`, `link_prediction.n_evaluated: 2000`) and
`src/kg_embedding.py`'s `max_eval` parameter before writing this.

**Extrinsic evaluation** (task-specific): on 455 real gold-confirmed TP
records (1,612 comparisons), the embedding placed a wrong-but-competing
candidate closer to the correct concept than a random unrelated concept
**68.9%** of the time (vs. 50% chance) — real signal.

**Decisive test — gold-validated tiebreak, threshold sweep**
(`evaluation/kg_tiebreak_validation.py`):

| Threshold | Full-population win/loss | KGE loss vs. hardcoded rule's own pattern |
|---|---|---|
| 0.01 | 12 win / 20 loss (net negative) | 0 / 0 (n=5) |
| 0.02 | 93 win / 104 loss (net negative) | 0 / 0 (n=93, tied) |
| 0.03 | 265 win / 181 loss (net +84) | **63 / 0** |
| 0.05 | 347 win / 200 loss | 63 / 0 |
| 0.08 | 347 win / 202 loss | 63 / 0 |

On the hardcoded `_prefer_lab_procedure_over_observable()` rule's own
pattern (Lab Test, Procedure vs. Observable-Entity/Qualifier-Value), the
rule has **zero losses at every threshold tested**; KGE has 63.
**Specific falsification**: a proposed narrative that KGE would
"naturally" resolve the third near-duplicate concept from §5 was checked
against the actual retrained model on the actual failing entity — KGE
picked the identical wrong concept, 0.0018 embedding-distance margin
(noise, not signal).

**Decision: not wired into production.** The hardcoded rule stays as the
safer specialist mechanism. KGE remains built, tested, checkpointed
(`models/kg_transe_v1.pt`, committed to git), and evaluated — a real,
positive-net-but-imperfect generalist signal for future work on patterns
the hardcoded rule doesn't cover, contingent on a calibrated gating
mechanism that does not yet exist.

---

## 7. Exhaustive-Candidate-Eval Net-Impact (`EXHAUSTIVE_CANDIDATE_EVAL_ENABLED`)

Closes an open question tracked since 2026-08-19: this flag (kept
Step B's per-candidate evaluation running past the first "yes," fixing
one verified pattern — wound-dehiscence-class duplicates) had a known
cost (~34% more LLM calls) but no matching accuracy read on the broader
population it also touches.

**Measured** (`evaluation/exhaustive_candidate_eval_impact.py`, 5-note
test scope, 989 decisions screened): entities where a model
independently accepted 2+ candidates (tiebreak-eligible) graded at
**14.3% precision (3/21)** vs. **84.7% (265/313)** for non-eligible
entities — spot-checked against real entities, not a grading artifact.

**Conclusion**: not neutral, not occasionally negative — substantially
negative on the broad population. The flag's one verified narrow win is
real and kept; a proposed mitigation (route tiebreak-eligible entities
straight to HITL rather than pay for the comparative LLM call) is
identified but **not implemented or verified**.

---

## 8. Calibrator Status

`ConsensusCalibrator` (`src/mollm_tier_calibrator.py`), **17-feature**
logistic regression (was 16 before 2026-08-30, see §9), scores `P(correct)`
only for entities that already fail every hard Tier 1/3 rule. Current
production model retrained on **144 notes**: **validation AUROC 0.852**
(up from 0.845 on the prior 114-note/16-feature model, and 0.74 before
that). `CALIBRATED_AUTO_THRESHOLD = 0.72` — **not re-derived against
either retrain**; see `docs/ConsensusCalibrator_Technical_Reference.md`
§13.5 for the open item this leaves. Two hard "trap" gates bypass the
calibrator entirely for known-fragile patterns
(`_is_coronary_segment_trap()`, `_is_short_alphanumeric_code()`).
Leakage guard (`ConsensusCalibrator.load(..., scoring_note_ids=...)`)
verified live: correctly degraded to untrained/no-op on 3 of the 10
fresh-validation notes (they were in its training set).

---

## 9. Experiment: `kg3_confirmation_count` as a calibrator feature (2026-08-30)

**Question**: KG3 (Memgraph) had been a pure write sink for the whole
project — confirmed by direct code search that zero references to
Memgraph/`kg3_query` existed in any decision-making module. Does reading
it back in as calibrator evidence actually move precision/AUROC, and by
how much specifically (isolated from simply having more training data)?

### 9.1 What was changed, and how the calibrator was retrained

- **New read function**, `src/kg3_query.py::count_kg3_confirmations(driver, entity_text, concept_id)`
  — how many `:PatientObservation` nodes in the live graph already confirm
  this exact (text, concept) pairing. Same never-raise, "0 on any failure"
  contract as the existing DuckDB-sourced `count_prior_confirmations()`.
- **New 17th feature**, `kg3_confirmation_count` (`min(count, 10) / 10.0`,
  identical scaling to the existing `prior_confirmation_count`).
  `FEATURE_SET_VERSION` bumped 1→2, so any model saved under the old
  16-feature layout safely degrades to untrained on load rather than
  silently scoring with misaligned coefficients — verified live before
  retraining.
- **`route_tier()`/`_score_with_calibrator()`** gained `kg3_driver`/
  `kg3_driver_factory` parameters mirroring the existing `conn`/
  `conn_factory` contract exactly. `scripts/run_stage3_tier_gate.py`
  passes `kg3_driver=memgraph_driver` — the same driver already opened
  for the (dry-run) KG3 write-path check, not a second connection.
- **Retraining procedure** (unchanged methodology from the 2026-08-20
  114-note retrain, documented in
  `docs/ConsensusCalibrator_Technical_Reference.md` §10 — only the note
  population and feature count differ): real historical
  `mollm_tier_gate_decisions` rows, restricted to `TIER_4_ENSEMBLE_SPLIT`
  (the population the calibrator is actually consulted for), labeled 1/0
  by SNOMED-crosswalk exact match against gold, split **note-disjoint**
  (every 4th note, sorted, to validation — never random), fit with
  `LogisticRegression(class_weight="balanced")`. The only change this
  round: `evaluation/tier_gate_cal_eval.py::build_labeled_examples()` now
  also takes a live Memgraph driver and computes each training example's
  real `kg3_confirmation_count` via the same function `route_tier()` calls
  at inference — previously this would have trained the new feature on an
  all-zero column, which was caught and fixed before the retrain ran, not
  after.
- **The abbreviation flywheel (`src/abbreviation_flywheel.py`) was NOT
  touched or retrained in this update.** It is a separate mechanism,
  upstream of this calibrator (Stage 1 expansion tie-breaking, not Stage
  3 tier routing), trained from a different data source entirely: real
  `hitl_review_queue` reviewer-confirmed resolutions mined into
  deterministic context-trigger rules (`mine_context_rules()`), plus a
  frequency-priority mechanism (`compute_frequency_priority()`) gated
  behind an explicit `VERIFIED_ALLOW_LIST` that starts **empty** — a
  posture inverted from an earlier block-list design after a real-data
  test found the block-list version re-selecting its own wrong guesses
  (7/7 wrong on gold-check, see the reorg plan's Phase 7 entry). Only the
  calibrator is the subject of this experiment.

### 9.2 Result — isolated from the corpus-growth effect

Comparing the new 144-note/17-feature model's AUROC (0.852) directly
against the old 114-note/16-feature baseline (0.845) conflates two
changes at once (more notes **and** a new feature). To isolate the
feature's own contribution, two models were fit on the **exact same**
144-note data and note-disjoint split — one with real
`kg3_confirmation_count`, one with that single feature zeroed
(`evaluation/tier_gate_cal_eval.py`'s existing ablation mechanism,
`fit_and_report(..., ablate_indices=(kg3_idx,))`):

| | AUROC | Coverage @ 0.72 | Precision @ 0.72 | Promoted |
|---|---|---|---|---|
| With `kg3_confirmation_count` | 0.852 | 20.9% | **97.0%** | 133 |
| Without (ablated) | 0.821 | 18.0% | **91.2%** | 114 |
| **Delta, feature-attributable** | **+0.031** | **+2.9pp** | **+5.8pp** | +19 net |

`kg3_confirmation_count > 0` on 23.9% of train / 23.8% of val examples —
a common signal, not a rare edge case, at the current corpus size.

**Inspected at the individual-example level, not just aggregate deltas.**
29 val examples are promoted only when the KG3 feature is present, and
**all 29 are correct against gold** — dominated by recurring lab-value
patterns (WBC/RBC/HGB/Creat measurements independently confirmed
elsewhere in the graph) and repeated findings (`NSTEMI` ×3, `headaches`
×2, `blindness` ×2). Going the other direction, 10 examples lose
promotion when the feature is added: 6 are real false positives correctly
suppressed — `aspirin` most dramatically (score 0.919→0.224 despite a
high `prior_confirmation_count=66`, once `kg3_confirmation_count=0`
contradicted it) — but 4 are genuine misses (`renal failure`, `Sclera`,
`lungs`, `left ovary` — the last essentially a tie, 0.71999 vs. the 0.72
threshold).

### 9.3 What the fitted coefficients show

Read directly from `models/consensus_calibrator_v1.pkl` (17-feature,
current production model):

| Feature | β (new, 17-feature) | β (old, 16-feature) |
|---|---:|---:|
| `kg3_confirmation_count` | **+7.1368** | *(did not exist)* |
| `top_candidate_similarity_score` | +2.7502 | +3.5188 |
| `frac_supported_1` | +2.3729 | +2.0133 |
| `prior_confirmation_count` | **−2.1114** | **+1.3456** |

`kg3_confirmation_count`'s coefficient (+7.14) is by far the largest
magnitude of any feature in the model — more than double the next
largest. **`prior_confirmation_count`'s coefficient flipped sign** between
the two fits (+1.35 → −2.11). The most defensible reading, stated as
inference rather than a fact read directly off the model: the two
confirmation-count features are correlated (an entity confirmed in
DuckDB's own decision history is often also the kind of entity KG3 was
populated with, since KG3's population was itself derived by grading
those same historical decisions — see §9.4), and a linear model facing
two collinear features can assign credit to one and a compensating
negative weight to the other without that implying either feature
individually predicts incorrectness. This is flagged as something to
verify (e.g. a variance-inflation check, or refitting with only one of
the two confirmation-count features) before treating the sign flip as a
substantive finding on its own, not just an artifact of adding a
correlated feature.

### 9.4 The honest caveat — this measures today's KG3, not real human review

KG3's current population is **100% gold-simulated** — written by grading
the pipeline's own historical decisions against gold and only ingesting
the matching ones (`ingest_reviewed_case()`/`ingest_auto_decision()`),
not real completed human review (still zero in production, see §9
[Known Limitations] below). So `kg3_confirmation_count > 0` is, to a
real and unquantified degree, a restatement of "this exact pattern
already matched gold once" — evaluated here against labels that are
**also** gold-based, on a note population that substantially overlaps the
notes KG3 was populated from. The note-disjoint train/val split guards
against literal row-level leakage (the same guard every other feature in
this calibrator relies on), but not this deeper population-level
circularity. §9.2's numbers are real **given KG3's contents as of
2026-08-30** and should not be read as a measurement of what happens once
KG3 accumulates independent, real human-reviewed confirmations.

**Decision**: adopted as production (`code_version=full_corpus_retrain_2026-08-30`)
on the reasoning that (a) the isolated ablation shows real, well-behaved
signal even acknowledging the caveat above, (b) it is also the only way
to have a genuinely trained calibrator at all post-`FEATURE_SET_VERSION`
bump, since the alternative was leaving production running the old
16-feature model in its safely-degraded, always-`None`-scoring untrained
state. Not treated as a validated claim about real-world unreviewed KG3
data — that measurement does not yet exist.

---

## 10. Known Limitations & Open Gaps — Stated Honestly

- **No false-deflection rate.** `hitl_review_queue` is populated (19,103
  cases) but has zero completed human reviews — the patient-safety
  metric the proposal names cannot be computed until real review happens.
- **No T0→T2 longitudinal trend.** Every deflection/precision figure
  above is a single point-in-time measurement.
- **No confidence intervals** anywhere in this document — none were
  computed; reported as point estimates only.
- **A third SNOMED near-duplicate pattern** ("Clinical Finding"-domain
  concepts) is confirmed real but unhandled by either the hardcoded rule
  or KGE.
- **The MCHC/RDW retrieval fix (§5) is verified on 28 entities**, not the
  full 26-term lab-abbreviation alias population — the other 24 terms are
  re-normalized but not yet graded at the Stage 3 ensemble level.
- **`EXHAUSTIVE_CANDIDATE_EVAL_ENABLED`'s proposed HITL-routing mitigation
  (§7) is not implemented** — only its net impact is measured.
- **Guideline-derived KG injection (Objective 2) and RotatE/CompGCN
  (Objective 4's other two named methods) remain unbuilt** — TransE was
  built as the simplest of the three named KGE methods; the other two are
  real, stated scope, not silently skipped.
- **The production pipeline's own automatic write path remains
  `dry_run=True`** — `scripts/run_stage3_tier_gate.py`'s call to
  `ingest_auto_decision()` never commits a live, unreviewed write, exactly
  as before. **Nuance added 2026-08-30, not previously true**: the live
  KG3 graph is no longer empty — a one-off, manually-run population script
  (`ingest_reviewed_case()`, real writes, not dry-run) wrote ~6,600 nodes
  derived by grading historical decisions against gold and treating a
  match as a simulated approval. This is **not** real human review and
  not an ongoing autonomous write path; it exists specifically so
  `kg3_confirmation_count` (§9) has real data to be evaluated against. Do
  not read "KG3 has data in it" as "the system writes to KG3 unreviewed
  in production" — those are two different claims, and only the second
  one was previously stated by this bullet.

---

## 11. Reproducibility

- Full code-flow trace: `docs/Code_Flow.md`.
- Every metric's exact formula + real implementing code:
  `docs/Code_Reference_Stages_And_Metrics.md`.
- Chronological session log with every decision's rationale:
  `docs/2026-08-20_Session_Results_And_Status.md` (§1–15).
- This project's own generated data (not the licensed OMOP/Athena
  reference vocabulary, which is excluded — see that export script's own
  docstring): `exports/*.parquet` (13 tables, 77,687 rows, 16.3MB),
  reproducible via `scripts/export_pipeline_tables.py`.
- KGE checkpoint: `models/kg_transe_v1.pt`.
- Fresh-10 validation note IDs: `ui/components/fresh10_notes.py`.
- §9's retrain: `scripts/retrain_calibrator_full_corpus.py --save`. §9's
  isolated feature ablation: `evaluation/kg3_feature_ablation.py`.
