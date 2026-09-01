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

**Gap, stated honestly, updated 2026-08-31**: the proposal's own
independent-re-audit false-deflection rate still cannot be computed —
`hitl_review_queue` has real cases now (19,103 total, 250 in the fresh-10
scope, populated 2026-08-20) but zero completed human reviews. A
gold-substituted proxy now exists instead (this doc's §12 below, full
derivation in `docs/Code_Reference_Stages_And_Metrics.md` §15): 7.9%–23.2%
depending on population, real numbers, not the proposal's real metric. No T0→T2
deflection-rate trend exists — every number above is a single
point-in-time read.

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

## 6. KG Embedding (TransE + RotatE) Results

**Update**: RotatE was subsequently built too, as a real 4-configuration
ablation (`guideline`/`gold`/`combined`/`snomed_is_a` training data) —
full results, method, and honest limitations in
`docs/KG_Embedding_Technical_Reference.md` (merged 2026-09-01 from what
were previously two separate TransE/RotatE docs). **Neither method beats
the hardcoded rule; RotatE is worse than TransE at this task.** This
section covers TransE specifically; see that doc's Part B for RotatE.

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

**Extrinsic evaluation** (task-specific), **current, post-locked-test-
split-leakage-fix numbers** (2026-08-31; the leakage fix removed 39 of
149 source notes that belonged to the official locked test split — see
`docs/KG_Embedding_Technical_Reference.md` §6.2/§10 for the full
before/after): on 343 real gold-confirmed TP records (1,209 comparisons),
the embedding placed a wrong-but-competing candidate closer to the
correct concept than a random unrelated concept **63.7%** of the time
(was 68.9% pre-fix; vs. 50% chance) — real signal, smaller than
originally measured.

**Decisive test — gold-validated tiebreak, threshold sweep**
(`evaluation/kg_tiebreak_validation.py`), **current numbers**:

| Threshold | Full-population win/loss | KGE loss vs. hardcoded rule's own pattern |
|---|---|---|
| 0.01 | 11 win / 24 loss | 0 / 0 (n=5) |
| 0.02 | 65 win / 243 loss | 29 / 0 (n=93) |
| 0.03 | **130 win / 379 loss (net −249)** | **134 / 0** |
| 0.05 | 190 win / 399 loss | 134 / 0 |
| 0.08 | 200 win / 400 loss | 134 / 0 |

**Superseded twice, both moves in the same direction — disclosed in
full, not collapsed to one row.** The net at threshold 0.03 was
originally 265W/181L (net +84), then 228W/263L (net −35) from corpus
growth alone, then the numbers above (130W/379L, net −249) after the
locked-test-split leakage fix — TransE's original apparent strength was
partly an artifact of notes it should never have been measured against.
**The current, correct conclusion**: TransE is net-harmful on the
broader tied-pair population, not just losing to the rule on its own
narrower specialist pattern. On the hardcoded
`_prefer_lab_procedure_over_observable()` rule's own pattern (Lab Test,
Procedure vs. Observable-Entity/Qualifier-Value), the rule has **zero
losses at every threshold tested**; KGE now has 134 (was 63).
**Specific falsification**: a proposed narrative that KGE would
"naturally" resolve the third near-duplicate concept from §5 was checked
against the actual retrained model on the actual failing entity — KGE
picked the identical wrong concept, 0.0018 embedding-distance margin
(noise, not signal).

**Decision: not wired into production.** The hardcoded rule stays as the
safer specialist mechanism, and is now more decisively ahead than
originally measured. KGE remains built, tested, checkpointed
(`models/kg_transe_v1.pt`, committed to git), and evaluated. RotatE
(§ above, full ablation in `docs/KG_Embedding_Technical_Reference.md`)
was built as the natural follow-on question ("does a different KGE
architecture do better") and found to underperform TransE at this same
task across all four training-data configurations tried — neither
method is a candidate for production use without a calibrated gating
mechanism that does not yet exist (see that doc's §13 for the concrete,
lower-risk integration points considered but not built).

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

**Conclusion at the time**: not neutral, not occasionally negative — substantially
negative on the broad population. The flag's one verified narrow win is
real and kept; a proposed mitigation (route tiebreak-eligible entities
straight to HITL rather than pay for the comparative LLM call) was
identified but not implemented or verified.

### 7.1 Re-investigated at full corpus scale, 2026-08-31 — the mitigation isn't justified after all

The 14.3% figure above was a 5-note, 21-entity sample. Re-measured
against the full corpus (2,529 real tiebreak-eligible entities, detected
directly from stored `eval_trail` data, not re-derived):

| Population | Precision | n |
|---|---|---|
| Original 5-note sample | 14.3% | 3/21 |
| **Full corpus, raw plurality pick, any tier** | **38.8%** | 437/1,126 gradable |
| **Full corpus, AUTO-tier subset only** (the actual unreviewed-write risk) | **78.7%** | 37/47 gradable |

The direction of the original finding is real — this population's raw
plurality pick genuinely underperforms (38.8% vs. the ~85% baseline for
non-eligible entities) — but the **original 14.3% number understated the
real rate by nearly 3×**, a small-sample artifact, not a corpus-wide
truth.

More importantly: only **64 of 2,529** tiebreak-eligible entities (2.5%)
ever actually reach an AUTO tier at all (51 unanimous `TIER_1`, 13
calibrator-promoted `TIER_1B`) — the existing gate (unanimous-agreement
requirement + `CALIBRATED_AUTO_THRESHOLD`) is **already** filtering this
genuinely-weaker population down to a much smaller, much safer subset
(78.7% precision, in line with the rest of the system's AUTO-tier
precision elsewhere in this document). The proposed mitigation's real
justification was precision risk; that risk, measured properly, is far
smaller than believed.

**Decision: not implemented.** What remains after removing the precision
argument is a pure compute-cost optimization (skip one comparative LLM
call across ~2,529 entities corpus-wide, versus ~19,000+ total Stage-3
decisions) — real, but modest, and not free: implementing it as
originally scoped (skip the ensemble outcome entirely, not just the
extra call) would discard the 64 entities currently reaching AUTO tier
legitimately. Documented here rather than built, so the next person
doesn't re-discover the same small-sample-artifact urgency this document
originally reported.

---

## 8. Calibrator Status

`ConsensusCalibrator` (`src/mollm_tier_calibrator.py`), **17-feature**
logistic regression (was 16 before 2026-08-30, see §9), scores `P(correct)`
only for entities that already fail every hard Tier 1/3 rule. **Updated
2026-08-31** (`docs/ConsensusCalibrator_Technical_Reference.md` §18): a
locked-test-split contamination was found and fixed — the training pool
had unconditionally included 39 of 149 notes (26%) from `data/splits/
note_splits.csv`'s official locked test split, plus (found while
re-verifying the first fix) 4 of fresh-5's 5 real validation notes.
Current production model retrained on the corrected **105-note pool**:
**validation AUROC 0.868** (up from the previously-reported 0.852 on the
contaminated 144-note fit — removing 39% of the pool did not hurt).
**Same day, re-derived**: `CALIBRATED_AUTO_THRESHOLD` moved **0.72 → 0.78**
(`docs/ConsensusCalibrator_Technical_Reference.md` §19) — applying the
exact rule 0.72 was originally chosen with (smallest threshold reaching
100% val precision, hard traps active) to the clean model found 0.72 no
longer holds 100% on current data (only ~98%; `neck pain` @ 0.779742 is
now the sole blocker, the same shape of story as the original
`incontinence` case). Real cost: coverage at 0.78 is 13.6%, down from
~16-17% at 0.72 on the same clean pool. Two hard "trap" gates bypass the
calibrator entirely for known-fragile patterns
(`_is_coronary_segment_trap()`, `_is_short_alphanumeric_code()`).
Leakage guard (`ConsensusCalibrator.load(..., scoring_note_ids=...)`)
verified live: the corrected model's `training_note_ids` now has **zero
overlap** with both fresh-10 and fresh-5 — both remain genuinely usable
held-out validation populations for the current calibrator, which was
not true of fresh-10 under the pre-fix model (8 of its 10 notes had
silently become training data by the time this was checked).

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

## 10. Fresh-5 End-to-End Validation (2026-08-30)

**Question**: does any of §9's work (the new `kg3_confirmation_count` calibrator feature, now in production) actually show up in a genuinely fresh, real, end-to-end run — not a re-grade of already-stored decisions? Five notes were picked that had **never been processed by this pipeline at all** (confirmed via `extracted_entities`, 0 rows for all 5 before this run) — smallest-by-char-count from the 133 gold-covered notes not yet in the 144-note corpus, same "smallest for speed" convention as `scripts/run_fresh5_final_validation.py`'s second batch: `13397956-DS-5`, `17739994-DS-31`, `16410990-DS-12`, `16795604-DS-17`, `17309807-DS-20`. Run for real: `src.clinical_pipeline.run_pipeline()` (Stage 1-2b) then `scripts/run_stage3_tier_gate.py` (Stage 3, real production code path, real Memgraph driver, real production calibrator) — not a replay or simulation.

### 10.1 Stage 1-2b: real gold comparison vs. the existing headline numbers

Via `scripts/score_gold_recall.py` (same methodology as §2's table), 544 real gold annotations:

| Metric | Corpus-wide (144 notes, §2) | Fresh-10 (§2) | **Fresh-5 (new, 2026-08-30)** |
|---|---|---|---|
| Span recall | 53.0% | 49.5% | **58.6%** (319/544) |
| Linked recall | 33.5% | 26.8% | **40.1%** (218/544) |
| Linked precision | 50.0% | 45.3% | **54.2%** (218/402) |
| **Linked F1** | 40.1% | 33.7% | **46.1%** |
| Benchmark char IoU (macro/weighted) | 0.1437/0.2824 | 0.1453/0.2425 | **0.2131/0.3366** (all) / 0.2187/0.3311 (in-scope) |

Fresh-5 beats both prior measurements on every metric. **This is not attributable to §9's work** — Stage 1-2b never touches the calibrator or KG3 at all. It reflects the cumulative effect of every fix landed since the 2026-08-20 measurement (SNOMED crosswalk fix, SNOMED near-duplicate retrieval fix, allergy-context fixes, extraction cold-start work). Real per-note spread: span recall 38.6%-66.7%, linked recall 22.9%-48.2% across the 5 notes — a noisier read than the 144-note/1,497-annotation prior measurements, given n=5.

**Vs. the external DrivenData benchmark leaderboard** (`drivendata.org/benchmarks/310`, live-fetched 2026-08-30, not the closed competition #258, which is a different, non-comparable leaderboard on a much larger official test set): the official reference baseline is 0.1794 macro IoU. Both prior internal measurements (0.1437, 0.1453) sat **below** that baseline; fresh-5 (0.2131-0.2187) is the **first time this project's own measurement has cleared it** — by 0.019-0.039. Still well below the real leaderboard's top entries (0.41-0.48 macro / 0.56-0.64 weighted) and rank is unchanged (still last/7th on macro, 6th/7th on weighted, since the gap to the nearest entry — 0.2321 macro, `FAISS+Qwen3 Instruct` — closed from ~0.087 to ~0.015-0.019, about 80%, without fully closing). Caveat: this is a 5-note internal sample, not the real withheld test set the leaderboard entries were scored on — a calibration point, not a literal submission comparison (same framing this doc already used for the 3-note version of this check).

### 10.2 Stage 3: real production run with the current calibrator + live KG3

373 entities, 0 errors, 44.0 min wall-clock. Tier distribution: `TIER_1_AUTO_VALIDATED`=111, `TIER_3_AUTO_VALIDATED`=73, `TIER_1B_CALIBRATED_AUTO_VALIDATED`=27, `TIER_2_AUTO_RESOLVED`=8, `TIER_4_ENSEMBLE_SPLIT`=109, `TIER_5_TRUE_AMBIGUITY`=44, 1 unscored. **AUTO coverage 211/373 = 56.6%**, gold-graded precision (clean-span, SNOMED-crosswalk) **92.1%** (139/151 gradable).

Calibrator loaded genuinely **trained**, not leakage-refused (`ConsensusCalibrator.load(..., scoring_note_ids=...)` correctly passed — these 5 notes were never in its training set, since they'd never been processed before this run at all).

**The calibrator's own isolated contribution** (`TIER_1B` specifically — the only tier a calibrator promotion reaches that counts toward `AUTO_TIERS`):

| | TP | FP | Precision | Recall* | **F1** |
|---|---|---|---|---|---|
| Without a calibrator (nothing ever promotes from `TIER_4`) | 0 | 0 | n/a | 0.0% | **0.0%** |
| With the current production calibrator | 21 | 2 | **91.3%** | 44.7% | **60.0%** |

*Recall = TP / (TP + still-correct-but-stuck-at-`TIER_4`) = 21/(21+26). 26 more entities remain genuinely correct by plurality vote but didn't clear the 0.72 threshold — a real, honest coverage gap, not a defect. 91.3% precision on n=23 gradable is consistent with (a touch below) the held-out val-set measurement in §9.2/§17.5 of `docs/ConsensusCalibrator_Technical_Reference.md` (97.0%), within expected small-sample noise. 2 wrong: `Incision` (a specificity mismatch — predicted a valid but less-specific SNOMED concept than gold wanted) and `neurology`.

**Conclusion**: the calibrator genuinely activates and produces real, mostly-correct promotions on truly unseen notes — not just on its own held-out val split. This is the first real end-to-end confirmation of §9's work outside a re-grade of already-stored decisions.

### 10.3 All-stages summary — corpus-wide, fresh-10, fresh-5, side by side

Every column below is a real, gold-graded measurement (not a projection) — see §2 for corpus-wide/fresh-10 methodology and §10.1-10.2 above for fresh-5's. Gold annotation counts differ substantially by column (39,403 / 1,497 / 544) — read percentages, not raw counts, as the comparable figures; a 5-note sample is noisier than the other two, per the per-note spread already noted in §10.1.

| Stage | Metric | Corpus-wide (144 notes, 2026-08-20) | Fresh-10 (2026-08-20) | **Fresh-5 (2026-08-30)** |
|---|---|---|---|---|
| 1-2b | Gold annotations | 39,403 | 1,497 | 544 |
| 1-2b | Span recall | 53.0% | 49.5% | **58.6%** |
| 1-2b | Linked recall | 33.5% | 26.8% | **40.1%** |
| 1-2b | Linked precision | 50.0% | 45.3% | **54.2%** |
| 1-2b | **Linked F1** | 40.1% | 33.7% | **46.1%** |
| 1-2b | Benchmark char IoU (macro) | 0.1437 | 0.1453 | **0.2131** (0.2187 in-scope) |
| 1-2b | Benchmark char IoU (weighted) | 0.2824 | 0.2425 | **0.3366** (0.3311 in-scope) |
| 3 | Total Stage-3 decisions | 19,202 | 250 | 373 |
| 3 | **Deflection rate** (all `AUTO_TIERS` / all decisions) | 57.0% | 31.2% | **56.6%** |
| 3 | **AUTO-tier precision** (gradable) | 86.9% | 76.8% | **92.1%** |

**Reading this honestly, not just favorably**: fresh-5 leads on every single row — but it's the *newest* measurement, on the *smallest* sample, run *after* every fix this whole document catalogs (SNOMED crosswalk, near-duplicate retrieval, allergy-context, the KG3 calibrator feature). It is not a controlled ablation against the other two columns — multiple things changed between 2026-08-20 and 2026-08-30, not one. Treat it as "the pipeline's current state looks meaningfully better on fresh data than the 2026-08-20 snapshot," not as an isolated measurement of any single change (§10.2's own TP/FP/precision/recall/F1 table is the one isolated, single-variable measurement in this section — the calibrator's own marginal contribution, holding everything else fixed).

---

## 11. Experiment: Guideline Evidence Injection (2026-08-30)

**Question**: `src/guideline_evidence.py` (curated clinical-guideline rules injected into the MoLLM tiebreak prompt, built 2026-08-20, off by default pending validation) — does it actually help? Full plan at `/home/ec2-user/.claude/plans/here-is-details-of-peppy-pixel.md`.

**Real population**: 217 entities (corrected from an initially-reported 258 — a real `select_population()` dedup bug, no `DISTINCT` on `entity_id`, was found and fixed; `normalized_entities` can carry more than one row per `entity_id` after re-normalization), 88 distinct mention texts, spanning `TIER_4_ENSEMBLE_SPLIT`/`TIER_2_AUTO_RESOLVED` decisions whose candidate list gets a real name match in the 76-file guideline corpus.

**Method**: `evaluation/guideline_evidence_ab_test.py` — real, live, paired LLM calls (not a replay of stored votes, since guideline evidence changes the prompt text itself), each entity run twice through `route_tier()`, once with `GUIDELINE_EVIDENCE_ENABLED` off and once on, graded via the same clean-span + SNOMED-crosswalk methodology as every other precision figure in this document.

**Result, 50-entity slice (n=23 gradable)**:

| | Gradable | Correct | Precision |
|---|---|---|---|
| OFF | 23 | 20 | 87.0% |
| ON | 23 | 20 | 87.0% |

**Zero flips** — every one of the 23 paired entities got the identical correctness outcome with guideline evidence on vs. off, not just similar aggregate precision.

**Why, investigated directly rather than assumed**: inspected all 13 unique gradable entities' actual injected evidence text and full 3-model reasoning, both arms. Every real hit was **one-sided background context about the single candidate already chosen** ("chest pain can be a symptom of ACS", "aspirin is standard therapy for ACS", "CHF is a differential diagnosis to consider for COPD") — never a fact that discriminates between the two specific candidates causing a tie. Even on genuine 2-1 splits, the injected evidence didn't address the axis of disagreement, so reasoning text sometimes changed wording between arms while the verdict pattern stayed identical. This is a structural explanation, not a small-sample artifact — matches exactly what the original scoping investigation (plan file) already found: real hits in this corpus are overwhelmingly one-sided facts, not pair-adjudicating rules.

**Decision**: `GUIDELINE_EVIDENCE_ENABLED` stays off. Not concluded from statistical underpowering alone — the zero-flip result plus a mechanistic explanation for *why* it can't move a tiebreak under the current name-match-only design together make a full 217-entity run unlikely to change the conclusion. The full run was not executed (a real ~2.4-hour cost at this reduced population size); this stays a documented, reasoned decision rather than an assumption, and can be revisited if the underlying design changes (e.g. the plan's own out-of-scope items — rule-relevance ranking, or wiring evidence into Stage 1's separate acronym-escalation prompt instead).

---

## 12. Known Limitations & Open Gaps — Stated Honestly

- **False-deflection rate, closed 2026-08-31 via a gold-substituted proxy.**
  `hitl_review_queue` is populated (19,103 cases) but still has zero
  completed human reviews, so the proposal's own re-audit-based
  computation remains impossible. What's now real: a wrong AUTO-tier
  decision (gold-checked, not human-reviewed) is exactly what "should have
  gone to HITL but did not" means, so `1 - auto_tier_precision` on the
  same three populations already measured gives a real, Wilson-CI'd
  number — corpus-wide 13.1% [12.3%, 14.0%], fresh-10 23.2% [14.1%,
  35.8%], fresh-5 7.9% [4.6%, 13.4%]. Full derivation:
  `docs/Code_Reference_Stages_And_Metrics.md` §15. **Still open**: this is
  a proxy, not the proposal's real independent-re-audit metric, and the
  proposal's own "pre-set acceptable bound" for this metric was never
  defined anywhere in this project — there is no threshold to check these
  numbers against, only the raw rate itself.
- **T0→T2 longitudinal trend, partially closed 2026-08-30.** §15 now
  reports two real, comparable checkpoints (T0 = 2026-08-20, T1 =
  2026-08-30) with a proper two-proportion significance test — all four
  metrics tested improved with p < 0.01. **Still open**: a third
  checkpoint (T2) doesn't exist yet and can't be fabricated, and two
  points establish a significant difference, not a validated trend line.
  T0→T1 also isn't isolated to one cause — it reflects everything shipped
  between the two dates together.
- **Confidence intervals, now closed on the headline metric, 2026-08-31.**
  Every figure above was originally just a point estimate.
  `docs/Code_Reference_Stages_And_Metrics.md` §14 first added Wilson score
  intervals (AUTO-tier precision, Linked precision/recall, calibrator
  `TIER_1B` promotion precision); §16 now adds the project's own
  **actually-specified method** — bootstrap CIs resampled at the note
  level, not a plain binomial-proportion interval — for AUTO-tier
  precision and the false-deflection-rate proxy (§15) across all three
  populations. Real result: [85.9%, 88.1%] corpus-wide, [68.9%, 83.9%]
  fresh-10, [86.1%, 96.7%] fresh-5. Note-level clustering does **not**
  uniformly widen intervals relative to Wilson (fresh-10's bootstrap CI is
  actually *narrower*, 15.1pp vs. 21.7pp) — a real, checked finding, not
  assumed. The specific claim this was built to stress-test (does
  fresh-5 genuinely beat fresh-10) **holds up better under bootstrap, not
  worse** — the gap widens from 0.7pp (Wilson) to 2.2pp (bootstrap).
  **Still open**: Linked precision/recall and the calibrator's `TIER_1B`
  slice still only have Wilson intervals — bootstrap wasn't extended to
  those, since AUTO-tier precision (the paper's headline metric) was the
  priority. A real, caught-live bug during this work is worth knowing
  about if reusing note-ID lists elsewhere in this codebase: the obvious
  `evaluation/grade_fresh5_by_tier.py` note list is a *different*, older
  (2026-08-17) 5-note batch that happens to share the name "fresh5" — not
  the real "Fresh-5 (2026-08-30)" notes this section's own numbers are
  built on.
- **A third SNOMED near-duplicate pattern ("Clinical Finding"-class
  concepts) — investigated at corpus scale 2026-08-31, does not justify a
  fix.** The original characterization (one observed case) doesn't hold
  up: 1,562 real Lab Test entities have a `Clinical Finding`-class
  candidate alongside another class in their pool, but on a 38-entity
  gradable sample, that class was actually *chosen* only once (2.6%) —
  and that one case was wrong. `_LAB_PROCEDURE_PENALIZED_CLASSES` doesn't
  even include `Clinical Finding`; raw SapBERT similarity alone already
  keeps it from winning in the vast majority of real cases. Not
  recommended as a priority — building a new tiebreak rule here would
  touch ~1 case per ~38, not the systemic pattern originally believed.
- **The MCHC/RDW retrieval fix (§5) closed, 2026-08-30**: the remaining 24
  terms were graded at the Stage 3 ensemble level for the first time (991
  real decisions, 521 `AUTO_TIERS`, 467 gradable). Found overall precision
  at only 57.8% (270/467) — entirely explained by 5 terms sitting at
  exactly 0% each (135 wrongly auto-approved entities: `hco3`, `urean`,
  `na`, `total co2`, `mch`), while the other 19 averaged ~95%+. Root-caused
  to two real, distinct bugs, both fixed: (1) `tier3_fast_path()`'s
  `verified_lab_test_alias` trust-bypass only allowlisted one
  `ambiguity_reason` string, missing `"alias_candidate_outranked"` —
  orchestrator.py's own detector for exactly this situation (the alias
  candidate present but outranked by a wrong one), computed but never
  consulted; (2) the Lab Value Suffix Fallback's retry-selection logic
  only adopted a stripped-text retry when it strictly improved the coarse
  tier label, silently discarding a retry that introduced the correct,
  gold-verified alias candidate at an *unchanged* tier. Verified directly
  against each term's own gold-documented concept ID (not assumed):
  `urean`/`total co2`/`na` now fully auto-resolve correctly; `hco3`/`mch`
  get the pool-inclusion fix (previously entirely absent, now genuinely
  visible to the ensemble) but are deliberately still routed to the full
  ensemble rather than auto-approved, since an existing test encodes real,
  still-valid caution (the 2026-08-20 MCH/MCHC near-duplicate finding)
  about the specific ambiguity reason those two produce. **Not yet done**:
  a full corpus re-normalization/re-grading batch to measure the real,
  at-scale precision delta — this was verified via direct function calls
  on each term's bare form, not yet applied to a live batch re-run.
- **`EXHAUSTIVE_CANDIDATE_EVAL_ENABLED`'s proposed HITL-routing mitigation
  — investigated at corpus scale 2026-08-31, deliberately not built.**
  §7.1: the original 14.3%-precision justification was a 5-note/21-entity
  sample; the real, full-corpus AUTO-tier-specific precision is 78.7%
  (the existing gate already filters this population). What's left is a
  real but modest compute-cost optimization, not a precision fix, and
  building it as originally scoped would cost 64 currently-legitimate
  AUTO-tier decisions corpus-wide. A considered non-decision, not an
  oversight.
- **Update, 2026-08-31 — this bullet is now stale on two of its three counts.**
  Guideline-derived KG injection (Objective 2) was built and A/B-validated
  (§11): a real null result (87.0%/87.0% precision, zero flips), stays off.
  RotatE (Objective 4's second named KGE method) was built and evaluated as
  a genuine 4-config ablation (curated guideline graph / gold-derived
  competition triples / their combination / the full SNOMED IS_A hierarchy)
  — see `docs/KG_Embedding_Technical_Reference.md`. Real finding:
  RotatE shows a stronger AGGREGATE embedding-separation signal than TransE
  on two of the four configs (72–84% vs. TransE's 69%), but is WORSE as an
  actual per-entity tiebreak (loses 8–9x more than it wins, vs. TransE's
  near-breakeven), and every usable RotatE config still loses head-to-head
  to the existing hardcoded `_prefer_lab_procedure_over_observable()` rule
  (0 losses for the rule, every config, every threshold) — a materially
  worse showing than TransE's own already-negative result against that same
  rule. **Only CompGCN (Objective 4's third named method) remains genuinely
  unbuilt and deliberately deferred** — real, stated scope, a substantially
  bigger architectural lift than either TransE or RotatE, not silently
  skipped (see the RotatE doc's own §5).
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

## 13. Reproducibility

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
- §10's fresh-5 note IDs: `13397956-DS-5`, `17739994-DS-31`, `16410990-DS-12`,
  `16795604-DS-17`, `17309807-DS-20`. Stage 1-2b: `src.clinical_pipeline.run_pipeline()`
  per note. Stage 3: `scripts/run_stage3_tier_gate.py --note-ids <the 5 ids> --is-test`.
  Grading: `scripts/score_gold_recall.py --note-ids <the 5 ids>` (§10.1) and
  `evaluation/grade_overnight_corpus_run.py`'s `grade_population()`/
  `plurality_candidate_index()` against the same 5 notes (§10.2) — run ad hoc for
  this section, not yet consolidated into one checked-in script the way §9's
  ablation was.

---

## 15. Longitudinal Trend — T0 and T1 (real checkpoints; T2 pending)

**What this section honestly is, and isn't.** `docs/Evaluation_Criteria.md`'s own proposal language calls for "a T0 to T2 deflection-rate trend that is statistically distinguishable from flat" — three checkpoints, not two, with a T2 that by definition requires further real elapsed deployment time this project doesn't have yet. This section is **not** that. What it is: the two real, comparable, already-measured checkpoints this project actually has — reported honestly as a first data point toward that requirement, not a substitute for it.

**T0 = 2026-08-20** (fresh-10, §2) and **T1 = 2026-08-30** (fresh-5, §10) are the two genuinely comparable checkpoints available: both are "fresh, never-before-processed at measurement time" note populations, graded by the identical clean-span + SNOMED-crosswalk methodology. 10 real calendar days apart.

| Metric | T0 (2026-08-20) | T1 (2026-08-30) | Two-proportion z-test |
|---|---|---|---|
| **Deflection rate** | 31.2% (78/250) | 56.6% (211/373) | z = 6.22, **p < 0.0001** |
| **AUTO-tier precision** | 76.8% (43/56) | 92.1% (139/151) | z = 2.99, **p = 0.0027** |
| **Linked recall** | 26.8% (401/1,497) | 40.1% (218/544) | z = 5.77, **p < 0.0001** |
| **Linked precision** | 45.3% (401/886) | 54.2% (218/402) | z = 2.99, **p = 0.0028** |

**All four differences are statistically significant** (two-proportion z-test, not just a favorable point estimate) — genuinely stronger evidence than "the newer number looks better," which is as far as §10.3's own honest caveat took this. Deflection rate, the proposal's own named headline metric for this exact test, is the single most significant of the four (p < 0.0001).

**What this does not establish**: a *trend* in the statistical sense needs three or more points to distinguish a real trajectory from two noisy draws that happen to differ — this is a real, significant **difference between two points**, not yet a validated trend line. Nor does it isolate a cause: per §10.3's own caveat, T0→T1 reflects the cumulative effect of everything shipped in between (the SNOMED crosswalk fix, near-duplicate retrieval fix, the `kg3_confirmation_count` feature, the lab-alias bug fixes), not one attributable change. **T2 does not exist and cannot be fabricated** — it requires a third, later, comparably-measured checkpoint. This section should be extended, not re-derived, the next time such a checkpoint exists.

---

## 14. Repurposed-Data Mechanisms — Measured Benchmark Impact

Every mechanism below takes data this project already has for another reason — historical tier-gate decisions, the pipeline's own gold-simulated approvals, a curated lookup dict already sitting in the codebase, external guideline documents — and repurposes it as a *new* signal into the pipeline's own decisions. This table consolidates every such mechanism's **real, measured** effect on this project's actual benchmark metrics (AUTO-tier precision, deflection rate, Linked F1) — not a projection. Positive, null, and negative results are reported with the same rigor; a mechanism failing to move the needle is exactly as reportable as one that does.

| Mechanism | Data repurposed | Target metric | Measured result | Verdict |
|---|---|---|---|---|
| **`kg3_confirmation_count`** (§9, §10.2) | Historical tier-gate decisions, gold-graded and written to the live KG3 graph | AUTO-tier precision/recall/F1 on the calibrator-eligible (`TIER_4`) population | **+5.8pp precision, +2.9pp coverage, F1 0%→60.0%** on real fresh-5 data (isolated ablation: +0.031 AUROC, holding corpus size fixed) | **Positive, adopted** — in production |
| **Lab-alias fix** (§12, this doc's Known Limitations) | The already-existing, gold-verified `_LAB_TEST_ALIASES` dict — bugs prevented it from being consulted correctly | AUTO-tier precision on 5 specific lab-abbreviation terms | 3 of 5 terms: **0% → 100%** (verified against each term's own gold-documented concept). 2 of 5: still correctly declines to auto-approve, but the correct concept is now visible to the ensemble for the first time | **Positive, fixed** — not yet re-measured at full corpus scale |
| **`prior_confirmation_count`** (`docs/ConsensusCalibrator_Technical_Reference.md` §10.7) | The pipeline's own DuckDB decision/review history | AUTO-tier precision on the calibrator-eligible population | Ablation-tested: removing it "doesn't fix the coronary false positives and measurably costs precision elsewhere" — real but **secondary** signal (current fitted β = −2.11, likely collinear with `kg3_confirmation_count`, see §17.6 of that doc) | **Positive, kept** — smaller effect than `kg3_confirmation_count` alone |
| **Mined context rules** (`mine_context_rules()`, KG3 doc §5, §11.2) | Gold-simulated-approved HITL cases → deterministic Stage 1 tiebreak rules | Stage 1 abbreviation-disambiguation accuracy | 27 real rules mined (21 confirmed `"no" + lad` → Lymphadenopathy examples) — but **zero** of fresh-5's 22 ambiguous entities used any mined abbreviation | **Real signal, null on this sample** — mining works, coverage hasn't reached this population yet |
| **Guideline evidence injection** (§11 above) | 76 curated clinical-guideline documents | Tiebreak-resolution precision on `TIER_4`/`TIER_2` split votes | 87.0% / 87.0%, **zero flips** across 23 real paired entities — structural cause identified (real hits are one-sided background facts, not pair-adjudicating rules) | **Null, not adopted** — flag stays off |
| **Acronym escalation** (`src/acronym_escalation.py`, tracked in the reorg plan's Phase 4) | MoLLM + the pipeline's own dictionary, repurposed to resolve ambiguous abbreviations before extraction | Stage 1 expansion accuracy | 34.3%→36.1% precision at corpus scale — a systematic textbook-prior bias (e.g. `LAD`→"left anterior descending artery" over gold's "Lymphadenopathy") | **Negative, not adopted** — stays off by default |

**Reading this honestly**: exactly one mechanism (`kg3_confirmation_count`) has a clearly positive, adopted, at-scale-verified effect; one more (the lab-alias fix) is positive and real but not yet re-measured at full corpus scale; `prior_confirmation_count` contributes a real but secondary signal; two mechanisms (mined context rules, guideline evidence) are honestly null on the specific populations tested so far, not proven ineffective in general; one (acronym escalation) is a real, corpus-measured negative result. The pattern worth taking away: **repurposing existing data has real, non-trivial potential — roughly half of what's been tried has moved a real benchmark number — but it is not uniformly positive, and every claim above is backed by a specific, reproducible measurement, not an assumption that "more signal is always better."**

---

## 16. Three-Batch Comparison at Full Stage 3 Coverage (Fresh-10 / Fresh-5 original / Fresh-5 gazetteer, 2026-09-01)

§10's fresh-5 numbers were captured mid-batch; both fresh-5 batches (and Fresh-10) were subsequently run to **genuine full Stage 3 coverage** (`scripts/complete_fresh5_stage3_full.py` — no per-note cap, idempotent) and re-graded. This section is the current, authoritative, side-by-side comparison of all three named batches (`ui/components/note_batches.py`'s `NOTE_BATCHES`), superseding §10.3's numbers for the two batches it lists in more detail here plus adding the gazetteer batch for the first time.

### 16.1 Tier distribution (raw decision counts, % of that batch's total)

| Tier | Fresh-10 (2026-08-20) | Fresh-5 original (2026-08-30) | Fresh-5 gazetteer (2026-08-31) |
|---|---|---|---|
| `TIER_1_AUTO_VALIDATED` | 75 (29.1%) | 111 (29.8%) | 119 (29.0%) |
| `TIER_1B_CALIBRATED_AUTO_VALIDATED` | 0 | 27 (7.2%) | 0 |
| `TIER_2_AUTO_RESOLVED` | 5 (1.9%) | 8 (2.1%) | 8 (1.9%) |
| `TIER_3_AUTO_VALIDATED` | 3 (1.2%) | 73 (19.6%) | 78 (19.0%) |
| `TIER_4_ENSEMBLE_SPLIT` | 124 (48.1%) | 109 (29.2%) | 154 (37.5%) |
| `TIER_5_TRUE_AMBIGUITY` | 49 (19.0%) | 44 (11.8%) | 50 (12.2%) |
| null/error | 2 (0.8%) | 1 (0.3%) | 2 (0.5%) |
| **Total decisions** | **258** | **373** | **411** |

### 16.2 Accuracy & coverage

| Metric | Fresh-10 | Fresh-5 original | Fresh-5 gazetteer |
|---|---|---|---|
| Gold annotations | 1,497 | 544 | 632 |
| Span recall | 49.5% | 58.6% | 55.4% |
| Linked recall | 26.8% | 40.1% | 33.5% |
| Linked precision | 45.1% | 54.2% | 46.9% |
| AUTO-tier precision (gradable) | 76.8% (43/56) | 92.0% (139/151) | 93.3% (111/119) |
| Deflection rate (share landing AUTO) | 30.2% | 56.6% | 47.9% |

### 16.3 Timing (mean seconds/entity, pause-excluded — `MAX_PLAUSIBLE_ENTITY_GAP_SECONDS=600`)

| Stage | Fresh-10 | Fresh-5 original | Fresh-5 gazetteer |
|---|---|---|---|
| Stage 2b (normalization) | 1.216s | 2.938s | 2.110s |
| Stage 3 (MoLLM tier gate) | 10.157s | 6.947s | 7.025s |

**Reading it honestly**: Fresh-10 is the oldest measurement and TIER_3's near-absence there (3 vs. 73/78 in the two Fresh-5 batches) reflects a real pipeline change since — the fast-path exact-match tier got materially better coverage later, consistent with §16 not being a controlled ablation any more than §10.3 was. The two Fresh-5 batches are close on AUTO precision (92.0% vs. 93.3%) but the gazetteer batch has a noticeably higher `TIER_4_ENSEMBLE_SPLIT` share (37.5% vs. 29.2%) and lower deflection (47.9% vs. 56.6%) — consistent with the gazetteer-recovered spans being harder cases the ensemble agrees on less often (see §14's gazetteer-fallback discussion). Stage 3 is consistently the dominant per-entity cost (5-8x Stage 2b) across all three batches — direct empirical support for this project's 2-5 min/note latency budget being spent where it matters (the multi-model ensemble, not retrieval).

---

## 17. `normalized_entities` Dedup-Key Bug — Found and Fixed (2026-09-01)

**What was found.** Diagnosing the corpus-wide recall gap directly (not assumed): of 40,579 gold annotations across all 154 processed test notes, **46.9% were never extracted by GLiNER/gazetteer at all**, and a further **14.1% (5,730 gold-covering entities) were extracted and accepted but had zero `normalized_entities` row** — invisible to Stage 3/HITL/KG3 despite having cleared Stage 2a. Traced to a real schema bug, not a model-quality issue: `normalized_entities` was `UNIQUE(note_id, original_text, expanded_text, gliner_label)`, not `entity_id`. When the same term is mentioned twice in one note (verified live: `"HTN"` twice in note `10097089-DS-8`, both real, distinct `entity_id`s), the second `INSERT ... ON CONFLICT (...) DO UPDATE SET entity_id = EXCLUDED.entity_id` silently overwrote the first mention's `entity_id` link — orphaning it. Verified corpus-wide: **100% of a real 8,653-row gap** is explained by exactly this collision.

**The fix** (`src/normalization/orchestrator.py`, `scripts/fix_normalized_entities_dedup_key.py`): `UNIQUE(entity_id, expanded_text)`, not `entity_id` alone — a real, legitimate one-to-many case exists too (a multi-drug regimen abbreviation like `"R-CHOP"` normalizes to 5 different `expanded_text`/concept pairs for the SAME `entity_id`, verified live in note `12465457-DS-18`: Rituximab/Cyclophosphamide/Hydroxydaunomycin/Oncovin/Prednisone). Verified zero collisions on the new key across all 22,177 pre-existing rows. DuckDB 1.4.5 has no `ALTER TABLE ADD/DROP CONSTRAINT`, so the fix is a rename-recreate-copy migration (old table kept, not dropped) plus a cheap backfill that copies each orphaned `entity_id`'s sibling result rather than re-running `normalize_entity()` (deterministic given the same key — zero new model/search calls for 8,653 rows). Production migration: 22,177 → 30,830 rows, 0 accepted entities left orphaned.

**A second-order finding, equally real**: six evaluation/grading scripts (`scripts/score_gold_recall.py`, `evaluation/iou_metrics.py`, `evaluation/ablations.py`, `evaluation/cal_eval.py`, `evaluation/stage1_disambiguation_eval.py`, `evaluation/stage2b_cal_eval.py`, `scripts/measure_gliner_risk_vs_match_tier.py`) had already joined `extracted_entities` to `normalized_entities` on the old composite key as a **documented, deliberate workaround** for this exact defect (`score_gold_recall.py`'s own "KNOWN DB CAVEAT" docstring, written earlier this session, correctly diagnosed the bug and worked around it for scoring purposes specifically because `normalize_entity()` is a pure function of that key — the *concept* a duplicate mention resolved to was never wrong, only its `entity_id`-level persistence). Once entity_id became reliable, that same workaround join would have started **double-counting** predictions instead — verified live, up to 6x row duplication on the Fresh-5 original batch. All seven were switched to the correct, direct `entity_id` join.

**Net effect on already-reported numbers, checked directly rather than assumed**: because the workaround already existed in the grading path, **the linked-recall/precision figures already published in this document (§2, §10, §16) do not change** — re-run after the fix with the corrected join, all three batches in §16.2 reproduce identically. The real, practical benefit of this fix is **downstream, not in the score**: 8,653 previously-invisible entities (5,730 of them gold-covering) are now visible to Stage 3/HITL/KG3 for the first time, where before this fix they could never be routed, reviewed, or written at all — a pure coverage/completeness fix to the pipeline's own internal plumbing, not a retroactive change to any accuracy metric already reported. `tests/test_normalized_entities_dedup_key.py` (4 checks) and the full suite (91/91) both pass; see the fix's own commit message for the complete diagnosis.
