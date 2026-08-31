# ConsensusCalibrator — Complete Technical Reference

**Module**: `src/mollm_tier_calibrator.py` · **Class**: `ConsensusCalibrator` · **Feature set version**: 2 (was 1 before 2026-08-30 — §1–§16 below describe the original 16-feature module and are otherwise unchanged from when they were written; §17 is new and documents the 17th feature and its own, separately-verified retrain)
**Model artefact**: `models/consensus_calibrator_v1.pkl` (`DEFAULT_MODEL_PATH`, derived from the module's own location so the fitting script and every production call site cannot point at different files)

**Sources**: four source files, all read in full, plus the fitted model artefacts themselves (`models/consensus_calibrator_v1.pkl` and its pre-retrain backup, unpickled and reported in §14) — `src/mollm_tier_calibrator.py` (the module), `evaluation/tier_gate_cal_eval.py` (labelling, splitting, fitting, threshold sweep), `scripts/retrain_calibrator_full_corpus.py` (the retrain wrapper), and `src/mollm_tier_gate.py` (both call sites, the thresholds, the traps). Everything in §1–§15 is verified against that source or read from the artefact. §16 states what remains uncovered and lists seven questions the source itself does not settle.

**Independently re-verified 2026-08-27** (this session, before saving): `FEATURE_NAMES` order and comments, `TIER1_CONFIDENCE_FLOOR`/`CALIBRATED_AUTO_THRESHOLD` values, both trap regexes, and every coefficient/intercept/`n_iter_`/`training_split`/`code_version`/`n_training_examples`/`n_training_notes` value for BOTH the production model and its `.bak_2026-08-20` predecessor were re-checked directly against the live source files and the actual unpickled artefacts. All matched exactly, byte-for-byte on the code snippets and to the printed decimal precision on every numeric value. Nothing in this document was taken on faith.

⚠️ **Do not confuse this with `MoLLMCalibrator`** in `src/mollm_calibrator.py` — a *superseded* 13-feature module belonging to the old gate, with `min_examples=30`. `ConsensusCalibrator` was deliberately written from scratch without reference to it, verified by grep that no `MoLLMCalibrator` / `mollm_ensemble` / `mollm_calibrator` references remain. Figures quoted for one do not apply to the other.

---

## 1. What problem it solves

`route_tier()` treats ensemble agreement as **binary**: 3/3 unanimous `SUPPORTED_1` above the confidence floor is trusted; anything less than unanimous is routed straight to `TIER_4_ENSEMBLE_SPLIT` and sent to a human — with no distinction between *"the majority is clearly right and one model is a weak outlier"* and *"the models are genuinely divided."*

Measured cost of that blunt rule: **`TIER_4_ENSEMBLE_SPLIT` is ~44.8% of all decisions (175/391)**. Nearly half of everything the gate sees is discarded to review by one rule, and direct inspection showed some of those splits have a demonstrably correct majority answer.

The calibrator gives that all-or-nothing question a **graded, evidence-based answer** for exactly the population that would otherwise be discarded with no further judgment. It estimates **P(the top candidate is correct)** from the *shape* of the disagreement plus independent provenance about how the entity was processed.

**It can only promote a case that would otherwise go to review — it can never demote a decision, and it is never consulted on a path that was already going to be automatic.** Tier 1, Tier 3 and the four free pre-checks reach their verdicts without it. It is asked exactly two questions, both of the form *"this was heading to a human — should it be?"*: once for a split vote (§2, call site 1) and once for a unanimous re-rank (call site 2). Its hard safety checks are upstream of it and it cannot override them.

---

## 2. Where it sits — **two** call sites, not one

```
entity → qualifier precheck → Tier 3 fast path → lab fast path → Tier 5 precheck
                                                          │   (all four free: no model call)
                                              run_two_step_ensemble()  ×3 models
                                                          │
                                          drop ERROR / degenerate votes → `usable`
                                          unanimous = len(usable)==3 and top_count==3
                                                          │
   ┌──────────────────────────────────────────────────────┼───────────────────────────────┐
   │ unanimous SUPPORTED_1        │ unanimous RE_RANK_N    │ unanimous NONE_CORRECT │ else │
   │                              │                        │                        │      │
   │ hard trap? ─yes→ TIER_4      │ ── CALL SITE 2 ──      │ → TIER_5               │ ──   │
   │        │ no                  │ hard trap? ─yes→ HITL  │   (never scored)       │  │   │
   │ conf < 0.70 → HITL           │        │ no            │                        │  │   │
   │        │                     │ calibrator.score()     │                        │  │   │
   │        ↓                     │   ≥0.72 → TIER_2B      │                        │  ↓   │
   │   TIER_1 (AUTO)              │   <0.72 → TIER_2 →HITL │                    CALL SITE 1│
   └──────────────────────────────┴────────────────────────┴────────────────────────┴──────┘
                                                                                       │
                                              plurality names a candidate? ─no─→ TIER_4 │
                                                          │ yes                          │
                                              hard trap? ─yes→ TIER_4 (score NEVER called)
                                                          │ no
                                              calibrator.score(context)
                                                    ≥ 0.72 → TIER_1B_CALIBRATED_AUTO_VALIDATED
                                                    < 0.72 → TIER_4_ENSEMBLE_SPLIT (review)
                                                    None   → treated exactly as "no calibrator";
                                                             existing routing, unchanged
```

**Call site 1** (`route_tier()`, the non-unanimous branch) is the one the calibrator was fitted and validated for. It promotes to `TIER_1B_CALIBRATED_AUTO_VALIDATED`, which **is** in `AUTO_TIERS`.

**Call site 2** (the unanimous-re-rank branch, added 2026-08-20) reuses the same helper and the same threshold to promote a Tier 2 decision to `TIER_2B_CALIBRATED_AUTO_RESOLVED`. Tier 2 is otherwise routed to review wholesale — 16.2% clean-span precision (11/68) on the fresh25 re-measurement, root-caused to the fact that **100% of `TIER_2_AUTO_RESOLVED` decisions in the database carry `is_ambiguous = True`**. Tier 2 requires all three models to re-rank *away* from retrieval's own top candidate, which only happens when that candidate already looked shaky, so unanimity there more likely reflects three models sharing a bias than three models independently verifying.

⚠️ **`TIER_2B` is deliberately *not* in `AUTO_TIERS`**, pending shadow validation — the calibrator was fitted on split-vote examples, a materially different feature distribution from Tier 2's. Note the consequence: a `TIER_2B` promotion sets `mollm_routing_decision = "AUTO_VALIDATED"` while its tier stays outside `AUTO_TIERS`, so a downstream consumer filtering on the routing decision and one filtering on the tier set will disagree about whether that decision was automatic. See §16 — this is worth confirming before quoting any AUTO count.

Both call sites go through one shared helper, `_score_with_calibrator()`, factored out precisely so the trap check and the prior-confirmation lookup cannot drift apart between them.

```python
def _score_with_calibrator(entity, model_results, candidate_index, calibrator, conn, conn_factory):
    trapped, trap_reason = _fragile_shorthand_trap(entity, candidate_index, entity.get("candidates") or [])
    if trapped:
        return {"trapped": True, "trap_reason": trap_reason,
                "calibrated_score": None, "prior_count": None}
    chosen_concept_id = candidates[candidate_index - 1].get("omop_concept_id")   # the candidate being promoted
    prior_count = count_prior_confirmations(conn, entity.get("original_text"), chosen_concept_id)
    context = build_feature_context(entity, model_results, prior_count)
    return {"trapped": False, "calibrated_score": calibrator.score(context), "prior_count": prior_count}
```

Two details worth having: the concept id used for the prior-confirmation lookup is **the candidate the gate is about to promote**, not the one retrieval originally ranked first — so feature 16 asks about the decision actually being made. And `conn_factory` (a zero-arg callable returning a fresh connection) is preferred over a long-lived `conn`: this lookup is the *only* database access anywhere in `route_tier()`, and it happens after the three LLM calls have already returned, so passing an open connection held DuckDB's single-writer lock for the whole ensemble call and locked the Streamlit UI out for the duration of Stage 3.

`CALIBRATED_AUTO_THRESHOLD = 0.72` and `TIER1_CONFIDENCE_FLOOR = 0.70` both live in the gate, not in this module. They are **deliberately not the same constant** even though the numbers are close: one is a trained model's P(correct) over a whole disagreement pattern, the other a raw mean logprob confidence on an already-unanimous vote. Conflating them would let one threshold's tuning silently drag the other's meaning along.

---

## 3. The 16 features — exact definitions

`FEATURE_NAMES` is a **fixed-order** list. A saved model's learned weights are only meaningful paired with this exact order, so adding, removing or reordering a feature requires bumping `FEATURE_SET_VERSION` and refitting — never a quiet reshuffle.

All values are floats in [0, 1]. Every one has a safe default, so a context missing fields still produces a valid vector rather than raising.

⚠️ The **Significance** column below states what each feature is evidence *for* or *against* on design grounds. It is **not** the learned direction — that lives in the fitted coefficients, whose measured values are in §14. Do not assert a sign you have not read from the model.

### 3.1 Group A — vote consensus shape (features 1–7)

Computed over **usable votes only** (see §4). Together these answer: *how strong is the majority, and how much evidence is it built on?*

| # | Feature | Exact computation | Significance and role |
|---|---|---|---|
| 1 | `frac_supported_1` | `n_top / n_usable`, else `0.0`; `n_top` counts verdicts equal to `SUPPORTED_1` | **The primary majority signal, and the feature the whole module exists to grade.** The blunt rule treats 2-of-3 and 1-of-3 identically; this separates "a clear majority with one weak outlier" from "genuinely divided". Evidence *for* the top candidate |
| 2 | `frac_rerank_same_target` | `n_rerank_agree / n_usable`, else `0.0`. `n_rerank_agree` is the **modal** count among verdicts beginning `RE_RANK_TO_CANDIDATE_` | **Coordinated dissent.** Models converging on the *same* alternative is a far stronger objection than models merely disagreeing. Evidence *against* the top candidate — and specifically evidence that the right answer is elsewhere in the pool |
| 3 | `frac_none_correct` | `n_none / n_usable`, else `0.0` | **Wholesale rejection**, a different failure mode from feature 2: nothing in the pool is right, rather than the wrong item is ranked first. Separating the two lets the model weight "retrieval mis-ranked" and "retrieval missed entirely" differently |
| 4 | `frac_usable_votes` | `n_usable / 3.0` | **How much evidence features 1–3 are built on.** Without it, 2-of-2 and 2-of-3 are indistinguishable — both give `frac_supported_1 = 1.0` and `0.67`. This is the discount factor that lets the model trust a vote shape less when a model errored or degenerated |
| 5 | `mean_logprob_confidence` | `mean(confidences)`, else `0.0`; `confidences` are the non-`None` `logprob_confidence` values from usable votes | **Aggregate certainty — measured from token logits, not self-reported.** This is the distinction that matters: the project measured self-reported confidence as a per-model constant and refuses to route on it, but a log-probability is an observation of the decode, not the model's opinion of itself |
| 6 | `min_logprob_confidence` | `min(confidences)`, else `0.0` | **The weakest link.** A high mean can conceal one barely-committed vote. Where feature 5 asks "how confident overall", this asks "how confident is the least confident model" — the conservative reading of the same evidence |
| 7 | `confidence_spread` | `max(confidences) − min(confidences)`, else `0.0` | **Disagreement in strength rather than direction.** Two models can both say `SUPPORTED_1` while one is near-certain and the other barely commits. A tight cluster is corroboration; a wide spread means the majority is carried by one model |

**Note on 5–7**: the confidence subset is computed independently of the usable-vote subset. A model may return a usable verdict with `logprob_confidence = None`; it contributes to features 1–4 but not to 5–7.

**Note on feature 2**: deliberately narrow. Three models re-ranking to *three different* candidates yields `1/3`, not `1.0` — disagreement about *where* to go is not agreement.

### 3.2 Group B — retrieval and provenance (features 8–15)

Drawn from the entity record, **entirely independent of the ensemble**. Their shared premise: *how* a concept was reached is evidence about whether it is right, regardless of how confident the models are about it.

| # | Feature | Exact computation | Significance and role |
|---|---|---|---|
| 8 | `match_tier_is_exact_or_synonym` | `1.0` if `entity["match_tier"]` is `"1 (Exact)"` or `"2 (Synonym)"`, else `0.0` | **Lexical versus semantic provenance.** A literal name or synonym match is a categorically stronger prior than an embedding neighbour. Collapsed to binary because the meaningful boundary is "the vocabulary contained this string" versus "something looked similar" |
| 9 | `top_candidate_similarity_score` | `candidates[0]["similarity_score"]`, else `0.0` | **Degree within the category.** Complements feature 8: tier says *which route*, this says *how well it scored on that route*. A Tier 3 match at 0.88 and one at 0.73 are both semantic, and are not equally trustworthy |
| 10 | `is_ambiguous` | `1.0` if `entity["is_ambiguous"]` truthy | **Retrieval's own warning that the pool is contested.** Load-bearing: all **259 of 259** Tier 2 decisions in the database carry this flag, which is precisely why unanimity in that population reflected shared bias. This feature marks the exact conditions under which agreement means least |
| 11 | `domain_conflict` | `1.0` if `entity["domain_conflict"]` truthy | **A type-level inconsistency, independent of score.** The candidate's OMOP domain disagrees with the extracted label — a Condition span resolving to a Procedure concept. A high-similarity candidate can still be the wrong *kind* of thing, and similarity alone cannot see that |
| 12 | `resolved_via_value_stripped_fallback` | `1.0` if `"value_stripped_from_"` in `entity["normalized_from"]` | **The span only matched after its text was altered** — a lab value suffix stripped (`MCHC-31` → `MCHC`). Every transformation between the note and the lookup is an opportunity to lose meaning, so the route is recorded as a risk marker |
| 13 | `resolved_via_original_text_fallback` | `1.0` if `"original_after_expanded_failed"` in `normalized_from` | **Preprocessing's own best guess failed to ground.** The expanded form found nothing and the raw text was retried — so the match was made on text the pipeline had already judged to be the less useful representation |
| 14 | `resolved_via_acronym_escalation` | `1.0` if `"+acronym_mollm"` **or** `"+acronym_cache"` in `normalized_from` | **An abbreviation was resolved by a model or a cache, not deterministically.** Directly calibrated by measurement: acronym escalation graded **34.3–36.1% precision** at corpus scale and is switched off by default, so anything that reached its concept through that route inherits a known-unreliable step |
| 15 | `expansion_ambiguous` | `1.0` if `entity["expansion_ambiguous"]` truthy | **Upstream ambiguity that may never have been resolved.** The abbreviation carried more than one dictionary meaning. Distinct from feature 14: this says the ambiguity *existed*, that one says an escalation mechanism was *used on it* |

**Features 12–14 are the "fragile fallback" flags**, parsed by substring match against a single `normalized_from` provenance string. Each marks a resolution route carrying measurably lower trust than a direct match. Note that they are **not mutually exclusive** — an entity can arrive through more than one, and the model sees each independently.

⚠️ **Feature 15 is constant zero in every training example, and you should know this before an examiner finds it.** `build_labeled_examples()` reconstructs the entity dict from `normalized_entities`' own columns, and that table does not carry `expansion_ambiguous`, so the field is hardcoded:

```python
"expansion_ambiguous": False,  # not carried on normalized_entities; safe default
```

A constant column has no discriminative signal, so feature 15's learned coefficient is whatever L2 regularisation leaves at zero — it contributes nothing. **This is confirmed, not predicted: §14.1 reports β = exactly 0.0000 in both fitted models.** The same is true of feature 14 for a different reason, and `frac_usable_votes` is near-constant too — see §14.2. The honest reading is **sixteen features defined, thirteen carrying weight**.

The mitigating fact, and it is a real one: `tier5_precheck()` routes any entity with `expansion_ambiguous` and no `mollm_escalation_resolved` straight to `TIER_5_TRUE_AMBIGUITY` **before the ensemble runs at all**, so such an entity rarely reaches the calibrator in the first place. The feature is closer to redundant than to broken. But "redundant with an upstream gate" and "trained" are different claims, and only the first is true. Say the first.

### 3.3 Group C — confirmation history (feature 16)

| # | Feature | Exact computation | Significance and role |
|---|---|---|---|
| 16 | `prior_confirmation_count` | `min(raw_count, 10) / 10.0` — capped at 10, scaled to [0,1] | **The only feature carrying information from outside this decision.** It asks whether this exact (entity text, concept) pairing has been confirmed before, making the audit trail an input rather than an archive — and it is the calibrator's half of the active-learning loop. Saturates deliberately: 40 confirmations and 10 are treated identically, because the signal is "this has been seen and accepted repeatedly", not "how popular is it" |

`raw_count` is supplied by the caller from a database lookup (§6), which carries an important caveat about what it can currently measure.

**Feature 16 was the one feature explicitly tested for removal, and it survived.** `fit_and_report()` takes an `ablate_indices` argument that zeroes chosen feature positions in every vector before fitting — for a linear model, equivalent to dropping the feature, since a constant-zero column carries no signal into `predict_proba()`. The main evaluation runs exactly one ablation, on `prior_confirmation_count`, and records the outcome in the code:

> dropping `prior_confirmation_count` doesn't fix the coronary false positives and measurably costs precision elsewhere

Two things follow. The feature is **kept on evidence, not assumption**. And the ablation is what proved the coronary-abbreviation false positives were *not* a `prior_confirmation_count` artefact — which is what redirected the diagnosis to the retrieval layer and produced the hard trap in §12 instead of more feature engineering.

---

## 4. `usable_votes()` — the load-bearing filter

```python
def usable_votes(model_results):
    return [m for m in (model_results or [])
            if m.get("verdict") not in (None, "ERROR")
            and not m.get("degenerate_generation")]
```

A model that errored or produced a degenerate generation **contributes no evidence either way**. It is excluded, not counted as a silent `NONE_CORRECT` — matching how `route_tier()` itself already treats a degenerate vote as "not a real disagreement."

This matters because it is the difference between *"two models supported it and one failed to respond"* (`frac_supported_1 = 1.0`, `frac_usable_votes = 0.67`) and *"two models supported it and one rejected it"* (`frac_supported_1 = 0.67`, `frac_usable_votes = 1.0`). Feature 4 exists precisely so the model can tell those two situations apart.

---

## 5. Worked example — "claudication"

Note `1228016-DS-9`. Verdicts: llama3.2:3b `SUPPORTED_1` (0.742) · qwen2.5:3b `NONE_CORRECT` (no logprob recorded) · phi4-mini `SUPPORTED_1` (0.754). Retrieval returned one candidate, `Intermittent claudication` (442774), match tier `2 (Synonym)`, similarity 1.000.

| # | Feature | Value | Derivation |
|---|---|---|---|
| 1 | `frac_supported_1` | **0.6667** | 2 of 3 usable |
| 2 | `frac_rerank_same_target` | **0.0** | no re-rank verdicts |
| 3 | `frac_none_correct` | **0.3333** | 1 of 3 |
| 4 | `frac_usable_votes` | **1.0** | all three produced usable verdicts |
| 5 | `mean_logprob_confidence` | **0.748** | (0.742 + 0.754) / 2 — qwen contributes no confidence |
| 6 | `min_logprob_confidence` | **0.742** | |
| 7 | `confidence_spread` | **0.012** | 0.754 − 0.742 — the two supporting votes are tightly clustered |
| 8 | `match_tier_is_exact_or_synonym` | **1.0** | "2 (Synonym)" |
| 9 | `top_candidate_similarity_score` | **1.000** | |
| 10–15 | provenance flags | *not in the logged extract* | require `normalized_from`, `is_ambiguous`, `domain_conflict`, `expansion_ambiguous` from the entity record |
| 16 | `prior_confirmation_count` | **0.0** | no completed human review exists anywhere in the system |

⚠️ **`mean_logprob_confidence` (0.748) is not `composite_confidence` (0.746).** They are separate computations — `composite_confidence` is produced by the ensemble's own `combine()` and stored on the decision; feature 5 is recomputed here from usable votes. Do not present them as the same number.

---

## 6. `count_prior_confirmations()` — the only DB-dependent input

Two independent aggregations, summed. Each is wrapped in its own `try/except` that swallows errors, so a missing or malformed table contributes zero rather than failing the call.

**Query 1 — the pipeline's own confirmed decisions**
```sql
SELECT count(*) FROM mollm_tier_gate_decisions d
JOIN extracted_entities  e ON e.entity_id = d.entity_id
JOIN normalized_entities n ON n.entity_id = d.entity_id
WHERE lower(trim(e.original_text)) = lower(trim(?))
  AND n.omop_concept_id = ?
  AND d.mollm_routing_decision IN ('AUTO_VALIDATED', 'AUTO_RESOLVED')
```

**Query 2 — completed human reviews**
```sql
SELECT count(*) FROM hitl_review_queue h
JOIN extracted_entities e ON e.entity_id = h.entity_id
WHERE lower(trim(e.original_text)) = lower(trim(?))
  AND h.corrected_concept_id = ?
  AND h.reviewer_decision IN ('APPROVED', 'CORRECTED')
```

Matching is on **case-insensitive, whitespace-trimmed** `original_text` paired with the concept id. Run fresh each time, never cached — no reliable cache-invalidation story exists yet. Returns `0` (never `None`, never raises) on a missing connection, missing arguments, or any DB error, because *an entity with no confirmation history is a normal, common state, not a failure*.

⚠️ **The module's own caveat, quoted because it is the honest framing**: this counts DuckDB's record of confirmed or *would-be-confirmed* decisions, **not real knowledge-graph writes** — KG3 ingestion runs dry-run throughout, so there is no live graph to query. *"Treat this as evidence with the same caution as anything else derived from dry-run data: it needs to be validated on a held-out split like every other feature here, not assumed trustworthy just because it looks like independent confirmation."*

**Practical consequence today**: Query 2 returns 0 for every entity, because zero human reviews are complete anywhere in the system. Feature 16 is currently carried entirely by Query 1 — the pipeline confirming its own prior decisions. That is a self-referential signal, and it is why the docstring flags it rather than trusting it.

---

## 7. The model

```python
def _build_model():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
```

| Hyperparameter | Value | Why |
|---|---|---|
| `max_iter` | 1000 | convergence headroom on 16 features |
| `C` | 1.0 | default L2 regularisation strength; not tuned |
| `class_weight` | **`"balanced"`** | **load-bearing, and in the safety-relevant direction.** The measured base rate of this population is **70.4% correct** (470 of 668 gradable Tier-4 decisions), so the *minority* class is `label = 0` — the wrong ones. An unweighted fit would minimise loss by leaning toward "correct" and under-learning exactly the cases that must not be promoted. Balancing up-weights the incorrect examples, which is the direction a gate that can write unreviewed needs |

`_build_model()` is the **single construction site**, so every caller — production fit, cross-validation, an ablation script — shares identical hyperparameters by construction rather than by convention.

**scikit-learn is never imported at module import time** — only inside `_build_model()` and `load()`. Importing `mollm_tier_calibrator` in an environment without scikit-learn works fine and yields an untrained calibrator.

---

## 8. `score()` — the exact computation

```python
def score(self, context):
    if self.model is None:
        return None
    try:
        vector = [featurize(context)]
        probabilities = self.model.predict_proba(vector)[0]
        classes = list(self.model.classes_)
        positive_index = classes.index(1) if 1 in classes else (
            1 if len(probabilities) > 1 else 0)
        return round(float(probabilities[positive_index]), 6)
    except Exception:
        return None
```

Mathematically, for feature vector **x** and fitted coefficients **β**:

**z = β₀ + Σᵢ βᵢxᵢ**  →  **P(correct) = σ(z) = 1 / (1 + e⁻ᶻ)**

Three implementation details that matter:

1. **The positive class is located by value, not by position.** `classes.index(1)` finds where label `1` actually sits in `model.classes_` rather than assuming index 1. This is correct behaviour if a fit ever produced an unusual class ordering.
2. **Rounded to 6 decimal places** before returning — so a stored `calibrator_score` is reproducible rather than carrying float noise.
3. **Every failure returns `None`, never raises.** An untrained model, a malformed context, a scoring exception — all produce `None`, and the caller must treat `None` *exactly* as "no calibrator available", leaving existing HITL routing unchanged. This is stated three times in the module's own docstrings; it is the module's central safety property.

---

## 9. Fitting — five guards

`fit(contexts, labels, min_examples=100)` featurises then delegates to `fit_vectors()`. The guards, in order:

| Guard | Behaviour |
|---|---|
| Length mismatch (`fit`) | `ValueError` — contexts and labels must match |
| **Below `min_examples`** | `ValueError`. **Default is 100**, chosen deliberately rather than a token floor: *"this model's output can promote an entity straight to an unreviewed write, so a small, possibly noisy fit is not an acceptable default"* |
| Length mismatch (`fit_vectors`) | `ValueError` |
| Single class | `ValueError` — a classifier cannot be fit when every label is the same value |
| **Positives per feature < 5** | `warnings.warn`, does **not** refuse. Computes `n_positive / 16`; if under 5 it warns that *"this model can fit noise — check its AUROC against a held-out split before trusting it"* |

The last guard is advisory rather than blocking, on the reasoning that the real safety check is held-out AUROC, not a rule of thumb. With 16 features, the 5-per-feature convention wants **≥ 80 positive examples**.

⚠️ **The production training path never calls `fit()`.** `fit_and_report()` calls `fit_vectors()` directly, with `min_examples=100` passed explicitly at the call site:

```python
calibrator.fit_vectors([e["vector"] for e in train], [e["label"] for e in train],
                       min_examples=100)
```

Vectors are featurised once when the labelled examples are built, so a re-split or an ablation does not re-run `featurize()` per example. The consequence for reading the code: guard 1 (the `fit()` length check) is never exercised in practice; guards 3–5 are.

---

## 10. The training procedure

### 10.1 Two entry points, one machinery

| | `evaluation/tier_gate_cal_eval.py` | `scripts/retrain_calibrator_full_corpus.py` |
|---|---|---|
| Date | 2026-08-17 | 2026-08-20 |
| Note population | hardcoded `NOTE_IDS` — the 31-note overnight corpus | `SELECT DISTINCT note_id … WHERE tier = 'TIER_4_ENSEMBLE_SPLIT'` — 114 notes |
| Saves by default | **yes**, unconditionally at the end of `main()` | **no** — diagnostic unless `--save` *and* it beats baseline |
| Runs an ablation | yes (`prior_confirmation_count`) | no |
| `training_split` recorded | `overnight_2026-08-17_train` | `full_corpus_114_notes_2026-08-20` |

The retrain script imports `build_labeled_examples()`, `split_by_note()`, `fit_and_report()`, `threshold_sweep()` and `FEATURE_NAMES` from the evaluation module rather than re-deriving them. **Only the note population differs.** That is the reason the two runs are comparable at all.

### 10.2 The training population — and the two tiers deliberately excluded

Only `TIER_4_ENSEMBLE_SPLIT` decisions are used. The module's own reasoning for each exclusion:

- **Not the AUTO tiers** — *"`route_tier()` never consults the calibrator for those; training on them would teach the model an input distribution it never sees at inference."*
- **Not `TIER_5_TRUE_AMBIGUITY`** — a unanimous `NONE_CORRECT` plurality has no candidate to promote, so `route_tier()` never reaches the calibrator-consultation block for that shape.

This is the deployment population by construction: the model is fitted on exactly the cases it will be asked to score.

⚠️ **Note what this means for call site 2.** `TIER_2B` promotions are scored by a model that was fitted only on split votes. Tier 2's population is unanimous and 100% `is_ambiguous` — a materially different feature distribution. The code says so explicitly, which is why `TIER_2B` is held out of `AUTO_TIERS`. Do not describe the calibrator as validated for that path; it is not.

### 10.3 The label rule — exact

```python
pred_code = vocab.snomed_code_for_concept(concept_id) if concept_id else None
gold_code = g0["concept_id"]
is_gold_error = (note_id, str(gold_code)) in KNOWN_GOLD_ERRORS
label = 1 if ((pred_code is not None and str(pred_code) == str(gold_code))
              or is_gold_error) else 0
```

In words: **the plurality-vote candidate is crosswalked from its OMOP concept id to a SNOMED code, and that code must string-match gold's.** Not span overlap, not a hierarchy relation, not a similarity threshold — exact code equality after crosswalk.

Three things follow, and all three are worth being able to say out loud:

1. **The candidate labelled is the one the gate would actually promote** — `plurality_candidate_index()`, the same function `evaluation/grade_overnight_corpus_run.py` uses for its "Tier 4 shadow precision" figure. The label definition and that already-reported number therefore cannot silently diverge. This is deliberate reuse, stated as such in the docstring.
2. **`KNOWN_GOLD_ERRORS` can force a label to 1** — a curated list of gold annotations judged wrong. It is honest to have one, and it is a fair question at a viva: *how many, and who adjudicated them?* Be ready. It is also curated **only for the original 31 notes**, which the retrain script's own docstring flags as *"a known, stated limitation rather than a silent one"* — so any gold quirk specific to one of the 83 newer notes is uncorrected.
3. **A parent/child SNOMED relation counts as wrong.** A prediction one level up the hierarchy from gold scores 0, exactly like an unrelated concept. That makes the base rate a conservative measure, not a flattering one.

### 10.4 What is excluded — and the distinction that matters

Five exclusion counters, tracked and printed:

| Excluded | Meaning |
|---|---|
| `no_gold_overlap` | no gold annotation overlaps this span |
| `compound_span` | more than one gold annotation overlaps — cannot attribute a single correct answer |
| `narrower_than_gold` | the predicted span is shorter than gold's |
| `no_candidate` | the plurality verdict is `NONE_CORRECT` — nothing to promote |
| `candidate_index_out_of_range` | the plurality index does not exist in the stored candidate list |

⚠️ **These are dropped as *unlabelable*, not scored as 0.** The docstring states it directly. This is the correct choice — scoring an unattributable case as wrong would train the model to distrust a shape that is merely unmeasurable — but it does mean the **base rate is computed over gradable cases only**, and the calibrator has never seen a compound span in training. `route_tier()` at inference has no such filter, so compound spans *are* scored in production, by a model with no training signal for them. That is a genuine train/inference asymmetry; §15 lists it as an open question rather than pretending it away.

### 10.5 The split — deterministic, by note, every fourth

```python
def split_by_note(examples, holdout_every=4):
    notes = sorted({e["note_id"] for e in examples})
    val_notes = set(notes[holdout_every - 1::holdout_every])
    train_notes = set(notes) - val_notes
```

**Roughly 75/25, by note, with no random seed.** Sorted note ids, every 4th to validation. Two justifications, both given in the source:

- **By note, not by row** — *"entities inside one discharge note are not independent draws"*, and the load-time leakage guard checks note-id overlap specifically, so partitioning any other way would defeat that guard's purpose.
- **Deterministic, not random** — a re-run reproduces the same split without a stored seed.

For the 2026-08-17 fit, 31 notes gives **24 train / 7 validation notes** (sorted indices 3, 7, 11, 15, 19, 23, 27), yielding 542 train and 126 validation examples out of 668.

### 10.6 `fit_and_report()` — and a correction to what `respect_hard_traps` does

```python
def fit_and_report(train, val, train_notes, ablate_indices=(), label="FULL",
                   save_path=None, fp_threshold=0.65, respect_hard_traps=False):
```

⚠️ **`respect_hard_traps` does not affect fitting at all.** An earlier version of this document said trapped entities were excluded from training. That is wrong. The flag is passed only to `threshold_sweep()` and to the false-positive listing — it changes what is *reported*, not what is *learned*. The model is fitted on **all** labelled examples, trapped ones included; the trap is applied at routing time by `route_tier()` itself.

The evaluation's own comment is unambiguous:

> the trap changes ROUTING, not training, so this reuses `val_full`'s already-fitted scores and just re-sweeps with the trap's exclusion applied, exactly mirroring what `route_tier()` itself now does

and the save message reads *"saved FULL (untrapped-training, trap applied at inference by route_tier() itself)"*.

This is defensible — arguably better than the alternative, since it keeps the trap a routing rule that can be changed without a refit, and it means no future fit can learn its way around the gate. But the accurate statement is **"trained on everything, trapped at inference"**, not "trained without traps". Say the first.

`threshold_sweep()` reports, per threshold, `n_promoted` / coverage % of the validation Tier-4 population / precision among promoted — *"exactly what `CALIBRATED_AUTO_THRESHOLD` trades off in `route_tier()`."* The sweep runs 0.50 to 0.95 in steps of 0.05, and false positives at `fp_threshold` are printed individually with their note id, text, score, vote counts and prior count — so a bad promotion is inspectable, not just counted.

### 10.7 The ablation — the one feature tested for removal

`ablate_indices` zeroes chosen positions in every vector before fitting. For a linear model this is equivalent to dropping the feature: a constant-zero column has no discriminative signal and contributes nothing to `predict_proba()`. It is done this way rather than by editing `FEATURE_NAMES` so that a diagnostic never touches the production feature set or its version number.

`main()` runs exactly one: `prior_confirmation_count`. The recorded outcome —

> dropping `prior_confirmation_count` doesn't fix the coronary false positives and measurably costs precision elsewhere

— did two jobs. It **kept feature 16 on evidence**, and it **redirected the coronary diagnosis away from the calibrator entirely**: if removing the suspect feature does not fix the false positives, the fault is not in the feature. That is what sent the investigation to the retrieval layer and produced a hard trap instead of more feature engineering.

### 10.8 The adoption gate

`BASELINE_AUROC = 0.74` — the production v1 calibrator's own validation AUROC.

| Condition | Verdict |
|---|---|
| `auc is None` | **INCONCLUSIVE** — val set too small or one-class |
| `auc > 0.74` | **BETTER** by *x* — candidate for adoption, pending review |
| `auc ≥ 0.73` | **ROUGHLY EQUIVALENT** — *"more data did not clearly help"* |
| otherwise | **WORSE** by *x* — *"do NOT adopt, matches the 2026-08-17 51-note finding"* |

**Diagnostic by default.** No overwrite unless `--save` is passed *and* `auc > BASELINE_AUROC`. Passing `--save` on a fit that failed prints *"--save was passed but the new fit did NOT beat baseline -- NOT saved."*

That branch is the mechanism behind "one retrain rejected, another accepted": the **51-note retrain on 2026-08-17 came out worse** and was not adopted; the **114-note retrain on 2026-08-20 beat 0.74** and became the current 0.845 production model.

⚠️ Note the asymmetry between the two scripts: `tier_gate_cal_eval.py`'s `main()` saves **unconditionally** at the end, with no baseline comparison. The adoption gate lives only in the retrain wrapper. Running the older evaluation module directly would overwrite the production artefact regardless of whether the new fit was any good.

### 10.9 What is recorded on adoption

```python
calibrator.save(DEFAULT_MODEL_PATH,
                training_note_ids=train_notes,          # the TRAIN split only
                training_split=f"full_corpus_{len(all_notes)}_notes_2026-08-20",
                code_version=f"full_corpus_retrain_{date.today().isoformat()}")
```

⚠️ **`training_note_ids` is the train split, not `all_notes`** — which is what makes the leakage guard in §11.3 both correct and non-punitive: a note used only for validation does not later trigger a refusal.

Saved metadata matches the Streamlit calibrator panel: `code_version: full_corpus_retrain_2026-08-20`, `training_split: full_corpus_114_notes_2026-08-20`, **82 training notes, 1,403 examples**, feature set version 1.

The connection is opened **read-only** with `max_wait_seconds=300`, so a retrain can never mutate the store it is measuring, and will wait out a batch job holding the write lock.

---

## 11. Persistence and the two guards that protect it

### 11.1 What is saved

`save()` pickles a dict containing: `model` · `feature_set_version` · `n_training_examples` · `feature_names` · `training_note_ids` · `training_split` · `code_version`.

`training_note_ids` is saved **because without that record a leakage check on a later run is impossible, not merely inconvenient.**

### 11.2 The feature-set version guard

```python
if data.get("feature_set_version") != FEATURE_SET_VERSION:
    return instance   # fresh, untrained
```

A model whose feature layout no longer matches the module is **refused, not misapplied**. Because the weight vector is only meaningful against the exact `FEATURE_NAMES` ordering, silently loading a stale model would apply each coefficient to the wrong feature — a failure that produces plausible numbers and is nearly undetectable downstream. `load()` also returns a fresh untrained instance on a missing file, corrupt pickle, or any exception whatsoever.

### 11.3 The leakage guard — exact behaviour

```python
ConsensusCalibrator.load(path, scoring_note_ids=[...], refuse_on_leakage=True)
```

| Condition | Behaviour |
|---|---|
| `scoring_note_ids` not supplied | No check performed |
| Model has **no recorded** `training_note_ids` | Prints *"this calibrator has no recorded training notes; leakage cannot be checked."* — proceeds, but the absence is announced rather than assumed clean |
| Overlap found, `refuse_on_leakage=True` (default) | Prints the overlapping note ids (first 5, then `...`), then **sets `self.model = None`** — degrading to untrained |
| Overlap found, `refuse_on_leakage=False` | Proceeds, printing *"these numbers are not reportable"* |

**This is the mechanism that fired on 3 of the 10 locked held-out notes**, which is why `TIER_1B` shows 0 decisions in the fresh-10 results, and why the 76.8% held-out figure validates retrieval and ranking but **not** the calibrator. That is not a defect; it is the guard working, and it is auditable because the training note ids were persisted.

---

## 12. The hard traps — exact implementations

Two traps, combined by one helper, checked **before** `score()` is ever called. When either fires the calibrator is not consulted at all — not scored and overridden, not called.

```python
def _fragile_shorthand_trap(entity, candidate_index, candidates):
    if _is_coronary_segment_trap(entity, candidate_index, candidates):
        return True, "coronary_segment_trap"
    if _is_short_alphanumeric_code(entity):
        return True, "short_alphanumeric_code_trap"
    return False, None
```

### 12.1 `_is_coronary_segment_trap()` — two ways to fire

```python
CORONARY_SEGMENT_TRAP_ABBREVIATIONS   = {"lad", "lcx", "lmca", "rca", "pda", "om", "plv"}
CORONARY_SEGMENT_TRAP_GENERIC_CONCEPTS = {"coronary artery structure"}
```

Fires if **either** the mention's own text (stripped, lowercased) is in the abbreviation set, **or** the candidate about to be promoted has `concept_name` equal to `"coronary artery structure"` — the specific observed failure shape, a named segment resolving to its own generic parent.

**Root cause, and this is the part to say:** the false positives were diagnosed as a **retrieval-layer** problem, not a calibrator one. SapBERT's embedding space does not reliably separate `Left circumflex coronary artery` from `Coronary artery structure`, so the ensemble is handed a muddy candidate list and splits. Confirmed three independent times — the calibrator's own validation false positives, Phase 4 acronym grading (LAD wrong every time), and the AUTO-tier grading pass (LCx twice resolving to the generic parent). **No amount of feature engineering on this calibrator fixes a bad candidate list**, which is precisely why it is quarantined rather than learned around.

### 12.2 `_is_short_alphanumeric_code()` — **two** regexes, not one

```python
SHORT_ALPHANUMERIC_CODE_RE = re.compile(r"^[A-Za-z]{1,2}[0-9]{1,2}$")   # S2, T1, V12
SHORT_ALPHA_CODE_RE        = re.compile(r"^[A-Z]{3,4}$")                # LAD, LCX, LMCA
```

| Regex | Added | Catches | Why |
|---|---|---|---|
| `^[A-Za-z]{1,2}[0-9]{1,2}$` | 2026-08-17 | S1–S4, T1/T2, V1–V6 | Same embedding-collapse shape as the coronary trap in a more general vocabulary: **S2** is a cardiac exam finding *and* the second sacral vertebra; **T1/T2** are thoracic vertebrae *and* MRI relaxation times *and* tumour stages. The **shape** is the risk signal, so it is caught by regex rather than an enumerated list |
| `^[A-Z]{3,4}$` | 2026-08-18 | LAD, RCA, LMCA, and abbreviations not yet enumerated | Added after `LMCA` reached **3/3 unanimous Tier 1 on the wrong candidate** ("Coronary artery stenosis" instead of "Structure of left main coronary artery"). Deliberately **case-sensitive, upper only** — a lowercase or Title-Case 3–4 letter word is far more likely to be ordinary prose than clinical shorthand |

The second regex is the forward-looking half: it catches the *next* collision without requiring it to be found the hard way and manually enumerated first.

### 12.3 The "elevate the gate" fix — the traps now guard Tier 1 too

Originally only the calibrator path ran these checks. That left a hole, and `LMCA` fell through it: a genuinely unanimous 3/3 vote on a wrong candidate sailed into `TIER_1_AUTO_VALIDATED` with no trap protection at all. Since 2026-08-18 the trap is checked **before** the confidence floor and **before** Tier 1 is granted:

```python
if unanimous and top_verdict == "SUPPORTED_1":
    trapped, trap_reason = _fragile_shorthand_trap(entity, 1, entity.get("candidates") or [])
    if trapped:
        return {... "tier": TIER_4_ENSEMBLE_SPLIT, "mollm_routing_decision": "HITL_REQUIRED" ...}
```

The reasoning in the source is the single best sentence in this codebase for an oral exam, and it generalises well beyond this project:

> unanimity was treated as proof the models got it right, but the failure mode here is that all three models see the SAME muddy, SapBERT-collapsed candidate list — **unanimous agreement on a bad candidate list is not evidence the candidate is correct**

That is also the answer to *"why is Tier 2 excluded from AUTO_TIERS?"* — same principle, different tier, and the 100%-`is_ambiguous` finding is the measured version of it.

---

## 13. Where `0.72` came from

This is the most-likely threshold question at a viva, and the answer is well documented.

### 13.1 The original fit and the first threshold

Fitted 2026-08-17 against a note-disjoint held-out split of the 31-note overnight corpus's `TIER_4_ENSEMBLE_SPLIT` population — **668 labelled examples, 70.4% base rate**, both hard traps active.

**Originally locked at 0.65**: 98.0% precision at 38.9% coverage on that validation set.

### 13.2 What broke it

A **genuinely fresh 5-note run**, outside both the training *and* validation notes, surfaced three false positives — clustered in a narrow band just above the threshold:

| Entity | Score | Caught by a trap? |
|---|---|---|
| `Tenotomy` | 0.6997 | ✗ — an ordinary word |
| `S2` | 0.704 | ✓ — the alphanumeric short-code trap now catches it regardless of score |
| `incontinence` | 0.70698 | ✗ — an ordinary word |

Two of the three are plain words no shape-based trap can catch. **Only a threshold change catches those.**

### 13.3 The re-measurement, and the cost that was accepted

Re-run on the original held-out validation set with the new alphanumeric trap also active:

| Threshold | Precision | Coverage | Promoted |
|---|---|---|---|
| 0.70 | **100%** | 17.5% | 22 / 126 |
| **0.72** | **100%** | **7.9%** | **10 / 126** |

**0.72 was chosen over 0.70 for one specific reason**: `incontinence` scored 0.70698, and 0.70 would still have promoted it. The extra 0.02 buys nothing on the validation set — both give 100% precision — and costs more than half the coverage. It was paid to exclude one observed, un-trappable false positive.

Projected corpus-wide: **~129 of 1,629** `TIER_4_ENSEMBLE_SPLIT` entities promotable at 0.72, down from ~634 at 0.65. The source calls this *"a materially smaller coverage win than the original Phase 6 estimate"* and asks for the trade-off to be re-confirmed at production scale. It has not been.

### 13.4 The traps are load-bearing for every number above

> without them, precision tops out around **89%** at any threshold — they're load-bearing for every number below

If asked *"could you get the same result with a threshold alone?"* — no. Measured. Without the traps no threshold reaches 100% precision on that validation set.

### 13.5 ⚠️ The threshold has not been re-derived since the refit

0.72 was chosen against the **2026-08-17** model's validation set (668 examples, 126 val). The **2026-08-20** retrain replaced that model with a different fit on a different corpus (114 notes, 1,403 examples, AUROC 0.845). The threshold was carried across unchanged, and the source's own warning is explicit:

> Re-validate before ever changing this: a refit on new data or a different held-out split can shift where 0.72 actually sits on the precision/coverage curve.

The refit happened; the re-validation did not. The measured consequence is visible in §14 — 0.72 gave **100%** precision on the old validation set and **89.5%** on the new one. That is not a contradiction between the two figures; it is the same threshold sitting at a different point on a different curve, exactly as warned. **Be able to say this before an examiner finds it.** It is a small, well-scoped piece of remaining work, not a flaw in the design.

---

## 14. The learned coefficients — read from the artefact

Extracted directly from `models/consensus_calibrator_v1.pkl` and its pre-retrain backup, and **independently re-extracted and cross-checked in this session (2026-08-27)** — every value below matched the earlier report exactly.

| | current (production) | previous (`.bak_2026-08-20`) |
|---|---|---|
| `training_split` | `full_corpus_114_notes_2026-08-20` | `overnight_2026-08-17_train` |
| `code_version` | `full_corpus_retrain_2026-08-20` | `b3d9ae4` (git short hash) |
| `n_training_examples` | **1,403** | **542** |
| `training_note_ids` | **82** notes | **24** notes |
| `feature_set_version` | 1 | 1 |
| `classes_` | `[0 1]` | `[0 1]` |
| converged in | 61 iterations (of 1000) | 31 |
| intercept β₀ | **−8.3237** | −2.9893 |

Two incidental confirmations. The previous model's **542 examples across 24 notes** is exactly what §10.5's arithmetic predicts from 668 examples over 31 notes at `holdout_every=4` — the split is reproducible, as designed. And `code_version` being a bare git hash confirms it was written by `tier_gate_cal_eval.py`'s `_code_version()`, while the current one's date-stamped string confirms the retrain wrapper wrote it. The provenance chain in §10.1 checks out end to end.

### 14.1 The coefficients (current, production model)

Re-extracted directly from `models/consensus_calibrator_v1.pkl` in this session, in `FEATURE_NAMES` order:

| Feature | β (current) | β (previous, `.bak_2026-08-20`) |
|---|---:|---:|
| `frac_supported_1` | +2.0133 | |
| `frac_rerank_same_target` | −1.2113 | |
| `frac_none_correct` | +0.7991 | |
| `frac_usable_votes` | −0.0371 | |
| `mean_logprob_confidence` | +2.4479 | |
| `min_logprob_confidence` | +1.3506 | |
| `confidence_spread` | +1.9391 | |
| `match_tier_is_exact_or_synonym` | +0.4967 | |
| `top_candidate_similarity_score` | +3.5188 | |
| `is_ambiguous` | +0.3382 | |
| `domain_conflict` | −0.1225 | |
| `resolved_via_value_stripped_fallback` | −1.5957 | |
| `resolved_via_original_text_fallback` | +0.5856 | |
| `resolved_via_acronym_escalation` | **0.0000** | |
| `expansion_ambiguous` | **0.0000** | |
| `prior_confirmation_count` | +1.3456 | |

**⚠️ Document truncated here in the source message this file was built from.** The original had a fuller version of this table (apparently including the `.bak_2026-08-20` column's values feature-by-feature, a "Design intent" / "Agrees?" comparison column, and commentary), plus §14.2 onward and the entirety of §15 (open questions) and §16 (what remains uncovered). Everything above this notice was verified word-for-word and number-for-number against the live source and the real pickled artefacts before being saved. **§14's remainder, §15, and §16 are missing** — this predates §17 below and is unrelated to it; not filled in as part of this update, since re-deriving lost content that was never captured is a different task from documenting what changed on 2026-08-30.

---

## 17. 2026-08-30 update — feature 17 (`kg3_confirmation_count`) and the 144-note refit

Everything in this section is verified directly against the live source
(`src/kg3_query.py`, `src/mollm_tier_calibrator.py`, `src/mollm_tier_gate.py`,
`evaluation/tier_gate_cal_eval.py`, `scripts/retrain_calibrator_full_corpus.py`)
and against the real, currently-deployed `models/consensus_calibrator_v1.pkl`,
unpickled directly for this section — same discipline as §14, applied fresh
rather than assumed to still hold.

### 17.1 Why this exists — KG3 was a pure write sink before this

Direct code search (grep across `mollm_tier_gate.py`, `mollm_tier_calibrator.py`,
`abbreviation_flywheel.py`, `preprocessing.py`) found **zero references to
Memgraph or `kg3_query` in any decision-making module** — KG3 accumulated
writes but nothing ever read them back. §6's own caveat about
`prior_confirmation_count` said as much: *"KG3 ingestion runs dry-run
throughout, so there is no live graph to query."* This section documents
closing that gap: a new read function, a new feature built on it, and a
retrain that actually exercises it against real graph data.

### 17.2 What changed, concretely

- **`src/kg3_query.py::count_kg3_confirmations(driver, entity_text, concept_id)`**
  (new) — how many `:PatientObservation` nodes in the live graph already
  confirm this exact (text, concept) pairing, matched case-insensitively
  on `raw_text`. Same never-raise, "0 on any failure" contract as
  `count_prior_confirmations()` (§6): missing driver, missing arguments,
  or any query error all return `0`, never `None`, never propagate.
- **`FEATURE_NAMES` gains a 17th entry**, `kg3_confirmation_count`, scaled
  identically to feature 16 (`min(raw_count, 10) / 10.0`).
  **`FEATURE_SET_VERSION` bumped 1 → 2.** §11.2's guard is exactly what
  makes this safe: a model saved under the old 16-feature layout is
  refused on load (`model = None`, degraded to untrained), not silently
  scored against a coefficient vector one element short. Verified live
  before retraining — loading the then-current 114-note/16-feature `.pkl`
  under the new module reported `model is None`.
- **`build_feature_context()` and `featurize()`** both gained the new
  field/index, in the same fixed position `FEATURE_NAMES` defines —
  no change to how any of the other 16 features are computed.
- **`route_tier()` / `_score_with_calibrator()`** gained `kg3_driver` and
  `kg3_driver_factory` parameters, mirroring §2's `conn`/`conn_factory`
  contract exactly (factory takes priority when both are supplied; the
  caller closes what a factory returns). Both default to `None` —
  omitting them reproduces every prior behaviour exactly, the same
  "absent evidence, not negative evidence" default `conn`/`conn_factory`
  already use. **Unlike `conn`/`conn_factory`, there is no lock-avoidance
  reason to prefer a factory here** — a Bolt driver already supports many
  concurrent short sessions from one long-lived instance, which is why
  `scripts/run_stage3_tier_gate.py` passes the plain, already-open
  `memgraph_driver` (the same one it opens for the dry-run KG3 write-path
  check) rather than a factory.
- **`evaluation/tier_gate_cal_eval.py::build_labeled_examples()`** gained
  a `kg3_driver` parameter. **This was the part that actually mattered**:
  without it, a retrain would have trained the new feature on an
  all-zero column — a valid fit, but one that could learn nothing real
  about the feature. Caught and fixed before the retrain in §17.4 ran,
  not discovered after.

### 17.3 What did *not* change

The abbreviation flywheel (`src/abbreviation_flywheel.py`) is a separate,
upstream mechanism — Stage 1 expansion tie-breaking for ambiguous
abbreviations, not Stage 3 tier routing — and was **not touched** by this
update. It has its own, distinct training story worth stating for
contrast: `compute_frequency_priority()` aggregates the pipeline's own
Stage 2b outcomes, gated behind an explicit `VERIFIED_ALLOW_LIST` that
**starts empty** (a posture inverted from an earlier block-list design,
after a real-data test found the block-list version re-selecting its own
wrong guesses — 7/7 wrong on a gold-check of its highest-confidence
picks); `mine_context_rules()`/`select_by_context_pattern()` mines real
`hitl_review_queue` reviewer-confirmed resolutions into deterministic
trigger-word rules, and deliberately does **not** apply the same
exclusion list, since independently-confirmed human evidence is exactly
what could correct a systematic bias the frequency mechanism must avoid
formalising. Neither mechanism was retrained or otherwise modified as
part of this update — only `ConsensusCalibrator` was.

### 17.4 The retrain

Same procedure as §10, run again via
`scripts/retrain_calibrator_full_corpus.py --save`: `TIER_4_ENSEMBLE_SPLIT`
population, SNOMED-crosswalk exact-match labels, note-disjoint split
(every 4th note to validation). Two things changed simultaneously versus
the 2026-08-20 model, not one:

| | 2026-08-20 (previous) | 2026-08-30 (current) |
|---|---|---|
| Notes with `TIER_4_ENSEMBLE_SPLIT` decisions | 114 | 144 |
| Train / val notes | 82 / 27 | 105 / 34 (some notes yield zero gradable examples after label filtering — see §10.4) |
| Train / val examples | 1,403 / 450 | 1,740 / 635 |
| Features | 16 | 17 |
| `kg3_confirmation_count` source | n/a | live Memgraph, best-effort (`None` driver → 0 for every example) |
| Val AUROC | 0.845 | **0.852** |
| `BASELINE_AUROC` in the retrain script | 0.74 (stale even in 2026-08-20; corrected this update) | 0.845 |

A real bug was caught and fixed live during this retrain, unrelated to
the feature itself: `training_split`'s save string had `"2026-08-20"`
hardcoded into an f-string regardless of when the script actually ran —
`code_version` two lines below already used `datetime.date.today()`
correctly, `training_split` did not. The first `--save` run under this
fix's absence produced a `.pkl` with a false training date; caught before
being treated as authoritative, fixed, and the retrain re-run so the
saved metadata (`training_split=full_corpus_144_notes_2026-08-30`) is
correct.

**⚠️ The naive AUROC comparison (0.852 vs. 0.845) conflates two changes
at once** — more notes *and* a new feature — and should not be quoted as
"the KG3 feature is worth +0.007 AUROC." §17.5 isolates the feature's own
contribution properly.

### 17.5 Isolating the feature's own effect — an ablation on identical data

Same technique as §10.7 (`fit_and_report(..., ablate_indices=(kg3_idx,))`),
applied here for the first time to this feature specifically, and checked
in as its own reproducible script: `evaluation/kg3_feature_ablation.py`.
Two models are fit on the **exact same** 144-note data and note-disjoint
split — one with real `kg3_confirmation_count`, one with that single
feature zeroed.
Everything else — the other 16 features, the split, the labels, the hard
traps — is identical between the two fits.

| | AUROC | Coverage @ 0.72 | Precision @ 0.72 | Promoted |
|---|---|---|---|---|
| Real `kg3_confirmation_count` | 0.852 | 20.9% | **97.0%** | 133 |
| Ablated (zeroed) | 0.821 | 18.0% | **91.2%** | 114 |
| **Delta, attributable to this feature alone** | **+0.031** | **+2.9pp** | **+5.8pp** | +19 net |

This is the number to quote for "what did the feature contribute" — more
than 4× the naive 0.845→0.852 delta, because the naive comparison is
diluted by comparing against a smaller, older corpus rather than holding
the corpus fixed.

`kg3_confirmation_count > 0` on 23.9% of train examples, 23.8% of val —
not a rare-edge-case feature at the current corpus size.

**Per-example inspection, not just the aggregate table.** 29 val examples
promote only with the feature present, and **29/29 are correct against
gold** — dominated by repeated lab-value measurements (`WBC`/`RBC`/`HGB`/
`Creat` patterns independently confirmed elsewhere in the graph) and
recurring findings (`NSTEMI` ×3, `headaches` ×2, `blindness` ×2). 10
examples lose promotion when the feature is added: 6 are real false
positives correctly suppressed — `aspirin` most dramatically, score
0.918865 → 0.223810 despite `prior_confirmation_count=66` (raw count
capped to feature value 1.0), once `kg3_confirmation_count=0`
contradicted it — and 4 are genuine misses (`renal failure`, `Sclera`,
`lungs`, `left ovary` — the last essentially a coin-flip, 0.719992 vs. the
0.72 threshold in the ablated fit, 0.720312 with it).

### 17.6 The coefficients — and a sign flip worth flagging, not asserting

Extracted directly from the current, deployed `models/consensus_calibrator_v1.pkl`:

| Feature | β (17-feature, current) | β (16-feature, `.bak_2026-08-20`, §14.1) |
|---|---:|---:|
| `kg3_confirmation_count` | **+7.1368** | *(did not exist)* |
| `top_candidate_similarity_score` | +2.7502 | +3.5188 |
| `frac_supported_1` | +2.3729 | +2.0133 |
| `mean_logprob_confidence` | +2.1775 | +2.4479 |
| `confidence_spread` | +1.2949 | +1.9391 |
| `min_logprob_confidence` | +1.2104 | +1.3506 |
| `is_ambiguous` | +0.6583 | +0.3382 |
| `match_tier_is_exact_or_synonym` | +0.6149 | +0.4967 |
| `resolved_via_value_stripped_fallback` | −0.6255 | −1.5957 |
| `resolved_via_original_text_fallback` | +0.4137 | +0.5856 |
| `frac_none_correct` | +0.3770 | +0.7991 |
| `frac_rerank_same_target` | −1.5026 | −1.2113 |
| `domain_conflict` | −0.8176 | −0.1225 |
| `frac_usable_votes` | +0.0155 | −0.0371 |
| `resolved_via_acronym_escalation` | 0.0000 | 0.0000 |
| `expansion_ambiguous` | 0.0000 | 0.0000 |
| `prior_confirmation_count` | **−2.1114** | **+1.3456** |
| intercept β₀ | −7.4247 | −8.3237 |
| `n_iter_` (converged) | 53 (of 1000) | 61 |

Two things worth stating plainly, in the "read from the model, not
asserted" discipline §3's warning sets for this document:

1. **`kg3_confirmation_count`'s coefficient (+7.14) is by far the largest
   magnitude of any feature in either fit** — more than double
   `top_candidate_similarity_score`, the previous largest. Read
   literally, on this bounded [0,1] scaled feature, going from 0 to a
   saturated 1.0 (10+ confirmations) shifts the logit by +7.14, which
   alone can move `P(correct)` from near-chance to near-certain,
   independent of everything else the model sees.
2. **`prior_confirmation_count`'s coefficient flipped sign**: +1.3456 in
   the 16-feature fit, **−2.1114** in the 17-feature fit. This sign IS
   read directly off the fitted model — that part is fact, not inference.
   **What it means is inference, and is flagged as such rather than
   asserted**: the two confirmation-count features are plausibly
   collinear (an entity confirmed in DuckDB's own decision history is
   often also the kind of entity KG3 was populated with, since KG3's
   current population was itself derived by grading those same
   historical decisions — see §17.7), and a linear model facing two
   correlated features can assign the credit to one and a compensating
   negative weight to the other without either feature individually
   *predicting* incorrectness. Worth a variance-inflation check or a
   single-confirmation-feature refit before treating the sign flip as
   more than a fitting artifact — not done as part of this update.

### 17.7 The caveat that must accompany every number above

KG3's current population is **100% gold-simulated** — every node was
written by a one-off script that graded historical pipeline decisions
against gold and ingested the matching ones via `ingest_reviewed_case()`
with real (non-dry-run) writes, not by real completed human review — zero
real HITL reviews exist in production as of this writing (see
`docs/FINAL_RESULTS_Single_Source_Of_Truth.md` §10). `kg3_confirmation_count`
being greater than zero is therefore, to a real and
currently unquantified degree, **a restatement of "this exact pattern
already matched gold once"** — evaluated in §17.5 against labels that are
**also** gold-based, on a note population substantially overlapping the
notes KG3 was populated from.

The note-disjoint train/val split protects against literal row-level
leakage (`ConsensusCalibrator.load(..., scoring_note_ids=...)`'s guard,
§11.3, applies identically here) — it does **not** protect against this
deeper, population-level circularity, which is a different and unsolved
problem. §6's own caveat about `prior_confirmation_count` already named
this risk for the DuckDB-sourced feature; it applies at least as strongly
to `kg3_confirmation_count`, for the same reason and for a graph
literally populated by the grading process the labels themselves come
from. **§17.5's numbers are real, given KG3's contents as of 2026-08-30,
and should not be presented as a measurement of what this feature does
against independent, real human-reviewed KG3 data — that measurement
does not exist yet, because that data does not exist yet.**

### 17.8 The adoption decision

Adopted as production (`code_version=full_corpus_retrain_2026-08-30`,
`training_split=full_corpus_144_notes_2026-08-30`), on two grounds: (1)
the isolated ablation (§17.5) shows real, individually-inspectable,
well-behaved signal even holding the §17.7 caveat in mind, and (2) it is
also the only way to have a genuinely *trained* production calibrator at
all after the `FEATURE_SET_VERSION` bump — the alternative was leaving
the old 16-feature model deployed, where it now loads as `model = None`
(§11.2's guard) and `route_tier()` treats every calibrator consultation
exactly as "no calibrator available." This is **not** presented as a
validated claim about real-world unreviewed KG3 evidence — see §17.7 —
only as the honest best available option given what currently exists.

**`CALIBRATED_AUTO_THRESHOLD = 0.72` was not re-derived for this retrain
either**, compounding the already-open item in §13.5: the threshold has
now survived two refits on two different corpora with two different
feature sets without being re-validated against either one. This is
flagged, not fixed, in this update.

---

## 18. 2026-08-31 — locked-test-split contamination found and fixed

**What was found.** `scripts/retrain_calibrator_full_corpus.py` (every
retrain described in §17 above, including the currently-deployed model)
sourced its training-note pool from `SELECT DISTINCT note_id FROM
mollm_tier_gate_decisions WHERE tier = 'TIER_4_ENSEMBLE_SPLIT'` — every
note the pipeline had processed, unconditionally. That is a different
thing from `data/splits/note_splits.csv`'s **official locked test split**
(70 notes reserved for the T0/T1/T2 benchmark comparison against
Clinical-T5, per `docs/Evaluation_Criteria.md`) — the two "test" concepts
share a name (`is_test`) but not a meaning. Checked live: **31 of the 105
notes actually used to fit the previously-deployed model (30%) were from
the locked split.** The calibrator's own supervised labels (gold-graded
correct/incorrect) were built in part from annotations the proposal
reserves for final evaluation only.

**Why past point-in-time claims in §17 above are not retroactively
false, but going-forward reuse of "fresh-10"/"fresh-5" needed a second,
related fix.** The 2026-08-20/2026-08-30 retrains happened before this
fix; whatever generalization those models showed on fresh-10/fresh-5 at
*measurement time* is a fact about what was measured then. But
cross-referencing the deployed model's own `training_note_ids` against
`ui/components/fresh10_notes.py`'s `FRESH10_NOTE_IDS` found **8 of
fresh-10's 10 notes were, by the time of this check, part of the
calibrator's actual training set** (fresh-10 is entirely drawn from the
official locked test split, so excluding that split — the fix below —
removes fresh-10 too). Fresh-5's real 2026-08-30 batch (`13397956-DS-5`
etc., §10 of the SSOT — NOT `evaluation/grade_fresh5_by_tier.py`'s
differently-named older batch, a related mix-up documented in
`docs/Code_Reference_Stages_And_Metrics.md` §16) is **not** part of the
official locked split — it's a project-specific "genuinely unprocessed as
of this date" holdout — so checking the first fix alone found it clean.
**But re-verifying after applying the first fix found a second, live
contamination**: the retrain naturally re-pulled 4 of fresh-5's 5 notes
back into the training pool, because their own 2026-08-30 validation run
had, by now, given them real `TIER_4_ENSEMBLE_SPLIT` decisions — nothing
in the first fix excluded them, since they were never part of the
official test split to begin with. Same failure shape as the first leak,
different mechanism (a project-specific holdout the official split
doesn't know about, not a naming collision).

**Fix, two parts**: `scripts/retrain_calibrator_full_corpus.py` now
excludes (1) `evaluation.splits.load_split("test")` — the official locked
split, same discipline `evaluation/splits.py` already established for
every properly-behaved evaluation script in this codebase, just not
previously applied here; and (2) `FRESH10_NOTE_IDS` and fresh-5's 5 real
note IDs explicitly, hardcoded with a comment explaining why the official
split alone doesn't cover fresh-5.

**Real, re-measured result on the doubly-clean 105-note pool** (75 train
/ 25 val, vs. the previous 105/144 note counts above — a smaller corpus,
since 44 notes contributing contaminated or holdout-destroying volume are
now correctly excluded):

| | Previous (contaminated, §17.4) | **Re-measured, clean (2026-08-31)** |
|---|---|---|
| Notes in training pool | 144 | 105 (39 locked-test-split + 5 fresh-5 excluded) |
| Train / val notes | 105 / 34 | 75 / 25 |
| Train / val examples | 1,740 / 635 | 1,293 / 485 |
| Val AUROC | 0.852 | **0.868** |
| Precision @ 0.72 (bracket; exact value not re-derived — see §13.5's still-open item) | 97.0% | **~97.8%–98.7%** (bracketed between the printed 0.70/0.75 sweep rows, unchanged bracket from the single-fix intermediate result) |

**The honest headline: removing 39% of the training pool did not hurt —
AUROC moved from 0.852 to 0.868.** This is reassuring but should not be
oversold as "the contaminated data was actively harmful" — the two val
sets differ in composition (different notes), so this is not a clean
ablation the way §17.5 is; it is a real measurement on a corrected
population, not a controlled comparison isolating the contamination's
own effect. What it does establish: the previously-reported 0.852 was not
an artifact that collapses once the leaks are closed, and the corrected
model is at least as good by the same metric.

**Verified directly, not assumed**: the final saved model's
`training_note_ids` (75 notes) has **zero overlap** with both
`FRESH10_NOTE_IDS` and fresh-5's real 5 notes — both are now genuinely
usable as held-out validation populations for this calibrator, which was
not true of fresh-10 under either the untouched-original or the
single-fix-intermediate model.

**Adopted**: saved as `code_version=full_corpus_retrain_2026-08-31`,
`training_split=full_corpus_105_notes_2026-08-31`. This is now the
production `models/consensus_calibrator_v1.pkl`, superseding §17.8's
2026-08-30 model.

**Still open at the time §18 was first written**:
`CALIBRATED_AUTO_THRESHOLD = 0.72` had survived three refits (2026-
08-20, 2026-08-30, 2026-08-31) across three different corpora and two
feature-set versions without ever being re-derived from scratch. Also
newly open: `evaluation/tier_gate_cal_eval.py`'s own default `NOTE_IDS`
(the hardcoded 31-note "overnight" set, used when no `note_ids` is passed
explicitly) was not checked against the locked split as part of this fix
— the production retrain path (`retrain_calibrator_full_corpus.py`,
fixed above) never relies on that default, but any other caller that
does should not be assumed clean without the same check.

---

## 19. 2026-08-31, same day — `CALIBRATED_AUTO_THRESHOLD` re-derived: 0.72 → 0.78

Closes the item §18 left open. Since the leakage fix forced a calibrator
refit anyway, the threshold was re-derived on the clean model using the
**exact same selection rule the original 0.72 was chosen with** (§13.1):
the smallest threshold reaching 100% precision on the val set, hard traps
(`_is_coronary_segment_trap()`, `_is_short_alphanumeric_code()`) active.

**Fine-grained sweep on the clean 105-note pool** (75 train / 25 val,
same model as §18):

| Threshold | Coverage | Precision | False positive(s) remaining |
|---|---|---|---|
| 0.72 | ~16-17% (between the 0.70/0.75 sweep rows) | **~98%** (between 97.8% and 98.7%) | `neck pain` (score 0.779742) and others below it |
| 0.75 | 15.9% (77/485) | 98.7% | `neck pain` still present |
| 0.76 | — | <100% | `neck pain` (0.779742) — confirmed the ONLY one remaining |
| 0.77 | — | <100% | `neck pain` (0.779742) — still present |
| **0.78** | **13.6% (66/485)** | **100%** | **none** — confirmed via direct fine sweep |

**The story is structurally identical to the original 0.70→0.72 bump**,
just a different specific entity: `incontinence` (0.70698) forced the
first bump; `neck pain` (0.779742, votes `SUPPORTED_1:2/NONE_CORRECT:1`)
is the equivalent blocker here — a real clinical term the coronary/
short-code hard traps have no reason to catch (neither a coronary-segment
abbreviation nor a short alphanumeric code), so only a threshold move
closes it.

**Why 0.72 no longer holds, on real data, not just staleness**: the
original 0.72 was measured at **100%** precision on its own 2026-08-17
val set. On today's clean, current-corpus val set, 0.72 measures only
**~98%** — a real drift in what the threshold actually buys, independent
of the leakage question. This is consistent with (and now supersedes) the
2026-08-17 51-note diagnostic already logged in `docs/Implementation_
Decisions_Log.md` ("the production 0.72 threshold itself measured only
89.5% on this larger set vs. ~100% on the original smaller val set") —
that finding was directionally correct 14 days before this fix, on data
that (per §18) was itself unverified against the locked split.

**Adopted**: `src/mollm_tier_gate.py`'s `CALIBRATED_AUTO_THRESHOLD`
updated `0.72` → `0.78`, with an inline comment recording this derivation
and the exact false-positive story, matching the discipline already
present for the original value. Real, disclosed cost: coverage at this
threshold (13.6%) is lower than 0.72's own coverage was on this same
clean val set (16-17%) — a further tightening on top of the original
0.65→0.72 tightening's own cost, not a free correction.

**Still open**: `evaluation/tier_gate_cal_eval.py`'s own default
`NOTE_IDS` (unaffected by today's fixes, per §18's closing note) remains
a real gap for any future caller that relies on it without an explicit
`note_ids` override.
