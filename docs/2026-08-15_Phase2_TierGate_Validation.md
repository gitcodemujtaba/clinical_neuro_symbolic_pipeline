# Phase 2 (Pass 4 Tier 1-5 gate) — real validation, 2026-08-15

**Status: the scope-mismatch hypothesis from the first run is DISPROVEN by a
second run. The real, actionable finding is a specific prompt-strictness
tradeoff, and a first attempt to fix it produced a textbook "consensus went
up, precision did not" failure -- reverted, not shipped. This is the most
important thing to read before building Phases 3-6 on top of this gate.**

## Run 1 — LOW-tier only (18 entities, 3 canonical gold notes)

```
TIER_1_AUTO_VALIDATED: 1  ( 5.6%) -- 1/1 gradable, 100.0% precision
TIER_4_ENSEMBLE_SPLIT: 12 (66.7%)
TIER_5_TRUE_AMBIGUITY: 5  (27.8%)
AUTO coverage: 5.6%   (target ~90%)
```

Initial hypothesis: this only tested Stage 2b's LOW-confidence tier (the
hardest, pre-filtered-toward-disagreement subset, and the only population
production Stage 3 has ever seen) -- maybe the full entity population, where
most entities are easy HIGH-tier confirmations, would look completely
different.

## Run 2 — HIGH-tier only (36 entities, same 3 notes) — hypothesis disproven

```
TIER_1_AUTO_VALIDATED: 1  ( 2.8%) -- 1/1 gradable, 0.0% precision
TIER_4_ENSEMBLE_SPLIT: 28 (77.8%)
TIER_5_TRUE_AMBIGUITY: 7  (19.4%)
AUTO coverage: 2.8%   (target ~90%)
```

Even Stage 2b's OWN most-confident tier -- entities Stage 2b already believed
it had resolved correctly -- split 77.8% of the time. The scope-mismatch
explanation does not hold; something in the two-step CoT itself is causing
unanimity to fail far more often than it should.

## Root cause, found by reading actual model reasoning (not guessed)

Pulled `eval_trail`/`clinical_meaning` for several split cases directly from
`reports/tier_gate_batch_results.json`:

- **`HEENT`**: unanimous, CORRECT rejection. Stage 2b's only candidate was
  "Structure of anterior portion of neck" -- genuinely wrong for HEENT
  (head/eyes/ears/nose/throat). All 3 models correctly said NONE_CORRECT.
  The gate working as intended; just correctly declining to auto-validate a
  bad upstream candidate, which is not free coverage but is not a bug either.
- **`pneumonia`, `heart failure`, `sodium`**: all three are cases where 2
  models said `SUPPORTED_1` (obviously correct) and **`qwen2.5:3b`
  consistently dissented**, demanding the candidate's bare name also capture
  contextual specifics the prompt never asked it to require -- rejecting
  plain "Pneumonia" for "lack[ing] specificity regarding fever and lung
  inflammation on the left side," rejecting plain "Heart failure" for not
  saying "chronic systolic." A real, diagnosable prompt-calibration gap, not
  random noise.

## Attempted fix #1 (REJECTED, not shipped) — loosening the match rule

Added a 5th rule to `_binary_match_prompt()`: "match the core concept, not
every detail -- a plain 'Pneumonia' candidate matches even if the note adds
laterality/severity." Re-ran the same 36-entity HIGH-tier batch:

```
TIER_1_AUTO_VALIDATED: 20 (55.6%) -- 17 gradable, 1 correct, 5.9% precision
TIER_4_ENSEMBLE_SPLIT: 16 (44.4%)
AUTO coverage: 55.6%   (target ~90%)
```

Coverage jumped from 2.8% to 55.6% -- and Tier 1 precision **collapsed to
5.9%** (1/17 correct). The looser rule let all three models unanimously
rubber-stamp WRONG matches on bare qualifier/fragment spans that are not
independently linkable clinical concepts at all: `'left'`, `'Removal'`,
`'Multiple'`, `'fixation'` were all AUTO_VALIDATED to some SNOMED code purely
because the prompt no longer let a model object that the candidate said
nothing distinctive. This is the exact "ensemble consensus went up, precision
did not" failure `src/mollm_ensemble.py`'s own Fragile Concept Gate
(`route()`, the `value_stripped_from_` cap) was built to catch on lab
values -- reproduced here by a different mechanism, on a different entity
class. **Reverted.** The stricter 4-rule prompt is back in force in
`src/mollm_tier_gate.py`; the rejected rule and this reasoning are recorded
in that file's own inline comment so it is not silently re-tried later
without this context.

## What this means for the plan

The two-step CoT mechanism itself is not obviously broken -- it correctly
rejected a genuinely-wrong candidate (HEENT) and correctly accepted a
genuinely-right one at least once each. The problem is narrower and more
tractable than "does the architecture work": **qwen2.5:3b's strictness
calibration relative to the other two models is the dominant driver of
splits on cases the other two get right**, and the naive fix (loosen the
rule for everyone) trades that off directly against precision on a different,
worse failure mode (fragment/qualifier spans getting confidently
mislabeled). A future attempt should be narrower and testable in isolation,
e.g.:

- Scope any specificity relaxation to GLiNER labels where it is safe
  (Condition/Medication/Lab Test) and never to single-word/qualifier-shaped
  spans, rather than loosening the rule globally.
- Investigate whether qwen2.5:3b specifically needs a different prompt, a
  different role in the ensemble (e.g. tie-breaker rather than an equal
  vote), or is simply a weaker fit for this task style than llama3.2:3b/
  phi4-mini -- this codebase already made a similar per-model judgment once
  (retiring BioMistral+OpenBioLLM for the current 3-model Ollama set).
- Consider whether "3/3 unanimous" is too strict a bar for THIS ensemble
  regardless of prompt wording, given `scripts/experiment_3b_voting.py`'s
  own prior finding that 2-1 splits still carry real signal (50.0% precision,
  not noise) -- a differently-shaped Tier 2 (e.g. "2/3 agree AND the
  dissent is qwen2.5:3b specifically, on a Condition/Medication label")
  might recover more coverage without the fragment-rubber-stamping failure
  mode, but this is a hypothesis, not yet tested.

## What is NOT yet done in Phase 2

- No fix has been found and validated yet -- the state as of this doc is
  "problem diagnosed, one candidate fix tried and rejected."
- `src.kg3_ingestion.ingest_auto_decision()` has been verified end-to-end in
  dry-run and one live-write-then-cleanup smoke test (commit `240796c`), but
  the batch runner never calls it -- read-only/measurement-only throughout,
  by design, and doubly appropriate given the 5.9% precision result above.
- No calibration of `TIER1_CONFIDENCE_FLOOR` (0.70, placeholder) or the Tier
  3 fast-path criteria against real data yet -- today's samples are far too
  small and, per the above, the gate isn't trustworthy enough yet to be
  worth calibrating a confidence floor for.
- `src.mollm_ensemble.py`'s production `route()` is untouched; this is still
  a standalone, additive module per the plan.

## Recommendation

Do not proceed to Phases 3-6 (retrieval/acronym/chunking/cache) under the
assumption that Phase 2's gate is production-ready -- it is not yet. The
higher-leverage next step is narrowly diagnosing and fixing the qwen2.5:3b
strictness gap (or the ensemble-composition question it points to) on a
larger, more diverse sample, since that is what is actually gating the 90%
target -- not upstream retrieval or acronym quality, which Phases 3/4 would
improve.
