# Phase 2 (Pass 4 Tier 1-5 gate) — first real validation, 2026-08-15

**Status: built and running against live models, but the 90% autonomous
target is not met on this sample, and the honest reason is a scope mismatch
worth fixing before concluding anything about the gate's own quality.**

## What ran

`scripts/run_tier_gate_batch.py` against the 3 canonical gold notes
(17751158-DS-19, 19442119-DS-15, 14490470-DS-11), 6 LOW-tier entities per
note, 18 total, real Ollama calls (qwen2.5:3b/llama3.2:3b/phi4-mini), no
mocking. Full per-decision output: `reports/tier_gate_batch_results.json`.

## Result

```
TIER_1_AUTO_VALIDATED: 1  ( 5.6%)   -- 1/1 gradable, 100.0% precision
TIER_4_ENSEMBLE_SPLIT: 12 (66.7%)
TIER_5_TRUE_AMBIGUITY: 5  (27.8%)
AUTO coverage (Tier 1+2+3): 5.6%   (target ~90%)
```

Zero Tier 2/3 decisions. HITL breakdown: `ensemble_split` 12,
`unresolved_acronym` 2, `verdict_none_correct` 3.

## Why this is not (yet) evidence the gate itself is broken

**This ran ONLY against Stage 2b's LOW-confidence tier**, because that is the
only entity population `src.mollm_ensemble.load_validation_records(tier="LOW")`
loads, and it is the only population production Stage 3 has ever seen
(`build_prompt()`'s own mode selection: `if tier == "LOW" and
len(candidates) > 1`). LOW tier is, by construction, the subset Stage 2b's
own confidence scoring already flagged as hard -- it is not a random sample
of all entities, it is pre-filtered toward disagreement. The spec's
~70/15/5/5/5 Tier 1-5 distribution reads as a target over the FULL entity
population Pass 4 would see in the blueprint's design (every entity, with
HIGH/MEDIUM-tier confirmations making up the bulk of Tier 1's ~70%) --
testing only the hardest pre-filtered slice and expecting 90% of THAT to
auto-resolve is very likely the wrong bar, not evidence the two-step CoT
itself underperforms. This distinction was not obvious until real numbers
came back, which is exactly why this is flagged now rather than after
building Phases 3-6 on an unvalidated assumption.

A secondary, real (not scope-driven) factor: strict 3/3 unanimity is a high
bar with 3B models, consistent with `scripts/experiment_3b_voting.py`'s own
prior measurement (2-1 splits "MEDIUM" 50.0% precision vs 3-0 "HIGH" 72.7% on
a 75-entity mini-test) -- a meaningful fraction of genuine 2-1 splits is
expected behavior on hard entities, not a bug in this implementation.

## What the one gradable Tier 1 decision showed

`'Laparoscopic appendectomy'` (1 candidate) -> 3/3 unanimous SUPPORTED_1 ->
AUTO_VALIDATED -> graded correct against gold. Single data point, but it is
at least a real, non-degenerate walk through the full two-step CoT ending in
a correct autonomous decision -- the mechanism works when the setup allows a
clean answer.

## Recommended next step (not yet done)

Re-run `scripts/run_tier_gate_batch.py` (or a variant) against the FULL
entity population for a note -- not just `tier="LOW"` -- so the tier
distribution is measured against the same population the spec's percentages
describe. `load_validation_records()` already supports this (drop the
`tier="LOW"` filter); the batch script would need Tier 1's fast-confirmation
path exercised on HIGH/MEDIUM-tier entities too, which the current script
does not attempt (it only ever loads LOW-tier records). This is the natural
next increment before drawing a conclusion about whether the 90% target is
reachable.

## What is NOT yet done in Phase 2

- No re-run at full-population scope (above).
- `src.kg3_ingestion.ingest_auto_decision()` has been verified end-to-end in
  dry-run and one live-write-then-cleanup smoke test (see commit `240796c`),
  but this batch script never calls it -- still read-only/measurement-only
  by design.
- No calibration of `TIER1_CONFIDENCE_FLOOR` (0.70, placeholder) or the Tier
  3 fast-path criteria against this real data yet -- n=18 is too small to
  calibrate anything.
- `src.mollm_ensemble.py`'s production `route()` is untouched; this is still
  a standalone, additive module per the plan.
