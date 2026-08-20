# 2026-08-19: TIER_2_AUTO_RESOLVED precision collapse -- the "MCV/MCHC/RDW problem"

## Summary

A corpus-scale grading pass (`evaluation/tier_gate_grading.py`) across 68 notes
and 8,815 Stage 3 decisions surfaced a real, systematic bug: `TIER_2_AUTO_RESOLVED`
(the "3/3 unanimous re-rank" auto-write tier) measured at **20.0% precision**
(115 clean-graded, 92 wrong) -- far below what an unreviewed auto-write tier
should be. Every wrong decision was the same repeatable pattern for common CBC
abbreviations (`RDW-13`, `MCV-98`, `MCHC-32`, etc., identical wrong concept
firing on every occurrence). Root-caused, one fix attempt tried and reverted
after it made things worse, then fixed properly. Retroactive reprocessing of
already-completed notes scheduled to run automatically once the in-flight
batch's Stage 3 finishes.

## Full grading results (context)

| Tier | n_decisions | clean_n | precision |
|---|---|---|---|
| TIER_1_AUTO_VALIDATED | 3234 | 1792 | 82.0% |
| TIER_3_AUTO_VALIDATED | 1130 | 868 | 97.9% |
| TIER_1B_CALIBRATED_AUTO_VALIDATED | 60 | 33 | 93.9% |
| TIER_2_AUTO_RESOLVED | 189 | 115 | **20.0%** |
| TIER_4_ENSEMBLE_SPLIT (shadow, still HITL) | 3001 | 1397 | 61.1% |

Calibrator-active (14 notes) vs calibrator-inactive/leakage-disabled (54
notes) subsets showed comparable precision on every tier both had in common
(TIER_1: 77.4% vs 82.6%, TIER_3: 97.8% vs 97.9%) -- no sign the calibrator
behaves differently in the wild than its held-out validation numbers.

## Root cause

`TIER_2_AUTO_RESOLVED` means the 3-model ensemble unanimously voted
`RE_RANK_TO_CANDIDATE_N` -- rejected Stage 2b's top-ranked candidate and
picked a different one instead, unanimously.

`src/normalization/tier_retrieval.py`'s `_prefer_lab_procedure_over_observable()`
(built earlier this session, 78/78-exceptionless corpus evidence) already
re-ranks a Lab-Test entity's Procedure-class candidate ("...determination")
ahead of its Observable-Entity-class sibling (the abstract property, e.g.
"Red blood cell distribution width") when SapBERT's raw cosine score would
otherwise put the wrong one first. Direct verification (`normalize_entity('RDW',
...)`) confirmed this re-ranking correctly places the right concept at
candidate #1 for every case checked.

The gap: all 3 ensemble models still unanimously **rejected candidate #1 and
re-ranked to candidate #2** anyway. Pulled the actual stored decision for
`RDW-13` directly (`mollm_tier_gate_decisions.models`):

```
phi4-mini:   RE_RANK_TO_CANDIDATE_2 | "more specific to red cells and their size variation"
qwen2.5:3b:  RE_RANK_TO_CANDIDATE_2 | "supports... being an observation rather than a condition/disorder"
llama3.2:3b: RE_RANK_TO_CANDIDATE_2 | "specifically targets red blood cells and has a clear domain classification"
```

Candidates were `[1] Red cell distribution width determination` (correct,
gold's target) and `[2] Red blood cell distribution width` (wrong). Every
model's own reasoning cites candidate #2's more literal-sounding wording as
the deciding factor -- the exact anchoring-on-surface-form bias
`_prefer_lab_procedure_over_observable()`'s rank bonus was built to
counteract, except the rank bonus alone wasn't strong enough to stop the
ensemble from independently re-ranking away from position 1.

No direct SNOMED relationship exists between the two concepts in any of the
three verified pairs (checked `athena_concept_relationship` directly) -- this
is a naming-convention duplicate, not a graph-linked one, same shape as the
brand-alias ("Lasix") problem from earlier this session.

## Fix attempt 1 (prompt-based) -- tried, failed, reverted

First approach: add a `LAB_PROCEDURE_PRIOR` instruction clause to
`_binary_match_prompt()`, modeled directly on the already-proven
`CONDITION_VS_OBSERVATION_PRIOR` pattern (a similar duplicate-concept prior
built earlier this session that measurably works).

**v1** (soft, reasoned wording): live-tested against the real `RDW-13` case.
No change -- qwen2.5:3b still re-ranked to candidate #2.

**v2** ("sledgehammer" strengthening, matching `CONDITION_VS_OBSERVATION_
PRIOR`'s own successful escalation pattern -- imperative default + high-bar
override condition): live-tested again. Still no change across any of the 3
models, and in one case actively **regressed**: llama3.2:3b, which had
previously (wrongly) picked candidate #2, now picked **candidate #4** ("Dry
body weight" -- a nonsensical, unrelated concept), worse than before.

**Why it failed**: traced to an architectural mismatch, not a wording
problem. `CONDITION_VS_OBSERVATION_PRIOR` lives in `_tiebreak_prompt()`, a
dedicated comparative call that shows 2+ candidates side by side -- the model
can literally see both options and weigh them against each other.
`LAB_PROCEDURE_PRIOR` was injected into `_binary_match_prompt()`, which
judges **exactly one candidate per call**, by deliberate design (documented
elsewhere in this module: a dense 1-to-N candidate list was already measured
to cause index-confusion in 3B models). Telling the model to prefer "this
candidate... even when a later candidate's wording looks closer" asks it to
reason about text that is not present anywhere in its own context window.
No amount of wording strength fixes an instruction that references
information the model cannot see.

Reverted both v1 and v2 entirely rather than ship an unproven, sometimes-
harmful prompt change.

## Fix attempt 2 (deterministic fast-path) -- verified working

Since persuasion isn't architecturally viable at this prompt site, applied
the same trust-tier pattern this codebase already uses successfully for
`verified_brand_alias`, `verified_lab_test_alias`, and
`PHYSEXAM_SHORTHAND_MATCH_BASIS`: bypass the ensemble entirely for a
curated/verified pattern instead of asking a small model to honor it.

1. `_prefer_lab_procedure_over_observable()` now tags its winning candidate's
   `match_basis = "lab_procedure_preferred"` when it swaps a Procedure-class
   concept ahead of an Observable-Entity sibling (only when the winner's own
   basis was still the generic Tier 3 default -- never overwrites a more
   specific existing basis like a verified alias).
2. New `_lab_procedure_fast_path()` in `src/mollm_tier_gate.py`: when a
   Lab-Test entity's top candidate carries that tag, with no ambiguous
   expansion, a PRESENT-or-unset assertion, and a similarity score clearing
   `TIER3_SIMILARITY_FLOOR`, routes directly to `TIER_3_AUTO_VALIDATED` --
   **zero model calls**.
3. Wired into `route_tier()` alongside the existing `tier3_fast_path()`/
   `tier5_precheck()` free pre-checks.

**Verified live** (real DB, real `normalize_entity()` + `route_tier()` calls,
no mocks):

```
RDW  -> 4281085 Red cell distribution width determination   | lab_procedure_preferred -> TIER_3_AUTO_VALIDATED, 0 model calls
MCV  -> 4016239 Erythrocyte mean corpuscular volume determination | lab_procedure_preferred -> TIER_3_AUTO_VALIDATED, 0 model calls
MCHC -> 4290193 Mean corpuscular hemoglobin concentration determination | lab_procedure_preferred -> TIER_3_AUTO_VALIDATED, 0 model calls
```

All three now resolve to the gold-correct concept. Full test suite 77/77
passing (`tests/test_tier_gate.py` updated to extract `_lab_procedure_fast_path`
into its AST-based pure-function harness, same mechanism already used for
`tier3_fast_path`/`tier5_precheck`).

## Separately: a narrower Tier 1/2 gap, also fixed

While diagnosing this, found and fixed a related but distinct gap:
`_prefer_lab_procedure_over_observable()` can only re-rank when BOTH classes
are already present among the candidates it's given -- true for Tier 3's
top-K semantic search, never true for Tier 1/2's literal exact/synonym
match (the Procedure-class sibling has entirely different text, so it's
never in the pool Tier 1/2 returns). Added `_lab_procedure_sibling_check()`/
`_lab_procedure_sibling()` in `tier_retrieval.py`: when a Lab-Test entity's
Tier 1/2 hit is *entirely* Observable-Entity-class, runs a supplementary
semantic search scoped to Procedure-class SNOMED Measurement concepts and
prepends it (ambiguous=True) if found above the similarity floor. This
scenario wasn't directly confirmed in the graded data (the observed failures
all went through Tier 3), but is a real, evidence-consistent defensive
complement, tested and harmless.

## Retroactive reprocessing

Both fixes only take effect when `normalize_entity()`/`process_and_
normalize_entities()` actually runs -- every note already fully processed
through Stage 2b before this fix landed has stale `normalized_entities.
candidates` with no `lab_procedure_preferred` tag at all, and Stage 3's
resume-check would otherwise skip past their already-decided entities
without ever re-evaluating them.

`scripts/retroactive_lab_procedure_fix.py` (new): re-runs
`process_and_normalize_entities()` for every Lab-Test-labeled entity in an
already-processed note, identifies which ones actually changed (now tagged
`lab_procedure_preferred`), deletes only THOSE entities' stale
`mollm_tier_gate_decisions` rows (so Stage 3 resume-check recomputes exactly
them, nothing else), and reports the distinct note_ids touched.

Scheduled via `logs/chain_retroactive_lab_fix.sh` (detached background
watcher, same pattern as this session's other batch-chaining scripts) to run
automatically once the in-flight 50-note batch's Stage 3 completes --
deliberately sequential, not parallel, after two earlier attempts at
parallel Stage 1-2b/Stage 3 execution this session caused real DB-lock
contention problems (one causing actual data loss, one causing full
starvation) that took real time to diagnose and recover from. Progress in
`logs/retroactive_lab_fix.log`.

## Files changed

- `src/normalization/tier_retrieval.py`: `_prefer_lab_procedure_over_observable()`
  now tags its winner; new `_lab_procedure_sibling_check()`/
  `_lab_procedure_sibling()`.
- `src/normalization/orchestrator.py`: Tier 1 and Tier 2 branches of
  `normalize_entity()` call the new sibling check/lookup before returning.
- `src/mollm_tier_gate.py`: new `_lab_procedure_fast_path()`, wired into
  `route_tier()`.
- `tests/test_tier_gate.py`: `_lab_procedure_fast_path` added to the
  AST-extracted pure-function set.
- `scripts/retroactive_lab_procedure_fix.py` (new).
- `logs/chain_retroactive_lab_fix.sh` (new, not committed -- operational
  script in the gitignored `logs/` working directory).
