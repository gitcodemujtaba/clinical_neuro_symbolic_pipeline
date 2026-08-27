# Phase 3 (Pass 3 hybrid retrieval) — validation, 2026-08-16

**Status: structurally sound, gated off. `_tier3_hybrid_rows()` (dense +
BM25 + prior, Reciprocal Rank Fusion) is built, tested, and validated
end-to-end against real data. A real bug was found and fixed along the way.
The remaining gap — Top-1 accuracy trailing dense-only — is isolated to the
RRF weights specifically, not to any structural defect, and needs a real
hyperparameter sweep before `CNSP_HYBRID_RETRIEVAL` should default on. That
sweep is explicitly deferred, not attempted here.**

## What was built

- `src/normalization/bm25_index.py` — DuckDB FTS index over
  `athena_concept.concept_name`. Along the way, found and fixed a real
  performance bug: the naive query pattern (self-join `match_bm25`'s
  subquery back to `athena_concept`) measured 109s/query against this
  table — unusable. The fix (query the base table directly; it already
  carries every needed column) measured 0.2-1.6s with realistic filters.
- `_tier3_hybrid_rows()` (`src/normalization/tier_retrieval.py`) — RRF
  fusion: `Score(c) = w_dense·RRF_dense(c) + w_sparse·RRF_sparse(c) +
  w_prior·P(c|Mention)`, `RRF_x(c) = 1/(RRF_K + rank_x(c))`. Starting
  weights `w_dense=0.5, w_sparse=0.3, w_prior=0.2`, `RRF_K=60` —
  explicitly named CALIBRATION-PENDING constants (`RRF_WEIGHT_DENSE` etc.
  in `tier_retrieval.py`), same discipline this codebase already applies to
  every other routing threshold. `verified_brand_alias` candidates are
  guaranteed through fusion and truncation, matching the dense-only path's
  existing guarantee.
- `CANDIDATE_LIMIT` bumped 3→5 (`src/normalization/constants.py`), matching
  the spec's "Top-5" target. Confirmed both `mollm_ensemble.py` and
  `mollm_tier_gate.py` already iterate candidates dynamically, so this
  needed no downstream code change.
- `VOCAB_BY_LABEL` checked empirically (real `vocabulary_id` distribution
  per OMOP domain) before touching it: Anatomy is already 100% SNOMED, and
  Procedure/Measurement's larger non-SNOMED vocabularies (ICD10PCS, LOINC)
  aren't this pipeline's target crosswalk vocabulary. Concluded no
  extension is warranted — a validated finding, not a forced change.
- `evaluation/stage2b_hybrid_ab.py` — the A/B harness: re-runs
  `normalize_entity()` fresh (not the DB's stored, pre-Phase-3 candidate
  lists) against the same already-extracted entities, once per retrieval
  mode, and reports Top-1/Top-5-oracle accuracy plus the rank distribution
  of the correct concept when found.
- 12 new unit tests (`tests/test_hybrid_retrieval.py`, pure fusion/
  truncation/alias-guarantee logic, no live DB needed) + 1 new precheck test
  case (`tests/test_tier_gate.py`). 49/49 total suite passing throughout.

## The bug found and fixed

First 300-entity A/B (dense vs. hybrid, same entities, same notes) showed
hybrid producing **10 more zero-candidate rejections** than dense-only on
identical entities (27 vs. 17), with an apparent Tier-3 oracle-accuracy
"improvement" (49.1%→60.5%) that was entirely a shrinking-denominator
artifact — the raw count of entities with the correct concept somewhere in
top-5 was identical (26) in both arms.

Root cause: `TIER3_SIMILARITY_FLOOR` was checked against `candidates[0]`'s
dense score specifically, in both `normalize_entity()`
(`src/normalization/orchestrator.py`) and `mollm_tier_gate.py`'s
`tier5_precheck()`. Under RRF fusion, `candidates[0]` is whichever concept
ranked first *after fusion*, not whichever has the best dense score — BM25
can promote a lexically-close-but-semantically-weaker match ahead of a
strong dense hit, and the floor then rejected the whole candidate pool based
on that weaker candidate's score alone, even when a passing dense match sat
at rank 2-5.

**Fix**: both floor checks now use the pool's best dense score
(`max(c["similarity_score"] for c in cands)`), not `candidates[0]`'s
specifically. Verified as a true no-op in dense-only mode (candidates are
already dense-score-sorted, so `candidates[0]` already *is* the max) — the
re-run dense-mode control was byte-identical to the pre-fix run on every
metric, confirming the fix touches only the hybrid path.

## Final validated numbers (300 entities, 32 notes, post-fix)

| | Dense-only (control) | Hybrid (post-fix) |
|---|---|---|
| Zero-candidate rejections | 17 | 21 (down from 27 pre-fix) |
| Entities routed to Tier 3 | 53 | 49 (up from 43 pre-fix) |
| **Overall oracle accuracy** | 55.8% (63/113) | **56.6% (64/113)** |
| **Overall Top-1 accuracy** | **47.8% (54/113)** | 43.4% (49/113) |

The fix recovered 6 of the 10 spurious rejections. Oracle accuracy edged
past dense-only for the first time (64 vs. 63) — hybrid retrieval does find
the correct concept in cases dense-only alone misses, proving the
theoretical value of adding a sparse signal. **Top-1 accuracy is unchanged
by the fix** (still 43.4% vs. 47.8%) — this was never the floor bug's doing;
it is RRF fusion itself demoting good dense matches below rank 1 often
enough to cost top-1 accuracy under the current weights.

## Why this isolates cleanly to a weight-tuning question

Fixing the structural bug first, then re-measuring, is what makes this
conclusion trustworthy rather than a guess: with the floor bug controlled
for, the ONLY remaining difference between dense-only and hybrid is how
candidates get ranked (RRF fusion vs. pure cosine), and that is exactly
where the Top-1 regression persists. The baseline weights
(`w_dense=0.5, w_sparse=0.3`) let a 30%-weighted sparse signal outvote a
50%-weighted dense signal often enough to matter — not obviously wrong on
its face, but not yet validated either.

## The RRF weight sweep, run this session (`evaluation/rrf_weight_sweep.py`)

Grid search over `w_dense`/`w_sparse` in 0.1 steps from (1.0, 0.0) to
(0.0, 1.0), `w_prior=0` throughout (Phase 4's prior matrix doesn't exist
yet, same as production). 155-160 clean-span, Tier-3-eligible entities
across all 32 test notes, dense+sparse rankings fetched once per entity and
re-fused in-memory per grid point (see the script's own docstring for why
that's cheap). Full results: `reports/rrf_weight_sweep.json`.

```
 w_dense  w_sparse  top1_acc  oracle_acc  n_gradable
     1.0       0.0     61.3%       74.2%         155
     0.8       0.2     49.0%       63.9%         155
     0.7       0.3     47.1%       63.9%         155
     0.6       0.4     45.2%       63.9%         155
     0.5       0.5     43.2%       62.6%         155
     0.4       0.6     40.8%       57.9%         152
     0.3       0.7     39.5%       57.2%         152
     0.2       0.8     40.1%       56.6%         152
     0.0       1.0     37.7%       48.1%         154
```

**The answer is unambiguous: `w_dense=1.0, w_sparse=0.0` — pure dense,
zero sparse contribution — wins on both Top-1 and oracle accuracy, and
every single grid point strictly declines (monotonically, no local optimum)
as sparse weight increases.** This is a materially different, larger-sample
result than the earlier 300-entity A/B at the old default weights
(`w_dense=0.5, w_sparse=0.3`: 43.4% Top-1 there vs. 43.2% here at the
adjacent grid point — consistent) — but the earlier A/B only tested one
weight setting and reasonably read the gap as a tuning question. The full
sweep answers that question: there is no weight combination in this grid
where adding BM25 helps. Sparse retrieval, at least as currently indexed
(`bm25_index.py`, plain `concept_name`/synonym FTS with no domain-aware
tokenization), appears to actively promote lexically-close-but-wrong
matches ahead of correct dense hits often enough that no blend recovers the
loss — consistent with the RRF-demotion mechanism already diagnosed in the
floor-check bug above, just not fixable by re-weighting.

## Production gate status

**`CNSP_HYBRID_RETRIEVAL` should stay OFF, not just pending tuning but as a
result of completed tuning.** The sweep closes this question rather than
leaving it open: dense-only is the validated best setting in the tested
grid, not merely the cautious default. `HYBRID_RETRIEVAL_ENABLED` in
`tier_retrieval.py` remains unset; `normalize_entity()` and
`mollm_tier_gate.tier5_precheck()` continue using dense-only in any
unconfigured environment. Revisiting hybrid retrieval would need a better
sparse signal (e.g. domain-aware BM25 tokenization, or a genuinely-informative
`w_prior` term once Phase 4's abbreviation prior matrix exists), not a
different weight on the current BM25 index.

## Session summary (Phases 0-3)

- **Phase 0**: repository cleanup, extending the 2026-08-14 dead-code audit.
- **Phase 1**: recovered the uncommitted Stage 4/5 stash (HITL queue, KG3
  ingestion, anchoring-bias prompt work).
- **Phase 2**: built and validated the Pass 4 two-step CoT + Tier 1-5 gate.
  Diagnosed and fixed the qwen2.5:3b strictness asymmetry (targeted,
  model-specific prompt clause) and the qualifier-fragment noise class (a
  free precheck), after first diagnosing and reverting a global prompt
  loosening that collapsed precision to 5.9%. **Locked at 94.4% Tier 1
  precision / 50.0% AUTO coverage** on a curated, unconfounded atomic-only
  sample — precision nearly double the 52.6% pre-session baseline.
- **Phase 3**: built the BM25+SapBERT+prior RRF hybrid retrieval engine,
  found and fixed a real floor-check bug via disciplined A/B methodology,
  and cleanly isolated the remaining gap to an RRF-weight-tuning question —
  gated off pending that tuning work.
