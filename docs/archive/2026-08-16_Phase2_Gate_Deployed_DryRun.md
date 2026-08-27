# Phase 2 gate deployed to the production Stage 3 entry point, dry-run, 2026-08-16

**Status: `src.mollm_tier_gate.route_tier()` is now wired into a real
production batch runner and verified end-to-end on live data. KG3
(Memgraph) writes remain dry-run only. Nothing changed about what a human
reviewer sees. Hybrid retrieval stays off.**

## Scope decision

User chose the conservative deploy option over a full production cutover:
wire the new gate into the pipeline's Stage 3 entry point so it runs for
real on the corpus, but keep `src.kg3_ingestion.ingest_auto_decision()` in
`dry_run=True` and leave `src.hitl_queue.enqueue_pending_cases()` queuing
every decision regardless of tier -- the same posture that module's own
docstring has held since 2026-08-14 ("a deliberate, temporary
conservatism"). The alternative (real, unreviewed KG3 writes for AUTO
tiers) was explicitly declined: Phase 2's 94.4% precision figure is based
on only 18 gradable entities, which isn't yet the scale this project's own
standards require before writing unreviewed.

## What was built

- **`src.mollm_tier_gate.store_tier_decision()`**: persists a `route_tier()`
  decision to a new table, `mollm_tier_gate_decisions` -- deliberately
  separate from `mollm_decisions` (different artifact shape; no
  `ensemble_agreement`/`citation_verified`/`mode`, since the two-step CoT
  doesn't use those contradiction-audit concepts). Returns the decision
  dict enriched with a generated `mollm_call_id`, so the same object can be
  passed straight to `ingest_auto_decision()`.
- **`src.hitl_queue`**: `enqueue_pending_cases()` now reads a third source
  table (`mollm_tier_gate_decisions`) alongside the existing two, queuing
  every decision for human review regardless of tier -- including Tier 1/2/3
  AUTO decisions, since the "auto" label is not yet trusted enough at scale
  to skip review. New `_presented_suggestion_from_tier_gate_decision()`
  helper builds the reviewer-facing payload; simpler than the other two
  sources' concept-id resolution since `route_tier()` already records
  `final_candidate_index` directly rather than requiring text-parsing.
- **`scripts/run_stage3_tier_gate.py`**: the production counterpart to
  `scripts/run_stage3_batch.py`, using `route_tier()` instead of the older
  binary `route()`. Resumable (same `already_processed_entity_ids()`
  pattern, checked against the new table), circuit-breaker on consecutive
  failures. For every Tier 1/2/3 decision, calls
  `ingest_auto_decision(driver, decision, entity_fields, dry_run=True)` and
  logs what would have been written -- exercising the real write-path logic
  at batch scale without ever calling Memgraph's write transaction.

## Verified end-to-end (real Ollama calls, real DB, real Memgraph connection)

Ran against note `10000032-DS-21` (10 entities, resumability re-tested
across two separate invocations):
- Preflight, Ollama models, Memgraph connectivity all confirmed live.
- Resumability confirmed: second run correctly skipped the 3 entities the
  first run had already stored.
- Got a real Tier 1 hit (`'orthopnea'` -> `'Orthopnea'`, AUTO_VALIDATED) and
  confirmed the dry-run KG3 write path fires correctly:
  `[dry-run KG3 write OK] would write concept_id=315361 (Orthopnea)` --
  Memgraph was never actually written to.
- Confirmed `HYBRID_RETRIEVAL_ENABLED: False` is what this run used, per
  the Phase 3 findings doc's recommendation.
- Ran `enqueue_pending_cases()` afterward: all 10 stored tier-gate
  decisions correctly appeared in `hitl_review_queue` tagged
  `source_table='mollm_tier_gate_decisions'`.

All 49 existing tests still pass.

## What did NOT change

- No row has ever been written to Memgraph by this deployment -- every
  `ingest_auto_decision()` call this script makes uses `dry_run=True`
  unconditionally; there is no flag or code path in
  `scripts/run_stage3_tier_gate.py` that flips it.
- `CNSP_HYBRID_RETRIEVAL` is not set by this script or anywhere in this
  deployment -- dense-only Tier 3 retrieval remains what production uses.
- `src.mollm_ensemble.route()` (the older binary gate) and
  `scripts/run_stage3_batch.py` are untouched and still exist; nothing was
  removed or deprecated. Running `run_stage3_batch.py` still uses the old
  path exactly as before.
- Human reviewers see no behavioral change: every decision, from either the
  old or new gate, still lands in `hitl_review_queue` as PENDING.

## Next steps (not this session)

- Run `scripts/run_stage3_tier_gate.py` across a much larger slice of the
  corpus to build a real-scale precision sample (18 gradable entities is
  not enough to trust for a real write-gate decision).
- Once that larger sample confirms precision holds, the actual "go live"
  step is narrow and identifiable: flip `ingest_auto_decision()`'s call
  site in this script from `dry_run=True` to `dry_run=False` for Tier 1/2/3,
  and update `enqueue_pending_cases()` to stop queuing those tiers (per the
  original plan's Phase 2 design) -- both deliberately NOT done here.
