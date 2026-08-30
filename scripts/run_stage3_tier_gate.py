"""
scripts/run_stage3_tier_gate.py — production Stage 3 batch runner using
src.mollm_tier_gate.route_tier() (Pass 4's two-step CoT + Tier 1-5 gate),
NOT src.mollm_ensemble's older binary route(). This is the counterpart to
scripts/run_stage3_batch.py for the new gate.

2026-08-16 DEPLOYMENT SCOPE (user decision after Phase 2/3 validation):
  - route_tier() runs for real, on real corpus data -- this is not a dry
    run or a diagnostic. Every decision is persisted to
    mollm_tier_gate_decisions (src.mollm_tier_gate.store_tier_decision()).
  - src.kg3_ingestion.ingest_auto_decision() IS called for every Tier 1/2/3
    (AUTO_VALIDATED/AUTO_RESOLVED) decision, but ALWAYS with dry_run=True.
    Nothing is written to Memgraph/KG3 by this script. The point of calling
    it at all (rather than skipping it) is to exercise the real write-path
    logic and log what WOULD have been written, at real batch scale, before
    that gate is ever flipped -- see docs/2026-08-15_Phase2_TierGate_Validation.md
    and docs/2026-08-16_Phase3_HybridRetrieval_Validation.md for why 94.4%
    precision on 18 gradable entities is not yet enough evidence to write
    unreviewed.
  - src.hitl_queue.enqueue_pending_cases() now reads mollm_tier_gate_decisions
    too (see that module's 2026-08-16 update) and queues EVERY decision this
    script produces for human review, tier notwithstanding -- same
    "deliberate, temporary conservatism" that module's docstring already
    applies to the other two decision sources. Nothing about running this
    script changes what a reviewer sees.
  - CNSP_HYBRID_RETRIEVAL is NOT set by this script. Candidate retrieval
    uses whatever src.normalization.tier_retrieval.HYBRID_RETRIEVAL_ENABLED
    resolves to in the calling environment -- default unset, i.e. dense-only
    Tier 3, per the Phase 3 findings doc's explicit recommendation to keep
    hybrid retrieval off pending an RRF-weight grid search.

2026-08-17 (plan Phase 6, build-order step 4): route_tier() now gets a
  fitted src.mollm_tier_calibrator.ConsensusCalibrator + this script's own
  DuckDB connection, loaded once before the note loop via
  ConsensusCalibrator.load(..., scoring_note_ids=note_ids) -- passing the
  actual note_ids being processed lets the load-time leakage guard refuse
  the model (falling back to untrained/no-op, never raising) if any of them
  were in its own training set. A missing or corrupt .pkl degrades to the
  same untrained no-op, so this wiring can never make a batch run fail that
  would otherwise have succeeded -- see ConsensusCalibrator.load()'s own
  docstring for the full degrade-gracefully contract.

2026-08-30: route_tier() also gets kg3_driver=memgraph_driver (the SAME
  driver this script already opens for the dry-run KG3 write-path check
  below, reused rather than opening a second one), feeding the calibrator's
  new kg3_confirmation_count feature (FEATURE_SET_VERSION=2). When Memgraph
  isn't reachable at startup, memgraph_driver is None and this reproduces
  prior behavior exactly -- the feature reads back 0 for every entity. The
  currently-loaded production .pkl was fit under FEATURE_SET_VERSION=1, so
  ConsensusCalibrator.load() will report it "untrained (no-op)" until it is
  refit on the new 17-feature set -- this wiring alone does not change any
  routing decision until that refit happens; see
  evaluation/tier_gate_cal_eval.py.

Run:  python3 scripts/run_stage3_tier_gate.py --note-ids 17739994-DS-31,10043750-DS-6,...
      python3 scripts/run_stage3_tier_gate.py --note-ids ... --limit-per-note 2   # light touch first
"""
import argparse
import collections
import os
import sys
import time
import traceback

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
sys.path.insert(0, PROJECT_DIR)

DEFAULT_MAX_CONSECUTIVE_FAILURES = 5


def already_processed_entity_ids(conn, note_ids) -> set:
    """entity_ids that already have a stored mollm_tier_gate_decisions row --
    same resumability contract as run_stage3_batch.py's identically-named
    function, against the new table instead of mollm_decisions. Not filtered
    on is_test: route_tier() runs exactly once per entity regardless of
    which run wrote it, so a re-run should skip it either way."""
    try:
        rows = conn.execute("""
            SELECT DISTINCT entity_id FROM mollm_tier_gate_decisions
            WHERE note_id IN ({})
        """.format(",".join("?" * len(note_ids))), note_ids).fetchall()
    except duckdb.Error:
        return set()  # table doesn't exist yet -- first run ever, nothing to skip
    return {r[0] for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None,
                    help="Comma-separated note_ids. Default: every note_id with "
                         "is_test=TRUE rows in extracted_entities.")
    ap.add_argument("--limit-per-note", type=int, default=None)
    ap.add_argument("--max-consecutive-failures", type=int,
                    default=DEFAULT_MAX_CONSECUTIVE_FAILURES)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--is-test", action="store_true", default=True,
                    help="Marks stored rows is_test=TRUE (default: on, matching "
                         "every other diagnostic/batch script in this repo -- "
                         "this is still a validation deployment, not a production "
                         "corpus run, until precision is re-confirmed at scale).")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    print("=" * 78)
    print("STAGE 3 BATCH RUN -- Pass 4 Tier 1-5 gate (src.mollm_tier_gate)")
    print("=" * 78)

    from src.llm_client import MODEL_NAMES
    from scripts.test_stage3_live import preflight
    if not preflight(MODEL_NAMES):
        print("\nAborting -- pull the missing model(s) with `ollama pull <model>` first.")
        return 1

    from src.mollm_ensemble import load_validation_records
    from src.mollm_tier_gate import AUTO_TIERS, build_clients, route_tier, store_tier_decision
    from src.mollm_tier_calibrator import ConsensusCalibrator, DEFAULT_MODEL_PATH
    from src.kg3_ingestion import UningestibleCase, get_memgraph_driver, ingest_auto_decision
    from src.normalization.tier_retrieval import HYBRID_RETRIEVAL_ENABLED
    from src.batch_status import clear_status, write_status
    from src.db_utils import connect_with_retry as _connect_with_retry_base

    # 2026-08-19: db_utils.connect_with_retry()'s own 300s default assumes
    # the OTHER side of a lock collision is also a short-lived, cycled
    # connection (its own docstring says so explicitly) -- true for another
    # Stage 3 run or Streamlit, but NOT true when this script is deliberately
    # run in parallel with Stage 1-2b (scripts/test_pipeline_e2e.py), which
    # holds ONE connection per NOTE for that note's whole processing time.
    # Proven live: two entities (MCH-30, PLT in note 10993096-DS-12) lost
    # their tier-gate decision entirely when a parallel Stage 1-2b note ran
    # longer than 300s and this script's write connection gave up instead of
    # waiting it out. 1800s (30 min) comfortably covers even a slow note
    # (GLiNER/SapBERT are GPU-accelerated now, but Tier 1-3 normalization
    # with CONTEXTUAL_CANDIDATES_ENABLED can still take several minutes of
    # genuine DB-holding time) while still failing loudly, not silently
    # hanging forever, if something is genuinely deadlocked rather than just
    # slow. Scoped to this script only -- other connect_with_retry() callers
    # (Streamlit, etc.) keep the original 300s, where a long wait would be
    # the wrong tradeoff (a stuck UI, not an unattended batch).
    STAGE3_LOCK_WAIT_SECONDS = 1800

    def connect_with_retry(db_path, read_only=False):
        return _connect_with_retry_base(db_path, read_only=read_only,
                                        max_wait_seconds=STAGE3_LOCK_WAIT_SECONDS)

    print(f"\nHYBRID_RETRIEVAL_ENABLED: {HYBRID_RETRIEVAL_ENABLED} "
          f"(should be False -- see this script's own docstring)")

    memgraph_driver = None
    try:
        memgraph_driver = get_memgraph_driver()
        with memgraph_driver.session() as s:
            s.run("RETURN 1")
        print("Memgraph: reachable (dry-run writes will still be exercised, "
              "nothing will be committed)")
    except Exception as exc:
        print(f"Memgraph: NOT reachable ({exc}) -- dry-run KG3 write-path checks "
              f"will be skipped for this run, everything else proceeds normally")
        memgraph_driver = None

    # 2026-08-18 ("don't lock Streamlit out for 10-45 minutes" fix). This
    # used to be one duckdb.connect() held open for the entire batch --
    # every note, every entity -- even though the actual DB reads/writes
    # inside that window are each well under a second; the rest is
    # GLiNER/SapBERT/3-model LLM inference that never touches the
    # connection. DuckDB's single-writer lock doesn't care how much of
    # that time is spent actually querying -- it excludes every other
    # connection (including Streamlit's) for the connection's WHOLE
    # lifetime. Below, a connection is opened fresh right before each
    # DB-touching operation and closed immediately after, so the lock is
    # only held for milliseconds at a time, not the whole run.
    # connect_with_retry() (not a bare duckdb.connect()) since a
    # short-lived connection can now legitimately collide with another
    # short-lived one (Streamlit's own read, or this same pattern in a
    # concurrently-run script) -- worth waiting out, not crashing on.
    conn = connect_with_retry(args.db, read_only=False)
    try:
        if args.note_ids:
            note_ids = [n.strip() for n in args.note_ids.split(",") if n.strip()]
        else:
            note_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test = TRUE"
            ).fetchall()]
        if not note_ids:
            raise SystemExit("No is_test=TRUE rows in extracted_entities.")

        print(f"db:    {args.db}")
        print(f"notes: {len(note_ids)}")

        calibrator = ConsensusCalibrator.load(DEFAULT_MODEL_PATH, scoring_note_ids=note_ids)
        print(f"calibrator: {'fitted' if calibrator.model is not None else 'untrained (no-op)'}"
              f", loaded from {DEFAULT_MODEL_PATH}")

        clients = build_clients()
        already_done = already_processed_entity_ids(conn, note_ids)
        print(f"resume check: {len(already_done)} entity_id(s) already have a stored "
              f"tier-gate decision and will be skipped\n")
    finally:
        # Setup phase done -- close this connection rather than holding it
        # through the whole note/entity loop below, which opens its own
        # short-lived connections per operation instead.
        conn.close()

    tier_totals = collections.Counter()
    routing_totals = collections.Counter()
    dry_run_write_ok = 0
    dry_run_write_blocked = 0
    n_processed = 0
    n_skipped = 0
    n_errors = 0
    consecutive_failures = 0
    start_time = time.time()

    try:
        for note_idx, note_id in enumerate(note_ids, 1):
            note_conn = connect_with_retry(args.db, read_only=False)
            try:
                records = load_validation_records(note_conn, note_id, tier=None)
            finally:
                note_conn.close()
            if args.limit_per_note:
                records = records[:args.limit_per_note]
            todo = [r for r in records if r["entity_id"] not in already_done]
            n_skipped += len(records) - len(todo)

            print(f"[note {note_idx}/{len(note_ids)}] {note_id}: {len(records)} "
                  f"record(s), {len(todo)} remaining after resume-skip")

            # 2026-08-17: DB-independent progress, so a UI/monitor (e.g.
            # ui/pages/1_🚀_Pipeline_Runner.py) can show real status/ETA even
            # though DuckDB's single-writer lock means it can't open the DB
            # itself while this script runs. Once per NOTE, not per entity --
            # see src.batch_status.write_status()'s own docstring for why.
            write_status("stage3_tier_gate", started_at=start_time,
                        notes_done=note_idx - 1, notes_total=len(note_ids),
                        entities_done=n_processed, current_note_id=note_id, errors=n_errors)

            for rec_idx, rec in enumerate(todo, 1):
                elapsed_min = (time.time() - start_time) / 60
                print(f"  [{rec_idx}/{len(todo)}] {rec['original_text']!r} "
                      f"({rec['gliner_label']})  "
                      f"[elapsed {elapsed_min:.1f}m | done {n_processed} | "
                      f"skipped {n_skipped} | errors {n_errors}]")

                # 2026-08-18 (round 2 -- the first "cycle a connection
                # around route_tier()" attempt was verified LIVE to still
                # lock Streamlit out almost continuously, since route_tier()
                # holds whatever connection it's given open across its own
                # 3-model LLM ensemble call, which is the actual slow part).
                # conn_factory is passed instead of an already-open conn --
                # route_tier() now opens its own brief connection ONLY at
                # the moment it actually needs the calibrator's
                # prior-confirmation lookup (AFTER the LLM calls return),
                # and closes it immediately. store_tier_decision()'s own
                # write still gets its own separate short-lived connection
                # below. Folded into the SAME try/except that already
                # handles per-entity errors, so a connect-retry timeout is
                # just one more entity-level failure, not a script crash.
                # 2026-08-30: kg3_driver=memgraph_driver (the SAME driver
                # already opened above for the dry-run KG3 write-path check,
                # not a fresh one per entity). Unlike conn/conn_factory,
                # there's no DuckDB-style single-writer lock to avoid
                # holding open here -- a Bolt driver already supports many
                # concurrent short sessions from one long-lived driver
                # instance, which is exactly how the write-path check below
                # already uses it for the whole batch. When Memgraph isn't
                # reachable (memgraph_driver is None, see the reachability
                # check above), this reproduces the pre-2026-08-30 behavior
                # exactly -- kg3_confirmation_count reads back 0 for every
                # entity, same "absent evidence" default as an omitted
                # kg3_driver anywhere else.
                try:
                    decision = route_tier(
                        rec, clients=clients, calibrator=calibrator,
                        conn_factory=lambda: connect_with_retry(args.db, read_only=False),
                        kg3_driver=memgraph_driver)
                    write_conn = connect_with_retry(args.db, read_only=False)
                    try:
                        decision = store_tier_decision(decision, rec["entity_id"], note_id,
                                                       write_conn, is_test=args.is_test)
                    finally:
                        write_conn.close()
                except Exception as exc:
                    n_errors += 1
                    consecutive_failures += 1
                    print(f"    ERROR: {exc.__class__.__name__}: {exc}")
                    traceback.print_exc()
                    if consecutive_failures >= args.max_consecutive_failures:
                        print(f"\n{consecutive_failures} consecutive failures -- "
                              f"stopping early. Re-run this EXACT command -- "
                              f"already-stored decisions will be skipped automatically.")
                        return 1
                    continue

                consecutive_failures = 0
                n_processed += 1
                already_done.add(rec["entity_id"])
                tier_totals[decision.get("tier") or "none"] += 1
                routing_totals[decision["mollm_routing_decision"]] += 1

                print(f"    -> tier={decision.get('tier')} "
                      f"routing={decision['mollm_routing_decision']} "
                      f"reason={decision.get('queue_reason') or 'ok'}")

                if decision.get("tier") in AUTO_TIERS and memgraph_driver is not None:
                    entity_fields = {
                        "original_text": rec["original_text"],
                        "entity_label": rec["gliner_label"],
                        "orig_start": rec["orig_start"], "orig_end": rec["orig_end"],
                        "confidence": rec.get("gliner_confidence"),
                        "candidates": rec.get("candidates") or [],
                    }
                    try:
                        write_result = ingest_auto_decision(
                            memgraph_driver, decision, entity_fields, dry_run=True)
                        dry_run_write_ok += 1
                        print(f"    [dry-run KG3 write OK] would write concept_id="
                              f"{write_result['params']['omop_concept_id']} "
                              f"({write_result['params']['concept_name']})")
                    except UningestibleCase as exc:
                        dry_run_write_blocked += 1
                        print(f"    [dry-run KG3 write BLOCKED] {exc}")

        write_status("stage3_tier_gate", started_at=start_time,
                    notes_done=len(note_ids), notes_total=len(note_ids),
                    entities_done=n_processed, current_note_id=None, errors=n_errors)

        elapsed_min = (time.time() - start_time) / 60
        print("\n" + "=" * 78)
        print("BATCH COMPLETE")
        print("=" * 78)
        print(f"processed: {n_processed}   skipped(resume): {n_skipped}   errors: {n_errors}")
        print(f"elapsed: {elapsed_min:.1f} minutes")
        print(f"\ntier distribution: {dict(tier_totals)}")
        print(f"routing: {dict(routing_totals)}")
        auto_n = sum(tier_totals[t] for t in tier_totals if t in AUTO_TIERS)
        print(f"AUTO coverage: {auto_n}/{n_processed} "
              f"({auto_n/n_processed*100:.1f}%)" if n_processed else "AUTO coverage: n/a")
        print(f"\ndry-run KG3 write checks: {dry_run_write_ok} would-succeed, "
              f"{dry_run_write_blocked} blocked (UningestibleCase) -- "
              f"NOTHING was actually written to Memgraph this run")
    finally:
        clear_status("stage3_tier_gate")
        if memgraph_driver is not None:
            memgraph_driver.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
