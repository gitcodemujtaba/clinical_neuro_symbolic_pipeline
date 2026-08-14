"""
scripts/run_stage3_batch.py — full-corpus Stage 3 (MoLLM) batch runner.

WHY THIS EXISTS. scripts/test_stage3_live.py is deliberately a single-note
diagnostic tool ("a diagnostic exercise, not a benchmark" per its own
docstring) -- it takes one --note-id, prints the full prompt for every
record, and exists to catch integration faults cheaply. Running it 25 times
by hand across a real batch (measured smoke test: ~85% of entities LOW-tier,
~2,350 entities corpus-wide) is not a serious plan for an unattended,
multi-hour run. This script reuses the exact same Stage 2 -> retrieval ->
validate_record() -> store_decision() pipeline test_stage3_live.py exercises,
but loops it across every requested note, tracks running totals, and is
built to survive being killed and restarted.

RESUMABILITY. Before calling validate_record() for an entity, this script
checks whether a mollm_decisions row already exists for that entity_id
(is_test=TRUE). If so, it is skipped. validate_record() is only ever called
once per entity regardless of tier, so this check is correct independent of
which --tier a given run used. This means: if vLLM hangs or the process is
killed at hour 8, re-running the EXACT SAME command resumes from wherever it
stopped rather than re-processing everything (and re-spending the LLM calls
already paid for) from zero.

CIRCUIT BREAKER. A single flaky entity should not stop a 10-hour run, but a
genuinely dead or hung vLLM server should not be ground through for the
remaining hours either -- each failure there costs a full request timeout,
not a fast failure. --max-consecutive-failures (default 5) stops the batch
early with a clear message once that many entities in a row raise, instead
of silently burning the rest of the night retrying a dead endpoint. A single
success anywhere resets the counter, so an intermittent issue does not trip
it.

WHAT THIS SCRIPT DOES NOT DO. It does not tell you what fraction of what it
processes will be GRADABLE by evaluation/cal_eval.py's threshold sweep --
see the smoke test findings this was built after: a `mode: contradiction`
entity with model disagreement is correctly routed HITL_REQUIRED but is not
gradable against gold either way. Coverage/gradability is
evaluation/cal_eval.py's job once this has run, not this script's.

Run (needs a writable DuckDB connection, so nothing else should hold it open
for writes concurrently -- DuckDB allows many readers OR one writer):
  python3 scripts/run_stage3_batch.py --note-ids 17739994-DS-31,10043750-DS-6,...
  python3 scripts/run_stage3_batch.py --note-ids ... --limit-per-note 2   # light touch across all notes first
  python3 scripts/run_stage3_batch.py --note-ids ... > /tmp/stage3_batch.log 2>&1 &   # unattended
"""

import argparse
import collections
import os
import sys
import time
import traceback

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
sys.path.insert(0, PROJECT_DIR)

# Same candidates and same fallback order as scripts/test_stage3_live.py --
# kept in sync deliberately rather than imported, since importing it would
# also import its argparse-driven main() machinery for no benefit here.
TRIPLETS_CANDIDATES = [
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded_rules_added"),
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded"),
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned"),
]

DEFAULT_MAX_CONSECUTIVE_FAILURES = 5


def already_processed_entity_ids(conn, note_ids) -> set:
    """entity_ids that already have a stored, ERROR-FREE is_test=TRUE decision,
    across ANY mode/tier -- validate_record() runs exactly once per entity
    regardless of which tier requested it, so this check does not need to
    know which --tier the earlier (possibly interrupted) run used.

    Deliberately excludes rows with error IS NOT NULL. store_decision() is
    called unconditionally in the loop below, including for artifacts that
    carry a caught exception's error message (e.g. a JSON-parse failure), so
    a transient failure still leaves a row here. Treating that row as "done"
    would make it permanently unskippable on resume -- exactly the entities
    a bugfix (like the 2026-08-11 salvage-regex fix) is meant to help would
    be silently skipped forever instead of retried. Only a clean, error-free
    decision counts as truly processed.
    """
    rows = conn.execute("""
        SELECT DISTINCT entity_id FROM mollm_decisions
        WHERE is_test = TRUE AND error IS NULL AND note_id IN ({})
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()
    return {r[0] for r in rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None,
                     help="Comma-separated note_ids. Default: every note_id with "
                          "is_test=TRUE rows in extracted_entities.")
    ap.add_argument("--tier", default="LOW", choices=["HIGH", "LOW"],
                     help="confidence_tier_in to validate (default LOW -- HIGH-tier "
                          "entities were never intended to reach Stage 3).")
    ap.add_argument("--limit-per-note", type=int, default=None,
                     help="Cap entities validated per note. Useful for a light pass "
                          "across every note first, to catch a note-specific crash "
                          "before committing to the full unattended run.")
    ap.add_argument("--max-consecutive-failures", type=int,
                     default=DEFAULT_MAX_CONSECUTIVE_FAILURES)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--calibrator", default=None,
                     help="path to a MoLLMCalibrator .pkl (src/mollm_calibrator.py, "
                          "fit via scripts/fit_mollm_calibrator.py) to score routing "
                          "decisions with instead of raw composite_confidence. Omit to "
                          "keep the pre-calibrator behavior (route() compares "
                          "composite_confidence directly against AUTO_VALIDATE_THRESHOLD/"
                          "MOLLM_RESOLVE_THRESHOLD) -- see cal_report.json's 2026-08-13 "
                          "finding for why that raw signal is currently unreliable "
                          "(ECE=0.773, [0.9,1.0)-confidence bin only 12.5% accurate).")
    ap.add_argument("--shuffle-candidates", action="store_true",
                     help="EXPERIMENT (2026-08-13 P2.2). Permute each entity's "
                          "candidate list before prompting, recording the "
                          "permutation per decision. Tests whether the 22.4%% "
                          "INTRODUCED_ERROR rate is partly candidate-POSITION bias "
                          "rather than model quality -- candidates are normally "
                          "presented in concept_id order, and the observed errors "
                          "cluster in near-identical lab strings. Run the same "
                          "slice with and without this and compare "
                          "evaluation/stage2b_cal_eval.py's cross-tab. Seeded per "
                          "entity_id, so a re-run reproduces the same permutation.")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    print("=" * 78)
    print("STAGE 3 BATCH RUN")
    print("=" * 78)

    from src.llm_client import BIOMISTRAL_BASE_URL, OPENBIOLLM_BASE_URL, build_clients
    from scripts.test_stage3_live import preflight

    if not preflight({"biomistral": BIOMISTRAL_BASE_URL, "openbiollm": OPENBIOLLM_BASE_URL}):
        print("\nAborting -- start both vLLM servers first: bash scripts/start_vllm.sh")
        return 1

    from src.mollm_ensemble import load_validation_records, store_decision, validate_record
    from src.retrieval import DuckDBHierarchy, GroundingRetriever, GuidelineIndex, VocabularyRetriever
    from src.mollm_calibrator import MoLLMCalibrator

    triplets = next((p for p in TRIPLETS_CANDIDATES if os.path.exists(p)), None)
    if not triplets:
        print(f"\nNo guideline corpus found. Tried: {TRIPLETS_CANDIDATES}")
        return 1

    # CALIBRATOR LOADING -- loud either way, never silent. MoLLMCalibrator.load()
    # itself already degrades gracefully (missing file, corrupt pickle, or a
    # feature_set_version mismatch all just leave .model = None rather than
    # raising -- see its docstring), but a batch run silently falling back to
    # raw composite_confidence because of a typo'd --calibrator path is exactly
    # the kind of mistake this project's "report exclusions loudly" policy
    # (evaluation/cal_eval.py's module docstring) exists to prevent elsewhere,
    # so it is surfaced here too rather than only being visible in artifact
    # rows after the fact.
    calibrator = None
    if args.calibrator:
        calibrator = MoLLMCalibrator.load(args.calibrator)
        if calibrator.model is None:
            print(f"\nWARNING: --calibrator {args.calibrator} failed to load or is "
                  f"untrained (missing file, corrupt pickle, or feature_set_version "
                  f"mismatch) -- every decision this run will fall back to raw "
                  f"composite_confidence, identical to not passing --calibrator at all.")
        else:
            print(f"\ncalibrator: {args.calibrator} "
                  f"({calibrator.n_training_examples} training examples, "
                  f"feature_set_version={calibrator.feature_set_version})")
    else:
        print("\ncalibrator: none -- routing on raw composite_confidence directly")

    # Read-write: this script writes via store_decision(). DuckDB allows many
    # readers OR one writer, so nothing else (another pipeline run, a manual
    # inspect_duckdb.py session held open) should be writing concurrently, or
    # this connect() call itself will fail or block.
    conn = duckdb.connect(args.db, read_only=False)
    try:
        if args.note_ids:
            note_ids = [n.strip() for n in args.note_ids.split(",") if n.strip()]
        else:
            note_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test = TRUE"
            ).fetchall()]
        if not note_ids:
            raise SystemExit("No is_test=TRUE rows in extracted_entities. "
                             "Run scripts/test_pipeline_e2e.py first.")

        print(f"db:    {args.db}")
        print(f"notes: {len(note_ids)}")
        print(f"tier:  {args.tier}")

        index = GuidelineIndex(triplets)
        vocab = VocabularyRetriever(conn)
        retriever = GroundingRetriever(index, vocab, hierarchy=DuckDBHierarchy(conn))
        clients = build_clients()
        print(f"guideline KG: {index.stats['nodes']} nodes, {index.stats['rules']} rules")

        already_done = already_processed_entity_ids(conn, note_ids)
        print(f"resume check: {len(already_done)} entity_id(s) already have a stored "
              f"decision and will be skipped\n")

        routing_totals = collections.Counter()
        mode_totals = collections.Counter()
        decoding_mode_totals = collections.Counter()
        n_processed = 0
        n_skipped = 0
        n_errors = 0
        n_calibrator_scored = 0
        n_degenerate = 0
        n_degenerate_retried = 0
        n_shuffled = 0
        consecutive_failures = 0
        start_time = time.time()

        for note_idx, note_id in enumerate(note_ids, 1):
            records = load_validation_records(conn, note_id, tier=args.tier)
            if args.limit_per_note:
                records = records[:args.limit_per_note]
            todo = [r for r in records if r["entity_id"] not in already_done]
            n_skipped += len(records) - len(todo)

            print(f"[note {note_idx}/{len(note_ids)}] {note_id}: {len(records)} "
                  f"{args.tier}-tier record(s), {len(todo)} remaining after resume-skip")

            for rec_idx, rec in enumerate(todo, 1):
                elapsed_min = (time.time() - start_time) / 60
                print(f"  [{rec_idx}/{len(todo)}] {rec['original_text']!r} "
                      f"({rec['gliner_label']})  "
                      f"[elapsed {elapsed_min:.1f}m | done {n_processed} | "
                      f"skipped {n_skipped} | errors {n_errors}]")

                try:
                    artifact = validate_record(
                        rec, retriever, clients=clients, calibrator=calibrator,
                        shuffle_candidates=args.shuffle_candidates)
                except Exception as exc:
                    n_errors += 1
                    consecutive_failures += 1
                    print(f"    ERROR: {exc.__class__.__name__}: {exc}")
                    traceback.print_exc()
                    if consecutive_failures >= args.max_consecutive_failures:
                        print(f"\n{consecutive_failures} consecutive failures -- "
                              f"stopping early. vLLM may be down or hung; check "
                              f"/tmp/biomistral.log and /tmp/openbiollm.log, restart "
                              f"if needed (bash scripts/start_vllm.sh), then re-run "
                              f"this EXACT command -- already-stored decisions will "
                              f"be skipped automatically.")
                        return 1
                    continue

                consecutive_failures = 0
                n_processed += 1
                already_done.add(rec["entity_id"])
                routing_totals[artifact.get("mollm_routing_decision")] += 1
                mode_totals[artifact.get("mode")] += 1
                if artifact.get("calibrator_score") is not None:
                    n_calibrator_scored += 1
                # 2026-08-13 (P4). These two counters are the point of the
                # degeneracy work: they are what make the FREQUENCY_PENALTY
                # fix's effect a NUMBER rather than an assertion. Compare
                # against the report S4.2 baseline of 23.8% of BioMistral
                # verdicts.
                if artifact.get("degenerate_generation"):
                    n_degenerate += 1
                if any(m.get("degenerate_retried")
                       for m in (artifact.get("models") or [])):
                    n_degenerate_retried += 1
                if artifact.get("candidate_permutation"):
                    n_shuffled += 1
                for m in artifact.get("models", []) or []:
                    decoding_mode_totals[m.get("decoding_mode")] += 1

                if artifact.get("error"):
                    print(f"    artifact error: {artifact['error']} "
                          f"-> {artifact.get('mollm_routing_decision')}")
                else:
                    print(f"    -> {artifact.get('mollm_routing_decision')} "
                          f"({artifact.get('queue_reason') or 'ok'}), "
                          f"mode={artifact.get('mode')}")

                store_decision(artifact, conn, is_test=True)

        elapsed_min = (time.time() - start_time) / 60
        print("\n" + "=" * 78)
        print("BATCH COMPLETE")
        print("=" * 78)
        print(f"processed: {n_processed}   skipped(resume): {n_skipped}   "
              f"errors: {n_errors}")
        print(f"elapsed: {elapsed_min:.1f} minutes")
        if calibrator is not None and calibrator.model is not None:
            print(f"calibrator_score used on {n_calibrator_scored}/{n_processed} "
                  f"decision(s) (the rest were forced to HITL_REQUIRED by one of "
                  f"route()'s three hard safety rules before the calibrator was "
                  f"ever consulted -- see route()'s docstring in src/mollm_ensemble.py)")
        print(f"\ndegenerate generations: {n_degenerate}/{n_processed} decision(s) "
              f"still degenerate after retry; {n_degenerate_retried} triggered a "
              f"retry at all")
        print(f"  (2026-08-13 report S4.2 baseline, BEFORE FREQUENCY_PENALTY: "
              f"11.9% of all per-model verdicts, 23.8% of BioMistral's)")
        if args.shuffle_candidates:
            print(f"candidate lists shuffled: {n_shuffled}/{n_processed} "
                  f"(the rest had <2 candidates or drew the identity permutation)")
            print("  EXPERIMENT RUN -- compare this batch's INTRODUCED_ERROR rate "
                  "against an unshuffled\n  run of the same slice "
                  "(evaluation/stage2b_cal_eval.py). Do NOT pool the two.")
        print(f"\nrouting: {dict(routing_totals)}")
        print(f"mode:    {dict(mode_totals)}")
        print(f"decoding modes: {dict(decoding_mode_totals)}")
        if decoding_mode_totals.get("json_object_unguided"):
            print("\nWARNING: some calls fell back to unguided decoding. "
                  "evaluation/cal_eval.py's decoding_purity split will separate "
                  "these out -- see its report, do not average with the guided subset.")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
