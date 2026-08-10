"""
scripts/profile_stage3.py

Answers two questions that Stage 3's design claims depend on but that nothing
has ever measured.

1. IS IT DETERMINISTIC?
   src/llm_client.py sets TEMPERATURE = 0.0 and justifies it thus: "Stage 3 is
   a validation gate whose outputs are written into a provenance ledger and
   cited in an audit trail; the same entity and the same evidence must produce
   the same verdict on a re-run, or the 'deterministic, traceable' claim
   fails." That claim has never been tested. Temperature 0 makes decoding
   greedy, which is necessary but NOT sufficient for reproducibility: vLLM
   batches requests dynamically, and floating-point reduction order in batched
   GPU kernels can vary with batch composition. AWQ dequantisation adds another
   opportunity for the same. Whether that perturbs a logprob in the 6th decimal
   or flips a verdict is an empirical question.

   This script runs the same records twice and compares, at three strictnesses:
     * routing decision   -- what actually reaches the ledger
     * verdict            -- what the model decided
     * logprob + reasoning-- bit-level reproducibility

   A verdict flip is a genuine problem for the audit claim. A logprob wobble in
   the 6th decimal is not, but it must be MEASURED rather than assumed, because
   composite_confidence is compared against a threshold and a value sitting on
   the boundary could route differently between runs.

2. HOW FAST IS IT?
   Every record costs two sequential LLM calls on a shared 15GB T4 hosting both
   models. No latency figure exists. This matters more than it sounds: if a
   record takes ~10s, a corpus of a few hundred notes at a few hundred entities
   each is measured in days, and the right response may be architectural
   (batching, async, or filtering which records reach Stage 3 at all) rather
   than a matter of patience. Better to know before building a batch runner
   around an assumption.

Both answers come from the same runs, so they are measured together.

NOTE: this opens DuckDB read-only and never writes. It can run alongside a
pipeline batch.

Run:
  python3 scripts/profile_stage3.py --limit 5
  python3 scripts/profile_stage3.py --limit 10 --runs 3 --note-id 10000032-DS-21
"""

import argparse
import collections
import json
import os
import statistics
import sys
import time

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
TRIPLETS_CANDIDATES = [
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded"),
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned"),
]


class TimedClient:
    """Wraps an LLMClient to record wall-clock time per completion.

    Wrapping rather than instrumenting llm_client.py itself: timing is a
    measurement concern and the transport module should not carry a profiling
    hook that only this script uses.
    """

    def __init__(self, inner, label):
        self.inner = inner
        self.label = label
        self.timings = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def complete(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return self.inner.complete(*args, **kwargs)
        finally:
            self.timings.append(time.perf_counter() - t0)


def summarise(values):
    if not values:
        return "no samples"
    vals = sorted(values)
    p95 = vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]
    return (f"n={len(vals)} mean={statistics.mean(vals):.2f}s "
            f"median={statistics.median(vals):.2f}s "
            f"min={vals[0]:.2f}s max={vals[-1]:.2f}s p95={p95:.2f}s")


def fingerprint(artifact):
    """The three comparison strictnesses, as comparable tuples."""
    models = artifact.get("models") or []
    return {
        "routing": artifact.get("mollm_routing_decision"),
        "verdicts": tuple(m.get("verdict") for m in models),
        "logprobs": tuple(m.get("logprob_confidence") for m in models),
        "composite": artifact.get("composite_confidence"),
        "reasoning": tuple((m.get("reasoning") or "")[:400] for m in models),
    }


def corpus_projection(conn, per_record_s):
    """Projects full-corpus runtime from the measured per-record cost.

    Guarded rather than assumed: the table may not exist in every deployment,
    and a failed projection should not lose the latency measurement that
    preceded it.
    """
    try:
        total = conn.sql(
            "SELECT count(*) FROM normalized_entities").fetchone()[0]
    except Exception as exc:
        print(f"  (could not count normalized_entities: {exc})")
        return
    secs = total * per_record_s
    print(f"  records in normalized_entities: {total:,}")
    print(f"  projected serial runtime: {secs/3600:.1f} h ({secs/86400:.1f} days)")
    print("  NOTE: serial, one record at a time, both models sequential.")
    print("  Concurrency would improve this; the KV cache sized at startup")
    print("  (~2-4 sequences per model) is the ceiling on how much.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note-id", default="10000032-DS-21")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--runs", type=int, default=2,
                    help="how many times to repeat each record")
    ap.add_argument("--tier", choices=["HIGH", "LOW"], default=None)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    from src.llm_client import build_clients
    from src.mollm_ensemble import load_validation_records, validate_record
    from src.retrieval import (
        DuckDBHierarchy, GroundingRetriever, GuidelineIndex, VocabularyRetriever,
    )

    triplets = next((p for p in TRIPLETS_CANDIDATES if os.path.exists(p)), None)
    if not triplets:
        print(f"No guideline corpus found. Tried: {TRIPLETS_CANDIDATES}")
        return 1

    conn = duckdb.connect(args.db, read_only=True)
    records = load_validation_records(conn, args.note_id, limit=args.limit,
                                      tier=args.tier)
    if not records:
        print(f"No records for note {args.note_id}. Run Stage 1/2 first.")
        return 1

    index = GuidelineIndex(triplets)
    vocab = VocabularyRetriever(conn)
    retriever = GroundingRetriever(index, vocab, hierarchy=DuckDBHierarchy(conn))

    clients = {k: TimedClient(v, k) for k, v in build_clients().items()}

    print("=" * 78)
    print(f"STAGE 3 PROFILE — {len(records)} record(s) x {args.runs} run(s)")
    print("=" * 78)
    print(f"note: {args.note_id}   guideline KG: {index.stats['nodes']} nodes, "
          f"{index.stats['rules']} rules")

    # ---------------------------------------------------------------- runs --
    per_record_times = []
    results = collections.defaultdict(list)   # entity_id -> [fingerprint,...]
    errors = 0

    for run_i in range(1, args.runs + 1):
        print(f"\n--- run {run_i}/{args.runs} ---")
        for rec in records:
            t0 = time.perf_counter()
            artifact = validate_record(rec, retriever, clients=clients)
            elapsed = time.perf_counter() - t0
            per_record_times.append(elapsed)
            if artifact.get("error"):
                errors += 1
                print(f"  {elapsed:6.2f}s  ERROR  {rec['original_text'][:34]!r} "
                      f"-> {str(artifact['error'])[:60]}")
                continue
            results[rec["entity_id"]].append(fingerprint(artifact))
            print(f"  {elapsed:6.2f}s  {artifact['mollm_routing_decision']:<16} "
                  f"{rec['original_text'][:34]!r}")

    # ------------------------------------------------------- determinism ----
    print("\n" + "=" * 78)
    print("DETERMINISM")
    print("=" * 78)

    comparable = {k: v for k, v in results.items() if len(v) >= 2}
    if not comparable:
        print("  Not enough successful repeat runs to compare.")
    else:
        counts = collections.Counter()
        for eid, fps in comparable.items():
            first = fps[0]
            for key in ("routing", "verdicts", "logprobs", "composite", "reasoning"):
                stable = all(fp[key] == first[key] for fp in fps[1:])
                counts[key] += 1 if stable else 0
                if not stable and key in ("routing", "verdicts"):
                    print(f"  UNSTABLE {key} for {eid}:")
                    for i, fp in enumerate(fps, 1):
                        print(f"    run {i}: {fp[key]}")
                elif not stable and key == "logprobs":
                    print(f"  logprob drift for {eid}:")
                    for i, fp in enumerate(fps, 1):
                        print(f"    run {i}: {fp[key]}")

        n = len(comparable)
        print(f"\n  {n} record(s) compared across {args.runs} runs:")
        for key in ("routing", "verdicts", "composite", "logprobs", "reasoning"):
            print(f"    {key:<10} identical in {counts[key]}/{n}")

        if counts["routing"] == n and counts["verdicts"] == n:
            print("\n  Routing and verdicts reproduce. The audit-trail claim in")
            print("  src/llm_client.py holds at the level that matters.")
            if counts["logprobs"] < n:
                print("  Logprobs vary in low-order digits (batched GPU reduction")
                print("  order). Harmless UNLESS a composite sits on a routing")
                print("  threshold -- worth noting in the calibration write-up.")
        else:
            print("\n  *** VERDICTS OR ROUTING ARE NOT REPRODUCIBLE. ***")
            print("  The 'deterministic, traceable' claim does not hold as")
            print("  written and must be either fixed (seed/batching controls)")
            print("  or restated honestly in the dissertation.")

    # ---------------------------------------------------------- throughput --
    print("\n" + "=" * 78)
    print("THROUGHPUT")
    print("=" * 78)
    print(f"  per record (both models): {summarise(per_record_times)}")
    for label, c in clients.items():
        print(f"  {label:<12} per call: {summarise(c.timings)}")
    if errors:
        print(f"  errors: {errors} (excluded from determinism, included in timing)")

    if per_record_times:
        print()
        corpus_projection(conn, statistics.median(per_record_times))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({
                "note_id": args.note_id, "runs": args.runs,
                "per_record_seconds": per_record_times,
                "per_call_seconds": {k: v.timings for k, v in clients.items()},
                "records_compared": len(comparable) if comparable else 0,
            }, fh, indent=2)
        print(f"\n  raw timings -> {args.json_out}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
