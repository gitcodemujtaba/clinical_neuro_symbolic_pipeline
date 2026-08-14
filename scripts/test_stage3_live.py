"""
scripts/test_stage3_live.py

Live exercise of Stage 3 against the real Ollama-served ensemble (2026-08-14:
qwen2.5:3b, llama3.2:3b, phi4-mini -- see src/llm_client.py's docstring for
why this replaced the earlier two-model vLLM setup). This script is
deliberately verbose rather than terse, because the first run of an untested
integration is a diagnostic exercise, not a benchmark.

WHAT IT CHECKS, IN ORDER (each gates the next):
  1. The Ollama server is reachable and each required model is pulled.
     Failing here costs a second; failing later costs a model load and a
     confusing traceback.
  2. Stage 2 output can be read back into ValidationRecord shape.
  3. Retrieval produces evidence (or explains why it did not).
  4. A prompt can be built and fits the token budget.
  5. --dry-run stops here. Everything above is free.
  6. The models return parseable JSON with an in-vocabulary verdict.
  7. Logprobs are present and a verdict confidence can be extracted.
  8. Routing and citation verification produce a decision artifact.

--dry-run exists because steps 1-4 catch most integration faults at zero LLM
cost, and because the prompts should be read by a human before anyone trusts
what a model says about them.

Run:
  python3 scripts/test_stage3_live.py --dry-run --limit 3 --verbose
  python3 scripts/test_stage3_live.py --limit 3 --verbose
"""

import argparse
import collections
import json
import os
import sys
import urllib.error
import urllib.request

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
TRIPLETS_CANDIDATES = [
    # 2026-08-11 Stage3 Issue1 rule backfill: 42->12 rule-less high-frequency
    # nodes fixed against the 51-file _grounded corpus, then the 25 files that
    # never went through grounding backfill at all were merged in and given
    # the same treatment (26->14 remaining). See
    # docs/Stage3_Issue1_Rule_Backfill.md. Non-destructive: the original
    # _grounded/ and _cleaned/ dirs below are untouched and still work as
    # fallbacks if this one is ever missing.
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded_rules_added"),
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded"),
    os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned"),
]


def preflight(model_names) -> bool:
    """Confirms the Ollama server is reachable and each required model is
    pulled (2026-08-14: replaces the old per-model vLLM :8000/:8001 HTTP
    check -- all three models now share one Ollama daemon, so there is one
    server to reach and N model names to confirm on it, not N separate
    endpoints).

    Checked BEFORE importing src.mollm_ensemble, because that import pulls in
    src.retrieval; discovering a dead server after a model load wastes
    minutes and buries the real cause in an unrelated traceback. Matched
    against `ollama list` because llm_client.build_clients() sends a `model`
    field that must match exactly what's pulled -- a mismatch produces a 404
    that reads like a network error. A bare name with no ":tag" (e.g.
    "phi4-mini") is Ollama's own shorthand for ":latest", so that suffix is
    tried too before declaring a model missing.
    """
    import ollama

    ok = True
    print("--- 1. OLLAMA MODEL PREFLIGHT ---")
    try:
        available = {m.model for m in ollama.list().models}
    except Exception as exc:
        print(f"  XX   Ollama server unreachable -> {exc}")
        print("\n  Start it (e.g. `ollama serve`), or run with --dry-run to")
        print("  exercise everything up to the LLM call.")
        return False

    for name in model_names:
        candidates = {name, f"{name}:latest"} if ":" not in name else {name}
        if candidates & available:
            print(f"  OK   {name}")
        else:
            ok = False
            print(f"  XX   {name}  -> not pulled (available: {sorted(available)})")
    if not ok:
        print("\n  One or more required models are missing. Run `ollama pull <model>`,")
        print("  or run with --dry-run to exercise everything up to the LLM call.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note-id", default="10000032-DS-21")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--tier", choices=["HIGH", "LOW"], default=None,
                    help="only validate records in this confidence tier")
    ap.add_argument("--dry-run", action="store_true",
                    help="build prompts but make no LLM calls")
    ap.add_argument("--verbose", action="store_true", help="print full prompts")
    ap.add_argument("--store", action="store_true",
                    help="persist decision artifacts to mollm_decisions")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    from src.llm_client import MODEL_NAMES, PROMPT_BUDGET_TOKENS, CONTEXT_WINDOW_TOKENS

    print("=" * 78)
    print("STAGE 3 LIVE TEST")
    print("=" * 78)

    endpoints_up = preflight(MODEL_NAMES)
    if not endpoints_up and not args.dry_run:
        return 1

    from src.mollm_ensemble import (
        SYSTEM_PROMPT, build_prompt, load_validation_records, store_decision,
        validate_record,
    )
    from src.retrieval import (
        DuckDBHierarchy, GroundingRetriever, GuidelineIndex, VocabularyRetriever,
    )

    triplets = next((p for p in TRIPLETS_CANDIDATES if os.path.exists(p)), None)
    if not triplets:
        print(f"\nNo guideline corpus found. Tried: {TRIPLETS_CANDIDATES}")
        return 1

    # Read-write only when --store; otherwise read-only so this can run
    # alongside a pipeline batch (DuckDB permits many readers OR one writer).
    conn = duckdb.connect(args.db, read_only=not args.store)

    print("\n--- 2. LOADING STAGE 2 OUTPUT ---")
    records = load_validation_records(conn, args.note_id, limit=args.limit, tier=args.tier)
    print(f"  {len(records)} record(s) for note {args.note_id}"
          + (f" (tier={args.tier})" if args.tier else ""))
    if not records:
        print("  Nothing to validate. Run scripts/test_pipeline_e2e.py first,")
        print("  and check that entity_id is populated in both Stage 2 tables.")
        return 1

    print("\n--- 3. RETRIEVAL ---")
    index = GuidelineIndex(triplets)
    vocab = VocabularyRetriever(conn)
    retriever = GroundingRetriever(index, vocab, hierarchy=DuckDBHierarchy(conn))
    print(f"  guideline KG: {index.stats['nodes']} nodes, {index.stats['rules']} rules")
    print(f"  hierarchy: DuckDBHierarchy (athena_concept_ancestor)")

    artifacts = []
    for i, rec in enumerate(records, 1):
        print("\n" + "=" * 78)
        print(f"[{i}/{len(records)}] {rec['original_text']!r} "
              f"({rec['gliner_label']}, tier={rec['confidence_tier_in']}, "
              f"assertion={rec['assertion_status']}/{rec['experiencer']})")
        print("=" * 78)

        retrieval = retriever.retrieve(rec)
        rules = retrieval.get("rules") or []
        print(f"  channels run: {retrieval.get('channels_run')}")
        print(f"  snomed_code: {retrieval.get('snomed_code')}")
        print(f"  rules retrieved: {len(rules)} "
              f"(pooled {retrieval.get('rules_pooled_before_cap', 0)})")
        if retrieval.get("retrieval_skipped_reason"):
            print(f"  SKIPPED: {retrieval['retrieval_skipped_reason']}")
        sup = retrieval.get("suppression") or {}
        if sup:
            print(f"  suppression: { {k: v for k, v in sup.items() if isinstance(v, int) and v} }")
        for r in rules[:3]:
            print(f"    [{r['match_channel']} {r['match_confidence']}] {r['predicate']}: "
                  f"{str(r['source_name'])[:34]} -> {str(r['target_name'])[:34]}")

        prompt, mode, allowed = build_prompt(rec, retrieval)
        approx_tokens = len(prompt) // 4
        print(f"\n  mode: {mode}")
        print(f"  allowed verdicts: {sorted(allowed)}")
        # Budgeted against what the SERVER was launched with, not the model's
        # architectural ceiling. vLLM 400s on prompt+max_tokens > max_model_len,
        # and that failure surfaces as an API error rather than a truncation,
        # so it must be caught here rather than discovered mid-run.
        print(f"  prompt: {len(prompt)} chars ~= {approx_tokens} tokens "
              f"{'OK' if approx_tokens < PROMPT_BUDGET_TOKENS else 'OVER BUDGET'} "
              f"(budget {PROMPT_BUDGET_TOKENS} = context {CONTEXT_WINDOW_TOKENS} - output)")
        if args.verbose:
            print("\n" + "-" * 78)
            print(prompt)
            print("-" * 78)

        if args.dry_run:
            continue

        print("\n  calling ensemble...")
        artifact = validate_record(rec, retriever)
        artifacts.append(artifact)

        if artifact.get("error"):
            print(f"  ERROR: {artifact['error']}")
            print(f"  routed: {artifact['mollm_routing_decision']} "
                  f"({artifact['queue_reason']})")
            continue

        if artifact.get("decoding_mode_mismatch"):
            print("  ! DECODING MODE MISMATCH across models -- their logprobs are")
            print("    not on the same scale; do not use this record for calibration.")
        for m in artifact["models"]:
            lp = m["logprob_confidence"]
            print(f"    {m['model'][:38]:<40} {m['verdict']:<24} "
                  f"logprob={lp if lp is not None else 'UNAVAILABLE'} "
                  f"self={m['raw_confidence_label']} "
                  f"decode={m.get('decoding_mode')}")
            if m.get("verdict_out_of_vocabulary"):
                print(f"      ! returned out-of-vocabulary verdict: "
                      f"{m['verdict_out_of_vocabulary']!r}")
            if m.get("reasoning"):
                print(f"      reasoning: {str(m['reasoning'])[:110]}")
            if m.get("request"):
                print(f"      requested more evidence: {m['request']}")

        print(f"  agreement={artifact['ensemble_agreement']} "
              f"composite={artifact['composite_confidence']} "
              f"({artifact.get('confidence_basis')})")
        print(f"  citations={artifact.get('citations_made', 0)} "
              f"verified={artifact['citation_verified']}")
        for c in artifact.get("citation_checks") or []:
            if not c.get("verified"):
                print(f"    ! {c.get('rule_id')}: {c.get('reason')}")
        print(f"  >>> {artifact['mollm_routing_decision']}"
              + (f"  ({artifact['queue_reason']})" if artifact.get("queue_reason") else ""))
        print(f"      {artifact.get('routing_basis')}")
        if artifact.get("expansion"):
            print(f"      expansion round: {artifact['expansion'].get('applied')}")

        if args.store:
            store_decision(artifact, conn, is_test=True)

    if artifacts:
        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        routes = collections.Counter(a["mollm_routing_decision"] for a in artifacts)
        for k, v in routes.most_common():
            print(f"  {k:<20} {v}")
        modes = collections.Counter(
            m.get("decoding_mode") for a in artifacts for m in a.get("models", []))
        print(f"\n  decoding modes: {dict(modes)}")
        if "json_object_unguided" in modes:
            print("  WARNING: some calls fell back to UNGUIDED json_object mode.")
            print("  vLLM rejected the guided-decoding request for those. Logprob")
            print("  calibration differs between guided and unguided output, so a")
            print("  calibration set must not mix them.")

        no_lp = sum(1 for a in artifacts
                    for m in a.get("models", []) if m.get("logprob_confidence") is None)
        if no_lp:
            print(f"\n  WARNING: {no_lp} model response(s) had no logprob. Routing falls")
            print("  back to HITL for those. Check vLLM was started in a configuration")
            print("  that returns logprobs.")
        if args.store:
            print(f"\n  {len(artifacts)} artifact(s) written to mollm_decisions (is_test=TRUE)")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
