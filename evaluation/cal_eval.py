"""
evaluation/cal_eval.py — Stage 3 confidence calibration (docs/Stage3_Open_Issues.md
Issue 4, docs/Evaluation_Criteria.md "MoLLM Gate Validation": "Expected
Calibration Error (or a reliability diagram) is computed on the validation
slice before setting production thresholds, and re-checked at T0, T1, and T2").

WHAT THIS SCRIPT DOES AND DOES NOT COVER. Read this before trusting a number
out of it.

Stage 3 issues three kinds of verdict (mollm_ensemble.build_prompt()'s three
`mode`s): resolution, contradiction, non_asserted_check. Only RESOLUTION-mode
decisions are gradable against data this project actually has: the SNOMED CT
Entity Linking Challenge's train_annotations.csv gives a gold concept_id per
span, so "did MoLLM resolve to the candidate whose SNOMED code matches gold"
(or correctly say NONE_CORRECT when none of the candidates do) is a fact this
script can check mechanically, the same way scripts/score_gold_recall.py
checks Stage 2b's own top-1 pick.

CONTRADICTION and NON_ASSERTED_CHECK verdicts (SUPPORTED / CONTRADICTED /
INSUFFICIENT_EVIDENCE) are judgments about guideline compliance, not concept
identity -- the gold set has no label for "was this SUPPORTED verdict
correct," because that was never what SNOMED annotators were asked. Grading
those needs the human "periodic re-audit sample" docs/Evaluation_Criteria.md
already names for exactly this reason. This script reports how many such
records exist and excludes them from every accuracy/ECE number, loudly,
rather than silently only covering part of the truth.

SAMPLE SIZE. As of 2026-08-11 only a handful of notes have been run through
Stage 3 (see docs/Stage3_Open_Issues.md). Every number below is for
validating THIS SCRIPT'S methodology and for re-running as the corpus grows,
not a production threshold fit off a double-digit sample -- same caveat
score_gold_recall.py states for its 3-note IoU numbers, for the same reason.

Run:
  python3 evaluation/cal_eval.py
  python3 evaluation/cal_eval.py --note-ids 10000032-DS-21 --out reports/cal_report.json
"""

import argparse
import collections
import json
import os
import re
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

sys.path.insert(0, PROJECT_DIR)

from src.retrieval import VocabularyRetriever  # noqa: E402
from src.mollm_calibrator import extract_features  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    accuracy as _accuracy, auroc, average_precision, bootstrap_ci,
    compute_ece_report, format_ci, group_by_note, null_model_report,
    print_interpretation_block,
)
from evaluation.splits import add_split_args, assert_no_contamination, resolve_note_ids  # noqa: E402
from scripts.score_gold_recall import (  # noqa: E402
    load_gold, overlaps, _first_existing, GOLD_CANDIDATES,
)

# Equal-width confidence bins for the reliability diagram / ECE sum. 10 bins
# is the conventional default in the calibration literature (Guo et al. 2017)
# and matches what a reliability-diagram figure in the dissertation would use.
ECE_BINS = 10

RESOLVED_RE = re.compile(r"^RESOLVED_TO_CANDIDATE_(\d+)$")

# The three real values src/llm_client.py's LLMClient.complete() writes to
# decoding_mode (see its "decoding_mode" comment): the first two are both
# genuinely constrained generation, just via different serving-API paths
# (OpenAI-spec json_schema vs vLLM's guided_json extra_body) -- their logprob
# semantics are the same "probability among the legal enum values" quantity
# verdict_schema()'s docstring (src/llm_client.py) describes, so both count
# as "guided" for calibration purposes. json_object_unguided is the fallback
# that only asks for valid JSON syntax with no vocabulary constraint at the
# token level -- that is the "free generation" case whose logprob is diluted
# across the whole vocabulary, and is NOT comparable to the guided cases.
# "unknown" is the LLMClient default if decoding_mode was somehow never set;
# treated as untrusted rather than assumed guided, for the same reason a
# missing crosses_sentence_boundary defaults to False rather than True
# elsewhere in this codebase -- an absent signal must not silently count as
# a passing one.
GUIDED_DECODING_MODES = {"guided_json_schema", "guided_json_extra_body"}


def _decoding_purity(decoding_modes: list, mismatch: bool) -> str:
    """Classifies a decision's decoding_modes (the mollm_decisions column,
    one entry per ensemble model that answered) for calibration-set
    inclusion.

    2026-08-11, docs/Stage3_Open_Issues.md Issue 4 follow-up: mollm_ensemble.
    validate_record() already computes decoding_modes and decoding_mode_
    mismatch per decision specifically so a run that silently fell back to
    unguided decoding would be visible -- but nothing downstream ever read
    either field. A calibration run before this fix would have silently
    pooled guided and unguided composite_confidence values into one ECE
    curve and one threshold sweep, which verdict_schema()'s own docstring
    (src/llm_client.py) names as the exact mistake to avoid: "absolute
    logprob values shift... a calibration set must not mix the two."

    Returns "guided" only when EVERY model's decoding_mode is a member of
    GUIDED_DECODING_MODES and there was no cross-model mismatch (a mismatch
    necessarily means at least one model was NOT guided, so this check is
    partly redundant with the modes list itself -- kept explicit anyway
    since decoding_mode_mismatch is the artifact's own documented signal for
    exactly this, and trusting it directly is more legible than re-deriving
    the same fact from list contents alone).

    Returns "unguided_or_mixed" for everything else, INCLUDING an empty/
    missing decoding_modes (older rows written before this column existed,
    or a decision where no model returned logprobs at all) -- absence of
    evidence that decoding was guided is not evidence it was.
    """
    if mismatch:
        return "unguided_or_mixed"
    if decoding_modes and all(m in GUIDED_DECODING_MODES for m in decoding_modes):
        return "guided"
    return "unguided_or_mixed"


def load_gradable_decisions(conn, note_ids):
    """mollm_decisions rows in resolution mode, joined back to their Stage 2
    candidates JSON via the SAME safe composite key
    scripts/score_gold_recall.py uses for normalized_entities (note_id,
    original_text, expanded_text, gliner_label) -- NOT entity_id. That
    caveat is real and documented in score_gold_recall.py's module
    docstring ("KNOWN DB CAVEAT..."): normalized_entities is unique on the
    composite key, not entity_id, so a duplicate-text entity's row can have
    been silently overwritten by a later mention sharing the same key. This
    script inherits the same exposure by reading the same table and avoids
    the same way: never join on entity_id into normalized_entities.

    Each returned decision now also carries `decoding_modes`,
    `decoding_mode_mismatch`, and a derived `decoding_purity` ("guided" /
    "unguided_or_mixed") -- see _decoding_purity()'s docstring. Filtering by
    decoding mode is deliberately NOT done here at the SQL/loading layer:
    every decision is still returned, and main() is what decides how to
    split and report them, so a caller inspecting coverage numbers can see
    the full population rather than a pre-filtered subset with no visible
    trace of what was excluded -- same "report exclusions loudly" policy
    this module's docstring already states for the contradiction/
    non_asserted_check modes.
    """
    # 2026-08-13 BUGFIX: this SELECT used to also read d.decoding_mode_mismatch.
    # That column does not exist. mollm_ensemble.py sets
    # artifact["decoding_mode_mismatch"] on the in-memory decision artifact
    # (~line 928) but store_decision()'s CREATE TABLE / ALTER TABLE / INSERT
    # column lists never included it -- unlike confidence_spread, which had
    # the identical "computed but never persisted" bug and was caught and
    # backfilled via ALTER TABLE on 2026-08-11 (see store_decision()'s
    # migration comment), decoding_mode_mismatch was never backfilled. Any
    # call to this function against a real mollm_decisions table raised a
    # DuckDB Binder Error before this fix -- it was never actually run against
    # live data. Dropped from the SELECT; _decoding_purity() below is called
    # with mismatch=False for every row instead of a persisted flag, which is
    # exactly the "re-derive from decoding_modes list contents alone" fallback
    # its own docstring already names as the alternative -- decoding_modes
    # (JSON) IS persisted, so purity is still correctly classified, just via
    # the less legible of the two equivalent paths the docstring describes.
    rows = conn.execute("""
        SELECT d.mollm_call_id, d.entity_id, d.note_id, d.mode,
               d.ensemble_agreement, d.composite_confidence, d.confidence_spread,
               d.models, d.mollm_routing_decision, d.decoding_modes,
               d.retrieved_context,
               d.grounding_basis, d.annotation_discrepancy, d.rejected_candidates,
               d.confidence_tier_in, d.citation_verified,
               e.original_text, e.expanded_text, e.entity_label,
               e.orig_start, e.orig_end,
               n.candidates
        FROM mollm_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n
          ON n.note_id = e.note_id
         AND n.original_text = e.original_text
         AND n.expanded_text = e.expanded_text
         AND n.gliner_label = e.entity_label
        WHERE d.is_test = TRUE
          AND d.mode = 'resolution'
          AND d.note_id IN ({})
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    # docs/MoLLM_Redesign_Proposal.md S7 step 2, S4.1: grounding_basis,
    # annotation_discrepancy, rejected_candidates, confidence_spread and
    # retrieved_context were all added 2026-08-11 alongside MoLLM-Cal --
    # older rows written before that will have NULL for these, which
    # build_feature_ctx() / extract_features() handle the same way every
    # other "not measured" value in this codebase is handled (defaults to
    # 0.0/False rather than raising), not treated as an error.
    cols = ["mollm_call_id", "entity_id", "note_id", "mode", "ensemble_agreement",
            "composite_confidence", "confidence_spread", "models",
            "mollm_routing_decision", "decoding_modes",
            "retrieved_context", "grounding_basis", "annotation_discrepancy",
            "rejected_candidates", "confidence_tier_in", "citation_verified",
            "original_text", "expanded_text", "entity_label",
            "orig_start", "orig_end", "candidates"]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        d["models"] = _json(d["models"], [])
        d["candidates"] = _json(d["candidates"], [])
        d["decoding_modes"] = _json(d["decoding_modes"], [])
        d["retrieved_context"] = _json(d["retrieved_context"], {})
        d["rejected_candidates"] = _json(d["rejected_candidates"], [])
        # mismatch=False: see the "2026-08-13 BUGFIX" comment above
        # load_gradable_decisions()'s SQL -- the persisted column doesn't
        # exist, so purity falls back to the decoding_modes-list-only check.
        d["decoding_purity"] = _decoding_purity(d["decoding_modes"], False)
        out.append(d)
    return out


def build_feature_ctx(decision: dict) -> dict:
    """Reconstructs the feature_ctx shape src.mollm_calibrator.extract_features()
    expects, from a persisted mollm_decisions row (docs/MoLLM_Redesign_Proposal.md
    S7 step 2). This is the training-data assembly MoLLM-Cal needs -- everything
    here was already computed once by validate_record() and just needs to be
    read back out of the DB in the shape extract_features() was written against.
    """
    return {
        "confidence_tier_in": decision.get("confidence_tier_in"),
        "mode": decision.get("mode"),
        "ensemble": {
            "composite_confidence": decision.get("composite_confidence"),
            "confidence_spread": decision.get("confidence_spread"),
        },
        "retrieval": {"rules": (decision.get("retrieved_context") or {}).get("rules") or []},
        "model_results": decision.get("models") or [],
        "grounding_basis": decision.get("grounding_basis"),
        "annotation_discrepancy": decision.get("annotation_discrepancy"),
        "rejected_candidates": decision.get("rejected_candidates") or [],
    }


def load_ungradable_counts(conn, note_ids):
    """How many contradiction/non_asserted_check decisions exist for these
    notes, so the report states its own coverage rather than letting the
    resolution-only numbers below imply they cover everything."""
    rows = conn.execute("""
        SELECT mode, count(*) FROM mollm_decisions
        WHERE is_test = TRUE AND mode != 'resolution'
          AND note_id IN ({})
        GROUP BY mode
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()
    return {mode: n for mode, n in rows}


def _json(v, default):
    if not v:
        return default
    try:
        return json.loads(v) if isinstance(v, str) else v
    except (ValueError, TypeError):
        return default


def official_verdict(decision):
    """The ensemble's single verdict, or None if the models disagreed.

    Mirrors mollm_ensemble.route()'s own logic: route() only reads
    model_results[0]['verdict'] for its CONTRADICTED/INSUFFICIENT_EVIDENCE
    checks, and only AFTER confirming ensemble_agreement -- disagreement
    routes straight to HITL_REQUIRED via the first safety rule before any
    single verdict is treated as "the" answer. A disagreeing pair has no
    verdict to grade here for the same reason: there isn't one.
    """
    if not decision["ensemble_agreement"]:
        return None
    models = decision["models"]
    if not models:
        return None
    return models[0].get("verdict")


def grade(decision, gold_by_note, vocab):
    """Returns 'correct' | 'incorrect' | None (ungradable: no verdict, no
    candidates, or no overlapping gold span for this entity at all -- the
    last case just means Stage 2a extracted something outside the annotated
    set, not a Stage 3 failure of any kind)."""
    verdict = official_verdict(decision)
    if verdict is None:
        return None, "model_disagreement"

    candidates = decision["candidates"]
    if not candidates:
        return None, "no_candidates_recorded"

    gold = gold_by_note.get(decision["note_id"], [])
    overlapping_gold = [
        g for g in gold
        if overlaps(decision["orig_start"], decision["orig_end"], g["start"], g["end"])
    ]
    if not overlapping_gold:
        return None, "no_overlapping_gold_span"

    gold_concept_ids = {g["concept_id"] for g in overlapping_gold}

    # Crosswalk every candidate to SNOMED once, same method
    # score_gold_recall.py uses for Stage 2b's own top-1 pick.
    candidate_snomed = [vocab.snomed_code_for_concept(c.get("omop_concept_id"))
                        for c in candidates]
    any_candidate_correct = any(code in gold_concept_ids for code in candidate_snomed if code)

    if verdict == "NONE_CORRECT":
        return ("correct" if not any_candidate_correct else "incorrect"), None

    m = RESOLVED_RE.match(verdict)
    if not m:
        # INSUFFICIENT_EVIDENCE or an out-of-vocabulary verdict -- not a
        # concept pick, nothing to check against gold.
        return None, f"verdict_not_a_resolution_pick:{verdict}"

    idx = int(m.group(1)) - 1
    if idx < 0 or idx >= len(candidate_snomed):
        return None, "candidate_index_out_of_range"

    chosen_code = candidate_snomed[idx]
    if chosen_code is None:
        return None, "chosen_candidate_uncrosswalked"

    return ("correct" if chosen_code in gold_concept_ids else "incorrect"), None


def compute_ece(graded, scheme="equal_width"):
    """Standard equal-width-bin ECE (Guo et al. 2017): sum over bins of
    (bin_size / N) * |bin_accuracy - bin_mean_confidence|. Also returns the
    per-bin table for a reliability diagram.

    `graded` is a list of (confidence, is_correct) with confidence in [0, 1]
    and confidence not None -- filter before calling.

    2026-08-13: THE ARITHMETIC MOVED TO evaluation/metrics.py; this is now a
    thin adapter that preserves the (ece, table) tuple every existing caller
    in this file and in stage_calibration.py already unpacks. The formula was
    written out three separate times across the repo (here, stage2b_cal_eval's
    tier3_ece(), fit_mollm_calibrator's _ece()) -- three correct copies with
    no mechanism to stay correct together, which is the same setup that
    produced the decoding_mode_mismatch bug. Numbers are unchanged: metrics.py's
    self-test asserts it reproduces this function's output, including the
    0.773 Stage 3 figure from the 2026-08-13 report.

    Prefer metrics.compute_ece_report() directly in NEW code -- it also
    returns MCE, Brier, base rate, n_nonempty_bins and the binning scheme,
    all of which matter for interpreting an ECE and none of which fit in this
    legacy two-tuple.
    """
    report = compute_ece_report(graded, n_bins=ECE_BINS, scheme=scheme)
    if report is None:
        return None, []
    return report["ece"], report["table"]


def threshold_sweep(graded, thresholds):
    """For each candidate threshold t: what fraction of the gradable sample
    would be auto-admitted at composite_confidence >= t, and what fraction of
    THOSE admitted decisions were actually correct. This is the curve
    AUTO_VALIDATE_THRESHOLD / MOLLM_RESOLVE_THRESHOLD should be read off of,
    once the sample is large enough that a single bin isn't noise -- see the
    module-level sample-size caveat.
    """
    n = len(graded)
    rows = []
    for t in thresholds:
        admitted = [(c, correct) for c, correct in graded if c >= t]
        rows.append({
            "threshold": t,
            "n_admitted": len(admitted),
            "coverage": round(len(admitted) / n, 4) if n else None,
            "precision_if_admitted": (round(sum(1 for _, ok in admitted if ok) / len(admitted), 4)
                                      if admitted else None),
        })
    return rows


def raw_label_breakdown(decisions_graded):
    """Per-model raw_confidence_label (HIGH/MEDIUM/LOW) vs empirical
    correctness, to check docs/Stage3_Open_Issues.md Issue 4's finding that
    self-reported confidence is a per-model constant rather than a
    per-record signal, at whatever sample size is currently available.
    Structure: {model_name: {label: {"n": int, "correct": int}}}.
    """
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: {"n": 0, "correct": 0}))
    for decision, outcome in decisions_graded:
        if outcome is None:
            continue
        for m in decision["models"]:
            name = m.get("model") or "?"
            label = m.get("raw_confidence_label") or "?"
            bucket = out[name][label]
            bucket["n"] += 1
            bucket["correct"] += int(outcome == "correct")
    return {model: dict(labels) for model, labels in out.items()}


def _print_subset(accuracy, ece, reliability_table, sweep, subset=None):
    """Shared printer for the guided and unguided_or_mixed report sections --
    factored out so the two subsets can never silently drift in what they
    display, the same anti-duplication reasoning src/normalization.py's
    _tier_queries() uses.

    `subset` is the full _subset_report() dict; when passed, the 2026-08-13
    additions (bootstrap CI, AUROC/AP, null-model baseline, equal-mass ECE)
    are printed too. Optional so any caller that still passes only the four
    legacy positional arguments keeps working unchanged.
    """
    if accuracy["n"] == 0:
        print("  n=0 -- nothing to report for this subset.")
        return

    print(f"\n--- Accuracy ---")
    ci = accuracy.get("ci_note_level")
    if ci:
        print(f"  {accuracy['correct']} / {accuracy['n']} = "
              f"{format_ci(ci)}   (95% CI, {ci['n_clusters']} notes resampled)")
    else:
        print(f"  {accuracy['correct']} / {accuracy['n']} = {accuracy['rate']*100:.2f}%")

    if subset:
        # All four interpretation blocks come from evaluation/metrics.py's
        # single printer, so this script, stage2a_cal_eval.py and
        # stage2b_cal_eval.py cannot describe the same quantities differently.
        print_interpretation_block(subset.get("_pairs") or [],
                                   accuracy_ci=accuracy.get("ci_note_level"))

    if ece is not None:
        print(f"\n--- Expected Calibration Error (composite_confidence, {ECE_BINS} bins) ---")
        print(f"  ECE = {ece}")
        print(f"  {'bin':<12} | {'n':>4} | {'mean conf':>10} | {'accuracy':>9}")
        for row in reliability_table:
            mc = f"{row['mean_confidence']:.4f}" if row["mean_confidence"] is not None else "-"
            ac = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "-"
            print(f"  {row['bin']:<12} | {row['n']:>4} | {mc:>10} | {ac:>9}")
    else:
        print("\n--- ECE: no decisions had a composite_confidence to bin "
              "(all disagreed or lacked logprobs) ---")

    print(f"\n--- Threshold sweep (coverage / precision if admitted) ---")
    print(f"  {'threshold':>9} | {'n_admitted':>10} | {'coverage':>8} | {'precision':>9}")
    for row in sweep:
        cov_s = f"{row['coverage']*100:.1f}%" if row["coverage"] is not None else "-"
        prec_s = f"{row['precision_if_admitted']*100:.1f}%" if row["precision_if_admitted"] is not None else "-"
        print(f"  {row['threshold']:>9.2f} | {row['n_admitted']:>10} | {cov_s:>8} | {prec_s:>9}")


def print_report(report):
    print("=" * 78)
    print("STAGE 3 CALIBRATION — resolution-mode decisions only (see module docstring)")
    print("=" * 78)

    sp = report.get("split_provenance")
    if sp:
        print(f"\nsplit: {sp['split']}   "
              f"(split file sha256 {str(sp.get('split_file_sha256'))[:12]}...)")
        if sp["split"] in ("explicit_note_ids", "all"):
            print("  !! not a reportable split -- development measurement only")

    cov = report["coverage"]
    print(f"\nResolution-mode decisions found: {cov['total_resolution_decisions']}")
    print(f"  gradable (verdict + candidates + overlapping gold span): "
          f"{cov['gradable']}")
    print(f"  ungradable, by reason:")
    for reason, n in cov["ungradable_reasons"].items():
        print(f"    {reason}: {n}")
    print(f"\nUngraded entirely (contradiction / non_asserted_check modes -- no gold "
          f"label exists for these, needs the Stage 5 human re-audit sample):")
    for mode, n in cov["ungraded_modes"].items():
        print(f"    {mode}: {n}")

    print(f"\nDecoding purity (all resolution decisions, gradable or not):")
    for purity, n in cov["decoding_purity_all_resolution_decisions"].items():
        print(f"    {purity}: {n}")
    print(f"Decoding purity (gradable only -- this is what feeds the numbers below):")
    for purity, n in cov["decoding_purity_gradable_only"].items():
        print(f"    {purity}: {n}")
    if cov["decoding_purity_gradable_only"].get("unguided_or_mixed", 0) > 0:
        print(f"  NOTE: {cov['decoding_purity_gradable_only']['unguided_or_mixed']} gradable "
              f"decision(s) used unguided or mixed decoding. They are reported separately under "
              f"'unguided_or_mixed' below and are NOT included in the ECE/threshold-sweep numbers "
              f"above that section -- do not average the two populations together.")

    if cov["gradable"] == 0:
        print("\nNo gradable resolution-mode decisions -- nothing further to report. "
              "Run scripts/test_stage3_live.py --store on a note with LOW-tier, "
              "multi-candidate entities that also has gold annotations.")
        return

    print(f"\n{'=' * 78}\nGUIDED-DECODING SUBSET -- the numbers to actually set thresholds from\n{'=' * 78}")
    _print_subset(report["accuracy"], report["ece"], report["reliability_table"],
                 report["threshold_sweep"], subset=report)

    other = report["unguided_or_mixed"]
    print(f"\n{'=' * 78}\nUNGUIDED-OR-MIXED SUBSET -- reported for visibility only, do not "
          f"use for threshold-setting\n{'=' * 78}")
    if other["accuracy"]["n"] == 0:
        print("  (none -- every gradable decision used guided decoding)")
    else:
        _print_subset(other["accuracy"], other["ece"], other["reliability_table"],
                     other["threshold_sweep"], subset=other)

    print(f"\n  current AUTO_VALIDATE_THRESHOLD=0.85, MOLLM_RESOLVE_THRESHOLD=0.60 "
          f"(src/mollm_ensemble.py) -- read the GUIDED subset's threshold-sweep table above "
          f"at those two rows first.")

    print(f"\n--- Self-reported confidence (raw_confidence_label) vs accuracy, per model ---")
    for model, labels in report["raw_label_breakdown"].items():
        print(f"  {model}:")
        for label in ("HIGH", "MEDIUM", "LOW"):
            b = labels.get(label)
            if not b or b["n"] == 0:
                continue
            rate = b["correct"] / b["n"]
            print(f"    {label:<7}: {b['correct']}/{b['n']} correct = {rate*100:.1f}%")

    print(f"\nSAMPLE SIZE CAVEAT: n={report['accuracy']['n']} gradable decisions. "
          f"See module docstring -- this is for methodology validation and re-running "
          f"as the corpus grows, not a production threshold fit.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None,
                     help="Comma-separated note_ids to evaluate. Default: every "
                          "note_id with is_test=TRUE resolution-mode rows in "
                          "mollm_decisions.")
    ap.add_argument("--gold", default=None, help="path to train_annotations.csv")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=None, help="write full JSON report here")
    ap.add_argument("--emit-training-data", default=None,
                     help="write MoLLM-Cal training data (JSONL: mollm_call_id, "
                          "features, feature_names, label) for every GUIDED-decoding "
                          "gradable decision to this path. label is 1 (verdict matched "
                          "gold) / 0 (did not) -- the same grading this script already "
                          "does for the accuracy/ECE numbers, just also written out "
                          "per-decision instead of only aggregated. See "
                          "src/mollm_calibrator.py's MoLLMCalibrator.fit().")
    # 2026-08-13 (report S4.3): --split, defaulting to `val`. Previously this
    # script defaulted to "every note_id in mollm_decisions", which is how the
    # ECE numbers came to be computed over a pool overlapping the notes used
    # to derive the fixes being measured.
    add_split_args(ap, default="val")
    args = ap.parse_args()

    gold_path = args.gold or _first_existing(GOLD_CANDIDATES, "gold annotations CSV")
    conn = duckdb.connect(args.db, read_only=True)
    try:
        note_ids, split_prov = resolve_note_ids(
            args, conn=conn, table="mollm_decisions",
            where="is_test = TRUE AND mode = 'resolution'")
        assert_no_contamination(conn)
        if not note_ids:
            raise SystemExit(
                f"No is_test=TRUE resolution-mode rows in mollm_decisions for "
                f"split '{split_prov['split']}'.\n"
                "Either run Stage 3 over that split "
                "(scripts/run_stage3_batch.py), or pass --split all to score "
                "whatever is present\n(a development measurement, not a "
                "reportable one).")

        print(f"gold:  {gold_path}")
        print(f"db:    {args.db}")
        print(f"notes: {note_ids}")

        gold_rows = load_gold(gold_path, note_ids)
        gold_by_note = collections.defaultdict(list)
        for g in gold_rows:
            gold_by_note[g["note_id"]].append(g)

        decisions = load_gradable_decisions(conn, note_ids)
        ungraded_modes = load_ungradable_counts(conn, note_ids)

        vocab = VocabularyRetriever(conn)
        ungradable_reasons = collections.Counter()
        decisions_graded = []
        for d in decisions:
            outcome, reason = grade(d, gold_by_note, vocab)
            if outcome is None:
                ungradable_reasons[reason] += 1
            decisions_graded.append((d, outcome))

        gradable = [(d, o) for d, o in decisions_graded if o is not None]

        # 2026-08-11 SPLIT BY DECODING PURITY. See _decoding_purity()'s
        # docstring for why this split exists. The GUIDED subset is the
        # primary report -- it's the population AUTO_VALIDATE_THRESHOLD and
        # MOLLM_RESOLVE_THRESHOLD will actually be applied against, since
        # guided decoding is the intended production path. The other subset
        # is reported alongside it, not discarded, so a reader can see the
        # excluded population's size rather than trusting a number with an
        # invisible gap.
        purity_counts = collections.Counter(d["decoding_purity"] for d in decisions)
        gradable_purity_counts = collections.Counter(d["decoding_purity"] for d, _ in gradable)

        def _subset_report(pairs):
            confs = [(d["composite_confidence"], o == "correct")
                    for d, o in pairs if d["composite_confidence"] is not None]
            ece, reliability_table = compute_ece(confs)
            n_correct = sum(1 for _, o in pairs if o == "correct")

            # 2026-08-13 additions (report P0.2 / P0.3). None of these change
            # the numbers above; they make them interpretable.
            #
            #  * accuracy_ci -- note-level bootstrap. 14.29% on n=140 was
            #    previously reported as a bare point estimate; the interval is
            #    what says whether it is distinguishable from anything else.
            #    Clusters are NOTES because entities within a note are not
            #    independent (docs/Evaluation_Criteria.md requires note-level
            #    resampling for exactly this reason).
            #  * auroc / average_precision -- can composite_confidence RANK?
            #    ECE cannot answer that, and the report's own finding (the
            #    higher-confidence bin being LESS accurate) is a ranking
            #    failure, not a calibration one. AUROC < 0.5 states it plainly.
            #  * null_model -- what predicting the base rate everywhere scores.
            #    Printed next to the real numbers so a base-rate-shaped result
            #    is obvious rather than needing a footnote.
            #  * ece_equal_mass -- the same data binned by quantile. With
            #    128/140 decisions inside one equal-width bin, the headline
            #    0.773 rests on 2 non-empty bins; this is the version that
            #    does not.
            per_note = group_by_note(
                [{"note_id": d["note_id"], "correct": (o == "correct")}
                 for d, o in pairs])

            def _acc(rows):
                return _accuracy([r["correct"] for r in rows])

            return {
                # Raw (confidence, correct) pairs, kept so print_report() can
                # re-derive the interpretation block without recomputing the
                # grading. Stripped before the JSON is written -- a report file
                # should carry conclusions, not a copy of the input.
                "_pairs": confs,
                "accuracy": {
                    "correct": n_correct, "n": len(pairs),
                    "rate": (n_correct / len(pairs)) if pairs else None,
                    "ci_note_level": bootstrap_ci(per_note, _acc),
                },
                "ece": ece,
                "ece_equal_mass": compute_ece_report(confs, scheme="equal_mass"),
                "ece_full": compute_ece_report(confs, scheme="equal_width"),
                "discrimination": {
                    "auroc": auroc(confs),
                    "average_precision": average_precision(confs),
                },
                "null_model": null_model_report(confs),
                "reliability_table": reliability_table,
                "threshold_sweep": threshold_sweep(confs, thresholds),
            }

        thresholds = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05..0.95
        gradable_guided = [(d, o) for d, o in gradable if d["decoding_purity"] == "guided"]
        gradable_other = [(d, o) for d, o in gradable if d["decoding_purity"] != "guided"]

        guided_report = _subset_report(gradable_guided)

        if args.emit_training_data:
            # docs/MoLLM_Redesign_Proposal.md S7 step 2 / S4.1 -- MoLLM-Cal's
            # training data. GUIDED-decoding gradable decisions only, same
            # population guided_report's ECE/threshold-sweep numbers are
            # computed from (S "mixing guided/unguided is exactly the mistake
            # verdict_schema()'s docstring names"). label is 1/0 from the SAME
            # grade() call already used for accuracy above -- not re-derived,
            # so the training labels and the reported accuracy can never
            # silently disagree about what "correct" meant for a given row.
            n_written = 0
            with open(args.emit_training_data, "w") as f:
                for d, outcome in gradable_guided:
                    ctx = build_feature_ctx(d)
                    row = {
                        "mollm_call_id": d["mollm_call_id"],
                        # 2026-08-13 (docs/2026-08-13_Code_Improvement_Proposals.md
                        # P3.4 / P0.3). note_id is what lets the fitting script
                        # (a) record training_note_ids in the .pkl so
                        # MoLLMCalibrator.load()'s leakage refusal has something
                        # to check against, and (b) bootstrap at the NOTE level
                        # rather than the row level -- entities inside one
                        # discharge note are not independent draws, so a
                        # row-level CI on n=140 would be far too narrow.
                        # Emitted per row rather than as a file header because
                        # the fitting script filters rows and must not have to
                        # reconstruct which notes survived.
                        "note_id": d.get("note_id"),
                        "split": split_prov.get("split"),
                        "split_file_sha256": split_prov.get("split_file_sha256"),
                        "features": extract_features(ctx),
                        "label": 1 if outcome == "correct" else 0,
                    }
                    f.write(json.dumps(row) + "\n")
                    n_written += 1
            print(f"\nwrote {n_written} training example(s) to {args.emit_training_data}")
            if n_written < 30:
                print(f"  NOTE: MoLLMCalibrator.fit()'s default min_examples=30 will "
                      f"refuse this file as-is (see src/mollm_calibrator.py) -- this is "
                      f"expected at the current sample size, not a bug in this script.")

        report = {
            # Which split produced these numbers, and the SHA256 of the split
            # file that defined it. A saved report outlives the terminal it was
            # printed in; without this, a JSON in the repo six months from now
            # cannot say whether it came from the validation slice or from
            # everything in the database.
            "split_provenance": split_prov,
            "coverage": {
                "total_resolution_decisions": len(decisions),
                "gradable": len(gradable),
                "ungradable_reasons": dict(ungradable_reasons),
                "ungraded_modes": ungraded_modes,
                "decoding_purity_all_resolution_decisions": dict(purity_counts),
                "decoding_purity_gradable_only": dict(gradable_purity_counts),
            },
            # Top-level accuracy/ece/reliability_table/threshold_sweep are
            # the GUIDED-ONLY numbers -- the ones to actually read thresholds
            # off of. See guided_report's construction above.
            "accuracy": guided_report["accuracy"],
            "_pairs": guided_report["_pairs"],
            "ece": guided_report["ece"],
            "ece_equal_mass": guided_report["ece_equal_mass"],
            "ece_full": guided_report["ece_full"],
            "discrimination": guided_report["discrimination"],
            "null_model": guided_report["null_model"],
            "reliability_table": guided_report["reliability_table"],
            "threshold_sweep": guided_report["threshold_sweep"],
            # Reported for transparency, never used for threshold-setting --
            # mixing this back into the numbers above is exactly the mistake
            # this patch exists to prevent.
            "unguided_or_mixed": _subset_report(gradable_other),
            "raw_label_breakdown": raw_label_breakdown(decisions_graded),
        }
    finally:
        conn.close()

    print_report(report)

    if args.out:
        # _pairs is the raw input, kept in memory only for the printer above.
        for sub in (report, report.get("unguided_or_mixed") or {}):
            sub.pop("_pairs", None)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
