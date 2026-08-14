"""
evaluation/stage2a_cal_eval.py — Stage 2a (GLiNER-BioMed extraction) confidence
calibration, the same question cal_eval.py asks of Stage 3 but one stage
earlier: is `extracted_entities.confidence` (GLiNER's own softmax score for
the span it extracted) a trustworthy probability, or just a number?

WHAT "CORRECT" MEANS HERE, AND WHY IT IS DELIBERATELY NOT THE SAME QUESTION
score_gold_recall.py ASKS. score_gold_recall.py's span_recall iterates GOLD
annotations and asks "was there ANY overlapping prediction" -- a per-gold-span
question, and the right one for measuring what Stage 2a missed. This script
needs the mirror-image, per-PREDICTION question -- "was this extracted span,
at this confidence, actually a real mention" -- because a calibration curve
needs exactly one (confidence, correct) pair per scored decision, and the
decision being scored is "GLiNER extracted this span at this confidence."
`span_correct` = the extracted span overlaps at least one gold annotation in
its note, irrespective of concept identity (that is Stage 2b/3's job, not
Stage 2a's).

GOLD SCOPE CAVEAT -- READ BEFORE TRUSTING A NUMBER FROM THIS SCRIPT.
The SNOMED CT Entity Linking Challenge's annotators tagged only Procedures,
Body Structures and Clinical Findings (score_gold_recall.py's
print_official_metrics() already documents this for the IoU metric). GLiNER's
CLINICAL_LABELS (src/entity_extraction.py) map onto that scope as:
    Condition, Symptom  -> Clinical Finding   (IN SCOPE)
    Anatomy              -> Body Structure     (IN SCOPE)
    Procedure             -> Procedure          (IN SCOPE)
    Medication            -> not annotated at all (RxNorm, not SNOMED)
    Lab Test               -> not annotated at all (confirmed empirically
                               this project: lab-suffix Stage 1 fixes showed
                               zero movement on score_gold_recall.py's
                               recall numbers for exactly this reason)
A Medication or Lab Test entity can NEVER show a gold overlap no matter how
correct the extraction is -- scoring it "incorrect" every single time would
poison the confidence-vs-correctness curve with a systematic floor that has
nothing to do with GLiNER's calibration. These two labels are therefore
excluded from the ECE population and reported separately, under the same
"report exclusions loudly, don't silently drop" policy cal_eval.py's module
docstring states for its own contradiction/non_asserted_check exclusion.

GOLD-LESS NOTE GUARD. Same reasoning as score_gold_recall.py main()'s
"GOLD-LESS NOTE GUARD" comment: a note_id with is_test=TRUE extracted_entities
rows but ZERO gold annotations (e.g. a note outside the 272-note annotated
set, or an old diagnostic run's leftover rows) would have every one of its
entities score span_correct=False regardless of extraction quality, since
gold_by_note.get() for that note is always []. Entities from such notes are
excluded from the ECE population and reported separately, not silently
folded in as "wrong."

Run:
  python3 evaluation/stage2a_cal_eval.py
  python3 evaluation/stage2a_cal_eval.py --note-ids 17751158-DS-19,19442119-DS-15 --out reports/stage2a_cal.json
"""

import argparse
import collections
import json
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

sys.path.insert(0, PROJECT_DIR)

from scripts.score_gold_recall import (  # noqa: E402
    load_gold, overlaps, _first_existing, GOLD_CANDIDATES,
)
from evaluation.cal_eval import compute_ece, threshold_sweep, ECE_BINS  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    accuracy as _accuracy, auroc, average_precision, bootstrap_ci,
    compute_ece_report, format_ci, group_by_note, null_model_report,
    print_interpretation_block,
)
from evaluation.splits import (  # noqa: E402
    add_split_args, assert_no_contamination, resolve_note_ids,
)

# See module docstring "GOLD SCOPE CAVEAT". Kept as an explicit allow-list
# (not a "everything except Medication" exclude-list) so a future new
# CLINICAL_LABEL added to src/entity_extraction.py without updating this
# script fails safe -- it is excluded and reported, not silently assumed
# in-scope.
IN_SCOPE_LABELS = {"Condition", "Symptom", "Anatomy", "Procedure"}


def load_extracted_entities(conn, note_ids):
    """Stage 2a output for the target notes: raw GLiNER confidence plus span.

    Reads extracted_entities DIRECTLY, not joined against normalized_entities
    -- this measures Stage 2a's own confidence, upstream of and independent
    from whatever Stage 2b does with the entity afterward. Same accepted-row
    filter score_gold_recall.py's load_predictions() applies (is_test=TRUE,
    excludes below_threshold / superseded_by_split / superseded_by_growth
    rows), for the same reason: those are rows Stage 2a itself withdrew or
    replaced, not live extraction decisions to grade.

    NOTE: the column is `confidence`, not `gliner_confidence` -- the latter
    is only an in-memory rename src/mollm_ensemble.py applies when it reads
    this same column for Stage 3 prompts (build_prompt() candidate section);
    the DDL in src/entity_extraction.py's ensure_extracted_entities_table()
    names it `confidence`.
    """
    rows = conn.execute("""
        SELECT note_id, orig_start, orig_end, entity_label, original_text,
               confidence, entity_id
        FROM extracted_entities
        WHERE is_test = TRUE
          AND note_id IN ({})
          AND confidence IS NOT NULL
          AND (below_threshold IS NULL OR below_threshold = FALSE)
          AND (superseded_by_split IS NULL OR superseded_by_split = FALSE)
          AND (superseded_by_growth IS NULL OR superseded_by_growth = FALSE)
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    cols = ["note_id", "orig_start", "orig_end", "entity_label", "original_text",
            "confidence", "entity_id"]
    return [dict(zip(cols, r)) for r in rows]


def grade(entities, gold_by_note):
    """Attaches span_correct (bool) to each entity dict: does its span
    overlap >=1 gold annotation in its note, regardless of concept identity.
    Mutates nothing -- returns new dicts."""
    graded = []
    for e in entities:
        gold = gold_by_note.get(e["note_id"], [])
        hit = any(overlaps(e["orig_start"], e["orig_end"], g["start"], g["end"])
                  for g in gold)
        graded.append({**e, "span_correct": hit})
    return graded


def label_breakdown(graded):
    """Per-GLiNER-label accuracy (span_correct rate) and mean confidence, for
    the in-scope population only -- lets a reader see whether, say, Anatomy
    is systematically better-calibrated than Condition, the same kind of
    per-model breakdown cal_eval.py's raw_label_breakdown() gives for Stage 3.
    """
    out = collections.defaultdict(lambda: {"n": 0, "correct": 0, "conf_sum": 0.0})
    for e in graded:
        b = out[e["entity_label"]]
        b["n"] += 1
        b["correct"] += int(e["span_correct"])
        b["conf_sum"] += e["confidence"]
    result = {}
    for label, b in out.items():
        result[label] = {
            "n": b["n"],
            "correct": b["correct"],
            "accuracy": round(b["correct"] / b["n"], 4) if b["n"] else None,
            "mean_confidence": round(b["conf_sum"] / b["n"], 4) if b["n"] else None,
        }
    return result


def print_report(report, note_ids):
    print("=" * 78)
    print("STAGE 2A CALIBRATION — GLiNER-BioMed extraction confidence vs gold overlap")
    print("=" * 78)

    cov = report["coverage"]
    print(f"\nnotes requested: {note_ids}")
    print(f"is_test=TRUE extracted_entities rows found: {cov['total_entities']}")
    print(f"  excluded, no gold annotations for this note_id: {cov['excluded_gold_less_note']}")
    print(f"  excluded, out-of-gold-scope label (Medication / Lab Test): "
          f"{cov['excluded_out_of_scope_label']}")
    print(f"  in ECE population: {cov['in_scope_n']}")
    if cov["excluded_gold_less_note_ids"]:
        print(f"  gold-less note_ids excluded: {sorted(cov['excluded_gold_less_note_ids'])}")

    if cov["in_scope_n"] == 0:
        print("\nNo in-scope, gold-coverable entities -- nothing further to report.")
        return

    print(f"\n--- Accuracy (span_correct rate, in-scope population) ---")
    acc = report["accuracy"]
    ci = report.get("accuracy_ci_note_level")
    if ci:
        print(f"  {acc['correct']} / {acc['n']} = {format_ci(ci)}"
              f"   (95% CI, {ci['n_clusters']} notes resampled)")
    else:
        print(f"  {acc['correct']} / {acc['n']} = {acc['rate']*100:.2f}%")

    # 2026-08-13: the same interpretation block every calibration script now
    # prints, from evaluation/metrics.py. Stage 2a is the one stage whose
    # signal the report found EMPIRICALLY SUPPORTED -- AUROC is what confirms
    # that holds up as a ranking claim and not only as a calibration one.
    print_interpretation_block(report.get("_pairs") or [], accuracy_ci=None)

    print(f"\n--- Expected Calibration Error (GLiNER `confidence`, {ECE_BINS} bins) ---")
    if report["ece"] is not None:
        print(f"  ECE = {report['ece']}")
        print(f"  {'bin':<12} | {'n':>5} | {'mean conf':>10} | {'accuracy':>9}")
        for row in report["reliability_table"]:
            mc = f"{row['mean_confidence']:.4f}" if row["mean_confidence"] is not None else "-"
            ac = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "-"
            print(f"  {row['bin']:<12} | {row['n']:>5} | {mc:>10} | {ac:>9}")
    else:
        print("  no gradable entities.")

    print(f"\n--- Threshold sweep (coverage / precision if admitted at >= t) ---")
    print(f"  {'threshold':>9} | {'n_admitted':>10} | {'coverage':>8} | {'precision':>9}")
    for row in report["threshold_sweep"]:
        cov_s = f"{row['coverage']*100:.1f}%" if row["coverage"] is not None else "-"
        prec_s = (f"{row['precision_if_admitted']*100:.1f}%"
                  if row["precision_if_admitted"] is not None else "-")
        print(f"  {row['threshold']:>9.2f} | {row['n_admitted']:>10} | {cov_s:>8} | {prec_s:>9}")

    print(f"\n--- Per-label breakdown (in-scope population) ---")
    print(f"  {'label':<12} | {'n':>5} | {'accuracy':>9} | {'mean conf':>10}")
    for label, b in sorted(report["label_breakdown"].items()):
        ac = f"{b['accuracy']*100:.1f}%" if b["accuracy"] is not None else "-"
        mc = f"{b['mean_confidence']:.4f}" if b["mean_confidence"] is not None else "-"
        print(f"  {label:<12} | {b['n']:>5} | {ac:>9} | {mc:>10}")

    if cov["out_of_scope_n"]:
        print(f"\n--- Out-of-gold-scope entities (Medication / Lab Test, excluded above) ---")
        for label, b in sorted(report["out_of_scope_breakdown"].items()):
            print(f"  {label:<12}: n={b['n']}, mean_confidence={b['mean_confidence']}")
        print("  These CANNOT show a gold overlap by construction (gold has zero "
              "lab-panel/medication annotations) -- reported for visibility only, "
              "never as 'incorrect'.")

    print(f"\nSAMPLE SIZE CAVEAT: n={acc['n']} gradable entities -- same caveat "
          f"score_gold_recall.py and cal_eval.py state for their own numbers: this "
          f"is for methodology validation and re-running as the corpus grows, not "
          f"a production threshold fit off a small sample.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None,
                     help="Comma-separated note_ids to evaluate. Default: every "
                          "note_id with is_test=TRUE rows in extracted_entities.")
    ap.add_argument("--gold", default=None, help="path to train_annotations.csv")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=None, help="write full JSON report here")
    # 2026-08-13 (report S4.3): default to the validation slice rather than
    # "whatever is in the database". See evaluation/splits.py.
    add_split_args(ap, default="val")
    args = ap.parse_args()

    gold_path = args.gold or _first_existing(GOLD_CANDIDATES, "gold annotations CSV")
    conn = duckdb.connect(args.db, read_only=True)
    try:
        note_ids, split_prov = resolve_note_ids(
            args, conn=conn, table="extracted_entities", where="is_test = TRUE")
        assert_no_contamination(conn)
        if not note_ids:
            raise SystemExit(
                f"No is_test=TRUE extracted_entities rows for split "
                f"'{split_prov['split']}'.\nRun scripts/test_pipeline_e2e.py "
                f"over that split, or pass --split all (development only).")

        print(f"gold:  {gold_path}")
        print(f"db:    {args.db}")
        print(f"notes: {note_ids}")

        gold_rows = load_gold(gold_path, note_ids)
        gold_by_note = collections.defaultdict(list)
        for g in gold_rows:
            gold_by_note[g["note_id"]].append(g)
        gold_note_ids = {g["note_id"] for g in gold_rows}

        entities = load_extracted_entities(conn, note_ids)

        excluded_gold_less = [e for e in entities if e["note_id"] not in gold_note_ids]
        scoreable = [e for e in entities if e["note_id"] in gold_note_ids]

        out_of_scope = [e for e in scoreable if e["entity_label"] not in IN_SCOPE_LABELS]
        in_scope = [e for e in scoreable if e["entity_label"] in IN_SCOPE_LABELS]

        graded = grade(in_scope, gold_by_note)
        out_of_scope_graded = grade(out_of_scope, gold_by_note)  # for visibility only

        confs = [(e["confidence"], e["span_correct"]) for e in graded]
        ece, reliability_table = compute_ece(confs)

        # 2026-08-13 (report P0.2/P0.3). Stage 2a's headline 89.87% was a bare
        # point estimate over 1944 entities drawn from only 31 notes -- the
        # effective sample size is closer to 31 than 1944, since entities in
        # one discharge note share a patient, a template and an author.
        # Resampling notes says so; resampling entities would have produced a
        # confidently narrow, wrong interval.
        per_note = group_by_note([{"note_id": e["note_id"],
                                   "correct": e["span_correct"]} for e in graded])
        accuracy_ci = bootstrap_ci(per_note, lambda rows: _accuracy(
            [r["correct"] for r in rows]))

        n_correct = sum(1 for _, ok in confs if ok)
        thresholds = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05..0.95

        report = {
            "split_provenance": split_prov,
            # raw pairs, printer-only; stripped before --out is written
            "_pairs": confs,
            "accuracy_ci_note_level": accuracy_ci,
            "discrimination": {
                "auroc": auroc(confs),
                "average_precision": average_precision(confs),
            },
            "null_model": null_model_report(confs),
            "ece_equal_mass": compute_ece_report(confs, scheme="equal_mass"),
            "ece_full": compute_ece_report(confs, scheme="equal_width"),
            "coverage": {
                "total_entities": len(entities),
                "excluded_gold_less_note": len(excluded_gold_less),
                "excluded_gold_less_note_ids": sorted({e["note_id"] for e in excluded_gold_less}),
                "excluded_out_of_scope_label": len(out_of_scope),
                "out_of_scope_n": len(out_of_scope),
                "in_scope_n": len(in_scope),
            },
            "accuracy": {
                "correct": n_correct, "n": len(confs),
                "rate": (n_correct / len(confs)) if confs else None,
            },
            "ece": ece,
            "reliability_table": reliability_table,
            "threshold_sweep": threshold_sweep(confs, thresholds),
            "label_breakdown": label_breakdown(graded),
            "out_of_scope_breakdown": label_breakdown(out_of_scope_graded),
        }
    finally:
        conn.close()

    print_report(report, note_ids)

    if args.out:
        report.pop("_pairs", None)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
