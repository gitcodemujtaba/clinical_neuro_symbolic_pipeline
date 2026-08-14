"""
scripts/fit_mollm_calibrator.py — fits MoLLMCalibrator from evaluation/cal_eval.py's
--emit-training-data output and saves it in the exact format
src.mollm_calibrator.MoLLMCalibrator.load() expects.

WHY THIS SCRIPT EXISTS, NOT JUST MoLLMCalibrator.fit() DIRECTLY.
MoLLMCalibrator.fit(feature_ctxs, labels) takes RAW feature_ctx dicts and calls
extract_features() on each one internally. But evaluation/cal_eval.py's
--emit-training-data already calls extract_features() itself before writing each
line -- the JSONL's "features" field is the FINAL 13-element vector, not a
feature_ctx dict. This script therefore calls MoLLMCalibrator.fit_vectors(),
which is the same code path fit() itself delegates to after extraction.

  2026-08-13 (docs/2026-08-13_Code_Improvement_Proposals.md P3). This script
  USED TO hand-roll `LogisticRegression(max_iter=1000, C=1.0)` with a comment
  promising those were "the same hyperparameters fit() itself uses". That
  promise broke silently when P3.3 added class_weight="balanced" to fit() and
  not here -- the saved production model was a different estimator from the
  one src/mollm_calibrator.py documents, and neither file could detect it.
  fit_vectors() and _new_estimator() now make a second construction site
  impossible. Do not reintroduce one.

WHAT CHANGED ON 2026-08-13 (P3), AND WHY

  P3.1  ACCURACY IS NO LONGER THE HEADLINE. On a 14.3% positive rate,
        "85.71% accurate" is what you get by predicting 0 for everything --
        which is exactly what the previous version of this script reported,
        alongside a held-out ECE of 0.0299 that looked like a large win over
        the raw 0.773 baseline. Both numbers were artifacts of the class
        balance. This version reports AUROC and average precision first --
        the two metrics that measure DISCRIMINATION (can the model rank a
        correct decision above an incorrect one) rather than agreement with
        the majority class -- and prints the null model's numbers in the same
        table so a base-rate predictor is self-evidently a base-rate
        predictor instead of requiring a footnote.

  P3.2  STRATIFIED K-FOLD, NOT ONE 80/20 SHUFFLE. The old held_out_check()
        justified an unstratified split with "n=140 is too small for
        stratification to matter much". The opposite is true: with 20
        positives, a random 28-row test fold expects ~4 positives and can
        easily draw 1. That is how n_test=28 produced an accuracy exactly
        equal to the base rate. StratifiedKFold holds the positive rate fixed
        in every fold and uses all 140 rows for evaluation instead of 28.

  P3.4  TRAINING PROVENANCE IS RECORDED. save() now receives
        training_note_ids/training_split/code_version, which is what makes
        MoLLMCalibrator.load()'s leakage refusal functional -- without them
        load() can only report "cannot be checked". This requires note_id on
        each training row; cal_eval.py --emit-training-data emits it as of the
        same date. Older JSONL without note_id still fits, and says loudly
        that the resulting .pkl cannot be leakage-checked.

  P0.2  ECE ARITHMETIC COMES FROM evaluation/metrics.py. This script used to
        carry its own _ece() -- the third of three copies of one formula.

  P0.3  NOTE-LEVEL BOOTSTRAP CIs. Entities within a discharge note are not
        independent, so a row-level interval on n=140 would be far too narrow.
        Resampling notes is what docs/Evaluation_Criteria.md specifies.

HONEST ABOUT SAMPLE SIZE. n=140 clears fit()'s min_examples=30 floor, but that
floor is a placeholder, not a calibrated minimum. The quantity that actually
binds is POSITIVES: 13 features against 20 positives is ~1.5 per parameter.
fit_vectors() warns about this; the CV block below is the evidence that should
decide whether the model is used at all. A wide AUROC interval spanning 0.5
means "this model does not discriminate", regardless of how good its ECE looks.

Run:
  python3 evaluation/cal_eval.py --split val --emit-training-data mollm_cal_train.jsonl
  python3 scripts/fit_mollm_calibrator.py --train-data mollm_cal_train.jsonl \
      --out models/mollm_calibrator_v1.pkl
"""

import argparse
import collections
import json
import os
import sys

# Derived from this file's location rather than hardcoded to
# /home/ec2-user/... (P3.5) -- the old absolute path made the script
# unrunnable anywhere but the EC2 box, including in the stub tests that are
# supposed to verify it before it touches real data.
PROJECT_DIR = os.environ.get(
    "CNSP_PROJECT_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from src.mollm_calibrator import (  # noqa: E402
    MoLLMCalibrator, FEATURE_NAMES, FEATURE_SET_VERSION,
)
from src.provenance import get_code_version  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    auroc, average_precision, bootstrap_ci, compute_ece_report, format_ci,
    null_model_report,
)


def load_training_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _pos_index(model):
    """Column of predict_proba corresponding to class 1 ("correct").

    scikit-learn sorts classes_, so this is normally 1 -- but "normally" is
    not a guarantee worth putting a routing decision on, and the same
    explicit lookup is used in MoLLMCalibrator.score().
    """
    classes = list(model.classes_)
    return classes.index(1) if 1 in classes else (1 if len(classes) > 1 else 0)


def cross_validated_scores(X, y, n_splits=5, seed=13):
    """Out-of-fold P(correct) for EVERY row, via stratified k-fold.

    Returns (pairs, n_splits_used) where pairs is [(prob, bool(label)), ...]
    aligned to X's original order, or (None, reason) when the data cannot
    support k folds.

    Out-of-fold rather than a single held-out slice: every row gets a
    prediction from a model that did not see it, so the evaluation set is all
    140 rows instead of 28, and the resulting AUROC/ECE have far less
    split-luck variance. Each fold's estimator comes from the same
    _new_estimator() the production model uses (via fit_vectors), so the CV
    numbers describe the model actually being saved.
    """
    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError:
        return None, "scikit-learn not available"

    n_pos = sum(1 for v in y if v)
    n_neg = len(y) - n_pos
    if min(n_pos, n_neg) < 2:
        return None, f"only {n_pos} positive / {n_neg} negative example(s)"
    # Cannot have more folds than members of the rarest class.
    n_splits = min(n_splits, n_pos, n_neg)
    if n_splits < 2:
        return None, "too few examples in the minority class for 2 folds"

    probs = [None] * len(X)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        fold = MoLLMCalibrator()
        # min_examples=2: the FLOOR belongs to the production fit, not to a
        # fold of it. Applying 30 here would silently skip CV on exactly the
        # small samples where CV matters most.
        fold.fit_vectors([X[i] for i in train_idx], [y[i] for i in train_idx],
                         min_examples=2)
        pos = _pos_index(fold.model)
        for i, row in zip(test_idx, fold.model.predict_proba([X[i] for i in test_idx])):
            probs[i] = float(row[pos])

    pairs = [(p, bool(label)) for p, label in zip(probs, y) if p is not None]
    return pairs, n_splits


def _print_cv_block(pairs, note_ids, n_splits):
    """The discrimination-first report. Order matters: AUROC before ECE,
    model beside null, because the failure this block exists to expose is a
    base-rate predictor with a beautiful ECE.
    """
    print(f"\n--- cross-validated performance ({n_splits}-fold stratified, "
          f"out-of-fold predictions for all {len(pairs)} rows) ---")

    model_auroc = auroc(pairs)
    model_ap = average_precision(pairs)
    report = compute_ece_report(pairs, scheme="equal_mass")
    null = null_model_report(pairs)

    # Note-level bootstrap (P0.3). Falls back to row-level with an explicit
    # warning when note_ids are absent, rather than silently reporting an
    # interval that assumes independence it does not have.
    if note_ids and len(set(note_ids)) > 1:
        clusters = collections.defaultdict(list)
        for pair, nid in zip(pairs, note_ids):
            clusters[nid].append(pair)
        units = list(clusters.values())
        unit_label = f"{len(units)} notes"
    else:
        units = [[p] for p in pairs]
        unit_label = f"{len(units)} rows (NO note_id: interval is too narrow)"

    auroc_ci = bootstrap_ci(units, lambda flat: auroc(flat))
    ap_ci = bootstrap_ci(units, lambda flat: average_precision(flat))

    print(f"  resampling unit: {unit_label}")
    print()
    print(f"  {'metric':<34}{'model':>10}{'null (base rate)':>18}")
    print(f"  {'-' * 62}")
    print(f"  {'AUROC (0.5 = no discrimination)':<34}"
          f"{_fmt(model_auroc):>10}{_fmt(null.get('auroc')):>18}")
    print(f"  {'  95% CI (note-level bootstrap)':<34}"
          f"{format_ci(auroc_ci, pct=False)}")
    print(f"  {'average precision':<34}"
          f"{_fmt(model_ap):>10}{_fmt(null.get('average_precision')):>18}")
    print(f"  {'  95% CI (note-level bootstrap)':<34}"
          f"{format_ci(ap_ci, pct=False)}")
    print(f"  {'ECE (equal-mass bins)':<34}"
          f"{_fmt(report.get('ece')):>10}{_fmt(null.get('ece')):>18}")
    print(f"  {'Brier score':<34}"
          f"{_fmt(report.get('brier')):>10}{_fmt(null.get('brier')):>18}")
    print(f"  {'accuracy @0.5 (NOT the headline)':<34}"
          f"{_fmt(_accuracy_at(pairs, 0.5)):>10}"
          f"{_fmt(null.get('majority_class_accuracy')):>18}")
    print(f"  {'base rate (positives / n)':<34}{_fmt(null.get('base_rate')):>10}")

    _print_verdict(model_auroc, auroc_ci, report, null)
    return {
        "auroc": model_auroc, "auroc_ci": auroc_ci,
        "average_precision": model_ap, "average_precision_ci": ap_ci,
        "ece_equal_mass": report.get("ece"), "brier": report.get("brier"),
        "null_model": null, "n_splits": n_splits, "n_scored": len(pairs),
    }


def _print_verdict(model_auroc, auroc_ci, report, null):
    """Says out loud what the numbers mean, so the interpretation does not
    depend on the reader remembering that ECE and accuracy are both
    uninformative at this class balance.
    """
    print("\n  reading:")
    if model_auroc is None:
        print("    AUROC could not be computed (needs both classes present).")
        return
    ci_spans_chance = bool(auroc_ci) and auroc_ci["lo"] <= 0.5 <= auroc_ci["hi"]
    if ci_spans_chance:
        print("    ** The AUROC confidence interval INCLUDES 0.5. On this sample "
              "this model\n       is not distinguishable from one that cannot rank "
              "correct above incorrect.")
        print("       A low ECE here means 'it predicts the base rate and the base "
              "rate is right\n       on average', which is calibration without "
              "discrimination -- it will not\n       improve routing, only make the "
              "confidence numbers look sane.")
    elif model_auroc > 0.5:
        print(f"    AUROC {model_auroc:.3f} with the interval clear of 0.5: the "
              f"model does carry\n    real ranking signal on this sample.")
    else:
        print(f"    AUROC {model_auroc:.3f} is BELOW chance with the interval clear "
              f"of 0.5 --\n    the ranking is inverted, which is a bug or a leak, "
              f"not a usable model.")

    null_ece = null.get("ece")
    model_ece = report.get("ece")
    if null_ece is not None and model_ece is not None and null_ece <= model_ece:
        print(f"    The null model's ECE ({null_ece}) is no worse than this model's "
              f"({model_ece}).\n    Any ECE improvement claimed against the raw "
              f"composite_confidence baseline (0.773)\n    is therefore attributable "
              f"to predicting the base rate, not to the features.")


def _accuracy_at(pairs, threshold):
    if not pairs:
        return None
    hits = sum(1 for p, ok in pairs if (p >= threshold) == ok)
    return round(hits / len(pairs), 4)


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-data", required=True,
                     help="JSONL from evaluation/cal_eval.py --emit-training-data")
    ap.add_argument("--out", required=True, help="path to save the fitted calibrator (.pkl)")
    ap.add_argument("--min-examples", type=int, default=30,
                     help="same floor as MoLLMCalibrator.fit() (default 30)")
    ap.add_argument("--folds", type=int, default=5,
                     help="stratified CV folds (default 5); capped at the size "
                          "of the minority class")
    ap.add_argument("--report-json", default=None,
                     help="also write the CV metrics to this path")
    args = ap.parse_args()

    rows = load_training_rows(args.train_data)
    print(f"loaded {len(rows)} training example(s) from {args.train_data}")

    if len(rows) < args.min_examples:
        raise SystemExit(
            f"only {len(rows)} example(s), refusing to fit below "
            f"--min-examples={args.min_examples} -- same floor MoLLMCalibrator.fit() uses.")

    for r in rows:
        if len(r["features"]) != len(FEATURE_NAMES):
            raise SystemExit(
                f"mollm_call_id={r.get('mollm_call_id')}: feature vector has "
                f"{len(r['features'])} values, expected {len(FEATURE_NAMES)} "
                f"(FEATURE_SET_VERSION={FEATURE_SET_VERSION}). Training data was "
                f"likely emitted by a different code version -- re-run "
                f"evaluation/cal_eval.py --emit-training-data first.")

    X = [r["features"] for r in rows]
    y = [r["label"] for r in rows]
    note_ids = [r.get("note_id") for r in rows]

    # Split/provenance carried through from cal_eval.py (P3.4).
    training_note_ids = sorted({n for n in note_ids if n})
    splits_seen = sorted({r.get("split") for r in rows if r.get("split")})
    training_split = splits_seen[0] if len(splits_seen) == 1 else (
        ",".join(splits_seen) if splits_seen else None)

    label_counts = collections.Counter(y)
    n_pos = label_counts.get(1, 0)
    print(f"label balance: {dict(label_counts)} ({n_pos / len(y) * 100:.1f}% correct)")
    print(f"positives per feature: {n_pos / max(1, len(FEATURE_NAMES)):.1f} "
          f"({n_pos} positives, {len(FEATURE_NAMES)} features; >=5 is the "
          f"conventional floor)")

    if training_note_ids:
        print(f"training notes: {len(training_note_ids)} distinct "
              f"(split={training_split or 'unrecorded'})")
    else:
        print("!! training rows carry NO note_id -- this .pkl cannot be "
              "leakage-checked by\n   MoLLMCalibrator.load(). Re-emit training "
              "data with a current evaluation/cal_eval.py.")

    if len(set(y)) < 2:
        raise SystemExit(
            f"all {len(y)} labels are the same value ({y[0]}) -- a logistic "
            f"regression cannot be fit on a single class. Need both correct "
            f"and incorrect examples in the training data.")

    cv_pairs, cv_info = cross_validated_scores(X, y, n_splits=args.folds)
    cv_summary = None
    if cv_pairs is None:
        print(f"\n--- cross-validated performance ---\n  skipped: {cv_info}")
    else:
        cv_summary = _print_cv_block(cv_pairs, note_ids, cv_info)

    print("\n--- fitting production model on all data ---")
    calibrator = MoLLMCalibrator().fit_vectors(X, y, min_examples=args.min_examples)
    model = calibrator.model
    pos_idx = _pos_index(model)

    print(f"\n--- coefficients (feature -> weight, class order "
          f"{list(model.classes_)}) ---")
    coefs = model.coef_[pos_idx if model.coef_.shape[0] > 1 else 0]
    for name, coef in zip(FEATURE_NAMES, coefs):
        print(f"  {name:<38} {coef:+.4f}")
    print(f"  {'intercept':<38} "
          f"{model.intercept_[pos_idx if len(model.intercept_) > 1 else 0]:+.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    calibrator.save(args.out,
                    training_note_ids=training_note_ids,
                    training_split=training_split,
                    code_version=get_code_version())
    print(f"\nsaved calibrator ({len(X)} training examples, "
          f"feature_set_version={FEATURE_SET_VERSION}, "
          f"{len(training_note_ids)} training note_ids recorded) to {args.out}")

    if args.report_json and cv_summary is not None:
        with open(args.report_json, "w", encoding="utf-8") as fh:
            json.dump({
                "train_data": args.train_data,
                "out": args.out,
                "n_examples": len(X),
                "n_positive": n_pos,
                "training_split": training_split,
                "training_note_ids": training_note_ids,
                "code_version": get_code_version(),
                "cross_validation": cv_summary,
            }, fh, indent=2)
        print(f"wrote CV report to {args.report_json}")

    print("\nNEXT: scripts/run_stage3_batch.py --calibrator "
          f"{args.out}\n  load() will refuse this model on any note in its own "
          "training set (P3.4),\n  so a batch over the val split will decline it. "
          "That is the guard working,\n  not a failure -- score the TEST split, or "
          "refit on a disjoint slice.")


if __name__ == "__main__":
    main()
