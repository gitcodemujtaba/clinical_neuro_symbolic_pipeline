"""
tests/test_calibrator_fit.py — the 2026-08-13 P3 calibrator rework
(src/mollm_calibrator.py, scripts/fit_mollm_calibrator.py).

WHAT IS BEING PROTECTED.

  1. THE DRIFT BUG ITSELF. Before P3 there were two places that constructed a
     LogisticRegression: MoLLMCalibrator.fit() and the fitting script's
     hand-rolled copy, joined only by a comment promising they matched. P3.3
     added class_weight="balanced" to the first and not the second, so the
     model actually saved to models/mollm_calibrator_v1.pkl was a DIFFERENT
     estimator from the documented one, and no test could see it. There is now
     one construction site (_new_estimator) and fit() delegates to
     fit_vectors(). test_single_estimator_construction_site() greps the source
     to keep it that way -- a behavioural test cannot catch a second copy that
     happens to be configured identically TODAY.

  2. THE LEAKAGE GUARD IS ACTUALLY ARMED. load()'s refusal only works if
     save() was given training_note_ids. The old script called save(path) with
     no provenance, so the guard would have reported "cannot be checked"
     forever while looking like it was working.

  3. OUT-OF-FOLD COVERAGE. cross_validated_scores() must return a prediction
     for every row, each from a model that did not train on it -- that is the
     difference between evaluating on 140 rows and evaluating on 28.

NO SCIKIT-LEARN REQUIRED. sklearn is absent from this sandbox (and from any
box that has not run requirements.txt), so a fake linear_model /
model_selection pair is injected into sys.modules before importing the
calibrator. The fake is deliberately dumb -- it scores on feature 0 -- because
these tests are about plumbing (delegation, provenance, fold coverage), not
about whether logistic regression works.

Run:  python3 tests/test_calibrator_fit.py
"""

import ast
import os
import sys
import tempfile
import types

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import numpy as np  # noqa: E402


# ==========================================================================
# Fake scikit-learn
# ==========================================================================

class FakeLogisticRegression:
    """Deterministic stand-in. P(correct) is a monotone function of feature 0,
    so ranking metrics are predictable, and every constructor kwarg is
    recorded so a test can assert what _new_estimator() actually asked for.
    """

    def __init__(self, max_iter=100, C=1.0, class_weight=None):
        self.max_iter = max_iter
        self.C = C
        self.class_weight = class_weight
        self.classes_ = None
        self.coef_ = None
        self.intercept_ = None
        self.seen_rows = None

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        n_features = len(X[0]) if X else 1
        self.coef_ = np.zeros((1, n_features))
        self.coef_[0][0] = 1.0
        self.intercept_ = np.array([0.0])
        # Recorded so a fold test can assert the fold never saw its own
        # held-out rows.
        self.seen_rows = [tuple(row) for row in X]
        return self

    def predict_proba(self, X):
        out = []
        for row in X:
            p = 1.0 / (1.0 + np.exp(-(row[0] * 4.0 - 2.0)))
            out.append([1.0 - p, p])
        return np.array(out)


class FakeStratifiedKFold:
    """Contiguous per-class striping: deterministic, and it genuinely holds
    the positive rate near-constant across folds, which is the property
    StratifiedKFold is being used for.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits

    def split(self, X, y):
        pos = [i for i, v in enumerate(y) if v]
        neg = [i for i, v in enumerate(y) if not v]
        folds = [[] for _ in range(self.n_splits)]
        for group in (pos, neg):
            for j, idx in enumerate(group):
                folds[j % self.n_splits].append(idx)
        all_idx = set(range(len(y)))
        for f in folds:
            test = sorted(f)
            yield sorted(all_idx - set(test)), test


def _install_fake_sklearn():
    sk = types.ModuleType("sklearn")
    lm = types.ModuleType("sklearn.linear_model")
    ms = types.ModuleType("sklearn.model_selection")
    lm.LogisticRegression = FakeLogisticRegression
    ms.StratifiedKFold = FakeStratifiedKFold
    sk.linear_model = lm
    sk.model_selection = ms
    sys.modules["sklearn"] = sk
    sys.modules["sklearn.linear_model"] = lm
    sys.modules["sklearn.model_selection"] = ms


_install_fake_sklearn()

from src.mollm_calibrator import (  # noqa: E402
    FEATURE_NAMES, MoLLMCalibrator, _new_estimator,
)
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "fit_mollm_calibrator",
    os.path.join(PROJECT_DIR, "scripts", "fit_mollm_calibrator.py"))
fitmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fitmod)


# ==========================================================================
# Fixtures
# ==========================================================================

def make_rows(n=60, pos_rate=0.15, with_note_ids=True):
    """Synthetic training rows in cal_eval.py --emit-training-data's shape.

    Feature 0 correlates with the label so the fake model has something to
    rank on; the remaining 12 are noise-free zeros -- these tests never assert
    a particular AUROC value, only that the plumbing computes one.
    """
    rows = []
    n_pos = max(2, int(n * pos_rate))
    for i in range(n):
        is_pos = i < n_pos
        feats = [0.0] * len(FEATURE_NAMES)
        feats[0] = 0.9 if is_pos else 0.1
        rows.append({
            "mollm_call_id": f"call-{i}",
            "note_id": f"note-{i % 6}" if with_note_ids else None,
            "split": "val",
            "features": feats,
            "label": 1 if is_pos else 0,
        })
    return rows


def make_ctx(conf=0.9, tier="LOW"):
    return {"confidence_tier_in": tier, "mode": "resolution",
            "ensemble": {"composite_confidence": conf, "confidence_spread": 0.1}}


# ==========================================================================
# Source-structure helpers (AST, not grep -- see test 1's comment)
# ==========================================================================

def _call_name(node):
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _n_constructions(src, name):
    """How many times `name` is CALLED in this source. Docstrings and
    comments quoting the call do not count, which is the entire reason this
    exists.
    """
    return sum(1 for node in ast.walk(ast.parse(src))
               if isinstance(node, ast.Call) and _call_name(node) == name)


def _constructed_inside(src, name):
    """Names of the functions that call `name`. Empty string for a call at
    module level.
    """
    out = set()
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and _call_name(node) == name:
                out.add(fn.name)
    return out


def _has_literal_prefix(src, prefix):
    """True when any STRING LITERAL in the code starts with `prefix`.

    ast.get_docstring-bearing nodes are skipped so that a docstring or comment
    explaining "this used to be hardcoded to /home/ec2-user" does not read as
    the hardcoding itself.
    """
    tree = ast.parse(src)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstrings \
                and node.value.startswith(prefix):
            return True
    return False


# ==========================================================================
# Tests
# ==========================================================================

def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ---- 1. single estimator construction site (the drift regression) -----
    with open(os.path.join(PROJECT_DIR, "scripts", "fit_mollm_calibrator.py"),
              encoding="utf-8") as fh:
        fit_src = fh.read()
    with open(os.path.join(PROJECT_DIR, "src", "mollm_calibrator.py"),
              encoding="utf-8") as fh:
        cal_src = fh.read()

    # AST, not substring search: the fitting script's docstring deliberately
    # QUOTES the old `LogisticRegression(max_iter=1000, C=1.0)` line to record
    # what the drift bug was, and a naive grep cannot tell a cautionary
    # docstring from a live construction. Same for the /home/ec2-user comment
    # below. Counting Call nodes asks the question that actually matters:
    # does this module CONSTRUCT an estimator.
    check("fitting script does not construct LogisticRegression itself",
          _n_constructions(fit_src, "LogisticRegression") == 0)
    check("calibrator module constructs LogisticRegression exactly once",
          _n_constructions(cal_src, "LogisticRegression") == 1)
    check("the one construction site is inside _new_estimator",
          _constructed_inside(cal_src, "LogisticRegression") == {"_new_estimator"})
    check("script imports fit_vectors path, not a private estimator",
          "fit_vectors(" in fit_src)

    est = _new_estimator()
    check("_new_estimator sets class_weight=balanced", est.class_weight == "balanced")
    check("_new_estimator sets max_iter=1000", est.max_iter == 1000)
    check("_new_estimator sets C=1.0", est.C == 1.0)

    # ---- 2. fit() delegates to fit_vectors() ------------------------------
    ctxs = [make_ctx(0.9)] * 20 + [make_ctx(0.2)] * 20
    labels = [1] * 20 + [0] * 20
    c_fit = MoLLMCalibrator().fit(ctxs, labels, min_examples=10)
    check("fit() produces a trained model", c_fit.model is not None)
    check("fit() records n_training_examples", c_fit.n_training_examples == 40)
    check("fit() inherits balanced weighting via _new_estimator",
          c_fit.model.class_weight == "balanced")

    from src.mollm_calibrator import extract_features
    X_equiv = [extract_features(c) for c in ctxs]
    c_vec = MoLLMCalibrator().fit_vectors(X_equiv, labels, min_examples=10)
    check("fit() and fit_vectors() see identical training rows",
          c_fit.model.seen_rows == c_vec.model.seen_rows)

    # ---- 3. fit_vectors guards -------------------------------------------
    def raises(fn, exc=ValueError):
        try:
            fn()
        except exc:
            return True
        except Exception:
            return False
        return False

    check("fit_vectors refuses below min_examples",
          raises(lambda: MoLLMCalibrator().fit_vectors([[0.0]] * 5, [0] * 5)))
    check("fit_vectors refuses length mismatch",
          raises(lambda: MoLLMCalibrator().fit_vectors([[0.0]] * 10, [0] * 9,
                                                        min_examples=2)))
    check("fit_vectors refuses a single class",
          raises(lambda: MoLLMCalibrator().fit_vectors([[0.0]] * 10, [0] * 10,
                                                        min_examples=2)))

    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MoLLMCalibrator().fit_vectors([[float(i % 2)] * len(FEATURE_NAMES)
                                       for i in range(40)],
                                      [i % 2 for i in range(40)], min_examples=2)
        check("fit_vectors warns on low positives-per-feature",
              any("positives per feature" in str(w.message) for w in caught))

    # ---- 4. provenance + leakage guard -----------------------------------
    rows = make_rows(60)
    X = [r["features"] for r in rows]
    y = [r["label"] for r in rows]
    note_ids = sorted({r["note_id"] for r in rows})

    with tempfile.TemporaryDirectory() as td:
        pkl = os.path.join(td, "cal.pkl")
        cal = MoLLMCalibrator().fit_vectors(X, y, min_examples=10)
        cal.save(pkl, training_note_ids=note_ids, training_split="val",
                 code_version="deadbeef")

        clean = MoLLMCalibrator.load(pkl)
        check("save/load round-trips the model", clean.model is not None)
        check("save/load round-trips training_note_ids",
              clean.training_note_ids == note_ids)
        check("save/load round-trips training_split", clean.training_split == "val")
        check("save/load round-trips code_version", clean.code_version == "deadbeef")

        disjoint = MoLLMCalibrator.load(pkl, scored_note_ids=["note-99", "note-98"])
        check("load keeps the model for disjoint notes", disjoint.model is not None)

        leaked = MoLLMCalibrator.load(pkl, scored_note_ids=["note-99", "note-1"])
        check("load REFUSES on training-set overlap", leaked.model is None)
        check("refused calibrator scores None", leaked.score(make_ctx()) is None)

        forced = MoLLMCalibrator.load(pkl, scored_note_ids=["note-1"],
                                      refuse_on_leakage=False)
        check("refuse_on_leakage=False proceeds despite overlap",
              forced.model is not None)

        check("assert_not_trained_on reports the overlap",
              clean.assert_not_trained_on(["note-1", "note-99"]) == ["note-1"])
        check("assert_not_trained_on is empty when disjoint",
              clean.assert_not_trained_on(["note-99"]) == [])

        # A pkl saved WITHOUT provenance must not silently look safe.
        pkl2 = os.path.join(td, "cal_noprov.pkl")
        bare = MoLLMCalibrator().fit_vectors(X, y, min_examples=10)
        bare.save(pkl2)
        unknown = MoLLMCalibrator.load(pkl2, scored_note_ids=["note-1"])
        check("pkl without training notes cannot be leakage-checked",
              unknown.training_note_ids == [])
        check("unknown-provenance pkl is not silently refused",
              unknown.model is not None)

    # ---- 5. cross-validated out-of-fold coverage --------------------------
    pairs, n_splits = fitmod.cross_validated_scores(X, y, n_splits=5)
    check("CV returns a prediction for every row", pairs is not None and
          len(pairs) == len(X))
    check("CV used the requested fold count", n_splits == 5)
    check("CV probabilities are in [0,1]",
          all(0.0 <= p <= 1.0 for p, _ in pairs))
    check("CV labels line up with input labels",
          [int(ok_) for _, ok_ in pairs] == y)

    # Folds must be capped by the minority class, not blow up.
    tiny_X = [[0.9] * len(FEATURE_NAMES)] * 2 + [[0.1] * len(FEATURE_NAMES)] * 8
    tiny_y = [1, 1] + [0] * 8
    tiny_pairs, tiny_info = fitmod.cross_validated_scores(tiny_X, tiny_y, n_splits=5)
    check("CV caps folds at the minority-class size",
          tiny_pairs is not None and tiny_info == 2)

    single_pairs, reason = fitmod.cross_validated_scores(
        [[0.5] * len(FEATURE_NAMES)] * 10, [0] * 10)
    check("CV declines a single-class sample", single_pairs is None)
    check("CV explains why it declined", isinstance(reason, str) and reason)

    one_pos, reason2 = fitmod.cross_validated_scores(
        [[0.9] * len(FEATURE_NAMES)] + [[0.1] * len(FEATURE_NAMES)] * 9,
        [1] + [0] * 9)
    check("CV declines when the minority class has <2 members", one_pos is None)

    # ---- 6. reporting helpers --------------------------------------------
    check("_accuracy_at counts threshold agreement",
          fitmod._accuracy_at([(0.9, True), (0.1, False)], 0.5) == 1.0)
    check("_accuracy_at penalises disagreement",
          fitmod._accuracy_at([(0.9, False), (0.1, True)], 0.5) == 0.0)
    check("_accuracy_at handles an empty sample",
          fitmod._accuracy_at([], 0.5) is None)
    check("_fmt renders None as n/a", fitmod._fmt(None) == "n/a")
    check("_fmt renders floats to 3dp", fitmod._fmt(0.12345) == "0.123")

    fake_model = FakeLogisticRegression()
    fake_model.fit([[0.0]], [0])
    fake_model.classes_ = np.array([0, 1])
    check("_pos_index finds class 1", fitmod._pos_index(fake_model) == 1)

    # The CV block must run end to end without raising, including the
    # note-level bootstrap path -- this is the code that prints the numbers a
    # human decides on, so an exception here is a silent loss of the check.
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summary = fitmod._print_cv_block(pairs, [r["note_id"] for r in rows], 5)
    out = buf.getvalue()
    check("CV block reports AUROC", "AUROC" in out)
    check("CV block prints the null-model column", "null (base rate)" in out)
    check("CV block demotes accuracy explicitly", "NOT the headline" in out)
    check("CV block resamples notes", "notes" in out)
    check("CV block returns a summary dict",
          isinstance(summary, dict) and "auroc" in summary)
    check("CV summary carries a bootstrap interval",
          summary.get("auroc_ci") is None or "lo" in summary["auroc_ci"])

    # Row-level fallback must announce itself rather than quietly pretending
    # to independence it does not have.
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        fitmod._print_cv_block(pairs, [None] * len(pairs), 5)
    check("CV block warns when note_ids are missing",
          "NO note_id" in buf2.getvalue())

    # ---- 7. no duplicate ECE implementation remains -----------------------
    check("fitting script no longer defines its own _ece",
          "def _ece" not in fit_src)
    check("fitting script imports ECE from evaluation.metrics",
          "from evaluation.metrics import" in fit_src)
    check("fitting script no longer hardcodes the EC2 project dir",
          not _has_literal_prefix(fit_src, "/home/ec2-user"))
    check("fitting script passes provenance to save()",
          "training_note_ids=training_note_ids" in fit_src)

    # ---- 8. cal_eval emits what the fitting script needs ------------------
    with open(os.path.join(PROJECT_DIR, "evaluation", "cal_eval.py"),
              encoding="utf-8") as fh:
        cal_eval_src = fh.read()
    check("cal_eval --emit-training-data emits note_id",
          '"note_id": d.get("note_id")' in cal_eval_src)
    check("cal_eval --emit-training-data emits the split",
          '"split": split_prov.get("split")' in cal_eval_src)

    print(f"calibrator-fit tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(run())
