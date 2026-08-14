"""
src/mollm_calibrator.py — MoLLM-Cal: a trained confidence calibrator for
Stage 3's route() (docs/MoLLM_Redesign_Proposal.md S4.1).

WHY THIS EXISTS. route()'s two threshold comparisons (AUTO_VALIDATE_THRESHOLD,
MOLLM_RESOLVE_THRESHOLD) compare a single hand-combined number
(composite_confidence -- combine()'s mean of two logprob-derived confidences)
against two constants. docs/Stage3_Open_Issues.md Issue 4 measured this
directly: ECE 0.7529, all 6 gradable decisions in one run landed in the
[0.9,1.0) confidence bin against 16.7% actual accuracy. Qi et al. 2026's
Table 2 shows the same failure mode in a different domain -- hand-combined
ensemble signals (their "logits-based"/"vote-based" baselines) losing to a
small TRAINED aggregator (MoLAM) on every benchmark tried. This module is
that trained aggregator's counterpart here, scoped to what this project's
data volume and safety requirements actually support.

WHAT THIS DOES NOT DO. It never touches route()'s three hard safety rules
(model disagreement, citation verification, CONTRADICTED/INSUFFICIENT_EVIDENCE/
NONE_CORRECT verdicts) -- see route()'s calibrator_score docstring in
src/mollm_ensemble.py. It only ever substitutes for the composite_confidence
comparison, and only once actually fit on real data; score() returns None
otherwise, which route() treats identically to "no calibrator passed at all".

GRACEFUL UNTRAINED STATE. Importing this module never requires scikit-learn.
It is only imported inside fit()/load(), which are the only paths that touch
an actual model object. Any caller that just wants "is a calibrator
available" gets a clean None from score() with no import error, so a pipeline
run on a box without scikit-learn installed behaves exactly as if this module
did not exist.
"""

# Feature order is fixed and versioned so a saved model's coefficients always
# line up with the same feature at the same index. Adding/removing a feature
# means bumping FEATURE_SET_VERSION and retraining -- never silently
# reordering an existing saved model's inputs.
FEATURE_SET_VERSION = 1

FEATURE_NAMES = [
    "confidence_tier_low",                  # 1.0 if confidence_tier_in == "LOW"
    "mode_resolution",                      # 1.0 if mode == "resolution"
    "composite_confidence",                 # ensemble composite, or 0.0 if unmeasured
    "confidence_spread",                    # |logprob_a - logprob_b|, or 0.0
    "has_guideline_evidence",               # 1.0 if any rules were retrieved (A/B/D/E)
    "n_rules_retrieved",                    # len(rules), capped and scaled to [0,1]
    "max_rule_match_confidence",            # best match_confidence among retrieved rules
    "grounding_basis_guideline_rule",       # one-hot (docs/MoLLM_Redesign_Proposal.md S9.5)
    "grounding_basis_ontology_only",        # one-hot
    "grounding_basis_model_terminology",    # one-hot -- least-checkable tier
    "raw_confidence_agree",                 # 1.0 if both models' self-reported labels match
    "annotation_discrepancy",               # d_anno (S4.2/S9.3)
    "n_rejected_candidates",                # len(rejected_candidates), capped and scaled
]


def extract_features(feature_ctx: dict) -> list:
    """Builds the fixed-order feature vector from a validate_record()-time
    context dict.

    feature_ctx is a plain dict of already-computed values (ensemble,
    citation, retrieval, model_results, grounding_basis, ...), NOT the final
    decision artifact -- this is built and scored BEFORE route() runs, and
    route()'s own output cannot be a calibrator input without being circular.
    See mollm_ensemble.validate_record() for where feature_ctx is assembled.

    Every value defaults to 0.0 rather than raising on a missing key, so a
    caller that builds a partial context (e.g. a contradiction-mode record
    with no candidates) still gets a valid vector -- the calibrator was never
    meant to require every feature to be meaningful for every record type,
    only to make the ones that are available legible to a shared model.
    """
    ensemble = feature_ctx.get("ensemble") or {}
    retrieval = feature_ctx.get("retrieval") or {}
    rules = retrieval.get("rules") or []
    models = feature_ctx.get("model_results") or []
    raw_labels = {m.get("raw_confidence_label") for m in models if m.get("raw_confidence_label")}
    grounding_basis = feature_ctx.get("grounding_basis")
    rejected = feature_ctx.get("rejected_candidates") or []
    match_confs = [r.get("match_confidence") for r in rules
                   if r.get("match_confidence") is not None]

    values = {
        "confidence_tier_low": 1.0 if feature_ctx.get("confidence_tier_in") == "LOW" else 0.0,
        "mode_resolution": 1.0 if feature_ctx.get("mode") == "resolution" else 0.0,
        "composite_confidence": ensemble.get("composite_confidence") or 0.0,
        "confidence_spread": ensemble.get("confidence_spread") or 0.0,
        "has_guideline_evidence": 1.0 if rules else 0.0,
        "n_rules_retrieved": min(len(rules), 5) / 5.0,
        "max_rule_match_confidence": max(match_confs) if match_confs else 0.0,
        "grounding_basis_guideline_rule": 1.0 if grounding_basis == "guideline_rule" else 0.0,
        "grounding_basis_ontology_only": 1.0 if grounding_basis == "ontology_only" else 0.0,
        "grounding_basis_model_terminology":
            1.0 if grounding_basis == "model_terminology_knowledge" else 0.0,
        "raw_confidence_agree": 1.0 if len(raw_labels) == 1 else 0.0,
        "annotation_discrepancy": 1.0 if feature_ctx.get("annotation_discrepancy") else 0.0,
        "n_rejected_candidates": min(len(rejected), 5) / 5.0,
    }
    return [values[name] for name in FEATURE_NAMES]


#: Hyperparameters live here and NOWHERE else. Every estimator this project
#: fits -- production model, cross-validation folds in
#: scripts/fit_mollm_calibrator.py, and any future ablation -- comes through
#: this function, so a change to C, max_iter or class_weight cannot apply to
#: one of them and not the others. The 2026-08-13 P3.3 drift (class_weight
#: added to fit() but not to the fitting script's hand-rolled copy) is the
#: concrete bug this closes.
#:
#: class_weight="balanced": at a 14.3% positive rate an unweighted fit
#: minimises loss by predicting "incorrect" almost everywhere. Weighting
#: inversely to class frequency removes that shortcut; it does not
#: manufacture discrimination that is not in the features -- a balanced model
#: with no signal reports AUROC ~0.5 rather than a flattering accuracy.
def _new_estimator():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")


class MoLLMCalibrator:
    """Wraps a scikit-learn LogisticRegression. Untrained (self.model is None)
    until fit() or a successful load() -- score() returns None in that state
    so every caller's fallback path is identical whether the calibrator was
    never built, failed to load, or is deliberately not yet trusted.
    """

    def __init__(self):
        self.model = None
        self.feature_set_version = FEATURE_SET_VERSION
        self.n_training_examples = 0
        # 2026-08-13 (docs/2026-08-13_Code_Improvement_Proposals.md P3.4).
        # WHICH NOTES THIS MODEL SAW. docs/Evaluation_Criteria.md's split
        # design exists to stop a threshold being fitted to the data it is then
        # evaluated on, and until now that separation was enforced entirely by
        # whoever remembered to pass the right --note-ids. Recording the
        # training notes IN the artifact makes the check mechanical: see
        # assert_not_trained_on() and load()'s scored_note_ids argument.
        self.training_note_ids = []
        self.training_split = None
        self.code_version = None

    def score(self, feature_ctx: dict):
        """Returns P(this decision is correct) in [0, 1], or None if
        untrained. Deliberately never raises -- a calibrator failure must
        degrade to the existing threshold behavior, not take down Stage 3.
        """
        if self.model is None:
            return None
        try:
            x = [extract_features(feature_ctx)]
            proba = self.model.predict_proba(x)[0]
            # predict_proba's column order follows self.model.classes_; find
            # the column for class 1 ("correct") explicitly rather than
            # assuming index 1, which silently breaks if classes_ is [1, 0]
            # (scikit-learn sorts classes_, so this is usually fine, but
            # "usually" is not a guarantee worth trusting a routing decision on).
            classes = list(self.model.classes_)
            idx = classes.index(1) if 1 in classes else (1 if len(proba) > 1 else 0)
            return round(float(proba[idx]), 6)
        except Exception:
            return None

    def fit(self, feature_ctxs: list, labels: list, min_examples: int = 30):
        """Fits a logistic regression. Refuses below min_examples rather than
        fitting a model nobody should trust -- docs/MoLLM_Redesign_Proposal.md
        S8 flags sample size as a real risk in both directions (too few
        examples: overfits or learns nothing). 30 is a placeholder floor, not
        a calibrated one; docs/Evaluation_Criteria.md's validation-slice
        design is the right place to set the real number, once the batch run
        has produced enough gradable decisions to make that call meaningfully.

        labels are 1 (verdict matched gold) / 0 (did not), the same grading
        evaluation/cal_eval.py already performs for resolution-mode decisions.

        Deliberately plain LogisticRegression, not XGBoost, at this data
        volume -- docs/MoLLM_Redesign_Proposal.md S4.1: fewer parameters than
        examples is a real overfitting risk below ~100 examples, and a linear
        model's coefficients are directly inspectable, which matters for the
        same "white box, auditable" standard applied to every other component
        in this pipeline (S9.5). Revisit only once the gradable sample is
        large enough that model capacity, not overfitting, is the binding
        constraint.

        2026-08-13 (P3.3): class_weight="balanced". The 2026-08-13 report S5.4
        fit this model on 140 examples of which 20 were positive -- 14.3%. An
        unweighted logistic regression on that balance minimises loss most
        cheaply by predicting "incorrect" almost everywhere, which is exactly
        what the held-out check then observed (mean predicted P(correct) =
        0.1274, held-out "accuracy" 85.71% = the base rate exactly). Weighting
        the classes inversely to their frequency removes that shortcut. It does
        not manufacture signal that is not there -- if the features cannot
        discriminate, a balanced model says so through AUROC ~0.5 rather than
        through a confident-looking accuracy number.

        NOTE ON THE REAL BINDING CONSTRAINT. min_examples counts ROWS, but the
        quantity that limits a binary classifier is POSITIVES: 13 features
        against 20 positives is roughly 1.5 positives per parameter, well
        inside the regime where a linear model can fit noise. fit() warns when
        that ratio is below 5 rather than refusing, because refusing would
        leave the pipeline on raw composite_confidence, which the same report
        measured at 14.29% accuracy across the entire threshold sweep -- a bad
        calibrator that knows it is bad is still an improvement on a signal
        measured to be inverted.
        """
        if len(feature_ctxs) != len(labels):
            raise ValueError(
                f"feature_ctxs ({len(feature_ctxs)}) and labels ({len(labels)}) "
                f"must be the same length")
        # Extract first, then hand off to fit_vectors(), which owns every
        # check and the single estimator construction. This method is now
        # only "turn dicts into vectors"; see fit_vectors() for why there is
        # exactly one construction site.
        X = [extract_features(ctx) for ctx in feature_ctxs]
        return self.fit_vectors(X, labels, min_examples=min_examples)

    def fit_vectors(self, X: list, labels: list, min_examples: int = 30):
        """fit(), but for feature vectors that have ALREADY been through
        extract_features().

        2026-08-13 (docs/2026-08-13_Code_Improvement_Proposals.md P3, and a
        drift bug this method exists to make impossible).

        WHY. evaluation/cal_eval.py --emit-training-data writes the FINAL
        13-element vector, not a feature_ctx dict, so fit() cannot consume it
        (it would call .get() on a list). scripts/fit_mollm_calibrator.py
        therefore hand-rolled its own LogisticRegression construction, with a
        comment promising it used "the same hyperparameters fit() itself
        uses". That promise silently broke the moment P3.3 added
        class_weight="balanced" to fit() and not to the script: the saved
        production model was then a DIFFERENT estimator from the one this
        class documents, and nothing in either file could notice.

        There is now exactly ONE place in the codebase that constructs the
        estimator -- fit() delegates here after extracting features, and the
        script calls here directly. The hyperparameters cannot drift again
        because there is no second copy to drift from.
        """
        if len(X) < min_examples:
            raise ValueError(
                f"only {len(X)} labeled example(s), refusing to fit "
                f"below min_examples={min_examples} -- see fit()'s docstring")
        if len(X) != len(labels):
            raise ValueError(
                f"X ({len(X)}) and labels ({len(labels)}) must be the same length")
        if len(set(labels)) < 2:
            raise ValueError(
                f"all {len(labels)} labels are the same value ({labels[0]}) -- "
                f"a logistic regression cannot be fit on a single class")

        # Positives, not rows, are the binding constraint -- see fit()'s
        # "NOTE ON THE REAL BINDING CONSTRAINT". Warns rather than refuses:
        # the fallback is raw composite_confidence, measured at 14.29%
        # accuracy across the whole threshold sweep.
        n_pos = sum(1 for y in labels if y)
        ratio = n_pos / max(1, len(FEATURE_NAMES))
        if ratio < 5:
            import warnings as _warnings
            _warnings.warn(
                f"fitting {len(FEATURE_NAMES)} features on only {n_pos} positive "
                f"example(s) ({ratio:.1f} positives per feature; >=5 is the "
                f"conventional floor). This model can fit noise. Read the AUROC "
                f"and the null-model baseline before trusting it -- accuracy on "
                f"this class balance is not informative.",
                stacklevel=2)

        self.model = _new_estimator()
        self.model.fit(X, labels)
        self.n_training_examples = len(X)
        return self

    def assert_not_trained_on(self, scored_note_ids):
        """Returns the note_ids this calibrator was TRAINED on that also appear
        in `scored_note_ids` -- i.e. the leakage, if any.

        Empty list means clean (including the case where no training notes were
        recorded, which is reported separately by `training_note_ids == []`).
        A caller that wants to refuse rather than warn checks the return value;
        this method does not decide policy, because "refuse" is right for a
        reported number and wrong for an exploratory re-run.
        """
        if not self.training_note_ids or not scored_note_ids:
            return []
        return sorted(set(self.training_note_ids) & set(scored_note_ids))

    def save(self, path: str, training_note_ids=None, training_split=None,
             code_version=None):
        """Persists the model AND its provenance.

        2026-08-13 (P3.4): training_note_ids/training_split are what turn
        docs/Evaluation_Criteria.md's leakage control from a policy sentence
        into something checkable. A .pkl that does not know which notes it saw
        cannot be audited later, and "which notes were in that calibrator's
        training set" is not recoverable from the coefficients.
        """
        import pickle
        if training_note_ids is not None:
            self.training_note_ids = sorted(set(training_note_ids))
        if training_split is not None:
            self.training_split = training_split
        if code_version is not None:
            self.code_version = code_version
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_set_version": self.feature_set_version,
                "n_training_examples": self.n_training_examples,
                "feature_names": FEATURE_NAMES,
                "training_note_ids": self.training_note_ids,
                "training_split": self.training_split,
                "code_version": self.code_version,
            }, f)

    @classmethod
    def load(cls, path: str, scored_note_ids=None,
             refuse_on_leakage: bool = True) -> "MoLLMCalibrator":
        """Returns an untrained calibrator (score() -> None) on ANY load
        failure -- missing file, corrupt pickle, feature-version mismatch --
        rather than raising. A calibrator that can't be trusted must degrade
        to "no calibrator", not crash Stage 3 or silently score against a
        feature vector it was never actually fit for.

        2026-08-13 (P3.4) LEAKAGE REFUSAL. `scored_note_ids` is the set of
        notes the caller is about to score. If any of them are in this
        calibrator's own training set, the model has seen their labels, and any
        routing decision it makes on them is not evidence of anything --
        reported numbers from such a run would be optimistically biased in a
        way nothing downstream could detect. With refuse_on_leakage=True (the
        default) the calibrator degrades to untrained and says why, which
        routes on raw composite_confidence exactly as if no calibrator had been
        passed. Set it False deliberately for an exploratory re-run.

        A calibrator with NO recorded training notes (fitted before P3.4)
        cannot be checked. That is reported once, loudly, rather than assumed
        safe -- "we cannot tell" must not silently read as "it is fine".
        """
        import pickle
        inst = cls()
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if data.get("feature_set_version") != FEATURE_SET_VERSION:
                return inst  # stale feature layout, refuse rather than misapply
            inst.model = data.get("model")
            inst.feature_set_version = data.get("feature_set_version", FEATURE_SET_VERSION)
            inst.n_training_examples = data.get("n_training_examples", 0)
            inst.training_note_ids = data.get("training_note_ids") or []
            inst.training_split = data.get("training_split")
            inst.code_version = data.get("code_version")
        except Exception:
            inst.model = None
            return inst

        if scored_note_ids is not None and inst.model is not None:
            if not inst.training_note_ids:
                print("  ! calibrator has NO recorded training note_ids "
                      "(fitted before 2026-08-13); leakage cannot be checked. "
                      "Refit with scripts/fit_mollm_calibrator.py to record them.")
            else:
                overlap = inst.assert_not_trained_on(scored_note_ids)
                if overlap:
                    msg = (f"  ! LEAKAGE: calibrator was TRAINED on "
                           f"{len(overlap)} of the note(s) about to be scored "
                           f"({', '.join(overlap[:5])}"
                           f"{' ...' if len(overlap) > 5 else ''}).")
                    if refuse_on_leakage:
                        print(msg + "\n    Refusing to use it -- routing falls "
                                    "back to raw composite_confidence. Refit on "
                                    "a disjoint split, or pass "
                                    "refuse_on_leakage=False for an exploratory "
                                    "run whose numbers you will not report.")
                        inst.model = None
                    else:
                        print(msg + "\n    refuse_on_leakage=False: proceeding. "
                                    "Numbers from this run are NOT reportable.")
        return inst
