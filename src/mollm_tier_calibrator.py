"""
src/mollm_tier_calibrator.py -- a trained consensus-confidence model for
entities where the 3-model MoLLM ensemble (src/mollm_tier_gate.py) does NOT
reach unanimous agreement.

THE PROBLEM. route_tier()'s Tier 1-5 table treats ensemble agreement as
binary: 3/3 unanimous SUPPORTED_1 above a fixed confidence floor is trusted
(TIER_1_AUTO_VALIDATED); anything less than unanimous -- a confident 2-1
split, a 1-1-1 split, doesn't matter which -- is routed straight to
TIER_4_ENSEMBLE_SPLIT and sent to a human, with no distinction between "the
majority is clearly right and one model is a weak outlier" and "the models
are genuinely divided." Measured on real corpus data: TIER_4_ENSEMBLE_SPLIT
is ~44.8% of all decisions (175/391) -- nearly half of everything the gate
sees gets discarded to HITL by this one rule, and direct inspection of
individual cases has already shown some of those splits have a demonstrably
CORRECT majority answer.

WHAT THIS MODULE DOES. For exactly that low-confidence, non-unanimous
population -- entities that already failed every hard Tier 1/2/3 rule --
this fits a small classifier that estimates P(the top candidate is
correct) from the actual SHAPE of the disagreement (how many models agree,
how confident each one is, how far the confidences are spread) plus
independent context about the entity's own processing history (which
retrieval tier it resolved through, whether it passed through a fragile
fallback path, how many times a similar resolution has been confirmed
before). A high estimate promotes the entity to a new, separately-labeled
approval tier; a low one leaves it routed to HITL exactly as today. This
never touches route_tier()'s existing Tier 1/2/3 logic or its hard safety
checks -- it only gives the currently-all-or-nothing "was the vote
unanimous" question a graded, evidence-based answer for the cases that
would otherwise be discarded with no further judgment at all.

SAFE BY DEFAULT. This module never imports scikit-learn at import time --
only when a model is actually fit or loaded. An untrained instance's
score() always returns None, and a None score must be treated by the
caller exactly like "no calibrator available" -- existing HITL routing,
unchanged. A caller must never let this module's failure (a bad load, a
scoring exception) change routing behavior; every failure path here
degrades to None rather than raising.
"""

import warnings

FEATURE_SET_VERSION = 1

# Each entry: (name, one-line meaning). Order is fixed -- a saved model's
# learned weights are only meaningful paired with this exact order, so
# adding/removing/reordering a feature means bumping FEATURE_SET_VERSION
# and refitting, never quietly reshuffling an existing model's inputs.
FEATURE_NAMES = [
    "frac_supported_1",                     # share of usable votes for the top candidate
    "frac_rerank_same_target",               # share agreeing on the same alternate candidate
    "frac_none_correct",                     # share rejecting every candidate
    "frac_usable_votes",                     # how many of the 3 models produced a usable vote
    "mean_logprob_confidence",               # average confidence across usable votes
    "min_logprob_confidence",                # weakest usable vote's confidence
    "confidence_spread",                     # gap between the strongest and weakest vote
    "match_tier_is_exact_or_synonym",        # the retrieval tier that produced the top candidate
    "top_candidate_similarity_score",        # that candidate's own retrieval score
    "is_ambiguous",                          # Stage 2b's own multi-candidate ambiguity flag
    "domain_conflict",                       # candidate's OMOP domain disagreed with the label
    "resolved_via_value_stripped_fallback",  # a noisy-lab-value suffix had to be stripped first
    "resolved_via_original_text_fallback",   # the expanded text failed and raw text was retried
    "resolved_via_acronym_escalation",       # an ambiguous abbreviation was MoLLM-resolved first
    "expansion_ambiguous",                   # the abbreviation itself had >1 dictionary meaning
    "prior_confirmation_count",              # how often this same resolution was confirmed before
]


def usable_votes(model_results: list) -> list:
    """The subset of model_results with a real, non-degenerate verdict.
    A model that errored or produced a degenerate generation contributes no
    evidence either way -- it is excluded here rather than counted as a
    silent NONE_CORRECT, matching how route_tier() itself already treats a
    degenerate vote as "not a real disagreement."
    """
    return [m for m in (model_results or [])
            if m.get("verdict") not in (None, "ERROR") and not m.get("degenerate_generation")]


def build_feature_context(entity: dict, model_results: list,
                          prior_confirmation_count: int = 0) -> dict:
    """Packages exactly what route_tier() already has in hand at the point
    it would consult this calibrator -- the entity record and the two-step
    ensemble's per-model verdicts -- plus one value the caller must supply
    from a DB lookup (prior_confirmation_count). Kept as a separate
    plain-dict step so featurize() itself stays a pure function of already-
    gathered data, testable with synthetic inputs and no DB connection.
    """
    return {
        "entity": entity or {},
        "model_results": model_results or [],
        "prior_confirmation_count": prior_confirmation_count or 0,
    }


def featurize(context: dict) -> list:
    """Turns a feature context into the fixed-order numeric vector the
    model consumes. Every value has a safe default, so a context missing
    some fields (a differently-shaped entity record, an empty vote list)
    still produces a valid vector instead of raising.
    """
    entity = context.get("entity") or {}
    votes = usable_votes(context.get("model_results"))
    n_usable = len(votes)

    verdicts = [v.get("verdict") for v in votes]
    n_top = sum(1 for v in verdicts if v == "SUPPORTED_1")
    n_none = sum(1 for v in verdicts if v == "NONE_CORRECT")
    reranks = [v for v in verdicts if v and v.startswith("RE_RANK_TO_CANDIDATE_")]
    n_rerank_agree = 0
    if reranks:
        top_target = max(set(reranks), key=reranks.count)
        n_rerank_agree = reranks.count(top_target)

    confidences = [v.get("logprob_confidence") for v in votes
                  if v.get("logprob_confidence") is not None]

    candidates = entity.get("candidates") or []
    top_similarity = candidates[0].get("similarity_score") if candidates else None
    normalized_from = entity.get("normalized_from") or ""

    fields = {
        "frac_supported_1": (n_top / n_usable) if n_usable else 0.0,
        "frac_rerank_same_target": (n_rerank_agree / n_usable) if n_usable else 0.0,
        "frac_none_correct": (n_none / n_usable) if n_usable else 0.0,
        "frac_usable_votes": n_usable / 3.0,
        "mean_logprob_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
        "min_logprob_confidence": min(confidences) if confidences else 0.0,
        "confidence_spread": (max(confidences) - min(confidences)) if confidences else 0.0,
        "match_tier_is_exact_or_synonym":
            1.0 if entity.get("match_tier") in ("1 (Exact)", "2 (Synonym)") else 0.0,
        "top_candidate_similarity_score": top_similarity if top_similarity is not None else 0.0,
        "is_ambiguous": 1.0 if entity.get("is_ambiguous") else 0.0,
        "domain_conflict": 1.0 if entity.get("domain_conflict") else 0.0,
        "resolved_via_value_stripped_fallback":
            1.0 if "value_stripped_from_" in normalized_from else 0.0,
        "resolved_via_original_text_fallback":
            1.0 if "original_after_expanded_failed" in normalized_from else 0.0,
        "resolved_via_acronym_escalation":
            1.0 if ("+acronym_mollm" in normalized_from or "+acronym_cache" in normalized_from)
            else 0.0,
        "expansion_ambiguous": 1.0 if entity.get("expansion_ambiguous") else 0.0,
        "prior_confirmation_count":
            min(context.get("prior_confirmation_count") or 0, 10) / 10.0,
    }
    return [fields[name] for name in FEATURE_NAMES]


def count_prior_confirmations(conn, entity_text: str, concept_id) -> int:
    """How many times this exact (entity text, concept) pairing has already
    been confirmed -- either as an AUTO_VALIDATED/AUTO_RESOLVED decision in
    mollm_tier_gate_decisions, or as an APPROVED/CORRECTED human review in
    hitl_review_queue. Nothing in the codebase already tracks this; it is a
    fresh aggregation query, run fresh each time rather than cached, since
    no reliable cache invalidation story exists yet.

    This counts DuckDB's own record of confirmed/would-be-confirmed
    decisions, not real knowledge-graph writes -- KG3 ingestion runs in
    dry-run mode throughout this pipeline today, so there is no live graph
    to query yet. Treat this as evidence with the same caution as anything
    else derived from dry-run data: it needs to be validated on a held-out
    split like every other feature here, not assumed trustworthy just
    because it looks like independent confirmation.

    Returns 0 (never raises, never returns None) on a missing connection,
    missing arguments, or any DB error -- an entity with no confirmation
    history is a normal, common state, not a failure.
    """
    if conn is None or not entity_text or concept_id is None:
        return 0
    total = 0
    try:
        total += conn.execute("""
            SELECT count(*) FROM mollm_tier_gate_decisions d
            JOIN extracted_entities e ON e.entity_id = d.entity_id
            JOIN normalized_entities n ON n.entity_id = d.entity_id
            WHERE lower(trim(e.original_text)) = lower(trim(?))
              AND n.omop_concept_id = ?
              AND d.mollm_routing_decision IN ('AUTO_VALIDATED', 'AUTO_RESOLVED')
        """, [entity_text, concept_id]).fetchone()[0]
    except Exception:
        pass
    try:
        total += conn.execute("""
            SELECT count(*) FROM hitl_review_queue h
            JOIN extracted_entities e ON e.entity_id = h.entity_id
            WHERE lower(trim(e.original_text)) = lower(trim(?))
              AND h.corrected_concept_id = ?
              AND h.reviewer_decision IN ('APPROVED', 'CORRECTED')
        """, [entity_text, concept_id]).fetchone()[0]
    except Exception:
        pass
    return total


def _build_model():
    """The one place a scikit-learn estimator gets constructed, so every
    fitting call site (production fit, cross-validation, an ablation
    script) shares identical hyperparameters by construction rather than by
    convention. class_weight="balanced" matters specifically because this
    population (currently HITL-routed, not-yet-approved cases) is expected
    to be mostly incorrect -- an unweighted fit on a skewed label
    distribution minimizes loss by predicting the majority class almost
    everywhere, which would defeat the purpose of building this at all.
    """
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")


class ConsensusCalibrator:
    """A small trained model over the ensemble's disagreement pattern plus
    entity provenance, estimating P(the leading candidate is actually
    correct) for entities the raw Tier 1-5 rules would otherwise send to
    HITL with no further judgment. Starts untrained; score() returns None
    until fit() or load() succeeds, and every caller must treat that None
    exactly like "this feature doesn't exist" -- unchanged HITL routing.
    """

    def __init__(self):
        self.model = None
        self.feature_set_version = FEATURE_SET_VERSION
        self.n_training_examples = 0
        self.training_note_ids = []
        self.training_split = None
        self.code_version = None

    def score(self, context: dict):
        """P(correct) in [0, 1], or None if untrained or scoring fails for
        any reason. Never raises -- a scoring failure must fall back to
        existing routing, not interrupt the gate.
        """
        if self.model is None:
            return None
        try:
            vector = [featurize(context)]
            probabilities = self.model.predict_proba(vector)[0]
            classes = list(self.model.classes_)
            positive_index = classes.index(1) if 1 in classes else (
                1 if len(probabilities) > 1 else 0)
            return round(float(probabilities[positive_index]), 6)
        except Exception:
            return None

    def fit(self, contexts: list, labels: list, min_examples: int = 100):
        """Fits on a list of feature contexts (see build_feature_context())
        and their 1/0 correctness labels. min_examples defaults to 100
        rather than a token minimum -- this model's output can promote an
        entity straight to an unreviewed write, so a small, possibly noisy
        fit is not an acceptable default; the actual number that's safe to
        trust should be set by checking real held-out performance, not
        assumed from this floor alone.
        """
        if len(contexts) != len(labels):
            raise ValueError(
                f"contexts ({len(contexts)}) and labels ({len(labels)}) must match")
        vectors = [featurize(c) for c in contexts]
        return self.fit_vectors(vectors, labels, min_examples=min_examples)

    def fit_vectors(self, vectors: list, labels: list, min_examples: int = 100):
        """fit(), starting from already-featurized vectors -- the single
        construction site _build_model() guarantees every caller (this
        method, a fitting script, a cross-validation loop) uses the same
        hyperparameters with no possibility of drifting apart.
        """
        if len(vectors) < min_examples:
            raise ValueError(
                f"only {len(vectors)} labeled example(s); refusing to fit below "
                f"min_examples={min_examples}")
        if len(vectors) != len(labels):
            raise ValueError(f"vectors ({len(vectors)}) and labels ({len(labels)}) must match")
        if len(set(labels)) < 2:
            raise ValueError(
                f"all {len(labels)} labels are the same value ({labels[0]}) -- "
                f"a classifier cannot be fit on a single class")

        n_positive = sum(1 for y in labels if y)
        positives_per_feature = n_positive / max(1, len(FEATURE_NAMES))
        if positives_per_feature < 5:
            warnings.warn(
                f"fitting {len(FEATURE_NAMES)} features on only {n_positive} positive "
                f"example(s) ({positives_per_feature:.1f} per feature; 5+ is the usual "
                f"floor). This model can fit noise -- check its AUROC against a "
                f"held-out split before trusting it.", stacklevel=2)

        self.model = _build_model()
        self.model.fit(vectors, labels)
        self.n_training_examples = len(vectors)
        return self

    def trained_on_any_of(self, note_ids) -> list:
        """Which of `note_ids` this model was trained on, if any -- the
        overlap that would make a reported number on those notes
        meaningless (the model has already seen their labels). Empty list
        means clean.
        """
        if not self.training_note_ids or not note_ids:
            return []
        return sorted(set(self.training_note_ids) & set(note_ids))

    def save(self, path: str, training_note_ids=None, training_split=None,
            code_version=None):
        """Persists the model together with which notes it was trained on
        -- without that record, a leakage check on a later run is
        impossible, not just inconvenient.
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
    def load(cls, path: str, scoring_note_ids=None, refuse_on_leakage: bool = True):
        """Loads a saved model, or returns a fresh untrained instance on
        any failure (missing file, corrupt data, a feature-set version that
        no longer matches this module) -- a caller must never crash because
        a calibrator artifact is missing or stale.

        If `scoring_note_ids` is given and overlaps this model's own
        training notes, the loaded model is refused (degraded back to
        untrained) by default, since anything it says about those notes
        would reflect memorized labels, not real generalization. Pass
        refuse_on_leakage=False only for an exploratory run whose numbers
        will not be reported anywhere.
        """
        import pickle
        instance = cls()
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if data.get("feature_set_version") != FEATURE_SET_VERSION:
                return instance
            instance.model = data.get("model")
            instance.feature_set_version = data.get("feature_set_version", FEATURE_SET_VERSION)
            instance.n_training_examples = data.get("n_training_examples", 0)
            instance.training_note_ids = data.get("training_note_ids") or []
            instance.training_split = data.get("training_split")
            instance.code_version = data.get("code_version")
        except Exception:
            instance.model = None
            return instance

        if scoring_note_ids is not None and instance.model is not None:
            if not instance.training_note_ids:
                print("  ! this calibrator has no recorded training notes; "
                      "leakage cannot be checked.")
            else:
                overlap = instance.trained_on_any_of(scoring_note_ids)
                if overlap:
                    message = (f"  ! leakage: trained on {len(overlap)} of the note(s) "
                               f"about to be scored ({', '.join(overlap[:5])}"
                               f"{' ...' if len(overlap) > 5 else ''}).")
                    if refuse_on_leakage:
                        print(message + "\n    refusing to use it; falling back to "
                                        "existing routing.")
                        instance.model = None
                    else:
                        print(message + "\n    proceeding anyway -- these numbers "
                                        "are not reportable.")
        return instance
