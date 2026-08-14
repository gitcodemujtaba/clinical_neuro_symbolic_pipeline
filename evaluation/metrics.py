"""
evaluation/metrics.py — the single implementation of every calibration and
discrimination metric this project reports.

WHY THIS EXISTS. As of 2026-08-13 the same equal-width-bin ECE formula was
written out THREE times: evaluation/cal_eval.py's compute_ece(), a private
_ece() in scripts/fit_mollm_calibrator.py, and inline in
evaluation/stage2b_cal_eval.py's tier3_ece(). Each copy was correct in
isolation. That is precisely the state the decoding_mode_mismatch bug
(2026-08-13 report S3.3) was in before it broke: one idea, several copies, no
mechanism to keep them honest. This module is the one copy.

NO DATABASE, NO sklearn, NO numpy. Pure stdlib, so scripts/fit_mollm_
calibrator.py can import it without pulling in cal_eval.py's DuckDB machinery,
and so a metric can be unit-tested with a hand-written list of pairs. sklearn
is available in this project but is deliberately not used for AUROC/AP here --
these are twenty lines each, and a hand-written version that can be read
against its own docstring is worth more in a dissertation appendix than a
library call.

WHAT CHANGED IN THE METHODOLOGY, AND WHY (2026-08-13).

  1. EQUAL-MASS BINNING IS AVAILABLE, AND IS THE RIGHT DEFAULT FOR STAGE 3.
     The report's S5.4 Stage 3 ECE of 0.773 was computed with 10 equal-WIDTH
     bins over a population where 128 of 140 points fell in a single bin
     ([0.9, 1.0)). An "expected calibration error" averaged over bins that are
     empty or near-empty is not measuring calibration across the confidence
     range -- it is reporting |0.918 - 0.125| with extra arithmetic. Equal-MASS
     (quantile) bins put an equal number of POINTS in each bin, so every bin's
     accuracy estimate rests on the same sample size. Guo et al. 2017's
     original formulation uses equal width; Nixon et al. 2019 and Roelofs
     et al. 2022 both show equal-width ECE is badly biased under exactly this
     kind of concentrated distribution. Both are offered; the scheme used is
     recorded in the report dict so no number is ever ambiguous about it.

  2. n_nonempty_bins IS ALWAYS REPORTED. A single-bin result is now visible on
     the face of the output rather than requiring someone to read the table.

  3. ACCURACY IS NOT A HEADLINE METRIC ON IMBALANCED DATA. On the Stage 3
     gradable set (14.3% positive), predicting "incorrect" for everything
     scores 85.7% accuracy -- which is exactly what the calibrator's held-out
     check reported, and exactly why that number needed a caveat. auroc() and
     average_precision() measure whether a score can RANK, which is the actual
     question; null_model_report() prints what you get for free so the
     comparison is unavoidable rather than optional.

  4. BOOTSTRAP CIs RESAMPLE NOTES, NOT ENTITIES. docs/Evaluation_Criteria.md
     requires note-level bootstrap CIs and a paired comparison design, and
     nothing in the repository implemented either as of 2026-08-13. Entities
     within one discharge note are not independent -- the same abbreviation,
     the same templated section headers, the same patient recur -- so
     resampling entities would produce intervals that are too narrow, i.e.
     confidently wrong. bootstrap_ci() therefore takes CLUSTERS (one per note)
     and resamples whole notes with replacement.
"""

import math
import random

DEFAULT_BINS = 10
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 13


# ==========================================================================
# Calibration
# ==========================================================================

def _clean_pairs(pairs):
    """(confidence, correct) pairs with a usable confidence, coerced to
    (float, bool). Drops None confidences rather than treating them as 0.0 --
    "no confidence was measured" is not "confidence zero", and the difference
    matters: mollm_decisions rows with no logprobs route to HITL precisely
    because the value is unmeasurable, and folding them in at 0.0 would make
    the low-confidence bins look artificially well-calibrated.
    """
    out = []
    for conf, correct in pairs:
        if conf is None:
            continue
        try:
            out.append((float(conf), bool(correct)))
        except (TypeError, ValueError):
            continue
    return out


def _equal_width_bins(pairs, n_bins):
    bins = [[] for _ in range(n_bins)]
    edges = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    for conf, correct in pairs:
        # min() keeps confidence == 1.0 in the last bin rather than indexing
        # off the end -- the standard off-by-one in every hand-rolled ECE.
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))
    return bins, edges


def _equal_mass_bins(pairs, n_bins):
    """Quantile bins: equal COUNT per bin, edges taken from the data.

    Ties are kept together in one bin rather than split across a boundary --
    splitting identical confidences into different bins would produce two bins
    with the same mean_confidence and different accuracies, which is an
    artefact of the binning, not of the model. Consequence: with heavily tied
    scores you get fewer than n_bins non-empty bins, which n_nonempty_bins
    reports honestly instead of padding.
    """
    ordered = sorted(pairs, key=lambda p: p[0])
    n = len(ordered)
    if n == 0:
        return [], []
    target = max(1, n // n_bins)
    bins, edges = [], []
    i = 0
    while i < n:
        j = min(i + target, n)
        # Extend past the nominal boundary while the next point ties the last
        # one already inside this bin.
        while j < n and ordered[j][0] == ordered[j - 1][0]:
            j += 1
        chunk = ordered[i:j]
        bins.append(chunk)
        edges.append((chunk[0][0], chunk[-1][0]))
        i = j
    # Merge any trailing runt bin into its predecessor so the last bin is not
    # a single point with accuracy 0.0 or 1.0 by construction.
    if len(bins) > 1 and len(bins[-1]) < max(2, target // 2):
        bins[-2].extend(bins[-1])
        edges[-2] = (edges[-2][0], edges[-1][1])
        bins.pop()
        edges.pop()
    return bins, edges


def compute_ece(pairs, n_bins=DEFAULT_BINS, scheme="equal_width"):
    """Expected Calibration Error. Returns None on an empty population.

    ECE = sum over bins of (bin_size / N) * |accuracy - mean_confidence|.
    0 = perfectly calibrated; higher = confidence is systematically
    misleading. Identical arithmetic to evaluation/cal_eval.py's original
    compute_ece() when scheme="equal_width", so historical numbers reproduce
    exactly -- verified by the stub tests at the bottom of this file.
    """
    report = compute_ece_report(pairs, n_bins=n_bins, scheme=scheme)
    return report["ece"] if report else None


def compute_ece_report(pairs, n_bins=DEFAULT_BINS, scheme="equal_width",
                       value_name="confidence"):
    """ECE plus everything needed to interpret it: the per-bin table, N,
    the number of NON-EMPTY bins, the scheme used, MCE and Brier score.

    value_name renames the confidence column in the returned table
    ("similarity" for Stage 2b's SapBERT scores, "confidence" elsewhere) so a
    printed table never mislabels what was actually binned.
    """
    clean = _clean_pairs(pairs)
    if not clean:
        return None

    if scheme == "equal_mass":
        bins, edges = _equal_mass_bins(clean, n_bins)
    elif scheme == "equal_width":
        bins, edges = _equal_width_bins(clean, n_bins)
    else:
        raise ValueError(f"unknown binning scheme {scheme!r}; "
                         f"expected 'equal_width' or 'equal_mass'")

    n = len(clean)
    ece = 0.0
    mce = 0.0
    table = []
    for b, (lo, hi) in zip(bins, edges):
        if not b:
            table.append({"bin": f"[{lo:.2f}, {hi:.2f})", "n": 0,
                          f"mean_{value_name}": None, "accuracy": None, "gap": None})
            continue
        mean_conf = sum(c for c, _ in b) / len(b)
        acc = sum(1 for _, ok in b if ok) / len(b)
        gap = abs(acc - mean_conf)
        ece += (len(b) / n) * gap
        mce = max(mce, gap)
        table.append({"bin": f"[{lo:.2f}, {hi:.2f})", "n": len(b),
                      f"mean_{value_name}": round(mean_conf, 4),
                      "accuracy": round(acc, 4), "gap": round(gap, 4)})

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier": brier_score(clean),
        "n": n,
        "n_bins_requested": n_bins,
        # The number that makes a degenerate result self-evident. A 10-bin ECE
        # resting on 1 non-empty bin is a different claim from one resting on
        # 10, and the single ECE figure cannot distinguish them.
        "n_nonempty_bins": sum(1 for b in bins if b),
        "scheme": scheme,
        "base_rate": round(sum(1 for _, ok in clean if ok) / n, 4),
        "table": table,
    }


def brier_score(pairs):
    """Mean squared error between confidence and outcome. Unlike ECE it is a
    PROPER scoring rule: it cannot be gamed by a model that predicts the base
    rate everywhere (which scores ECE ~0 while carrying no information), so
    reporting it next to ECE is what makes that failure visible. Lower better.
    """
    clean = _clean_pairs(pairs)
    if not clean:
        return None
    return round(sum((c - (1.0 if ok else 0.0)) ** 2 for c, ok in clean) / len(clean), 4)


def max_calibration_error(pairs, n_bins=DEFAULT_BINS, scheme="equal_width"):
    """The worst single bin's |accuracy - mean_confidence|. ECE averages away
    a catastrophically wrong bin if it is small; MCE does not."""
    report = compute_ece_report(pairs, n_bins=n_bins, scheme=scheme)
    return report["mce"] if report else None


# ==========================================================================
# Discrimination -- "can this score RANK, regardless of its absolute value"
# ==========================================================================

def auroc(pairs):
    """Area under the ROC curve, via the Mann-Whitney U identity (the
    probability a random correct case is scored above a random incorrect one).
    Ties contribute 0.5, handled by ranking with average ranks.

    0.5 = no discriminative signal whatsoever. Returns None when one class is
    absent (AUROC is undefined, not 0.5 -- an important distinction on small
    gradable samples where a fold can legitimately contain one class).

    WHY THIS IS THE HEADLINE, NOT ACCURACY. The 2026-08-13 report S5.4 found
    Stage 3's composite_confidence INVERTED: the [0.9,1.0) bin was less
    accurate than [0.8,0.9). ECE cannot express "inverted" -- it is an
    absolute-difference measure, so a perfectly anti-correlated score and a
    perfectly uninformative one can produce similar ECEs. AUROC below 0.5
    says "inverted" unambiguously, which is the actionable finding.
    """
    clean = _clean_pairs(pairs)
    pos = [c for c, ok in clean if ok]
    neg = [c for c, ok in clean if not ok]
    if not pos or not neg:
        return None

    ordered = sorted(clean, key=lambda p: p[0])
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum_pos = sum(ranks[k] for k, (_, ok) in enumerate(ordered) if ok)
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return round(u / (n_pos * n_neg), 4)


def average_precision(pairs):
    """Area under the precision-recall curve (the AP / interpolation-free
    form: sum of precision at each positive, divided by the number of
    positives).

    Preferred over AUROC as the SECOND number on heavily imbalanced data:
    AUROC's baseline is always 0.5 regardless of balance, whereas AP's
    baseline IS the positive base rate -- so on the Stage 3 gradable set an AP
    of 0.15 is immediately legible as "no better than chance", where an AUROC
    of 0.55 is not.
    """
    clean = _clean_pairs(pairs)
    n_pos = sum(1 for _, ok in clean if ok)
    if not clean or n_pos == 0:
        return None
    # TIES ARE HANDLED AT THE GROUP LEVEL, NOT BY INPUT ORDER. The naive
    # implementation walks the sorted list one item at a time, so a block of
    # identical scores gets whatever order the input happened to have -- and
    # if the positives sort first by accident, a completely uninformative
    # score reports AP 1.0. That is not a hypothetical here: Stage 3's
    # composite_confidence is heavily tied (128 of 140 gradable decisions in
    # one 0.1-wide band, report S5.4), which is exactly the regime where the
    # naive version lies. Every positive inside a tie group is therefore
    # credited with the precision at the END of its group, which is the
    # order-independent expected value.
    ordered = sorted(clean, key=lambda p: -p[0])
    total = 0.0
    hits = 0
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        group = ordered[i:j + 1]
        group_pos = sum(1 for _, ok in group if ok)
        hits += group_pos
        if group_pos:
            precision_at_group_end = hits / (j + 1)
            total += group_pos * precision_at_group_end
        i = j + 1
    return round(total / n_pos, 4)


def null_model_report(pairs):
    """What you get for free: predict the base rate for every example.

    Returned alongside any model's numbers so a base-rate-predicting model is
    obvious rather than requiring a caveat. On the 2026-08-13 calibrator's
    n=140 / 14.3%-positive training set this returns accuracy 0.857 (predict
    the majority class) and ECE ~0.0 -- i.e. exactly the two numbers that
    made the fitted calibrator look good, produced by a model with no inputs.
    """
    clean = _clean_pairs(pairs)
    if not clean:
        return None
    n = len(clean)
    base = sum(1 for _, ok in clean if ok) / n
    baseline_pairs = [(base, ok) for _, ok in clean]
    return {
        "base_rate": round(base, 4),
        "majority_class_accuracy": round(max(base, 1 - base), 4),
        "ece": compute_ece(baseline_pairs),
        "brier": brier_score(baseline_pairs),
        "auroc": 0.5,          # constant score cannot rank; stated, not computed
        "average_precision": round(base, 4),
    }


# ==========================================================================
# Uncertainty -- note-level bootstrap
# ==========================================================================

def bootstrap_ci(clusters, statistic_fn, n_resamples=DEFAULT_RESAMPLES,
                 alpha=0.05, seed=DEFAULT_SEED):
    """Percentile bootstrap CI for `statistic_fn`, resampling CLUSTERS with
    replacement.

    `clusters` is a list of lists: one inner list per NOTE, containing that
    note's observations in whatever shape statistic_fn expects. `statistic_fn`
    receives the flattened concatenation of the resampled clusters.

    RESAMPLING NOTES, NOT ENTITIES, IS THE WHOLE POINT.
    docs/Evaluation_Criteria.md specifies note-level resampling. Entities
    inside one discharge note share a patient, a template, a set of
    abbreviations and an author -- they are strongly correlated, so treating
    them as independent draws produces intervals far narrower than the truth.
    On a 31-note corpus the effective sample size is closer to 31 than to
    1944, and an interval that pretends otherwise is worse than none: it
    converts "we do not know" into a confident, wrong claim.

    Returns None when there are fewer than 2 clusters (no variability to
    estimate) or the statistic is undefined on the observed data.
    """
    clusters = [c for c in clusters if c]
    if len(clusters) < 2:
        return None
    flat = [x for c in clusters for x in c]
    point = statistic_fn(flat)
    if point is None:
        return None

    rng = random.Random(seed)
    draws = []
    k = len(clusters)
    for _ in range(n_resamples):
        sample = []
        for _ in range(k):
            sample.extend(clusters[rng.randrange(k)])
        val = statistic_fn(sample)
        if val is not None:
            draws.append(val)
    if len(draws) < 2:
        return None

    draws.sort()
    lo_i = int((alpha / 2) * len(draws))
    hi_i = min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))
    return {
        "point": round(point, 4),
        "lo": round(draws[lo_i], 4),
        "hi": round(draws[hi_i], 4),
        "n_clusters": k,
        "n_resamples": len(draws),
        "alpha": alpha,
    }


def paired_bootstrap_diff(clusters_a, clusters_b, statistic_fn,
                          n_resamples=DEFAULT_RESAMPLES, alpha=0.05,
                          seed=DEFAULT_SEED):
    """CI on (statistic(A) - statistic(B)) under PAIRED note-level resampling:
    the same note indices are drawn for both systems on every resample.

    Built now, before it is needed, because docs/Evaluation_Criteria.md's
    success criterion -- "concept-level F1 at T2 meets or exceeds Clinical-T5's
    on the locked test set with non-overlapping bootstrap confidence
    intervals" -- is a paired comparison, and the report's S6 NET VALUE
    (CAUGHT_AND_FIXED minus INTRODUCED_ERROR = -17) is already one: the same
    entities scored two ways. Pairing removes between-note variance, which is
    the dominant term here; an unpaired comparison of two systems on the same
    notes throws away that reduction and will fail to detect real differences.

    clusters_a[i] and clusters_b[i] MUST describe the same note. Raises on a
    length mismatch rather than silently zipping to the shorter -- a silently
    truncated paired test is an invalid one.

    "diff excludes 0" is the readable form of the significance claim: if the
    returned interval does not contain 0, the difference survives note-level
    resampling.
    """
    if len(clusters_a) != len(clusters_b):
        raise ValueError(
            f"paired bootstrap needs aligned clusters: got {len(clusters_a)} "
            f"vs {len(clusters_b)}. Each index must be the same note.")
    k = len(clusters_a)
    if k < 2:
        return None

    def _diff(idxs):
        a = [x for i in idxs for x in clusters_a[i]]
        b = [x for i in idxs for x in clusters_b[i]]
        va, vb = statistic_fn(a), statistic_fn(b)
        if va is None or vb is None:
            return None
        return va - vb

    point = _diff(list(range(k)))
    if point is None:
        return None

    rng = random.Random(seed)
    draws = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(k) for _ in range(k)]
        val = _diff(idxs)
        if val is not None:
            draws.append(val)
    if len(draws) < 2:
        return None

    draws.sort()
    lo = draws[int((alpha / 2) * len(draws))]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return {
        "point": round(point, 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "n_clusters": k,
        "n_resamples": len(draws),
        "alpha": alpha,
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def accuracy(observations):
    """Fraction of truthy outcomes. Written as a named function rather than a
    lambda so it can be handed to bootstrap_ci() and appear by name in output.
    `observations` is a flat list of booleans (or (conf, correct) pairs, in
    which case the second element is used).
    """
    if not observations:
        return None
    vals = [o[1] if isinstance(o, (tuple, list)) else o for o in observations]
    return sum(1 for v in vals if v) / len(vals)


def group_by_note(rows, note_key="note_id"):
    """Rows -> list of per-note lists, in stable note order. The shape
    bootstrap_ci() expects. Stable ordering keeps a seeded bootstrap
    reproducible across runs on the same data.
    """
    buckets = {}
    for r in rows:
        buckets.setdefault(r[note_key], []).append(r)
    return [buckets[k] for k in sorted(buckets)]


def print_interpretation_block(pairs, accuracy_ci=None, indent="  "):
    """Prints the four things that turn a bare ECE into an interpretable
    result: the accuracy CI, discrimination (AUROC/AP), the null-model
    baseline, and equal-width-vs-equal-mass binning sensitivity.

    Lives here rather than in each evaluation script so all of them say the
    same thing in the same order -- the same anti-duplication reasoning that
    motivated this module. `pairs` is (confidence, correct).
    """
    ew = compute_ece_report(pairs, scheme="equal_width")
    em = compute_ece_report(pairs, scheme="equal_mass")
    if ew is None:
        print(f"{indent}(no scoreable pairs)")
        return
    null = null_model_report(pairs)
    au, ap_ = auroc(pairs), average_precision(pairs)

    if accuracy_ci:
        print(f"\n{indent}--- Accuracy, note-level 95% CI ---")
        print(f"{indent}  {format_ci(accuracy_ci)}  "
              f"({accuracy_ci['n_clusters']} notes resampled, "
              f"{accuracy_ci['n_resamples']} draws)")

    print(f"\n{indent}--- Discrimination (can this score RANK correct above incorrect?) ---")
    if au is None:
        print(f"{indent}  AUROC             = undefined (only one class present)")
    else:
        note = ""
        if au < 0.5:
            note = "   <-- BELOW 0.5: INVERTED, not merely uninformative"
        elif abs(au - 0.5) < 0.03:
            note = "   <-- ~0.5: no discriminative signal"
        print(f"{indent}  AUROC             = {au}{note}")
    print(f"{indent}  Average precision = {ap_}   "
          f"(base rate {null['base_rate']} is the no-signal floor)")

    print(f"\n{indent}--- Null model (predict the base rate for every example) ---")
    print(f"{indent}  majority-class accuracy = {null['majority_class_accuracy']}")
    print(f"{indent}  ECE                     = {null['ece']}   "
          f"<-- an input-free model is well calibrated; low ECE alone proves nothing")
    print(f"{indent}  Brier                   = {null['brier']}   "
          f"(proper score -- the null model cannot game this one)")

    print(f"\n{indent}--- Binning sensitivity ---")
    print(f"{indent}  equal-width ECE = {ew['ece']}  "
          f"({ew['n_nonempty_bins']}/{ew['n_bins_requested']} bins non-empty)")
    print(f"{indent}  equal-mass  ECE = {em['ece']}  ({em['n_nonempty_bins']} quantile bins)")
    print(f"{indent}  Brier (model)   = {ew['brier']}  vs null {null['brier']}")
    if ew["n_nonempty_bins"] <= 2:
        print(f"{indent}  !! the equal-width figure rests on {ew['n_nonempty_bins']} "
              f"non-empty bin(s); prefer the equal-mass number.")


def format_ci(ci, pct=True):
    """'52.71% [48.10, 57.30]' -- one place, so every script's CI looks the
    same and no caller invents its own bracket convention."""
    if not ci:
        return "n/a"
    mul = 100.0 if pct else 1.0
    suffix = "%" if pct else ""
    return (f"{ci['point'] * mul:.2f}{suffix} "
            f"[{ci['lo'] * mul:.2f}, {ci['hi'] * mul:.2f}]")


# ==========================================================================
# Stub tests -- run: python3 evaluation/metrics.py
# ==========================================================================

def _selftest():
    ok = 0
    fail = []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # 1. Perfect calibration -> ECE 0.
    perfect = [(1.0, True)] * 50 + [(0.0, False)] * 50
    check("perfect calibration ece==0", compute_ece(perfect) == 0.0)

    # 2. Maximally miscalibrated -> ECE 1.
    worst = [(1.0, False)] * 50 + [(0.0, True)] * 50
    check("worst calibration ece==1", compute_ece(worst) == 1.0)

    # 3. Reproduces cal_eval.py's original arithmetic on the report's own
    #    Stage 3 shape: 12 points at ~0.889/33.3% + 128 at ~0.9184/12.5%.
    s3 = ([(0.8892, True)] * 4 + [(0.8892, False)] * 8
          + [(0.9184, True)] * 16 + [(0.9184, False)] * 112)
    r = compute_ece_report(s3)
    check("stage3 ece ~0.773", abs(r["ece"] - 0.773) < 0.005)
    check("stage3 single nonempty bin flagged", r["n_nonempty_bins"] == 2)

    # 4. Equal-mass binning spreads the same data across more bins.
    spread = [(0.90 + i * 0.0001, i % 8 == 0) for i in range(200)]
    ew = compute_ece_report(spread, scheme="equal_width")
    em = compute_ece_report(spread, scheme="equal_mass")
    check("equal_width degenerates to 1 bin", ew["n_nonempty_bins"] == 1)
    check("equal_mass uses many bins", em["n_nonempty_bins"] >= 8)

    # 5. None confidences are dropped, not coerced to 0.
    check("None conf dropped", compute_ece_report([(None, True), (1.0, True)])["n"] == 1)

    # 6. AUROC: perfect, inverted, uninformative, undefined.
    check("auroc perfect", auroc([(0.9, True), (0.8, True), (0.2, False), (0.1, False)]) == 1.0)
    check("auroc inverted", auroc([(0.1, True), (0.2, True), (0.8, False), (0.9, False)]) == 0.0)
    check("auroc ties==0.5", auroc([(0.5, True), (0.5, False)]) == 0.5)
    check("auroc one class -> None", auroc([(0.9, True), (0.8, True)]) is None)

    # 7. AP baseline equals the base rate on an uninformative score.
    flat = [(0.5, True)] * 2 + [(0.5, False)] * 8
    check("ap uninformative ~ base rate", abs(average_precision(flat) - 0.2) < 0.01)

    # 8. Null model reproduces the artefact the report flagged: 14.3% positive
    #    -> 85.7% "accuracy" and near-zero ECE, from a model with no inputs.
    nm = null_model_report([(0.9, True)] * 20 + [(0.9, False)] * 120)
    check("null model accuracy 0.857", abs(nm["majority_class_accuracy"] - 0.8571) < 0.001)
    check("null model ece ~0", nm["ece"] < 0.001)
    check("null model auroc 0.5", nm["auroc"] == 0.5)

    # 9. Brier is NOT fooled by the null model the way ECE is.
    check("brier penalises null model", nm["brier"] > 0.1)

    # 10. Bootstrap: CI brackets the point estimate and widens with clustering.
    clusters = [[True] * 8 + [False] * 2 for _ in range(20)]
    ci = bootstrap_ci(clusters, accuracy, n_resamples=300)
    check("ci brackets point", ci["lo"] <= ci["point"] <= ci["hi"])
    check("ci n_clusters", ci["n_clusters"] == 20)
    check("ci needs >=2 clusters", bootstrap_ci([[True]], accuracy) is None)

    # 11. A clustered signal produces a WIDER interval than the same
    #     observations spread evenly -- the whole reason for note-level
    #     resampling. All-correct notes vs all-wrong notes is the extreme case.
    clumped = [[True] * 10 for _ in range(10)] + [[False] * 10 for _ in range(10)]
    even = [[True] * 5 + [False] * 5 for _ in range(20)]
    ci_clumped = bootstrap_ci(clumped, accuracy, n_resamples=400)
    ci_even = bootstrap_ci(even, accuracy, n_resamples=400)
    check("clustered data -> wider CI",
          (ci_clumped["hi"] - ci_clumped["lo"]) > (ci_even["hi"] - ci_even["lo"]))

    # 12. Paired bootstrap: identical systems -> interval containing 0;
    #     clearly different systems -> interval excluding it.
    a = [[True] * 9 + [False] for _ in range(20)]
    b = [[True] + [False] * 9 for _ in range(20)]
    same = paired_bootstrap_diff(a, list(a), accuracy, n_resamples=300)
    diff = paired_bootstrap_diff(a, b, accuracy, n_resamples=300)
    check("paired identical -> 0 in CI", same["point"] == 0.0 and not same["excludes_zero"])
    check("paired different -> excludes 0", diff["excludes_zero"])
    try:
        paired_bootstrap_diff([[True]], [[True], [False]], accuracy)
        check("paired length mismatch raises", False)
    except ValueError:
        check("paired length mismatch raises", True)

    # 13. group_by_note / format_ci.
    grouped = group_by_note([{"note_id": "b", "x": 1}, {"note_id": "a", "x": 2},
                             {"note_id": "a", "x": 3}])
    check("group_by_note", len(grouped) == 2 and len(grouped[0]) == 2)
    check("format_ci", format_ci({"point": 0.5271, "lo": 0.481, "hi": 0.573})
          == "52.71% [48.10, 57.30]")

    # 14. Empty input is None everywhere, never a crash or a fake 0.
    check("empty -> None", all(f([]) is None for f in
                               (compute_ece, brier_score, auroc,
                                average_precision, null_model_report, accuracy)))

    # 15. Unknown scheme raises rather than silently falling back.
    try:
        compute_ece_report([(0.5, True)], scheme="nonsense")
        check("bad scheme raises", False)
    except ValueError:
        check("bad scheme raises", True)

    print(f"metrics.py self-test: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
