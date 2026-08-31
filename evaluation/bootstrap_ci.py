"""evaluation/bootstrap_ci.py -- 2026-08-31: note-level bootstrap confidence
intervals, the actual method `docs/Evaluation_Criteria.md` specifies
("Bootstrap confidence intervals are resampled at the note level") --
distinct from the Wilson score interval already built (`docs/Code_Reference_
Stages_And_Metrics.md` S14), which treats every graded ENTITY as an
independent Bernoulli trial. That's the wrong independence assumption for
this project's data: entities cluster within notes (a clinical note's own
vocabulary/complexity is shared across every entity in it), so resampling
entities directly understates true uncertainty. Resampling NOTES with
replacement, then pooling every entity belonging to each resampled note
(duplicates included when a note is drawn more than once), is the standard
fix and matches this project's own repeated "note-disjoint" discipline
elsewhere (calibrator train/val splits, the leakage guard in
ConsensusCalibrator.load()).

WHAT THIS DOES NOT CHANGE: the point estimates themselves are identical to
what's already reported (same records, same metric function) -- only the
interval around them is more honest. A wider bootstrap interval than the
Wilson interval it replaces is the EXPECTED, correct outcome for a
population dominated by a few large notes, not a bug to explain away.
"""
import collections
import random


def bootstrap_note_level_ci(records: list, metric_fn, n_boot: int = 2000,
                            seed: int = 42, alpha: float = 0.05):
    """`records`: list of dicts, each carrying a `note_id` key (plus
    whatever `metric_fn` needs -- e.g. `correct`). `metric_fn(records) ->
    float|None` computes the point statistic for a (possibly resampled)
    record list; returns None for a pooled draw with an empty/ungradable
    result (excluded from the bootstrap distribution, not treated as 0).

    Resamples NOTE IDs (not records) with replacement, `n_boot` times;
    each resample pools every record belonging to each drawn note_id
    (including its full duplicate weight if a note_id is drawn more than
    once). Returns {"point": float, "ci_lo": float, "ci_hi": float,
    "n_notes": int, "n_records": int, "n_boot_usable": int}.

    Deterministic given `seed` -- same discipline as every other
    stochastic evaluation in this codebase (TransE/RotatE training,
    negative sampling) using a fixed seed for reproducibility.
    """
    by_note = collections.defaultdict(list)
    for r in records:
        by_note[r["note_id"]].append(r)
    note_ids = list(by_note.keys())

    point = metric_fn(records)
    rng = random.Random(seed)
    boot_estimates = []
    for _ in range(n_boot):
        sample_notes = rng.choices(note_ids, k=len(note_ids))
        pooled = []
        for nid in sample_notes:
            pooled.extend(by_note[nid])
        est = metric_fn(pooled)
        if est is not None:
            boot_estimates.append(est)
    boot_estimates.sort()

    n = len(boot_estimates)
    if n == 0:
        return {"point": point, "ci_lo": None, "ci_hi": None,
               "n_notes": len(note_ids), "n_records": len(records), "n_boot_usable": 0}
    lo_idx = int((alpha / 2) * n)
    hi_idx = min(n - 1, int((1 - alpha / 2) * n))
    return {"point": point, "ci_lo": boot_estimates[lo_idx], "ci_hi": boot_estimates[hi_idx],
           "n_notes": len(note_ids), "n_records": len(records), "n_boot_usable": n}


def precision_metric(records):
    """Standard 'fraction correct' point statistic -- None on an empty pool
    (a resample can, in principle, draw zero of a rare tier's notes)."""
    if not records:
        return None
    return sum(1 for r in records if r["correct"]) / len(records)


def false_deflection_metric(records):
    """1 - precision_metric -- the false-deflection-rate proxy (docs/
    Code_Reference_Stages_And_Metrics.md S15), computed directly rather
    than derived by complementing an already-computed CI, so the bootstrap
    resampling is genuinely re-run on this statistic (not assumed to
    transform losslessly -- it does, since complementation is monotonic,
    but computing it directly avoids relying on that assumption silently)."""
    p = precision_metric(records)
    return None if p is None else 1 - p
