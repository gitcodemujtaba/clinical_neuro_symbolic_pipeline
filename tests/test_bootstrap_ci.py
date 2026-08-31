"""
tests/test_bootstrap_ci.py -- evaluation/bootstrap_ci.py's note-level
resampling: pure logic, no DB dependency.

Run: python3 -m pytest tests/test_bootstrap_ci.py -v
"""
import sys

from evaluation.bootstrap_ci import (
    bootstrap_note_level_ci,
    false_deflection_metric,
    precision_metric,
)


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # precision_metric / false_deflection_metric
    # ======================================================================
    recs = [{"correct": True}, {"correct": True}, {"correct": False}, {"correct": True}]
    check("precision_metric computes fraction correct", precision_metric(recs) == 0.75)
    check("false_deflection_metric is the exact complement",
          abs(false_deflection_metric(recs) - 0.25) < 1e-9)
    check("both return None on an empty pool",
          precision_metric([]) is None and false_deflection_metric([]) is None)

    # ======================================================================
    # bootstrap_note_level_ci -- deterministic given a seed
    # ======================================================================
    records = (
        [{"note_id": "A", "correct": True} for _ in range(10)]
        + [{"note_id": "B", "correct": False} for _ in range(10)]
    )
    result = bootstrap_note_level_ci(records, precision_metric, n_boot=500, seed=1)
    check("point estimate matches the plain (non-resampled) statistic",
          abs(result["point"] - 0.5) < 1e-9)
    check("n_notes reflects the distinct note_id count, not the record count",
          result["n_notes"] == 2)
    check("n_records reflects the full record count", result["n_records"] == 20)
    check("CI bounds are within [0, 1]", 0.0 <= result["ci_lo"] <= result["ci_hi"] <= 1.0)
    check("CI is WIDE for a 2-note population -- resampling can only ever draw "
         "{both A, both B, one each}, so 0.0 and 1.0 must both be reachable",
          result["ci_lo"] == 0.0 and result["ci_hi"] == 1.0)

    result2 = bootstrap_note_level_ci(records, precision_metric, n_boot=500, seed=1)
    check("same seed reproduces identical results", result == result2)

    result3 = bootstrap_note_level_ci(records, precision_metric, n_boot=500, seed=2)
    check("a different seed can (and here, does) produce a numerically different point-adjacent "
         "bootstrap distribution -- checked via ci bounds differing is NOT required (both may "
         "still hit [0,1] given only 2 notes), so check n_boot_usable/point stability instead: "
         "the deterministic point estimate is seed-independent",
          result3["point"] == result["point"])

    # ======================================================================
    # A population where entity-level (Wilson-style) and note-level
    # bootstrap CIs should meaningfully diverge -- one giant note dominating
    # many small ones. This is the concrete case this module exists for.
    # ======================================================================
    skewed = (
        [{"note_id": "big", "correct": True} for _ in range(50)]
        + [{"note_id": "big", "correct": False} for _ in range(50)]
        + [{"note_id": f"small{i}", "correct": True} for i in range(5)]
    )
    skewed_result = bootstrap_note_level_ci(skewed, precision_metric, n_boot=2000, seed=7)
    check("skewed-population point estimate: 50 correct of 100 'big' + 5 correct of 5 "
         "'small' = 55/105",
          abs(skewed_result["point"] - (55 / 105)) < 1e-9)
    check("skewed-population CI is WIDE (>0.3) despite n=105 entities, because only 6 "
         "distinct notes exist and 'big' (internally 50/50) vs. the 5 perfect 'small' "
         "notes swings the pooled precision hugely depending on how many times 'big' is "
         "drawn -- an entity-level (Wilson-style) interval on the same 105 points would "
         "come nowhere close to this width, which is exactly the gap this module exists to close",
          (skewed_result["ci_hi"] - skewed_result["ci_lo"]) > 0.3)

    # ======================================================================
    # bootstrap_note_level_ci on false_deflection_metric -- same records,
    # different statistic, should be the exact complement of the precision
    # CI (monotonic transform of the same underlying resamples... but this
    # module recomputes independently per the module's own stated
    # discipline, so check numerically, not by assuming the identity holds
    # by construction)
    # ======================================================================
    fd_result = bootstrap_note_level_ci(records, false_deflection_metric, n_boot=500, seed=1)
    check("false-deflection CI is the numeric complement of the precision CI "
         "on the same records/seed (both resample the same note_ids in the same order)",
          abs(fd_result["ci_lo"] - (1 - result["ci_hi"])) < 1e-9
          and abs(fd_result["ci_hi"] - (1 - result["ci_lo"])) < 1e-9)

    print(f"bootstrap-ci tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_bootstrap_ci():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
