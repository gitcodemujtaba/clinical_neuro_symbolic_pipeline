"""evaluation/run_bootstrap_ci.py -- 2026-08-31: real note-level bootstrap
CIs for the headline AUTO-tier-precision figure (and its direct complement,
the false-deflection-rate proxy, docs/Code_Reference_Stages_And_Metrics.md
S15) across the same three populations the Wilson-interval section (S14)
already reports, so the two can be directly compared.

Read-only. No LLM calls, no pipeline run -- reuses evaluation.tier_gate_
grading.grade_by_tier() (already-stored mollm_tier_gate_decisions rows) and
evaluation.bootstrap_ci.bootstrap_note_level_ci() (pure resampling, no DB).

Run: python3 -m evaluation.run_bootstrap_ci
"""
import json
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.mollm_tier_gate import AUTO_TIERS  # noqa: E402
from evaluation.bootstrap_ci import (  # noqa: E402
    bootstrap_note_level_ci, false_deflection_metric, precision_metric)
from evaluation.tier_gate_grading import grade_by_tier  # noqa: E402
from ui.components.fresh10_notes import FRESH10_NOTE_IDS  # noqa: E402

# NOT evaluation.grade_fresh5_by_tier.NOTE_IDS -- that's a DIFFERENT, older
# (2026-08-17) 5-note calibrator-validation batch that happens to share the
# name "fresh5". Confirmed live (2026-08-31): those 5 notes currently carry
# only 125 tier-gate decisions from a single 2026-08-20 run, vs. the real
# "Fresh-5 (2026-08-30)" headline population's 373 decisions
# (docs/FINAL_RESULTS_Single_Source_Of_Truth.md S10) -- reusing the wrong
# list silently would have reported a bootstrap CI around the wrong 79.2%
# point estimate instead of the real, documented 92.1%. These 5 note IDs
# are quoted directly from S10's own text, not re-derived.
FRESH5_NOTE_IDS = [
    "13397956-DS-5", "17739994-DS-31", "16410990-DS-12",
    "16795604-DS-17", "17309807-DS-20",
]


def auto_tier_records(conn, note_ids):
    """Pools clean-span-gradable records across every AUTO_TIERS tier for
    `note_ids` -- the exact population S3/S14's "AUTO-tier precision"
    already measures, via grade_by_tier() (correct, current AUTO_TIERS
    import -- not the stale hardcoded copy some older scripts carry)."""
    report = grade_by_tier(conn, note_ids, tiers=list(AUTO_TIERS))
    records = []
    for tier in AUTO_TIERS:
        records.extend(report.get(tier, {}).get("clean", {}).get("records", []))
    return records


def report_population(name, records):
    prec = bootstrap_note_level_ci(records, precision_metric, n_boot=2000, seed=42)
    fdr = bootstrap_note_level_ci(records, false_deflection_metric, n_boot=2000, seed=42)
    print(f"\n=== {name} ===")
    print(f"  n_notes={prec['n_notes']}  n_gradable={prec['n_records']}")
    if prec["ci_lo"] is None:
        print("  (no gradable records)")
        return None
    print(f"  AUTO-tier precision: {prec['point']*100:.1f}%  "
         f"bootstrap 95% CI [{prec['ci_lo']*100:.1f}%, {prec['ci_hi']*100:.1f}%]  "
         f"width={(prec['ci_hi']-prec['ci_lo'])*100:.1f}pp")
    print(f"  False deflection rate: {fdr['point']*100:.1f}%  "
         f"bootstrap 95% CI [{fdr['ci_lo']*100:.1f}%, {fdr['ci_hi']*100:.1f}%]  "
         f"width={(fdr['ci_hi']-fdr['ci_lo'])*100:.1f}pp")
    return {"precision": prec, "false_deflection": fdr}


def main():
    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb",
                              read_only=True, max_wait_seconds=120)

    corpus_note_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test=TRUE").fetchall()]
    print(f"corpus-wide: {len(corpus_note_ids)} distinct is_test notes "
         f"(the '144 notes' figure elsewhere in this project's docs is stale -- "
         f"the corpus has grown since; this run uses the real, current count)")

    populations = {
        "corpus-wide (current, real count -- see note above)": corpus_note_ids,
        "fresh-10": FRESH10_NOTE_IDS,
        "fresh-5": FRESH5_NOTE_IDS,
    }

    results = {}
    for name, note_ids in populations.items():
        records = auto_tier_records(conn, note_ids)
        results[name] = report_population(name, records)

    conn.close()

    out_path = f"{PROJECT_DIR}/logs/bootstrap_ci_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o)
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    main()
