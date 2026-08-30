"""evaluation/kg3_feature_ablation.py -- 2026-08-30.

Isolates kg3_confirmation_count's OWN marginal contribution to precision/
AUROC, separate from the corpus-growth effect (114->144 notes) that a
naive "new AUROC vs. old baseline" comparison conflates with it (see
scripts/retrain_calibrator_full_corpus.py, and
docs/ConsensusCalibrator_Technical_Reference.md §17.4's warning about
exactly this). Same 144-note population, same note-disjoint split, same
everything else -- the ONLY difference between the two fits below is
whether the 17th feature (kg3_confirmation_count) is real or zeroed.

Reuses evaluation/tier_gate_cal_eval.py's build_labeled_examples()/
split_by_note()/fit_and_report()/threshold_sweep() rather than
re-deriving them -- same reuse discipline
scripts/retrain_calibrator_full_corpus.py already established.

Diagnostic only: does not touch the saved production .pkl. See
docs/ConsensusCalibrator_Technical_Reference.md §17.5 and
docs/FINAL_RESULTS_Single_Source_Of_Truth.md §9.2 for the results this
script produced and how they were interpreted (including the §17.7/§9.4
circularity caveat -- KG3's current population is 100% gold-simulated,
not real human review, so this measures the feature's behavior given
today's KG3 contents, not against independent real-world evidence).
"""
import collections
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402
from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from evaluation.tier_gate_cal_eval import (  # noqa: E402
    build_labeled_examples, split_by_note, fit_and_report, FEATURE_NAMES,
    threshold_sweep)
from scripts.score_gold_recall import load_gold  # noqa: E402
from src.kg3_ingestion import get_memgraph_driver  # noqa: E402

DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
PRODUCTION_THRESHOLD = 0.72  # src.mollm_tier_gate.CALIBRATED_AUTO_THRESHOLD


def main():
    conn = connect_with_retry(DB_PATH, read_only=True, max_wait_seconds=300)
    vocab = VocabularyRetriever(conn)

    # Best-effort reachability check, matching the same degrade-gracefully
    # pattern scripts/run_stage3_tier_gate.py and
    # scripts/retrain_calibrator_full_corpus.py already use -- but for THIS
    # script specifically, an unreachable Memgraph makes the ablation
    # meaningless (both fits would train on an all-zero kg3_confirmation_count
    # column, and the "isolated effect" would trivially be zero), so it
    # exits rather than silently producing a null result.
    try:
        kg3_driver = get_memgraph_driver()
        with kg3_driver.session() as s:
            s.run("RETURN 1")
        print("Memgraph reachable\n")
    except Exception as exc:
        print(f"Memgraph NOT reachable ({exc}) -- this ablation is meaningless "
              f"without real KG3 data (both fits would see an all-zero "
              f"kg3_confirmation_count column). Aborting rather than reporting "
              f"a trivial zero delta.")
        conn.close()
        return 1

    all_notes = sorted(r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM mollm_tier_gate_decisions WHERE tier = 'TIER_4_ENSEMBLE_SPLIT'"
    ).fetchall())
    print(f"{len(all_notes)} notes have TIER_4_ENSEMBLE_SPLIT decisions\n")

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, all_notes)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    examples = build_labeled_examples(conn, vocab, gold_by_note, note_ids=all_notes,
                                      kg3_driver=kg3_driver)
    train, val, train_notes, val_notes = split_by_note(examples)
    print(f"split: {len(train_notes)} train notes ({len(train)}), "
          f"{len(val_notes)} val notes ({len(val)})\n")

    kg3_idx = FEATURE_NAMES.index("kg3_confirmation_count")

    n_nonzero_train = sum(1 for e in train if e["vector"][kg3_idx] > 0)
    n_nonzero_val = sum(1 for e in val if e["vector"][kg3_idx] > 0)
    print(f"kg3_confirmation_count > 0: {n_nonzero_train}/{len(train)} train "
          f"({n_nonzero_train/len(train)*100:.1f}%), "
          f"{n_nonzero_val}/{len(val)} val ({n_nonzero_val/len(val)*100:.1f}%)\n")

    print("#" * 78)
    print("# FULL (17 features, real kg3_confirmation_count)")
    print("#" * 78)
    cal_full, val_full, auc_full = fit_and_report(
        train, val, train_notes, ablate_indices=(), label="FULL (17 features, real KG3)",
        save_path=None, respect_hard_traps=True)

    print("\n" + "#" * 78)
    print("# ABLATED (kg3_confirmation_count zeroed -- everything else identical)")
    print("#" * 78)
    cal_abl, val_abl, auc_abl = fit_and_report(
        train, val, train_notes, ablate_indices=(kg3_idx,),
        label="ABLATED (kg3_confirmation_count zeroed)",
        save_path=None, respect_hard_traps=True)

    print("\n" + "=" * 78)
    print(f"HEAD-TO-HEAD AT THE PRODUCTION THRESHOLD ({PRODUCTION_THRESHOLD})")
    print("=" * 78)
    for name, val_scored in [("FULL (real KG3)", val_full), ("ABLATED (KG3 zeroed)", val_abl)]:
        row = threshold_sweep(val_scored, [PRODUCTION_THRESHOLD], respect_hard_traps=True)[0]
        print(f"  {name:24s}  coverage={row['coverage_pct']}%  "
              f"precision={row['precision_pct']}%  n_promoted={row['n_promoted']}")

    print(f"\nAUROC: FULL={auc_full}  ABLATED={auc_abl}  "
          f"delta={None if auc_full is None or auc_abl is None else round(auc_full - auc_abl, 4)}")

    # Which specific val examples change tier (promoted under one, not the
    # other) at the production threshold -- concrete cases, not just
    # aggregate deltas. This is what let the interpretation in the docs
    # above go past "the number went up" to "here's what actually flipped,
    # and whether it was correct."
    full_by_id = {(e["note_id"], e["text"]): e for e in val_full}
    abl_by_id = {(e["note_id"], e["text"]): e for e in val_abl}
    print("\n--- val examples where FULL vs ABLATED disagree on promotion "
          f"at {PRODUCTION_THRESHOLD} ---")
    n_diff = 0
    for key, e_full in full_by_id.items():
        e_abl = abl_by_id.get(key)
        if e_abl is None:
            continue
        full_promoted = e_full["score"] is not None and e_full["score"] >= PRODUCTION_THRESHOLD \
            and not e_full.get("is_trapped")
        abl_promoted = e_abl["score"] is not None and e_abl["score"] >= PRODUCTION_THRESHOLD \
            and not e_abl.get("is_trapped")
        if full_promoted != abl_promoted:
            n_diff += 1
            print(f"  [{key[0]}] {key[1]!r} label={e_full['label']} "
                  f"full_score={e_full['score']} abl_score={e_abl['score']} "
                  f"kg3_count={e_full.get('kg3_confirmation_count', 0)} "
                  f"-> {'promoted only WITH kg3' if full_promoted else 'promoted only WITHOUT kg3'}")
    if n_diff == 0:
        print("  (none -- identical promotion set at this threshold)")

    conn.close()
    kg3_driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
