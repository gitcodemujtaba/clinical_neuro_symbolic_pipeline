"""evaluation/kg3_calibrator_previous_regrade.py -- 2026-08-30.

"Previous" half of the requested before/after: re-grades the notes ALREADY
processed through Stage 3, comparing the tier/AUTO-coverage each entity
actually landed at (as currently stored -- reflects whatever calibrator,
if any, was live in production when that entity was scored) against what
it would land at if re-scored TODAY with the current production
calibrator (144-note/17-feature, kg3_confirmation_count included).

No new LLM calls: model_results are already stored in
mollm_tier_gate_decisions.models from when the ensemble actually ran.
Only the calibrator-consultation step is replayed.

METHODOLOGY, stated precisely so it's reproducible:

- Restricted to the calibrator's own genuinely-clean notes (NOT in its
  `training_note_ids`) -- scoring notes the model was trained on would be
  leakage, and ConsensusCalibrator.load(..., scoring_note_ids=...)'s own
  guard would (correctly) refuse the whole model rather than silently
  report an inflated number. 39 of the 144 total notes qualify as of this
  writing; 2 of the locked fresh-10 notes (ui/components/fresh10_notes.py)
  are among them -- reported as its own, explicitly tiny (n=2), slice.
- Only entities CURRENTLY at TIER_4_ENSEMBLE_SPLIT are re-scored for
  promotion to TIER_1B_CALIBRATED_AUTO_VALIDATED (the only route into
  `AUTO_TIERS` a calibrator promotion can take -- TIER_2B stays outside
  `AUTO_TIERS` by design, see docs/ConsensusCalibrator_Technical_Reference.md
  §2, so a TIER_2->TIER_2B promotion is tracked and reported separately,
  not folded into "new AUTO-tier precision/deflection rate"). Entities
  already in AUTO_TIERS are left exactly as stored -- nothing un-promotes
  an already-finalized decision, matching this project's "can only
  promote, never demote" discipline everywhere else a calibrator is used.
- `AUTO_TIERS` is imported directly from src.mollm_tier_gate, NOT
  redefined locally -- two evaluation scripts (grade_overnight_corpus_run.py,
  grade_allergy_shadow_run.py) carry their own stale hardcoded copy that
  is missing TIER_1B and wrongly includes TIER_2_AUTO_RESOLVED, the same
  drift bug already found and fixed once in src/kg3_ingestion.py. Not
  fixed in those two files as part of this script; noted as a real,
  separate, still-open finding.
- Grading (correct/incorrect vs. gold): identical clean-span + SNOMED-
  crosswalk methodology as evaluation/grade_overnight_corpus_run.py's
  grade_population() (reused directly, not re-derived): single overlapping
  gold entity, predicted span not narrower than gold, exact SNOMED code
  match after crosswalk (or a curated KNOWN_GOLD_ERRORS override).
- Linked precision/recall/F1 (the corpus-wide headline in
  docs/FINAL_RESULTS_Single_Source_Of_Truth.md §2) are UNCHANGED by any
  of this and are not recomputed here: the calibrator only ever changes
  ROUTING (AUTO vs. HITL), never WHICH candidate is chosen -- an entity's
  final_candidate_index is identical whether it lands at TIER_4 (HITL) or
  gets promoted to TIER_1B (AUTO). The only metrics a calibrator swap can
  move are AUTO-tier precision and deflection rate, which is what this
  script reports.
"""
import collections
import json
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import duckdb  # noqa: E402

from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from evaluation.grade_overnight_corpus_run import (  # noqa: E402
    plurality_candidate_index, grade_population, KNOWN_GOLD_ERRORS)
from scripts.score_gold_recall import load_gold  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402
from src.kg3_ingestion import get_memgraph_driver  # noqa: E402
from src.mollm_tier_gate import (  # noqa: E402
    AUTO_TIERS, CALIBRATED_AUTO_THRESHOLD, TIER_4_ENSEMBLE_SPLIT,
    TIER_2_AUTO_RESOLVED, TIER_1B_CALIBRATED_AUTO_VALIDATED,
    TIER_2B_CALIBRATED_AUTO_RESOLVED, _score_with_calibrator)
from src.mollm_tier_calibrator import ConsensusCalibrator, DEFAULT_MODEL_PATH
from ui.components.fresh10_notes import FRESH10_NOTE_IDS

DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"


def pct(n, d):
    return f"{n/d*100:.1f}% ({n}/{d})" if d else "n/a"


def load_decisions(conn, note_ids):
    ph = ",".join("?" * len(note_ids))
    rows = conn.execute(f"""
        SELECT d.entity_id, d.note_id, d.tier, d.mollm_routing_decision,
               d.final_candidate_index, d.models,
               e.original_text, e.entity_label, e.orig_start, e.orig_end,
               n.candidates, n.match_tier, n.is_ambiguous, n.domain_conflict,
               n.normalized_from
        FROM mollm_tier_gate_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.note_id IN ({ph})
    """, note_ids).fetchall()
    cols = [c[0] for c in conn.description]
    return [dict(zip(cols, r)) for r in rows]


def regrade(note_ids, label, conn, vocab, gold_by_note, calibrator, kg3_driver):
    print("\n" + "=" * 78)
    print(f"{label}  ({len(note_ids)} notes)")
    print("=" * 78)

    decisions = load_decisions(conn, note_ids)
    total_decisions = len(decisions)
    print(f"total Stage-3 decisions: {total_decisions}")

    # --- OLD: exactly as currently stored ---
    old_auto = [d for d in decisions if d["tier"] in AUTO_TIERS]
    old_raw, old_clean, old_skip = grade_population(
        old_auto, gold_by_note, vocab, lambda d: (d["final_candidate_index"], {}))
    old_correct = sum(1 for r in old_clean if r["correct"])
    old_gradable = len(old_clean)

    print(f"\nOLD (as currently stored):")
    print(f"  AUTO_TIERS decisions: {len(old_auto)}")
    print(f"  deflection rate: {pct(len(old_auto), total_decisions)}")
    print(f"  AUTO-tier precision: {pct(old_correct, old_gradable)}")

    # --- NEW: re-score currently-HITL TIER_4 (-> TIER_1B) and TIER_2 (-> TIER_2B) ---
    tier4 = [d for d in decisions if d["tier"] == TIER_4_ENSEMBLE_SPLIT]
    tier2 = [d for d in decisions if d["tier"] == TIER_2_AUTO_RESOLVED]

    newly_promoted_1b = []  # (decision, candidate_index) -- counts toward AUTO_TIERS
    newly_promoted_2b = []  # tracked separately -- NOT in AUTO_TIERS by design

    for d in tier4:
        idx, top_verdict, vote_counts = plurality_candidate_index(d["models"])
        if idx is None:
            continue
        candidates = d["candidates"]
        if isinstance(candidates, str):
            candidates = json.loads(candidates)
        entity = {
            "original_text": d["original_text"], "candidates": candidates,
            "match_tier": d["match_tier"], "is_ambiguous": d["is_ambiguous"],
            "domain_conflict": d["domain_conflict"], "normalized_from": d["normalized_from"],
            "expansion_ambiguous": False,
        }
        models = d["models"]
        if isinstance(models, str):
            models = json.loads(models)
        result = _score_with_calibrator(entity, models, idx, calibrator, conn, None,
                                        kg3_driver, None)
        if result["trapped"]:
            continue
        score = result["calibrated_score"]
        if score is not None and score >= CALIBRATED_AUTO_THRESHOLD:
            newly_promoted_1b.append((d, idx))

    for d in tier2:
        idx = d["final_candidate_index"]  # already the unanimous re-rank target
        if idx is None:
            continue
        candidates = d["candidates"]
        if isinstance(candidates, str):
            candidates = json.loads(candidates)
        entity = {
            "original_text": d["original_text"], "candidates": candidates,
            "match_tier": d["match_tier"], "is_ambiguous": d["is_ambiguous"],
            "domain_conflict": d["domain_conflict"], "normalized_from": d["normalized_from"],
            "expansion_ambiguous": False,
        }
        models = d["models"]
        if isinstance(models, str):
            models = json.loads(models)
        result = _score_with_calibrator(entity, models, idx, calibrator, conn, None,
                                        kg3_driver, None)
        if result["trapped"]:
            continue
        score = result["calibrated_score"]
        if score is not None and score >= CALIBRATED_AUTO_THRESHOLD:
            newly_promoted_2b.append((d, idx))

    def grade_promoted(promoted_list):
        raw, clean, skip = grade_population(
            [d for d, _ in promoted_list], gold_by_note, vocab,
            lambda d, _idx_map={id(d): idx for d, idx in promoted_list}:
                (_idx_map[id(d)], {}))
        return clean

    new1b_clean = grade_promoted(newly_promoted_1b)
    new2b_clean = grade_promoted(newly_promoted_2b)
    new1b_correct = sum(1 for r in new1b_clean if r["correct"])
    new2b_correct = sum(1 for r in new2b_clean if r["correct"])

    new_auto_count = len(old_auto) + len(newly_promoted_1b)
    new_correct = old_correct + new1b_correct
    new_gradable = old_gradable + len(new1b_clean)

    print(f"\nNEW (re-scored with current production calibrator + live KG3):")
    print(f"  newly promoted TIER_4 -> TIER_1B: {len(newly_promoted_1b)} "
          f"(gradable {len(new1b_clean)}, correct {new1b_correct}, "
          f"precision {pct(new1b_correct, len(new1b_clean))})")
    print(f"  AUTO_TIERS decisions: {new_auto_count}")
    print(f"  deflection rate: {pct(new_auto_count, total_decisions)}")
    print(f"  AUTO-tier precision: {pct(new_correct, new_gradable)}")
    print(f"  [separately tracked, NOT counted toward AUTO_TIERS -- TIER_2B pending "
          f"shadow validation] newly promoted TIER_2 -> TIER_2B: "
          f"{len(newly_promoted_2b)} (gradable {len(new2b_clean)}, correct "
          f"{new2b_correct}, precision {pct(new2b_correct, len(new2b_clean))})")

    print(f"\nDELTA:")
    print(f"  deflection rate: {len(old_auto)}/{total_decisions} -> "
          f"{new_auto_count}/{total_decisions}  "
          f"({(new_auto_count-len(old_auto))/total_decisions*100:+.1f}pp)")
    old_prec = old_correct / old_gradable if old_gradable else None
    new_prec = new_correct / new_gradable if new_gradable else None
    if old_prec is not None and new_prec is not None:
        print(f"  AUTO-tier precision: {old_prec*100:.1f}% -> {new_prec*100:.1f}%  "
              f"({(new_prec-old_prec)*100:+.1f}pp)")
    print(f"  Linked precision/recall/F1: UNCHANGED (routing-only change, "
          f"final_candidate_index never differs)")

    return {
        "total_decisions": total_decisions, "old_auto": len(old_auto),
        "old_correct": old_correct, "old_gradable": old_gradable,
        "new_auto": new_auto_count, "new_correct": new_correct, "new_gradable": new_gradable,
        "newly_promoted_1b": len(newly_promoted_1b), "new1b_correct": new1b_correct,
        "new1b_gradable": len(new1b_clean),
        "newly_promoted_2b": len(newly_promoted_2b), "new2b_correct": new2b_correct,
        "new2b_gradable": len(new2b_clean),
    }


def main():
    conn = duckdb.connect(DB_PATH, read_only=True)
    vocab = VocabularyRetriever(conn)
    kg3_driver = get_memgraph_driver()
    with kg3_driver.session() as s:
        s.run("RETURN 1")
    print("Memgraph reachable")

    all_notes = sorted(r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM mollm_tier_gate_decisions"
    ).fetchall())

    calibrator_probe = ConsensusCalibrator.load(DEFAULT_MODEL_PATH)
    train_ids = set(calibrator_probe.training_note_ids)
    clean_notes = sorted(set(all_notes) - train_ids)
    print(f"total notes: {len(all_notes)}  calibrator training notes: {len(train_ids)}  "
          f"clean (not in training): {len(clean_notes)}")

    clean_fresh10 = [n for n in FRESH10_NOTE_IDS if n not in train_ids]
    print(f"fresh-10 notes clean of calibrator training: {clean_fresh10} "
          f"({len(clean_fresh10)}/10)")

    # Real load with the leakage guard active, scoped to the clean population
    # -- should NOT be refused, since clean_notes is exactly its complement.
    calibrator = ConsensusCalibrator.load(DEFAULT_MODEL_PATH, scoring_note_ids=clean_notes)
    print(f"calibrator loaded for scoring: "
          f"{'TRAINED (clean)' if calibrator.model is not None else 'REFUSED (leakage!) -- unexpected'}")

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, all_notes)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    results = {}
    results["clean39"] = regrade(clean_notes, "PREVIOUS -- all calibrator-clean notes",
                                  conn, vocab, gold_by_note, calibrator, kg3_driver)
    if clean_fresh10:
        results["fresh10_clean"] = regrade(
            clean_fresh10, "PREVIOUS -- fresh-10 subset clean of calibrator training "
                          "(small sample, n={})".format(len(clean_fresh10)),
            conn, vocab, gold_by_note, calibrator, kg3_driver)

    conn.close()
    kg3_driver.close()
    return results


if __name__ == "__main__":
    main()
