"""
Grade the 6-note post-allergy-fix Stage 3 tier-gate run against gold.
Same methodology as docs/2026-08-16_Shadow_Run_Precision_At_Scale.md's
original 76.1%/78.3% measurement: SNOMED crosswalk via
VocabularyRetriever.snomed_code_for_concept(), clean-span only (single
overlapping gold entity, predicted span not narrower than gold), and the
RCA occlusion gold-label-error adjudication carried forward as a known
correction (not re-litigated here).
"""
import collections
import os
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
sys.path.insert(0, PROJECT_DIR)

from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from scripts.score_gold_recall import load_gold, overlaps  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402

NOTE_IDS = [
    "10912090-DS-33", "13164440-DS-18", "11649745-DS-4",
    "15853461-DS-4", "17739994-DS-31", "12314513-DS-16",
]

# Known gold-labeling error adjudicated this session (RCA occlusion,
# 11649745-DS-4) -- carried forward, not re-derived.
KNOWN_GOLD_ERRORS = {
    ("11649745-DS-4", "285171000119104"),
}

AUTO_TIERS = {"TIER_1_AUTO_VALIDATED", "TIER_2_AUTO_RESOLVED", "TIER_3_AUTO_VALIDATED"}


def main():
    conn = duckdb.connect(DB_PATH, read_only=True)
    vocab = VocabularyRetriever(conn)

    decisions = conn.execute(f"""
        SELECT d.entity_id, d.note_id, d.tier, d.final_candidate_index,
               e.original_text, e.expanded_text, e.entity_label,
               e.orig_start, e.orig_end, e.assertion_status,
               n.candidates
        FROM mollm_tier_gate_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.note_id IN ({",".join("?" * len(NOTE_IDS))})
          AND d.tier IN ({",".join("?" * len(AUTO_TIERS))})
    """, NOTE_IDS + list(AUTO_TIERS)).fetchall()
    cols = [c[0] for c in conn.description]
    decisions = [dict(zip(cols, row)) for row in decisions]

    print(f"AUTO-tier decisions fetched: {len(decisions)}")

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, NOTE_IDS)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    import json as _json

    raw_gradable = []
    clean_span = []
    skipped_no_gold = 0
    skipped_compound = 0
    skipped_narrower = 0
    skipped_no_candidate = 0

    for d in decisions:
        note_id = d["note_id"]
        gold = gold_by_note.get(note_id, [])
        overlapping = [g for g in gold
                      if overlaps(d["orig_start"], d["orig_end"], g["start"], g["end"])]
        if not overlapping:
            skipped_no_gold += 1
            continue
        if len(overlapping) != 1:
            skipped_compound += 1
            continue
        g0 = overlapping[0]
        is_narrower = (d["orig_end"] - d["orig_start"]) < (g0["end"] - g0["start"])

        candidates = d["candidates"]
        if isinstance(candidates, str):
            candidates = _json.loads(candidates)
        idx = (d["final_candidate_index"] or 0) - 1
        if candidates is None or idx < 0 or idx >= len(candidates):
            skipped_no_candidate += 1
            continue
        chosen = candidates[idx]
        concept_id = chosen.get("omop_concept_id") or chosen.get("concept_id")
        concept_name = chosen.get("concept_name")

        pred_code = vocab.snomed_code_for_concept(concept_id) if concept_id else None
        gold_code = g0["concept_id"]
        is_gold_error = (note_id, str(gold_code)) in KNOWN_GOLD_ERRORS

        correct_raw = (pred_code is not None and str(pred_code) == str(gold_code)) or is_gold_error
        rec = {
            "note_id": note_id, "text": d["original_text"], "label": d["entity_label"],
            "tier": d["tier"], "pred_concept_id": concept_id, "pred_concept_name": concept_name,
            "pred_snomed": pred_code, "gold_snomed": gold_code, "gold_concept_name": g0.get("concept_name"),
            "correct": correct_raw, "is_gold_error_adjudicated": is_gold_error,
            "narrower_than_gold": is_narrower,
        }
        raw_gradable.append(rec)
        if not is_narrower:
            clean_span.append(rec)
        else:
            skipped_narrower += 1

    n_raw = len(raw_gradable)
    n_raw_correct = sum(1 for r in raw_gradable if r["correct"])
    n_clean = len(clean_span)
    n_clean_correct = sum(1 for r in clean_span if r["correct"])

    print(f"\nskipped: no_gold_overlap={skipped_no_gold} compound_span={skipped_compound} "
          f"no_candidate={skipped_no_candidate} (narrower-than-gold counted separately: {skipped_narrower})")
    print(f"\nALL gradable (raw):  {n_raw_correct}/{n_raw} = "
          f"{n_raw_correct/n_raw*100:.1f}%" if n_raw else "ALL gradable: n/a")
    print(f"CLEAN-span only:     {n_clean_correct}/{n_clean} = "
          f"{n_clean_correct/n_clean*100:.1f}%" if n_clean else "CLEAN-span: n/a")

    print("\n--- clean-span incorrect cases ---")
    for r in clean_span:
        if not r["correct"]:
            print(f"  [{r['note_id']}] {r['text']!r} ({r['label']}, {r['tier']}) "
                 f"pred={r['pred_concept_name']!r}/{r['pred_snomed']}  "
                 f"gold={r['gold_concept_name']!r}/{r['gold_snomed']}")

    print("\n--- clean-span correct cases (sample, first 15) ---")
    for r in [r for r in clean_span if r["correct"]][:15]:
        tag = " [GOLD-ERROR-ADJUDICATED]" if r["is_gold_error_adjudicated"] else ""
        print(f"  [{r['note_id']}] {r['text']!r} ({r['label']}) "
             f"pred={r['pred_concept_name']!r}{tag}")

    allergy_related = [r for r in clean_span
                       if r["label"] == "Medication" and
                       ("allerg" in (r["pred_concept_name"] or "").lower())]
    print(f"\n--- allergy-context entities resolved to an Allergy concept: {len(allergy_related)} ---")
    for r in allergy_related:
        print(f"  [{r['note_id']}] {r['text']!r} -> {r['pred_concept_name']!r} "
             f"correct={r['correct']}")


if __name__ == "__main__":
    main()
