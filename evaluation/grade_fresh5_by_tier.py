"""
evaluation/grade_fresh5_by_tier.py -- grades the 2026-08-17 5-fresh-note
calibrator validation run (notes outside the 31-note corpus used to train
models/consensus_calibrator_v1.pkl, per data/splits/note_splits.csv's
official 'test' split) against gold, broken down PER TIER rather than
lumped into one AUTO_TIERS bucket -- the whole point of this run is to see
whether TIER_1B_CALIBRATED_AUTO_VALIDATED holds its validated ~98% on
genuinely unseen notes, separately from TIER_1/TIER_3's own numbers.

Same clean-span + SNOMED-crosswalk methodology as
evaluation/grade_overnight_corpus_run.py (reuses nothing else from it --
that script's NOTE_IDS/KNOWN_GOLD_ERRORS are specific to the 31-note
corpus and don't apply here).
"""
import collections
import json
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
    "13538696-DS-11", "19895550-DS-7", "11516225-DS-20",
    "14652764-DS-17", "12298181-DS-9",
]

TIERS_OF_INTEREST = [
    "TIER_1_AUTO_VALIDATED", "TIER_1B_CALIBRATED_AUTO_VALIDATED",
    "TIER_2_AUTO_RESOLVED", "TIER_3_AUTO_VALIDATED",
]


def pct(n, d):
    return f"{n}/{d} = {n/d*100:.1f}%" if d else f"{n}/{d} = n/a"


def main():
    conn = duckdb.connect(DB_PATH, read_only=True)
    vocab = VocabularyRetriever(conn)

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, NOTE_IDS)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)
    print(f"gold annotations loaded for {len(gold_by_note)}/{len(NOTE_IDS)} notes, "
          f"{len(gold_rows)} total.\n")

    note_ph = ",".join("?" * len(NOTE_IDS))
    rows = conn.execute(f"""
        SELECT d.entity_id, d.note_id, d.tier, d.final_candidate_index, d.routing_basis,
               e.original_text, e.entity_label, e.orig_start, e.orig_end,
               n.candidates
        FROM mollm_tier_gate_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.note_id IN ({note_ph})
    """, NOTE_IDS).fetchall()
    cols = [c[0] for c in conn.description]
    decisions = [dict(zip(cols, row)) for row in rows]
    print(f"total mollm_tier_gate_decisions rows fetched: {len(decisions)}")
    print(f"overall tier distribution: {dict(collections.Counter(d['tier'] for d in decisions))}\n")

    by_tier = collections.defaultdict(list)
    for d in decisions:
        by_tier[d["tier"]].append(d)

    summary = {}
    for tier in TIERS_OF_INTEREST:
        tier_decisions = by_tier.get(tier, [])
        raw, clean = [], []
        skipped = collections.Counter()
        for d in tier_decisions:
            note_id = d["note_id"]
            gold = gold_by_note.get(note_id, [])
            overlapping = [g for g in gold
                          if overlaps(d["orig_start"], d["orig_end"], g["start"], g["end"])]
            if not overlapping:
                skipped["no_gold_overlap"] += 1
                continue
            if len(overlapping) != 1:
                skipped["compound_span"] += 1
                continue
            g0 = overlapping[0]
            is_narrower = (d["orig_end"] - d["orig_start"]) < (g0["end"] - g0["start"])

            candidates = d["candidates"]
            if isinstance(candidates, str):
                candidates = json.loads(candidates)
            idx = (d["final_candidate_index"] or 0) - 1
            if candidates is None or idx < 0 or idx >= len(candidates):
                skipped["no_candidate"] += 1
                continue
            chosen = candidates[idx]
            concept_id = chosen.get("omop_concept_id") or chosen.get("concept_id")
            concept_name = chosen.get("concept_name")

            pred_code = vocab.snomed_code_for_concept(concept_id) if concept_id else None
            gold_code = g0["concept_id"]
            correct = pred_code is not None and str(pred_code) == str(gold_code)

            rec = {"note_id": note_id, "text": d["original_text"], "label": d["entity_label"],
                  "pred_concept_name": concept_name, "pred_snomed": pred_code,
                  "gold_snomed": gold_code, "correct": correct, "narrower": is_narrower,
                  "routing_basis": d["routing_basis"]}
            raw.append(rec)
            if not is_narrower:
                clean.append(rec)
            else:
                skipped["narrower_than_gold"] += 1

        n_raw_c = sum(1 for r in raw if r["correct"])
        n_clean_c = sum(1 for r in clean if r["correct"])
        summary[tier] = {
            "n_decisions": len(tier_decisions), "raw_gradable": len(raw), "raw_correct": n_raw_c,
            "clean_gradable": len(clean), "clean_correct": n_clean_c, "skipped": dict(skipped),
        }

        print("=" * 78)
        print(f"TIER: {tier}  ({len(tier_decisions)} decisions)")
        print("=" * 78)
        print(f"  skipped: {dict(skipped)}")
        print(f"  ALL gradable (raw):  {pct(n_raw_c, len(raw))}")
        print(f"  CLEAN-span only:     {pct(n_clean_c, len(clean))}")
        if tier == "TIER_1B_CALIBRATED_AUTO_VALIDATED" and clean:
            print("  --- every clean-span TIER_1B decision (small population, show all) ---")
            for r in clean:
                tag = "OK" if r["correct"] else "WRONG"
                print(f"    [{tag}] [{r['note_id']}] {r['text']!r} ({r['label']}) "
                      f"-> {r['pred_concept_name']!r}  basis={r['routing_basis'][:100]}")
        elif clean:
            print("  --- clean-span incorrect cases ---")
            for r in clean:
                if not r["correct"]:
                    print(f"    [{r['note_id']}] {r['text']!r} ({r['label']}) "
                          f"pred={r['pred_concept_name']!r}/{r['pred_snomed']} "
                          f"gold={r['gold_snomed']}")
        print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for tier in TIERS_OF_INTEREST:
        s = summary[tier]
        print(f"{tier:38s} n={s['n_decisions']:4d}  clean-span: "
              f"{pct(s['clean_correct'], s['clean_gradable'])}")

    other_tiers = {d["tier"] for d in decisions} - set(TIERS_OF_INTEREST)
    if other_tiers:
        print(f"\n(other tiers present, not broken out here: "
              f"{dict(collections.Counter(d['tier'] for d in decisions if d['tier'] in other_tiers))})")


if __name__ == "__main__":
    main()
