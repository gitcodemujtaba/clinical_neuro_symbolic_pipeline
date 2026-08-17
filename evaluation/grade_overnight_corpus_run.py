"""
Grade the 2026-08-17 overnight 31-note corpus run (Stage 1->2b with
CNSP_ACRONYM_ESCALATION=1, then Stage 3 tier-gate) against gold. Three
questions, same overlap/SNOMED-crosswalk methodology as
evaluation/grade_allergy_shadow_run.py (clean-span only: single overlapping
gold entity, predicted span not narrower than gold):

1. AUTO-tier precision (TIER_1/2/3, the 659 decisions that cleared the
   strict unanimous gate) -- expected near-perfect; this is the foundation
   Phase 6's calibrator will be layered on top of.
2. TIER_4_ENSEMBLE_SPLIT "shadow precision" -- for each split-vote entity,
   derive the plurality candidate using the exact same logic route_tier()
   itself uses (collections.Counter over usable, non-degenerate/non-error
   votes; top_verdict = most common), then grade THAT candidate against
   gold even though the entity was routed to HITL. This estimates how much
   of the 1,629-entity Tier 4 bucket the ConsensusCalibrator has a shot at
   safely promoting.
3. Acronym escalation (Phase 4) final full-corpus precision -- entities
   whose normalized_from contains "+acronym_mollm"/"+acronym_cache",
   across all 31 notes now that the run has finished, closing out the
   34.3%/36.1% partial reads from the interrupted earlier runs.

Read-only. No LLM calls, no pipeline run.
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
    "10043750-DS-6", "10060142-DS-9", "10097089-DS-8", "10124346-DS-4",
    "10371195-DS-9", "10848570-DS-12", "10860165-DS-24", "10912090-DS-33",
    "11134545-DS-21", "11532659-DS-11", "11649745-DS-4", "11838076-DS-20",
    "11997336-DS-3", "12128814-DS-15", "12247014-DS-9", "12314513-DS-16",
    "12545016-DS-17", "12962702-DS-14", "12970259-DS-4", "13164440-DS-18",
    "14102739-DS-16", "14280440-DS-8", "14490470-DS-11", "14975962-DS-19",
    "15853461-DS-4", "16393593-DS-5", "16991646-DS-11", "17739994-DS-31",
    "17751158-DS-19", "18570237-DS-10", "19442119-DS-15",
]

AUTO_TIERS = {"TIER_1_AUTO_VALIDATED", "TIER_2_AUTO_RESOLVED", "TIER_3_AUTO_VALIDATED"}

# Carried forward from the earlier allergy shadow-run grading -- a real,
# already-adjudicated gold-label error, not re-litigated here.
KNOWN_GOLD_ERRORS = {("11649745-DS-4", "285171000119104")}


def plurality_candidate_index(models_json):
    """Reproduces route_tier()'s own top_verdict derivation (src/mollm_tier_gate.py
    ~line 583-585) exactly: usable votes = not degenerate_generation and
    verdict != 'ERROR'; top_verdict = most common verdict via Counter.
    Returns (candidate_index, top_verdict, vote_counts) or (None, None, None)
    if there's no usable vote or the plurality verdict has no candidate
    (NONE_CORRECT) -- matching the calibrator's own "never consulted for
    that shape" rule in route_tier().
    """
    models = models_json
    if isinstance(models, str):
        models = json.loads(models)
    usable = [m for m in (models or [])
              if not m.get("degenerate_generation") and m.get("verdict") != "ERROR"]
    if not usable:
        return None, None, None
    verdicts = [m["verdict"] for m in usable]
    vote_counts = collections.Counter(verdicts)
    top_verdict, _ = vote_counts.most_common(1)[0]
    if top_verdict == "SUPPORTED_1":
        return 1, top_verdict, vote_counts
    if top_verdict.startswith("RE_RANK_TO_CANDIDATE_"):
        return int(top_verdict.rsplit("_", 1)[1]), top_verdict, vote_counts
    return None, top_verdict, vote_counts  # NONE_CORRECT -- no candidate to grade


def grade_population(decisions, gold_by_note, vocab, candidate_index_fn):
    """Shared grading core for both AUTO-tier and Tier-4-shadow grading.
    candidate_index_fn(decision) -> (idx_1based_or_None, extra_info_dict).
    """
    raw, clean = [], []
    skipped = collections.Counter()
    for d in decisions:
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

        idx, extra = candidate_index_fn(d)
        if idx is None:
            skipped["no_candidate"] += 1
            continue

        candidates = d["candidates"]
        if isinstance(candidates, str):
            candidates = json.loads(candidates)
        i = idx - 1
        if candidates is None or i < 0 or i >= len(candidates):
            skipped["candidate_index_out_of_range"] += 1
            continue
        chosen = candidates[i]
        concept_id = chosen.get("omop_concept_id") or chosen.get("concept_id")
        concept_name = chosen.get("concept_name")

        pred_code = vocab.snomed_code_for_concept(concept_id) if concept_id else None
        gold_code = g0["concept_id"]
        is_gold_error = (note_id, str(gold_code)) in KNOWN_GOLD_ERRORS
        correct = (pred_code is not None and str(pred_code) == str(gold_code)) or is_gold_error

        rec = {
            "note_id": note_id, "text": d["original_text"], "label": d["entity_label"],
            "pred_concept_name": concept_name, "pred_snomed": pred_code,
            "gold_snomed": gold_code, "correct": correct,
            "is_gold_error_adjudicated": is_gold_error, "narrower_than_gold": is_narrower,
            **extra,
        }
        raw.append(rec)
        if not is_narrower:
            clean.append(rec)
        else:
            skipped["narrower_than_gold"] += 1
    return raw, clean, skipped


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
    print(f"Gold annotations loaded for {len(gold_by_note)}/{len(NOTE_IDS)} notes, "
          f"{len(gold_rows)} total.\n")

    note_ph = ",".join("?" * len(NOTE_IDS))

    # ------------------------------------------------------------------
    # 1. AUTO-tier precision (Tier 1/2/3, the 659 decisions)
    # ------------------------------------------------------------------
    tier_ph = ",".join("?" * len(AUTO_TIERS))
    auto_rows = conn.execute(f"""
        SELECT d.entity_id, d.note_id, d.tier, d.final_candidate_index,
               e.original_text, e.expanded_text, e.entity_label,
               e.orig_start, e.orig_end, n.candidates
        FROM mollm_tier_gate_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.note_id IN ({note_ph}) AND d.tier IN ({tier_ph})
    """, NOTE_IDS + list(AUTO_TIERS)).fetchall()
    cols = [c[0] for c in conn.description]
    auto_decisions = [dict(zip(cols, row)) for row in auto_rows]
    print(f"=== 1. AUTO-tier (Tier 1/2/3) precision -- {len(auto_decisions)} decisions fetched ===")

    def auto_idx(d):
        return (d["final_candidate_index"] or None), {"tier": d["tier"]}

    raw, clean, skipped = grade_population(auto_decisions, gold_by_note, vocab, auto_idx)
    n_raw_c = sum(1 for r in raw if r["correct"])
    n_clean_c = sum(1 for r in clean if r["correct"])
    print(f"  skipped: {dict(skipped)}")
    print(f"  ALL gradable (raw):  {pct(n_raw_c, len(raw))}")
    print(f"  CLEAN-span only:     {pct(n_clean_c, len(clean))}")
    print("  --- clean-span incorrect cases ---")
    for r in clean:
        if not r["correct"]:
            print(f"    [{r['note_id']}] {r['text']!r} ({r['label']}, {r['tier']}) "
                  f"pred={r['pred_concept_name']!r}/{r['pred_snomed']} gold={r['gold_snomed']}")
    print()

    # ------------------------------------------------------------------
    # 2. TIER_4_ENSEMBLE_SPLIT shadow precision (the 1,629 decisions)
    # ------------------------------------------------------------------
    t4_rows = conn.execute(f"""
        SELECT d.entity_id, d.note_id, d.tier, d.final_candidate_index, d.models,
               e.original_text, e.expanded_text, e.entity_label,
               e.orig_start, e.orig_end, n.candidates
        FROM mollm_tier_gate_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.note_id IN ({note_ph}) AND d.tier = 'TIER_4_ENSEMBLE_SPLIT'
    """, NOTE_IDS).fetchall()
    cols = [c[0] for c in conn.description]
    t4_decisions = [dict(zip(cols, row)) for row in t4_rows]
    print(f"=== 2. TIER_4_ENSEMBLE_SPLIT shadow precision -- {len(t4_decisions)} decisions fetched ===")

    def t4_idx(d):
        idx, top_verdict, vote_counts = plurality_candidate_index(d["models"])
        return idx, {"top_verdict": top_verdict, "vote_counts": dict(vote_counts or {})}

    raw4, clean4, skipped4 = grade_population(t4_decisions, gold_by_note, vocab, t4_idx)
    n_raw4_c = sum(1 for r in raw4 if r["correct"])
    n_clean4_c = sum(1 for r in clean4 if r["correct"])
    print(f"  skipped: {dict(skipped4)}")
    print(f"  ALL gradable (raw):  {pct(n_raw4_c, len(raw4))}")
    print(f"  CLEAN-span only:     {pct(n_clean4_c, len(clean4))}")
    print("  --- clean-span: plurality-candidate CORRECT despite the split (sample, first 20) ---")
    for r in [r for r in clean4 if r["correct"]][:20]:
        print(f"    [{r['note_id']}] {r['text']!r} ({r['label']}) votes={r['vote_counts']} "
              f"-> {r['pred_concept_name']!r}")
    print("  --- clean-span: plurality-candidate INCORRECT (sample, first 15) ---")
    for r in [r for r in clean4 if not r["correct"]][:15]:
        print(f"    [{r['note_id']}] {r['text']!r} ({r['label']}) votes={r['vote_counts']} "
              f"pred={r['pred_concept_name']!r} gold={r['gold_snomed']}")
    print()

    # ------------------------------------------------------------------
    # 3. Acronym escalation (Phase 4) final full-corpus precision
    # ------------------------------------------------------------------
    acro_rows = conn.execute(f"""
        SELECT e.entity_id, e.note_id, e.original_text, e.expanded_text, e.entity_label,
               e.orig_start, e.orig_end, n.candidates, n.omop_concept_id, n.omop_concept_name,
               n.match_tier, n.normalized_from
        FROM extracted_entities e
        JOIN normalized_entities n ON n.entity_id = e.entity_id
        WHERE e.note_id IN ({note_ph})
          AND (n.normalized_from LIKE '%+acronym_mollm%' OR n.normalized_from LIKE '%+acronym_cache%')
    """, NOTE_IDS).fetchall()
    cols = [c[0] for c in conn.description]
    acro_decisions = [dict(zip(cols, row)) for row in acro_rows]
    print(f"=== 3. Acronym escalation (Phase 4) full-corpus precision -- "
          f"{len(acro_decisions)} escalated+normalized entities found ===")

    def acro_idx(d):
        # Not a tier-gate decision -- the entity's own top (Tier1/2/3-cleared)
        # concept IS candidate 1 by construction of process_and_normalize_entities().
        return (1 if d.get("omop_concept_id") else None), {
            "match_tier": d["match_tier"], "normalized_from": d["normalized_from"]}

    raw_a, clean_a, skipped_a = grade_population(acro_decisions, gold_by_note, vocab, acro_idx)
    n_raw_a_c = sum(1 for r in raw_a if r["correct"])
    n_clean_a_c = sum(1 for r in clean_a if r["correct"])
    print(f"  skipped: {dict(skipped_a)}")
    print(f"  ALL gradable (raw):  {pct(n_raw_a_c, len(raw_a))}")
    print(f"  CLEAN-span only:     {pct(n_clean_a_c, len(clean_a))}")
    print("  --- clean-span incorrect cases (all) ---")
    for r in clean_a:
        if not r["correct"]:
            print(f"    [{r['note_id']}] {r['text']!r} ({r['label']}) "
                  f"pred={r['pred_concept_name']!r}/{r['pred_snomed']} gold={r['gold_snomed']}")
    print()

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"1. AUTO-tier (Tier 1/2/3) clean-span precision:      {pct(n_clean_c, len(clean))}")
    print(f"2. TIER_4_ENSEMBLE_SPLIT shadow clean-span precision: {pct(n_clean4_c, len(clean4))}")
    print(f"3. Acronym escalation full-corpus clean-span precision: {pct(n_clean_a_c, len(clean_a))}")


if __name__ == "__main__":
    main()
