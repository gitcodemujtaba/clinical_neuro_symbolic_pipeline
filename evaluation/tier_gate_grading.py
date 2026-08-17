"""
evaluation/tier_gate_grading.py -- general-purpose, note_ids-parameterized
version of the per-tier grading methodology proven in
evaluation/grade_overnight_corpus_run.py and evaluation/grade_fresh5_by_tier.py
(both of which hardcode their own NOTE_IDS list for a specific run). Factored
out here so a THIRD caller -- ui/pages/4_📊_Evaluation_Metrics.py -- doesn't
duplicate the SNOMED-crosswalk + clean-span grading logic a third time; those
two scripts are left as-is (they're dated, one-off records of specific runs,
not meant to be rewritten to import this after the fact) but any NEW caller
should use this module instead of copy-pasting the pattern again.

Same clean-span + SNOMED-crosswalk methodology throughout this project:
overlaps() defines "any character overlap", clean-span means exactly one
overlapping gold annotation and the prediction is not narrower than it,
concept identity is compared via VocabularyRetriever.snomed_code_for_concept()
so RxNorm-vocabulary Medication links crosswalk to SNOMED like everywhere
else in this codebase.
"""
import collections
import json
import os
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing  # noqa: E402
from scripts.score_gold_recall import load_gold, overlaps  # noqa: E402
from src.mollm_tier_gate import AUTO_TIERS  # noqa: E402
from src.retrieval import VocabularyRetriever  # noqa: E402


def plurality_candidate_index(models_json):
    """Reproduces route_tier()'s own top_verdict derivation exactly (same
    logic as evaluation/grade_overnight_corpus_run.py's identical function,
    duplicated here rather than imported FROM that script -- that script is
    a dated, note-specific artifact, not a shared library this module
    should depend on). Returns (candidate_index, top_verdict, vote_counts)
    or (None, None, None)/(None, verdict, counts) when there's no usable
    vote or the plurality has no candidate (NONE_CORRECT).
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
    return None, top_verdict, vote_counts


def _grade_rows(rows, gold_by_note, vocab, candidate_index_fn):
    """Shared clean-span grading core -- see module docstring."""
    raw, clean = [], []
    skipped = collections.Counter()
    for d in rows:
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
        correct = pred_code is not None and str(pred_code) == str(gold_code)

        rec = {
            "note_id": note_id, "text": d["original_text"], "label": d["entity_label"],
            "pred_concept_name": concept_name, "pred_snomed": pred_code,
            "gold_snomed": gold_code, "correct": correct, "narrower_than_gold": is_narrower,
            **extra,
        }
        raw.append(rec)
        if not is_narrower:
            clean.append(rec)
        else:
            skipped["narrower_than_gold"] += 1
    return raw, clean, skipped


def grade_by_tier(conn, note_ids: list, tiers: list = None) -> dict:
    """Grades every mollm_tier_gate_decisions row for `note_ids`, broken
    down per tier, against gold. Returns
    {tier: {"n_decisions": int, "raw": {...}, "clean": {...}}}, where each
    of raw/clean is {"n": int, "n_correct": int, "precision": float|None,
    "skipped": {...}}.

    `tiers` defaults to every AUTO tier plus TIER_4_ENSEMBLE_SPLIT (graded
    via its plurality candidate -- the "shadow precision" methodology from
    evaluation/grade_overnight_corpus_run.py, showing how much of the
    currently-HITL'd split-vote population the calibrator has a shot at,
    same as that script's own analysis). TIER_5_TRUE_AMBIGUITY is excluded
    by default -- most of its rows have no candidate at all (no_candidates/
    unresolved_acronym/standalone_qualifier_span), so grading it the same
    way as the others would mostly just report "no_candidate" skips.
    """
    tiers = tiers or list(AUTO_TIERS) + ["TIER_4_ENSEMBLE_SPLIT"]
    if not note_ids:
        return {}

    vocab = VocabularyRetriever(conn)
    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, note_ids)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    note_ph = ",".join("?" * len(note_ids))
    tier_ph = ",".join("?" * len(tiers))
    rows = conn.execute(f"""
        SELECT d.entity_id, d.note_id, d.tier, d.final_candidate_index, d.models,
               e.original_text, e.entity_label, e.orig_start, e.orig_end, n.candidates
        FROM mollm_tier_gate_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n ON n.entity_id = d.entity_id
        WHERE d.note_id IN ({note_ph}) AND d.tier IN ({tier_ph})
    """, note_ids + tiers).fetchall()
    cols = [c[0] for c in conn.description]
    decisions = [dict(zip(cols, r)) for r in rows]

    by_tier = collections.defaultdict(list)
    for d in decisions:
        by_tier[d["tier"]].append(d)

    def _auto_idx(d):
        return (d["final_candidate_index"] or None), {}

    def _t4_idx(d):
        idx, top_verdict, vote_counts = plurality_candidate_index(d["models"])
        return idx, {"top_verdict": top_verdict, "vote_counts": dict(vote_counts or {})}

    report = {}
    for tier in tiers:
        tier_decisions = by_tier.get(tier, [])
        idx_fn = _t4_idx if tier == "TIER_4_ENSEMBLE_SPLIT" else _auto_idx
        raw, clean, skipped = _grade_rows(tier_decisions, gold_by_note, vocab, idx_fn)
        n_raw_c = sum(1 for r in raw if r["correct"])
        n_clean_c = sum(1 for r in clean if r["correct"])
        report[tier] = {
            "n_decisions": len(tier_decisions),
            "raw": {"n": len(raw), "n_correct": n_raw_c,
                   "precision": (n_raw_c / len(raw)) if raw else None, "skipped": dict(skipped)},
            "clean": {"n": len(clean), "n_correct": n_clean_c,
                     "precision": (n_clean_c / len(clean)) if clean else None,
                     "records": clean},
        }
    return report
