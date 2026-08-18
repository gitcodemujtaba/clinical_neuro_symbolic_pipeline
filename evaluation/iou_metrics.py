"""
evaluation/iou_metrics.py — per-stage IoU (Intersection over Union), the
set-overlap counterpart to evaluation/stage_calibration.py's ECE curves.
Built 2026-08-18 while grading a single fresh note (16410990-DS-12)
end-to-end against gold and asked to report IoU alongside ECE for every
stage — factored out here, and wired into
ui/pages/4_📊_Evaluation_Metrics.py, rather than left as a one-off script,
so the same numbers are reproducible for any note selection.

TWO FORMS OF IoU, BECAUSE ONLY ONE STAGE PRODUCES SPANS.
  set IoU   = TP / (TP + FP + FN) at the DECISION level. TP/FP/FN are
              defined per stage below (span-is-real for 2a, concept-is-
              right for 2b, AUTO-tier-decision-is-right for 3) -- this is
              the generalization of IoU/Jaccard to any correct/incorrect
              decision set, not just bounding boxes, and is comparable
              across all three stages.
  char IoU  = the DrivenData SNOMED-CT entity-linking benchmark's own
              definition (https://www.drivendata.org/benchmarks/310/
              benchmark-snomed-ct/page/983/): IoU_class = |chars(pred) ∩
              chars(gold)| / |chars(pred) ∪ chars(gold)|, aggregated macro
              (unweighted mean per class) and support-weighted (weighted
              by gold span count per class). Meaningful only for Stage 2a
              (extraction is the only stage that changes span boundaries).
              This project's gold CSV (train_annotations.csv) carries no
              entity-label/class field -- only span + concept_id -- so
              "class" collapses to one aggregate bucket rather than a
              fabricated taxonomy gold doesn't provide; that's the
              single-class case of the same formula, not a different one.

Same conventions as every other evaluation/ script: overlaps() defines
"any character overlap" (scripts/score_gold_recall.py), is_test=TRUE
scopes to smoke-run data, superseded_by_split/superseded_by_growth rows
are excluded (a replaced entity and its replacement would otherwise both
contribute a data point for what is really one decision), concept
identity is compared via VocabularyRetriever.snomed_code_for_concept()
so RxNorm-vocabulary Medication links crosswalk to SNOMED like everywhere
else in this codebase.
"""
import json
import os
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts.score_gold_recall import overlaps  # noqa: E402


def set_iou(tp, fp, fn):
    denom = tp + fp + fn
    return round(tp / denom, 4) if denom else None


def char_iou_by_class(pred_spans, gold_spans):
    """pred_spans / gold_spans: list of (label, start, end). Returns
    (macro_iou, support_weighted_iou, per_class dict)."""
    classes = set(l for l, _, _ in pred_spans) | set(l for l, _, _ in gold_spans)
    per_class = {}
    for cls in classes:
        pred_chars = set()
        for l, s, e in pred_spans:
            if l == cls:
                pred_chars.update(range(s, e))
        gold_chars = set()
        gold_support = 0
        for l, s, e in gold_spans:
            if l == cls:
                gold_chars.update(range(s, e))
                gold_support += 1
        union = pred_chars | gold_chars
        inter = pred_chars & gold_chars
        iou = len(inter) / len(union) if union else None
        per_class[cls] = {"iou": round(iou, 4) if iou is not None else None,
                          "gold_support": gold_support}
    valid = [(c, v) for c, v in per_class.items() if v["iou"] is not None]
    macro = sum(v["iou"] for _, v in valid) / len(valid) if valid else None
    total_support = sum(v["gold_support"] for _, v in valid)
    weighted = (sum(v["iou"] * v["gold_support"] for _, v in valid) / total_support
                if total_support else None)
    return (round(macro, 4) if macro is not None else None,
            round(weighted, 4) if weighted is not None else None,
            per_class)


# ==========================================================================
# Stage 2a -- extraction: is the predicted SPAN real at all (any gold
# overlap, any concept)?
# ==========================================================================

def stage2a_iou(conn, note_ids, gold_by_note):
    rows = conn.execute("""
        SELECT orig_start, orig_end, note_id
        FROM extracted_entities
        WHERE is_test = TRUE AND note_id IN ({}) AND (below_threshold IS NULL OR below_threshold = FALSE)
          AND (superseded_by_split IS NULL OR superseded_by_split = FALSE)
          AND (superseded_by_growth IS NULL OR superseded_by_growth = FALSE)
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    tp = fp = 0
    pred_spans, gold_spans = [], []
    for s, e, note_id in rows:
        gold = gold_by_note.get(note_id, [])
        if any(overlaps(s, e, g["start"], g["end"]) for g in gold):
            tp += 1
        else:
            fp += 1
        pred_spans.append(("ALL", s, e))
    fn = 0
    for note_id, gold in gold_by_note.items():
        if note_id not in note_ids:
            continue
        preds = [(r[0], r[1]) for r in rows if r[2] == note_id]
        for g in gold:
            gold_spans.append(("ALL", g["start"], g["end"]))
            if not any(overlaps(s, e, g["start"], g["end"]) for s, e in preds):
                fn += 1

    macro_char_iou, weighted_char_iou, _ = char_iou_by_class(pred_spans, gold_spans)
    return {"tp": tp, "fp": fp, "fn": fn, "set_iou": set_iou(tp, fp, fn),
            "char_iou": macro_char_iou}


# ==========================================================================
# Stage 2b -- normalization/linking: given a span with a gold match, is the
# CONCEPT this entity's chosen candidate names the right one?
# ==========================================================================

def stage2b_iou(conn, note_ids, gold_by_note, vocab):
    rows = conn.execute("""
        SELECT n.orig_start, n.orig_end, n.note_id, n.omop_concept_id
        FROM (
            SELECT e.orig_start, e.orig_end, e.note_id, n.omop_concept_id
            FROM normalized_entities n
            JOIN extracted_entities e
              ON e.note_id = n.note_id AND e.original_text = n.original_text
             AND e.expanded_text = n.expanded_text AND e.entity_label = n.gliner_label
            WHERE n.is_test = TRUE AND n.note_id IN ({})
        ) n
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    tp = fp = 0
    for s, e, note_id, concept_id in rows:
        gold = gold_by_note.get(note_id, [])
        overlapping = [g for g in gold if overlaps(s, e, g["start"], g["end"])]
        if not overlapping:
            continue  # no gold span here -- 2a already accounts for pure extraction misses
        code = vocab.snomed_code_for_concept(concept_id) if concept_id else None
        if any(str(code) == str(g["concept_id"]) for g in overlapping):
            tp += 1
        else:
            fp += 1

    fn = 0
    for note_id, gold in gold_by_note.items():
        if note_id not in note_ids:
            continue
        note_rows = [(r[0], r[1], r[3]) for r in rows if r[2] == note_id]
        for g in gold:
            hit = any(overlaps(s, e, g["start"], g["end"])
                     and str(vocab.snomed_code_for_concept(c) if c else None) == str(g["concept_id"])
                     for s, e, c in note_rows)
            if not hit:
                fn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "set_iou": set_iou(tp, fp, fn)}


# ==========================================================================
# Stage 3 -- tier gate: is each AUTO-tier decision's resolved concept right?
# Uses the CURRENT production table (mollm_tier_gate_decisions / route_tier
# tiers), not the superseded mollm_decisions/route() gate.
# ==========================================================================

def stage3_iou(conn, note_ids, gold_by_note, vocab):
    from src.mollm_tier_gate import AUTO_TIERS

    rows = conn.execute("""
        SELECT d.tier, d.final_candidate_index, e.orig_start, e.orig_end, e.note_id, nm.candidates
        FROM mollm_tier_gate_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id AND e.is_test = TRUE
        JOIN normalized_entities nm ON nm.entity_id = d.entity_id AND nm.is_test = TRUE
        WHERE d.note_id IN ({}) AND d.is_test = TRUE
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    def resolved_code(final_idx, cand_json):
        candidates = json.loads(cand_json) if cand_json else []
        if final_idx is None or not (1 <= final_idx <= len(candidates)):
            return None
        return vocab.snomed_code_for_concept(candidates[final_idx - 1].get("omop_concept_id"))

    tp = fp = 0
    for tier, final_idx, s, e, note_id, cand_json in rows:
        if tier not in AUTO_TIERS:
            continue
        gold = gold_by_note.get(note_id, [])
        overlapping = [g for g in gold if overlaps(s, e, g["start"], g["end"])]
        if not overlapping:
            continue
        code = resolved_code(final_idx, cand_json)
        if any(str(code) == str(g["concept_id"]) for g in overlapping):
            tp += 1
        else:
            fp += 1

    fn = 0
    for note_id, gold in gold_by_note.items():
        if note_id not in note_ids:
            continue
        note_rows = [r for r in rows if r[4] == note_id and r[0] in AUTO_TIERS]
        for g in gold:
            hit = any(overlaps(s, e, g["start"], g["end"])
                     and str(resolved_code(final_idx, cand_json)) == str(g["concept_id"])
                     for _tier, final_idx, s, e, _nid, cand_json in note_rows)
            if not hit:
                fn += 1

    n_auto = sum(1 for r in rows if r[0] in AUTO_TIERS)
    return {"tp": tp, "fp": fp, "fn": fn, "set_iou": set_iou(tp, fp, fn),
            "n_decisions": len(rows), "n_auto": n_auto}
