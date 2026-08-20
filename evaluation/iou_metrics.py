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
              by gold span count per class).

              2026-08-20 CORRECTION: an earlier version of this module
              treated "class" as a fabricated single "ALL" bucket, on the
              premise that this project's gold CSV carries no per-class
              field. That premise was wrong -- re-read directly against
              the benchmark page's own metric section: "class" IS the
              SNOMED CT concept ID itself (~7,000 distinct concepts in
              the training notes), and "the predicted concept ID must
              match exactly; relationships between concepts are not
              taken into account for scoring" -- i.e. a predicted span's
              characters only ever count toward a class's predicted set
              if that span's OWN resolved concept equals the class being
              scored. gold's concept_id column IS that class field; it
              was there the whole time. This means the benchmark's char
              IoU is inherently a JOINT span+concept metric, not a pure
              extraction metric -- it cannot be computed from Stage 2a
              alone (spans have no resolved concept yet at that stage).
              See benchmark_char_iou() below for the corrected, concept-
              gated computation; stage2a_iou()'s own char_iou stays as a
              deliberately concept-blind span-only diagnostic, relabeled
              to stop claiming it's the benchmark number.

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
    """pred_spans / gold_spans: list of (label, note_id, start, end).
    Returns (macro_iou, support_weighted_iou, per_class dict).

    Character positions are keyed by (note_id, offset), not raw offset
    alone -- offsets are only meaningful within one note, so pooling two
    different notes' spans by raw int would let a note-A span at [10:20]
    spuriously "overlap" a totally unrelated note-B span that happens to
    sit at the same numeric offset. Keying by (note_id, offset) makes
    cross-note collision structurally impossible while being identical to
    the single-note case (all callers' typical use) when note_id is
    constant across every span passed in.
    """
    classes = set(l for l, _, _, _ in pred_spans) | set(l for l, _, _, _ in gold_spans)
    per_class = {}
    for cls in classes:
        pred_chars = set()
        for l, nid, s, e in pred_spans:
            if l == cls:
                pred_chars.update((nid, p) for p in range(s, e))
        gold_chars = set()
        gold_support = 0
        for l, nid, s, e in gold_spans:
            if l == cls:
                gold_chars.update((nid, p) for p in range(s, e))
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


def benchmark_char_iou(conn, note_ids, gold_by_note, vocab):
    """The DrivenData SNOMED-CT benchmark's own char IoU, computed exactly
    as specified: "class" = SNOMED CT concept ID. A predicted span's
    characters only count toward a class's predicted set if that span's
    OWN resolved concept equals the class; ditto gold via its concept_id
    column. Macro = unweighted mean over classes present in pred ∪ gold;
    support-weighted = weighted by each class's gold span count.

    Uses each entity's Stage 2b top candidate (normalized_entities'
    omop_concept_id, crosswalked to SNOMED the same way every other
    concept-identity comparison in this file does) as "our answer" -- a
    real benchmark submission has no HITL-deferral option, so this is the
    faithful stand-in for "what would we submit" on every span, not just
    the subset that happened to reach an AUTO tier in Stage 3.

    Predictions whose concept can't be crosswalked to a SNOMED code are
    bucketed under the literal class "UNMAPPED" (a value that can never
    collide with a real numeric SNOMED code) so they still count as
    intersection-free noise against whatever class they overlap, rather
    than silently vanishing from the metric.
    """
    rows = conn.execute("""
        SELECT e.orig_start, e.orig_end, e.note_id, n.omop_concept_id
        FROM extracted_entities e
        JOIN normalized_entities n
          ON n.note_id = e.note_id AND n.original_text = e.original_text
         AND n.expanded_text = e.expanded_text AND n.gliner_label = e.entity_label
         AND n.is_test = TRUE
        WHERE e.is_test = TRUE AND e.note_id IN ({})
          AND (e.below_threshold IS NULL OR e.below_threshold = FALSE)
          AND (e.superseded_by_split IS NULL OR e.superseded_by_split = FALSE)
          AND (e.superseded_by_growth IS NULL OR e.superseded_by_growth = FALSE)
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    pred_spans = []
    for s, e, note_id, concept_id in rows:
        code = vocab.snomed_code_for_concept(concept_id) if concept_id else None
        pred_spans.append((str(code) if code else "UNMAPPED", note_id, s, e))

    gold_spans = [(str(g["concept_id"]), note_id, g["start"], g["end"])
                  for note_id in note_ids for g in gold_by_note.get(note_id, [])]

    macro, weighted, per_class = char_iou_by_class(pred_spans, gold_spans)
    return {"macro_char_iou": macro, "weighted_char_iou": weighted,
            "n_classes": len(per_class), "per_class": per_class}


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
        pred_spans.append(("ALL", note_id, s, e))
    fn = 0
    for note_id, gold in gold_by_note.items():
        if note_id not in note_ids:
            continue
        preds = [(r[0], r[1]) for r in rows if r[2] == note_id]
        for g in gold:
            gold_spans.append(("ALL", note_id, g["start"], g["end"]))
            if not any(overlaps(s, e, g["start"], g["end"]) for s, e in preds):
                fn += 1

    # Deliberately concept-blind (single "ALL" class): this is a Stage 2a
    # -only span diagnostic, NOT the DrivenData benchmark's char IoU --
    # that metric is concept-gated (see benchmark_char_iou() above) and
    # can't be computed before a concept has even been resolved.
    span_only_char_iou, _, _ = char_iou_by_class(pred_spans, gold_spans)
    return {"tp": tp, "fp": fp, "fn": fn, "set_iou": set_iou(tp, fp, fn),
            "span_only_char_iou": span_only_char_iou}


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
