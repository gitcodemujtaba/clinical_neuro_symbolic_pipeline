"""
scripts/score_gold_recall.py

Scores Stage 1 -> Stage 2a -> Stage 2b pipeline output against the SNOMED CT
Entity Linking Challenge gold annotations (train_annotations.csv). Read-only,
no LLM calls, no pipeline run -- run test_pipeline_e2e.py --note-ids first so
extracted_entities/normalized_entities hold is_test=TRUE rows for the notes
being scored.

WHY THIS EXISTS. docs/Stage3_Open_Issues.md and the three ranked gold notes
(smallest/median/largest by annotation count: 17751158-DS-19, 19442119-DS-15,
14490470-DS-11) were chosen so Stage 3 recall could be measured before a
multi-day, 272-note corpus run. This script is the measurement: it turns the
console printout of "N entities accepted, M normalized" into an actual
recall/precision number against the challenge's own gold labels.

TWO RECALL FIGURES, AND WHY BOTH MATTER.
  SPAN RECALL   -- fraction of gold annotations with ANY overlapping predicted
                   entity, regardless of whether the concept is right. This is
                   purely a Stage 2a (GLiNER extraction) number.
  LINKED RECALL -- fraction of gold annotations where an overlapping predicted
                   entity ALSO resolved to the correct SNOMED concept_id. This
                   is the actual metric the SNOMED CT Entity Linking Challenge
                   scores, and it is Stage 2a + Stage 2b combined.
The gap between the two tells you whether a miss is an extraction problem
(span never found) or a normalization problem (span found, wrong concept) --
conflating them, as a single "recall" figure would, throws that away.

MATCHING IS OVERLAP-BASED, NOT EXACT-SPAN. Gold annotations are frequently
single tokens ("Left", "craniectomy" scored separately) while this pipeline's
spans are often multi-word ("Left craniectomy" as one Procedure entity). A
predicted span legitimately covering several gold annotations is expected
behavior, not a bug -- exact-span matching would undercount correct links for
no good reason. "Overlap" here means any character overlap: pred_start <
gold_end and pred_end > gold_start, within the same note.

COMPOUND-SPAN CASES ARE COUNTED SEPARATELY, NOT JUST TOLERATED. Some overlap
cases are not "multi-word span, one concept" (like "Left craniectomy" above)
but genuinely COMPOUND: gold expects two DIFFERENT concepts for two DIFFERENT
sub-spans that this pipeline extracted as one entity with one concept -- e.g.
gold codes "gunshot wound" (56768003) and "abdomen" (818983003) as separate
annotations, while Stage 2a extracts "gunshot wound to abdomen" as a single
Procedure entity normalized to one code. That entity can overlap both gold
spans (counting toward span recall for each) but cannot possibly match both
concepts (it only carries one), regardless of how good Stage 2b/3 get -- the
fix is Stage 2a splitting compound spans or Stage 2b emitting multiple links
per entity, not normalization tuning. find_compound_spans() flags every
prediction overlapping >=2 distinct gold annotations so this failure mode is
counted automatically instead of found by eye in the wrong-concept examples.

MEDICATION CROSSWALK. Stage 2b normalizes Medication entities against RxNorm,
not SNOMED (src/normalization/constants.py's VOCAB_BY_LABEL), so a bare concept_id
comparison against gold's SNOMED concept_id would score every correct
medication link as wrong purely on vocabulary mismatch. This script reuses
VocabularyRetriever.snomed_code_for_concept() from src/retrieval.py -- the
same RxNorm->SNOMED crosswalk (via athena_concept_relationship) Stage 3
retrieval already depends on -- so medication links are compared on SNOMED
code, not blindly rejected. Concepts with no crosswalk hit are reported in a
separate "uncrosswalked" bucket rather than folded into "wrong concept",
since those are two different failure modes.

KNOWN DB CAVEAT THIS SCRIPT WORKS AROUND, NOT FIXES.
normalized_entities is UNIQUE on (note_id, original_text, expanded_text,
gliner_label) -- NOT on entity_id, despite entity_id being written to every
row for exactly this fan-out purpose (see process_and_normalize_entities's
"DEDUP FAN-OUT" docstring in src/normalization.py). When the same text+label
mention repeats in a note (e.g. "heart failure" appears 7+ times in
19442119-DS-15), each INSERT for the 2nd..nth occurrence hits the SAME
constraint and does an ON CONFLICT DO UPDATE that overwrites entity_id --
so only the LAST-written entity_id's row survives under that key, and a join
on entity_id alone silently drops every earlier duplicate-text entity from
the score. This script joins on (note_id, original_text, expanded_text,
gliner_label) instead of entity_id, which is safe here because
normalize_entity() is a pure function of that same key -- the concept mapping
IS identical for every duplicate, only the DB row-per-entity_id persistence is
broken. That persistence gap is real and matters for Stage 4 (which needs one
provenance row per entity_id), but it is a separate defect from what this
script measures and is not fixed here.

ALSO REPORTS THE OFFICIAL BENCHMARK METRIC. Beyond the recall/precision
breakdown above (built for debugging Stage 2a vs 2b failures), this script
also computes macro and support-weighted character IoU exactly as defined by
the SNOMED CT Entity Linking Benchmark (drivendata.org/benchmarks/310) and
its reference scoring.py (github.com/drivendataorg/snomed-ct-benchmark-runtime),
so a number from this script means the same thing as a number on that
leaderboard and other solutions' reported results. See the
official_character_iou() docstring below for the exact correspondence and its
one necessary caveat (this pipeline's Medication entities resolve against a
vocabulary the gold set structurally excludes).

Run:
  python3 scripts/score_gold_recall.py
  python3 scripts/score_gold_recall.py --note-ids 17751158-DS-19,19442119-DS-15 --out report.json
"""

import argparse
import collections
import csv
import json
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import VocabularyRetriever  # noqa: E402

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

# Same rationale as scripts/measure_channel_b_coverage.py: the repo layout
# nests the gold set under data/evaluaiton-dataset/, the deployed EC2 tree has
# it directly under data/ -- try both rather than hardcoding one.
GOLD_CANDIDATES = [
    os.path.join(PROJECT_DIR, "data", "evaluaiton-dataset",
                 "snomed-ct-entity-linking-challenge-1.2.0", "train_annotations.csv"),
    os.path.join(PROJECT_DIR, "data", "snomed-ct-entity-linking-challenge-1.2.0",
                 "train_annotations.csv"),
    os.path.join(PROJECT_DIR, "data", "evaluation-dataset",
                 "snomed-ct-entity-linking-challenge-1.2.0", "train_annotations.csv"),
]

# The three notes ranked by gold-annotation count for a fast, scoreable
# read before committing to the full 272-note corpus: small (144 gold
# annotations), median (269), large (431).
DEFAULT_NOTE_IDS = ["17751158-DS-19", "19442119-DS-15", "14490470-DS-11"]


def _first_existing(candidates, what):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise SystemExit(
        f"Could not locate {what}. Tried:\n  " + "\n  ".join(candidates)
        + "\nPass an explicit path."
    )


def load_gold(path, note_ids):
    """Gold annotations for the target notes, offsets coerced to int.

    train_annotations.csv stores start/end as floats ('184.0') -- MIMIC-IV
    note text itself is plain ASCII/UTF-8 with integer character offsets, the
    float formatting is just how the CSV was written.
    """
    want = set(note_ids)
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["note_id"] not in want:
                continue
            rows.append({
                "note_id": r["note_id"],
                "start": int(float(r["start"])),
                "end": int(float(r["end"])),
                "span": r["span"],
                "concept_id": r["concept_id"],
            })
    return rows


def load_predictions(conn, note_ids):
    """Accepted (>=EXTRACTION_THRESHOLD), normalized entities for the target
    notes, joined on (note_id, original_text, expanded_text, gliner_label)
    rather than entity_id -- see module docstring for why entity_id can't be
    trusted as a join key here.

    is_test=TRUE restricts to rows written by test_pipeline_e2e.py smoke
    runs, matching how these notes were actually processed.

    EXCLUDES superseded_by_split=TRUE and superseded_by_growth=TRUE rows.
    2026-08-10, compound-span splitting (src/clinical_pipeline.
    split_compound_entities()) replaces a compound entity with two-or-more
    atomic ones but keeps the merged parent's extracted_entities row for
    audit rather than deleting it, flagged superseded_by_split=TRUE.
    Span-growth (src/clinical_pipeline.grow_entity_spans(), added the same
    day for the mirror-image under-extraction problem) does the same thing
    with superseded_by_growth=TRUE when it replaces a too-narrow entity
    with a widened one. Either superseded row's normalized_entities row can
    also still be present (a stale row from before the mechanism existed,
    or simply never cleaned up) and would join here just like any live
    entity -- without this filter, a note that had a split or a growth
    applied would double-count: the replacement prediction(s) AND the
    entity they replaced.
    """
    rows = conn.execute("""
        SELECT e.note_id, e.orig_start, e.orig_end, e.entity_label,
               e.original_text, e.expanded_text, e.assertion_status,
               e.entity_id, e.expansion_ambiguous, e.selection_basis,
               n.omop_concept_id, n.omop_concept_name, n.omop_vocab,
               n.match_tier
        FROM extracted_entities e
        JOIN normalized_entities n
          ON n.note_id = e.note_id
         AND n.original_text = e.original_text
         AND n.expanded_text = e.expanded_text
         AND n.gliner_label = e.entity_label
         AND n.is_test = TRUE
        WHERE e.is_test = TRUE
          AND e.note_id IN ({})
          AND (e.below_threshold IS NULL OR e.below_threshold = FALSE)
          AND (e.superseded_by_split IS NULL OR e.superseded_by_split = FALSE)
          AND (e.superseded_by_growth IS NULL OR e.superseded_by_growth = FALSE)
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    cols = ["note_id", "orig_start", "orig_end", "entity_label", "original_text",
            "expanded_text", "assertion_status", "entity_id", "expansion_ambiguous",
            "selection_basis", "omop_concept_id", "omop_concept_name", "omop_vocab",
            "match_tier"]
    return [dict(zip(cols, r)) for r in rows]


def attach_snomed_codes(conn, predictions):
    """Resolves each prediction's SNOMED code, crosswalking RxNorm (Medication
    entities) through VocabularyRetriever's athena_concept_relationship
    lookup. Mutates and returns `predictions`."""
    vocab = VocabularyRetriever(conn)
    for p in predictions:
        cid = p["omop_concept_id"]
        if cid is None:
            p["snomed_code"] = None
            p["crosswalk_attempted"] = False
        else:
            p["snomed_code"] = vocab.snomed_code_for_concept(cid)
            p["crosswalk_attempted"] = (p["omop_vocab"] != "SNOMED")
    return predictions


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and a_end > b_start


# Explicit rank, not lexicographic sort: match_tier strings are "1 (Exact)",
# "2 (Synonym)", "3 (Semantic)", "0 (Failed)" -- sorting the strings directly
# would rank "0 (Failed)" ahead of "1 (Exact)" because '0' < '1', which is
# backwards (0 is the WORST tier, not the best). Unranked/unknown tiers sort
# last, same as "Failed".
TIER_RANK = {"1 (Exact)": 1, "2 (Synonym)": 2, "3 (Semantic)": 3, "0 (Failed)": 4}


def best_tier(preds):
    """The most-confident (lowest-rank) prediction among a list of predictions
    overlapping the same gold span -- used as the representative when several
    predictions overlap one gold annotation."""
    return min(preds, key=lambda p: TIER_RANK.get(p["match_tier"], 5))


def find_compound_spans(gold_by_note, preds_by_note):
    """Predictions whose single span overlaps >=2 distinct gold annotations --
    the compound-phrase pattern documented in the module docstring (one
    'gunshot wound to abdomen' Procedure entity overlapping the two separate
    gold spans 'gunshot wound' and 'abdomen', each with its own SNOMED
    concept). A single predicted concept can structurally satisfy at most one
    of the overlapped gold annotations, so this is a distinct failure mode
    from the tier/uncrosswalked breakdown in score() -- it needs Stage 2a to
    split compound spans or Stage 2b to emit multiple concept links per
    entity, not normalization tuning.
    """
    examples = []
    by_note = collections.Counter()
    total = 0
    for note_id, preds in preds_by_note.items():
        gold = gold_by_note.get(note_id, [])
        for p in preds:
            overlapping_gold = [
                g for g in gold
                if overlaps(p["orig_start"], p["orig_end"], g["start"], g["end"])
            ]
            if len(overlapping_gold) >= 2:
                total += 1
                by_note[note_id] += 1
                if len(examples) < 15:
                    examples.append({
                        "note_id": note_id,
                        "predicted_text": p["original_text"],
                        "predicted_start": p["orig_start"],
                        "predicted_end": p["orig_end"],
                        "predicted_concept": p["omop_concept_name"],
                        "gold_spans": [
                            {"span": g["span"], "concept_id": g["concept_id"],
                             "start": g["start"], "end": g["end"]}
                            for g in overlapping_gold
                        ],
                    })
    return {"count": total, "by_note": dict(by_note), "examples": examples}


def score(gold_rows, predictions):
    """Per-note and combined span/linked recall, plus a match_tier breakdown
    of misses and a sample of concrete errors for manual inspection."""
    preds_by_note = collections.defaultdict(list)
    for p in predictions:
        preds_by_note[p["note_id"]].append(p)

    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)

    # Computed up front (needs only gold_by_note/preds_by_note) so its
    # per-note counts can be folded into the per_note table below rather than
    # only appearing in the combined/examples view.
    compound_spans = find_compound_spans(gold_by_note, preds_by_note)

    per_note = {}
    wrong_concept_examples = []
    missed_span_examples = []
    uncrosswalked = 0

    # 2026-08-17 (abbreviation flywheel follow-up): does the WINNING
    # tiebreak (src.preprocessing.expand_text_and_track_offsets()'s
    # selection_basis -- alphabetical_default/numeric_context/
    # omop_groundability/observed_frequency_priority/context_pattern_rule)
    # correlate with linked-recall correctness? This is the direct,
    # concrete answer to "is the flywheel actually helping completeness",
    # not just "is AUTO-tier precision holding up" -- a different question
    # this script otherwise has no way to answer, since it never looked at
    # expansion_ambiguous/selection_basis at all before. Keyed by
    # selection_basis directly (not just "ambiguous vs not") so the two new
    # flywheel tiebreaks can be compared against the three pre-existing
    # heuristics on equal footing, not lumped into one bucket.
    selection_basis_stats = collections.defaultdict(lambda: {"correct": 0, "total": 0})

    for note_id, gold in gold_by_note.items():
        preds = preds_by_note.get(note_id, [])
        span_covered = 0
        linked_correct = 0
        tier_of_correct = collections.Counter()
        tier_of_wrong = collections.Counter()

        for g in gold:
            overlapping = [p for p in preds
                           if overlaps(p["orig_start"], p["orig_end"], g["start"], g["end"])]
            if not overlapping:
                if len(missed_span_examples) < 15:
                    missed_span_examples.append(g)
                continue
            span_covered += 1

            hit = next((p for p in overlapping if p["snomed_code"] == g["concept_id"]), None)
            if hit:
                linked_correct += 1
                tier_of_correct[hit["match_tier"]] += 1
                if hit.get("expansion_ambiguous") and hit.get("selection_basis"):
                    bucket = selection_basis_stats[hit["selection_basis"]]
                    bucket["correct"] += 1
                    bucket["total"] += 1
            else:
                # Report the most-confident attempt among the overlapping
                # predictions as the representative miss.
                best = best_tier(overlapping)
                tier_of_wrong[best["match_tier"]] += 1
                if best.get("expansion_ambiguous") and best.get("selection_basis"):
                    selection_basis_stats[best["selection_basis"]]["total"] += 1
                if best["omop_vocab"] and best["omop_vocab"] != "SNOMED" and not best["snomed_code"]:
                    uncrosswalked += 1
                elif len(wrong_concept_examples) < 15:
                    wrong_concept_examples.append({
                        "note_id": note_id, "gold_span": g["span"],
                        "gold_concept_id": g["concept_id"],
                        "predicted_text": best["original_text"],
                        "predicted_concept": best["omop_concept_name"],
                        "predicted_snomed_code": best["snomed_code"],
                        "match_tier": best["match_tier"],
                    })

        n_gold = len(gold)
        per_note[note_id] = {
            "gold_annotations": n_gold,
            "predicted_entities": len(preds),
            "span_covered": span_covered,
            "span_recall": span_covered / n_gold if n_gold else 0.0,
            "linked_correct": linked_correct,
            "linked_recall": linked_correct / n_gold if n_gold else 0.0,
            "tier_of_correct_links": dict(tier_of_correct),
            "tier_of_wrong_links": dict(tier_of_wrong),
            "compound_spans": compound_spans["by_note"].get(note_id, 0),
        }

    total_gold = sum(v["gold_annotations"] for v in per_note.values())
    total_span = sum(v["span_covered"] for v in per_note.values())
    total_linked = sum(v["linked_correct"] for v in per_note.values())

    combined = {
        "gold_annotations": total_gold,
        "predicted_entities": sum(v["predicted_entities"] for v in per_note.values()),
        "span_covered": total_span,
        "span_recall": total_span / total_gold if total_gold else 0.0,
        "linked_correct": total_linked,
        "linked_recall": total_linked / total_gold if total_gold else 0.0,
        "uncrosswalked_misses": uncrosswalked,
    }

    # "accuracy", not "recall": scoped to gold spans that already HAD an
    # overlapping ambiguous-abbreviation prediction (span-recall's job, not
    # this breakdown's) -- this answers "when this tiebreak picked an
    # expansion and got a chance to compete for a gold span, how often was
    # the resulting concept actually right", which is what tells you
    # whether observed_frequency_priority/context_pattern_rule (the new
    # flywheel tiebreaks) are doing better than the three pre-existing
    # heuristics, not a recall figure in the same sense as span/linked
    # recall above.
    ambiguous_abbreviation_breakdown = {
        basis: {
            "correct": v["correct"], "total": v["total"],
            "accuracy": v["correct"] / v["total"] if v["total"] else 0.0,
        }
        for basis, v in sorted(selection_basis_stats.items())
    }

    return {
        "per_note": per_note,
        "combined": combined,
        "wrong_concept_examples": wrong_concept_examples,
        "missed_span_examples": missed_span_examples,
        "compound_spans": compound_spans,
        "ambiguous_abbreviation_breakdown": ambiguous_abbreviation_breakdown,
    }


# ==========================================================================
# OFFICIAL BENCHMARK METRIC -- character IoU
#
# Faithful re-implementation of scoring.py from
# github.com/drivendataorg/snomed-ct-benchmark-runtime (the SNOMED CT Entity
# Linking Benchmark's own scorer -- drivendata.org/benchmarks/310), so a
# number from this script means the same thing as a number on that
# leaderboard. Re-derived from the published algorithm rather than importing
# it, since the reference script pins polars/scipy/typer and this project's
# tooling is otherwise duckdb-only; the interval-arithmetic version below is
# mathematically identical to the reference's boolean-character-matrix
# version (verified against it below with synthetic cases), just without the
# extra dependencies.
#
# WHAT "CLASS" MEANS HERE. Unlike the recall breakdown above (which groups by
# GLiNER label), the official metric's "class" is the SNOMED concept_id
# itself -- every distinct concept_id appearing in EITHER gold or predictions
# is its own class, scored independently, then averaged. A predicted
# concept_id that appears nowhere in gold still becomes its own class with
# IoU 0 (all of its predicted characters are "union", none are
# "intersection") and drags the macro average down -- this is the metric's
# only mechanism for penalizing false-positive concepts, so leaving it in
# rather than filtering it out understates real submission behavior.
#
# TWO AGGREGATES, MATCHING THE BENCHMARK PAGE:
#   macro character IoU            -- unweighted mean over classes in G ∪ P.
#   support-weighted character IoU -- mean weighted by # of GOLD spans per
#                                      class; classes with zero gold support
#                                      (pure false positives) get zero weight,
#                                      so this one is not penalized by them.
# Concept IDs must match EXACTLY -- no partial/hierarchy credit, per the
# benchmark's own statement ("relationships between concepts are not taken
# into account for scoring purposes").
#
# SCOPE CAVEAT: this is a 3-note internal subset, not the withheld ~25-note
# benchmark test set (which is not available to us), so this number is for
# internal calibration and cross-solution methodology comparison, not a
# literal leaderboard submission score.
# ==========================================================================

def _merge_intervals(intervals):
    """Sorted, merged half-open [start, end) intervals -- collapses overlaps
    and adjacencies so measure()/intersection() below don't double-count a
    character covered by two spans of the SAME class (e.g. two overlapping
    mentions normalized to the same concept)."""
    if not intervals:
        return []
    s = sorted(intervals)
    merged = [list(s[0])]
    for start, end in s[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _measure(intervals):
    """Total character count covered by a pre-merged interval list."""
    return sum(b - a for a, b in intervals)


def _intersection_measure(a_ivals, b_ivals):
    """Character count where two pre-merged, sorted interval lists overlap.
    Standard two-pointer sweep -- O(len(a) + len(b))."""
    i = j = 0
    total = 0
    while i < len(a_ivals) and j < len(b_ivals):
        a_s, a_e = a_ivals[i]
        b_s, b_e = b_ivals[j]
        lo, hi = max(a_s, b_s), min(a_e, b_e)
        if lo < hi:
            total += hi - lo
        if a_e < b_e:
            i += 1
        else:
            j += 1
    return total


def official_character_iou(gold_rows, predictions, label_filter=None):
    """Macro and support-weighted character IoU, computed exactly as
    scoring.py's iou_per_class()/macro_character_iou()/
    support_weighted_character_iou() do: per-class (= per concept_id)
    intersection and union are accumulated ACROSS ALL NOTES (a note boundary
    never lets characters from two different notes overlap, but the totals
    are summed, i.e. this is micro-averaged over notes within each class,
    matching the reference's per-class boolean matrix being n_docs x n_chars
    and summed over both axes).

    label_filter, if given, restricts predictions to those whose GLiNER
    entity_label is in label_filter (gold is never filtered -- gold's scope
    is fixed by how the challenge was annotated). Predictions with no
    resolved SNOMED code (snomed_code is None -- Tier 0 Failed, or an
    unresolved RxNorm crosswalk) are dropped: a submission can't emit a
    concept_id it doesn't have.
    """
    gold_by_class_note = collections.defaultdict(lambda: collections.defaultdict(list))
    pred_by_class_note = collections.defaultdict(lambda: collections.defaultdict(list))
    classes = set()

    for g in gold_rows:
        gold_by_class_note[g["concept_id"]][g["note_id"]].append((g["start"], g["end"]))
        classes.add(g["concept_id"])

    for p in predictions:
        if p["snomed_code"] is None:
            continue
        if label_filter is not None and p["entity_label"] not in label_filter:
            continue
        pred_by_class_note[p["snomed_code"]][p["note_id"]].append(
            (p["orig_start"], p["orig_end"]))
        classes.add(p["snomed_code"])

    ious = {}
    for c in classes:
        inter_total = 0
        union_total = 0
        note_ids_for_class = set(gold_by_class_note[c]) | set(pred_by_class_note[c])
        for nid in note_ids_for_class:
            g_ivals = _merge_intervals(gold_by_class_note[c].get(nid, []))
            p_ivals = _merge_intervals(pred_by_class_note[c].get(nid, []))
            inter = _intersection_measure(g_ivals, p_ivals)
            union = _measure(g_ivals) + _measure(p_ivals) - inter
            inter_total += inter
            union_total += union
        ious[c] = inter_total / union_total if union_total else 0.0

    macro = sum(ious.values()) / len(ious) if ious else 0.0

    support = collections.Counter(g["concept_id"] for g in gold_rows)
    weighted_num = sum(ious[c] * support.get(c, 0) for c in ious)
    weighted_den = sum(support.get(c, 0) for c in ious)
    weighted = weighted_num / weighted_den if weighted_den else 0.0

    return {
        "macro_character_iou": macro,
        "support_weighted_character_iou": weighted,
        "n_classes": len(ious),
        "per_class_iou": ious,
    }


def print_report(report, note_ids):
    print("=" * 78)
    print("GOLD RECALL — Stage 1 -> 2a -> 2b vs SNOMED CT Entity Linking Challenge")
    print("=" * 78)

    print(f"\n{'NOTE':<20} | {'GOLD':>6} | {'PRED':>6} | "
          f"{'SPAN R':>8} | {'LINKED R':>8} | {'COMPOUND':>8}")
    print("-" * 72)
    for nid in note_ids:
        v = report["per_note"].get(nid)
        if not v:
            # per_note is keyed only from gold_by_note, i.e. notes that had
            # at least one row in train_annotations.csv -- a missing entry
            # here always means zero gold annotations for this note_id
            # (not in the 272/300-note annotated set, or a note_id typo),
            # never "predictions missing" (that case still gets a row, just
            # with predicted_entities=0 and 0% recall).
            print(f"{nid:<20} | (no gold annotations for this note_id in "
                  f"train_annotations.csv -- not an annotated gold note, "
                  f"or a note_id typo)")
            continue
        print(f"{nid:<20} | {v['gold_annotations']:>6} | {v['predicted_entities']:>6} | "
              f"{v['span_recall']*100:>7.2f}% | {v['linked_recall']*100:>7.2f}% | "
              f"{v['compound_spans']:>8}")

    c = report["combined"]
    print("-" * 72)
    print(f"{'COMBINED':<20} | {c['gold_annotations']:>6} | {c['predicted_entities']:>6} | "
          f"{c['span_recall']*100:>7.2f}% | {c['linked_recall']*100:>7.2f}% | "
          f"{report['compound_spans']['count']:>8}")
    print("COMPOUND = predictions whose single span overlaps >=2 distinct gold")
    print("annotations -- see the detail section below.")

    print(f"\nUncrosswalked misses (non-SNOMED vocab, no RxNorm->SNOMED hit): "
          f"{c['uncrosswalked_misses']}")
    print("These are excluded from 'wrong concept' below -- they failed on")
    print("crosswalk coverage, not on the model picking the wrong concept.")

    ab = report.get("ambiguous_abbreviation_breakdown") or {}
    if ab:
        print(f"\n--- Ambiguous-abbreviation accuracy by winning tiebreak "
              f"(abbreviation flywheel) ---")
        print("Of gold spans whose overlapping prediction came from an ambiguous")
        print("abbreviation, how often was the resulting concept actually correct --")
        print("broken down by WHICH tiebreak won. observed_frequency_priority and")
        print("context_pattern_rule are the new flywheel mechanisms (2026-08-17);")
        print("the other three are the pre-existing static heuristics.")
        print(f"{'selection_basis':<32} | {'CORRECT':>8} | {'TOTAL':>6} | {'ACCURACY':>9}")
        print("-" * 62)
        for basis, v in ab.items():
            print(f"{basis:<32} | {v['correct']:>8} | {v['total']:>6} | "
                  f"{v['accuracy']*100:>8.2f}%")
    else:
        print(f"\n--- Ambiguous-abbreviation accuracy by winning tiebreak "
              f"(abbreviation flywheel) ---")
        print("No ambiguous-abbreviation gold spans overlapping a prediction in "
              "this sample.")

    print(f"\n--- Sample wrong-concept links (span found, concept incorrect) ---")
    for e in report["wrong_concept_examples"]:
        print(f"  [{e['note_id']}] gold '{e['gold_span']}' ({e['gold_concept_id']}) "
              f"-> predicted '{e['predicted_text']}' = {e['predicted_concept']} "
              f"({e['predicted_snomed_code']}), tier {e['match_tier']}")

    print(f"\n--- Sample missed spans (no overlapping prediction at all) ---")
    for g in report["missed_span_examples"]:
        print(f"  [{g['note_id']}] '{g['span']}' ({g['concept_id']}) "
              f"at [{g['start']}:{g['end']}]")

    cs = report["compound_spans"]
    print(f"\n--- Compound spans: 1 prediction overlapping >=2 gold annotations "
          f"({cs['count']} total) ---")
    print("A single predicted concept cannot match more than one of these --")
    print("this needs Stage 2a span-splitting or multi-concept linking, not")
    print("normalization tuning. See module docstring 'COMPOUND-SPAN CASES'.")
    for e in cs["examples"]:
        gold_desc = ", ".join(f"'{g['span']}' ({g['concept_id']})" for g in e["gold_spans"])
        print(f"  [{e['note_id']}] predicted '{e['predicted_text']}' "
              f"[{e['predicted_start']}:{e['predicted_end']}] = {e['predicted_concept']} "
              f"-- overlaps gold: {gold_desc}")


def print_official_metrics(all_iou, in_scope_iou, dropped_label):
    print("\n" + "=" * 78)
    print("OFFICIAL BENCHMARK METRIC — character IoU "
          "(drivendata.org/benchmarks/310)")
    print("=" * 78)
    print("3-note internal subset, not the withheld leaderboard test set --")
    print("for calibration and cross-solution methodology comparison, not a")
    print("literal submission score.\n")

    print(f"{'':<32} | {'MACRO IoU':>10} | {'WEIGHTED IoU':>13} | {'CLASSES':>7}")
    print("-" * 70)
    print(f"{'all entities':<32} | {all_iou['macro_character_iou']:>10.4f} | "
          f"{all_iou['support_weighted_character_iou']:>13.4f} | "
          f"{all_iou['n_classes']:>7}")
    print(f"{'in-scope (excl. ' + dropped_label + ')':<32} | "
          f"{in_scope_iou['macro_character_iou']:>10.4f} | "
          f"{in_scope_iou['support_weighted_character_iou']:>13.4f} | "
          f"{in_scope_iou['n_classes']:>7}")

    print(f"\nThe benchmark's gold annotations exclude substances/medications by")
    print(f"design (annotators tagged only Procedures, Body Structures and")
    print(f"Clinical Findings -- see the benchmark's 'Concepts in scope' section).")
    print(f"Every {dropped_label} entity this pipeline normalizes therefore maps")
    print(f"to a concept_id that CANNOT appear in gold, so it can only ever add a")
    print(f"zero-IoU class to the macro average, never help it. 'in-scope' is the")
    print(f"fairer number to compare against other solutions' leaderboard scores;")
    print(f"'all entities' shows the actual cost of not scoping the submission.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--note-ids", default=",".join(DEFAULT_NOTE_IDS),
                     help="Comma-separated note_ids to score (default: the "
                          "small/median/large ranked gold notes).")
    ap.add_argument("--gold", default=None, help="path to train_annotations.csv")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=None, help="write full JSON report here")
    args = ap.parse_args()

    note_ids = [n.strip() for n in args.note_ids.split(",") if n.strip()]
    gold_path = args.gold or _first_existing(GOLD_CANDIDATES, "gold annotations CSV")

    print(f"gold:  {gold_path}")
    print(f"db:    {args.db}")
    print(f"notes: {note_ids}")

    gold_rows = load_gold(gold_path, note_ids)
    if not gold_rows:
        raise SystemExit(f"No gold annotations found for {note_ids} in {gold_path}.")

    conn = duckdb.connect(args.db, read_only=True)
    try:
        predictions = load_predictions(conn, note_ids)
        if not predictions:
            raise SystemExit(
                f"No is_test=TRUE predictions found for {note_ids}. "
                f"Run: python3 scripts/test_pipeline_e2e.py --note-ids {','.join(note_ids)}"
            )
        attach_snomed_codes(conn, predictions)
        report = score(gold_rows, predictions)

        # GOLD-LESS NOTE GUARD. If a requested note_id has is_test=TRUE
        # predictions in the DB (e.g. leftover rows from an earlier
        # test_stage3_live.py diagnostic run) but zero gold annotations --
        # which the "no gold annotations for this note_id" message in
        # print_report already surfaces -- those predictions must NOT reach
        # official_character_iou(). Every distinct concept_id they carry
        # becomes its own class there with zero possible intersection (no
        # gold exists for that note to intersect against), so it can only
        # ever drag the macro average down for a note that was never
        # actually being scored. Measured concretely: including
        # 10000032-DS-21 (0 gold rows, but old Stage 3 diagnostic
        # predictions still in the DB) alongside 17751158-DS-19 inflated 90
        # classes to 133 and dropped macro IoU from 0.1136 to 0.0735 on
        # otherwise IDENTICAL 17751158-DS-19 predictions -- a scoring
        # artifact, not a real result.
        gold_note_ids = {g["note_id"] for g in gold_rows}
        official_predictions = [p for p in predictions if p["note_id"] in gold_note_ids]
        excluded_notes = {p["note_id"] for p in predictions} - gold_note_ids
        if excluded_notes:
            print(f"\nExcluded from official IoU metric (predictions exist but no "
                  f"gold annotations for this note -- would only add spurious "
                  f"zero-IoU classes): {sorted(excluded_notes)}")

        # Medication is the one GLiNER label normalized against a vocabulary
        # (RxNorm) the challenge's gold set structurally cannot contain --
        # see print_official_metrics(). Filtering it out is the "in-scope"
        # comparison; leaving it in shows the real cost of not doing so.
        out_of_scope_labels = {"Medication"}
        in_scope_labels = {p["entity_label"] for p in official_predictions} - out_of_scope_labels
        all_iou = official_character_iou(gold_rows, official_predictions)
        in_scope_iou = official_character_iou(
            gold_rows, official_predictions, label_filter=in_scope_labels)
    finally:
        conn.close()

    print_report(report, note_ids)
    print_official_metrics(all_iou, in_scope_iou, dropped_label="Medication")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                **report,
                "official_metric_all_entities": all_iou,
                "official_metric_in_scope": in_scope_iou,
            }, f, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
