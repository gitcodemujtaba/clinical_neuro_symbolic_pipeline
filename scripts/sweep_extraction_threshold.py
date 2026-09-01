"""scripts/sweep_extraction_threshold.py -- 2026-09-01: real, zero-new-
inference sweep of EXTRACTION_THRESHOLD (src/entity_extraction.py,
currently 0.35) against the current, grown corpus.

WHY THIS IS FREE. extracted_entities already stores every GLiNER
prediction down to SUBTHRESHOLD_FLOOR=0.35 (flagged below_threshold=TRUE
for anything under the live EXTRACTION_THRESHOLD), with its real
gliner_confidence. Re-grading at a different cutoff is a pure
re-filter + re-score against gold -- no model call, no re-extraction.

METHOD. For each candidate threshold T, "accepted" = every
extracted_entities row (not superseded) with gliner_confidence >= T.
Span recall/precision against gold computed the same overlap-based way
scripts/score_gold_recall.py already does (reused via its own
load_gold/overlaps, not reimplemented). This measures the SPAN-level
question specifically (Stage 2a alone) -- it does not touch Stage 2b
linking, matching the existing "span recall is a pure Stage 2a number"
discipline documented in score_gold_recall.py's own module docstring.

Run: python3 scripts/sweep_extraction_threshold.py
"""
import sys
import collections

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def main():
    from src.db_utils import connect_with_retry
    from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing
    from scripts.score_gold_recall import load_gold, overlaps

    conn = connect_with_retry(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb", read_only=True)

    note_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test = TRUE").fetchall()]
    print(f"{len(note_ids)} processed test notes in scope")

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, note_ids)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)
    n_gold = len(gold_rows)
    print(f"{n_gold} gold annotations across {len(gold_by_note)} notes\n")

    # Pull every real GLiNER-sourced row (not gazetteer/cold-start fallback
    # injections, which carry their own fixed score=1.0/verified provenance
    # and shouldn't be swept by a GLiNER confidence threshold at all) with
    # its confidence and span, once.
    rows = conn.execute("""
        SELECT note_id, orig_start, orig_end, confidence
        FROM extracted_entities
        WHERE is_test = TRUE
          AND (superseded_by_split IS NULL OR superseded_by_split = FALSE)
          AND (superseded_by_growth IS NULL OR superseded_by_growth = FALSE)
          AND (extraction_source IS NULL OR extraction_source = 'gliner')
          AND confidence IS NOT NULL
    """).fetchall()
    conn.close()
    print(f"{len(rows)} real GLiNER-scored rows to sweep\n")

    by_note = collections.defaultdict(list)
    for note_id, s, e, conf in rows:
        by_note[note_id].append((s, e, conf))

    print(f"{'threshold':>10} {'n_accepted':>11} {'span_recall':>12} {'span_precision':>15}")
    for t in THRESHOLDS:
        n_accepted = 0
        n_correct_pred = 0  # predictions overlapping >=1 gold span
        gold_covered = set()  # (note_id, gold_index) pairs covered
        for note_id, preds in by_note.items():
            accepted = [(s, e) for s, e, conf in preds if conf >= t]
            n_accepted += len(accepted)
            golds = gold_by_note.get(note_id, [])
            for s, e in accepted:
                hit = False
                for gi, g in enumerate(golds):
                    if overlaps(s, e, g["start"], g["end"]):
                        hit = True
                        gold_covered.add((note_id, gi))
                if hit:
                    n_correct_pred += 1
        span_recall = len(gold_covered) / n_gold if n_gold else 0.0
        span_precision = n_correct_pred / n_accepted if n_accepted else 0.0
        print(f"{t:>10.2f} {n_accepted:>11} {span_recall*100:>11.2f}% {span_precision*100:>14.2f}%")


if __name__ == "__main__":
    main()
