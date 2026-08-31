"""evaluation/mine_gliner_misses.py -- 2026-08-31: systematic mining of
GLiNER's own span-recall misses (gold entities GLiNER never extracted as
ANY span, not just mislabeled or boundary-shifted), on the TRAIN split
only.

WHY THIS EXISTS. Span recall is the single largest measured gap in this
pipeline (53.0% corpus-wide, 49.5% fresh-10, 58.6% fresh-5) -- nothing
downstream (Stage 2b linking, Stage 3 tier gate, the calibrator) can ever
recover an entity GLiNER never emitted a span for in the first place. A
hand-check of two real "wound" misses (2026-08-31 session) found two
DIFFERENT concrete failure modes: (1) a template physical-exam line
("Wound: steristrips in place, c/d/i...") where GLiNER extracts the
SAME-shaped short abbreviations right next to it (Abd, Ext) but skips the
plain word Wound -- likely dense clinical shorthand right after an
unabbreviated anchor word suppressing it; (2) narrative/process framing
("The wound dressings were monitored daily") where GLiNER extracts direct
mentions in the very next sentence but nothing here. This script
generalizes that hand-check into a systematic, repeatable pass across
every gold entity, not two anecdotes.

WHY TRAIN SPLIT ONLY. Same locked-test-split discipline fixed elsewhere
this session (evaluation/splits.py) -- this is diagnostic/pattern-mining
work that could inform a real prompt or extraction-rule change, so it
must never touch val or test.

WHAT THIS DOES NOT DO. It does not propose or build a fix. It reports
ranked, real, countable patterns with concrete examples so a fix (if any
pattern clears a real volume bar) can be scoped deliberately, the same
discipline mine_context_rules() (src/abbreviation_flywheel.py) already
established for the analogous disambiguation-stage mining.

Run: python3 -m evaluation.mine_gliner_misses [--out logs/gliner_miss_report.json]
"""
import argparse
import collections
import csv
import json
import os
import re
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

GOLD_NOTES_PATH = os.path.join(PROJECT_DIR, "data", "raw_notes", "gold_notes.csv")

_ABBREV_RE = re.compile(r"\b[A-Z]{2,5}\b|\b[a-z](?:/[a-z]){1,4}\b")  # ALL-CAPS 2-5 letter
                                                                     # tokens (WBC, HTN) OR
                                                                     # slash-shorthand (c/d/i)
_WINDOW = 40  # chars each side, for abbreviation-density and colon checks


def load_note_texts(note_ids: set) -> dict:
    """{note_id: raw text}, read from the fast 2.8MB gold-notes extract
    (data/raw_notes/gold_notes.csv) -- same convention as scripts/
    test_pipeline_e2e.py, avoids a 3.5GB discharge.csv scan."""
    texts = {}
    with open(GOLD_NOTES_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            nid = row.get("note_id")
            if nid in note_ids:
                texts[nid] = row.get("text", "")
    return texts


def _line_start(text, start):
    """True if `start` is the first non-whitespace character on its line."""
    line_begin = text.rfind("\n", 0, start) + 1
    return text[line_begin:start].strip() == ""


def _followed_by_colon(text, end):
    """True if the next non-whitespace character after the span is ':'."""
    i = end
    while i < len(text) and text[i] in " \t":
        i += 1
    return i < len(text) and text[i] == ":"


def classify_miss(text: str, start: int, end: int, span: str) -> dict:
    """Real, checkable pattern features for one missed gold entity --
    no judgment calls, just measurements, so the aggregation step ranks by
    real recurrence, not by which explanation sounds plausible."""
    window_after = text[end:min(len(text), end + _WINDOW)]
    window_before = text[max(0, start - _WINDOW):start]
    return {
        "span": span,
        "is_multiword": " " in span.strip(),
        "n_words": len(span.split()),
        "line_start": _line_start(text, start),
        "followed_by_colon": _followed_by_colon(text, end),
        "n_abbrev_after": len(_ABBREV_RE.findall(window_after)),
        "n_abbrev_before": len(_ABBREV_RE.findall(window_before)),
        "context_before": window_before[-30:],
        "context_after": window_after[:40],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"{PROJECT_DIR}/logs/gliner_miss_report.json")
    ap.add_argument("--top-n", type=int, default=25,
                    help="how many most-frequent missed surface forms to report")
    args = ap.parse_args()

    import duckdb
    from evaluation.cal_eval import GOLD_CANDIDATES, _first_existing
    from evaluation.splits import load_split
    from scripts.score_gold_recall import load_gold, overlaps
    from src.preprocessing import section_for_offset, segment_sections

    conn = duckdb.connect(f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb", read_only=True)

    processed = {r[0] for r in conn.execute(
        "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test = TRUE").fetchall()}
    train_notes = sorted(processed & load_split("train"))
    print(f"{len(processed)} processed notes total; {len(train_notes)} are train-split "
         f"(mining scope -- val/test never touched)")

    gold_path = _first_existing(GOLD_CANDIDATES, "gold")
    gold_rows = load_gold(gold_path, train_notes)
    gold_by_note = collections.defaultdict(list)
    for g in gold_rows:
        gold_by_note[g["note_id"]].append(g)
    print(f"{len(gold_rows)} gold annotations across {len(gold_by_note)} train notes")

    note_ph = ",".join("?" * len(train_notes))
    extracted_rows = conn.execute(f"""
        SELECT note_id, orig_start, orig_end FROM extracted_entities
        WHERE note_id IN ({note_ph})
          AND (superseded_by_split IS NULL OR superseded_by_split = FALSE)
          AND (superseded_by_growth IS NULL OR superseded_by_growth = FALSE)
    """, train_notes).fetchall()
    conn.close()
    extracted_by_note = collections.defaultdict(list)
    for note_id, s, e in extracted_rows:
        extracted_by_note[note_id].append((s, e))

    note_texts = load_note_texts(set(train_notes))
    missing_text = set(train_notes) - set(note_texts)
    if missing_text:
        print(f"WARNING: {len(missing_text)} train notes have no raw text in "
             f"gold_notes.csv (skipped for pattern context, still counted as gold "
             f"entities in the recall total): {sorted(missing_text)[:5]}...")

    n_gold, n_found, n_missed = 0, 0, 0
    misses = []
    surface_counts = collections.Counter()
    section_counts = collections.Counter()
    section_cache = {}

    for note_id, golds in gold_by_note.items():
        extracted = extracted_by_note.get(note_id, [])
        text = note_texts.get(note_id)
        for g in golds:
            n_gold += 1
            hit = any(overlaps(g["start"], g["end"], s, e) for s, e in extracted)
            if hit:
                n_found += 1
                continue
            n_missed += 1
            surface_counts[g["span"].strip().lower()] += 1
            if text is None:
                continue
            feats = classify_miss(text, g["start"], g["end"], g["span"])
            feats["note_id"] = note_id
            if note_id not in section_cache:
                section_cache[note_id] = segment_sections(text)
            sec_hit = section_for_offset(section_cache[note_id], g["start"])
            sec_name = sec_hit["name"] if sec_hit else None
            feats["section"] = sec_name
            section_counts[sec_name] += 1
            misses.append(feats)

    print(f"\n=== OVERALL ===")
    print(f"gold entities: {n_gold}  found: {n_found}  missed: {n_missed} "
         f"({n_missed/n_gold*100:.1f}% miss rate, i.e. span recall {n_found/n_gold*100:.1f}%)")

    # ------------------------------------------------------------------
    # Pattern clusters -- the actual output this script exists to produce.
    # ------------------------------------------------------------------
    def _pattern_tag(f):
        if f["followed_by_colon"] and f["line_start"]:
            return "template_header (line-start, colon-terminated)"
        if f["n_abbrev_after"] >= 2:
            return "dense_shorthand_after (>=2 abbrev/shorthand tokens within 40 chars)"
        if f["is_multiword"]:
            return "multiword_phrase"
        return "other_single_word"

    tag_counts = collections.Counter(_pattern_tag(f) for f in misses)
    print(f"\n=== PATTERN CLUSTERS (n={len(misses)} misses with usable raw text) ===")
    for tag, count in tag_counts.most_common():
        print(f"  {count:>5}  ({count/len(misses)*100:5.1f}%)  {tag}")

    print(f"\n=== TOP {args.top_n} MOST-FREQUENTLY-MISSED SURFACE FORMS ===")
    for span, count in surface_counts.most_common(args.top_n):
        print(f"  {count:>4}x  {span!r}")

    print(f"\n=== MISSES BY SECTION (top 15) ===")
    for sec, count in section_counts.most_common(15):
        print(f"  {count:>5}  {sec}")

    print(f"\n=== EXAMPLE MISSES PER PATTERN CLUSTER (up to 3 each) ===")
    by_tag = collections.defaultdict(list)
    for f in misses:
        by_tag[_pattern_tag(f)].append(f)
    for tag in tag_counts:
        print(f"\n  -- {tag} --")
        for f in by_tag[tag][:3]:
            print(f"     {f['note_id']}  {f['span']!r}  "
                 f"...{f['context_before']!r} [[{f['span']}]] {f['context_after']!r}...")

    report = {
        "n_gold": n_gold, "n_found": n_found, "n_missed": n_missed,
        "span_recall_train_split": n_found / n_gold if n_gold else None,
        "pattern_clusters": dict(tag_counts),
        "top_surface_forms": surface_counts.most_common(args.top_n),
        "section_counts": dict(section_counts),
        "n_train_notes": len(train_notes),
        "n_notes_missing_raw_text": len(missing_text),
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nfull report written to {args.out}")


if __name__ == "__main__":
    main()
