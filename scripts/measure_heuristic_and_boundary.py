"""
scripts/measure_heuristic_and_boundary.py — two pre-implementation checks
requested before finalizing the short-token Stage 3 routing trigger and
before assuming `crosses_sentence_boundary` already covers the newline-
containing compound-span cases.

CHECK 1 -- HEURISTIC TRIGGER RATE. Measures what fraction of all extracted
entities in a run would be flagged by each candidate short-token/
abbreviation heuristic (len<=4, isupper(), alphanumeric-mix), individually
and combined. Answers the isupper()-over-triggering risk directly: if it
flags a large fraction of the corpus (e.g. from all-caps template headers
like "PAST MEDICAL HISTORY"), it needs constraining or dropping before it's
wired into compute_confidence_tier() -- adding it blind risks reintroducing
the exact cost/latency problem the GLINER_CONFIDENCE_FLOOR fix was meant to
solve.

CHECK 2 -- BOUNDARY CONFIRMATION. For the five compound-span predictions
from the 25-note scripts/score_gold_recall.py run whose printed text
contains an embedded newline (candidates for "already caught by
crosses_sentence_boundary"), looks up their actual
crosses_sentence_boundary / sentence_ids_spanned values directly by
(note_id, orig_start, orig_end) -- the offsets printed alongside each
example uniquely identify the row, so this confirms the claim instead of
inferring it from newline presence in a terminal printout.

Run:
  python3 scripts/measure_heuristic_and_boundary.py --note-ids <25 note ids>
  python3 scripts/measure_heuristic_and_boundary.py --out reports/heuristic_report.json
"""

import argparse
import json
import os
import re
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")


# ==========================================================================
# Check 1 -- heuristic trigger rate
# ==========================================================================

# Adjacent letter-digit or digit-letter transition with no space -- catches
# T2DM, RR18, L4 (from "L4-L5"). Deliberately does not require a delimiter
# (unlike src/assertion.py's _LAB_NAME_VALUE, which requires a hyphen before
# the digit) -- this heuristic is for routing to Stage 3, not for the
# separate lab-panel extraction regex, so it's fine for it to be broader.
_ALNUM_MIX_RE = re.compile(r"[A-Za-z][0-9]|[0-9][A-Za-z]")

ISUPPER_BUDGET = 0.05  # share of corpus above which isupper() alone is judged too broad to ship as-is


def is_short(text: str) -> bool:
    return len(text) <= 4


def is_upper(text: str) -> bool:
    # str.isupper() is False for strings with no cased characters at all
    # (e.g. "12", "-", "L4-L5" -- the letters ARE upper but this still
    # evaluates True there, which is fine; the guard below is only to stop
    # a pure-punctuation/numeric token from spuriously counting as
    # "all-caps text"). Requires at least one alphabetic character.
    return bool(text) and text.isupper() and any(c.isalpha() for c in text)


def is_alnum_mix(text: str) -> bool:
    return bool(_ALNUM_MIX_RE.search(text))


def measure_heuristics(conn, note_ids):
    """Independent and combined hit rates for the three candidate
    short-token/abbreviation heuristics, over every extracted_entities row
    for the given notes (excludes superseded rows for the same
    double-count reason scripts/score_gold_recall.py excludes them)."""
    rows = conn.execute("""
        SELECT original_text FROM extracted_entities
        WHERE is_test = TRUE AND note_id IN ({})
          AND (superseded_by_split IS NULL OR superseded_by_split = FALSE)
          AND (superseded_by_growth IS NULL OR superseded_by_growth = FALSE)
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    texts = [r[0] for r in rows if r[0]]
    n = len(texts)

    flags = {"len<=4": [], "isupper": [], "alnum_mix": []}
    any_flag = []
    combo_counts = {}
    examples = {"len<=4": [], "isupper": [], "alnum_mix": []}

    for t in texts:
        f_len, f_up, f_mix = is_short(t), is_upper(t), is_alnum_mix(t)
        flags["len<=4"].append(f_len)
        flags["isupper"].append(f_up)
        flags["alnum_mix"].append(f_mix)
        any_flag.append(f_len or f_up or f_mix)

        if f_len and len(examples["len<=4"]) < 10:
            examples["len<=4"].append(t)
        if f_up and len(examples["isupper"]) < 10:
            examples["isupper"].append(t)
        if f_mix and len(examples["alnum_mix"]) < 10:
            examples["alnum_mix"].append(t)

        key = (f_len, f_up, f_mix)
        combo_counts[key] = combo_counts.get(key, 0) + 1

    def rate(flag_list):
        return (sum(flag_list) / n) if n else None

    return {
        "n_total_entities": n,
        "rate_len<=4": rate(flags["len<=4"]),
        "rate_isupper": rate(flags["isupper"]),
        "rate_alnum_mix": rate(flags["alnum_mix"]),
        "rate_any_combined": rate(any_flag),
        "n_any_combined": sum(any_flag),
        "examples": examples,
        "combo_breakdown": {
            f"len<=4={k[0]},isupper={k[1]},alnum_mix={k[2]}": v
            for k, v in sorted(combo_counts.items(), key=lambda kv: -kv[1])
        },
    }


# ==========================================================================
# Check 2 -- boundary confirmation for newline-containing compound spans
# ==========================================================================

# (note_id, orig_start, orig_end, human-readable label) -- pulled directly
# from the printed compound-span examples in the 25-note
# scripts/score_gold_recall.py run. The offsets are the actual lookup key;
# the label is display-only (backslash-n shown literally, not reconstructing
# the real embedded newline) and is not used to match anything.
NEWLINE_COMPOUND_CASES = [
    ("10043750-DS-6", 1132, 1140, "S1S2 / Abd (embedded newline)"),
    ("10371195-DS-9", 3216, 3232, "L / clavicular fx (embedded newline)"),
    ("10371195-DS-9", 3278, 3305, "L L2 / transverse process fx (embedded newline)"),
    ("10848570-DS-12", 819, 860, "left acetabular and iliac crest / fracture (embedded newline)"),
    ("10860165-DS-24", 3794, 3822, "left / lung base / atelectasis (embedded newlines)"),
]


def check_boundaries(conn):
    results = []
    for note_id, start, end, label in NEWLINE_COMPOUND_CASES:
        row = conn.execute("""
            SELECT crosses_sentence_boundary, sentence_ids_spanned, original_text
            FROM extracted_entities
            WHERE note_id = ? AND orig_start = ? AND orig_end = ? AND is_test = TRUE
        """, [note_id, start, end]).fetchone()
        if row is None:
            results.append({"note_id": note_id, "orig_start": start, "orig_end": end,
                            "label": label, "found": False})
            continue
        crosses, sentence_ids, actual_text = row
        results.append({
            "note_id": note_id, "orig_start": start, "orig_end": end, "label": label,
            "found": True, "actual_text": actual_text,
            "crosses_sentence_boundary": crosses,
            "sentence_ids_spanned": sentence_ids,
            "already_caught": bool(crosses),
        })
    return results


# ==========================================================================
# Reporting
# ==========================================================================

def print_report(heuristics, boundaries):
    print("=" * 78)
    print("CHECK 1 -- SHORT-TOKEN/ABBREVIATION HEURISTIC TRIGGER RATE")
    print("=" * 78)
    n = heuristics["n_total_entities"]
    print(f"n total entities: {n}\n")
    for label, key in [("len<=4", "rate_len<=4"), ("isupper (alpha-only)", "rate_isupper"),
                       ("alnum-mix", "rate_alnum_mix")]:
        r = heuristics[key]
        print(f"  {label:<24}: {r*100:5.2f}%" if r is not None else f"  {label:<24}: n/a")
    combined = heuristics["rate_any_combined"]
    print(f"  {'ANY (combined)':<24}: "
          f"{combined*100:5.2f}% ({heuristics['n_any_combined']}/{n})" if combined is not None else "n/a")

    r_up = heuristics["rate_isupper"]
    if r_up is not None:
        verdict = ("CONSTRAIN OR DROP -- exceeds "
                  f"{ISUPPER_BUDGET*100:.0f}% budget") if r_up > ISUPPER_BUDGET else \
                  f"within {ISUPPER_BUDGET*100:.0f}% budget, OK as proposed"
        print(f"\n  isupper() verdict: {verdict}")

    print("\n  Example matches per rule (up to 10 each):")
    for rule, exs in heuristics["examples"].items():
        print(f"    {rule}: {exs}")

    print("\n  Combination breakdown (len<=4, isupper, alnum_mix) -> count:")
    for combo, count in heuristics["combo_breakdown"].items():
        print(f"    {combo}: {count}")

    print("\n" + "=" * 78)
    print("CHECK 2 -- crosses_sentence_boundary CONFIRMATION FOR NEWLINE COMPOUND SPANS")
    print("=" * 78)
    for r in boundaries:
        if not r["found"]:
            print(f"  [{r['note_id']}] [{r['orig_start']}:{r['orig_end']}] "
                  f"expected '{r['label']}' -- NOT FOUND at that offset "
                  f"(row may not exist under is_test=TRUE with this key; check manually)")
            continue
        verdict = "ALREADY CAUGHT" if r["already_caught"] else "NOT CAUGHT -- needs new handling"
        print(f"  [{r['note_id']}] [{r['orig_start']}:{r['orig_end']}] '{r['actual_text']}'")
        print(f"      crosses_sentence_boundary={r['crosses_sentence_boundary']}, "
              f"sentence_ids_spanned={r['sentence_ids_spanned']} -- {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None,
                     help="Comma-separated note_ids for Check 1's corpus. Check 2 always "
                          "looks up its five fixed cases directly by (note_id, orig_start, "
                          "orig_end), independent of this arg. Default: every note_id with "
                          "is_test=TRUE rows in extracted_entities.")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    conn = duckdb.connect(args.db, read_only=True)
    try:
        if args.note_ids:
            note_ids = [n.strip() for n in args.note_ids.split(",") if n.strip()]
        else:
            note_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT note_id FROM extracted_entities WHERE is_test = TRUE"
            ).fetchall()]
        if not note_ids:
            raise SystemExit("No is_test=TRUE rows in extracted_entities. "
                             "Run scripts/test_pipeline_e2e.py first.")

        print(f"db:    {args.db}")
        print(f"notes: {note_ids}\n")

        heuristics = measure_heuristics(conn, note_ids)
        boundaries = check_boundaries(conn)
    finally:
        conn.close()

    print_report(heuristics, boundaries)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"heuristics": heuristics, "boundaries": boundaries}, f, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
