"""
scripts/verify_stage2_fixes.py — targeted confirmation that the 2026-08-11
fixes actually fire in a real pipeline run, not just in the stubbed unit
tests they were verified against before shipping (no DB/GPU access in the
dev sandbox those tests ran in).

Run AFTER re-ingesting the same 25-note test split (so extracted_entities /
normalized_entities reflect the new code -- ON CONFLICT DO UPDATE means a
rerun overwrites stale rows in place, it does not require a fresh DB). A
prior run of this script caught exactly this mistake: ingestion had not
actually been rerun, so every check below silently reported pre-fix
results (tier_reasons still showed the old low_gliner_confidence name,
0 rows showed any of the new signals, and the five newline spans all
showed superseded_by_split=False). If Check 2 shows low_gliner_confidence
> 0, STOP and rerun ingestion before trusting anything else here.

This script does NOT replace scripts/score_gold_recall.py. It cannot tell
you whether the laterality-floor change in find_compound_split() helped or
hurt overall IoU -- only a full score_gold_recall.py rerun, compared
against the existing baseline, can answer that. That comparison matters
here specifically because find_compound_split() has a documented history
of a measured regression from a similar relaxation (see its "REVERT
NOTICE" docstring in src/normalization.py) -- run that comparison, not
just this script, before treating any of these changes as safe.

What this DOES check, precisely:
  1. The five newline compound spans -- did any of them actually split, and
     into what.
  2. tier_reasons in normalized_entities -- do high_gliner_risk /
     short_token / isupper_abbreviation / alnum_mix actually appear, and
     does the OLD name (low_gliner_confidence) still show up anywhere
     (would mean a row didn't get overwritten by the rerun).
  3. assertion_engine in extracted_entities -- do delimiter-less Lab Test
     tokens (RR18-shaped) now get structured_result_override instead of
     whatever ConText/section-prior decided before.
  4. NEW -- strip_lab_value_suffix()'s candidate-arbitration rewrite: which
     entities actually resolved via the lab-value-suffix fallback, what
     candidate string won for each, and specifically whether any winning
     candidate itself ends in a digit (the SpO2-style case the rewrite
     exists for -- the OLD single-regex version could never produce this).

Run:
  python3 scripts/verify_stage2_fixes.py --note-ids <your 25 note ids>
"""

import argparse
import json
import os
import re

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

# Same five cases as scripts/measure_heuristic_and_boundary.py's
# NEWLINE_COMPOUND_CASES -- (note_id, orig_start, orig_end) of the PARENT
# (pre-split) entity as it was extracted before any of today's changes.
NEWLINE_PARENT_CASES = [
    ("10043750-DS-6", 1132, 1140, "S1S2/Abd -- expected to remain unsplit (sub-token fusion, documented as unaddressed)"),
    ("10371195-DS-9", 3216, 3232, "L/clavicular fx -- expected to split now (laterality-floor fix)"),
    ("10371195-DS-9", 3278, 3305, "L L2/transverse process fx -- expected to split now (laterality-floor fix)"),
    ("10848570-DS-12", 819, 860, "left acetabular/iliac crest fracture -- depends on vocab coverage, not this fix"),
    ("10860165-DS-24", 3794, 3822, "left lung base/atelectasis -- depends on vocab coverage, not this fix"),
]

# Same alnum-mix shape used in measure_heuristic_and_boundary.py, for
# spot-checking assertion_engine on delimiter-less Lab Test tokens.
_ALNUM_MIX_RE = re.compile(r"[A-Za-z][0-9]|[0-9][A-Za-z]")


def check_newline_splits(conn):
    print("=" * 78)
    print("CHECK 1 -- NEWLINE COMPOUND SPANS: DID THEY SPLIT")
    print("=" * 78)
    for note_id, start, end, label in NEWLINE_PARENT_CASES:
        parent = conn.execute("""
            SELECT entity_id, original_text, superseded_by_split
            FROM extracted_entities
            WHERE note_id = ? AND orig_start = ? AND orig_end = ? AND is_test = TRUE
        """, [note_id, start, end]).fetchone()
        print(f"\n  [{note_id}] [{start}:{end}] {label}")
        if parent is None:
            print("      PARENT ROW NOT FOUND -- rerun ingestion for this note before checking.")
            continue
        entity_id, orig_text, superseded = parent
        print(f"      parent original_text={orig_text!r} superseded_by_split={superseded}")
        if not superseded:
            print("      NOT SPLIT.")
            continue
        children = conn.execute("""
            SELECT e.original_text, e.orig_start, e.orig_end,
                   n.match_tier, n.omop_concept_id, n.omop_concept_name
            FROM extracted_entities e
            LEFT JOIN normalized_entities n
              ON n.entity_id = e.entity_id AND n.is_test = TRUE
            WHERE e.compound_split_of = ? AND e.is_test = TRUE
            ORDER BY e.orig_start
        """, [entity_id]).fetchall()
        if not children:
            print("      superseded_by_split=TRUE but no child rows found (compound_split_of mismatch?) -- investigate.")
            continue
        for child_text, c_start, c_end, tier, concept_id, concept_name in children:
            print(f"      SPLIT -> [{c_start}:{c_end}] {child_text!r} "
                  f"-> {tier} concept_id={concept_id} ({concept_name})")


def check_tier_reasons(conn, note_ids):
    print("\n" + "=" * 78)
    print("CHECK 2 -- tier_reasons: NEW SIGNALS PRESENT, OLD SIGNAL GONE")
    print("=" * 78)
    rows = conn.execute("""
        SELECT tier_reasons FROM normalized_entities
        WHERE is_test = TRUE AND note_id IN ({})
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    counts = {"high_gliner_risk": 0, "short_token": 0, "isupper_abbreviation": 0,
              "alnum_mix": 0, "low_gliner_confidence": 0}
    n_rows_with_reasons = 0
    for (raw,) in rows:
        if not raw:
            continue
        try:
            reasons = raw if isinstance(raw, list) else json.loads(raw)
        except (TypeError, ValueError):
            continue
        if reasons:
            n_rows_with_reasons += 1
        for key in counts:
            if key in reasons:
                counts[key] += 1

    print(f"  n normalized_entities rows checked: {len(rows)} "
          f"({n_rows_with_reasons} with at least one LOW reason)")
    for key, n in counts.items():
        flag = "  <-- SHOULD BE 0 (old signal name; nonzero means a stale, un-rerun row)" \
            if key == "low_gliner_confidence" and n else ""
        print(f"    {key:<24}: {n}{flag}")


def check_structured_result_override(conn, note_ids):
    print("\n" + "=" * 78)
    print("CHECK 3 -- assertion_engine ON DELIMITER-LESS LAB TOKENS")
    print("=" * 78)
    rows = conn.execute("""
        SELECT original_text, assertion_engine, assertion_status
        FROM extracted_entities
        WHERE is_test = TRUE AND note_id IN ({})
          AND entity_label = 'Lab Test'
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    alnum_mix_lab_tokens = [(t, eng, status) for t, eng, status in rows
                            if t and _ALNUM_MIX_RE.search(t)]
    print(f"  n Lab Test entities checked: {len(rows)}, "
          f"{len(alnum_mix_lab_tokens)} alnum-mix-shaped (RR18/S1S2-style)")
    n_overridden = sum(1 for _, eng, _ in alnum_mix_lab_tokens if eng == "structured_result_override")
    print(f"  of those, {n_overridden} now show assertion_engine='structured_result_override'")
    print("\n  Examples (up to 15):")
    for t, eng, status in alnum_mix_lab_tokens[:15]:
        print(f"    {t!r:<30} assertion_engine={eng!r:<35} assertion_status={status}")


def check_lab_suffix_arbitration(conn, note_ids):
    print("\n" + "=" * 78)
    print("CHECK 4 -- LAB-VALUE-SUFFIX CANDIDATE ARBITRATION (strip_lab_value_suffix)")
    print("=" * 78)
    rows = conn.execute("""
        SELECT original_text, normalized_from, match_tier, omop_concept_name
        FROM normalized_entities
        WHERE is_test = TRUE AND note_id IN ({})
          AND normalized_from LIKE 'value_stripped_from_%'
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    print(f"  n entities resolved via the lab-suffix fallback: {len(rows)}")

    digit_ending_wins = []
    for orig_text, normalized_from, match_tier, concept_name in rows:
        # normalized_from is "value_stripped_from_<source>:<winning candidate>"
        winning = normalized_from.split(":", 1)[1] if ":" in normalized_from else None
        if winning and winning[-1].isdigit():
            digit_ending_wins.append((orig_text, winning, match_tier, concept_name))

    print(f"  of those, {len(digit_ending_wins)} won with a candidate that itself ends in a "
          f"digit (the SpO2-style case -- unreachable under the old single-regex version).")

    print("\n  All resolutions (up to 20):")
    for orig_text, normalized_from, match_tier, concept_name in rows[:20]:
        print(f"    {orig_text!r:<30} {normalized_from:<40} -> {match_tier} ({concept_name})")

    if digit_ending_wins:
        print("\n  Digit-ending winners specifically (the fix's target case):")
        for orig_text, winning, match_tier, concept_name in digit_ending_wins:
            print(f"    {orig_text!r:<30} won on {winning!r:<15} -> {match_tier} ({concept_name})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None,
                     help="Comma-separated note_ids. Default: every note_id with "
                          "is_test=TRUE rows in extracted_entities.")
    ap.add_argument("--db", default=DB_PATH)
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
            raise SystemExit("No is_test=TRUE rows in extracted_entities. Re-run ingestion first.")

        check_newline_splits(conn)
        check_tier_reasons(conn, note_ids)
        check_structured_result_override(conn, note_ids)
        check_lab_suffix_arbitration(conn, note_ids)
    finally:
        conn.close()

    print("\n" + "=" * 78)
    print("Remember: this script does not check IoU. Re-run scripts/score_gold_recall.py")
    print("on the same 25 notes and compare against the 0.079/0.095 macro/weighted baseline")
    print("before treating the find_compound_split() change as safe.")
    print("=" * 78)


if __name__ == "__main__":
    main()
