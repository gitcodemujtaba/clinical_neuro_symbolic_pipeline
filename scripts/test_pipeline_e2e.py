import os
import sys
import duckdb
import csv
import json
import argparse

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
sys.path.append(PROJECT_DIR)

# 2026-08-07: was three separate imports (src.preprocessing,
# src.entity_extraction, src.normalization) manually chained below in
# run_e2e(). Now delegates to src.clinical_pipeline.run_pipeline instead, so
# this e2e test exercises the real reusable orchestrator (previously an
# empty file -- Stage 1 -> 2a -> 2b only ever got chained together here, ad
# hoc) instead of duplicating its own copy of the wiring.
from src.clinical_pipeline import run_pipeline

DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
NOTES_PATH = os.path.join(PROJECT_DIR, "data", "raw_notes", "discharge.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Stage 1 -> 2a -> 2b pipeline end-to-end on the first N notes "
                     "from data/raw_notes/discharge.csv (the MIMIC-IV-Note data)."
    )
    # 2026-08-07 addition: this script previously always processed exactly
    # one note (hardcoded via next(reader)), with no way to smoke-test
    # against more than that without editing the script. --notes generalizes
    # this instead of hardcoding a second fixed count (e.g. a literal
    # --notes-10 flag) -- `--notes 10` gets you the "process only ten notes"
    # behavior directly, and any other count works the same way without
    # needing a new flag added per desired count.
    parser.add_argument("--notes", type=int, default=1,
                         help="Number of notes to process, in file order (default: 1). "
                              "e.g. --notes 10 processes the first 10 notes.")
    return parser.parse_args()


def run_e2e(num_notes: int = 1):
    print("=" * 80)
    print(f"🚀 RUNNING END-TO-END PIPELINE (STAGE 1 -> STAGE 2a -> STAGE 2b) "
          f"ON {num_notes} NOTE(S)")
    print("=" * 80)

    conn = duckdb.connect(DB_PATH)

    total_entities = 0
    # total_after_dedup removed 2026-08-08 -- the orchestrator no longer drops
    # duplicate mentions (see run_pipeline's docstring).
    total_normalized = 0
    notes_processed = 0

    try:
        with open(NOTES_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for i, row in enumerate(reader):
                if i >= num_notes:
                    break

                note_id = row.get('note_id', f'test_note_{i}')
                full_text = row.get('text', '')
                # 2026-08-07: at least the first note is always run in FULL --
                # this is the one result that should reflect real Stage 1/2
                # behavior on an actual discharge summary (abbreviation
                # expansion, offset reconciliation, and OMOP normalization all
                # behave differently over a full note vs. a 1000-char prefix,
                # e.g. expansions/entities near the truncation point get cut
                # off). Any additional notes (i > 0, only reachable with
                # --notes > 1) are still sliced to 1000 chars to keep the
                # smoke test's total runtime reasonable, since GLiNER + Tier-3
                # SapBERT scan over a full discharge summary is slow and that
                # cost multiplies by --notes.
                raw_text = full_text if i == 0 else full_text[:1000]

                print(f"\n[{i + 1}/{num_notes}] Running Stage 1 -> Stage 2a -> Stage 2b "
                      f"via src.clinical_pipeline.run_pipeline() on {note_id}...")
                # is_test=True: this is a smoke-test script, so rows written
                # to note_expansions/extracted_entities/normalized_entities
                # get flagged is_test=TRUE and can be purged before production.
                result = run_pipeline(note_id, raw_text, conn, is_test=True)
                normalized = result["normalized"]

                # 2026-08-08: 'entities_normalized_input' is gone -- the
                # orchestrator no longer drops duplicate mentions before
                # normalizing. Stage 2b now caches by (expanded_text, label)
                # and fans the result out to every entity_id, so entity count
                # and normalized count should now MATCH; a mismatch is a real
                # defect rather than expected dedup shrinkage.
                n_entities = len(result['entities'])
                n_normalized = len(normalized)
                total_entities += n_entities
                total_normalized += n_normalized
                notes_processed += 1

                n_sub = len(result.get('subthreshold_entities') or [])
                n_fallback = sum(1 for n in normalized
                                 if n.get('normalized_from', '').startswith('original'))
                n_low = sum(1 for n in normalized if n['confidence_tier_in'] == 'LOW')
                n_nonassert = sum(
                    1 for e in result['entities']
                    if e['assertion_status'] != 'PRESENT' or e['experiencer'] != 'PATIENT'
                )
                print(f"... {n_entities} entities accepted, {n_normalized} normalized, "
                      f"{n_low} routed LOW, {n_nonassert} non-asserted, "
                      f"{n_sub} retained below threshold, "
                      f"{n_fallback} recovered via original-form fallback.")
                if n_entities != n_normalized:
                    print(f"⚠️  entity/normalized count mismatch ({n_entities} vs "
                          f"{n_normalized}) -- dedup fan-out may be dropping rows.")

                print(f"{'DOCTOR WROTE':<20} | {'LABEL':<12} | "
                      f"{'OMOP MAPPING':<28} | {'TIER':<14} | {'ASSERTION':<10} | {'IN'}")
                print("-" * 105)
                assertion_by_id = {e['entity_id']: e for e in result['entities']}
                for n in normalized:
                    orig = n['original_text'].replace('\n', ' ')[:18]
                    omop = n['omop_concept_name'][:26] if n['omop_concept_name'] else "None"
                    ent = assertion_by_id.get(n['entity_id'], {})
                    print(f"{orig:<20} | {n['gliner_label']:<12} | {omop:<28} | "
                          f"{n['match_tier']:<14} | {ent.get('assertion_status', '?'):<10} | "
                          f"{n['confidence_tier_in']}")

        if notes_processed < num_notes:
            print(f"\n⚠️  Requested {num_notes} note(s) but {NOTES_PATH} only had "
                  f"{notes_processed} available.")

        print("\n" + "=" * 80)
        print(f"📊 SUMMARY ACROSS {notes_processed} NOTE(S)")
        print("=" * 80)
        print(f"Total entities extracted: {total_entities}")
        print(f"Total normalized:         {total_normalized}")

    finally:
        conn.close()


if __name__ == "__main__":
    args = parse_args()
    run_e2e(num_notes=args.notes)
