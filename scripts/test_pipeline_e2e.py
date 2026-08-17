import os
import sys
import duckdb
import csv
import json
import argparse
import time

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
sys.path.append(PROJECT_DIR)

# 2026-08-07: was three separate imports (src.preprocessing,
# src.entity_extraction, src.normalization) manually chained below in
# run_e2e(). Now delegates to src.clinical_pipeline.run_pipeline instead, so
# this e2e test exercises the real reusable orchestrator (previously an
# empty file -- Stage 1 -> 2a -> 2b only ever got chained together here, ad
# hoc) instead of duplicating its own copy of the wiring.
from src.clinical_pipeline import run_pipeline
from src.batch_status import clear_status, write_status

DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
NOTES_PATH = os.path.join(PROJECT_DIR, "data", "raw_notes", "discharge.csv")
# 2026-08-10: the 272 gold notes (data/snomed-ct-entity-linking-challenge-*/
# train_annotations.csv) live inside discharge.csv but a full scan of the
# 3.3GB file costs minutes per run. gold_notes.csv is that same 272-row
# subset, extracted once. --note-ids defaults to reading from here instead,
# since named-note requests are almost always gold notes being scored for
# recall. Falls back to discharge.csv if the extract hasn't been made yet.
GOLD_NOTES_PATH = os.path.join(PROJECT_DIR, "data", "raw_notes", "gold_notes.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Stage 1 -> 2a -> 2b pipeline end-to-end on notes from "
                     "data/raw_notes/discharge.csv (the MIMIC-IV-Note data), either the "
                     "first N notes in file order or a specific set of note_ids."
    )
    # 2026-08-07 addition: this script previously always processed exactly
    # one note (hardcoded via next(reader)), with no way to smoke-test
    # against more than that without editing the script. --notes generalizes
    # this instead of hardcoding a second fixed count (e.g. a literal
    # --notes-10 flag) -- `--notes 10` gets you the "process only ten notes"
    # behavior directly, and any other count works the same way without
    # needing a new flag added per desired count.
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--notes", type=int, default=1,
                        help="Number of notes to process, in file order (default: 1). "
                             "e.g. --notes 10 processes the first 10 notes. Ignored if "
                             "--note-ids is given.")
    # 2026-08-10 addition: --notes only ever gets the first N notes in file
    # order, which has no way to target specific notes (e.g. the small/
    # median/large notes picked by gold-annotation count for a scoreable
    # recall measurement before a multi-day corpus run). --note-ids selects
    # by note_id instead of position, and every note it selects is run in
    # FULL -- there is no "first one full, rest truncated" split here,
    # because a truncated note can't be scored for recall against the gold
    # annotations, which reference offsets across the whole note.
    group.add_argument("--note-ids", type=str, default=None,
                        help="Comma-separated note_id values to process instead of the "
                             "first N (e.g. --note-ids 17751158-DS-19,19442119-DS-15). "
                             "Every requested note runs in FULL, no 1000-char slicing. "
                             "Reads from data/raw_notes/gold_notes.csv if it exists "
                             "(fast -- a few MB), else falls back to discharge.csv.")
    parser.add_argument("--input", type=str, default=None,
                         help="Override the notes CSV path (default depends on mode -- "
                              "see --note-ids).")
    return parser.parse_args()


def run_e2e(num_notes: int = 1, note_ids=None, input_path: str = None):
    target_count = len(note_ids) if note_ids else num_notes

    print("=" * 80)
    if note_ids:
        print(f"🚀 RUNNING END-TO-END PIPELINE (STAGE 1 -> STAGE 2a -> STAGE 2b) "
              f"ON {target_count} NAMED NOTE(S)")
    else:
        print(f"🚀 RUNNING END-TO-END PIPELINE (STAGE 1 -> STAGE 2a -> STAGE 2b) "
              f"ON {target_count} NOTE(S)")
    print("=" * 80)

    if input_path:
        notes_path = input_path
    elif note_ids:
        notes_path = GOLD_NOTES_PATH if os.path.exists(GOLD_NOTES_PATH) else NOTES_PATH
    else:
        notes_path = NOTES_PATH
    print(f"Reading from {notes_path}")

    wanted = set(note_ids) if note_ids else None
    missing = set(note_ids) if note_ids else None

    conn = duckdb.connect(DB_PATH)

    total_entities = 0
    # total_after_dedup removed 2026-08-08 -- the orchestrator no longer drops
    # duplicate mentions (see run_pipeline's docstring).
    total_normalized = 0
    notes_processed = 0
    start_time = time.time()

    try:
        with open(notes_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for i, row in enumerate(reader):
                note_id = row.get('note_id', f'test_note_{i}')

                if wanted is not None:
                    if note_id not in wanted:
                        continue
                    missing.discard(note_id)
                elif i >= num_notes:
                    break

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
                # cost multiplies by --notes. Named notes (--note-ids) are
                # always full -- see the flag's help text for why.
                raw_text = full_text if (wanted is not None or i == 0) else full_text[:1000]

                # 2026-08-17: DB-independent progress -- see
                # src.batch_status's own docstring for why this matters
                # (DuckDB's write lock blocks even a read-only connection
                # for this whole run, so a UI can't ask the DB itself).
                write_status("stage1_2b", started_at=start_time,
                            notes_done=notes_processed, notes_total=target_count,
                            entities_done=total_entities, current_note_id=note_id)

                print(f"\n[{notes_processed + 1}/{target_count}] Running Stage 1 -> "
                      f"Stage 2a -> Stage 2b via src.clinical_pipeline.run_pipeline() "
                      f"on {note_id}...")
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

                if wanted is not None and not missing:
                    # Found every requested note_id -- no need to keep scanning
                    # the rest of the file (matters most on the discharge.csv
                    # fallback, where the file is 3.3GB).
                    break

        if wanted is not None:
            if missing:
                print(f"\n⚠️  {len(missing)}/{len(wanted)} requested note_id(s) not found "
                      f"in {notes_path}: {sorted(missing)}")
        elif notes_processed < num_notes:
            print(f"\n⚠️  Requested {num_notes} note(s) but {notes_path} only had "
                  f"{notes_processed} available.")

        write_status("stage1_2b", started_at=start_time,
                    notes_done=notes_processed, notes_total=target_count,
                    entities_done=total_entities, current_note_id=None)

        print("\n" + "=" * 80)
        print(f"📊 SUMMARY ACROSS {notes_processed} NOTE(S)")
        print("=" * 80)
        print(f"Total entities extracted: {total_entities}")
        print(f"Total normalized:         {total_normalized}")

    finally:
        clear_status("stage1_2b")
        conn.close()


if __name__ == "__main__":
    args = parse_args()
    note_ids = None
    if args.note_ids:
        note_ids = [n.strip() for n in args.note_ids.split(",") if n.strip()]
    run_e2e(num_notes=args.notes, note_ids=note_ids, input_path=args.input)
