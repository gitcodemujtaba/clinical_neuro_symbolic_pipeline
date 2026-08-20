"""scripts/run_stage3_scoped_lab_fix.py -- 2026-08-20: Stage 3 tier-gate
run scoped to EXACTLY the entity_ids scripts/refix_uk_extension_lab_candidates.py
just re-normalized (the 26-term curated lab-abbreviation alias population,
2,484 entities), not the whole note or the whole corpus.

WHY A SEPARATE SCRIPT INSTEAD OF --note-ids. run_stage3_tier_gate.py's
--note-ids scoping is per-NOTE: it would still process every OTHER
undecided entity in each of those notes too, which is most of the corpus
(these lab abbreviations occur in the majority of notes) -- not the fast,
targeted before/after measurement this is for. Reuses the exact same
route_tier()/store_tier_decision() call pattern as the production script,
just filtered to entity_id IN (the target set) within each note's records.

Run: python3 scripts/run_stage3_scoped_lab_fix.py
"""
import sys
import time

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    from src.db_utils import connect_with_retry
    from src.mollm_ensemble import load_validation_records
    from src.mollm_tier_gate import build_clients, route_tier, store_tier_decision
    from src.mollm_tier_calibrator import ConsensusCalibrator, DEFAULT_MODEL_PATH
    from src.normalization.tier_retrieval import _LAB_TEST_ALIASES

    # 2026-08-20, rescoped under real time pressure: the full 26-term/2,484-
    # entity population measured ~15-20 sec/entity (the short_alphanumeric_
    # code_trap forces full 3-model ensemble for nearly all of them) -- at
    # that pace the full set would take ~11-12 hours. Narrowed to MCHC/RDW
    # specifically, the exact two terms this whole investigation was about,
    # for a fast, real, directly-relevant before/after number.
    terms = ["mchc", "rdw"]
    conn = connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=1800)
    placeholders = ",".join("?" * len(terms))
    rows = conn.execute(f"""
        SELECT DISTINCT entity_id, note_id FROM extracted_entities
        WHERE entity_label = 'Lab Test' AND is_test = TRUE
        AND lower(original_text) IN ({placeholders})
        AND (superseded_by_split IS NULL OR superseded_by_split = FALSE)
        AND (superseded_by_growth IS NULL OR superseded_by_growth = FALSE)
    """, terms).fetchall()
    # Further capped to 80 (from the full ~304 MCHC/RDW entities) per the
    # explicit "shrink sample size" instruction -- at ~15-20 sec/entity
    # (short_alphanumeric_code_trap forces full ensemble), 80 is ~20-25 min,
    # still a real, gradable sample rather than a token handful.
    rows = sorted(rows, key=lambda r: r[1])[:80]
    target_ids = {r[0] for r in rows}
    notes = sorted({r[1] for r in rows})
    print(f"{len(target_ids)} target entities across {len(notes)} notes")

    already_done = {r[0] for r in conn.execute(
        f"SELECT DISTINCT entity_id FROM mollm_tier_gate_decisions "
        f"WHERE entity_id IN ({','.join('?'*len(target_ids))})", list(target_ids)).fetchall()}
    print(f"{len(already_done)} already decided (unexpected -- refix just cleared these; "
          f"continuing with the rest)")
    conn.close()

    calibrator = ConsensusCalibrator.load(DEFAULT_MODEL_PATH, scoring_note_ids=notes)
    print(f"calibrator: {'fitted' if calibrator.model is not None else 'untrained (no-op)'}")
    clients = build_clients()

    n_processed, n_errors = 0, 0
    start_time = time.time()

    for note_idx, note_id in enumerate(notes, 1):
        note_conn = connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=1800)
        try:
            records = load_validation_records(note_conn, note_id, tier=None)
        finally:
            note_conn.close()
        todo = [r for r in records if r["entity_id"] in target_ids
               and r["entity_id"] not in already_done]
        if not todo:
            continue
        print(f"[note {note_idx}/{len(notes)}] {note_id}: {len(todo)} target entity(ies)")

        for rec in todo:
            elapsed_min = (time.time() - start_time) / 60
            try:
                decision = route_tier(
                    rec, clients=clients, calibrator=calibrator,
                    conn_factory=lambda: connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=1800))
                write_conn = connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=1800)
                try:
                    decision = store_tier_decision(decision, rec["entity_id"], note_id,
                                                   write_conn, is_test=True)
                finally:
                    write_conn.close()
            except Exception as exc:
                n_errors += 1
                print(f"    ERROR on {rec['original_text']!r}: {exc.__class__.__name__}: {exc}")
                continue
            n_processed += 1
            print(f"  [{elapsed_min:.1f}m] {rec['original_text']!r} -> "
                 f"tier={decision.get('tier')} routing={decision['mollm_routing_decision']}")

    print(f"\ndone: {n_processed} processed, {n_errors} errors, "
         f"{(time.time()-start_time)/60:.1f} min total")


if __name__ == "__main__":
    main()
