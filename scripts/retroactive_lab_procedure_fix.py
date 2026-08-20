"""scripts/retroactive_lab_procedure_fix.py -- 2026-08-19 retroactive fix for
the "MCV/MCHC/RDW problem" (see docs/2026-08-19_Lab_Procedure_Vs_Observable_
Entity_Finding.md for the full writeup).

Scope: every Lab-Test-labeled entity in a note that was already fully
processed through Stage 2b/3 BEFORE src/normalization/tier_retrieval.py's
_prefer_lab_procedure_over_observable() started tagging its winning
candidate's match_basis="lab_procedure_preferred", and before
src/mollm_tier_gate.py's _lab_procedure_fast_path() existed to bypass the
ensemble for that tag. Those notes' stored normalized_entities.candidates
predate both changes, so simply re-running Stage 3 on them would resume-skip
straight past every already-decided entity without ever re-evaluating it.

Steps:
  1. Pull every Lab-Test extracted_entities row whose note already has SOME
     normalized_entities row (i.e. already Stage-2b-processed), reconstruct
     the entity dicts process_and_normalize_entities() expects.
  2. Re-run process_and_normalize_entities() on that whole set in ONE call --
     normalize_entity()'s cache_key doesn't depend on note_id, so identical
     (text, label) pairs across different notes share one computation, a
     free efficiency win, not a correctness risk (the search itself has no
     per-note state).
  3. Re-read normalized_entities for those same entity_ids; the AFFECTED
     subset is whichever now has candidates[0].match_basis ==
     "lab_procedure_preferred" (i.e. the fix actually changed something for
     that specific entity -- most Lab-Test entities are unaffected and stay
     exactly as they were).
  4. Delete the AFFECTED entity_ids' stale mollm_tier_gate_decisions rows,
     so Stage 3's resume-check will recompute them instead of skipping.
  5. Print the distinct note_ids touched, for the caller to hand to
     scripts/run_stage3_tier_gate.py --note-ids.

Does NOT re-run Stage 3 itself -- kept as a separate, explicit step (see the
wrapper shell script this is launched from) so the DB-write phase here and
the Stage-3 LLM-ensemble phase after it are cleanly separated, same
discipline as every other batch script in this repo.
"""
import json
import sys

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.db_utils import connect_with_retry  # noqa: E402
from src.normalization.orchestrator import process_and_normalize_entities  # noqa: E402

DB_PATH = f"{PROJECT_DIR}/db/kg2_lexical_store.duckdb"

ENTITY_COLUMNS = [
    "note_id", "entity_label", "expanded_text", "original_text", "confidence",
    "orig_start", "orig_end", "exp_start", "exp_end", "is_test", "entity_id",
    "assertion_status", "experiencer", "temporality", "assertion_cue",
    "assertion_cue_start", "assertion_cue_end", "assertion_cue_category",
    "assertion_engine", "section_name", "sentence_id", "local_context",
    "expansion_ambiguous", "candidate_expansions", "gliner_model_version",
    "extraction_threshold", "below_threshold", "flat_ner",
    "crosses_sentence_boundary", "sentence_ids_spanned", "compound_split_of",
    "superseded_by_split", "grown_from", "superseded_by_growth",
    "selection_basis", "possibly_truncated", "gliner_input_token_count",
]


def main():
    conn = connect_with_retry(DB_PATH, read_only=False, max_wait_seconds=1800)
    try:
        cols_sql = ", ".join(f"e.{c}" for c in ENTITY_COLUMNS)
        rows = conn.execute(f"""
            SELECT {cols_sql}
            FROM extracted_entities e
            WHERE e.entity_label = 'Lab Test'
            AND e.note_id IN (SELECT DISTINCT note_id FROM normalized_entities)
        """).fetchall()
        entities = []
        for r in rows:
            d = dict(zip(ENTITY_COLUMNS, r))
            for jsoncol in ("candidate_expansions", "sentence_ids_spanned"):
                if isinstance(d.get(jsoncol), str):
                    try:
                        d[jsoncol] = json.loads(d[jsoncol])
                    except (TypeError, ValueError):
                        pass
            entities.append(d)

        print(f"Reprocessing {len(entities)} Lab-Test entities across "
              f"{len({e['note_id'] for e in entities})} already-processed notes...")
        if not entities:
            print("Nothing to do.")
            return

        process_and_normalize_entities(entities, conn, is_test=True)

        entity_ids = [e["entity_id"] for e in entities]
        ph = ",".join("?" * len(entity_ids))
        post = conn.execute(f"""
            SELECT entity_id, note_id, candidates
            FROM normalized_entities
            WHERE entity_id IN ({ph})
        """, entity_ids).fetchall()

        affected = []
        for entity_id, note_id, candidates_json in post:
            cands = candidates_json
            if isinstance(cands, str):
                cands = json.loads(cands)
            if cands and cands[0].get("match_basis") == "lab_procedure_preferred":
                affected.append((entity_id, note_id))

        print(f"{len(affected)} entities actually changed (now tagged "
              f"lab_procedure_preferred) out of {len(entities)} re-normalized.")

        if affected:
            affected_ids = [a[0] for a in affected]
            ph2 = ",".join("?" * len(affected_ids))
            deleted = conn.execute(
                f"DELETE FROM mollm_tier_gate_decisions WHERE entity_id IN ({ph2})",
                affected_ids)
            print(f"Deleted stale tier-gate decisions for {len(affected_ids)} "
                  f"affected entities (Stage 3 resume-check will recompute them).")

            affected_notes = sorted({a[1] for a in affected})
            print(f"\n{len(affected_notes)} notes touched:")
            print(",".join(affected_notes))
        else:
            print("No entities changed -- nothing further to do.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
