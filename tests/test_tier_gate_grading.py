"""
tests/test_tier_gate_grading.py -- evaluation/tier_gate_grading.py's
general-purpose grade_by_tier(), against a real (throwaway, in-memory)
DuckDB connection -- this module's whole point is a real SNOMED crosswalk
(src.retrieval.VocabularyRetriever) and real SQL joins, so a fake connection
would just be re-testing a mock instead of the actual logic.

Run: python3 -m pytest tests/test_tier_gate_grading.py -v
"""
import json
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.tier_gate_grading import grade_by_tier, plurality_candidate_index  # noqa: E402


def _build_conn():
    conn = duckdb.connect(":memory:")
    conn.sql("""
        CREATE TABLE athena_concept (
            concept_id BIGINT, concept_code VARCHAR, vocabulary_id VARCHAR
        );
    """)
    conn.sql("""
        CREATE TABLE extracted_entities (
            entity_id VARCHAR, note_id VARCHAR, original_text VARCHAR,
            entity_label VARCHAR, orig_start INT, orig_end INT
        );
    """)
    conn.sql("CREATE TABLE normalized_entities (entity_id VARCHAR, candidates JSON);")
    conn.sql("""
        CREATE TABLE mollm_tier_gate_decisions (
            entity_id VARCHAR, note_id VARCHAR, tier VARCHAR,
            final_candidate_index INTEGER, models JSON
        );
    """)
    # SNOMED concepts: 111 = correct "Chest pain", 222 = a wrong concept
    conn.execute("INSERT INTO athena_concept VALUES (111, '29857009', 'SNOMED')")
    conn.execute("INSERT INTO athena_concept VALUES (222, '99999999', 'SNOMED')")
    return conn


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    conn = _build_conn()

    # --- TIER_1_AUTO_VALIDATED: correct, clean-span ---
    conn.execute("INSERT INTO extracted_entities VALUES "
                 "('e1', 'n1', 'chest pain', 'Symptom', 0, 10)")
    conn.execute("INSERT INTO normalized_entities VALUES ('e1', ?)",
                [json.dumps([{"omop_concept_id": 111, "concept_name": "Chest pain"}])])
    conn.execute("INSERT INTO mollm_tier_gate_decisions VALUES "
                 "('e1', 'n1', 'TIER_1_AUTO_VALIDATED', 1, NULL)")

    # --- TIER_4_ENSEMBLE_SPLIT: plurality candidate is WRONG ---
    conn.execute("INSERT INTO extracted_entities VALUES "
                 "('e2', 'n1', 'LCX', 'Anatomy', 20, 23)")
    conn.execute("INSERT INTO normalized_entities VALUES ('e2', ?)",
                [json.dumps([{"omop_concept_id": 222, "concept_name": "Wrong concept"}])])
    conn.execute("INSERT INTO mollm_tier_gate_decisions VALUES ('e2', 'n1', "
                 "'TIER_4_ENSEMBLE_SPLIT', NULL, ?)", [json.dumps([
                     {"model": "a", "verdict": "SUPPORTED_1"},
                     {"model": "b", "verdict": "SUPPORTED_1"},
                     {"model": "c", "verdict": "NONE_CORRECT"},
                 ])])

    gold_path = "/tmp/cnsp_test_gold.csv"
    with open(gold_path, "w") as f:
        f.write("note_id,start,end,span,concept_id\n")
        f.write("n1,0,10,chest pain,29857009\n")     # matches e1's chosen concept
        f.write("n1,20,23,LCX,12345678\n")            # does NOT match e2's chosen concept

    import evaluation.tier_gate_grading as tgg
    original_candidates = tgg.GOLD_CANDIDATES
    tgg.GOLD_CANDIDATES = [gold_path]
    try:
        report = grade_by_tier(conn, ["n1"])
    finally:
        tgg.GOLD_CANDIDATES = original_candidates

    check("TIER_1_AUTO_VALIDATED graded as fully correct",
          report["TIER_1_AUTO_VALIDATED"]["clean"]["precision"] == 1.0)
    check("TIER_1_AUTO_VALIDATED has exactly 1 gradable clean-span decision",
          report["TIER_1_AUTO_VALIDATED"]["clean"]["n"] == 1)
    check("TIER_4_ENSEMBLE_SPLIT shadow precision correctly grades the plurality "
          "candidate as WRONG (0/1)",
          report["TIER_4_ENSEMBLE_SPLIT"]["clean"]["precision"] == 0.0)
    check("TIER_4_ENSEMBLE_SPLIT's graded record carries the real vote_counts",
          report["TIER_4_ENSEMBLE_SPLIT"]["clean"]["records"][0]["vote_counts"]
          == {"SUPPORTED_1": 2, "NONE_CORRECT": 1})

    check("an empty note_ids list returns an empty report, not an error",
          grade_by_tier(conn, []) == {})

    check("plurality_candidate_index: NONE_CORRECT plurality has no candidate",
          plurality_candidate_index(json.dumps([
              {"model": "a", "verdict": "NONE_CORRECT"},
              {"model": "b", "verdict": "NONE_CORRECT"},
              {"model": "c", "verdict": "SUPPORTED_1"},
          ]))[0] is None)

    os.remove(gold_path)

    print(f"tier-gate-grading tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_tier_gate_grading():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
