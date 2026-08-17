"""
tests/test_snomed_crosswalk.py -- src/retrieval.py's
VocabularyRetriever.snomed_code_for_concept(), against a real (throwaway,
in-memory) DuckDB connection.

Reproduces the exact real-data shape that caught the 2026-08-17 bug: a
single RxNorm Drug concept ("warfarin") with MULTIPLE relationship rows to
SNOMED spanning Drug, Observation, and Procedure domains -- the old
`ORDER BY concept_id ASC LIMIT 1` picked an arbitrary one (a Procedure
concept, "Warfarin prophylaxis") instead of the actual drug-product
concept, silently corrupting every Medication-domain precision/recall
number that used this crosswalk. Confirms the fix (prefer 'RxNorm - SNOMED
eq' relationship type, then domain match, then lowest concept_id purely
for determinism) picks the right one.

Run: python3 -m pytest tests/test_snomed_crosswalk.py -v
"""
import os
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import VocabularyRetriever  # noqa: E402


def _build_conn():
    conn = duckdb.connect(":memory:")
    conn.sql("""
        CREATE TABLE athena_concept (
            concept_id BIGINT, concept_code VARCHAR, concept_name VARCHAR,
            domain_id VARCHAR, vocabulary_id VARCHAR, standard_concept VARCHAR
        );
    """)
    conn.sql("""
        CREATE TABLE athena_concept_relationship (
            concept_id_1 BIGINT, concept_id_2 BIGINT, relationship_id VARCHAR,
            invalid_reason VARCHAR
        );
    """)
    rows = [
        # (concept_id, concept_code, concept_name, domain_id, vocabulary_id, standard_concept)
        (1310149, "11289", "warfarin", "Drug", "RxNorm", "S"),
        # Real-shape distractors: all legitimately reference warfarin, none
        # of them is the concept gold/grading actually wants.
        (3544588, "791651000000100", "Warfarin prophylaxis", "Procedure", "SNOMED", None),
        (4187015, "372756006", "Warfarin", "Observation", "SNOMED", None),
        (36716147, "722045009", "Warfarin therapy", "Procedure", "SNOMED", None),
        # The correct target: an explicit RxNorm-SNOMED equivalence AND a
        # Drug-domain match, but NOT the lowest concept_id among candidates.
        (4174989, "48603004", "Warfarin-containing product", "Drug", "SNOMED", None),
    ]
    for r in rows:
        conn.execute("INSERT INTO athena_concept VALUES (?, ?, ?, ?, ?, ?)", r)

    rels = [
        (1310149, 3544588, "Mapped from"),
        (1310149, 4187015, "RxNorm - SNOMED eq"),   # eq, but wrong domain
        (1310149, 36716147, "Mapped from"),
        (1310149, 4174989, "RxNorm - SNOMED eq"),   # eq AND right domain -- wants this
    ]
    for c1, c2, rel in rels:
        conn.execute("INSERT INTO athena_concept_relationship VALUES (?, ?, ?, NULL)", [c1, c2, rel])
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
    vocab = VocabularyRetriever(conn)

    code = vocab.snomed_code_for_concept(1310149)
    check("picks the RxNorm-SNOMED-eq + domain-matched concept, not the "
          "lowest concept_id among all candidates",
          code == "48603004")
    check("does NOT pick the lowest-concept_id Procedure distractor "
          "(the pre-fix bug)",
          code != "791651000000100")

    # A SNOMED-vocabulary concept needs no crosswalk at all -- its own
    # concept_code IS the SNOMED code.
    conn.execute("INSERT INTO athena_concept VALUES (999, '12345678', "
                 "'Some finding', 'Condition', 'SNOMED', 'S')")
    check("a SNOMED-vocabulary concept returns its own code directly",
          vocab.snomed_code_for_concept(999) == "12345678")

    check("None concept_id returns None, not an error",
          vocab.snomed_code_for_concept(None) is None)

    check("an unmapped concept_id returns None, not an error",
          vocab.snomed_code_for_concept(424242) is None)

    check("result is cached (second call doesn't re-query)",
          vocab.snomed_code_for_concept(1310149) == "48603004")

    print(f"snomed-crosswalk tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_snomed_crosswalk():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
