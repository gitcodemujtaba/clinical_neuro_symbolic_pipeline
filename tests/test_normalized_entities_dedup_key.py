"""tests/test_normalized_entities_dedup_key.py -- 2026-09-01: regression
test for the real production bug found and fixed this session (see
scripts/fix_normalized_entities_dedup_key.py's docstring for the full
diagnosis): normalized_entities used to be keyed on
(note_id, original_text, expanded_text, gliner_label), NOT entity_id, so
two distinct entity_ids sharing that tuple (two mentions of the same term
in one note -- verified live as the exact "HTN" case in note
10097089-DS-8) silently collapsed into one physical row, orphaning every
duplicate mention but the last-written one from normalized_entities
entirely.

Exercises the REAL src.normalization.orchestrator.process_and_normalize_
entities() write path against a throwaway on-disk DuckDB file -- not a
hand-rolled INSERT -- using the physexam_shorthand cold-start injection
point (a real, already-existing mechanism that skips the SapBERT/Athena
Tier 1-3 search entirely for a pre-verified concept) so this test needs no
GPU/model/vocabulary-table dependency, matching how this project already
tests other orchestrator.py code paths cheaply.

Run: python3 -m pytest tests/test_normalized_entities_dedup_key.py -v
"""
import os
import sys
import tempfile

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


def _duplicate_mention_entities():
    """Two distinct real entity_ids, one note, IDENTICAL
    (note_id, original_text, expanded_text, entity_label) -- exactly the
    shape that used to collapse into one normalized_entities row. Each
    carries a physexam_shorthand cold-start mapping so no real search runs.
    """
    shorthand = {"omop_concept_id": 38341003, "concept_name": "Hypertensive disorder",
                 "omop_domain": "Condition"}
    base = {
        "note_id": "TESTNOTE-1", "original_text": "HTN", "expanded_text": "Hypertension",
        "entity_label": "Condition", "gliner_confidence": 0.9, "is_test": True,
        "physexam_shorthand": shorthand,
    }
    return [
        {**base, "entity_id": "TESTNOTE-1-mention-A"},
        {**base, "entity_id": "TESTNOTE-1-mention-B"},
    ]


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")
        conn = duckdb.connect(db_path)

        from src.normalization.orchestrator import process_and_normalize_entities
        entities = _duplicate_mention_entities()
        process_and_normalize_entities(entities, conn, is_test=True)

        rows = conn.execute(
            "SELECT entity_id, omop_concept_id FROM normalized_entities "
            "WHERE note_id = 'TESTNOTE-1' ORDER BY entity_id"
        ).fetchall()
        conn.close()

        check("both duplicate mentions got a normalized_entities row "
              "(the actual bug: only 1 of 2 used to survive)",
              len(rows) == 2)
        entity_ids = {r[0] for r in rows}
        check("both real entity_ids are present, not one overwriting the other",
              entity_ids == {"TESTNOTE-1-mention-A", "TESTNOTE-1-mention-B"})
        check("both rows resolved to the correct concept",
              all(r[1] == 38341003 for r in rows))

    # Second case: the legitimate one-to-many shape (a multi-drug regimen
    # abbreviation normalizing to several expanded_text/concept pairs for
    # the SAME entity_id) must still produce multiple rows, not collapse to
    # one under the new UNIQUE(entity_id, expanded_text) key.
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test2.duckdb")
        conn = duckdb.connect(db_path)
        from src.normalization.orchestrator import process_and_normalize_entities

        drug_a = {"omop_concept_id": 1, "concept_name": "Drug A", "omop_domain": "Drug"}
        drug_b = {"omop_concept_id": 2, "concept_name": "Drug B", "omop_domain": "Drug"}
        entities = [
            {"note_id": "TESTNOTE-2", "original_text": "REGIMEN", "expanded_text": "Drug A",
             "entity_label": "Medication", "gliner_confidence": 0.9, "is_test": True,
             "entity_id": "TESTNOTE-2-regimen", "physexam_shorthand": drug_a},
            {"note_id": "TESTNOTE-2", "original_text": "REGIMEN", "expanded_text": "Drug B",
             "entity_label": "Medication", "gliner_confidence": 0.9, "is_test": True,
             "entity_id": "TESTNOTE-2-regimen", "physexam_shorthand": drug_b},
        ]
        process_and_normalize_entities(entities, conn, is_test=True)
        rows2 = conn.execute(
            "SELECT entity_id, expanded_text, omop_concept_id FROM normalized_entities "
            "WHERE note_id = 'TESTNOTE-2' ORDER BY expanded_text"
        ).fetchall()
        conn.close()
        check("one entity_id with two different expanded_text values keeps BOTH rows "
              "(the multi-drug-regimen case this fix must not regress)",
              len(rows2) == 2 and {r[1] for r in rows2} == {"Drug A", "Drug B"})

    print(f"normalized-entities-dedup-key tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_normalized_entities_dedup_key():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
