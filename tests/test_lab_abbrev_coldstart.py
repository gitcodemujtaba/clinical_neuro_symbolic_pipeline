"""
tests/test_lab_abbrev_coldstart.py — src/lab_abbrev_coldstart.py's "cold
start" direct-injection for bare CBC/chemistry-panel abbreviations
(Creat/Hgb/RBC/Na/Hct/Cl/MCH/MCHC/RDW/HCO3/WBC/Phos/Calcium/AnGap/UreaN)
GLiNER never proposes as entities at any confidence -- see that module's
own docstring for the corpus-wide sizing evidence.

Real (not mocked) imports throughout, same discipline as
tests/test_physexam_shorthand.py -- src.assertion's medspacy pipeline
loads once per process.

Run: python3 -m pytest tests/test_lab_abbrev_coldstart.py -v
"""
import sys

import duckdb

from src.lab_abbrev_coldstart import (
    LAB_ABBREV_COLDSTART_TERMS,
    build_lab_abbrev_coldstart_entities,
    find_lab_abbrev_spans,
)


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # find_lab_abbrev_spans -- word-boundary, case-sensitive matching
    # ======================================================================
    text = "Creat trended up to 2.1. Na was stable. Pt denies chest pain."
    spans = find_lab_abbrev_spans(text)
    span_texts = [s["text"] for s in spans]
    check("finds 'Creat' as its own span", "Creat" in span_texts)
    check("finds 'Na' as its own span", "Na" in span_texts)
    check("does NOT match 'pain' (not a lab-abbrev term)", "pain" not in span_texts)
    check("spans are sorted by start offset",
          [s["start"] for s in spans] == sorted(s["start"] for s in spans))

    check("word-boundary matching does not fire mid-word ('Sodium' contains "
         "no standalone 'Na' token, wrong case anyway)",
          not any(s["text"] == "Na" for s in find_lab_abbrev_spans("Sodium bicarbonate given.")))

    lowercase_text = "na was low and the patient was in pain, unrelated to na."
    check("lowercase 'na' (not the gold-evidenced 'Na' case) does NOT match -- "
         "case-sensitive by design, narrowest collision surface the evidence supports",
          find_lab_abbrev_spans(lowercase_text) == [])

    check("CV (cardiovascular, NOT a lab test -- deliberately excluded, see "
         "module docstring) never matches",
          not any(s["text"] == "CV" for s in find_lab_abbrev_spans("CV: RRR, no m/r/g.")))

    both_case_text = "CREAT 1.8, Hct 32, HCT 33."
    both_case_spans = [s["text"] for s in find_lab_abbrev_spans(both_case_text)]
    check("both evidenced case-variants of the same term match independently "
         "(CREAT and Hct/HCT)",
          "CREAT" in both_case_spans and "Hct" in both_case_spans and "HCT" in both_case_spans)

    # ======================================================================
    # build_lab_abbrev_coldstart_entities -- full entity shape + assertion
    # ======================================================================
    entity_text = ("Creat trended up to 2.1 over the admission. Na was stable.\n"
                   "No evidence of Hgb drop. Pt denies chest pain.")
    entities = build_lab_abbrev_coldstart_entities(entity_text, "TEST-NOTE-1", existing_entities=[])
    by_text = {e["original_text"]: e for e in entities}

    check("all three lab terms in the sample text produce an entity",
          set(by_text.keys()) == {"Creat", "Na", "Hgb"})
    check("entity_label is always 'Lab Test'",
          all(e["entity_label"] == "Lab Test" for e in entities))
    check("confidence is 1.0 (curated, not a model guess)",
          all(e["confidence"] == 1.0 for e in entities))
    check("orig_start/orig_end equal exp_start/exp_end (no expansion step involved)",
          all(e["orig_start"] == e["exp_start"] and e["orig_end"] == e["exp_end"] for e in entities))
    check("below_threshold is always False",
          all(e["below_threshold"] is False for e in entities))
    check("Creat/Na (no negation cue nearby) are PRESENT",
          by_text["Creat"]["assertion_status"] == "PRESENT"
          and by_text["Na"]["assertion_status"] == "PRESENT")
    check("'No evidence of Hgb drop' -- Hgb correctly detected ABSENT via real "
         "assertion detection, batched per line",
          by_text["Hgb"]["assertion_status"] == "ABSENT")
    check("gliner_model_version records the cold-start provenance",
          all(e["gliner_model_version"] == "lab_abbrev_coldstart" for e in entities))

    # ======================================================================
    # overlap with an existing (e.g. GLiNER-found) entity is skipped
    # ======================================================================
    na_start = entity_text.index("Na")
    existing = [{"orig_start": na_start, "orig_end": na_start + 2}]
    entities_with_overlap = build_lab_abbrev_coldstart_entities(
        entity_text, "TEST-NOTE-1", existing_entities=existing)
    check("a span already covered by an existing entity is skipped, others still found",
          "Na" not in [e["original_text"] for e in entities_with_overlap]
          and "Creat" in [e["original_text"] for e in entities_with_overlap])

    # ======================================================================
    # empty input
    # ======================================================================
    check("no matching terms -> empty list, no crash",
          build_lab_abbrev_coldstart_entities("Patient reports chest pain.", "TEST-NOTE-2", []) == [])

    # ======================================================================
    # dictionary consistency + concept correctness against the live vocabulary
    # ======================================================================
    check("every term maps to a 2-tuple (concept_id, concept_name)",
          all(len(v) == 2 and isinstance(v[0], int) and isinstance(v[1], str)
              for v in LAB_ABBREV_COLDSTART_TERMS.values()))

    try:
        conn = duckdb.connect(
            "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder/db/kg2_lexical_store.duckdb",
            read_only=True)
        mismatches = []
        for term, (cid, name) in LAB_ABBREV_COLDSTART_TERMS.items():
            row = conn.execute(
                "SELECT concept_name, domain_id, standard_concept FROM athena_concept WHERE concept_id=?",
                [cid]).fetchone()
            if row is None or row[0] != name or row[1] != "Measurement" or row[2] != "S":
                mismatches.append((term, cid, name, row))
        conn.close()
        check(f"every concept_id in the dict resolves to its stated name, "
             f"Measurement domain, standard concept (live DB check): {mismatches}",
              not mismatches)
    except duckdb.IOException:
        pass  # DB locked by a concurrent batch job -- skip the live check, not a test failure

    print(f"lab-abbrev-coldstart tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_lab_abbrev_coldstart():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
