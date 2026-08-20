"""
tests/test_narrative_state_word_coldstart.py —
src/narrative_state_word_coldstart.py's "cold start" direct-injection for
common single-word clinical state descriptors (alert/improved/baseline/
warm/clinic) GLiNER never proposes as entities at any confidence.

Real (not mocked) imports throughout, same discipline as
tests/test_physexam_shorthand.py and tests/test_lab_abbrev_coldstart.py.

Run: python3 -m pytest tests/test_narrative_state_word_coldstart.py -v
"""
import sys

import duckdb

from src.narrative_state_word_coldstart import (
    NARRATIVE_STATE_WORD_MATCH_BASIS,
    NARRATIVE_STATE_WORD_TERMS,
    build_narrative_state_word_entities,
    find_narrative_state_word_spans,
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
    # find_narrative_state_word_spans -- word-boundary, case-sensitive
    # ======================================================================
    text = "Patient remains Alert and oriented. Improved since baseline. Warm extremities."
    spans = find_narrative_state_word_spans(text)
    span_texts = [s["text"] for s in spans]
    check("finds 'Alert'", "Alert" in span_texts)
    check("finds 'Improved'", "Improved" in span_texts)
    check("finds 'baseline'", "baseline" in span_texts)
    check("finds 'Warm'", "Warm" in span_texts)
    check("spans sorted by start offset",
          [s["start"] for s in spans] == sorted(s["start"] for s in spans))

    check("deliberately-excluded polysemous terms (pain/stable/negative/tender/"
         "masses/wound/procedure/support) never appear as keys in the dict",
          not ({"pain", "Pain", "stable", "Stable", "negative", "Negative",
               "tender", "Tender", "masses", "Masses", "wound", "Wound",
               "procedure", "Procedure", "support", "Support"}
              & set(NARRATIVE_STATE_WORD_TERMS.keys())))

    check("'edema' is deliberately excluded (owned by src.physexam_shorthand "
         "under a different concept -- see module docstring)",
          "edema" not in NARRATIVE_STATE_WORD_TERMS
          and "Edema" not in NARRATIVE_STATE_WORD_TERMS)

    check("word-boundary matching does not fire mid-word",
          not any(s["text"] == "warm" for s in find_narrative_state_word_spans("lukewarmish")))

    # ======================================================================
    # build_narrative_state_word_entities -- full entity shape + assertion
    # ======================================================================
    entity_text = ("Patient remains Alert and oriented.\n"
                   "Not improved since admission. Pt denies chest pain.")
    entities = build_narrative_state_word_entities(entity_text, "TEST-NOTE-1", existing_entities=[])
    by_text = {e["original_text"]: e for e in entities}

    check("only the two dictionary terms in the sample produce an entity",
          set(by_text.keys()) == {"Alert", "improved"})
    check("confidence is 1.0 (curated, not a model guess)",
          all(e["confidence"] == 1.0 for e in entities))
    check("orig_start/orig_end equal exp_start/exp_end (no expansion step)",
          all(e["orig_start"] == e["exp_start"] and e["orig_end"] == e["exp_end"] for e in entities))
    check("'Alert' (no negation cue nearby) is PRESENT",
          by_text["Alert"]["assertion_status"] == "PRESENT")
    check("'Not improved since admission' -- correctly detected ABSENT via real "
         "assertion detection (these terms are genuinely negatable, unlike "
         "physexam's inherently-negated NT/ND)",
          by_text["improved"]["assertion_status"] == "ABSENT")
    check("narrative_coldstart marker field carries the pre-verified concept",
          by_text["Alert"]["narrative_coldstart"]["omop_concept_id"] == 4086843)
    check("gliner_model_version records the cold-start provenance",
          all(e["gliner_model_version"] == "narrative_state_word_coldstart" for e in entities))

    # ======================================================================
    # overlap with an existing entity is skipped
    # ======================================================================
    alert_start = entity_text.index("Alert")
    existing = [{"orig_start": alert_start, "orig_end": alert_start + 5}]
    entities_with_overlap = build_narrative_state_word_entities(
        entity_text, "TEST-NOTE-1", existing_entities=existing)
    check("a span already covered by an existing entity is skipped, others still found",
          "Alert" not in [e["original_text"] for e in entities_with_overlap]
          and "improved" in [e["original_text"] for e in entities_with_overlap])

    check("no matching terms -> empty list, no crash",
          build_narrative_state_word_entities("Patient reports chest pain.", "TEST-NOTE-2", []) == [])

    # ======================================================================
    # dictionary consistency + concept correctness against the live vocabulary
    # ======================================================================
    check("every term maps to a 4-tuple", all(len(v) == 4 for v in NARRATIVE_STATE_WORD_TERMS.values()))
    check("NARRATIVE_STATE_WORD_MATCH_BASIS is the expected string",
          NARRATIVE_STATE_WORD_MATCH_BASIS == "verified_narrative_state_word")

    try:
        conn = duckdb.connect(
            "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder/db/kg2_lexical_store.duckdb",
            read_only=True)
        mismatches = []
        for term, (cid, name, domain, _label) in NARRATIVE_STATE_WORD_TERMS.items():
            row = conn.execute(
                "SELECT concept_name, domain_id, standard_concept FROM athena_concept WHERE concept_id=?",
                [cid]).fetchone()
            if row is None or row[0] != name or row[1] != domain or row[2] != "S":
                mismatches.append((term, cid, name, row))
        conn.close()
        check(f"every concept_id resolves to its stated name/domain, standard "
             f"concept (live DB check): {mismatches}",
              not mismatches)
    except duckdb.IOException:
        pass  # DB locked by a concurrent batch job -- skip, not a test failure

    print(f"narrative-state-word-coldstart tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_narrative_state_word_coldstart():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
