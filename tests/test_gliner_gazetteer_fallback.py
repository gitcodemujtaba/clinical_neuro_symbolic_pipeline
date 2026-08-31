"""
tests/test_gliner_gazetteer_fallback.py -- src/gliner_gazetteer_fallback.py's
recover_missed_entities(): per-term structural context rules, overlap
suppression, and the deliberate exclusions. Pure logic, no DB/model.

Run: python3 -m pytest tests/test_gliner_gazetteer_fallback.py -v
"""
import sys

from src.gliner_gazetteer_fallback import recover_missed_entities


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # Mg -- only recovered before a dash-terminated lab value
    # ======================================================================
    text = "Calcium-7.6* Phos-3.5 Mg-1.8\n"
    recovered = recover_missed_entities(text, existing_spans=[])
    mg_hits = [r for r in recovered if r["text"] == "Mg"]
    check("Mg recovered when immediately followed by a dash-terminated value",
          len(mg_hits) == 1 and mg_hits[0]["label"] == "Lab Test")

    text_no_context = "Mg is the chemical symbol for magnesium.\n"
    recovered2 = recover_missed_entities(text_no_context, existing_spans=[])
    check("Mg NOT recovered without the dash-value context",
          not any(r["text"] == "Mg" for r in recovered2))

    # ======================================================================
    # RA -- only recovered immediately after a '%'
    # ======================================================================
    text = "VS: 97.9, 88, 113/46, 16, 96% RA\n"
    recovered = recover_missed_entities(text, existing_spans=[])
    check("RA recovered when immediately preceded by '% '",
          any(r["text"] == "RA" and r["label"] == "Condition" for r in recovered))

    text_no_context = "He has a history of RA (rheumatoid arthritis).\n"
    recovered2 = recover_missed_entities(text_no_context, existing_spans=[])
    check("RA NOT recovered without the '%' vitals context",
          not any(r["text"] == "RA" for r in recovered2))

    # ======================================================================
    # CV -- only recovered at line-start, immediately before ':'
    # ======================================================================
    text = "GENERAL: WDWN\nCV: RRR, no murmurs\nRESP: CTAB\n"
    recovered = recover_missed_entities(text, existing_spans=[])
    check("CV recovered as a line-start, colon-terminated header",
          any(r["text"] == "CV" and r["label"] == "Procedure" for r in recovered))

    text_no_context = "The CV risk factors were reviewed with the patient.\n"
    recovered2 = recover_missed_entities(text_no_context, existing_spans=[])
    check("CV NOT recovered mid-sentence (not line-start/colon-terminated)",
          not any(r["text"] == "CV" for r in recovered2))

    # ======================================================================
    # EOMI / CTAB -- plain whole-word match, no context gate
    # ======================================================================
    text = "HEENT: PERRL, EOMI, no icterus\nLUNGS: CTAB, no wheezes\n"
    recovered = recover_missed_entities(text, existing_spans=[])
    check("EOMI recovered on a plain whole-word match",
          any(r["text"] == "EOMI" and r["label"] == "Condition" for r in recovered))
    check("CTAB recovered on a plain whole-word match",
          any(r["text"] == "CTAB" and r["label"] == "Condition" for r in recovered))

    # ======================================================================
    # The 8 additional _always-gated terms -- case-insensitive
    # ======================================================================
    text = ("Diff shows normal Monos and Eos. RDWSD "
           "elevated. CXR clear. Abdomen non-tender. Wheezes noted "
           "bilaterally. Ambulatory - Independent at baseline. Level of "
           "Consciousness intact.")
    recovered = recover_missed_entities(text, existing_spans=[])
    found_texts = {r["text"].lower() for r in recovered}
    for expected in ["monos", "eos", "rdwsd", "cxr", "non-tender", "wheezes"]:
        check(f"{expected!r} recovered case-insensitively",
              any(expected in t for t in found_texts))
    check("'glucose' is NOT in the gazetteer (excluded -- unreliable panel-context signal)",
          not any("glucose" in t for t in found_texts))
    check("'ambulatory - independent' (with varying spacing/case) recovered",
          any("ambulatory" in t.lower() and "independent" in t.lower() for t in found_texts))
    check("'level of consciousness' recovered case-insensitively",
          any("level of" in t.lower() and "consciousness" in t.lower() for t in found_texts))

    # ======================================================================
    # Overlap suppression -- never recover something GLiNER already found
    # ======================================================================
    text = "CV: RRR, no murmurs\n"
    cv_start = text.index("CV")
    existing = [(cv_start, cv_start + 2)]  # GLiNER already extracted "CV"
    recovered = recover_missed_entities(text, existing_spans=existing)
    check("Overlapping existing GLiNER span suppresses the gazetteer match",
          not any(r["text"] == "CV" for r in recovered))

    # ======================================================================
    # Provenance marker and score
    # ======================================================================
    text = "Mg-1.8\n"
    recovered = recover_missed_entities(text, existing_spans=[])
    check("every recovered entity carries the gazetteer provenance marker",
          all(r["_extraction_source"] == "gazetteer_fallback_gliner_miss" for r in recovered))
    check("every recovered entity has a fixed score of 1.0, not a model confidence",
          all(r["score"] == 1.0 for r in recovered))

    # ======================================================================
    # Deliberately excluded terms never appear, even in favorable text
    # ======================================================================
    text = "Patient has pain. Vitals stable, exam normal. Cultures negative."
    recovered = recover_missed_entities(text, existing_spans=[])
    check("deliberately excluded ambiguous terms (pain/stable/normal/negative/"
         "culture) are never recovered -- not in the gazetteer at all",
          len(recovered) == 0)

    print(f"gliner-gazetteer-fallback tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_gliner_gazetteer_fallback():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
