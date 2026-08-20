"""
tests/test_physexam_shorthand.py — src/physexam_shorthand.py's "cold start"
direct-injection for physical-exam telegraphic shorthand GLiNER never
proposes as entities at all (VS/BP/HEENT/NAD/S-NT-ND/COR/... -- see that
module's own docstring for the corpus-wide sizing evidence).

Real (not mocked) imports throughout -- src.assertion's medspacy pipeline
loads once per process (a few seconds), same cost every other assertion-
touching test file in this repo already pays; no value in mocking it away
given the whole point of this module's most subtle bug (the colon-breaks-
negation-detection fix) is real interaction with that exact pipeline.

Run: python3 -m pytest tests/test_physexam_shorthand.py -v
"""
import sys

from src.physexam_shorthand import (
    PHYSEXAM_FINDING_TERMS,
    PHYSEXAM_HEADER_TERMS,
    PHYSEXAM_SHORTHAND_MATCH_BASIS,
    build_physexam_shorthand_entities,
    find_physexam_shorthand_spans,
    is_physexam_family_section,
)
from src.preprocessing import segment_sections


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # is_physexam_family_section / find_physexam_shorthand_spans
    # ======================================================================
    note1 = ("Physical Exam:\nGen: NAD\nAbd: S/NT/ND, wound dehiscence\n"
            "Ext: WNL\n\nPertinent Results:\nWBC-19.8\n")
    sections1 = segment_sections(note1)
    names1 = {s["name"] for s in sections1}
    check("segment_sections splits Gen/Abd/Ext as their own sections (the "
         "quirk this whole module works around)",
          {"Gen", "Abd", "Ext"} <= names1)

    spans1 = find_physexam_shorthand_spans(note1, sections1)
    texts1 = {s["text"] for s in spans1}
    check("finds the header terms themselves (Gen, Abd, Ext, Physical Exam)",
          {"Gen", "Abd", "Ext", "Physical Exam"} <= texts1)
    check("finds the compressed S/NT/ND triplet's NT and ND",
          {"NT", "ND"} <= texts1)
    check("finds NAD and WNL", {"NAD", "WNL"} <= texts1)
    check("does NOT match inside 'Pertinent Results' (not a physexam-family section)",
          not any(s["text"] == "WBC-19.8" for s in spans1))

    nt_span = next(s for s in spans1 if s["text"] == "NT")
    nd_span = next(s for s in spans1 if s["text"] == "ND")
    check("NT is flagged inherently_negated", nt_span["inherently_negated"] is True)
    check("ND is flagged inherently_negated", nd_span["inherently_negated"] is True)
    gen_span = next(s for s in spans1 if s["text"] == "Gen")
    check("Gen (a header term, not a finding) is NOT inherently_negated",
          gen_span["inherently_negated"] is False)

    # ======================================================================
    # COR / Chest / lowercase 'general' / 'Room air' / 'unremarkable'
    # (round 2 additions, found live on note 19895550-DS-7)
    # ======================================================================
    note2 = ("Physical Exam:\nVS; 98.6, 78, 110/70, 18, 97% Room air\n"
            "general: young in NAD\nHEENT: unremarkable\n"
            "Chest: breath sounds decreased\nCOR: RRR S1, S2\n"
            "abd: soft, NT, ND, +BS\nextrem: no edema\n")
    sections2 = segment_sections(note2)
    spans2 = find_physexam_shorthand_spans(note2, sections2)
    texts2 = {s["text"] for s in spans2}
    check("COR is recognized as its own header (round 2 addition)", "COR" in texts2)
    check("Chest is recognized as its own header (round 2 addition)", "Chest" in texts2)
    check("lowercase 'general' (not its own detected section -- stays inside "
         "Physical Exam's body) is still caught via PHYSEXAM_FINDING_TERMS",
          "general" in texts2)
    check("'Room air' (spelled out, not 'RA') is caught", "Room air" in texts2)
    check("'unremarkable' (HEENT's finding) is caught", "unremarkable" in texts2)
    check("RRR/S2/soft (inside COR's body, only reachable once COR is "
         "recognized as physexam-family) are all caught",
          {"RRR", "S2", "soft"} <= texts2)

    # ======================================================================
    # overlap avoidance -- build_physexam_shorthand_entities()
    # ======================================================================
    existing = [{"orig_start": nt_span["start"], "orig_end": nt_span["end"]}]
    built = build_physexam_shorthand_entities(note1, sections1, "test-note-1", existing)
    built_texts = {e["original_text"] for e in built}
    check("an entity already covered by an 'existing' (e.g. GLiNER) span is "
         "NOT re-injected",
          "NT" not in built_texts)
    check("everything else still gets injected despite one overlap",
          "ND" in built_texts and "Gen" in built_texts)

    # ======================================================================
    # entity dict shape -- every key extracted_entities/store_entities()
    # needs must be present (see src.entity_extraction.store_entities())
    # ======================================================================
    required_keys = {
        "entity_id", "note_id", "entity_label", "expanded_text", "original_text",
        "confidence", "orig_start", "orig_end", "exp_start", "exp_end",
        "assertion_status", "experiencer", "temporality", "assertion_cue",
        "assertion_engine", "section_name", "sentence_id", "local_context",
        "expansion_ambiguous", "candidate_expansions", "selection_basis",
        "gliner_model_version", "extraction_threshold", "below_threshold",
        "flat_ner", "crosses_sentence_boundary", "sentence_ids_spanned",
        "compound_split_of", "superseded_by_split", "grown_from",
        "superseded_by_growth", "possibly_truncated", "gliner_input_token_count",
        "physexam_shorthand",
    }
    all_present = all(required_keys <= set(e.keys()) for e in built)
    check("every injected entity dict carries every key store_entities() reads",
          all_present)
    check("confidence is 1.0 (pre-verified, not a GLiNER guess)",
          all(e["confidence"] == 1.0 for e in built))
    check("below_threshold is always False (these bypass the threshold gate entirely)",
          all(e["below_threshold"] is False for e in built))
    check("physexam_shorthand marker carries the pre-resolved concept",
          all(e["physexam_shorthand"]["omop_concept_id"] for e in built))

    nd_built = next(e for e in built if e["original_text"] == "ND")
    check("ND's injected entity is correctly ABSENT (inherently_negated override)",
          nd_built["assertion_status"] == "ABSENT")
    gen_built = next(e for e in built if e["original_text"] == "Gen")
    check("Gen's injected entity is correctly PRESENT",
          gen_built["assertion_status"] == "PRESENT")

    # ======================================================================
    # the colon-breaks-negation-detection bug (round 2) -- regression guard
    # ======================================================================
    note3 = "Physical Exam:\nextrem: no edema\n"
    sections3 = segment_sections(note3)
    built3 = build_physexam_shorthand_entities(note3, sections3, "test-note-3", [])
    edema = next((e for e in built3 if e["original_text"] == "edema"), None)
    check("'edema' entity was actually built", edema is not None)
    if edema:
        check("'no edema' (colon-prefixed header immediately before the "
             "negation cue) is correctly detected as ABSENT -- this is the "
             "exact live bug found on note 19895550-DS-7: a bare "
             "annotate_assertions() call on 'extrem: no edema' returns "
             "PRESENT (wrong) due to a medspacy/spaCy tokenization quirk "
             "where the colon breaks negation-trigger matching; stripping "
             "the colon-prefixed header before detection (see this "
             "function's own comment) is what fixes it",
              edema["assertion_status"] == "ABSENT")

    # No-colon-on-the-line case must still work normally (regression guard
    # for the strip-up-to-first-colon logic not over-stripping).
    note4 = "Physical Exam:\nno edema noted throughout\n"
    sections4 = segment_sections(note4)
    built4 = build_physexam_shorthand_entities(note4, sections4, "test-note-4", [])
    edema4 = next((e for e in built4 if e["original_text"] == "edema"), None)
    if edema4:
        check("a line with NO colon at all still gets correct negation detection",
              edema4["assertion_status"] == "ABSENT")

    # ======================================================================
    # match_basis constant + dictionary consistency
    # ======================================================================
    check("PHYSEXAM_SHORTHAND_MATCH_BASIS is the expected string",
          PHYSEXAM_SHORTHAND_MATCH_BASIS == "verified_physexam_shorthand")
    check("every header term maps to a 4-tuple",
          all(len(v) == 4 for v in PHYSEXAM_HEADER_TERMS.values()))
    check("every finding term maps to a 5-tuple",
          all(len(v) == 5 for v in PHYSEXAM_FINDING_TERMS.values()))
    check("no gliner_label in either dictionary is 'Qualifier' (would trip "
         "src.mollm_tier_gate.qualifier_fragment_precheck() and defeat the "
         "whole fast-path point)",
          all(v[-1] != "Qualifier" for v in PHYSEXAM_HEADER_TERMS.values())
          and all(v[-2] != "Qualifier" for v in PHYSEXAM_FINDING_TERMS.values()))

    print(f"physexam-shorthand tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_physexam_shorthand():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
