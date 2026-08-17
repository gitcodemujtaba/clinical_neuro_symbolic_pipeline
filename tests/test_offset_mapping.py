"""
tests/test_offset_mapping.py

Unit tests for the offset/context/linking logic added or changed on
2026-08-08. This file was 0 lines despite docs/Implementation_Checklist.md
flagging offset mapping as "the most bug-sensitive part of the working code so
far" -- and the 2026-08-08 changes added three more pieces of offset
arithmetic (section lookup, sentence-bounded context windows, relation
endpoint overlap linking), so it needed to stop being empty.

WHY THESE FUNCTIONS ARE IMPORTED THE WAY THEY ARE: src/preprocessing.py,
src/entity_extraction.py and src/extraction.py load scispaCy, GLiNER-BioMed
and GLiNER-relex at module import time (multi-GB downloads, minutes of
startup). Importing them here would make the test suite unrunnable anywhere
except a fully provisioned EC2 box, which defeats the point of having fast
unit tests for pure arithmetic. _load_pure_functions() therefore parses each
module's AST and executes ONLY its module-level constants and the specific
pure functions under test, with no imports and no model loading.

That is a deliberate trade-off with a real downside worth stating: these tests
exercise the functions in isolation, so they will NOT catch a break caused by
a change elsewhere in the module (a renamed constant used by a function not
extracted here, for example). They are a fast correctness check on the
arithmetic, not a substitute for scripts/test_pipeline_e2e.py against real
models and a real DuckDB.

Run: python3 -m pytest tests/test_offset_mapping.py -v
     (or: python3 tests/test_offset_mapping.py for a plain-output run)
"""

import ast
import csv
import hashlib
import os
import re
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLD_NOTES = os.path.join(
    PROJECT_ROOT, "data", "evaluaiton-dataset",
    "snomed-ct-entity-linking-challenge-1.2.0", "train_notes.csv",
)


def _load_pure_functions(module_filename: str, wanted: set, extra_globals: dict = None) -> dict:
    """Executes only the named functions plus module-level constants from a
    source file, without running its imports. See this module's docstring."""
    path = os.path.join(SRC_DIR, module_filename)
    tree = ast.parse(open(path, encoding="utf-8").read())

    def _is_literal_assign(node):
        """Keeps only constant module-level assignments.

        Module-level statements include things like
        `model = GLiNER.from_pretrained(...)`, which is exactly the multi-GB
        model load this extractor exists to avoid -- so assignments whose
        right-hand side is a call (or anything else non-literal) are skipped
        rather than executed."""
        if not isinstance(node, ast.Assign):
            return False
        # re.compile(...) is explicitly allowed -- SECTION_HEADER_RE is a
        # module-level compiled pattern that segment_sections() depends on, and
        # compiling a regex has none of the cost or side effects that make
        # arbitrary calls unsafe to execute here.
        if (isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "compile"
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "re"):
            return True
        try:
            ast.literal_eval(node.value)
            return True
        except (ValueError, SyntaxError, TypeError):
            return False

    body = [
        n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted) or _is_literal_assign(n)
    ]
    ns = {"re": re, "hashlib": hashlib, "os": os}
    ns.update(extra_globals or {})
    exec(compile(ast.Module(body=body, type_ignores=[]), f"<{module_filename}>", "exec"), ns)
    return ns


PRE = _load_pure_functions(
    "preprocessing.py",
    # 2026-08-13: the four helpers below are the tiebreak chain
    # expand_text_and_track_offsets() calls -- _numeric_context_kind and
    # _select_by_numeric_context arrived with the numeric-context fix, and
    # _select_by_groundability (plus its _omop_domain_for_meaning helper) with
    # the "fx" -> fractions fix earlier the same day. Neither change added them
    # to this set, so every test in this file raised NameError at
    # expand_text_and_track_offsets()'s first ambiguous abbreviation -- a whole
    # suite silently red, discovered during the P0-P5 verification pass rather
    # than reported by it. They are pure functions (the conn-touching ones take
    # conn=None and return early), so extracting them costs nothing.
    {"segment_sections", "section_for_offset", "expand_text_and_track_offsets",
     "_numeric_context_kind", "_select_by_numeric_context",
     "_select_by_groundability", "_omop_domain_for_meaning"},
)
EXT = _load_pure_functions(
    "entity_extraction.py",
    {"map_offsets_to_original", "make_entity_id", "find_sentence", "build_local_context"},
)
REL = _load_pure_functions(
    "extraction.py",
    {"_overlap_ratio", "_link_endpoint", "_span_offsets", "make_relation_id",
     "_passes_label_constraint"},
)


class _FakeToken:
    """Stands in for a spaCy token so expansion can be tested without loading
    scispaCy. Only the four attributes expand_text_and_track_offsets() actually
    reads are provided."""

    def __init__(self, text, idx):
        self.text = text
        self.idx = idx
        self.is_stop = self.is_punct = self.is_space = False


# ---------------------------------------------------------------- sections

def test_sections_found_in_every_gold_note():
    """Section segmentation is a formatting heuristic, so it is tested against
    all 272 real notes rather than a synthetic example -- a regex that works on
    a handmade string and fails on real MIMIC layout would be worse than
    useless, because downstream code would silently receive 'unknown section'
    for everything."""
    if not os.path.exists(GOLD_NOTES):
        return  # gold set not present in this checkout; skip rather than fail
    notes = list(csv.DictReader(open(GOLD_NOTES, encoding="utf-8")))
    empties = [n["note_id"] for n in notes if not PRE["segment_sections"](n["text"])]
    assert not empties, f"{len(empties)} notes yielded no sections at all"

    with_family = sum(
        1 for n in notes
        if any(s["name_norm"] == "family history" for s in PRE["segment_sections"](n["text"]))
    )
    # Not all 272 -- a handful of notes genuinely omit the section. Asserting
    # equality here would encode a false assumption about the corpus.
    assert with_family >= 250, f"Family History found in only {with_family}/{len(notes)} notes"


def test_sections_are_ordered_and_non_overlapping():
    text = "Chief Complaint:\nchest pain\n\nPast Medical History:\nCOPD\n\nFamily History:\nmother had MI\n"
    secs = PRE["segment_sections"](text)
    assert [s["name"] for s in secs] == ["Chief Complaint", "Past Medical History", "Family History"]
    for a, b in zip(secs, secs[1:]):
        assert a["end"] <= b["header_start"]


def test_section_for_offset_returns_none_before_first_header():
    """None must mean 'unknown', never 'the first section' -- a preamble
    entity wrongly attributed to a section would inherit that section's
    assertion prior."""
    text = "preamble text\n\nFamily History:\nmother had MI\n"
    secs = PRE["segment_sections"](text)
    assert PRE["section_for_offset"](secs, 0) is None
    inside = text.index("mother")
    assert PRE["section_for_offset"](secs, inside)["name_norm"] == "family history"


# ------------------------------------------------------- abbreviation expansion

def test_ambiguous_abbreviation_is_flagged_and_keeps_all_candidates():
    """The regression this guards: the old dict comprehension silently kept one
    meaning per abbreviation, so 'MS' resolved to whichever row came last.

    2026-08-17 posture inversion: an ambiguous abbreviation not on
    VERIFIED_ALLOW_LIST is no longer silently guessed at all -- it's left
    unexpanded and routed downstream to fail safely (Tier 4/5/HITL) rather
    than risk a confident wrong answer. 'ms' is not allow-listed (the list
    starts empty), so this now asserts the FAIL-SAFE behavior, not the old
    always-guess-alphabetically one."""
    abbrevs = {"ms": ["mitral stenosis", "morphine sulfate", "multiple sclerosis"],
               "sob": ["shortness of breath"]}
    raw = "Pt has MS and SOB today"
    PRE["nlp"] = lambda _t: [_FakeToken("Pt", 0), _FakeToken("has", 3), _FakeToken("MS", 7),
                             _FakeToken("and", 10), _FakeToken("SOB", 14), _FakeToken("today", 18)]
    expanded, log = PRE["expand_text_and_track_offsets"](raw, abbrevs)

    assert log[0]["ambiguous"] is True
    assert len(log[0]["candidate_expansions"]) == 3
    assert log[0]["expansion"] == "MS", "not allow-listed -> left unexpanded, not guessed"
    assert log[0]["selection_basis"] == "unvetted_ambiguous_unexpanded"
    assert log[1]["ambiguous"] is False
    assert "candidate_expansions" not in log[1]
    assert expanded == "Pt has MS and shortness of breath today"


def test_allow_listed_ambiguous_abbreviation_still_expands():
    """The other side of the same gate: once an abbreviation is explicitly
    verified-safe (VERIFIED_ALLOW_LIST), the pre-2026-08-17 behavior --
    deterministic alphabetical-default expansion -- still applies. Proves
    the gate is a real gate (both directions), not an accidental full stop."""
    import src.abbreviation_flywheel as flywheel

    abbrevs = {"ms": ["mitral stenosis", "morphine sulfate", "multiple sclerosis"]}
    raw = "Pt has MS today"
    PRE["nlp"] = lambda _t: [_FakeToken("Pt", 0), _FakeToken("has", 3), _FakeToken("MS", 7),
                             _FakeToken("today", 10)]
    flywheel.VERIFIED_ALLOW_LIST.add("ms")
    try:
        expanded, log = PRE["expand_text_and_track_offsets"](raw, abbrevs)
    finally:
        flywheel.VERIFIED_ALLOW_LIST.discard("ms")

    assert log[0]["ambiguous"] is True
    assert log[0]["expansion"] == "mitral stenosis"
    assert expanded == "Pt has mitral stenosis today"


def test_expansion_offsets_are_correct_on_both_sides():
    abbrevs = {"ms": ["mitral stenosis"], "sob": ["shortness of breath"]}
    raw = "Pt has MS and SOB today"
    PRE["nlp"] = lambda _t: [_FakeToken("MS", 7), _FakeToken("SOB", 14)]
    expanded, log = PRE["expand_text_and_track_offsets"](raw, abbrevs)

    assert raw[log[0]["orig_start"]:log[0]["orig_end"]] == "MS"
    assert expanded[log[0]["exp_start"]:log[0]["exp_end"]] == "mitral stenosis"
    # The second expansion must account for the shift introduced by the first.
    assert raw[log[1]["orig_start"]:log[1]["orig_end"]] == "SOB"
    assert expanded[log[1]["exp_start"]:log[1]["exp_end"]] == "shortness of breath"


def test_time_machine_round_trip():
    """The core invariant: an offset found in expanded text must map back to
    the exact original characters the clinician wrote."""
    abbrevs = {"ms": ["mitral stenosis"], "sob": ["shortness of breath"]}
    raw = "Pt has MS and SOB today"
    PRE["nlp"] = lambda _t: [_FakeToken("MS", 7), _FakeToken("SOB", 14)]
    expanded, log = PRE["expand_text_and_track_offsets"](raw, abbrevs)

    i = expanded.index("shortness of breath")
    start, end = EXT["map_offsets_to_original"](i, i + len("shortness of breath"), log)
    assert raw[start:end] == "SOB"

    j = expanded.index("mitral stenosis")
    start, end = EXT["map_offsets_to_original"](j, j + len("mitral stenosis"), log)
    assert raw[start:end] == "MS"


# ---------------------------------------------------------------- entity_id

def test_entity_id_is_deterministic_and_span_sensitive():
    """Stage 3 decisions and Stage 4 provenance chains reference entity_id, so
    a re-run producing different IDs would orphan the entire audit trail."""
    a = EXT["make_entity_id"]("note1", 10, 20, "Condition")
    assert a == EXT["make_entity_id"]("note1", 10, 20, "Condition")
    assert a != EXT["make_entity_id"]("note1", 10, 21, "Condition")
    assert a != EXT["make_entity_id"]("note1", 10, 20, "Symptom")
    assert a != EXT["make_entity_id"]("note2", 10, 20, "Condition")
    assert a.startswith("note1-e")


# ------------------------------------------------------------ local context

def _sentences_for(text):
    spans, pos = [], 0
    parts = text.split(". ")
    for i, part in enumerate(parts):
        seg = part + (". " if i < len(parts) - 1 else "")
        spans.append({"sentence_id": i, "start": pos, "end": pos + len(seg)})
        pos += len(seg)
    return spans


def test_local_context_keeps_the_negation_cue():
    """The whole reason the window is sentence-bounded: a fixed character
    window can cut immediately before 'denies' and yield context that looks
    complete while having removed the word that inverts the meaning."""
    text = ("Admitted overnight. She denies any chest pain radiating to the left arm "
            "at this time. Discharged home.")
    pos = text.index("chest pain")
    ctx = EXT["build_local_context"](text, _sentences_for(text), pos, pos + 10)
    assert "chest pain" in ctx["text"]
    assert "denies" in ctx["text"]


def test_local_context_respects_cap_and_never_cuts_the_entity():
    long_text = "y" * 3000
    sentences = [{"sentence_id": 0, "start": 0, "end": 3000}]
    ctx = EXT["build_local_context"](long_text, sentences, 1500, 1510)
    assert len(ctx["text"]) <= EXT["LOCAL_CONTEXT_MAX_CHARS"]
    assert ctx["start"] <= 1500 and ctx["end"] >= 1510


def test_local_context_falls_back_without_sentence_data():
    ctx = EXT["build_local_context"]("abc" * 100, [], 150, 155)
    assert ctx["text"]
    assert "char_window" in ctx["basis"]
    assert ctx["sentence_id"] is None


# -------------------------------------------------- relation endpoint linking

def test_overlap_ratio_uses_shorter_span_denominator():
    """A canonical entity fully contained in a wider relex span refers to the
    same mention and must score 1.0 -- Jaccard would penalise exactly the
    nesting the two models are expected to disagree about."""
    ov = REL["_overlap_ratio"]
    assert ov(10, 20, 10, 20) == 1.0
    assert ov(5, 25, 10, 20) == 1.0
    assert ov(0, 5, 10, 20) == 0.0
    assert ov(10, 20, 15, 25) == 0.5


def test_endpoint_linking_resolves_or_reports_why_not():
    entities = [
        {"entity_id": "n1-eaaa", "exp_start": 10, "exp_end": 20},
        {"entity_id": "n1-ebbb", "exp_start": 50, "exp_end": 60},
    ]
    link = REL["_link_endpoint"]
    assert link(10, 20, entities)[0] == "n1-eaaa"
    assert link(8, 22, entities)[0] == "n1-eaaa", "nested relex span should still link"
    assert link(50, 60, entities)[0] == "n1-ebbb"
    # Every non-link must say WHY -- an unexplained None would be
    # indistinguishable from 'not attempted' in the provenance record.
    assert link(18, 40, entities)[2] == "unresolved_low_overlap"
    assert link(None, None, entities)[2] == "no_offsets"
    assert link(10, 20, [])[2] == "no_candidate_entities"


def test_span_offsets_reads_variants_and_refuses_to_guess():
    """Returning (0, 0) for a missing offset would link the endpoint to
    whatever entity sits at the start of the note."""
    assert REL["_span_offsets"]({"start": 3, "end": 9}) == (3, 9)
    assert REL["_span_offsets"]({"start_idx": 3, "end_idx": 9}) == (3, 9)
    assert REL["_span_offsets"]({"text": "no offsets here"}) == (None, None)
    assert REL["_span_offsets"](None) == (None, None)


def test_relation_id_is_deterministic():
    mk = REL["make_relation_id"]
    assert mk("n", 1, 2, "causes", 3, 4) == mk("n", 1, 2, "causes", 3, 4)
    assert mk("n", 1, 2, "causes", 3, 4) != mk("n", 1, 2, "indicates", 3, 4)


def test_relation_label_constraints():
    ok = REL["_passes_label_constraint"]
    assert ok("treated with", "Medication", "Condition") is True
    assert ok("treated with", "Anatomy", "Condition") is False
    assert ok("treated with", None, "Condition") is False
    # Unknown labels pass through, so extending RELATION_LABELS without a
    # matching constraint entry doesn't silently drop every prediction.
    assert ok("some_new_relation", "X", "Y") is True


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    print("\nALL PASS" if not failures else f"\n{failures} FAILURE(S)")
    sys.exit(1 if failures else 0)
