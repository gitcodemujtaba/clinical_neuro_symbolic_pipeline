"""
tests/test_abbreviation_flywheel.py -- src/abbreviation_flywheel.py's two
data-driven abbreviation-disambiguation mechanisms (observed-frequency
priority, context-pattern rules) and the shared bias-exclusion list both
respect. No live DB or Ollama server needed: a small in-memory FakeConn
simulates just the SQL shapes this module issues.

Run: python3 -m pytest tests/test_abbreviation_flywheel.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.abbreviation_flywheel as flywheel  # noqa: E402
from src.abbreviation_flywheel import (  # noqa: E402
    _is_bias_excluded, compute_frequency_priority, context_window,
    mine_context_rules, record_ambiguous_expansion_outcome, select_by_context_pattern)


class FakeConn:
    """In-memory stand-in supporting exactly the SQL shapes
    abbreviation_flywheel.py issues -- not a general SQL engine."""

    def __init__(self):
        self.observations = {}   # (abbrev, context, expansion) -> hit_count
        self.context_rules = {}  # (abbrev, meaning, word, position) -> (support, score)
        self.hitl_rows = []      # list of (original_text, concept_id, orig_start, orig_end, note_id)

    def sql(self, _ddl):
        pass

    def execute(self, query, params=None):
        params = params or []
        q = " ".join(query.split())

        if q.startswith("INSERT INTO abbreviation_observed_expansions"):
            abbrev, ctx, expansion, domain, basis = params
            key = (abbrev, ctx, expansion)
            self.observations[key] = self.observations.get(key, 0) + 1
            return _Result([])

        if q.startswith("SELECT expansion, sum(hit_count) FROM abbreviation_observed_expansions"):
            abbrev = params[0]
            totals = {}
            for (a, _ctx, expansion), count in self.observations.items():
                if a == abbrev:
                    totals[expansion] = totals.get(expansion, 0) + count
            return _Result(list(totals.items()))

        if q.startswith("INSERT INTO abbreviation_context_rules"):
            abbrev, meaning, word, position, support, score = params
            self.context_rules[(abbrev, meaning, word, position)] = (support, score)
            return _Result([])

        if q.startswith("SELECT meaning, trigger_word, position, score FROM abbreviation_context_rules"):
            abbrev = params[0]
            meanings = set(params[1:])
            rows = [(m, w, p, s) for (a, m, w, p), (_sup, s) in self.context_rules.items()
                   if a == abbrev and m in meanings]
            return _Result(rows)

        if "FROM hitl_review_queue" in q:
            return _Result(self.hitl_rows)

        raise AssertionError(f"FakeConn: unexpected query: {q[:80]}")


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # Bias exclusion list
    # ======================================================================
    check("coronary-segment abbreviations are excluded", _is_bias_excluded("lad"))
    check("the additional confirmed-wrong list is excluded", _is_bias_excluded("RA"))
    check("short-alphanumeric-code shape is excluded even if not enumerated",
          _is_bias_excluded("T7"))
    check("an ordinary, non-flagged abbreviation is NOT excluded",
          not _is_bias_excluded("copd"))
    check("empty/None text is excluded (safe default)", _is_bias_excluded(""))

    # ======================================================================
    # Mechanism 1: observed-frequency priority, gated by VERIFIED_ALLOW_LIST
    # (2026-08-17 posture inversion -- see module docstring). Tests must not
    # leak allow-list membership across each other, so each one adds/removes
    # its own entry.
    # ======================================================================
    conn = FakeConn()
    check("no data -> None, not a forced pick",
          compute_frequency_priority(conn, "copd", ["chronic obstructive pulmonary disease",
                                                     "cyclophosphamide"]) is None)

    for _ in range(4):
        record_ambiguous_expansion_outcome(
            conn, "copd", "General", "chronic obstructive pulmonary disease",
            "Condition", "alphabetical_default")
    record_ambiguous_expansion_outcome(
        conn, "copd", "General", "cyclophosphamide", "Drug", "alphabetical_default")

    check("a clearly dominant meaning (4 vs 1) is NOT returned when the "
          "abbreviation is not on VERIFIED_ALLOW_LIST, no matter how "
          "dominant the ledger signal is",
          compute_frequency_priority(
              conn, "copd", ["chronic obstructive pulmonary disease", "cyclophosphamide"]
          ) is None)

    flywheel.VERIFIED_ALLOW_LIST.add("copd")
    try:
        winner = compute_frequency_priority(
            conn, "copd", ["chronic obstructive pulmonary disease", "cyclophosphamide"])
        check("the SAME dominant ledger data IS returned once explicitly "
              "allow-listed",
              winner == "chronic obstructive pulmonary disease")
    finally:
        flywheel.VERIFIED_ALLOW_LIST.discard("copd")

    conn2 = FakeConn()
    record_ambiguous_expansion_outcome(conn2, "ms", "General", "multiple sclerosis",
                                       "Condition", "alphabetical_default")
    record_ambiguous_expansion_outcome(conn2, "ms", "General", "morphine sulfate",
                                       "Drug", "alphabetical_default")
    flywheel.VERIFIED_ALLOW_LIST.add("ms")
    try:
        check("a genuine near-tie (1 vs 1) returns None -- not enough support "
              "anyway, even when allow-listed",
              compute_frequency_priority(conn2, "ms", ["multiple sclerosis", "morphine sulfate"])
              is None)
    finally:
        flywheel.VERIFIED_ALLOW_LIST.discard("ms")

    conn3 = FakeConn()
    for _ in range(3):
        record_ambiguous_expansion_outcome(conn3, "lad", "General",
                                           "left anterior descending artery",
                                           "Spec Anatomic Site", "alphabetical_default")
    flywheel.VERIFIED_ALLOW_LIST.add("lad")
    try:
        check("a bias-excluded abbreviation NEVER returns a frequency-priority "
              "pick even if mistakenly allow-listed -- the bias check is a "
              "second, redundant gate, not a replacement for the allow-list",
              compute_frequency_priority(conn3, "lad", ["left anterior descending artery",
                                                         "lymphadenopathy"]) is None)
    finally:
        flywheel.VERIFIED_ALLOW_LIST.discard("lad")

    check("VERIFIED_ALLOW_LIST starts empty in the real module (no "
          "speculative pre-seeding)",
          len(flywheel.VERIFIED_ALLOW_LIST) == 0)

    # ======================================================================
    # context_window()
    # ======================================================================
    text = "The patient has vertebral tenderness at S2 without radiculopathy today."
    idx = text.index("S2")
    pre, post = context_window(text, idx, idx + 2)
    check("pre-window picks up the distinguishing word 'vertebral'", "vertebral" in pre)
    check("post-window picks up 'without'", "without" in post)

    # ======================================================================
    # Mechanism 2: select_by_context_pattern()
    # ======================================================================
    conn4 = FakeConn()
    conn4.context_rules[("s2", "second sacral vertebra", "vertebral", "pre")] = (5, 2.1)
    conn4.context_rules[("s2", "second heart sound", "murmur", "post")] = (5, 1.8)
    meanings = ["second sacral vertebra", "second heart sound"]

    text_vert = "Exam notes vertebral S2 tenderness today."
    idx = text_vert.index("S2")
    pick = select_by_context_pattern(conn4, meanings, "s2", text_vert, idx, idx + 2)
    check("a matched 'vertebral' pre-trigger picks the sacral-vertebra meaning",
          pick == "second sacral vertebra")

    text_murmur = "S2 murmur was noted on exam."
    idx = text_murmur.index("S2")
    pick = select_by_context_pattern(conn4, meanings, "s2", text_murmur, idx, idx + 2)
    check("a matched 'murmur' post-trigger picks the heart-sound meaning",
          pick == "second heart sound")

    text_neither = "S2 was documented in the chart without further detail."
    idx = text_neither.index("S2")
    pick = select_by_context_pattern(conn4, meanings, "s2", text_neither, idx, idx + 2)
    check("no matching trigger word -> None, falls through to the existing tiebreak",
          pick is None)

    check("no rules at all for this abbreviation -> None",
          select_by_context_pattern(FakeConn(), meanings, "s2", text_vert, idx, idx + 2) is None)

    # ======================================================================
    # mine_context_rules()
    # ======================================================================
    conn5 = FakeConn()
    raw_texts = {}
    hitl_rows = []
    # 3 confirmed "second sacral vertebra" examples, all near "vertebral"/"spine"
    sacral_sentences = [
        "There is vertebral tenderness noted at S2 today.",
        "Exam shows spine tenderness over S2 without other findings.",
        "Point tenderness at the vertebral level S2 was documented.",
    ]
    for i, sent in enumerate(sacral_sentences):
        note_id = f"note-sacral-{i}"
        raw_texts[note_id] = sent
        idx = sent.index("S2")
        hitl_rows.append(("S2", "10001", idx, idx + 2, note_id))
    # 3 confirmed "second heart sound" examples, all near "murmur"/"auscultation"
    heart_sentences = [
        "On auscultation S2 was normal with no murmur.",
        "S2 was split, a soft murmur was also appreciated.",
        "Cardiac exam: S1 S2 normal, no murmur heard.",
    ]
    for i, sent in enumerate(heart_sentences):
        note_id = f"note-heart-{i}"
        raw_texts[note_id] = sent
        idx = sent.index("S2")
        hitl_rows.append(("S2", "20002", idx, idx + 2, note_id))
    conn5.hitl_rows = hitl_rows

    n_written = mine_context_rules(conn5, raw_texts, min_support=5)
    check("mining real (synthetic) confirmed data writes at least one rule",
          n_written > 0)
    check("a rule was written FOR the sacral meaning with a distinguishing pre-word",
          any(m == "10001" and p == "pre" for (_a, m, _w, p) in conn5.context_rules))
    check("a rule was written FOR the heart-sound meaning with a distinguishing word",
          any(m == "20002" for (_a, m, _w, _p) in conn5.context_rules))

    conn6 = FakeConn()
    conn6.hitl_rows = hitl_rows[:2]  # too few examples
    check("too little data (below min_support) writes nothing",
          mine_context_rules(conn6, raw_texts, min_support=5) == 0)

    conn7 = FakeConn()
    # Same shape, but for a KNOWN-BIAS abbreviation (LAD) -- unlike
    # compute_frequency_priority(), mine_context_rules() must NOT exclude
    # it: this mechanism's whole purpose is eventually correcting exactly
    # these entities from real reviewer-confirmed data, which is
    # independent ground truth, not the pipeline's own guess repeating
    # itself. See the module docstring's "ONLY MECHANISM 1 EXCLUDES..."
    # paragraph.
    lad_rows = []
    for i, sent in enumerate(sacral_sentences + heart_sentences):
        note_id = f"note-lad-{i}"
        raw_texts[note_id] = sent
        idx = sent.index("S2")
        concept = "10001" if i < 3 else "20002"
        lad_rows.append(("LAD", concept, idx, idx + 2, note_id))
    conn7.hitl_rows = lad_rows
    check("a KNOWN-BIAS abbreviation (LAD) IS mined into rules from real "
          "reviewer-confirmed data -- mechanism 2 deliberately does not "
          "exclude it, unlike mechanism 1",
          mine_context_rules(conn7, raw_texts, min_support=5) > 0)

    check("conn=None returns 0, never raises", mine_context_rules(None, {}) == 0)
    check("no hitl rows at all returns 0", mine_context_rules(FakeConn(), {}) == 0)

    print(f"abbreviation-flywheel tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_abbreviation_flywheel():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
