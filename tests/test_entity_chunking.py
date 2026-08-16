"""
tests/test_entity_chunking.py -- src/entity_extraction.py's _build_chunks(),
the plan's Phase 5 (Pass 2 sliding-window chunking) core logic.

Pure-logic tests only, no GLiNER model / DB needed: _build_chunks() takes
pre-tokenized sentences/words as plain data. AST-extraction technique (same
as tests/test_tier_gate.py etc.) specifically because src/entity_extraction.py
loads the actual GLiNER-BioMed model at IMPORT time (`model =
GLiNER.from_pretrained(...)` at module level) -- a real, multi-second cost a
pure-logic test of chunk-boundary math has no reason to pay.

Run: python3 -m pytest tests/test_entity_chunking.py -v
"""
import ast
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _load_pure_functions(module_filename: str, wanted: set, extra_globals: dict = None) -> dict:
    """See tests/test_offset_mapping.py's identical helper."""
    path = os.path.join(SRC_DIR, module_filename)
    tree = ast.parse(open(path, encoding="utf-8").read())

    def _is_literal_assign(node):
        if not isinstance(node, ast.Assign):
            return False
        try:
            ast.literal_eval(node.value)
            return True
        except (ValueError, SyntaxError, TypeError):
            return False

    body = [
        n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted) or _is_literal_assign(n)
    ]
    ns = dict(extra_globals or {})
    exec(compile(ast.Module(body=body, type_ignores=[]), f"<{module_filename}>", "exec"), ns)
    return ns


EE = _load_pure_functions("entity_extraction.py", {"_build_chunks"})
_build_chunks = EE["_build_chunks"]


def _make_note(sentence_word_counts, words_per_char=1):
    """Builds a synthetic (sentences, words) pair: one sentence per entry in
    sentence_word_counts, each word given a distinct 1-char span so word
    counting by (start,end) containment is unambiguous and easy to reason
    about by hand. Returns (sentences, words, total_chars)."""
    sentences = []
    words = []
    pos = 0
    for i, n_words in enumerate(sentence_word_counts):
        sent_start = pos
        for _ in range(n_words):
            words.append((f"w{pos}", pos, pos + 1))
            pos += 1
        sentences.append({"sentence_id": i, "start": sent_start, "end": pos})
        pos += 1  # a 1-char gap (e.g. a space) between sentences, not counted as a word
    return sentences, words, pos


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # Small note, well under budget -- single chunk covering everything.
    # ======================================================================
    sentences, words, total_chars = _make_note([10, 15, 20])
    chunks = _build_chunks(sentences, words, budget=1800, overlap=128)
    check("under-budget note produces exactly one chunk",
          len(chunks) == 1)
    check("the single chunk covers from the first sentence's start...",
          chunks[0][0] == sentences[0]["start"])
    check("...to the last sentence's end",
          chunks[0][1] == sentences[-1]["end"])

    # ======================================================================
    # Over-budget note -- must split into multiple chunks, each snapped to
    # sentence boundaries, jointly covering the whole note with no gaps.
    # ======================================================================
    # 20 sentences of 150 words each = 3000 words total, budget 1000 -> must chunk.
    sentence_word_counts = [150] * 20
    sentences, words, total_chars = _make_note(sentence_word_counts)
    chunks = _build_chunks(sentences, words, budget=1000, overlap=100)
    check("over-budget note produces more than one chunk", len(chunks) > 1)

    check("every chunk boundary is a real sentence start", all(
        any(s["start"] == c[0] for s in sentences) for c in chunks))
    check("every chunk boundary is a real sentence end", all(
        any(s["end"] == c[1] for s in sentences) for c in chunks))

    check("chunks jointly cover from the note's first sentence start",
          chunks[0][0] == sentences[0]["start"])
    check("chunks jointly cover to the note's last sentence end",
          chunks[-1][1] == sentences[-1]["end"])

    def words_in(start, end):
        return sum(1 for _w, ws, we in words if ws >= start and we <= end)

    check("no chunk exceeds the budget (every sentence here is well under "
          "budget alone, so this must hold exactly)",
          all(words_in(c[0], c[1]) <= 1000 for c in chunks))

    check("consecutive chunks overlap (chunk N+1 starts before chunk N ends)",
          all(chunks[i + 1][0] < chunks[i][1] for i in range(len(chunks) - 1)))

    # Every point in the note is covered by at least one chunk -- no gap
    # a real entity could fall entirely outside every window.
    covered = [False] * total_chars
    for c_start, c_end in chunks:
        for p in range(c_start, min(c_end, total_chars)):
            covered[p] = True
    # only check inside actual sentence spans -- inter-sentence gap chars
    # (the 1-char separators _make_note() inserts) are allowed to be uncovered
    sentence_chars_covered = all(
        covered[p] for s in sentences for p in range(s["start"], s["end"]))
    check("every character inside every sentence is covered by some chunk",
          sentence_chars_covered)

    # ======================================================================
    # Edge cases
    # ======================================================================
    check("empty sentence list falls back to a single (0, last_word_end) chunk",
          _build_chunks([], [("w", 0, 5)], budget=1800, overlap=128) == [(0, 5)])
    check("empty sentence list AND empty words -> (0, 0), no crash",
          _build_chunks([], [], budget=1800, overlap=128) == [(0, 0)])

    # A single sentence alone exceeding the budget cannot be split further
    # (sentence-boundary-snapping takes priority over the budget) -- it
    # still forms its own chunk rather than crashing or looping forever.
    sentences, words, _ = _make_note([5000])
    chunks = _build_chunks(sentences, words, budget=1800, overlap=128)
    check("a single oversized sentence still produces exactly one chunk "
          "covering it whole, not a crash/infinite loop",
          len(chunks) == 1 and chunks[0] == (sentences[0]["start"], sentences[0]["end"]))

    # Many tiny sentences -- must still make forward progress every
    # iteration (regression guard against an infinite loop in the overlap
    # back-up logic when overlap >= a whole chunk's sentence count).
    sentences, words, _ = _make_note([1] * 500)
    chunks = _build_chunks(sentences, words, budget=50, overlap=128)
    check("many tiny sentences with overlap >= budget still terminates "
          "and covers the whole note",
          chunks[-1][1] == sentences[-1]["end"] and len(chunks) > 1)

    print(f"entity-chunking tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_entity_chunking():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
