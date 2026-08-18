"""
tests/test_exam_header_context.py — src/normalization/orchestrator.py's
_is_physical_exam_header_context(), the 2026-08-18 fix for the "Abdomen"
wrong-concept finding: gold codes physical-exam sub-header mentions
("Abdomen: soft, non-tender") to the EXAMINATION of the body region (a
SNOMED Procedure) rather than the body region itself (Anatomy/Body
Structure) -- confirmed at real scale (1,374 gold annotations corpus-wide).
Scoped to ':' only after checking both '-' and '/' introduced real false
positives on real data (spine-level ranges, echo-report measurement labels,
narrative compound anatomical references) -- see the function's own
docstring/comment block for the measured numbers.

Pure-logic tests only, no DB/model calls -- same AST-extraction technique as
tests/test_allergy_domain_tiebreak.py, since src/normalization/orchestrator.py
imports `from .constants import *` which loads the actual SapBERT model at
import time (~5.7s).

Run: python3 -m pytest tests/test_exam_header_context.py -v
"""

import ast
import os
import re
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src",
                       "normalization")


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


ORC = _load_pure_functions(
    "orchestrator.py", {"_is_physical_exam_header_context"},
    # _EXAM_HEADER_TRAILING_RE is a re.compile() call, not a literal --
    # _is_literal_assign can't extract it, so it's built here identically to
    # the real module-level pattern instead (same pattern used throughout
    # this session's other AST-extracted test files).
    extra_globals={"_EXAM_HEADER_TRAILING_RE": re.compile(r"^\s*:")},
)
_is_physical_exam_header_context = ORC["_is_physical_exam_header_context"]


def _ent(text, context):
    return {"original_text": text, "local_context": context}


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # -- real, confirmed positive cases (from actual note text this session) --
    check("'Abdomen:' header -> True",
          _is_physical_exam_header_context(
              _ent("Abdomen", "\nAbdomen: distended, minimally invasive..."), "Anatomy"))
    check("'ABDOMEN:' (all caps) header -> True",
          _is_physical_exam_header_context(
              _ent("ABDOMEN", "\nABDOMEN: Soft, non-tender, non-distended."), "Anatomy"))
    check("whitespace before the colon is tolerated",
          _is_physical_exam_header_context(
              _ent("Neck", "Neck  : supple"), "Anatomy"))

    # -- real, confirmed negative cases: narrative anatomical references,
    # NOT headers -- must stay False so they keep resolving as plain Anatomy --
    check("'gunshot wound to abdomen' (narrative, no colon) -> False",
          not _is_physical_exam_header_context(
              _ent("abdomen", "\ngunshot wound to abdomen\n \nMajor Surgical..."), "Anatomy"))
    check("'Worsening abdomen distension and pain' (narrative) -> False",
          not _is_physical_exam_header_context(
              _ent("abdomen", "\nWorsening abdomen distension and pain \n"), "Anatomy"))

    # -- scoped to Anatomy label only --
    check("non-Anatomy label never fires, even with a colon right after",
          not _is_physical_exam_header_context(
              _ent("Pre-eclampsia", "\nPre-eclampsia: severe features"), "Condition"))

    # -- dropped triggers (dash/slash) must NOT fire, per the measured
    # false-positive cases that got them dropped --
    check("dash trigger ('LUNGS - CTA bilat') does NOT fire (dropped, real FP risk)",
          not _is_physical_exam_header_context(
              _ent("LUNGS", "\nLUNGS - CTA bilat, no wheezes"), "Anatomy"))
    check("slash trigger ('liver/spleen edge') does NOT fire (dropped, real FP risk)",
          not _is_physical_exam_header_context(
              _ent("liver", "cannot percuss liver/spleen edge"), "Anatomy"))

    # -- missing/empty inputs degrade safely --
    check("empty local_context -> False, no crash",
          not _is_physical_exam_header_context(_ent("Abdomen", ""), "Anatomy"))
    check("entity text not found in local_context -> False, no crash",
          not _is_physical_exam_header_context(_ent("Abdomen", "unrelated text"), "Anatomy"))

    print(f"exam-header-context tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_exam_header_context():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
