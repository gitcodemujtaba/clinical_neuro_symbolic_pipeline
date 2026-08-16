"""
tests/test_allergy_context_override.py — src/assertion.py's
apply_allergy_context_override() (2026-08-16, docs/2026-08-16_Shadow_Run_Precision_At_Scale.md).

No heavy-import stubbing needed here (unlike tests/test_tier12_ranking.py/
test_hybrid_retrieval.py) -- src/assertion.py imports only `re`/`warnings`,
no model or DB dependency.

Run: python3 -m pytest tests/test_allergy_context_override.py -v
     (or: python3 tests/test_allergy_context_override.py for plain output)
"""
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.assertion import (  # noqa: E402
    STATUS_ALLERGY,
    STATUS_PRESENT,
    apply_allergy_context_override,
)


def _assertion(status=STATUS_PRESENT, engine="context_default"):
    return {"assertion_status": status, "experiencer": "PATIENT",
            "temporality": "CURRENT", "assertion_cue": None,
            "assertion_engine": engine}


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # Medication in an Allergies section -> STATUS_ALLERGY.
    result = apply_allergy_context_override(_assertion(), "allergies", "Medication")
    check("Allergies section + Medication label -> STATUS_ALLERGY",
          result["assertion_status"] == STATUS_ALLERGY)
    check("assertion_engine records the override",
          "allergy_section_override" in result["assertion_engine"])

    # Condition in an Allergies section (e.g. "anaphylaxis") -> untouched.
    # Only Medication-labeled spans are drug-vs-allergy-finding ambiguous;
    # a Condition already correctly names the reaction itself.
    result = apply_allergy_context_override(_assertion(), "allergies", "Condition")
    check("Allergies section + Condition label -> untouched",
          result["assertion_status"] == STATUS_PRESENT)

    # Medication OUTSIDE an Allergies section -> untouched.
    result = apply_allergy_context_override(_assertion(), "discharge medications", "Medication")
    check("non-Allergies section + Medication label -> untouched",
          result["assertion_status"] == STATUS_PRESENT)

    # No section at all -> untouched (matches apply_section_priors()'s own
    # "if not section_name_norm: return assertion" early exit).
    result = apply_allergy_context_override(_assertion(), None, "Medication")
    check("no section -> untouched", result["assertion_status"] == STATUS_PRESENT)

    # Original dict is never mutated in place -- caller's copy stays intact,
    # matching apply_section_priors()'s own contract.
    original = _assertion()
    apply_allergy_context_override(original, "allergies", "Medication")
    check("original assertion dict is not mutated in place",
          original["assertion_status"] == STATUS_PRESENT)

    print(f"allergy-context-override tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_allergy_context_override():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
