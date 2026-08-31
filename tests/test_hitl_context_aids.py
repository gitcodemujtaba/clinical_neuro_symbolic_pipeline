"""
tests/test_hitl_context_aids.py -- ui/components/hitl_context_aids.py's
pure functions: agreement_summary(), known_risk_flags(). No DB, no
Streamlit.

Run: python3 -m pytest tests/test_hitl_context_aids.py -v
"""
import sys

from ui.components.hitl_context_aids import agreement_summary, known_risk_flags


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ======================================================================
    # agreement_summary
    # ======================================================================
    unanimous_models = [
        {"verdict": "SUPPORTED_1", "degenerate_generation": False},
        {"verdict": "SUPPORTED_1", "degenerate_generation": False},
        {"verdict": "SUPPORTED_1", "degenerate_generation": False},
    ]
    r = agreement_summary(unanimous_models)
    check("3/3 unanimous is flagged unanimous=True", r["unanimous"] is True)
    check("unanimous label uses a checkmark", r["label"].startswith("✅"))
    check("unanimous total_usable is 3", r["total_usable"] == 3)

    split_models = [
        {"verdict": "SUPPORTED_1", "degenerate_generation": False},
        {"verdict": "SUPPORTED_1", "degenerate_generation": False},
        {"verdict": "NONE_CORRECT", "degenerate_generation": False},
    ]
    r2 = agreement_summary(split_models)
    check("2-1 split is NOT unanimous", r2["unanimous"] is False)
    check("split label uses a warning marker", r2["label"].startswith("⚠️"))
    check("split top_verdict is the majority verdict", r2["top_verdict"] == "SUPPORTED_1")

    all_error_models = [
        {"verdict": "ERROR", "degenerate_generation": False},
        {"verdict": "ERROR", "degenerate_generation": False},
    ]
    r3 = agreement_summary(all_error_models)
    check("all-error models produce 0 usable votes", r3["total_usable"] == 0)
    check("0-usable-vote case is flagged, not unanimous", r3["unanimous"] is False)

    check("empty model list doesn't crash", agreement_summary([])["total_usable"] == 0)
    check("None model list doesn't crash", agreement_summary(None)["total_usable"] == 0)

    # ======================================================================
    # known_risk_flags
    # ======================================================================
    check("a short alphanumeric mention (S2) is flagged",
          any("Short alphanumeric" in f for f in known_risk_flags("S2", [], None)))
    check("a plain, unambiguous mention (pneumonia) is NOT flagged",
          known_risk_flags("pneumonia", [], None) == [])

    check("a coronary-segment abbreviation (LAD) is flagged",
          any("Coronary-artery-segment" in f for f in known_risk_flags("LAD", [], None)))

    domain_conflict_candidates = [
        {"omop_concept_id": 1, "concept_name": "White blood cell count", "domain_id": "Procedure"},
        {"omop_concept_id": 2, "concept_name": "Leucocyte count", "domain_id": "Observation"},
    ]
    check("differing top-2 domains raises a domain-conflict flag",
          any("Domain conflict" in f for f in known_risk_flags(
              "WBC", domain_conflict_candidates, None)))

    same_domain_candidates = [
        {"omop_concept_id": 1, "concept_name": "Pneumonia", "domain_id": "Condition"},
        {"omop_concept_id": 2, "concept_name": "Bacterial pneumonia", "domain_id": "Condition"},
    ]
    check("matching top-2 domains raises no domain-conflict flag",
          not any("Domain conflict" in f for f in known_risk_flags(
              "pneumonia", same_domain_candidates, None)))

    check("empty candidates list doesn't crash", known_risk_flags("text", [], None) == [] or True)
    check("None original_text doesn't crash", known_risk_flags(None, [], None) is not None)

    print(f"hitl-context-aids tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_hitl_context_aids():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
