"""
tests/test_allergy_domain_tiebreak.py — src/normalization/orchestrator.py's
_apply_allergy_domain_tiebreak(), the 2026-08-16 fix for the NSAIDS/
Penicillins finding (docs/2026-08-16_Shadow_Run_Precision_At_Scale.md):
widening is_allergy_context's search_domain_override to ['Condition',
'Observation'] lets Tier 3 see SNOMED's own "Allergy to X" (Observation)
concepts, but near-tied scores still need this small-margin preference to
pick correctly.

Pure-logic tests only, no DB/model calls -- same AST-extraction technique
as tests/test_tier_gate.py, since src/normalization/orchestrator.py imports
`from .constants import *` which loads the actual SapBERT model at import
time (~5.7s).

Run: python3 -m pytest tests/test_allergy_domain_tiebreak.py -v
"""

import ast
import os
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


ORC = _load_pure_functions("orchestrator.py", {"_apply_allergy_domain_tiebreak"})
_apply_allergy_domain_tiebreak = ORC["_apply_allergy_domain_tiebreak"]


def _cand(name, domain, score, basis="semantic_similarity", concept_id=1):
    return {"omop_concept_id": concept_id, "concept_name": name, "domain_id": domain,
            "vocabulary_id": "SNOMED", "similarity_score": score, "match_basis": basis}


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # Real NSAIDS case: Condition on top, Observation within margin (0.021 gap).
    mapping = {"candidates": [
        _cand("Allergic reaction caused by nonsteroidal antiinflammatory agent",
              "Condition", 0.8078, concept_id=36687109),
        _cand("Allergy to non-steroidal anti-inflammatory agent",
              "Observation", 0.7867, concept_id=4163066),
    ]}
    r = _apply_allergy_domain_tiebreak(mapping)
    check("NSAIDS-shaped near-tie promotes the Observation candidate",
          r["candidates"][0]["omop_concept_id"] == 4163066)
    check("promoted candidate's match_basis records the tiebreak",
          "allergy_domain_tiebreak" in r["candidates"][0]["match_basis"])
    check("mapping-level fields follow the promoted candidate",
          r["concept_id"] == 4163066 and r["domain_id"] == "Observation")
    check("original mapping dict is not mutated in place", "candidates" not in mapping or
          mapping["candidates"][0]["omop_concept_id"] == 36687109)

    # Already-Observation top candidate -- no-op (covers the 6 exact-override
    # cases, e.g. aspirin/morphine, where candidates[0] is already correct).
    already_right = {"candidates": [
        _cand("Allergy to aspirin", "Observation", 1.0, concept_id=99),
        _cand("Aspirin", "Drug", 0.5, concept_id=100),
    ]}
    r = _apply_allergy_domain_tiebreak(already_right)
    check("top candidate already Observation-domain -> no-op",
          r["candidates"][0]["omop_concept_id"] == 99)

    # Gap too wide -- must NOT override a confidently-wrong-domain top hit.
    wide_gap = {"candidates": [
        _cand("Allergic reaction caused by X", "Condition", 0.90, concept_id=1),
        _cand("Allergy to X (weak match)", "Observation", 0.50, concept_id=2),
    ]}
    r = _apply_allergy_domain_tiebreak(wide_gap)
    check("gap wider than the margin -> no promotion",
          r["candidates"][0]["omop_concept_id"] == 1)

    # No Observation candidate present at all.
    no_observation = {"candidates": [
        _cand("Allergic reaction caused by X", "Condition", 0.90, concept_id=1),
        _cand("Some other Condition concept", "Condition", 0.89, concept_id=2),
    ]}
    r = _apply_allergy_domain_tiebreak(no_observation)
    check("no Observation-domain candidate present -> no-op",
          r["candidates"][0]["omop_concept_id"] == 1)

    # Fewer than 2 candidates -- nothing to tiebreak.
    check("single-candidate mapping -> no-op",
          _apply_allergy_domain_tiebreak(
              {"candidates": [_cand("X", "Condition", 0.9)]})["candidates"][0]["concept_name"]
          == "X")
    check("empty-candidate mapping -> no-op, no crash",
          _apply_allergy_domain_tiebreak({"candidates": []}) == {"candidates": []})

    # Real Penicillins case, in the REAL rank order Tier 3 would actually
    # produce once part 1 (widening search_domain_override) lands: Observation
    # (0.9880) already outscores Condition (0.9165), so it's already
    # candidates[0] before this function ever runs -- a no-op, confirming
    # widening the domain filter alone was sufficient here (no tiebreak
    # needed, unlike NSAIDS above).
    penicillins = {"candidates": [
        _cand("Allergy to penicillin", "Observation", 0.9880, concept_id=4240903),
        _cand("Allergic reaction caused by penicillin", "Condition", 0.9165, concept_id=36687105),
    ]}
    r = _apply_allergy_domain_tiebreak(penicillins)
    check("Penicillins-shaped case, real rank order -> already correct, no-op",
          r["candidates"][0]["omop_concept_id"] == 4240903)

    print(f"allergy-domain-tiebreak tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_allergy_domain_tiebreak():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
