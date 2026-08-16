"""
tests/test_acronym_escalation.py -- src/acronym_escalation.py's
resolve_ambiguous_acronyms(), build-order step 1 (mock-backed). Pure-logic
tests; the module has no DB/model imports at module level so a plain import
is fast (unlike src/normalization/orchestrator.py, which loads SapBERT).

Run: python3 -m pytest tests/test_acronym_escalation.py -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.acronym_escalation import resolve_ambiguous_acronyms, MOCK_RESOLUTIONS


def _entity(entity_id, expansion_ambiguous=True, **overrides):
    base = {"entity_id": entity_id, "expansion_ambiguous": expansion_ambiguous}
    base.update(overrides)
    return base


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    known_id = next(iter(MOCK_RESOLUTIONS))

    # The one entity_id present in MOCK_RESOLUTIONS, flagged ambiguous.
    resolved = resolve_ambiguous_acronyms([_entity(known_id)], "", "note1", conn=None)
    check("known ambiguous entity_id resolves via the mock",
          known_id in resolved)
    check("resolution carries the expected fields",
          resolved[known_id]["expansion"] == MOCK_RESOLUTIONS[known_id]["expansion"]
          and resolved[known_id]["source"] == "mock")

    # Ambiguous but NOT in MOCK_RESOLUTIONS -- absent from the result, not a KeyError.
    resolved = resolve_ambiguous_acronyms(
        [_entity("some-other-entity-id")], "", "note1", conn=None)
    check("ambiguous entity not in the mock table is simply absent from the result",
          resolved == {})

    # Not ambiguous at all -- skipped even if (hypothetically) present in the mock.
    resolved = resolve_ambiguous_acronyms(
        [_entity(known_id, expansion_ambiguous=False)], "", "note1", conn=None)
    check("non-ambiguous entity is never resolved, even if its id is in the mock table",
          resolved == {})

    # expansion_ambiguous missing entirely (ordinary entity dict shape) -- same as False.
    resolved = resolve_ambiguous_acronyms(
        [{"entity_id": known_id}], "", "note1", conn=None)
    check("entity dict with no expansion_ambiguous key at all -> not resolved",
          resolved == {})

    # Empty entity list -- no crash.
    check("empty entity list -> empty result, no crash",
          resolve_ambiguous_acronyms([], "", "note1", conn=None) == {})

    # Multiple entities: only the ambiguous+known one resolves.
    resolved = resolve_ambiguous_acronyms(
        [_entity(known_id), _entity("unrelated-id"), _entity("x", expansion_ambiguous=False)],
        "", "note1", conn=None)
    check("mixed batch resolves only the ambiguous, known entity",
          set(resolved.keys()) == {known_id})

    print(f"acronym-escalation tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_acronym_escalation():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
