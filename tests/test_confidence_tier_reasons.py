"""
tests/test_confidence_tier_reasons.py — asserts the 2026-08-13 P1.2 change to
src/normalization.py's compute_confidence_tier() does what it claims:

    ROUTING IS UNCHANGED. REASONS ARE NO LONGER CENSORED.

That claim is the whole justification for making the change under time
pressure, so it is worth a test rather than an assertion in a comment. The
critical case is a Tier 1/2 entity that trips high_gliner_risk and the
short-token trio: before, its tier_reasons said only ["weak_match_tier"];
after, it lists every signal that fired -- and the tier is "LOW" either way.

Run:  python3 tests/test_confidence_tier_reasons.py
"""

import os
import sys
import types

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

# Same import stubs as tests/test_tier12_ranking.py -- importing
# src/normalization.py otherwise loads SapBERT.
sys.path.insert(0, os.path.join(PROJECT_DIR, "tests"))
from test_tier12_ranking import _install_stubs  # noqa: E402

_stubbed_modules = _install_stubs()

import src.normalization as N  # noqa: E402

# 2026-08-17 fix: this call used to leave a fake `duckdb` (and possibly
# torch/transformers) permanently in sys.modules with no cleanup -- the same
# bug already fixed in test_tier12_ranking.py's and test_hybrid_retrieval.py's
# own internal calls, just reached here via the imported helper instead of a
# local copy. Only remove what THIS call actually stubbed.
for _name in _stubbed_modules:
    sys.modules.pop(_name, None)


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ------------------------------------------------------------------
    # 1. THE CASE THE FIX EXISTS FOR. A Tier 1 entity that is a short
    #    all-caps abbreviation extracted at high GLiNER confidence. Every
    #    signal is true of it; before P1.2 only weak_match_tier was recorded.
    # ------------------------------------------------------------------
    tier, reasons = N.compute_confidence_tier(
        gliner_confidence=0.95, normalization_ambiguous=False,
        expansion_ambiguous=False, short_token_text="WBC",
        match_tier="1 (Exact)")
    check("tier1 still routes LOW", tier == "LOW")
    check("tier1 records high_gliner_risk", "high_gliner_risk" in reasons)
    check("tier1 records short_token", "short_token" in reasons)
    check("tier1 records isupper_abbreviation", "isupper_abbreviation" in reasons)
    check("tier1 records weak_match_tier", "weak_match_tier" in reasons)
    check("tier1 is no longer a single-reason row", len(reasons) >= 4)

    # ------------------------------------------------------------------
    # 2. Tier 2 behaves identically to Tier 1 -- the exemption covered both.
    # ------------------------------------------------------------------
    _t2, r2 = N.compute_confidence_tier(
        gliner_confidence=0.95, normalization_ambiguous=False,
        expansion_ambiguous=False, short_token_text="RBC",
        match_tier="2 (Synonym)")
    check("tier2 matches tier1 reasons", set(r2) == set(reasons))

    # ------------------------------------------------------------------
    # 3. ROUTING IS UNCHANGED for every known tier. This is the claim that
    #    made the change safe to ship without a re-run.
    # ------------------------------------------------------------------
    for mt in ("1 (Exact)", "2 (Synonym)", "3 (Semantic)", "0 (Failed)"):
        t, r = N.compute_confidence_tier(0.1, False, False, match_tier=mt)
        check(f"{mt} routes LOW", t == "LOW")
        check(f"{mt} fires weak_match_tier", "weak_match_tier" in r)

    # ------------------------------------------------------------------
    # 4. match_tier=None is STILL the conservative "we don't know" path:
    #    weak_match_tier must not fire, and a clean entity must still be
    #    able to reach HIGH. This is the one route to skipping Stage 3 and
    #    P1.2 must not have widened it.
    # ------------------------------------------------------------------
    t, r = N.compute_confidence_tier(0.1, False, False, match_tier=None)
    check("None tier does not fire weak_match_tier", "weak_match_tier" not in r)
    check("clean entity with unknown tier can still reach HIGH", t == "HIGH")
    check("clean entity has no reasons", r == [])

    # ------------------------------------------------------------------
    # 5. Signals still fire for match_tier=None exactly as before -- the
    #    gate being removed was `not in (tier1, tier2)`, which was already
    #    True for None, so None's behaviour must be bit-identical.
    # ------------------------------------------------------------------
    t, r = N.compute_confidence_tier(0.95, False, False, short_token_text="K",
                                     match_tier=None)
    check("None tier still fires high_gliner_risk", "high_gliner_risk" in r)
    check("None tier still fires short_token", "short_token" in r)
    check("None tier routes LOW when signals fire", t == "LOW")

    # ------------------------------------------------------------------
    # 6. The independent signals are untouched by any of this.
    # ------------------------------------------------------------------
    _t, r = N.compute_confidence_tier(
        0.1, True, True, domain_conflict=True,
        crosses_sentence_boundary=True, match_tier=None)
    for expected in ("normalization_ambiguous", "expansion_ambiguous",
                     "label_domain_conflict",
                     "entity_span_crosses_sentence_boundary"):
        check(f"{expected} still fires", expected in r)

    # ------------------------------------------------------------------
    # 7. alnum_mix on a Tier 1 entity -- the third sub-signal, previously
    #    censored, on the delimiter-less lab pattern it was measured for.
    # ------------------------------------------------------------------
    _t, r = N.compute_confidence_tier(0.1, False, False,
                                      short_token_text="HCO3-22",
                                      match_tier="1 (Exact)")
    check("alnum_mix fires on tier 1", "alnum_mix" in r)

    # ------------------------------------------------------------------
    # 8. Empty/whitespace short_token_text must not fire anything.
    # ------------------------------------------------------------------
    _t, r = N.compute_confidence_tier(0.1, False, False, short_token_text="   ",
                                      match_tier=None)
    check("blank short_token_text fires nothing", r == [])

    print(f"confidence-tier reason tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
