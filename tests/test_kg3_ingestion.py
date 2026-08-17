"""
tests/test_kg3_ingestion.py -- src/kg3_ingestion.py's ingest_auto_decision()
tier-membership check.

Regression test for a real bug caught live on the 2026-08-17 fresh-note
calibrator validation run: ingest_auto_decision() used to check tier
membership against its OWN hardcoded 3-tuple ("TIER_1_AUTO_VALIDATED",
"TIER_2_AUTO_RESOLVED", "TIER_3_AUTO_VALIDATED") -- a second, independent
copy of route_tier()'s AUTO_TIERS set that silently missed
TIER_1B_CALIBRATED_AUTO_VALIDATED when Phase 6 added it. Every calibrator-
promoted decision was rejected as UningestibleCase even though it is a
genuine AUTO tier (6/6 TIER_1B decisions blocked on the first real run).
Fixed by importing AUTO_TIERS from src.mollm_tier_gate directly, so the two
can never drift apart again -- this test guards against that regressing.

dry_run=True never touches the Memgraph driver (returns before
driver.session() is ever called -- see the function's own code), so these
tests pass driver=None and need no live Memgraph connection.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kg3_ingestion import ingest_auto_decision, UningestibleCase  # noqa: E402
from src.mollm_tier_gate import AUTO_TIERS  # noqa: E402


def _decision(tier, final_candidate_index=1):
    return {
        "tier": tier, "final_candidate_index": final_candidate_index,
        "mollm_routing_decision": "AUTO_VALIDATED", "mollm_call_id": "test-call-1",
        "entity_id": "test-entity-1", "note_id": "test-note-1",
        "composite_confidence": 0.9, "queue_reason": None,
    }


def _entity_fields():
    return {
        "candidates": [{"omop_concept_id": 12345, "concept_name": "Test concept",
                       "vocabulary_id": "SNOMED", "domain_id": "Condition"}],
        "original_text": "test", "entity_label": "Condition",
        "orig_start": 0, "orig_end": 4, "confidence": 0.9,
    }


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    check("TIER_1B_CALIBRATED_AUTO_VALIDATED is in route_tier()'s AUTO_TIERS "
          "(sanity check the fix targets a real membership, not a typo)",
          "TIER_1B_CALIBRATED_AUTO_VALIDATED" in AUTO_TIERS)

    # The actual regression: this used to raise UningestibleCase.
    result = ingest_auto_decision(
        None, _decision("TIER_1B_CALIBRATED_AUTO_VALIDATED"), _entity_fields(), dry_run=True)
    check("a TIER_1B_CALIBRATED_AUTO_VALIDATED decision now passes the "
          "tier-membership check and reaches a dry-run result",
          result["dry_run"] is True and result["params"]["omop_concept_id"] == 12345)

    for tier in AUTO_TIERS:
        r = ingest_auto_decision(None, _decision(tier), _entity_fields(), dry_run=True)
        check(f"every AUTO_TIERS member ({tier}) passes the tier check",
              r["dry_run"] is True)

    try:
        ingest_auto_decision(
            None, _decision("TIER_4_ENSEMBLE_SPLIT"), _entity_fields(), dry_run=True)
        check("a genuine non-AUTO tier (TIER_4_ENSEMBLE_SPLIT) still raises "
              "UningestibleCase", False)
    except UningestibleCase as exc:
        check("a genuine non-AUTO tier (TIER_4_ENSEMBLE_SPLIT) still raises "
              "UningestibleCase, with the tier-rejection message",
              "is not an auto-write tier" in str(exc))

    print(f"kg3-ingestion tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_kg3_ingestion():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
