"""
tests/test_kg3_query.py -- src/kg3_query.py's count_kg3_confirmations().

Added 2026-08-30 alongside the new kg3_confirmation_count calibrator
feature (src/mollm_tier_calibrator.py, FEATURE_SET_VERSION=2) and its
route_tier() wiring (src/mollm_tier_gate.py's kg3_driver/kg3_driver_factory
params, covered separately in tests/test_tier_gate.py).

No live Memgraph connection is used here -- a minimal fake driver
(session()/run()/single()) stands in, matching this module's own
never-raise contract: missing driver, missing args, or any query error
must all come back 0, never None, never propagate an exception.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.kg3_query import count_kg3_confirmations  # noqa: E402


class _FakeRecord(dict):
    def __getitem__(self, key):
        return dict.get(self, key)


class _FakeResult:
    def __init__(self, n):
        self._n = n

    def single(self):
        return None if self._n is None else _FakeRecord(n=self._n)


class _FakeSession:
    def __init__(self, n=0, raise_on_run=False):
        self._n = n
        self._raise = raise_on_run

    def run(self, query, **kwargs):
        if self._raise:
            raise RuntimeError("simulated query failure")
        return _FakeResult(self._n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    def __init__(self, n=0, raise_on_session=False, raise_on_run=False):
        self._n = n
        self._raise_on_session = raise_on_session
        self._raise_on_run = raise_on_run

    def session(self):
        if self._raise_on_session:
            raise RuntimeError("simulated connection failure")
        return _FakeSession(self._n, raise_on_run=self._raise_on_run)


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    check("driver=None returns 0, never raises",
          count_kg3_confirmations(None, "aspirin", 12345) == 0)
    check("empty entity_text returns 0",
          count_kg3_confirmations(_FakeDriver(9), "", 12345) == 0)
    check("None entity_text returns 0",
          count_kg3_confirmations(_FakeDriver(9), None, 12345) == 0)
    check("concept_id=None returns 0",
          count_kg3_confirmations(_FakeDriver(9), "aspirin", None) == 0)

    check("a real match returns the driver's own count, unmodified",
          count_kg3_confirmations(_FakeDriver(3), "aspirin", 12345) == 3)
    check("a zero-match result (single() returns n=0) returns 0, "
         "not falsely treated as 'no record found'",
          count_kg3_confirmations(_FakeDriver(0), "aspirin", 12345) == 0)

    check("driver.session() raising is swallowed, returns 0",
          count_kg3_confirmations(
              _FakeDriver(raise_on_session=True), "aspirin", 12345) == 0)
    check("session.run() raising is swallowed, returns 0",
          count_kg3_confirmations(
              _FakeDriver(raise_on_run=True), "aspirin", 12345) == 0)

    # single() returning None (a real, if unlikely, driver behavior) must
    # not crash the record["n"] lookup.
    class _NoneResultSession(_FakeSession):
        def run(self, query, **kwargs):
            return _FakeResult(None)

    class _NoneResultDriver(_FakeDriver):
        def session(self):
            return _NoneResultSession()

    check("single() returning None (no record) is treated as 0, not a crash",
          count_kg3_confirmations(_NoneResultDriver(), "aspirin", 12345) == 0)

    print(f"kg3-query tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_kg3_query():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
