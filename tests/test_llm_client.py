"""
tests/test_llm_client.py -- src/llm_client.py's TRANSPORT_MAX_RETRIES logic
(LLMClient.complete()'s bounded retry for a genuine connection/timeout
failure, added 2026-08-17 after confirming a real, latent gap by reading
ollama's client source: httpx.ConnectError is translated into a bare
ConnectionError, and httpx.TimeoutException isn't translated/caught at all,
so neither was previously in _run()'s except clause -- a real transport
failure would have propagated uncaught with zero retry).

No live Ollama server needed: LLMClient._client.generate is monkeypatched
directly, and time.sleep is patched to a no-op so the retry-backoff tests
run instantly rather than actually waiting.
"""
import os
import sys
import types

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.llm_client as llm_client  # noqa: E402
from src.llm_client import LLMClient, LLMUnavailable  # noqa: E402


class _FakeResponse:
    def __init__(self, text='{"verdict": "SUPPORTED_1", "reasoning": "ok", '
                             '"confidence": "HIGH", "cited_evidence": []}'):
        self.response = text
        self.done_reason = "stop"
        self.logprobs = []


def _client_with_fake_generate(side_effects):
    """An LLMClient whose _client.generate() call sequence is scripted:
    each entry in side_effects is either an Exception instance (raised) or
    a _FakeResponse (returned). Never touches a real Ollama host.
    """
    client = LLMClient(model_name="test-model", host="http://fake-host:0")
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        i = len(calls) - 1
        outcome = side_effects[min(i, len(side_effects) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client._client = types.SimpleNamespace(generate=fake_generate)
    return client, calls


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    original_sleep = llm_client.time.sleep
    llm_client.time.sleep = lambda *_a, **_k: None
    try:
        # ------------------------------------------------------------
        # A connection error on the first attempt, success on the retry.
        # ------------------------------------------------------------
        client, calls = _client_with_fake_generate(
            [ConnectionError("refused"), _FakeResponse()])
        result = client.complete("system", "user")
        check("a transient ConnectionError is retried and eventually succeeds",
              result["text"].startswith("{"))
        check("transport_retries reflects how many retries were actually needed",
              result["transport_retries"] == 1)
        check("exactly 2 generate() calls were made (1 failure + 1 success)",
              len(calls) == 2)

        # ------------------------------------------------------------
        # httpx.TimeoutException specifically (the untranslated case) --
        # confirms it's caught at all, not just ConnectionError.
        # ------------------------------------------------------------
        client, calls = _client_with_fake_generate(
            [httpx.TimeoutException("timed out"), _FakeResponse()])
        result = client.complete("system", "user")
        check("httpx.TimeoutException (not translated by ollama's client) is "
              "also caught and retried -- previously would have propagated "
              "uncaught with zero retry",
              result["text"].startswith("{") and result["transport_retries"] == 1)

        # ------------------------------------------------------------
        # Exhausts all retries -- must raise LLMUnavailable, not hang or
        # silently return something.
        # ------------------------------------------------------------
        always_fails = ConnectionError("refused")
        client, calls = _client_with_fake_generate([always_fails])
        try:
            client.complete("system", "user")
            check("exhausting all transport retries raises LLMUnavailable", False)
        except LLMUnavailable as exc:
            check("exhausting all transport retries raises LLMUnavailable",
                  True)
            check("the raised message records how many retries were attempted",
                  f"{llm_client.TRANSPORT_MAX_RETRIES}" in str(exc)
                  or "transport retry attempt" in str(exc))
        # Each logical retry attempt tries BOTH the schema and plain-json
        # variants inside _run() -- with no schema passed here, that's 1
        # call per _run(), so total calls == 1 (first) + TRANSPORT_MAX_RETRIES.
        check("exactly 1 + TRANSPORT_MAX_RETRIES generate() calls were made, "
              "not fewer (retry didn't fire) or unbounded (no cap)",
              len(calls) == 1 + llm_client.TRANSPORT_MAX_RETRIES)

        # ------------------------------------------------------------
        # A non-transport error (ollama.ResponseError) must NOT trigger the
        # transport retry loop -- retrying a malformed request/HTTP error
        # can't help, and doing so anyway would just slow every genuine
        # failure down for no benefit.
        # ------------------------------------------------------------
        import ollama
        client, calls = _client_with_fake_generate(
            [ollama.ResponseError("bad request", 400)])
        try:
            client.complete("system", "user")
            check("a non-transport ResponseError does not trigger retries", False)
        except LLMUnavailable:
            check("a non-transport ResponseError does not trigger retries", True)
        # No schema passed -> _build_attempts() has exactly 1 entry (plain
        # json), so _run() makes exactly 1 call; the retry loop must not add
        # more on top of that for a non-transport failure.
        check("no transport retry was attempted for a non-transport failure",
              len(calls) == 1)
    finally:
        llm_client.time.sleep = original_sleep

    print(f"llm-client tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


def test_llm_client():
    assert run()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
