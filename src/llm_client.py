"""
src/llm_client.py — Stage 3 LLM transport.

Talks to the two locally-served vLLM endpoints (MedGemma 4B on :8000,
OpenBioLLM 8B on :8001 per scripts/boot_infra.sh) over vLLM's
OpenAI-compatible API. Deliberately thin: this module knows how to call a
model, get structured JSON back, and measure how confident the model was. It
knows nothing about clinical validation, prompts, or the KG -- that is
mollm_ensemble.py's job.

WHY LOGPROBS RATHER THAN SELF-REPORTED CONFIDENCE.
docs/Provenance_Schema.md names both `raw_confidence_label` (the model saying
"HIGH") and `logprob_confidence` as separate Stage 3 fields, and the
distinction is load-bearing rather than decorative. LLM self-reported
confidence is well documented as poorly calibrated -- models say "high
confidence" in a stable, largely context-independent way. Token-level
log-probabilities measure what the model's own distribution actually put on
the answer it gave. So:

  * logprob_confidence drives the ensemble math and the routing thresholds.
  * raw_confidence_label is recorded but NOT used in any decision. It is kept
    so the dissertation can report whether the two agree, which is a genuinely
    useful calibration finding either way (docs/Evaluation_Criteria.md already
    plans an ECE/reliability analysis this feeds directly).

Getting a per-verdict probability out of a JSON response takes some care: the
verdict is a value inside a JSON object, not the whole completion, so the
completion-level average logprob would be dominated by structural tokens
(braces, field names, the reasoning string) that carry no decision
information. extract_verdict_confidence() locates the tokens that actually
spell the verdict value and scores only those.

PRIVACY NOTE: base URLs default to localhost. This matters beyond
configuration convenience -- the proposal's Legal section commits to running
the LLM ensemble "exclusively on local infrastructure or within a secure
private network" specifically to avoid third-party API exposure of clinical
text. Anything that points these at a remote endpoint breaks that commitment,
so the defaults are local and overriding them is an explicit act.
"""

import json
import math
import os
import re
import warnings

warnings.filterwarnings("ignore")

MEDGEMMA_BASE_URL = os.environ.get("MEDGEMMA_BASE_URL", "http://localhost:8000/v1")
OPENBIOLLM_BASE_URL = os.environ.get("OPENBIOLLM_BASE_URL", "http://localhost:8001/v1")

# OpenBioLLM 8B fine-tunes base Meta-Llama-3-8B (NOT 3.1), so its context
# window is 8,192 tokens -- not the 128K MedGemma 4B inherits from Gemma 3.
# Both ensemble members must receive identical input for their votes to be
# comparable, so 8,192 is the binding budget for the whole ensemble and the
# figure docs/MoLLM_Stage3_Retrieval_Design.md S6.5 budgets against.
CONTEXT_WINDOW_TOKENS = 8192
MAX_OUTPUT_TOKENS = 800

# Temperature 0. Stage 3 is a validation gate whose outputs are written into a
# provenance ledger and cited in an audit trail; the same entity and the same
# evidence must produce the same verdict on a re-run, or the "deterministic,
# traceable" claim fails the same way the Stage 2b LIMIT-1 bug made it fail.
TEMPERATURE = 0.0

# Number of alternatives requested per token position. Only needed so the
# chosen token's own logprob is reliably present in the response payload.
TOP_LOGPROBS = 5


class LLMUnavailable(RuntimeError):
    """Raised when a model endpoint cannot be reached or returns unusable
    output. Deliberately a distinct type: mollm_ensemble.py must be able to
    tell "this model is down" (which invalidates the ensemble and must route
    to HITL) apart from "this model returned a valid CONTRADICTED verdict".
    Collapsing those two would let an outage masquerade as a clinical finding.
    """


class LLMClient:
    """One vLLM endpoint."""

    def __init__(self, model_name: str, base_url: str, api_key: str = "EMPTY", timeout: float = 120.0):
        from openai import OpenAI

        self.model_name = model_name
        self.base_url = base_url
        # vLLM ignores the key but the OpenAI SDK requires one to be set.
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(self, system_prompt: str, user_prompt: str) -> dict:
        """Single completion. Returns the raw text plus token logprobs.

        Every failure path raises LLMUnavailable rather than returning a
        sentinel value, so a transport problem can never be silently scored as
        a low-confidence verdict.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise LLMUnavailable(f"{self.model_name} at {self.base_url}: {exc}") from exc

        choice = response.choices[0]
        tokens = []
        try:
            for item in choice.logprobs.content:
                tokens.append({"token": item.token, "logprob": item.logprob})
        except (AttributeError, TypeError):
            # Some serving configurations omit logprobs. Recorded as absent
            # rather than defaulted to a number -- a fabricated 1.0 would
            # silently inflate composite_confidence and could push a case past
            # the auto-accept threshold on no evidence at all.
            tokens = []

        return {
            "model": self.model_name,
            "text": choice.message.content or "",
            "tokens": tokens,
            "finish_reason": choice.finish_reason,
        }


def parse_json_response(text: str) -> dict:
    """Parses the model's JSON, tolerating markdown fences.

    Raises ValueError on unparseable output rather than returning {}: an empty
    dict would flow downstream as a verdict-less record and be indistinguishable
    from a model that genuinely abstained.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: the outermost {...} span. Models occasionally prepend a
        # sentence despite the schema instruction.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"could not parse JSON from response: {cleaned[:200]!r}")


def extract_verdict_confidence(tokens: list, verdict: str):
    """Probability the model assigned to the tokens spelling its own verdict.

    Walks the token stream looking for the contiguous run whose concatenation
    reconstructs `verdict`, then returns the geometric mean of those tokens'
    probabilities. Rationale for each choice:

      * Only the verdict tokens are scored. Averaging the whole completion
        would mix in braces, field names and the free-text reasoning, none of
        which say anything about how sure the model was of its decision, and
        all of which are high-probability filler that would wash out the
        signal.
      * Geometric mean (not arithmetic) because these are probabilities of a
        joint event -- the model had to emit every one of those tokens. It is
        also the length-normalised form of the sequence probability, so a
        three-token verdict is comparable with a one-token verdict rather than
        being systematically penalised.
      * Returns None, never a default, when the verdict cannot be located or
        logprobs are absent. The caller must be able to distinguish "we could
        not measure this" from "the model was unconfident"; a silent 0.5 would
        corrupt both the routing decision and the calibration analysis.
    """
    if not tokens or not verdict:
        return None

    target = verdict.strip()
    if not target:
        return None

    for i in range(len(tokens)):
        acc = ""
        for j in range(i, min(i + 24, len(tokens))):
            acc += tokens[j]["token"]
            stripped = acc.strip().strip('"').strip()
            if stripped == target:
                span = tokens[i:j + 1]
                logprob_sum = sum(t["logprob"] for t in span)
                return round(math.exp(logprob_sum / len(span)), 6)
            if len(stripped) > len(target) + 4:
                break
    return None


def build_clients() -> dict:
    """Constructs both ensemble members.

    Model names must match what each vLLM server was launched with (`--model`),
    which is what /v1/models reports; scripts/boot_infra.sh already probes both
    endpoints on boot.
    """
    return {
        "medgemma": LLMClient(
            model_name=os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it"),
            base_url=MEDGEMMA_BASE_URL,
        ),
        "openbiollm": LLMClient(
            model_name=os.environ.get("OPENBIOLLM_MODEL", "aaditya/Llama3-OpenBioLLM-8B"),
            base_url=OPENBIOLLM_BASE_URL,
        ),
    }
