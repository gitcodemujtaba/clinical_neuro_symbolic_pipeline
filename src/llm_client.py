"""
src/llm_client.py — Stage 3 LLM transport.

Talks to a local Ollama server (default http://localhost:11434) serving the
three-model ensemble decided on 2026-08-14 (docs/2026-08-14_Stage2_Alias_Fixes_And_Stage3_Provenance.md
§6): qwen2.5:3b, llama3.2:3b, phi4-mini. Deliberately thin: this module knows
how to call a model, get structured JSON back, and measure how confident the
model was. It knows nothing about clinical validation, prompts, or the KG --
that is mollm_ensemble.py's job.

2026-08-14 REPLACES the earlier two-model vLLM setup (BioMistral-7B-AWQ +
OpenBioLLM-Llama3-8B-AWQ, served over vLLM's OpenAI-compatible API on a single
T4). That design is fully retired, not kept as a fallback -- running two
different transports side by side is exactly the kind of ambiguity that was
causing confusion between which models a given run/doc/decision actually
used. If a doc still describes BioMistral/OpenBioLLM as current, it predates
this switch; see docs/2026-08-14_Stage2_Alias_Fixes_And_Stage3_Provenance.md
§6 for why the switch happened and docs/2026-08-14_Dead_Code_Audit.md for
what else was cleaned up alongside it.

WHY THREE SMALL MODELS RATHER THAN TWO LARGE ONES. Validated overnight
(2026-08-13/14) via scripts/experiment_3b_voting.py against real Stage 2
candidate lists: qwen2.5:3b, llama3.2:3b and phi4-mini are independently
sourced (Alibaba/Meta/Microsoft respectively), run comfortably on this box's
GPU concurrently via Ollama, and a 3-way vote gives a genuine majority
concept (3-0 / 2-1 / 1-1-1) that a 2-model ensemble structurally cannot --
with 2 voters there is no "majority", only "agree" or "disagree". Whether
combine()'s current strict unanimous-agreement rule (src/mollm_ensemble.py)
should grow a separate, lower-confidence 2-1 tier is an open question this
migration deliberately does NOT decide -- see that module for the current
rule, which now applies to 3 voters unchanged, and needs its own calibration
work before any 2-1 tier would be safe to trust (2026-08-14 report's finding
that composite_confidence is currently uninformative-to-harmful applies
regardless of ensemble size).

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

Confirmed empirically (2026-08-14) before relying on it: Ollama's Python
client (>=0.5, this box has 0.6.2) supports BOTH a JSON-schema `format` (real
guided decoding, not just the string "json" -- an out-of-vocabulary verdict
is structurally impossible, same property the old vLLM guided_json gave) AND
`logprobs=True`/`top_logprobs=N` on `generate()`, for all three target
models. Getting a per-verdict probability out of a JSON response still takes
care: the verdict is a value inside a JSON object, not the whole completion,
so the completion-level average logprob would be dominated by structural
tokens (braces, field names, the reasoning string) that carry no decision
information. extract_verdict_confidence() locates the tokens that actually
spell the verdict value and scores only those -- unchanged from before, since
Ollama's per-token logprob shape (token, logprob, top_logprobs) is
structurally the same list-of-dicts shape the old vLLM transport produced.

PRIVACY NOTE: the Ollama host defaults to localhost. This matters beyond
configuration convenience -- the proposal's Legal section commits to running
the LLM ensemble "exclusively on local infrastructure or within a secure
private network" specifically to avoid third-party API exposure of clinical
text. Anything that points this at a remote endpoint breaks that commitment,
so the default is local and overriding it is an explicit act.
"""

import json
import math
import os
import re
import warnings

import ollama

warnings.filterwarnings("ignore")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Overridable by environment variable (comma-separated) rather than a code
# edit, same reasoning the old BIOMISTRAL_MODEL/OPENBIOLLM_MODEL env vars
# had: the name must match what `ollama list` actually shows, and a mismatch
# should be a clear 404-style failure, not a silent no-op.
MODEL_NAMES = [
    m.strip() for m in os.environ.get(
        "MOLLM_MODELS", "qwen2.5:3b,llama3.2:3b,phi4-mini").split(",") if m.strip()
]

# All three target models natively support >=32K context (qwen2.5:3b: 32768,
# llama3.2:3b/phi4-mini: 131072 -- confirmed via `ollama show` 2026-08-14), so
# unlike the old two-model setup there is no hardware-forced ceiling. Kept
# deliberately modest anyway: every ensemble member must receive identical
# input for votes to be comparable (same principle as before), and nothing in
# Stage 3's prompts approaches this budget today. Revisit if a prompt is ever
# observed to truncate.
CONTEXT_WINDOW_TOKENS = 8192
MAX_OUTPUT_TOKENS = 800
PROMPT_BUDGET_TOKENS = CONTEXT_WINDOW_TOKENS - MAX_OUTPUT_TOKENS

# Temperature 0. Stage 3 is a validation gate whose outputs are written into a
# provenance ledger and cited in an audit trail; the same entity and the same
# evidence must produce the same verdict on a re-run, or the "deterministic,
# traceable" claim fails the same way the Stage 2b LIMIT-1 bug made it fail.
TEMPERATURE = 0.0

# 2026-08-13 (docs/2026-08-13_Code_Improvement_Proposals.md P4), carried
# forward from the vLLM transport: greedy decoding (TEMPERATURE=0.0) with no
# repetition guard is a known trigger for a model looping on the same
# sentence/clause until it hits the output cap (Holtzman et al. 2019).
# Ollama's `frequency_penalty` option (confirmed present on
# ollama._types.Options, 2026-08-14) is the same deterministic
# token-selection penalty vLLM's was -- not a source of randomness, so
# TEMPERATURE=0.0's reproducibility guarantee stays intact. The numeric value
# is carried over as a starting point, not re-validated against these three
# specific models yet; is_degenerate() below is what actually measures
# whether it is working, so drift here is self-detecting.
FREQUENCY_PENALTY = 0.4

# 2026-08-13 (docs/2026-08-13_Code_Improvement_Proposals.md P4). The penalty
# above is a FIX. These constants are the DETECTOR, and they matter more.
#
# WHY A DETECTOR IS NEEDED WHEN A FIX HAS ALREADY SHIPPED. Without a recorded
# flag there is no before/after number, only a hope. Worse, the symptom is
# otherwise INVISIBLE downstream: a repetition-looped verdict arrives
# indistinguishable from a model that genuinely lacked evidence, and then
# counts as a real disagreement in route()'s model_disagreement safety rule.
#
# THE SIGNATURE: the same sentence repeated 10-15+ times, ALWAYS terminating
# exactly at MAX_OUTPUT_TOKENS because the loop never reaches a stop token.
# Both conditions are required below: long legitimate output that happens to
# hit the cap is not degenerate, and a repetitive short answer that
# terminated normally is not either.
DEGENERATE_NGRAM = 6          # n-gram width for the repetition scan
DEGENERATE_MIN_TOKENS = 40    # below this, repetition is not diagnostic

# THE METRIC IS DISTINCT-N, NOT "how often does the single most common n-gram
# appear". distinct-n (unique n-grams / total n-grams, Li et al. 2016; the
# standard measure in the degeneration literature Holtzman et al. 2019 works
# in) does not care how the repetition is distributed -- a repeated SENTENCE
# produces many DISTINCT n-grams, each repeated, which a "top n-gram share"
# threshold would miss entirely.
DEGENERATE_DISTINCT_RATIO = 0.35  # flag below this

# Retained as a SECOND, independent trigger for the case distinct-n is
# weakest on: a single token or very short phrase repeated many times.
DEGENERATE_TOP_NGRAM_RATIO = 0.35

# One retry at a harder penalty when degeneration is detected. Deterministic,
# so the retry is reproducible; a second failure is accepted and FLAGGED
# rather than retried again.
DEGENERATE_RETRY_PENALTY = 1.0

# Number of alternatives requested per token position. Only needed so the
# chosen token's own logprob is reliably present in the response payload.
TOP_LOGPROBS = 5


def verdict_schema(allowed_verdicts, require_citation: bool = True) -> dict:
    """JSON Schema constraining a response to the legal verdict set.

    `require_citation` (docs/Stage3_Open_Issues.md Issue 2 experiment,
    2026-08-11). `cited_evidence` was unconditionally `required`, and the
    hypothesis on file was that a required-but-often-empty array under guided
    decoding invites the model to fill it rather than leave it `[]` -- callers
    should pass `require_citation=False` when retrieval found zero rules,
    since there is nothing legitimate to cite in that case regardless of what
    the model does. Deliberately record-scoped rather than global: a record
    WITH evidence should still be required to engage with it.

    WHY GUIDED DECODING RATHER THAN TRUSTING THE PROMPT. Without it, an
    out-of-vocabulary verdict is indistinguishable from genuine uncertainty:
    _query_one() converts it to INSUFFICIENT_EVIDENCE, which routes to HITL. A
    model that simply formats badly therefore inflates human review volume
    while looking like a model that is appropriately unsure. Constraining
    generation makes malformed output impossible rather than merely detected.

    A SECOND, LESS OBVIOUS BENEFIT. composite_confidence is derived from the
    logprob of the verdict tokens. Under free generation that probability is
    taken over the model's entire vocabulary, so it is diluted by tokens that
    were never legal answers. Constrained to the enum, it becomes the
    probability of THIS verdict AMONG THE VALID ONES -- which is the quantity
    the routing thresholds actually want.

    That is a genuine change to the confidence distribution, not a free win:
    absolute logprob values shift, so AUTO_VALIDATE_THRESHOLD and
    MOLLM_RESOLVE_THRESHOLD must be calibrated against guided output and cannot
    be carried over from an unguided run.

    `cited_evidence` is deliberately NOT constrained to known rule_ids. A
    fabricated citation is the hallucination the pipeline exists to catch, and
    making it structurally impossible would remove the signal rather than the
    problem -- verify_citations() must still be able to catch a model inventing
    a rule_id, because that tells us something about the model that a
    constrained decode would hide.

    `reasoning_basis` (docs/MoLLM_Redesign_Proposal.md S9.5/S9.6): self-reported
    ONLY when the model reasoned to a verdict with no guideline evidence
    retrieved (require_citation=False is the caller's signal for that case).
    Two values, both scoped to labeling correctness, never to clinical
    decision-making: "ontology_only" (rests on Channel C's structured facts
    alone) or "medical_terminology_knowledge" (required the model's own
    knowledge beyond what Channel C states). Deliberately left OUT of
    `required` even when require_citation=False: forcing a field under guided
    decoding does not reliably improve what gets filled in, only that
    decoding fails less.
    """
    required = ["verdict", "reasoning", "confidence"]
    if require_citation:
        required.append("cited_evidence")

    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": sorted(allowed_verdicts)},
            "reasoning": {"type": "string"},
            "cited_evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_id": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ["rule_id", "quote"],
                },
            },
            "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
            "request": {
                "type": "string",
                "enum": ["MORE_RULES", "SUPPRESSED_RULES", "CANDIDATE_DETAIL", "NONE"],
            },
            "reasoning_basis": {
                "type": "string",
                "enum": ["ontology_only", "medical_terminology_knowledge"],
            },
        },
        "required": required,
    }


class LLMUnavailable(RuntimeError):
    """Raised when a model endpoint cannot be reached or returns unusable
    output. Deliberately a distinct type: mollm_ensemble.py must be able to
    tell "this model is down" (which invalidates the ensemble and must route
    to HITL) apart from "this model returned a valid CONTRADICTED verdict".
    Collapsing those two would let an outage masquerade as a clinical finding.
    """


class LLMClient:
    """One Ollama-served model."""

    def __init__(self, model_name: str, host: str = None, timeout: float = 120.0):
        self.model_name = model_name
        self.host = host or OLLAMA_HOST
        self.decoding_mode = "unknown"
        self._client = ollama.Client(host=self.host, timeout=timeout)

    def complete(self, system_prompt: str, user_prompt: str,
                 schema: dict = None, max_tokens: int = None) -> dict:
        """Single completion. Returns the raw text plus token logprobs.

        `schema` enables Ollama's structured-output mode (confirmed 2026-08-14
        against all three target models), which makes an out-of-vocabulary
        verdict structurally impossible rather than merely detectable -- see
        verdict_schema() for why that matters and what it costs. Falls back to
        plain "json" mode if the server rejects the schema, because a model
        or Ollama version that doesn't support structured output should
        degrade rather than fail the whole run.

        `max_tokens` (2026-08-15, added for scripts/analysis/mollm_wholenote_ensemble.py's
        experiment): overrides MAX_OUTPUT_TOKENS=800, which is sized for a
        single verdict object and far too small for a guided-decoding array
        response covering many entities at once. None (default) preserves
        today's behavior exactly for every existing caller.

        Every failure path raises LLMUnavailable rather than returning a
        sentinel, so a transport problem can never be silently scored as a
        low-confidence verdict.
        """
        def _build_attempts(frequency_penalty=FREQUENCY_PENALTY):
            base_kwargs = dict(
                model=self.model_name,
                prompt=user_prompt,
                system=system_prompt,
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": max_tokens or MAX_OUTPUT_TOKENS,
                    "frequency_penalty": frequency_penalty,
                },
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
            )
            out = []
            if schema:
                out.append(dict(base_kwargs, format=schema))
            out.append(dict(base_kwargs, format="json"))
            return out

        def _run(attempts):
            for i, kwargs in enumerate(attempts):
                try:
                    resp = self._client.generate(**kwargs)
                    self.decoding_mode = (
                        "guided_json_schema" if i == 0 and schema else "json_object_unguided"
                    )
                    return resp, None
                except (ollama.RequestError, ollama.ResponseError) as exc:
                    last = exc
            return None, last

        response, last_exc = _run(_build_attempts())
        if response is None:
            raise LLMUnavailable(
                f"{self.model_name} at {self.host}: {last_exc}") from last_exc

        # 2026-08-13 (P4), carried forward: one deterministic retry at a
        # harder frequency penalty when the first generation degenerated into
        # a repetition loop. Both outcomes are recorded: `degenerate_generation`
        # reflects the FINAL text (so a successful retry clears it), while
        # `degenerate_retried` records that a retry happened at all.
        degenerate, degenerate_detail = is_degenerate(
            response.response or "", response.done_reason)
        degenerate_retried = False
        if degenerate:
            degenerate_retried = True
            retry_resp, _retry_exc = _run(
                _build_attempts(frequency_penalty=DEGENERATE_RETRY_PENALTY))
            if retry_resp is not None:
                retry_degenerate, retry_detail = is_degenerate(
                    retry_resp.response or "", retry_resp.done_reason)
                # Keep the retry only if it actually helped.
                if not retry_degenerate:
                    response = retry_resp
                    degenerate, degenerate_detail = False, retry_detail

        tokens = []
        for lp in (response.logprobs or []):
            top = [{"token": t.token, "logprob": t.logprob}
                  for t in (lp.top_logprobs or [])]
            tokens.append({"token": lp.token, "logprob": lp.logprob, "top_logprobs": top})

        return {
            "model": self.model_name,
            "text": response.response or "",
            "tokens": tokens,
            "finish_reason": response.done_reason,
            # Ollama's done_reason is "length" when num_predict was hit, same
            # string the old vLLM finish_reason used for the equivalent case
            # -- no translation needed downstream.
            "truncated": response.done_reason == "length",
            "decoding_mode": getattr(self, "decoding_mode", "unknown"),
            # 2026-08-13 (P4), carried forward: True when the text below is
            # STILL a repetition loop after the retry. src/mollm_ensemble.py's
            # combine()/route() read this to keep a degenerate verdict out of
            # the model_disagreement safety rule.
            "degenerate_generation": degenerate,
            "degenerate_retried": degenerate_retried,
            "degenerate_detail": degenerate_detail,
        }


def _ngram_ratios(text: str, n: int) -> dict:
    """Distinct-n and top-n-gram-share over whitespace-split tokens. Shared by
    both passes of is_degenerate() so the literal and template-normalized
    checks apply the exact same arithmetic to different strings, rather than
    risking two implementations that could drift apart.

    Returns None (not a dict of zeros) when there are too few tokens to form
    even one n-gram -- a distinct signal from "computed and found repetitive"
    that callers must not treat as "not repetitive".
    """
    tokens = text.split()
    if len(tokens) < n:
        return None
    grams = {}
    total = 0
    for i in range(len(tokens) - n + 1):
        g = " ".join(tokens[i:i + n])
        grams[g] = grams.get(g, 0) + 1
        total += 1
    if not total:
        return None
    top_gram, top_count = max(grams.items(), key=lambda kv: kv[1])
    return {
        "distinct_ratio": len(grams) / total,
        "top_ratio": top_count / total,
        "top_count": top_count,
        "top_gram": top_gram,
        "n_ngrams": total,
        "n_unique_ngrams": len(grams),
    }


# 2026-08-13 (verification follow-up, docs/2026-08-13_Implementation_Verification.md):
# a case the literal check above misses entirely -- a chain of "The 'X' is a
# Y of the 'Z' concept" clauses cycling through ~20 unrelated ontology terms,
# hitting the output cap without ever repeating the same 6 words twice.
# Replacing quoted spans with a single placeholder before re-running
# _ngram_ratios() collapses that skeleton back to a small repeated set, the
# same way it would for literal repetition -- reusing the same thresholds
# rather than tuning a second pair that could drift out of sync with the
# first.
_QUOTED_SPAN_RE = re.compile(r"'[^']*'|\"[^\"]*\"")


def is_degenerate(text: str, finish_reason: str = None) -> tuple:
    """Detects the repetition-loop failure mode described at
    DEGENERATE_NGRAM above. Returns (is_degenerate, detail_dict).

    THE TEST: distinct-n. What fraction of the output's n-grams are UNIQUE.
    Prose that is going somewhere keeps producing new n-grams, so the ratio
    stays above ~0.9; a model looping over the same clause reuses the same
    small set forever, so the ratio collapses.

    A SECOND pass runs the identical test on the text with quoted spans
    blanked out, catching a "conceptual drift" loop that reuses the same
    templated sentence around a DIFFERENT quoted name each cycle. Either pass
    flagging is sufficient.

    Combined with finish_reason == "length" -- the loop's other invariant,
    since it terminates only when the token budget runs out.

    WHY finish_reason IS REQUIRED RATHER THAN ADVISORY. Clinical text is
    genuinely repetitive -- medication lists, templated section headers, "no
    acute distress" repeated per system. Flagging on repetition alone would
    fire on legitimate output. Requiring that the model ALSO ran out of
    budget is what separates "this text repeats" from "this model could not
    stop".

    detail_dict is returned for persistence: the ratio and the offending
    n-gram make a flagged decision auditable rather than asking anyone to
    take the classifier's word for it.
    """
    if not text:
        return False, {"reason": "empty_text"}

    tokens = text.split()
    if len(tokens) < DEGENERATE_MIN_TOKENS:
        return False, {"reason": "too_short_to_judge", "n_tokens": len(tokens)}

    literal = _ngram_ratios(text, DEGENERATE_NGRAM)
    if literal is None:
        return False, {"reason": "no_ngrams"}

    normalized_text = _QUOTED_SPAN_RE.sub("_", text)
    templated = _ngram_ratios(normalized_text, DEGENERATE_NGRAM)

    def _repetitive(r):
        return bool(r) and (r["distinct_ratio"] < DEGENERATE_DISTINCT_RATIO
                             or r["top_ratio"] >= DEGENERATE_TOP_NGRAM_RATIO)

    literal_repetitive = _repetitive(literal)
    templated_repetitive = _repetitive(templated)
    repetitive = literal_repetitive or templated_repetitive
    hit_cap = finish_reason == "length"

    detail = {
        "distinct_ngram_ratio": round(literal["distinct_ratio"], 4),
        "top_ngram_ratio": round(literal["top_ratio"], 4),
        "top_ngram_count": literal["top_count"],
        "n_ngrams": literal["n_ngrams"],
        "n_unique_ngrams": literal["n_unique_ngrams"],
        "finish_reason": finish_reason,
        "hit_output_cap": hit_cap,
        "top_ngram": literal["top_gram"][:120],
        "template_repetitive": templated_repetitive,
    }
    if templated and templated_repetitive:
        detail["template_distinct_ngram_ratio"] = round(templated["distinct_ratio"], 4)
        detail["template_top_ngram_ratio"] = round(templated["top_ratio"], 4)
        detail["template_top_ngram"] = templated["top_gram"][:120]

    if repetitive and hit_cap:
        detail["reason"] = ("repetition_loop" if literal_repetitive
                             else "template_repetition_loop")
        return True, detail
    detail["reason"] = ("repetitive_but_terminated_normally" if repetitive else "ok")
    return False, detail


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

        # Salvage for output truncated at max_tokens. `verdict` is the first
        # property in verdict_schema(), and guided decoding emits properties in
        # schema order, so a response cut off mid-`reasoning` still contains a
        # complete, schema-constrained verdict. Recovering it converts a
        # spurious HITL routing into a usable vote.
        #
        # Deliberately narrow: it recovers ONLY the verdict, and the caller
        # still sees truncated=True and an empty cited_evidence list. It does
        # not invent a citation, so verify_citations() will find nothing to
        # verify and the record cannot be auto-validated on salvaged output.
        m = re.search(r'"verdict"\s*:\s*"([A-Z0-9_]+)"', cleaned)
        if m:
            reasoning = re.search(r'"reasoning"\s*:\s*"(.*)', cleaned, re.DOTALL)
            return {
                "verdict": m.group(1),
                "reasoning": (reasoning.group(1)[:2000] if reasoning else ""),
                "cited_evidence": [],
                "salvaged_from_truncation": True,
            }
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


def extract_candidate_alternatives(tokens: list, verdict: str, allowed_verdicts) -> dict:
    """Probability the model assigned to verdicts it did NOT choose
    (docs/MoLLM_Redesign_Proposal.md S4.3, "rejected-candidate bookkeeping").

    Locates the verdict token span exactly as extract_verdict_confidence()
    does, then reads the LAST token's top_logprobs (see complete() -- these
    were being requested via TOP_LOGPROBS) for alternative tokens. For each
    alternate, reconstructs what the full string would be with that token
    substituted in and keeps it ONLY if the reconstruction is itself a member
    of allowed_verdicts.

    This sidesteps assuming anything about tokenizer boundaries (e.g. that a
    candidate index is always exactly one token) -- an alternate is trusted
    only when swapping it in produces another real, legal verdict string, not
    because of where it fell in the token stream. The cost is conservatism:
    it can only surface alternates that differ from the chosen verdict in
    exactly its final token, which covers the common case
    (RESOLVED_TO_CANDIDATE_2 vs RESOLVED_TO_CANDIDATE_3) but not every
    possible tokenization. Returns {} rather than guessing when it can't
    confirm a reconstruction -- consistent with extract_verdict_confidence()
    returning None rather than a default when it can't measure something.
    """
    if not tokens or not verdict:
        return {}
    target = verdict.strip()
    if not target:
        return {}
    allowed = set(allowed_verdicts or [])

    for i in range(len(tokens)):
        acc = ""
        for j in range(i, min(i + 24, len(tokens))):
            acc += tokens[j]["token"]
            stripped = acc.strip().strip('"').strip()
            if stripped == target:
                last = tokens[j]
                alternates = last.get("top_logprobs") or []
                prefix = "".join(t["token"] for t in tokens[i:j])
                out = {}
                for alt in alternates:
                    alt_token = alt.get("token", "")
                    if alt_token == last.get("token"):
                        continue  # the chosen token itself, not an alternative
                    candidate_str = (prefix + alt_token).strip().strip('"').strip()
                    if candidate_str in allowed and candidate_str != target:
                        out[candidate_str] = round(math.exp(alt["logprob"]), 6)
                return out
            if len(stripped) > len(target) + 4:
                break
    return {}


def build_clients(timeout: float = None) -> dict:
    """Constructs all ensemble members (2026-08-14: qwen2.5:3b, llama3.2:3b,
    phi4-mini -- see MODEL_NAMES above).

    Model names must match what `ollama list` shows on this host. Returns a
    dict keyed by model name (rather than the old fixed "biomistral"/
    "openbiollm" keys) -- mollm_ensemble.py's `for name, client in
    clients.items()` loop was already written generically over however many
    entries this returns, so the ensemble size lives in exactly one place:
    MODEL_NAMES above.

    `timeout` (2026-08-15): overrides LLMClient's 120s default. None
    preserves today's behavior for every existing caller -- added for the
    much larger prompts/completions in scripts/analysis/mollm_wholenote_ensemble.py's
    experiment, which need more headroom than a single-verdict call does.
    """
    return {name: LLMClient(model_name=name, timeout=timeout or 120.0) for name in MODEL_NAMES}
