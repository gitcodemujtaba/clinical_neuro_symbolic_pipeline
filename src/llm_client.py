"""
src/llm_client.py — Stage 3 LLM transport.

Talks to the two locally-served vLLM endpoints (BioMistral 7B on :8000,
OpenBioLLM 8B on :8001 per scripts/boot_infra.sh) over vLLM's
OpenAI-compatible API. Deliberately thin: this module knows how to call a
model, get structured JSON back, and measure how confident the model was. It
knows nothing about clinical validation, prompts, or the KG -- that is
mollm_ensemble.py's job.

WHY BIOMISTRAL RATHER THAN MEDGEMMA (2026-08-09).
The proposal named MedGemma 4B as the first ensemble member. It cannot run on
the project's Tesla T4 (compute capability 7.5), and this is a hard hardware
limit rather than a configuration problem -- every dtype is closed:

  * float16  -- vLLM refuses it for the `gemma3` model type outright
                ("does not support float16. Reason: Numerical instability"),
                raised during ModelConfig construction.
  * bfloat16 -- CUDAPlatform.check_if_supports_dtype() requires compute
                capability >= 8.0; T4 is 7.5. (Note that PyTorch's
                torch.cuda.is_bf16_supported() returns True on a T4 because it
                counts emulation, so the torch-level check is NOT a valid
                proxy for what vLLM will accept.)
  * float32  -- ~16GB of weights for a 4B model, against 15.36GB of VRAM.

This applies to the whole MedGemma family, since 4B, 1.5-4B and 27B are all
Gemma-3-based, and quantisation does not help because the rejected quantity is
the compute dtype rather than the weight format.

BioMistral 7B (AWQ) replaces it. The substitution preserves the property the
ensemble actually depends on: OpenBioLLM fine-tunes Llama-3 and BioMistral
fine-tunes Mistral-7B, so the two members have genuinely independent
pre-training and failure modes. An ensemble of two Llama-3 derivatives would
agree for correlated reasons and quietly weaken the disagreement rule that
routes contested extractions to human review.

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

BIOMISTRAL_BASE_URL = os.environ.get("BIOMISTRAL_BASE_URL", "http://localhost:8000/v1")
OPENBIOLLM_BASE_URL = os.environ.get("OPENBIOLLM_BASE_URL", "http://localhost:8001/v1")

# OpenBioLLM 8B fine-tunes base Meta-Llama-3-8B (NOT 3.1), so its context
# window is 8,192 tokens. BioMistral inherits Mistral-7B-v0.1's 32K positions,
# so OpenBioLLM is the binding constraint. Both ensemble members must receive
# identical input for their votes to be comparable, so 8,192 is the budget for
# the whole ensemble and the figure docs/MoLLM_Stage3_Retrieval_Design.md S6.5
# budgets against. Note the servers are launched with --max-model-len 4096 to
# fit both models on one 15.36GB card, so 4,096 is the operative limit in
# practice; 8,192 is the ceiling if they are ever run on separate GPUs.
CONTEXT_WINDOW_TOKENS = 8192
MAX_OUTPUT_TOKENS = 800

# What the servers are ACTUALLY launched with (scripts/start_vllm.sh). This is
# the number that matters at request time: vLLM rejects prompt+max_tokens
# exceeding max_model_len with a 400, so a prompt sized against the 8,192
# architectural ceiling would fail against a 4,096-token server. Kept separate
# from CONTEXT_WINDOW_TOKENS rather than folded into it because they answer
# different questions -- "what could this model take" versus "what did we give
# this process room for" -- and conflating them is how the ceiling silently
# becomes the budget again if the launch flags ever change.
SERVED_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
PROMPT_BUDGET_TOKENS = SERVED_MODEL_LEN - MAX_OUTPUT_TOKENS

# Temperature 0. Stage 3 is a validation gate whose outputs are written into a
# provenance ledger and cited in an audit trail; the same entity and the same
# evidence must produce the same verdict on a re-run, or the "deterministic,
# traceable" claim fails the same way the Stage 2b LIMIT-1 bug made it fail.
TEMPERATURE = 0.0

# 2026-08-13: greedy decoding (TEMPERATURE=0.0) with no repetition guard was
# found to cause a genuine, corpus-wide reliability bug -- 11.9% of ALL
# per-model verdicts (all of them BioMistral, 0 from OpenBioLLM) have
# `reasoning` degenerate into the same sentence repeated a dozen+ times
# (once, into a multi-thousand-character garbage digit string inside
# cited_evidence.rule_id), always paired with raw_confidence_label=None and
# a default verdict of INSUFFICIENT_EVIDENCE regardless of how explicit the
# entity is in the note text. This is the textbook failure mode for greedy
# decoding once genuine content is exhausted (Holtzman et al. 2019), not a
# real second opinion -- and it was inflating route()'s model_disagreement
# rate against entities OpenBioLLM answered correctly and confidently.
# frequency_penalty is a deterministic token-selection penalty, NOT a source
# of randomness -- applying it leaves TEMPERATURE=0.0's reproducibility
# guarantee (same entity + same evidence -> same verdict on a re-run) fully
# intact while breaking the repetition loop. Applied to both models for
# symmetry (the ensemble is only comparable if both members see identical
# generation settings, same reasoning as PROMPT_BUDGET_TOKENS above), even
# though OpenBioLLM was not observed to degenerate in this diagnosis.
FREQUENCY_PENALTY = 0.4

# 2026-08-13 (docs/2026-08-13_Code_Improvement_Proposals.md P4). The penalty
# above is a FIX. These constants are the DETECTOR, and they matter more.
#
# WHY A DETECTOR IS NEEDED WHEN A FIX HAS ALREADY SHIPPED. FREQUENCY_PENALTY
# was added and unit-verified but never re-validated against a live batch --
# the 2026-08-13 report lists that as outstanding (S9.5) precisely because
# nothing in the pipeline records whether a generation degenerated. Without a
# recorded flag there is no before/after number, only a hope. Worse, the
# symptom is currently INVISIBLE downstream: a repetition-looped verdict
# arrives as INSUFFICIENT_EVIDENCE, structurally identical to a model that
# genuinely lacked evidence, and then counts as a real disagreement in
# route()'s model_disagreement safety rule -- the single largest HITL category
# at 69.8% of decisions. Some unknown fraction of that is a decoding bug being
# read as clinical caution.
#
# THE SIGNATURE, from the report's S4.2 measurement of 475 affected verdicts:
# the same sentence repeated 10-15+ times, ALWAYS terminating exactly at
# MAX_OUTPUT_TOKENS (finish_reason == "length") because the loop never reaches
# a stop token. Both conditions are required below: long legitimate output that
# happens to hit the cap is not degenerate, and a repetitive short answer that
# terminated normally is not either.
DEGENERATE_NGRAM = 6          # n-gram width for the repetition scan
DEGENERATE_MIN_TOKENS = 40    # below this, repetition is not diagnostic

# THE METRIC IS DISTINCT-N, NOT "how often does the single most common n-gram
# appear". That distinction cost a test failure worth recording here, because
# the wrong version looks reasonable: an obvious repetition loop -- one
# 11-word sentence repeated 15 times -- has its most-common 6-gram appearing
# only 15 times out of 160, i.e. 9%, which no sensible "top n-gram share"
# threshold would flag. The repeated SENTENCE produces many DISTINCT 6-grams,
# each repeated. Measuring the top n-gram therefore misses precisely the
# failure mode this exists to catch.
#
# distinct-n (unique n-grams / total n-grams, Li et al. 2016; the standard
# measure in the degeneration literature Holtzman et al. 2019 works in) does
# not care how the repetition is distributed. Same example: 11 unique 6-grams
# over 160 = 0.069. Varied prose sits above 0.9.
DEGENERATE_DISTINCT_RATIO = 0.35  # flag below this

# Retained as a SECOND, independent trigger for the degenerate case distinct-n
# is weakest on: a single token or very short phrase repeated thousands of
# times (the report's multi-thousand-character digit string). distinct-n
# catches that too, but this makes the intent explicit and gives the audit
# detail something readable to point at.
DEGENERATE_TOP_NGRAM_RATIO = 0.35

# One retry at a harder penalty when degeneration is detected. Deterministic
# (frequency_penalty is not a randomness source, same argument as above), so
# the retry is reproducible; a second failure is accepted and FLAGGED rather
# than retried again, because three round-trips per degenerate call would cost
# more than the verdict is worth on a 24GB single-GPU box.
DEGENERATE_RETRY_PENALTY = 1.0

# Number of alternatives requested per token position. Only needed so the
# chosen token's own logprob is reliably present in the response payload.
TOP_LOGPROBS = 5

# Model defaults. Both are AWQ 4-bit: the two together must fit in one 15.36GB
# T4, which fp16 weights (14.5GB + 16GB) cannot. AWQ's GEMM kernels require
# compute capability >= 7.5, which the T4 exactly meets.
#
# Overridable by environment variable rather than code edit, because the string
# must match whatever `--model` the vLLM server was launched with -- a mismatch
# returns a 404 that reads like a network failure.
DEFAULT_BIOMISTRAL = os.environ.get(
    "BIOMISTRAL_MODEL", "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM")
DEFAULT_OPENBIOLLM = os.environ.get("OPENBIOLLM_MODEL", "bartowski/OpenBioLLM-Llama3-8B-AWQ")


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
    generation makes malformed output impossible rather than merely detected,
    which matters more here than swapping in a better instruction-follower --
    on a 15GB Turing card the choice of models is narrow anyway.

    A SECOND, LESS OBVIOUS BENEFIT. composite_confidence is derived from the
    logprob of the verdict tokens. Under free generation that probability is
    taken over the model's entire vocabulary, so it is diluted by tokens that
    were never legal answers. Constrained to the enum, it becomes the
    probability of THIS verdict AMONG THE VALID ONES -- which is the quantity
    the routing thresholds actually want.

    That is a genuine change to the confidence distribution, not a free win:
    absolute logprob values shift, so AUTO_VALIDATE_THRESHOLD and
    MOLLM_RESOLVE_THRESHOLD must be calibrated against guided output and cannot
    be carried over from an unguided run. Both were already flagged as
    calibration targets (docs/MoLLM_Stage3_Retrieval_Design.md S8); this makes
    recalibration mandatory rather than optional.

    `cited_evidence` is deliberately NOT constrained to known rule_ids. A
    fabricated citation is the hallucination the pipeline exists to catch, and
    making it structurally impossible would remove the signal rather than the
    problem -- verify_citations() must still be able to catch a model inventing
    a rule_id, because that tells us something about the model that a
    constrained decode would hide.

    `reasoning_basis` (docs/MoLLM_Redesign_Proposal.md S9.5/S9.6, added
    2026-08-11): self-reported ONLY when the model reasoned to a verdict with
    no guideline evidence retrieved (require_citation=False is the caller's
    signal for that case). Two values, both scoped to labeling correctness,
    never to clinical decision-making (S9.6): "ontology_only" (the verdict
    rests on Channel C's structured facts -- FSN, semantic tag, parents --
    alone) or "medical_terminology_knowledge" (the verdict required the
    model's own knowledge of what the term/concept means, beyond what Channel
    C states). Deliberately left OUT of `required` even when
    require_citation=False: Issue 2's required-field experiment
    (docs/Stage3_Open_Issues.md) found that forcing a field under guided
    decoding does not reliably improve what gets filled in, only that decoding
    fails less; an unforced self-report a model chooses to give is not more
    trustworthy for being unforced, but a forced one is not obviously more
    trustworthy either, so this stays optional and its absence just means
    "not reported" rather than being read as a failure.
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


def _is_role_error(exc) -> bool:
    """True if a request failed because the chat template rejected the roles.

    Matched on message text because the OpenAI SDK surfaces this as a generic
    BadRequestError with no machine-readable discriminator. Kept narrow on
    purpose: a broad match would fold the system prompt away in response to
    unrelated 400s, quietly changing what the model was asked without anything
    in the record saying so.
    """
    msg = str(exc or "").lower()
    return ("roles must alternate" in msg
            or "system role not supported" in msg
            or ("system" in msg and "not supported" in msg)
            or "only user and assistant roles" in msg)


class LLMClient:
    """One vLLM endpoint."""

    def __init__(self, model_name: str, base_url: str, api_key: str = "EMPTY", timeout: float = 120.0):
        from openai import OpenAI

        self.model_name = model_name
        self.base_url = base_url
        self.decoding_mode = "unknown"
        # Whether this endpoint's chat template accepts a `system` role.
        # Detected on first use rather than configured, because it is a property
        # of the served model's template, not of our configuration, and hard-
        # coding it per model would silently break on the next model swap.
        # BioMistral inherits Mistral-7B-v0.1's template and rejects the system
        # role with a 400 ("Conversation roles must alternate
        # user/assistant/user/assistant/..."); OpenBioLLM's Llama-3 template
        # accepts it. Both members must still receive the SAME instructions for
        # their votes to be comparable, so the fix folds the system prompt into
        # the user turn rather than dropping it.
        self._fold_system = False
        # vLLM ignores the key but the OpenAI SDK requires one to be set.
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def _messages(self, system_prompt: str, user_prompt: str) -> list:
        if self._fold_system:
            return [{"role": "user",
                     "content": f"{system_prompt}\n\n---\n\n{user_prompt}"}]
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def complete(self, system_prompt: str, user_prompt: str,
                 schema: dict = None) -> dict:
        """Single completion. Returns the raw text plus token logprobs.

        `schema` enables vLLM guided decoding (xgrammar), which makes an
        out-of-vocabulary verdict structurally impossible rather than merely
        detectable -- see verdict_schema() for why that matters and what it
        costs. Falls back to plain json_object mode if the server rejects the
        guided request, because an older or differently-built vLLM should
        degrade to the previous behaviour rather than fail the whole run.

        Every failure path raises LLMUnavailable rather than returning a
        sentinel, so a transport problem can never be silently scored as a
        low-confidence verdict.
        """
        def _build_attempts(frequency_penalty=FREQUENCY_PENALTY):
            base_kwargs = dict(
                model=self.model_name,
                messages=self._messages(system_prompt, user_prompt),
                temperature=TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
                frequency_penalty=frequency_penalty,
            )
            out = []
            if schema:
                out.append(dict(base_kwargs, response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "mollm_verdict", "schema": schema,
                                    "strict": True},
                }))
                # vLLM's own extension, accepted by some builds that reject the
                # OpenAI json_schema spelling.
                out.append(dict(base_kwargs, extra_body={"guided_json": schema}))
            out.append(dict(base_kwargs, response_format={"type": "json_object"}))
            return out

        def _run(attempts):
            for i, kwargs in enumerate(attempts):
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                    self.decoding_mode = (
                        "guided_json_schema" if i == 0 and schema else
                        "guided_json_extra_body" if i == 1 and schema else
                        "json_object_unguided"
                    )
                    return resp, None
                except Exception as exc:
                    last = exc
            return None, last

        response, last_exc = _run(_build_attempts())

        # A chat template that refuses the system role fails EVERY attempt
        # above, because the rejection happens during templating and so is
        # independent of the decoding mode. Retry once with the prompt folded
        # into the user turn, and remember the answer so later calls skip the
        # doomed attempts rather than paying three failed round-trips each.
        if response is None and not self._fold_system and _is_role_error(last_exc):
            self._fold_system = True
            response, last_exc = _run(_build_attempts())

        if response is None:
            raise LLMUnavailable(
                f"{self.model_name} at {self.base_url}: {last_exc}") from last_exc

        # 2026-08-13 (P4). One deterministic retry at a harder frequency
        # penalty when the first generation degenerated into a repetition loop.
        # Both outcomes are recorded: `degenerate_generation` reflects the
        # FINAL text (so a successful retry clears it), while
        # `degenerate_retried` records that a retry happened at all -- which is
        # the number that says how often the loop is still occurring even
        # though the pipeline now recovers from it. Without the second field,
        # a fully-effective retry would make the underlying problem look
        # solved rather than merely handled.
        degenerate, degenerate_detail = is_degenerate(
            response.choices[0].message.content or "",
            response.choices[0].finish_reason)
        degenerate_retried = False
        if degenerate:
            degenerate_retried = True
            retry_resp, _retry_exc = _run(
                _build_attempts(frequency_penalty=DEGENERATE_RETRY_PENALTY))
            if retry_resp is not None:
                retry_degenerate, retry_detail = is_degenerate(
                    retry_resp.choices[0].message.content or "",
                    retry_resp.choices[0].finish_reason)
                # Keep the retry only if it actually helped. A retry that
                # degenerated too is no better than the original and swapping
                # it in would only obscure which text produced the verdict.
                if not retry_degenerate:
                    response = retry_resp
                    degenerate, degenerate_detail = False, retry_detail

        choice = response.choices[0]
        tokens = []
        try:
            for item in choice.logprobs.content:
                # top_logprobs was already being REQUESTED (TOP_LOGPROBS=5,
                # above) but discarded here until 2026-08-11 -- captured now so
                # extract_candidate_alternatives() can read what probability
                # the model assigned to verdicts it did NOT choose, at no
                # extra cost: the alternates were already paid for on every
                # call, just thrown away.
                top = []
                try:
                    top = [{"token": t.token, "logprob": t.logprob}
                           for t in (item.top_logprobs or [])]
                except (AttributeError, TypeError):
                    top = []
                tokens.append({"token": item.token, "logprob": item.logprob,
                               "top_logprobs": top})
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
            # finish_reason == "length" means the model was still writing when
            # it hit MAX_OUTPUT_TOKENS, so the JSON is very likely unterminated.
            # Recorded explicitly because the downstream symptom (a parse
            # failure becoming INSUFFICIENT_EVIDENCE and routing to HITL) is
            # indistinguishable from genuine model uncertainty unless the cause
            # is carried alongside it. Observed in practice: these models pad
            # `reasoning` with restatements until the budget runs out.
            "truncated": choice.finish_reason == "length",
            # Whether the system prompt had to be folded into the user turn for
            # this endpoint. Both ensemble members must receive the same
            # instructions for their votes to be comparable; if this differs
            # across members, they were prompted differently and that is a
            # confound worth seeing rather than a detail worth hiding.
            "prompt_role_folded": self._fold_system,
            # Recorded per call: a run that silently fell back to unguided
            # decoding has different logprob calibration from a guided one, and
            # mixing the two in one calibration set would be invisible without
            # this.
            "decoding_mode": getattr(self, "decoding_mode", "unknown"),
            # 2026-08-13 (P4). True when the text below is STILL a repetition
            # loop after the retry. src/mollm_ensemble.py's combine()/route()
            # read this to keep a degenerate verdict out of the
            # model_disagreement safety rule -- a model that could not stop
            # talking did not disagree with anything.
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
# hitting the output cap (finish_reason == "length") without ever repeating
# the same 6 words twice. distinct_ngram_ratio scored 0.58 on that text
# (comfortably above DEGENERATE_DISTINCT_RATIO) because the quoted concept
# names change every cycle even though the SENTENCE SKELETON around them does
# not. Replacing quoted spans with a single placeholder before re-running
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
    small set forever, so the ratio collapses. See DEGENERATE_DISTINCT_RATIO
    for why this is measured instead of "share of the most common n-gram",
    which misses repeated multi-word sentences entirely.

    A SECOND pass runs the identical test on the text with quoted spans
    blanked out, catching a "conceptual drift" loop that reuses the same
    templated sentence around a DIFFERENT quoted name each cycle -- see
    _QUOTED_SPAN_RE above. Either pass flagging is sufficient.

    Combined with finish_reason == "length" -- the loop's other invariant,
    since it terminates only when the token budget runs out.

    WHY NOT JUST "does a sentence appear more than k times". Because the
    observed failures are not all sentence-shaped: the report records one case
    that degenerated into a multi-thousand-character run of digits inside
    cited_evidence.rule_id, which no sentence-splitter would catch. Word
    n-grams cover both shapes with one rule.

    WHY finish_reason IS REQUIRED RATHER THAN ADVISORY. Clinical text is
    genuinely repetitive -- medication lists, templated section headers, "no
    acute distress" repeated per system. Flagging on repetition alone would
    fire on legitimate output. Requiring that the model ALSO ran out of budget
    is what separates "this text repeats" from "this model could not stop".

    detail_dict is returned for persistence: the ratio and the offending
    n-gram make a flagged decision auditable rather than asking anyone to take
    the classifier's word for it.
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
        # Truncated: the point is to recognise the loop, not to store it again.
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
    were being requested via TOP_LOGPROBS but discarded until 2026-08-11) for
    alternative tokens. For each alternate, reconstructs what the full string
    would be with that token substituted in and keeps it ONLY if the
    reconstruction is itself a member of allowed_verdicts.

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


def build_clients() -> dict:
    """Constructs both ensemble members.

    Model names must match what each vLLM server was launched with (`--model`),
    which is what /v1/models reports; scripts/boot_infra.sh already probes both
    endpoints on boot.
    """
    return {
        "biomistral": LLMClient(model_name=DEFAULT_BIOMISTRAL, base_url=BIOMISTRAL_BASE_URL),
        "openbiollm": LLMClient(model_name=DEFAULT_OPENBIOLLM, base_url=OPENBIOLLM_BASE_URL),
    }
