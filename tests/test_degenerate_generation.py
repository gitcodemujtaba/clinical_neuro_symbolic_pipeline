"""
tests/test_degenerate_generation.py — isolated tests for the 2026-08-13 P4
degenerate-generation detector (src/llm_client.py is_degenerate) and its
effect on ensemble agreement (src/mollm_ensemble.py combine/route).

WHAT THESE PROTECT. The detector's whole job is to tell three things apart
that all look identical downstream today:

    (a) a repetition loop that ran out of budget      -> not a vote
    (b) legitimately repetitive clinical text          -> a vote
    (c) a genuine INSUFFICIENT_EVIDENCE verdict        -> a vote

Getting (b) wrong would silently discard real verdicts, which is a worse
failure than the bug being fixed -- clinical notes are full of templated
repetition. So the false-positive cases are tested at least as hard as the
true-positive ones.

Run:  python3 tests/test_degenerate_generation.py
"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.llm_client import (  # noqa: E402
    DEGENERATE_DISTINCT_RATIO, DEGENERATE_MIN_TOKENS, is_degenerate,
)
from src.mollm_ensemble import combine, route  # noqa: E402


def _model(name, verdict, conf=0.9, degenerate=False):
    return {"model": name, "verdict": verdict, "logprob_confidence": conf,
            "degenerate_generation": degenerate}


def run():
    ok, fail = 0, []

    def check(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # ==================================================================
    # is_degenerate()
    # ==================================================================

    # 1. THE OBSERVED FAILURE: one sentence repeated until the cap.
    loop = ("The entity is not clearly supported by the evidence provided here. " * 15)
    hit, detail = is_degenerate(loop, "length")
    check("repetition loop at cap is degenerate", hit is True)
    check("reason recorded", detail["reason"] == "repetition_loop")
    check("distinct-n collapsed", detail["distinct_ngram_ratio"] < 0.35)

    # 2. THE SAME TEXT that terminated normally is NOT flagged. The loop's
    #    defining property is that it could not stop, not that it repeated.
    hit, detail = is_degenerate(loop, "stop")
    check("repetition without hitting cap is not degenerate", hit is False)
    check("but it IS recorded as repetitive",
          detail["reason"] == "repetitive_but_terminated_normally")

    # 3. FALSE-POSITIVE GUARD: long, varied clinical reasoning that hit the
    #    cap. Truncation alone must never be enough to discard a verdict.
    varied = " ".join(
        f"Finding {i} was reviewed against guideline rule R{i} and the mapping "
        f"to concept {i * 7} appears consistent with the documented context."
        for i in range(30))
    hit, _d = is_degenerate(varied, "length")
    check("varied long text at cap is not degenerate", hit is False)

    # 4. FALSE-POSITIVE GUARD: genuinely repetitive clinical boilerplate that
    #    hit the cap but is not a single repeated n-gram.
    boiler = " ".join(
        f"{system}: within normal limits on examination today, no acute findings."
        for system in ["Cardiac", "Pulmonary", "Abdominal", "Neurologic",
                       "Skin", "Extremities", "HEENT", "Psychiatric",
                       "Lymphatic", "Vascular", "Renal", "Hepatic"])
    hit, _d = is_degenerate(boiler, "length")
    check("varied boilerplate at cap is not degenerate", hit is False)

    # 5. The digit-string case the report recorded (a multi-thousand-character
    #    run inside cited_evidence.rule_id) -- not sentence-shaped, still caught.
    digits = ("1234567890 " * 200)
    hit, _d = is_degenerate(digits, "length")
    check("repeated digit run is degenerate", hit is True)

    # 6. Short outputs are never judged: repetition is not diagnostic there,
    #    and a terse correct answer must not be discarded.
    short = " ".join(["word"] * (DEGENERATE_MIN_TOKENS - 5))
    hit, detail = is_degenerate(short, "length")
    check("short output not judged", hit is False)
    check("short output says why", detail["reason"] == "too_short_to_judge")

    # 7. Degenerate inputs must not raise.
    for bad in ("", None):
        hit, _d = is_degenerate(bad, "length")
        check(f"safe on {bad!r}", hit is False)

    # 7b. TEMPLATE DRIFT LOOP (2026-08-13, verification follow-up
    #     docs/2026-08-13_Implementation_Verification.md): a real BioMistral
    #     transcript for entity MCHC-34 that cycles through a fixed sentence
    #     skeleton ("The 'X' is a Y of the 'Z' concept.") around ~20 DIFFERENT
    #     quoted ontology terms, hitting the output cap. distinct_ngram_ratio
    #     alone scores this 0.58 (well above the 0.35 flag line) because the
    #     quoted names change every cycle -- the literal check missed this
    #     entirely before the template-normalized second pass was added.
    terms = ["Knowledge representation system", "Formal knowledge structure",
             "Conceptual hierarchy node", "Abstract reasoning framework",
             "Biological classification schema", "Organism taxonomy level",
             "Cellular component category", "Blood vessel anatomical region",
             "Erythrocyte membrane structure", "Hematologic tissue type",
             "Circulatory system component", "Vascular endothelium layer",
             "Bodily fluid compartment", "Living organism subsystem",
             "Scientific subject domain", "Knowledge management discipline",
             "Semantic ontology branch", "Information taxonomy class",
             "Cognitive science field", "Epistemological framework unit"]
    template_drift = " ".join(
        f"The '{terms[i % len(terms)]}' is a part of the "
        f"'{terms[(i + 1) % len(terms)]}' concept."
        for i in range(50))
    hit, detail = is_degenerate(template_drift, "length")
    check("template drift loop at cap is degenerate", hit is True)
    check("reason distinguishes template from literal loop",
          detail["reason"] == "template_repetition_loop")
    check("literal distinct-ratio alone would have missed it",
          detail["distinct_ngram_ratio"] >= DEGENERATE_DISTINCT_RATIO)
    check("template-normalized ratio collapsed",
          detail["template_distinct_ngram_ratio"] < 0.35)

    # 7c. FALSE-POSITIVE GUARD: varied prose that happens to use quotation
    #     marks (e.g. quoting different span text per sentence) must not be
    #     flagged just because normalization collapses the quoted parts --
    #     the SKELETON around the quotes must also repeat, not merely the
    #     presence of quotes.
    varied_quotes = " ".join(
        f"Sentence {i} discusses '{'term ' + str(i) * 2}' in a context that "
        f"varies its own structure each time, referencing finding {i} and "
        f"guideline section {i + 3} without repeating the same wording."
        for i in range(30))
    hit, detail = is_degenerate(varied_quotes, "length")
    check("varied prose with quotes is not degenerate", hit is False)

    # ==================================================================
    # combine() -- degenerate verdicts are not votes
    # ==================================================================

    # 8. THE CASE FROM THE REPORT. OpenBioLLM answers SUPPORTED confidently;
    #    BioMistral emits a repetition loop that defaults to
    #    INSUFFICIENT_EVIDENCE. Before P4 this was model_disagreement -> HITL.
    results = [_model("llama3.2:3b", "SUPPORTED", 0.9),
               _model("qwen2.5:3b", "INSUFFICIENT_EVIDENCE", 0.4, degenerate=True)]
    ens = combine(results)
    check("degenerate voter excluded -> agreement", ens["ensemble_agreement"] is True)
    check("exclusion counted", ens["n_degenerate_models"] == 1)
    check("exclusion names the model", ens["degenerate_models"] == ["qwen2.5:3b"])
    check("only one model scored", ens["n_models_scored"] == 1)
    check("basis says a voter was excluded",
          "degenerate_excluded" in ens["confidence_basis"])
    check("confidence is the surviving model's, not an average",
          abs(ens["composite_confidence"] - 0.9) < 1e-6)

    # 9. GENUINE disagreement is untouched -- the safety rule must still fire.
    results = [_model("llama3.2:3b", "SUPPORTED"), _model("qwen2.5:3b", "CONTRADICTED")]
    ens = combine(results)
    check("genuine disagreement still disagrees", ens["ensemble_agreement"] is False)
    check("no degeneracy claimed", ens["n_degenerate_models"] == 0)
    r = route(ens, {"citation_verified": True}, results)
    check("genuine disagreement still routes HITL",
          r["queue_reason"] == "model_disagreement")

    # 10. ALL models degenerate: not agreement-by-elimination, its own route.
    results = [_model("llama3.2:3b", "INSUFFICIENT_EVIDENCE", 0.3, degenerate=True),
               _model("qwen2.5:3b", "INSUFFICIENT_EVIDENCE", 0.3, degenerate=True)]
    ens = combine(results)
    check("all-degenerate flagged", ens["all_models_degenerate"] is True)
    r = route(ens, {"citation_verified": True}, results)
    check("all-degenerate routes HITL", r["mollm_routing_decision"] == "HITL_REQUIRED")
    check("all-degenerate has its own queue_reason",
          r["queue_reason"] == "degenerate_generation")
    check("all-degenerate is NOT reported as disagreement",
          r["queue_reason"] != "model_disagreement")

    # 11. Clean agreement is completely unaffected by any of this.
    results = [_model("llama3.2:3b", "SUPPORTED", 0.9), _model("qwen2.5:3b", "SUPPORTED", 0.9)]
    ens = combine(results)
    check("clean agreement unaffected", ens["ensemble_agreement"] is True)
    check("clean agreement has no exclusion note",
          "degenerate_excluded" not in ens["confidence_basis"])
    check("clean agreement counts both", ens["n_models_scored"] == 2)

    # 12. Models with no logprobs still report degeneracy fields -- the
    #     early-return path must not drop them.
    results = [{"model": "a", "verdict": "SUPPORTED", "logprob_confidence": None,
                "degenerate_generation": True},
               {"model": "b", "verdict": "SUPPORTED", "logprob_confidence": None}]
    ens = combine(results)
    check("no-logprob path keeps degeneracy fields",
          ens["n_degenerate_models"] == 1 and ens["composite_confidence"] is None)

    print(f"degenerate-generation tests: {ok} passed, {len(fail)} failed")
    for f in fail:
        print(f"  FAILED: {f}")
    return not fail


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
