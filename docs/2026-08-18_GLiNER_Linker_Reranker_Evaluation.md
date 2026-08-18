# 2026-08-18: GLiNER-Linker as a Stage 2b Reranking Signal — Evaluation and Result

## Motivation

This session root-caused a real, recurring class of Stage 2b linking error: SNOMED
sometimes models the identical clinical idea twice, once as a `Condition/Disorder`
concept and once as an `Observation/Morphologic-Abnormality` concept (e.g. `wound
dehiscence` → 225553008 vs. 410723003). Corpus-wide sizing (109-note test corpus) found
41 wrong-concept cases sharing this exact shape, 11 of 14 unique pairs favoring the
Observation/Morph-Abnormality sense. A fix was built and validated this session: widen
`GLINER_LABEL_TO_DOMAIN["Condition"]` so both senses reach the candidate pool, then use
an exhaustive per-candidate MoLLM evaluation with a comparative tiebreak (a corpus-measured
prior, "prefer Observation/Morph-Abnormality unless the note gives specific contrary
evidence") when 2+ candidates are independently accepted. Measured result: 7/7 AUTO-tier
precision on the 11 known cases (see `src/mollm_tier_gate.py`'s
`CONDITION_VS_OBSERVATION_PRIOR`/`_condition_vs_observation_duplicate`).

Given that mechanism's cost — the exhaustive evaluation measurably increased per-entity
Stage 3 latency (~3s → ~5.3s, confirmed against a live batch run) — this investigation
asked whether a purpose-built, much cheaper *reranking* model could do the same
disambiguation job earlier in the pipeline (Stage 2b, right after retrieval), either
replacing or triaging ahead of the expensive LLM tiebreak.

## Candidate: Knowledgator's GLiNER-Linker / GLiNKER framework

[Knowledgator](https://huggingface.co/knowledgator) publishes a 3-layer entity-linking
framework ("GLiNKER"): L1 extraction (GLiNER), L2 candidate retrieval, L3 disambiguation
via a bi-encoder. A 4th component, L4 (`gliner-linker-rerank-v1.0`, an `ettin-encoder-68m`
cross-encoder), is offered specifically for reranking. Confirmed directly against
Knowledgator's own official collection page
(https://huggingface.co/collections/knowledgator/gliner-relex — and the sibling
`gliner-biomed`/`gliner-linker` collections) rather than assumed:

| Model | Backbone | Role |
|---|---|---|
| `gliner-linker-base-v1.0` | DeBERTa-v3-base | Bi-encoder disambiguation, balanced |
| `gliner-linker-large-v1.0` | DeBERTa-v3-large | Bi-encoder disambiguation, max accuracy |
| `gliner-linker-rerank-v1.0` | ettin-encoder-68m | Cross-encoder reranking |

All three are **general-domain** — no biomedical-specific variant exists in Knowledgator's
own official collection. The only "biomedical" GLiNER-relex model found anywhere
(`grsilva/gliner-relex-biomedical-lora`) has zero downloads/likes and no disclosed training
data or evaluation — not a credible alternative.

## Environment compatibility (real, structural, worth documenting)

Two independent, hard blockers, confirmed empirically before attempting any workaround:

1. **The `glinker` package requires Python ≥3.10.** This project's main pipeline runs
   Python 3.9.25 (a deliberate pin — scispaCy/medspacy require `spacy<3.8`, and several
   other dependencies are version-locked against it). Confirmed via both `pip install
   glinker` and `pip install git+https://github.com/Knowledgator/GLinker.git` — both fail
   identically with `Package 'glinker' requires a different Python: 3.9.25 not in '>=3.10'`.
2. **All three checkpoints were saved in a `transformers==5.0.0` config/tokenizer format**
   (`tokenizer_class: "TokenizersBackend"`), incompatible with this project's installed
   `transformers==4.57.6`. Attempting to load `luqh/ClinicalT5-base`-style flax conversion
   or a bare `transformers.AutoModel` load both fail or silently drop critical weights.

**Resolution used**: an isolated Python 3.11 venv (`.venv_gliner_linker`, not committed —
recreate via the steps below), completely separate from the main pipeline's environment,
called via subprocess + JSON files — the same architectural pattern already used for Ollama
(a separately-managed model service, not something embedded in-process). This is the
correct way to use packages with incompatible dependency requirements without risking the
main pipeline's carefully-pinned environment; it should NOT be worked around by upgrading
the main environment's Python or `transformers` version, which would risk breaking multiple
other hard-won compatibility pins.

**To recreate**, from the project root:
```
python3.11 -m venv .venv_gliner_linker
source .venv_gliner_linker/bin/activate
pip install "git+https://github.com/Knowledgator/GLinker.git"
pip install "gliner==0.2.25"   # REQUIRED pin -- see below
```

**A second, subtler compatibility bug**: `glinker==0.1.1` (latest on PyPI at time of
writing) has no upper pin on its own `gliner` dependency, so a plain install pulls in
`gliner==0.2.28` (latest). That combination is broken — `glinker`'s own `GLinkerModel.forward()`
crashes with `AttributeError: 'str' object has no attribute 'shape'` deep inside its
scorer, because a version-drifted internal API returns a string where `glinker`'s glue code
expects a tensor. Pinning `gliner==0.2.25` (the release closest in time to when these
checkpoints were published, per PyPI release history) fixes this. Confirmed via direct
before/after testing, not guessed.

## API usage (both variants confirmed working)

`L3Component`/`L3Config` (bi-encoder, `base`/`large`) and `L4Component`/`L4Config`
(cross-encoder, `rerank`) are separate classes with slightly different call shapes — the
model card's own quick-start example conflates them, which caused an initial failed attempt
(`config.labels_encoder` was `None` for `rerank`, because it isn't a bi-encoder at all).
Working calls, `input_spans` given in true character offsets over `text` so the model scores
an entity's *own* known span rather than re-detecting spans itself:

```python
# L3 (base/large)
from glinker.l3 import L3Component, L3Config
comp = L3Component(L3Config(model_name="knowledgator/gliner-linker-large-v1.0",
                             device="cuda", threshold=0.0))
comp.predict_entities(text=text, labels=candidate_strings,
                       input_spans=[{"start": s, "end": e}],
                       span_label_indices=[list(range(len(candidate_strings)))])

# L4 (rerank) -- note input_spans is List[List[dict]], not List[dict]
from glinker.l4 import L4Component, L4Config
comp = L4Component(L4Config(model_name="knowledgator/gliner-linker-rerank-v1.0",
                             device="cuda", threshold=0.0))
comp.predict_entities(text=text, labels=candidate_strings,
                       input_spans=[[{"start": s, "end": e}]])
```

Reusable batch script: `dormant/scripts/gliner_linker_score.py` (run inside
`.venv_gliner_linker`, takes a JSON request file, writes a JSON result file — designed to be
called via subprocess from the main Python 3.9 pipeline).

## Validation result: negative

First smoke test (`wound dehiscence`, one case) looked very promising — `large` scored
0.96 for the correct (Observation/Morph-Abnormality) candidate, matching the exact case our
own MoLLM ensemble had split 2-1 the wrong way. That result was **not representative**.
Scored properly against all 10 gradable cases from the same known-duplicate-concept
population used to validate the MoLLM tiebreak fix (real `local_context` pulled from the
database, not synthetic snippets):

| Model | Correct | Cases gotten wrong |
|---|---|---|
| `gliner-linker-large-v1.0` (bi-encoder) | **4/10** | rhabdomyolysis, wound, cyst, Metastatic Renal Cell Cancer, compression fractures, mantle cell lymphoma |
| `gliner-linker-rerank-v1.0` (cross-encoder) | **5/10** | rhabdomyolysis, wound, Metastatic Renal Cell Cancer, compression fractures, Osteopenia |
| **Our MoLLM tiebreak (strengthened prior)**, same population | **7/7** | none |

Both GLiNER-Linker variants land near chance. Given 9 of the 10 cases are drawn from the
corpus's own 11/14-majority direction (gold favors Observation/Morph-Abnormality), even a
naive "always guess the majority direction" heuristic would likely outperform both.

**Methodological caveat, stated honestly**: candidate descriptions were phrased by hand
(`"{name}: a disorder, a diagnosed medical condition"` / `"{name}: a morphologic
abnormality, a descriptive clinical finding"`). A different phrasing might change the
result. This was not explored further — iterating on prompt/description phrasing until a
number improves is exactly the kind of result-hunting this project's evaluation discipline
avoids elsewhere, and there's no principled reason to expect a different phrasing to close
a 3-point gap against a 100%-precision existing mechanism.

## Conclusion and disposition

**Do not integrate.** The infrastructure (isolated venv, working subprocess-callable
scorer, confirmed API usage for both architectures) is real and reusable, but the model
itself — general-domain, no biomedical fine-tuning — does not perform competitively on this
specific SNOMED disambiguation task. This is a genuine negative result reached through
proper validation (not a setup/calling-convention failure), consistent with this session's
broader finding that general-domain models systematically underperform on fine-grained
SNOMED ontological distinctions without domain adaptation.

`scripts/gliner_linker_score.py` moved to `dormant/scripts/` (see that folder's `README.md`
for the archiving convention) rather than deleted, in case a different use case or an
improved candidate-phrasing approach is worth revisiting later. `.venv_gliner_linker` was
removed (5.2GB, trivially reproducible from the steps above) rather than kept, since venvs
have no independent value once documented.

## For the presentation

- **What was tried**: a fast, purpose-built entity-disambiguation model as an alternative to
  expensive LLM-based tiebreaking for the SNOMED-duplicate-concept problem.
- **What we learned**: environment isolation (separate venv, subprocess interface) is a
  viable, low-risk pattern for evaluating external models without touching the main
  pipeline's dependency pins — worth reusing for future experiments.
- **The actual finding**: general-domain entity-linking models, even purpose-built
  disambiguation/reranking architectures, don't reliably capture SNOMED's specific
  ontological distinctions (Condition/Disorder vs. Observation/Morphologic-Abnormality)
  without domain adaptation — reinforcing that our MoLLM-ensemble-plus-corpus-measured-prior
  approach, despite its latency cost, is currently the better-performing mechanism for this
  problem, not a stopgap waiting to be replaced.
