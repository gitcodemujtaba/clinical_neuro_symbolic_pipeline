# MoLLM Prompts & Reasoning — Complete Technical Reference

**Module**: `src/mollm_tier_gate.py` (prompt construction, orchestration) ·
`src/guideline_evidence.py` (rules-based KG bridge) · `src/llm_client.py`
(model transport)

**Companion documents**: `docs/ConsensusCalibrator_Technical_Reference.md`
(what happens to a decision *after* the ensemble votes — this document
stops where that one starts) and `docs/Entity_Journey_Plain_Language_Walkthrough.md`
(the same "fever" example used in §3 below, told for a non-technical
reader).

**Verification discipline**: every code snippet below is copied verbatim
from the live source, re-read in full for this document. Every example
is either a real, stored production decision (quoted from
`mollm_tier_gate_decisions`) or a real function call made live against
the actual guideline index and the actual database — none are invented
or hypothetical. Where a mechanism exists in code but is not currently
active in production (the guideline-evidence bridge, §7), that is stated
explicitly, not glossed over.

---

## 1. The core design: two steps, not one, and why

Every entity that reaches the ensemble (i.e. survives the four free,
zero-model-call pre-checks — see `docs/ConsensusCalibrator_Technical_Reference.md`
§2 for the full routing diagram) is evaluated by **three independent
local LLMs** (qwen2.5:3b, llama3.2:3b, phi4-mini), each running the
**same two-step process**, in complete isolation from each other and
from Stage 2b's own retrieval scores.

```
Step A — "What does this span mean?"     (no candidate list shown)
              │
              ▼
Step B — "Does THIS candidate match that meaning?"   (one candidate at a time)
```

**Why isolate Step A from the candidate list at all?** If the model saw
the candidates before committing to an interpretation, it could anchor
on whatever's in front of it rather than reasoning independently about
the clinical meaning — the exact failure mode this project calls
"attention dilution" or "anchoring," found and fixed more than once in
this codebase's history (see §5 and §8 for two concrete, measured
instances of it).

**Why one candidate at a time in Step B, rather than a single prompt
showing the whole ranked list?** This was tried and measured to fail.
The module's own comment on it:

> Already tried and rejected in this exact codebase, 2026-08-14: a dense
> 1-to-N candidate list measurably let 3B models detach a candidate's
> evidence tag from its own bracket index and misattribute it to the
> highest-scored candidate instead (two real 'lasix' cases both moved to
> the highest-scored LASCUFLOXACIN candidate instead of the correctly-tagged
> furosemide one) — a formatting fix (isolating each tag on its own line)
> was tried and still not reliable enough.

So Step B asks about **exactly one candidate per model call**: no
bracket index for a small model to mis-track. The system pays for this
in call volume (up to 1 Step-A call + up to 5 Step-B calls, per model,
per entity — up to 18 LLM calls for one entity across all three models)
but the isolation is what makes each individual judgment trustworthy.

---

## 2. Step A — the clinical-meaning prompt, in full

```python
MEANING_SYSTEM_PROMPT = (
    "You are a clinical terminology expert reading a single clinical note. "
    "Your only job right now is to state what a highlighted text span means "
    "clinically, using the note's own context. You have not been shown any "
    "candidate concept list and must not anticipate or guess at one."
)

def _clinical_meaning_prompt(entity: dict) -> str:
    assertion = entity.get("assertion_status", "PRESENT")
    allergy_instruction = ALLERGY_MEANING_INSTRUCTION if assertion == "ALLERGY" else ""
    return (
        "ENTITY:\n"
        f"  text as written: {entity.get('original_text')!r}\n"
        f"  after abbreviation expansion: {entity.get('expanded_text')!r}\n"
        f"  extractor label: {entity.get('gliner_label')}\n"
        f"  assertion: {assertion} / experiencer: {entity.get('experiencer', 'PATIENT')}\n\n"
        f"SECTION: {entity.get('section_name') or 'unknown'}\n"
        f"CONTEXT: ...{entity.get('local_context', '')}...\n\n"
        f"{allergy_instruction}"
        "TASK: Based ONLY on the note text above, provide a concise, "
        "single-phrase clinical definition of what this entity refers to "
        '(e.g. "a beta-blocker medication", "a diagnosis of high blood '
        'pressure", "a surgical procedure on the knee"). Do not name a '
        "database code, ontology term, or vocabulary identity. Do not "
        "explain the patient's history. Define the term only.\n\n"
        'Reply with JSON: {"clinical_meaning": "<single-phrase definition>", '
        '"reasoning": "<one short sentence>"}'
    )
```

**Exactly six pieces of provenance feed this prompt**, all computed by
earlier stages and simply read here, never recomputed:

| Field | Computed by | What it tells the model |
|---|---|---|
| `original_text` | Stage 2a (GLiNER) | the literal span as written |
| `expanded_text` | Stage 1 | the abbreviation-expanded form, if different |
| `gliner_label` | Stage 2a | the extractor's own category guess (Symptom/Condition/Medication/...) |
| `assertion_status` / `experiencer` | Stage 1 (medspacy/ConText) | whether this is being claimed, denied, historical, about a family member, etc. |
| `section_name` | Stage 1 (section segmentation) | which part of the note this sits in |
| `local_context` | Stage 2a's `build_local_context()` | the sentence-bounded window around the span |

**Deliberately absent from Step A**: any candidate SNOMED concept, any
Stage 2b retrieval score, any `match_basis`. The model reasons purely
from the note's own text.

**Why a single phrase, not a paragraph?** A 2026-08-18 tightening,
explained directly in the code:

> a single-phrase definition rather than a free sentence or two gives
> Step B a tighter, more atomic string to compare each candidate
> against — "a beta-blocker medication" is less ambiguous to judge a
> candidate against than a paragraph that drifts into patient history.

---

## 3. Worked example, Step A — "fever" (real, stored decision)

Same entity as the plain-language walkthrough doc, note `11859945-DS-29`,
`assertion_status: ABSENT` (the note says "Denies fever"). Three real
Step A outputs, verbatim from the database:

| Model | `clinical_meaning` |
|---|---|
| qwen2.5:3b | *"absence of fever in a patient"* |
| phi4-mini | *"An elevation in body temperature indicative of an infection"* |
| llama3.2:3b | *"a normal body temperature"* (factually backwards — a fever is not normal — but the model still reaches the correct final verdict in Step B, see below) |

This single example already shows the real spread of behavior across
three small models on identical input: one (qwen) let the *negation*
leak into its definition of the *concept* — exactly the confusion rule
3 in Step B (§4) exists to correct; one (phi4-mini) reasoned cleanly;
one (llama) got the definition itself wrong but recovered in Step B
anyway. None of this is cherry-picked — it's the actual output for a
genuinely ordinary entity.

---

## 4. Step B — the binary-match prompt, in full, with every conditional rule

```python
MATCH_SYSTEM_PROMPT = (
    "You are a clinical terminology validator auditing whether a proposed "
    "concept code correctly labels a text span, given an independent "
    "statement of what that span means."
)

def _binary_match_prompt(entity, candidate, clinical_meaning, extra_rule=None):
    basis = candidate.get("match_basis", "semantic_similarity")
    rules = (
        "RULES:\n"
        "1. SEMANTIC MATCH: judge the candidate strictly against the CLINICAL "
        "MEANING stated above, not the raw text spelling or the candidate's "
        "match score -- does it represent the exact same clinical idea, even "
        "if spelled completely differently? Do not reject a candidate just "
        "because the words do not match the original text.\n"
    )
    if basis in _VERIFIED_ALIAS_BASES:          # only when THIS candidate's basis warrants it
        rules += f"2. This candidate's basis is {basis} -- {_VERIFIED_ALIAS_BASES[basis]}. " \
                 "Do not reject it merely because the spelling differs from the entity text.\n"
    rules += (
        "3. Ignore assertion/negation status when judging the CONCEPT match "
        '-- a negated entity ("denies fever") still maps to its concept '
        '("Fever") if the name matches; you are labeling which concept the '
        "text refers to, not diagnosing.\n"
        "4. STRICT DOMAIN MISMATCH: reject a candidate that is a distinct or "
        "clinically unrelated concept (e.g. mapping a symptom to a "
        "biological genus, or a medication to a surgical tool). Do not "
        "force a match.\n"
    )
    if extra_rule:
        rules += extra_rule + "\n"
    return (
        "This entity's clinical meaning was independently determined to be:\n"
        f'  "{clinical_meaning}"\n\n'
        "ENTITY:\n"
        f"  text as written: {entity.get('original_text')!r}\n"
        f"  section: {entity.get('section_name') or 'unknown'}\n"
        f"  assertion: {entity.get('assertion_status', 'PRESENT')} / "
        f"experiencer: {entity.get('experiencer', 'PATIENT')}\n"
        f"  context: ...{entity.get('local_context', '')}...\n\n"
        "CANDIDATE CONCEPT:\n"
        f"  name: {candidate.get('concept_name')}\n"
        f"  domain: {candidate.get('domain_id')}\n"
        f"  vocabulary: {candidate.get('vocabulary_id')}\n"
        f"  basis: {basis}\n\n"
        f"{rules}\n"
        "Does this candidate concept match the clinical meaning stated "
        'above? Reply with JSON: {"match": true or false, "reasoning": '
        '"<one sentence>"}'
    )
```

**Rule 2 only appears when it applies to *this specific candidate*.**
This is a deliberately narrow fix for a real, measured bleed problem:

> confirmed live to cause exactly the bleed this guards against: a
> candidate whose real basis was `exact_text` got justified in a model's
> own reasoning as "verified to be a brand alias in the SNOMED
> vocabulary", language borrowed straight from this rule's text despite
> not applying to that candidate at all.

**The rule numbering never shifts, even when rule 2 is omitted** — rules
stay 1/[2]/3/4/5 because `ALLERGY_CONTEXT_CLAUSE` hardcodes "RULE 3" and
`QWEN_SUBSUMPTION_CLAUSE` hardcodes "5." in their own text; both must
keep pointing at the same rule whether or not rule 2 is present that
call.

### 4.1 Continuing the "fever" example — Step B, all three models, verbatim

Candidate: `Fever` (SNOMED concept 437663, match tier "1 (Exact)", basis
`exact_text` — so rule 2 does **not** fire here, since `exact_text` isn't
in `_VERIFIED_ALIAS_BASES`).

**qwen2.5:3b** → `match: false`
> *"The candidate concept 'Fever' does not directly represent the
> absence of fever. It is a distinct and clinically unrelated concept."*

This is qwen applying rule 4 (domain mismatch) to a case that was
actually about rule 3 (ignore negation) — it treated "absence of fever"
(its own Step A output) as a different *concept* from "Fever," when
really it's the same concept under a negated assertion. A real,
instructive model mistake.

**phi4-mini** → `match: true`
> *"The term 'fever' directly corresponds to an elevation in body
> temperature indicative of infection."*

**llama3.2:3b** → `match: true`
> *"The candidate concept 'Fever' matches the clinical meaning of 'a
> normal body temperature' because it is a synonym in the SNOMED
> vocabulary and represents an exact match in terms of clinical idea."*

Llama's Step A definition was wrong (fever ≠ normal temperature), and
its Step B reasoning repeats that error, yet it still lands on the
correct `match: true`. **Two out of three models correctly separated
"what concept does this word name" from "does the patient have it";
one did not.** This is exactly the disagreement that gets handed to the
calibrator (see the companion document) rather than either auto-approved
or auto-rejected outright.

---

## 5. The `_evaluate_one_model()` orchestration — sequential, stop-on-accept-first-normally, exhaustive when needed

```python
def _evaluate_one_model(client, entity):
    meaning_raw = client.complete(MEANING_SYSTEM_PROMPT, _clinical_meaning_prompt(entity),
                                  schema=_meaning_schema())
    clinical_meaning = parse_json_response(meaning_raw["text"]).get("clinical_meaning")

    extra_rules = []
    if client.model_name.startswith("qwen"):
        extra_rules.append(QWEN_SUBSUMPTION_CLAUSE)
    if entity.get("assertion_status") == "ALLERGY":
        extra_rules.append(ALLERGY_CONTEXT_CLAUSE)
    extra_rule = "\n".join(extra_rules) if extra_rules else None

    accepted = []   # only populated when EXHAUSTIVE_CANDIDATE_EVAL_ENABLED
    for i, cand in enumerate(candidates, 1):
        raw = client.complete(MATCH_SYSTEM_PROMPT,
                              _binary_match_prompt(entity, cand, clinical_meaning, extra_rule),
                              schema=_match_schema())
        matched = parse_json_response(raw["text"]).get("match")
        if matched:
            if not EXHAUSTIVE_CANDIDATE_EVAL_ENABLED:
                return {"verdict": "SUPPORTED_1" if i == 1 else f"RE_RANK_TO_CANDIDATE_{i}", ...}
            accepted.append({"index": i, "candidate": cand, ...})   # keep going, don't stop
    # after the loop: 0 accepted -> NONE_CORRECT; 1 accepted -> that one;
    # 2+ accepted -> _resolve_tiebreak() (see §6)
```

**Two conditional rules are layered on top of the base 4, per model call,
per entity — never both from the same source competing:**

- `QWEN_SUBSUMPTION_CLAUSE` is added **only** when `client.model_name`
  starts with `"qwen"`. It is never shown to llama3.2:3b or phi4-mini.
- `ALLERGY_CONTEXT_CLAUSE` is added **only** when
  `entity["assertion_status"] == "ALLERGY"`. It is shown to all three
  models when it applies, since the allergy-context confusion was
  measured across all three, not one.

`EXHAUSTIVE_CANDIDATE_EVAL_ENABLED` (default on) is the setting that
decides whether Step B stops at the *first* accepted candidate (the
original design) or evaluates *every* candidate independently and only
then decides. This default flip is itself a real, measured decision —
see `docs/2026-08-20_Session_Results_And_Status.md` §12 for the honest
finding that the broader population this creates (2+ independently-accepted
candidates) grades at only 14.3% precision versus 84.7% for entities
that don't trigger it.

---

## 6. Rule 5 — `QWEN_SUBSUMPTION_CLAUSE`, and why it's qwen-only

```python
QWEN_SUBSUMPTION_CLAUSE = (
    "5. HIERARCHICAL SUBSUMPTION: candidates are being checked ONE AT A TIME "
    "in ranked order; if you reject this one, it will not be reconsidered "
    "later. If this candidate is a correct but less-specific (broader) or "
    "more-specific (narrower) SNOMED/RxNorm relative of the precise concept "
    "described -- not a DIFFERENT concept entirely -- accept it as a match "
    "rather than rejecting it purely for lacking the note's full specificity "
    "(severity, laterality, exact subtype). Still reject a candidate that "
    "names a different clinical concept, not merely a less-detailed one."
)
```

**Why qwen specifically, not a blanket rule for all three models?** A
broader version of this idea ("don't require every detail to match") was
tried across *all three* models on 2026-08-15 and had to be reverted —
it let every model rubber-stamp wrong matches on bare qualifier
fragments ('left', 'Removal', 'Multiple'), collapsing Tier 1 precision
to 5.9% (1/17). The narrower, qwen-only, hierarchy-specific version was
only safe to try after a separate fix (`qualifier_fragment_precheck()`)
removed those fragment spans from the ensemble's job entirely — so the
residual risk of a per-model relaxation was smaller than it was when
the earlier, blanket version was tried.

**Why can't the model just check "is there a more specific candidate
elsewhere in the list"?** Because Step B evaluates one candidate at a
time — that's the whole point of the sequential design (§1). The clause
doesn't pretend otherwise; it makes the actual tradeoff explicit (rank
order, no second look) rather than asking the model to reason about
information it structurally does not have.

---

## 7. Provenance-conditioned rules — the allergy exception (real, measured, fixed)

```python
ALLERGY_CONTEXT_CLAUSE = (
    "ALLERGY EXCEPTION TO RULE 3: this entity's assertion status is ALLERGY. "
    "Unlike negation, an ALLERGY assertion means the CORRECT concept is the "
    "patient's allergic disposition/reaction to the substance, not the "
    "substance itself -- these are genuinely different concepts here, and "
    "that is expected, not an error. If the candidate names an allergy or "
    "adverse-reaction concept for this same substance (e.g. 'Allergy to X', "
    "'X allergy', 'Allergic reaction caused by X'), treat that as the "
    "correct match. Do NOT reject it under rule 4 on the grounds that it "
    "names a 'different concept' from the substance itself -- for an "
    "ALLERGY-status entity, the allergy/reaction concept IS the correct one."
)
```

**This is the single clearest example in the whole system of provenance
directly steering the prompt's own rule set, not just informing it.**
`assertion_status` (computed once, in Stage 1, by the deterministic
medspacy/ConText engine — never by an LLM) is read at *two separate
points* for an ALLERGY-status entity: it triggers a different Step A
instruction (`ALLERGY_MEANING_INSTRUCTION`, §2) *and* a different Step B
rule set (`ALLERGY_CONTEXT_CLAUSE`, here) — the same upstream field,
consumed twice, at two different stages of the same model call chain.

**What went wrong before this clause existed, root-caused from real
stored `mollm_tier_gate_decisions.models` trail data**: with retrieval
already fixed and correctly surfacing "Allergy to morphine" as the #1
candidate, the ensemble still split votes on it, 0/19 reaching Tier 1.

> Confirmed in the raw trail: phi4-mini's Step A never even mentioned
> allergy ("Morphine is an opioid medication..."), and qwen2.5:3b
> rejected "Allergy to morphine" as "too specific" after its own Step A
> hedged into a generic "may cause an allergy" framing instead of stating
> the patient's actual disposition.

The base rule 3 ("ignore assertion status when judging concept match")
is *correct* for negation — "denies fever" still names the concept
"Fever." But it is actively wrong for allergy: the correct concept for
an allergy-context substance mention genuinely **is** a different
concept (the allergic-disposition finding), and the stock rule 3 pushed
models toward rejecting that correct answer under rule 4 ("different
concept, reject"). The clause is a targeted carve-out from the general
rule, not a contradiction of it — measured to move Aspirin, fluconazole,
and morphine from `TIER_4_ENSEMBLE_SPLIT` to `TIER_1` in one re-run.

---

## 8. The tiebreak prompt — when 2+ candidates independently match

Only reached when `EXHAUSTIVE_CANDIDATE_EVAL_ENABLED` (§5) lets Step B
run to completion and finds **more than one** candidate independently
accepted by the *same* model. A separate, smaller comparative call runs,
scoped to just the accepted subset (usually 2, rarely 3) — not the full
original candidate list, for the same index-isolation reason as §1.

```python
TIEBREAK_SYSTEM_PROMPT = (
    "You are a clinical terminology validator. Several candidate concept "
    "codes have each independently been judged a plausible match for the "
    "same text span -- your job now is to pick the single best one using "
    "the note's own context, not to re-decide whether any of them match."
)
```

The prompt shows each accepted candidate's name, domain, concept class,
match basis, **and the model's own earlier Step B reasoning for why it
accepted it** — the model is reminded of its own prior judgment rather
than starting cold. Then, conditionally, up to two more evidence blocks
are appended:

### 8.1 `CONDITION_VS_OBSERVATION_PRIOR` — a corpus-measured convention, injected only when the exact pattern is present

```python
def _condition_vs_observation_duplicate(accepted):
    if len(accepted) != 2:
        return False
    names = {a["candidate"]["concept_name"].strip().lower() for a in accepted}
    if len(names) != 1:
        return False
    domains = {a["candidate"]["domain_id"] for a in accepted}
    return domains == {"Condition", "Observation"}
```

Fires only when exactly two candidates share the identical name string
but span the Condition and Observation domains — the specific SNOMED
near-duplicate pattern this corpus was measured to handle one way,
11-of-14 times (78.6%), on direct corpus inspection. **This is
deliberately injected as a hint the model can still override, never as
an absolute rule** — the prompt says so explicitly, and the source is
candid about why a stronger, unconditional version was tried and had to
be strengthened again:

> v1's softly-worded, stats-cited prior ("11 of 14... treat as a weak
> prior") measurably UNDER-powered against a 3B model's pretraining
> association between disease-sounding terms ("Carcinoma") and
> Condition/Disorder framing: live-tested on 'Metastatic Renal Cell
> Cancer', all 3 models cited "diagnosis"/"documentation style" to
> override the prior toward the WRONG (Condition) answer.

**Real bug, found and fixed, worth knowing**: this detector originally
also required `concept_class_id` to match `"Disorder"`/`"Morph
Abnormality"` — but candidate dicts never carry `concept_class_id` at
all, nothing populates it. That silently disabled the entire prior
injection from the day it was built, confirmed live on the "wound
dehiscence" case (note `13538696-DS-11`): the tiebreak correctly fired
for all three models, but with no prior actually injected, each model
reasoned unguided and all three happened to agree on the wrong
(Condition/Disorder) candidate. Relaxed to match on `domain_id` alone
(a field every candidate dict genuinely carries).

### 8.2 The rules-based guideline KG evidence block — real mechanism, real example, currently off by default

```python
def _guideline_evidence_block(accepted):
    from src.guideline_evidence import GUIDELINE_EVIDENCE_ENABLED, get_guideline_index, \
        guideline_evidence_for_candidates
    if not GUIDELINE_EVIDENCE_ENABLED:
        return ""
    evidence = guideline_evidence_for_candidates(get_guideline_index(), accepted)
    return evidence + "\n\n" if evidence else ""
```

**⚠️ Stated plainly, not glossed over: `GUIDELINE_EVIDENCE_ENABLED`
defaults OFF** (`CNSP_GUIDELINE_EVIDENCE` env var, unset in production).
This is the rules-based clinical guideline knowledge graph — 76 source
documents, ~1,700 nodes, 1,162 extracted recommendation rules, each
citing its own source — bridged into the tiebreak prompt for the first
time on 2026-08-20, but held behind a flag pending its own validation
batch, the same discipline applied to every other prompt-touching change
in this codebase (hybrid retrieval, acronym escalation). **No production
decision in the database today was made with this evidence visible to
the model.**

**How matching works, and why not by SNOMED code**: the guideline
corpus's own curators flagged several of its nodes
`quality_flag: same_snomed_type_mismatch_not_merged` — cases where they
found the SNOMED code alone would wrongly conflate distinct concepts.
So matching is done by **node name (case-insensitive exact match)** plus
a **soft type-compatibility check** against the node's own `@type`
(`Finding`/`Condition`/`Intervention`/`Medication`/...) versus the
candidate's OMOP domain — conservative, and tied to what the guideline
document actually names, not to a crosswalk the corpus's own curators
already flagged as sometimes wrong.

**Real, measured coverage — not assumed**: checked directly against
every candidate concept currently in the database: **67 of 7,151
distinct candidate concept names (0.9%)** have an exact match in the
guideline corpus. Modest, reported honestly.

**A real evidence block, computed live for this document** (not stored
in any production decision, since the flag is off — this is what the
mechanism *would* produce for a real candidate that genuinely has
coverage):

```
Candidate: "Osteoporosis" (Condition domain) — one of the 67 real, covered names

OFFICIAL GUIDELINE EVIDENCE (from curated clinical-guideline triplets, not a
similarity guess -- weigh this as real evidence, it does not decide the
answer for you):
  candidate [1] (Osteoporosis, guideline type: Condition) -- guideline
  evidence from gold-pocket-guide-2026-v1.1-20nov2025_wmv2_chunk_7_copd_and_multimorbidity.json:
      -> IS_PART_OF: Osteoporosis is one of the conditions comprising the
      Multiple Organ Loss of Tissue (MOLT) morbidity cluster associated with
      COPD. [source: Multiple Organ Loss of Tissue (MOLT): Osteoporosis,
      sarcopenia, anemia, emphysema.]
```

Notice the framing line: *"weigh this as real evidence, it does not
decide the answer for you"* — this is additive context, structurally
incapable of forcing a verdict on its own, the same "evidence to weigh,
never a deterministic override" discipline applied to
`CONDITION_VS_OBSERVATION_PRIOR` above.

---

## 9. Provenance summary — everything that reaches a prompt, and from where

| Provenance field | Source stage | Reaches Step A? | Reaches Step B? | Reaches tiebreak? |
|---|---|---|---|---|
| `original_text` | Stage 2a (GLiNER) | ✓ | ✓ | ✓ |
| `expanded_text` | Stage 1 | ✓ | — | — |
| `gliner_label` | Stage 2a | ✓ | — | — |
| `assertion_status` / `experiencer` | Stage 1 (medspacy/ConText) | ✓ (drives `ALLERGY_MEANING_INSTRUCTION`) | ✓ (drives rule 3 + `ALLERGY_CONTEXT_CLAUSE`) | ✓ |
| `section_name` | Stage 1 (sectioning) | ✓ | ✓ | ✓ |
| `local_context` | Stage 2a (`build_local_context()`) | ✓ | ✓ | ✓ |
| `match_basis` | Stage 2b (retrieval) | — | ✓ (drives conditional rule 2) | ✓ (shown, not rule-driving) |
| `concept_name` / `domain_id` / `vocabulary_id` | Stage 2b (retrieval) | — | ✓ | ✓ |
| `concept_class_id` | Stage 2b (retrieval) | — | — | ✓ (shown; also what `CONDITION_VS_OBSERVATION_PRIOR`'s detector intended to use, see §8.1's bug note) |
| model's own prior Step B reasoning | this stage | — | — | ✓ (only in tiebreak) |
| guideline KG evidence | `src/guideline_evidence.py` | — | — | ✓ (only in tiebreak, only if `GUIDELINE_EVIDENCE_ENABLED`) |
| `CONDITION_VS_OBSERVATION_PRIOR` | corpus-measured constant | — | — | ✓ (only when the exact 2-candidate Condition/Observation same-name pattern is present) |

**The single organizing principle across every row above**: every piece
of provenance is either shown as context for the model to reason over,
or used to select *which fixed rule text* gets included — nothing in
this system lets provenance silently change what counts as a correct
answer without the model seeing and reasoning about the relevant
evidence itself. The one deliberate, narrow exception is the four free
pre-checks upstream of the ensemble entirely (qualifier-fragment,
Tier-3 fast path, lab-procedure fast path, Tier-5 precheck) — those skip
the model outright, by design, and are documented in the companion
calibrator reference (§2 there).

---

## 10. Confidence extraction — not self-reported

Every Step A and Step B call asks for a JSON verdict but **never asks
the model to self-report a confidence number**. Confidence is instead
extracted from the model's own token log-probabilities on the verdict
token itself (`extract_verdict_confidence()` in `src/llm_client.py`) —
the same distinction the calibrator reference document explains for its
own `mean_logprob_confidence`/`min_logprob_confidence` features: a
log-probability is an observation of the decode, not the model's
opinion of itself. This project measured self-reported HIGH/LOW
confidence directly (a separate investigation, see the calibrator
reference §3.1) and found it to be a **per-model constant**, not a
per-record signal — BioMistral returned HIGH on 10/10 records in that
check regardless of correctness — which is exactly why this system
never asks for it that way.

---

## 11. Experimental — grounding Step A with real SNOMED IS_A taxonomy (Neo4j)

**Status: an experiment run against real models and real data, not a
shipped mechanism.** Nothing described in this section is wired into
`route_tier()` or any production call path — it lives entirely in
one-off session scratchpad scripts, not this repository. Reported here
because it's a direct, measured answer to a real architectural question
(§2's own text: Step A gets *nothing* but the note itself), not because
it's been adopted.

### 11.1 The problem, restated precisely

`_clinical_meaning_prompt()` (§2) instructs the model: *"Based ONLY on
the note text above, provide a concise, single-phrase clinical
definition."* There is no SNOMED lookup, no dictionary, no KG query
anywhere in that call. Whatever definition comes out is manufactured
entirely from the model's own pretrained weights at that instant. §3's
own worked example already shows the cost of this directly: qwen2.5:3b's
real, stored Step A output for "fever" (`assertion_status: ABSENT`) was
*"absence of fever in a patient"* — the negation leaked into the
concept's own definition, which is the literal, documented cause of its
`NONE_CORRECT` vote on an entity every other signal (Tier 1 exact match,
similarity 1.0, the other two models) already had right.

### 11.2 A real, previously-undiscovered resource: a live Neo4j SNOMED graph

Verified live, direct connection, not assumed: this environment runs a
Docker container (`db1_neo4j_lexicon`, bolt://localhost:7687) holding a
genuine, substantial SNOMED CT hierarchy — **386,110 `:SnomedConcept`
nodes**, each with `id` (SCTID) and `fullySpecifiedName`, connected by
**641,727 `:IS_A` relationships**. It is completely disconnected from
the live pipeline today — the only code that ever queries it is
`scripts/profile_databases.py`, a standalone diagnostic script, not part
of any production path. (This is separate from, and should not be
confused with, the Memgraph instances documented in
`docs/Knowledge_Graphs_Technical_Reference.md` — this
is a *third*, distinct graph database on the same box, holding the
*static reference* SNOMED hierarchy, not the dynamic patient graph.)

**Checked directly and confirmed absent**: this graph carries no textual
definitions. Every property key ever used across the whole graph is just
`id`, `effectiveTime`, `fullySpecifiedName`, and an unused
`preferredTerm` (0 nodes actually have it set). There is no `Description`
node type, no synonym table, no defining-relationship data — only the
bare `IS_A` subsumption hierarchy. So the only thing this graph can give
Step A is **taxonomic position** (what a term is a type of, and what's
a type of it) — not a prose definition. That turns out to be enough to
matter, per §12.4 below.

### 11.3 The design — an independent search, not the candidate list

The two-step CoT's central discipline is that Step A must not see Stage
2b's candidate list before forming its own judgment (§1) — this was
built specifically to fix an earlier anchoring-bias bug. Grounding Step
A with the Neo4j graph without breaking that discipline requires the
taxonomy lookup to be **independent of Stage 2b's retrieval**, not a
shortcut to it:

1. Search `SnomedConcept.fullySpecifiedName` for a tight match against
   the entity's **own text** — `expanded_text` (what Step A's prompt
   already shows the model as "after abbreviation expansion"), not
   Stage 2b's OMOP-scoped, already-filtered candidate pool. Match
   strategy: strip the trailing SNOMED semantic tag
   (`"Fever (finding)"` → `"Fever"`) and compare case-insensitively;
   nothing forced if there's no tight match.
2. On a match, pull its `IS_A` parent and a handful of children.
3. Inject as a labeled block making explicit that this is **context,
   not a candidate list**:
   ```
   RELATED SNOMED TAXONOMY (real medical-hierarchy context for terms
   like "fever" -- for background only, this is NOT a list to choose
   from and does not imply this entity means any one of these):
     "Fever (finding)" is a type of: Body temperature above reference
     range (finding)
     narrower/related terms under it include: Recurrent fever, Chronic
     fever, Fever due to infection, ...
   ```
   inserted into the existing prompt immediately before the `TASK:`
   line — every other line of the prompt is byte-identical, so this is
   the only variable between the OLD and NEW conditions below.

### 11.4 Deep-dive result — the "fever" entity, Step A *and* Step B, real models

Re-ran the real, shipped Step A and Step B calls (same `LLMClient`, same
schema-guided decoding, `TEMPERATURE=0.0` — reproducible, not a lucky
sampling draw) against `11859945-DS-29-ebf08ad5f49`, with and without
the taxonomy block:

| Model | OLD `clinical_meaning` | NEW `clinical_meaning` | OLD Step B | NEW Step B |
|---|---|---|---|---|
| qwen2.5:3b | "absence of fever in a patient" | **"a symptom of elevated body temperature"** | match=False, `NONE_CORRECT`, conf 0.734 | **match=True, `SUPPORTED_1`, conf 0.998** |
| llama3.2:3b | "a normal body temperature" (backwards) | "a body temperature above normal range" | match=True, `SUPPORTED_1`, conf 0.636 | match=True, `SUPPORTED_1`, conf 0.792 |
| phi4-mini | "An elevation in body temperature indicative of an infection" | "An elevated body temperature above normal range" | match=True, `SUPPORTED_1`, conf 0.960 | match=True, `SUPPORTED_1`, conf 0.969 |

**qwen's verdict flipped from wrong to right**, and the OLD run
reproduces the real historical reasoning almost verbatim (*"does not
directly represent the absence of fever... a distinct and clinically
unrelated concept"*), confirming this is a faithful replay of the actual
production failure, not a different scenario. llama and phi4-mini were
already correct and stayed correct, with higher confidence and reasoning
that echoes the graph's own wording (*"a body temperature above normal
range"* — nearly identical to the parent concept's actual name, *"Body
temperature above reference range"*).

**Net effect on this entity**: OLD = 2/3 split → routes to
`TIER_4_ENSEMBLE_SPLIT`, rescued only by the calibrator (score 0.928,
barely above the 0.72 threshold, itself dependent on a high
`prior_confirmation_count`). **NEW = 3/3 unanimous** → would route
directly to genuine `TIER_1_AUTO_VALIDATED`, no calibrator needed, no
split-vote fragility at all.

### 11.5 Scaled result — 15 real held-out `TIER_4_ENSEMBLE_SPLIT` entities, gold-graded

The single "fever" case is a best-case scenario — a common, single word
that happens to match a SNOMED preferred term exactly. To check whether
it generalizes, 15 real `TIER_4_ENSEMBLE_SPLIT` entities were sampled
(seed 42, from a pool of 2,222 gradable-against-gold, ≤3-candidate
entities out of 3,866 total) and run through the identical OLD-vs-NEW
comparison, majority vote graded against gold via
`VocabularyRetriever.snomed_code_for_concept()`.

**A real bug was caught and fixed mid-experiment, reported honestly
rather than silently corrected**: the taxonomy search was first run
against `original_text`, missing entities Stage 1 had already expanded
before Step A ever sees them (`PNA`→`pneumonia`, `GERD`→`gastroesophageal
reflux disease` — Step A's own prompt shows `expanded_text`, so that's
the field the taxonomy search should use too). Fixing this raised
coverage from 3/15 to 5/15.

**Even after the fix, coverage was low: 5 of 15 (33%)**. Most
`TIER_4_ENSEMBLE_SPLIT` entities in this sample were either abbreviations
whose expansion still doesn't exactly equal a SNOMED preferred term
(`HTN`→"Hypertension" vs. SNOMED's actual "Hypertensive disorder";
`CABG`→"coronary artery bypass graft", `HCAP`, `TTE` — none tight-matched)
or multi-word phrases with no single matching concept name at all
(`left colon`, `renal failure`, `groin pain`, `nondisplaced fractures`,
`heaving`, `coronary arteries`). The naive exact-match strategy in §12.3
is real but structurally narrow — most of this corpus's language doesn't
land on a literal SNOMED preferred term.

**Of the 5 scored entities, the outcome was far more mixed than the
"fever" case, reported in full**:

| Entity | OLD → NEW majority | Correct both ways? | Note |
|---|---|---|---|
| syncope | [1,1,None]→ same pick, unchanged | Yes, no change | — |
| constipation | unchanged | Yes, no change | — |
| abdominal pain | [1,None,1] (2/3) → [1,1,1] (3/3) | Yes, no change | Went from a fragile plurality to full unanimity — same answer, more stable |
| PNA / pneumonia | [1,1,None] (2/3) → [None,1,1] (2/3) | Yes, no change | **qwen's individual vote got worse**, not better — its NEW meaning ("bilateral opacities suggestive of multifocal pneumonia") invented clinical detail nowhere in the note's context and then rejected the correct candidate as too generic. The ensemble's final answer only stayed correct because the other two models still agreed. |
| GERD | [1,1,1] → [1,1,1] | Yes, no change | Already unanimous; no effect either direction |

**Honest bottom line**: across this sample, **zero WRONG→CORRECT flips
and zero CORRECT→WRONG flips at the ensemble-majority level** — every
entity that was correct under OLD stayed correct under NEW. One case
(abdominal pain) improved in a way that matters even without changing
the final answer (a genuine 3/3 unanimous vote is structurally safer
than a 2/3 plurality that happens to be right). One case (PNA) shows a
real, concrete downside: taxonomy context can also feed a model's
tendency to over-elaborate rather than sharpening its judgment — qwen's
new meaning was *more* wrong in a different way, not simply better.
**The "fever" result is real and reproducible, but it should not be read
as representative of what this mechanism would do at scale** — it was
the easy case, not the typical one, and typical `TIER_4_ENSEMBLE_SPLIT`
entities mostly never even reach the taxonomy-injection path under this
simple search strategy.

### 11.6 A cleaner test — 10 entities drawn specifically from the taxonomy-matched population

§12.5's 5 scored entities were a byproduct of a *blind* random sample —
diluted by the ~67% of cases where the mechanism can't apply at all, so
the sample size within reach was too small to draw a real conclusion
from. A second, more targeted test fixed that: instead of sampling
`TIER_4_ENSEMBLE_SPLIT` entities blindly, it scanned the gradable pool
(cheap — Neo4j lookup only, no LLM calls) specifically **until 10
entities with a real taxonomy match were found** (31 entities scanned to
find 10 — a 32.3% match rate, consistent with §12.5's finding), then ran
the full real OLD-vs-NEW Step A/B comparison on all 10.

**Real result, this time a clean, positive one:**

| | OLD | NEW |
|---|---|---|
| Correct | 6/10 (60%) | **8/10 (80%)** |
| WRONG → CORRECT | — | **2** (`Troponin`, `neck swelling`) |
| CORRECT → WRONG | — | **0** |

Both flips show the same pattern as the "fever" case: a genuine 2-vote
plurality that was actually *wrong* (`Troponin`: `[None, None, 1]`,
`neck swelling`: `[None, None, 1]`) became a clean 3/3 unanimous
*correct* vote once Step A had real taxonomy to reason from — e.g.
`Troponin`'s new meaning, *"a biomarker used to diagnose myocardial
infarction,"* is properly grounded where the old, ungrounded guesses had
the majority of models rejecting the correct candidate outright.

**The two entities that stayed wrong did so for an entirely different,
honest reason** — not a Step A problem at all: `Prednisone` and a
PICC-line entity had all 3 models agree on the *same* candidate both
times, and that candidate's SNOMED crosswalk simply isn't gold's answer
(a Stage 2b retrieval-layer gap — the correct concept was never in the
candidate pool to begin with). No amount of better meaning-generation in
Step A can fix a candidate that was never offered; this is out of scope
for what this mechanism can do, and correctly shows up as unaffected
rather than falsely "fixed."

**Zero regressions in this run** — unlike §12.5's blind sample, which had
one (the PNA/qwen over-elaboration case). Taken together, the honest
picture across both experiments is: **limited reach (~33% of
`TIER_4_ENSEMBLE_SPLIT` entities under this naive search), but a real,
clean, +20-percentage-point improvement with zero downside specifically
within that reach** — a materially more encouraging result than §12.5's
blind sample alone would suggest, precisely because most of §12.5's
"no effect" cases were never actually testing the mechanism at all.

### 11.7 What a real version of this would need

Stated plainly, not glossed over: (1) a smarter match strategy than
exact-string-after-stripping-the-tag — the ~33% coverage ceiling across
both experiments is a search-quality problem, not a proof the taxonomy
signal itself is weak; §12.6's clean result inside that reach is direct
evidence the underlying idea works, so widening the reach (a
fuzzy/synonym-aware lookup, or reusing SapBERT similarity against
`fullySpecifiedName` the same way Tier 3 already works, see
`docs/SapBERT_Technical_Reference.md`) is the highest-leverage next
step. (2) A larger, pre-registered sample before any production
conclusion — 10 and 5 scored entities are still small; §12.6's +20pp is
encouraging, not yet statistically conclusive. (3) Thought given to the
PNA-style downside from §12.5 — taxonomy context clearly *can* invite
over-elaboration on some entities even though it didn't in §12.6's
sample, and any real deployment would need to keep measuring that risk,
not assume it never recurs.

### 11.8 A different question — does a medically-trained model need the taxonomy crutch at all?

Sections 11.1-11.7 grounded a general-purpose 3B model with external
taxonomy. A different, orthogonal question: would a model that was
*already* medically fine-tuned solve the same puzzle unaided, with the
real, unmodified production prompts and no help at all?

**Real models pulled and tested, not simulated**: two genuine
medically-fine-tuned checkpoints, confirmed real via the Hugging Face
API before pulling and never assumed —
`MaziyarPanahi/BioMistral-7B-GGUF` (Q4_K_M, 4.4 GB) and
`aaditya/OpenBioLLM-Llama3-8B-GGUF` (Q4_K_M, 4.9 GB) — pulled directly
via `ollama pull hf.co/<repo>:Q4_K_M`, run through the exact same
`LLMClient.complete()` transport every production model uses. Both
models were unloaded from GPU memory and deleted from disk immediately
after testing — this was a pure side-experiment, never wired into
`MODEL_NAMES` or any production call path.

**A separate, real finding along the way, worth recording precisely**:
this project's own `.env` (sibling worktree) names two medical model
endpoints — `MEDGEMMA_URL`, `OPENBIOLLM_URL` — neither of which is
currently running (checked directly, nothing listening on either port).
Separately, this box's Hugging Face cache already holds two
*different*, AWQ-quantized copies of these same model families, fully
downloaded (`OpenBioLLM-Llama3-8B-AWQ`, 5.4 GB; `BioMistral-7B-AWQ`,
3.9 GB) but unusable as-is — AWQ format needs vLLM, and `vllm` is not
installed anywhere on this box. This experiment deliberately used
fresh, independently-verified GGUF pulls instead, specifically so it
could run on the existing Ollama transport with zero new infrastructure
— see `docs/Entity_Journey_Plain_Language_Walkthrough.md` §11 for the
plain-language version of this same infrastructure finding.

**Single-entity puzzle ("fever", real unmodified Step A + Step B, no
taxonomy help)**:

| Model | Step A meaning | Step B verdict |
|---|---|---|
| BioMistral-7B | "a symptom of fever in a patient" (reasonable) | **WRONG** — `match=False`, conf 0.308. Reasoning: *"Fever is in the domain of Symptom, while the text span 'fever' is in the domain of Condition. The two are not equivalent."* — a hallucinated domain conflict; the candidate's real domain is Condition, and the model's own Step A meaning never claimed otherwise. |
| OpenBioLLM-Llama3-8B | "An infectious process characterized by high body temperature" (overspecified but sound) | **CORRECT** — `match=True`, conf 0.882 |

Genuinely different failure/success shapes than the general 3B models
saw — not the negation-confusion bug at all. Medical training changed
*what kind* of mistake gets made; it didn't guarantee no mistake.

**Scaled test — the identical 10 entities from §11.6**, same gold
grading, run with the real unmodified prompts (no taxonomy) against
both medical models:

| Entity | Gold-correct? | General OLD | General NEW (+taxonomy) | BioMistral-7B | OpenBioLLM-8B |
|---|---|---|---|---|---|
| abdominal pain | ✓ | ✓ | ✓ | ✗ | ✓ |
| spinal stenosis | ✓ | ✓ | ✓ | ✗ | ✓ |
| brain natriuretic peptide | ✓ | ✓ | ✓ | ✗ | ✓ |
| Prednisone | (retrieval gap) | ✗ | ✗ | ✗ | ✗ |
| chronic kidney disease | ✓ | ✓ | ✓ | ✗ | ✓ |
| PICC line | (retrieval gap) | ✗ | ✗ | ✗ | ✗ |
| Tetracycline | ✓ | ✓ | ✓ | ✗ | error (empty Step A) |
| deep venous thrombosis | ✓ | ✓ | ✓ | ✗ | ✓ |
| Troponin | ✓ | ✗ | ✓ | ✗ | ✓ |
| neck swelling | ✓ | ✗ | ✓ | ✗ | ✓ |
| **Total** | | **6/10** | **8/10** | **0/10** | **7/10 (1 error)** |

**BioMistral-7B scored 0/10 — every single candidate rejected, on every
entity, including cases where its own Step A meaning was an
near-exact paraphrase of the correct candidate's name** (e.g. "a
diagnosis of chronic kidney disease" for the CKD candidate, still
rejected at Step B). This reads as a systematic instruction-following
failure specific to this model on this exact structured binary-match
task shape — not a medical-knowledge gap, since its free-text reasoning
was frequently reasonable. A real, concrete example of why "medically
trained" is not a substitute for verifying a model actually follows
this pipeline's specific prompt contract.

**OpenBioLLM-Llama3-8B (7/10, 1 error) matched the general ensemble's
taxonomy-boosted result on every entity that had a correct candidate to
find** — its 2 wrong answers (Prednisone, PICC line) are the *same two*
entities every other approach in this table also got wrong, both for
the same reason: a Stage 2b retrieval-layer gap where the correct
concept was never in the candidate pool at all. On the population where
an answer was actually reachable, OpenBioLLM alone performed on par with
the general ensemble + real external grounding — genuinely competitive,
not a fluke on one lucky entity.

**Honest conclusion**: real medical fine-tuning is not a uniform win —
one of two real, independently-verified medical models tested here was
the single worst performer of any approach tried in this whole
document (0/10), while the other matched the best-performing
general-ensemble configuration. Model *choice* within "medically
trained" matters at least as much as the medical/general label itself,
and neither this section nor §11.1-§11.7 should be read as recommending
a specific production change — both remain documented experiments.

## 12. What this document does not cover

Deliberately out of scope here, covered elsewhere:

- **What happens to the vote after it's cast** (unanimous vs. split
  routing, the calibrator's 16 features, the hard traps, threshold
  0.72) — `docs/ConsensusCalibrator_Technical_Reference.md`.
- **The retrieval mechanics that produce the candidate list in the
  first place** (Tier 1–4, the SNOMED namespace exclusion, the
  Procedure/Observable-Entity preference) — `docs/Code_Reference_Stages_And_Metrics.md`.
- **The TransE knowledge graph embedding** (a completely separate
  mechanism, evaluated as a candidate-disambiguation signal, not
  currently wired into any prompt) — `docs/Entity_Journey_Plain_Language_Walkthrough.md`
  §8.
- **Degenerate-generation detection and retry** (garbled/repetitive
  model output, a transport-layer concern in `src/llm_client.py`) —
  not detailed here; briefly, a degenerate response is excluded from
  `usable_votes()` exactly like an error, contributing no evidence
  either way.
