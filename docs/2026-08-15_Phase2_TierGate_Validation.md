# Phase 2 (Pass 4 Tier 1-5 gate) — real validation, 2026-08-15/16

**Status: three rounds of real-data diagnosis, each overturning part of the
previous conclusion. The two-step CoT mechanism looks structurally sound on
hand-inspection; the barriers to the 90% target are (a) a fixable per-model
strictness asymmetry (partially mitigated), (b) a class of entities that
should never have reached the ensemble at all (now filtered out), and (c) a
measurement confound in this validation script's own grading, unrelated to
gate quality, that has consumed the entire gradable sample so far. Net
result: still no reliable precision read. Read this whole document before
drawing conclusions from any single number in it.**

## Round 1 — LOW-tier only (18 entities)

```
AUTO coverage: 5.6% (target ~90%), 66.7% Tier 4 splits
```
Hypothesis: only tested Stage 2b's hardest, pre-filtered LOW tier.

## Round 2 — HIGH-tier only (36 entities) — hypothesis disproven

```
AUTO coverage: 2.8%, 77.8% Tier 4 splits
```
Even Stage 2b's own most-confident entities split constantly. Reading actual
model reasoning from the stored results showed why: **`qwen2.5:3b` was the
consistent dissenting vote** on obviously-correct cases (plain
"pneumonia"/"heart failure"/"sodium"), demanding contextual specificity
(laterality, subtype, severity) the candidate's bare name was never going to
carry.

## Rejected fix — loosening the match rule for all three models

A 5th rule ("match the core concept, not every detail") dropped Tier 1
precision to 5.9% (1/17) by letting all three models unanimously
rubber-stamp WRONG matches on bare qualifier/fragment spans ('left',
'Removal', 'Multiple') that are not independently linkable concepts at all.
**Reverted.**

## Round 3 — two targeted, narrower fixes (user's combined proposal)

**Fix A — qualifier-fragment precheck** (`qualifier_fragment_precheck()`,
`src/mollm_tier_gate.py`): confirmed against real DB data (not a guessed
word list) that `gliner_label == "Qualifier"` is exactly the label covering
the fragment entities that caused the precision collapse ('left', 'right',
'multiple', 'third', 'R', 'Cranial' all label Qualifier; genuine
single-word concepts like 'chest'→Anatomy and 'pain'→Symptom carry different
labels and are unaffected). These now route straight to HITL
(`queue_reason=standalone_qualifier_span`) without spending a single model
call — removed from the ensemble's job entirely rather than asking a prompt
to get them right.

**Fix B — qwen-only subsumption clause** (`QWEN_SUBSUMPTION_CLAUSE`): a 5th
rule about accepting a correct-but-less-specific hierarchical relative,
applied ONLY to `qwen2.5:3b`'s Step B prompt; `llama3.2:3b`/`phi4-mini`'s
prompts are unchanged, since they were not the ones over-rejecting.

Re-ran the same 36-entity HIGH-tier batch:

```
TIER_1_AUTO_VALIDATED: 6-7 (16.7-19.4%)
TIER_4_ENSEMBLE_SPLIT: 15 (41.7%)
TIER_5_TRUE_AMBIGUITY: 14 (38.9%), 10 of which are standalone_qualifier_span
AUTO coverage: 16.7-19.4% (target ~90%)
Tier 1 precision: 16.7% (1/6 gradable)
```

Real, measured improvement: AUTO coverage 2.8% → ~19%, with the qualifier
spans correctly diverted before ever reaching the ensemble. Still far short
of target, and Tier 1 precision still looks bad at 16.7%.

## The precision number is more confounded than it looks — a grading bug, not a gate bug

Inspecting the "incorrect" Tier 1 cases by hand (`fixation`→`Fixation`,
`clavicular fracture`→`Fracture of clavicle`, `pneumonia`→`Pneumonia`) showed
**sound, defensible model reasoning in every case** — these did not look like
errors. Checking the raw gold/candidate data directly confirmed why:

- **`clavicular fracture`**: gold annotates `"left clavicular"` and
  `"fracture"` as TWO SEPARATE spans with two separate SNOMED codes. The
  entity extracted `"clavicular fracture"` as ONE span with ONE (arguably
  *better*) candidate, `"Fracture of clavicle"`. A single candidate
  structurally cannot match both gold codes at once. This is
  `scripts/score_gold_recall.py`'s own **already-documented** compound-span
  caveat ("the fix is Stage 2a splitting compound spans... not normalization
  tuning") — my batch script's `grade()` was not accounting for it and was
  scoring these as flatly wrong.
- **`fixation`**: gold annotates the fuller phrase `"intermaxillary
  fixation"`; the entity's own span is only `"fixation"`, a strict substring,
  normalized to the more generic candidate `"Fixation"`. Not a compound
  mismatch, but the entity was never shown the more specific concept gold
  expects, because its own span is narrower than gold's.
- **`pneumonia`**: same pattern — gold's span is `"aspiration \npneumonia"`
  (a specific subtype), the entity's span is just `"pneumonia"`.

Fixed `grade()` (`scripts/run_tier_gate_batch.py`) to detect and separately
flag both patterns (`compound_span`, `narrower_than_gold`) rather than
silently scoring them as plain wrong, and to report a "clean-span-only"
precision figure alongside the raw one. Re-running the same batch with the
fixed grader: **all 6 gradable Tier 1 decisions carry one of these two
flags — zero clean, unconfounded data points (0/0 = n/a)**. The 16.7% number
is not a measurement of the gate's judgment quality; it is a measurement of
how often this specific 36-entity HIGH-tier sample happens to have
compound/narrow-span gold annotations, which is a Stage 2a extraction
question this validation run cannot separate from gate quality.

## Where this leaves things

Three additive findings, not one:
1. **qwen2.5:3b strictness asymmetry** — real, diagnosed, partially mitigated
   (Fix B). Not yet re-measured cleanly due to (3) below.
2. **Standalone qualifier/fragment spans reaching the ensemble** — real,
   diagnosed, fully fixed (Fix A) at essentially zero cost (a free precheck).
3. **This validation script's own grading conflated compound-span and
   narrower-than-gold entities with genuine errors** — a measurement bug in
   `scripts/run_tier_gate_batch.py`, not in `src/mollm_tier_gate.py`. Fixed,
   but it consumed the ENTIRE gradable sample on this particular 36-entity
   batch, leaving no clean signal to report yet.

None of the three findings individually explains the gap to 90% AUTO
coverage; each closes off one confound so the next run can see more clearly.
Coverage (19.4%) is still the harder number to move — most of the remaining
volume is Tier 4 splits (41.7%) on entities the qualifier filter correctly
does not touch (real clinical nouns like `heart failure`, `Hepatopathy`,
`rib fractures`), meaning the qwen mitigation (Fix B) has not yet closed the
gap on its own.

## What is NOT yet done in Phase 2

- No batch has yet produced a clean, unconfounded precision measurement.
  Needed: either a larger sample (so clean-span cases appear by chance) or a
  sample deliberately restricted to entities whose span already matches a
  gold annotation exactly, before trusting any precision number from this
  gate.
- Fix B's effect on qwen's dissent rate specifically has not been isolated
  from Fix A's effect (both landed together in this round) — worth a
  follow-up run with only one fix active if the combined effect needs
  decomposing.
- `src.kg3_ingestion.ingest_auto_decision()` remains verified-but-unused by
  the batch runner (measurement only, by design) -- more true than ever
  given (3) above.
- `src.mollm_ensemble.py`'s production `route()` is untouched.

## Recommendation

Do not draw a final precision conclusion from any number in this document —
each round has been confounded by something different, and this round's
confound (grading methodology) happened to consume 100% of the gradable
sample. The next useful step is a larger and/or more carefully sampled batch
run specifically to get clean-span data points, not another prompt or
filter change on top of an unmeasured baseline.
