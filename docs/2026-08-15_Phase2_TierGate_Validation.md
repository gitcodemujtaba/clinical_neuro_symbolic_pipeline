# Phase 2 (Pass 4 Tier 1-5 gate) — real validation, 2026-08-15/16

**Status (updated after Round 4): a clean, unconfounded measurement finally
landed. Tier 1 precision 94.4% (17/18) on a curated atomic-only sample,
AUTO coverage 50.0% (up from 2.8% at the start of this diagnosis). The one
error traces to upstream retrieval quality (a single, close-but-wrong Tier-3
semantic candidate), not the gate's judgment. Coverage is still short of the
90% target -- most of the remaining volume is genuine Tier 4 splits on
harder multi-word conditions and Tier 5 unresolved acronyms (expected,
since Pass 1's MoLLM escalation, plan Phase 4, is not built yet) -- but
precision, the dimension that actually protects KG3 integrity, now has real
evidence behind it. Rounds 1-3 below are kept for the diagnostic trail; read
them for how this conclusion was reached, not as the final word.**

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

## Round 4 — curated "atomic-only" batch (user's proposal) — clean signal at last

Round 3 ended with zero clean data points: every gradable Tier 1 decision on
that 36-entity sample happened to carry a `compound_span` or
`narrower_than_gold` flag. Rather than run a bigger batch of the same
unfiltered kind and hope clean cases show up, built a **curated selection**
(`select_atomic_entities()`, `scripts/run_tier_gate_batch.py`) that pulls
candidates from ALL 32 already-processed notes (not just the 3 canonical
ones -- a much bigger pool, 549 eligible entities found), keeping only
entities where:
- `gliner_label` is `Condition`/`Procedure`/`Medication` (substantive
  clinical nouns; `Qualifier` is never in this set, so fragment spans are
  excluded by construction, on top of `qualifier_fragment_precheck()`
  already catching them at routing time).
- Exactly ONE gold annotation overlaps the entity's span (unambiguous
  mapping target -- excludes compound phrases upstream, before any grading
  even happens).
- The entity's own span is at least as long as that gold annotation's
  (excludes the `narrower_than_gold` pattern upstream too).

Also added a **general** grader improvement (not curated-batch-specific):
`_is_precoordination_match()` in `grade()` now promotes an otherwise-"wrong"
`compound_span` case to `correct`/`compound_span_precoordinated` when the
chosen candidate's name textually subsumes every gold fragment's text (e.g.
`'Fracture of clavicle'` subsumes `'clavicular'` + `'fracture'`) -- a
heuristic proxy for SNOMED pre-coordination, not an authoritative ontology
check (see that function's docstring for exactly what it does and does not
verify). This applies on any future run, curated or not, not just this one.

36-entity curated batch, real Ollama calls, both Fix A (qualifier precheck)
and Fix B (qwen subsumption clause) active:

```
TIER_1_AUTO_VALIDATED: 18 (50.0%) -- 18 gradable, 17 correct, 94.4% precision
  (clean-span only: 17/18 -- 94.4%, i.e. essentially no confound left in this tier)
TIER_4_ENSEMBLE_SPLIT: 13 (36.1%)
TIER_5_TRUE_AMBIGUITY: 5  (13.9%) -- 3 unresolved_acronym (CAD/MR/ACS -- expected,
  Pass 1 MoLLM escalation is plan Phase 4, not built), 2 verdict_none_correct
AUTO coverage: 50.0% (target ~90%)
```

The single error (`'Bilateral rib fractures'`): all 3 models unanimously and
reasonably accepted the only candidate Stage 2b offered
(`'Fracture of two ribs'`, Tier-3 semantic match, similarity 0.88) against a
gold code for a different specific rib-fracture concept. Checked directly
against the DB -- this is a retrieval-quality gap (the correct candidate was
never in the list to begin with), not a two-step CoT reasoning failure.
Exactly the kind of gap Phase 3's planned hybrid retrieval work would close,
not evidence against Pass 4's gating logic.

## Recommendation (updated)

Precision is no longer the open question -- 94.4% on a real, unconfounded
sample is strong evidence the two-step CoT + Tier 1-5 gate makes sound
judgments once qualifier fragments are filtered and the qwen asymmetry is
addressed. **Coverage is the remaining gap** (50.0% vs. ~90% target), and the
breakdown suggests where it comes from: `ensemble_split` (36.1%, harder
multi-word conditions the ensemble still doesn't unanimously agree on) and
`unresolved_acronym` (part of the 13.9% Tier 5 share, structurally expected
until Phase 4's Pass 1 MoLLM escalation exists). Two reasonable next moves,
not mutually exclusive: (a) apply the same kind of targeted diagnosis Round
2-3 used to the `ensemble_split` cases specifically, now that qualifier
noise and grading confounds are out of the way, or (b) treat 50% coverage
with 94% precision as a legitimate, shippable Phase 2 milestone on its own
(the plan's own Tier 4/5 both route to human review regardless, so a lower
coverage number does not compromise KG3 integrity -- it just means more
volume goes to HITL than the spec's target) and move on to Phase 3
(retrieval), whose improvement would directly reduce both the
`ensemble_split` rate (better candidates, less to disagree about) and the
one measured error class above.
