# Shadow run at scale — precision does NOT hold, 2026-08-16

**Status: the shadow run did its job. It disproves, rather than confirms,
that 94.4% precision holds at natural-population scale, and surfaces a
specific, actionable root cause along the way. `dry_run=False` should NOT
be considered until this is addressed. This is a successful use of the
shadow-run methodology, even though the headline number is bad news.**

## What ran

`scripts/run_stage3_tier_gate.py` across all 32 already-processed notes,
`--limit-per-note 5` (~150 entities, bounded per the user's chosen scope
after the originally-requested 500-1000 notes turned out to exceed the
entire 272-note dataset). 155 processed, 0 errors, 14.3 minutes, all KG3
writes dry-run. Combined with the prior smoke-test's 10 entities: **56 total
AUTO-tier decisions** now stored in `mollm_tier_gate_decisions`.

## The number

Graded all 56 AUTO decisions against gold, applying the same
compound-span/narrower-than-gold corrections validated in Phase 3's A/B
work (see `docs/2026-08-16_Phase3_HybridRetrieval_Validation.md`):

```
ALL gradable (raw):  37/51 = 72.5%
CLEAN-span only:     35/46 = 76.1%
```

This is a substantial, real drop from the 94.4% (17/18) figure the Phase 2
lock was based on. That figure came from a **curated "atomic-only" sample**
— deliberately filtered to unambiguous, single-concept Condition/Procedure/
Medication entities. This shadow run pulled from the natural entity
population (every GLiNER label, no atomic filtering), which is a fairer
estimate of what real deployment would actually see.

## Why the drop, not just that it dropped

Inspected all 11 clean-span "incorrect" cases directly (predicted concept
name/class vs. gold's own concept name/class, via `athena_concept`):

**A real, systematic, diagnosable bug**: `'morphine'` resolved to a drug
PRODUCT concept, but gold expected `'Allergy to morphine'` (a Clinical
Finding). `'trazodone'` shows the identical pattern (`'Allergy to
trazodone'`). Both entities almost certainly came from an Allergies note
section, extracted/labeled Medication, and resolved as if the patient is
*taking* the drug rather than *allergic to* it. This is not ensemble noise
-- it's a context/section-handling gap somewhere upstream of Pass 4 (Stage
2a's labeling or Stage 2b's domain restriction not accounting for allergy
context), and it is exactly the kind of error a human reviewer must catch,
not an argument that Pass 4's gate itself is unreliable.

**Likely vocabulary duplicate-concept pairs, not clear errors**:
`'gunshot wound'` predicted "Gunshot wound" (concept_class=Disorder) against
gold's own "Gunshot wound" (concept_class=Morph Abnormality) -- identical
display name, different SNOMED axis. `'blurred vision'` shows the same
shape ("Blurred vision"/Disorder vs. gold's "Blurring of visual image"/
Clinical Finding). These read as SNOMED's own duplicate/near-duplicate
concept pairs, not a case where the model or gate reasoned incorrectly --
worth checking against `athena_concept_relationship` for a formal
duplicate/SAME_AS marker before concluding either way, not yet done here.

**`'RCA occlusion'` — ADJUDICATED, gold annotation error.** Pulled the
entity's actual `local_context` (`extracted_entities.local_context`, note
`11649745-DS-4`): "...status post cardiac Catheterization and ___ for
**right coronary artery occlusion** ___. Pre and post cath he received
Antiplatelet therapy... as well as intraprocedural heparin continuous
infusion." Unambiguously a coronary-PCI note (catheterization, antiplatelet
therapy, heparin infusion) with no carotid/neurology/vascular-surgery
context anywhere nearby. The gate's prediction ("Right coronary artery
occlusion") is correct; gold's code (`285171000119104`, "Right carotid
artery occlusion") is the error here, not the system. **Recomputed
clean-span precision with this one case corrected: 36/46 = 78.3%**
(up from 76.1%) -- still a real, substantial gap from 94.4%, but confirms
at least one "error" was actually a correct output penalized by a bad
label, not evidence the gate reasoned incorrectly.

## What this means for the deploy decision

**Precision does not hold at the scale/generality this shadow run tested.**
The honest, corrected number is ~78%, not 94.4% (adjudicating the RCA
occlusion case as a gold error recovers one point, 76.1%→78.3%). Most of
the remaining gap traces to identifiable causes rather than random
unreliability -- one real, fixable bug (allergy-context mishandling), likely
vocabulary-level ambiguity (SNOMED duplicate concepts), and at least one
gold-labeling error -- which is useful, actionable signal, but "most of our
errors are explainable" is not the same claim as "safe to write
unreviewed." `dry_run=False` remains correctly withheld. The earlier
decision to keep `enqueue_pending_cases()` queuing every AUTO-tier decision
for human review (rather than excluding them per the original plan) is now
directly validated by this data, not just a cautious default.

## Update, same session: allergy-context bug fixed, and a deeper limit found

Implemented the fix. Confirmed `SECTION_EXPERIENCER_OVERRIDE`/
`SECTION_TEMPORALITY_OVERRIDE` (`src/preprocessing.py`) don't cover this --
neither dict has an "allergies" entry, despite `preprocessing.py`'s own
module docstring naming this exact case as a 2026-08-08 target that was
apparently never implemented. Sized the population precisely: **19 entities**
across the 32 processed notes fall in an "Allergies" section, **100% labeled
Medication with `assertion_status=PRESENT`** -- a small, completely
systematic, 100%-affected population.

**Built**: `src.assertion.STATUS_ALLERGY` + `apply_allergy_context_override()`
(wired into `entity_extraction.py` right after `apply_section_priors()`),
plus the retrieval-side fix in `orchestrator.py`'s
`process_and_normalize_entities()`: when `assertion_status == STATUS_ALLERGY`
and the entity is Medication-labeled, the SEARCH (not the entity's own
stored label) becomes `"Allergy to {text}"` against the Condition domain
(reusing the existing `domain_override` mechanism, `gliner_label="Condition"`
for the search only -- so `VOCAB_BY_LABEL`'s Medication->RxNorm restriction
doesn't wrongly apply). 6 new unit tests
(`tests/test_allergy_context_override.py`), full suite still 49/49 [55/55
after this addition].

**A deeper, separate limitation surfaced while verifying this end-to-end.**
Live-tested `normalize_entity("Allergy to morphine", ..., domain_override=
["Condition"])` against the real DB: it does NOT find SNOMED's own
"Allergy to morphine" concept (id 4164683) -- because that concept has
`standard_concept = NULL`, and EVERY tier of this pipeline's retrieval
(Tier 1/2/3, not just this fix) filters `WHERE standard_concept = 'S'`
everywhere, as a pre-existing, pipeline-wide convention unrelated to this
fix. Checked `athena_concept_relationship`: `4164683` "Maps to" only a
single, generic standard concept, `439224` "Allergy to drug" -- OMOP
collapses every specific-drug allergy to that one generic concept in its
standard hierarchy. So even with this fix, morphine/trazodone allergy
entities will resolve to the generic "Allergy to drug" or a semantically
nearby standard concept (Tier 3 found "Poisoning by morphine" 0.72,
"Morphine dependence" 0.64, "Trazodone poisoning" 0.73 -- real ADR-adjacent
concepts, not exact allergy matches), not gold's specific SNOMED code. **The
fix is directionally correct and real** (Condition-domain allergy search
instead of Medication-domain drug-product search), **but will not by itself
fully close the precision gap for this error class** -- gold apparently
expects non-standard-OMOP SNOMED codes for at least some allergy findings,
which this pipeline's standard-concept-only convention cannot reach at any
tier. Whether to relax that convention (a bigger, cross-cutting question
well beyond this one bug) is unresolved and not attempted here.

## Update, same session: the OMOP boundary decision was made and implemented

Of the three options considered for the standard-concept-only limitation
(build a general non-standard<->standard crosswalk, relax the filter
narrowly, or accept the generic "Allergy to drug" ceiling), implemented the
middle option: `_apply_allergy_nonstandard_exact_override()`
(`src/normalization/orchestrator.py`). An exact, case-insensitive match on
the synthesized `"Allergy to {drug}"` string is looked up directly against
`athena_concept` WITHOUT the `standard_concept = 'S'` filter -- scoped ONLY
to this one deliberately-constructed search pattern, not a global relaxation
of Tier 1/2/3 for every entity. Verified correct in isolation (direct
function call and a full `process_and_normalize_entities()` call both
return `"Allergy to morphine"`, concept_id 4164683, as intended). Full
pipeline-level validation (does this actually land correctly when driven by
`test_pipeline_e2e.py` end to end) is in progress -- see below.

**A known, pre-existing performance sink found along the way, logged but
NOT touched (out of scope for this fix, deferred to a future optimization
phase / plan Phase 6)**: `process_and_normalize_entities()`'s Lab Value
Suffix Fallback (`orchestrator.py`, guarded by `label == "Lab Test"`) runs a
nested double loop where EVERY iteration calls `normalize_entity()` again --
a fresh SapBERT embedding computation and full Tier 1/2/3 search per
stripped candidate. This is pre-existing code, unrelated to the allergy fix,
but it measurably slows any batch containing several Lab Test entities with
multiple `strip_lab_value_suffix()` candidates. Confirmed via `ps` during a
stalled-looking re-run: 878% CPU (heavy multi-threaded model inference, not
a hung loop) is consistent with this fallback legitimately churning through
several embedding calls in sequence, not a bug in the new allergy-context
code (which has no loops at all -- reviewed line-by-line, see commit
history). Worth a future optimization (e.g. try DuckDB Tier 1/2 for each
stripped candidate before ever paying for a SapBERT call), not attempted
here to avoid scope creep on top of the allergy investigation.

## Update, same session: the allergy fix's final, measured effect

Re-ran Stage 1/2a/2b (`test_pipeline_e2e.py`, 522 entities, 0 errors) and the
new Stage 3 tier gate (`run_stage3_tier_gate.py`, 391 entities, 0 errors, 32.6
min) on all 6 notes containing the 19 allergy-context entities, with both
fixes (`STATUS_ALLERGY` + `_apply_allergy_nonstandard_exact_override()`)
live. Two separate numbers came out of this, and they tell different parts
of the story:

**AUTO-tier precision on this 6-note population: 43/52 clean-span = 82.7%**
(up from 78.3% on the earlier, larger, differently-sampled 155-entity run --
not a strict apples-to-apples comparison, different notes/entities, but
directionally consistent with a real improvement, not a regression). 9
clean-span misses inspected -- all distinct-vocabulary-duplicate-style
Anatomy/Lab-Test/Condition mismatches (e.g. 'STEMI'->'ST elevation' vs.
gold's own STEMI code, 'Abdomen'->generic vs. gold's more specific code),
same shape as the 'gunshot wound'/'blurred vision' class already flagged as
Recommended-next-step #3 below, not evidence of new problems from this
session's fixes.

**But zero of the 19 allergy entities reached AUTO tier at all** -- graded
their retrieval-layer top candidate directly against gold instead (bypassing
the tier gate, since none were gated AUTO): of 16 re-extracted this run (3
were not re-extracted, GLiNER non-determinism, unrelated to this fix):

```
6/9 gradable = 66.7% -- exact match via the override:
  aspirin, fluconazole, morphine, trazodone, prochlorperazine (both aspirin
  mentions) all resolve to the EXACT gold SNOMED code
  ('Allergy to aspirin'=293586001, 'Allergy to morphine'=293601001, etc.)
3/9 wrong -- NSAIDS, Penicillins, Phenergan resolve to a semantically-related
  but non-matching Allergy concept (different SNOMED extension code, or a
  dense-retrieval semantic neighbor when the exact override found no hit)
7/16 no_candidates -- brand/combination names ('Reglan', 'Elavil',
  'Tylenol-Codeine', 'Spiriva with HandiHaler', 'Abacavir Sulfate',
  'levetiracetam', 'lisinopril') where no exact "Allergy to {name}" SNOMED
  string exists and the semantic/hybrid Tier 3 fallback also comes up empty
```

**This is a real, verified fix at the retrieval layer** -- morphine and
trazodone, the two originally-diagnosed failures, are now both exact
matches, and the fix generalizes cleanly to 4 more entities in this batch.
**It does not yet show up as an AUTO-tier precision gain**, because every
one of these 16 entities routes to TIER_4_ENSEMBLE_SPLIT or
TIER_5_TRUE_AMBIGUITY (HITL_REQUIRED), never TIER_1. For the 7 cases where
the top candidate is exactly correct, the MoLLM ensemble still splits votes
rather than unanimously accepting Candidate #1 -- plausibly because the
gate's prompt still frames the entity by its original Medication label/text,
not the section-driven "this is an allergy, not a current medication"
reframing the retrieval fix applies internally, so models may be relitigating
whether "Allergy to aspirin" is a correct read of a bare "Aspirin" mention
without that context. Not investigated further this session -- flagged as a
concrete next step below rather than guessed at.

## Update, same session: the ensemble-split root cause fixed, with a real caveat

Traced the "0/19 reach AUTO tier" finding above to its root cause by reading
the actual stored `mollm_tier_gate_decisions.models` trail JSON for
Aspirin/fluconazole/morphine, rather than guessing. Both prompts already
carried `assertion_status` -- the actual defect was narrower: Step A gave no
guidance on what an ALLERGY assertion means (phi4-mini's Step A output for
'morphine' never mentioned allergy at all: "Morphine is an opioid medication
used for pain relief"), and Step B's rule 3 ("ignore assertion/negation
status when judging the CONCEPT match") is correct for negation -- a denied
entity still names the same concept -- but wrong for ALLERGY, where the
correct concept genuinely IS a different one (the allergic-disposition
finding, not the substance). Rule 3 as written pushed models toward
rejecting the exactly-correct "Allergy to X" candidate under rule 4 ("reject
a distinct concept").

**Fix** (`src/mollm_tier_gate.py`): an `ALLERGY_MEANING_INSTRUCTION` added to
Step A's prompt when `assertion_status == "ALLERGY"`, explicitly stating the
entity represents a documented allergy, not current medication use; and an
`ALLERGY_CONTEXT_CLAUSE` added to Step B's rules (all three models, not just
qwen) carving out the allergy exception to rule 3. 4 new unit tests
(`tests/test_tier_gate.py`), full suite 50/50.

**Micro-test on Note 1 confirmed the hypothesis immediately**: Aspirin,
fluconazole, morphine all moved TIER_4_ENSEMBLE_SPLIT -> TIER_1_AUTO_VALIDATED
in one re-run, each writing the exact correct concept. Extended to the other
5 notes: 5 more moved off HITL (trazodone, a second aspirin mention,
Prochlorperazine, NSAIDS, and 'Penicillins' in `17739994-DS-31` -- the last
jumping straight from TIER_5 to TIER_1).

**The honest final number, graded against gold: 6/8 = 75% precision on the
allergy entities that now reach AUTO tier** (Aspirin x2, fluconazole,
morphine, trazodone, Prochlorperazine all exactly correct; NSAIDS and
'Penicillins' wrong). Both wrong cases were checked directly against
`athena_concept` and are a **genuine SNOMED near-duplicate concept pair**,
the same class already flagged above ('gunshot wound'/'blurred vision'):

```
NSAIDS:      chosen 'Allergic reaction caused by nonsteroidal antiinflammatory
             agent' (Condition/Disorder) vs gold 'Allergy to non-steroidal
             anti-inflammatory agent' (Observation/Clinical Finding)
Penicillins: chosen 'Allergic reaction caused by penicillin'
             (Condition/Disorder) vs gold 'Allergy to penicillin'
             (Observation/Clinical Finding)
```

Both pairs are standard SNOMED concepts describing the same clinical fact
under two different concept classes. This is not a defect in the new prompt
clause -- the clause explicitly asks "does this candidate name an allergy/
reaction concept for this same substance," and both chosen candidates
correctly do. The gap is upstream, in which of two near-duplicate SNOMED
concepts retrieval ranks first, and the ensemble-split fix's real effect is
that once ensemble votes stop splitting, this class of ambiguity now writes
to AUTO tier instead of stalling at HITL. **This is a genuine, if narrow,
new false-positive-risk path introduced by fixing the split-vote problem**,
not free -- worth weighing directly, not glossed over.

**Combined with the 8/16 already-processed allergy entities' final states**:
6 correct AUTO, 2 wrong AUTO, 1 still TIER_4 (Phenergan -- its top candidate,
'Allergy to ergoline derivative', is itself wrong, so it correctly still
contests), 8 TIER_5 no_candidates (brand/combo names the exact override
can't reach), 3 not re-extracted this run (GLiNER non-determinism, tracked
separately).

## Recommended next steps (not done this session)

1. ~~Complete the in-progress re-run...~~ DONE above.
2. Optimize the Lab Value Suffix Fallback's nested normalize_entity() calls
   (see above) -- logged as a real performance issue, explicitly deferred.
3. **Now higher-priority, not just a grading-harness nicety.** Check
   `athena_concept_relationship` for a formal duplicate-concept marker to
   distinguish genuine vocabulary duplicates from real errors. This started
   as a grading-methodology question ('gunshot wound'/'blurred vision',
   'STEMI'/'Abdomen') but the ensemble-split fix below just turned it into a
   live false-positive-risk path: NSAIDS and 'Penicillins' now AUTO-validate
   to the wrong member of a genuine SNOMED near-duplicate pair
   (Condition/Disorder 'Allergic reaction caused by X' vs Observation/
   Clinical Finding 'Allergy to X'). A concrete fix candidate worth
   evaluating: prefer the Observation/Clinical Finding member over the
   Condition/Disorder member when multiple near-identical allergy concepts
   tie, rather than relying on retrieval's un-tie-broken rank-1 pick.
4. ~~Investigate why the MoLLM ensemble splits votes...~~ DONE: root-caused
   (Step B rule 3 told models to ignore assertion status, correct for
   negation but wrong for ALLERGY) and fixed
   (`ALLERGY_MEANING_INSTRUCTION`/`ALLERGY_CONTEXT_CLAUSE`,
   `src/mollm_tier_gate.py`). 6/8 of the entities this unstuck are correct;
   2/8 are the SNOMED duplicate-pair issue in (3) above -- not a clean win,
   a real trade needing that follow-up before considering `dry_run=False`.
5. Consider a broader (not just exact-match) non-standard-concept fallback
   for the 7/16 "no_candidates" brand/combination-name allergy entities --
   the current override requires an exact string match on the synthesized
   "Allergy to {text}" pattern, which brand names and multi-drug combos
   never satisfy.
6. Re-run this same shadow-run methodology on a larger sample once (2) and
   (3) are addressed, to measure whether precision recovers at scale.
