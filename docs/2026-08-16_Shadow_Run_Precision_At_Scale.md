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

## Recommended next steps (not done this session)

1. Investigate the allergy-context mishandling pattern specifically:
   check how many Medication-labeled entities in this corpus fall inside an
   Allergies/Adverse Reactions section, and whether `src/assertion.py`'s
   existing section-based priors (`SECTION_EXPERIENCER_OVERRIDE`/
   `SECTION_TEMPORALITY_OVERRIDE`, `preprocessing.py`) already cover this
   case or need extending -- this looks like exactly the kind of section
   context that machinery exists to handle, worth checking before building
   something new.
2. Check `athena_concept_relationship` for a formal duplicate-concept
   marker to distinguish genuine vocabulary duplicates (safe to treat as
   correct either way) from real errors, rather than eyeballing name
   similarity as done here.
3. Re-run this same shadow-run methodology on a larger sample once (1) is
   addressed, to measure whether precision recovers.
