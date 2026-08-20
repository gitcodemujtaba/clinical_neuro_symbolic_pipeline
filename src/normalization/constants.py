"""src/normalization/constants.py — module-wide config constants (split from src/normalization.py, 2026-08-14)."""
import os
import re

# `from .constants import *` (used by every other submodule in this package)
# skips underscore-prefixed names by default -- __all__ overrides that, since
# some of these (_ALNUM_MIX_RE) are genuinely used outside this file.
__all__ = [
    "PROJECT_DIR", "DB_PATH", "GLINER_LABEL_TO_DOMAIN", "VOCAB_BY_LABEL", "DEFAULT_VOCAB",
    "DOMAIN_TO_GLINER_LABEL", "TIER3_SIMILARITY_FLOOR", "CANDIDATE_LIMIT", "TIER3_AMBIGUITY_MARGIN",
    "RANKED_TIER12", "TIER12_RANK_SEMANTIC", "TIER12_CLASS_PREFERENCE", "TIER12_CLASS_DEMOTED",
    "TIER12_TIE_EPSILON", "HIGH_GLINER_RISK_FLOOR", "SHORT_TOKEN_MAX_LEN", "ISUPPER_MAX_LEN",
    "_ALNUM_MIX_RE", "FUZZY_MAX_EDIT_DISTANCE", "FUZZY_MIN_TEXT_LENGTH",
    "CONTEXTUAL_CANDIDATES_ENABLED",
]

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"


DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")


# DOMAIN FILTERING: docs/Databases.md's "Domain Filtering" open item calls for
# vector-search fallbacks to be domain-filtered by mapping GLiNER labels to
# OMOP domain IDs, to reduce cross-domain mismatches. Applied to ALL THREE
# tiers, not just Tier 3 -- the `ED` -> `Ed District` collision that motivated
# this was an exact-SYNONYM match at Tier 2, not a vector fallback, so gating
# only Tier 3 would have left the observed failure in place. A label with no
# entry gets no domain filter (safe default: matches pre-fix behavior).
GLINER_LABEL_TO_DOMAIN = {
    "Condition": ["Condition"],
    "Symptom": ["Condition", "Observation"],
    "Medication": ["Drug"],
    "Procedure": ["Procedure"],
    "Anatomy": ["Spec Anatomic Site"],
    "Lab Test": ["Measurement"],
    # "Qualifier" is never produced by GLiNER itself (not in CLINICAL_LABELS
    # in src/entity_extraction.py) -- it exists only as a relabeling target
    # for compound-split/span-growth halves whose resolved concept is a
    # standalone modifier (laterality words like "Left"/"Right" being the
    # concrete case that motivated this; see docs/Stage2_Compound_And_
    # Qualifier_Gaps.md gap 3). Both listed domain strings are best-effort
    # guesses at where SNOMED qualifier-value concepts land in this OMOP
    # dump -- NOT verified against the real vocabulary, since this sandbox
    # has no DB access. Getting the guess wrong here is harmless: callers
    # that already know the correct domain (the split/growth detectors) pass
    # it explicitly via normalize_entity()'s domain_override instead of
    # relying on this dict -- see that parameter's docstring for why this
    # entry existing or not doesn't change correctness, only the label's
    # display friendliness on a fallback path.
    "Qualifier": ["Qualifier Value", "Meas Value"],
}


# 2026-08-18 ("cold start" fix, paired with src.mollm_tier_gate's
# EXHAUSTIVE_CANDIDATE_EVAL_ENABLED -- same env var, must be enabled
# together, see that flag's own comment for why). Sized directly against
# this corpus: of 4,580 wrong-concept cases (span found, SNOMED code wrong)
# across the 109-note test corpus, 41 share an IDENTICAL concept_name with
# gold's own answer but a different code and domain -- e.g. "wound
# dehiscence" resolving to 225553008 (Condition/Disorder) when gold wants
# 410723003 (Observation/Morphologic Abnormality). 11 of those 14
# domain-mismatch pairs point the SAME direction: gold prefers the
# Observation/Morph-Abnormality sense, we pick the Condition/Disorder
# sibling, because GLINER_LABEL_TO_DOMAIN["Condition"] = ["Condition"]
# above excludes the Observation-domain concept from ever reaching Stage 2b's
# candidate pool -- Stage 3 (MoLLM) never gets a chance to judge it
# contextually because it never sees it at all. Widening the pool alone is
# NOT safe without also fixing route_tier()'s "stop at first accept" Step B
# behavior (see EXHAUSTIVE_CANDIDATE_EVAL_ENABLED) -- otherwise a bigger
# candidate list just gives the arbitrary-pick problem more room, not less.
# 2026-08-18: promoted to default-ON (opt-out, not opt-in) after the
# wound-dehiscence case above was traced end-to-end and confirmed this exact
# flag was the fix -- it had been sitting off-by-default since 2026-08-18,
# meaning every note processed without manually exporting
# CNSP_CONTEXTUAL_CANDIDATES silently kept the Condition-only pool. Set
# CNSP_CONTEXTUAL_CANDIDATES=0 (or "false"/"no") to opt back out.
CONTEXTUAL_CANDIDATES_ENABLED = os.environ.get(
    "CNSP_CONTEXTUAL_CANDIDATES", "1").strip().lower() not in ("0", "false", "no")

if CONTEXTUAL_CANDIDATES_ENABLED:
    GLINER_LABEL_TO_DOMAIN["Condition"] = ["Condition", "Observation"]



# VOCABULARY RESTRICTION: previously none of the queries filtered on
# vocabulary_id, so an entity could match a concept from ANY vocabulary in
# athena_concept (ICD10CM, LOINC, RxNorm...) rather than SNOMED, which this KG
# is otherwise built around. OMOP codes medications via RxNorm rather than
# SNOMED's Substance hierarchy, so Medication is the one exception.
VOCAB_BY_LABEL = {
    "Medication": ["RxNorm", "RxNorm Extension"],
}


DEFAULT_VOCAB = ["SNOMED"]



# Reverse of GLINER_LABEL_TO_DOMAIN (defined above), used by
# src/clinical_pipeline.py's compound-span splitter (and now span-growth) to
# relabel a split/grown entity for DISPLAY/storage purposes -- e.g. "abdomen"
# split off "gunshot wound to abdomen" (parent label Condition) resolves to
# a Spec Anatomic Site concept, so it is stored as entity_label="Anatomy"
# rather than the misleading inherited "Condition". Not a semantic
# classifier, just "which label is honest about this domain".
#
# NOTE ON CORRECTNESS vs DISPLAY (2026-08-10, gap 3 fix). This dict used to
# ALSO be the mechanism that kept Stage 2b's domain restriction from
# excluding the very candidate the split detector already confirmed --
# relabeling "Left" as "Qualifier" only helped if "Qualifier" then had a
# correct, verified domain entry, which it doesn't (see the "Qualifier"
# comment on GLINER_LABEL_TO_DOMAIN above). That correctness job now belongs
# to normalize_entity()'s domain_override parameter, which
# src/clinical_pipeline.py populates directly from the domain the detector
# found -- no dictionary lookup, no guessing. This dict is now cosmetic: a
# domain with no entry here just keeps the parent's original label (as
# before), which is a harmless, purely-informational fallback rather than a
# correctness bug, because domain_override no longer depends on it.
DOMAIN_TO_GLINER_LABEL = {
    "Condition": "Condition",
    "Procedure": "Procedure",
    "Spec Anatomic Site": "Anatomy",
    "Measurement": "Lab Test",
    "Observation": "Symptom",
    "Drug": "Medication",
    "Drug Exposure": "Medication",
    "Qualifier Value": "Qualifier",
    "Meas Value": "Qualifier",
}



# Documented in docs/Implementation_Methodology.md but previously not
# implemented at all -- normalize_entity() always returned top-1 regardless of
# score, so "0 (Failed)" was effectively unreachable and confidently wrong
# matches (`lasix` -> `Laslades`, `bioplar` -> `Bourgvilain`) flowed through as
# if fully resolved.
TIER3_SIMILARITY_FLOOR = 0.72



# Stage 3 routing parameters. All three are calibration targets against the
# validation slice (docs/Evaluation_Criteria.md) rather than settled values --
# see docs/MoLLM_Stage3_Retrieval_Design.md S8.
#
# 2026-08-16 (plan Phase 3): bumped 3 -> 5, matching the blueprint's "Top-5
# candidate pool >98% of the time" target. This is a real behavior change,
# not a cosmetic one -- it changes what every existing normalized_entities
# row's `candidates` list looks like going forward, so any comparison against
# rows written before this change should account for the differing list
# length (same caution RANKED_TIER12's own docstring gives for the tiebreak
# change below). Downstream consumers were checked before this change:
# src/mollm_ensemble.py's build_prompt() and src/mollm_tier_gate.py's Step B
# both already iterate `range(1, len(candidates) + 1)` dynamically rather
# than assuming exactly 3, so neither needed a code change for this bump.
CANDIDATE_LIMIT = 5


TIER3_AMBIGUITY_MARGIN = 0.05



# ==========================================================================
# TIER 1/2 RANKING (2026-08-13, docs/2026-08-13_Code_Improvement_Proposals.md
# P1.1). DEFAULT OFF -- see RANKED_TIER12 below.
# ==========================================================================
#
# THE PROBLEM. Every Tier 1/2 lookup in this module ends `ORDER BY concept_id
# ASC LIMIT n`. When a clinical string legitimately matches several distinct
# standard SNOMED concepts, the winner is therefore the one Athena happened to
# assign the lowest integer id -- a property of the vocabulary's build process,
# not of medicine. evaluation/stage2b_cal_eval.py measured the consequence
# (2026-08-13 report S5.3):
#
#     Tier 1 (Exact)   776 gradable   52.71% accurate
#     Tier 2 (Synonym) 532 gradable   61.84% accurate
#
# "Exact" performing WORSE than "Synonym" is the signature of an arbitrary
# tiebreak rather than a retrieval failure: the exact-match set is larger and
# more collision-prone (short, common strings like "Left" and "Initial" match
# many qualifier concepts), so the arbitrary pick is wrong more often there.
# The same report's cross-check found note 10371195-DS-9 at Tier 1 9/28 (32.1%)
# vs Tier 2 16/19 (84.2%) -- the same pattern in a single note.
#
# WHY THIS IS THE HIGHEST-VALUE FIX IN THE PIPELINE. It is upstream of every
# confidence signal. Roughly 47% of exact-matched entities are handed the wrong
# concept BEFORE any tier, threshold or ensemble sees them; no amount of
# downstream calibration recovers a link that was mis-selected here.
#
# WHY IT IS DEFAULT OFF. Turning it on changes what Stage 2b returns, which
# invalidates every mollm_decisions row already recorded against the old
# candidate lists (report S6's drift confound -- now detectable via
# src/provenance.py's candidates_hash, but still a real invalidation). Run the
# A/B on the validation slice first:
#
#     python3 evaluation/stage2b_cal_eval.py --per-candidate --split val
#         (read the ranking HEADROOM: oracle minus top-1 accuracy. That number
#          is the ceiling this ranker can reach. If it is near zero, retrieval
#          is the bug and this will not help.)
#     CNSP_RANKED_TIER12=1 python3 scripts/test_pipeline_e2e.py ...
#     python3 evaluation/stage2b_cal_eval.py --split val   (compare tiers)
#
# then flip the default here once measured.
RANKED_TIER12 = os.environ.get("CNSP_RANKED_TIER12", "").strip() in ("1", "true", "yes")



# SEMANTIC CRITERION -- SEPARATELY GATED, DEFAULT OFF.
#
# The ranker's third criterion is SapBERT cosine between the entity text and
# each candidate concept name. It is the only criterion that costs a model
# call, and it breaks the property _lookup_tier12()'s docstring explicitly
# defends: "Tier 1/2 ... keeps this cheap enough to try at every candidate
# boundary (no embedding call per attempt)". Compound-split partition search
# calls into Tier 1/2 repeatedly per entity, so enabling this is not a fixed
# cost -- it scales with the number of boundaries tried.
#
# Default OFF means the ranker ships as concept_class -> domain -> specificity
# -> concept_id: pure SQL and dict lookups, no embeddings, no latency change,
# and every ordering decision traceable to a vocabulary field a human can
# inspect. Criteria 1 and 2 are also the ones with the clearest mechanism for
# the measured failure (the "Left"/"Initial" Qualifier Value collisions), so
# the cheap half is where the expected gain is concentrated.
#
# Turn it on to measure whether semantics adds anything ON TOP of the metadata
# ordering -- that is a separate A/B from the ranker itself, and running both
# changes at once would make the result uninterpretable:
#
#     CNSP_RANKED_TIER12=1 CNSP_TIER12_SEMANTIC=1 python3 ...
#
# tier12_rank_basis records which of the two ran ("ranked_v1" vs
# "ranked_v1_semantic"), so the comparison is gradable per row.
TIER12_RANK_SEMANTIC = os.environ.get(
    "CNSP_TIER12_SEMANTIC", "").strip() in ("1", "true", "yes")



# Preferred SNOMED concept_class_id per GLiNER label, best first. A clinical
# note's "fracture" is a Clinical Finding, not the Qualifier Value or the
# Staging/Scales concept that share the string; ordering by class is the single
# most reliable non-arbitrary signal available at Tier 1/2 because it needs no
# embedding call and no hierarchy join. Classes not listed rank after every
# listed one but ahead of the explicit demotions in TIER12_CLASS_DEMOTED.
TIER12_CLASS_PREFERENCE = {
    "Condition": ["Clinical Finding", "Disorder", "Event"],
    "Symptom": ["Clinical Finding", "Disorder", "Observable Entity"],
    "Procedure": ["Procedure", "Regime/Therapy"],
    "Anatomy": ["Body Structure", "Anatomical Structure", "Morph Abnormality"],
    "Lab Test": ["Procedure", "Observable Entity", "Measurement"],
    "Medication": ["Ingredient", "Clinical Drug", "Branded Drug", "Substance"],
}



# Classes that are almost never what a clinical mention means, but that collide
# heavily on short strings. Demoted, NOT excluded -- "Left" genuinely IS a
# Qualifier Value when the extracted span really is a bare laterality token,
# and a hard exclusion would turn a wrong link into a failed one, which is not
# obviously an improvement. Demotion lets a better-classed candidate win when
# one exists and changes nothing when none does.
TIER12_CLASS_DEMOTED = {
    "Qualifier Value", "Navigational Concept", "Staging / Scales",
    "Linkage Concept", "Attribute", "Record Artifact", "Context-dependent",
}



# Two candidates whose SapBERT similarity to the entity's own text differ by
# less than this, after class and domain agreement, are treated as a genuine
# unresolved tie: is_ambiguous=True, both forwarded, Stage 3 decides. This is
# the OTHER half of the fix and arguably the more important one -- the existing
# `multiple_exact_concept_name_matches` ambiguity flag fires on len(rows) > 1
# BEFORE any ranking, so today Stage 3 is flooded with easy multi-match cases
# while the genuinely undecidable ones are silently resolved by integer id.
# After ranking, ambiguity means "ranking could not separate them", which is
# the question Stage 3 is actually good at.
TIER12_TIE_EPSILON = 0.02



# 2026-08-11 REPLACES GLINER_CONFIDENCE_FLOOR (was 0.60, flagged LOW when
# confidence fell BELOW the floor). evaluation/stage_calibration.py's
# per-stage ECE measurement on the 25-note test split found the GLiNER
# confidence-vs-accuracy relationship INVERTED for this model/domain: higher
# reported confidence correlated with LOWER extraction accuracy, not higher.
# A "flag LOW confidence as risky" rule was therefore steering Stage 3
# attention in exactly the wrong direction -- protecting the extractions the
# calibration data says are actually MORE likely to be wrong, and starving
# attention from the ones the model was overconfident about. This flips the
# polarity: HIGH reported confidence is now what raises the flag.
HIGH_GLINER_RISK_FLOOR = 0.70



# Short-token / abbreviation heuristics (2026-08-11, measured via
# scripts/measure_heuristic_and_boundary.py against the 25-note test split
# BEFORE being wired in here -- see docs/Stage2_Improvement_Areas_Technical_
# Brief.md's second-opinion addendum). Three independent signals, each
# catching a different failure shape:
#   * SHORT_TOKEN_MAX_LEN -- bare short tokens/abbreviations (Abd, NAD, R).
#   * ISUPPER_MAX_LEN     -- longer acronyms (HEENT, NSTEMI, FODMAP) while
#                            excluding all-caps narrative section headers
#                            (SOCIAL HISTORY, DISCHARGE MEDS). Measured:
#                            isupper() with NO length cap hit 28.93% of the
#                            corpus (5.8x over a 5% routing budget); of that,
#                            10.3 points came from isupper()-but-not-short
#                            tokens (the header risk), while the len<=8
#                            slice stayed within budget and was almost
#                            entirely genuine abbreviations.
#   * _ALNUM_MIX_RE        -- delimiter-less name+value gluing (HCO3-22,
#                            pT2), measured at 5.04% of the corpus.
SHORT_TOKEN_MAX_LEN = 4


ISUPPER_MAX_LEN = 8


_ALNUM_MIX_RE = re.compile(r"[A-Za-z][0-9]|[0-9][A-Za-z]")



# Fuzzy/typo fallback (2026-08-11, docs/Stage3_Open_Issues.md Issue 3).
# `spirnolactone` (a misspelling of spironolactone) embedded closer to
# SPIRAPRILAT and SPIRILENE than to the correctly-spelled concept under
# Tier 3's semantic search -- the true concept never made the candidate list
# at all, so both MoLLM models picked the closest-SPELLED wrong drug from a
# candidate set that never contained the right one. Edit distance catches
# what embedding similarity missed: "spirnolactone" is one character from
# "spironolactone". Deliberately not its own tier -- edit distance on short
# clinical tokens and abbreviations is noisy, so this only SUPPLEMENTS an
# already-uncertain Tier 3 result (below-floor or margin-ambiguous) rather
# than ever driving normalization on its own or overriding a confident match.
FUZZY_MAX_EDIT_DISTANCE = 2


FUZZY_MIN_TEXT_LENGTH = 5  # below this, a 2-edit budget is most of the string



