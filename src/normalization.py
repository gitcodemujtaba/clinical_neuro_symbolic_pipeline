"""
src/normalization.py — Stage 2b grounding & normalization.

2026-08-07 added the domain/vocabulary restrictions and the Tier-3 similarity
floor documented below.

2026-08-08 adds the Stage 3 prerequisites from
docs/MoLLM_Stage3_Retrieval_Design.md S3 and
docs/Stage1_2_Completeness_Audit.md:

  * TOP-3 CANDIDATES INSTEAD OF LIMIT 1. Stage 3 does two different jobs
    depending on how confident Stage 2b was. When the match is clean it only
    needs to answer "does this contradict a guideline?", and extra candidates
    are wasted context. When the match is ambiguous it is genuinely
    disambiguating, and it cannot do that if Stage 2b already discarded the
    alternatives. So every tier now retrieves 3 and the caller forwards 1 or
    all 3 depending on the ambiguity gate below.

  * DETERMINISM FIX (pre-existing correctness bug, docs/Proposal_Alignment_Review.md
    S7). No tier had an ORDER BY, just LIMIT 1. When multiple OMOP rows share a
    lowercased concept_name -- routine in a multi-vocabulary Athena dump --
    DuckDB's row order under parallel/vectorized execution is not stable, so
    two runs over the same note could return different concepts (confirmed:
    `ED` resolved to `Ed District` on one run and `Erectile dysfunction` on the
    next). That directly contradicts the proposal's deterministic/traceable
    claim, since the same note could produce two different provenance chains
    depending on when it was run. Every tier now orders explicitly and breaks
    ties on the lowest concept_id.

  * AMBIGUITY DETECTION. The old LIMIT 1 didn't just pick arbitrarily, it
    concealed that there was anything to pick between. A Tier-1 lookup
    returning three distinct standard concepts for the same string is exactly
    the case Stage 3 exists to resolve, and it was previously indistinguishable
    from a clean single match.

  * confidence_tier_in. Computed here because this is the first point where
    all three uncertainty signals are available at once.

2026-08-11 adds a fuzzy/edit-distance supplement to Tier 3 (docs/
Stage3_Open_Issues.md Issue 3): when Tier 3's result is already uncertain
(below TIER3_SIMILARITY_FLOOR or margin-ambiguous between the top two), a
levenshtein()-based lookup is merged into the candidate list. Motivated by
`spirnolactone` (a misspelling of spironolactone) resolving to three
semantically-close but pharmacologically wrong candidates -- the correctly-
spelled concept never appeared under semantic search at all, so Stage 3 had
no way to pick it even in its deeper-resolution mode. Additive only: never
runs on a confident match, never overrides Tier 1-3, only widens an
already-ambiguous candidate list.
"""

import os
import re
import json
import math
import itertools
import duckdb
import torch
from transformers import AutoTokenizer, AutoModel
import warnings

from src.provenance import (
    provenance_alter_statements,
    provenance_column_sql,
    provenance_params,
    provenance_placeholders,
)

warnings.filterwarnings("ignore")

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

print("Loading SapBERT model for vector normalization... (this may take a moment)")
MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
SAPBERT_POOLING = "cls"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sapbert = AutoModel.from_pretrained(MODEL_NAME)

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
CANDIDATE_LIMIT = 3
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


_VOCAB_RELEASE = None


def get_athena_vocabulary_release(conn) -> str:
    """Identifies which Athena vocabulary release produced a mapping.

    docs/Provenance_Schema.md Stage 2b specifies this field and it was never
    written. It is not bookkeeping: SNOMED concept IDs are stable across
    releases but concept NAMES, standard_concept flags and the ancestor closure
    are not, so a mapping produced against one release is not guaranteed
    reproducible against another. Without the stamp there is no way to tell,
    after the fact, whether two runs disagreed because the code changed or
    because the vocabulary did -- which is exactly the ambiguity that made the
    Stage 2b non-determinism bug hard to characterise.

    Prefers a real `vocabulary` table if the Athena dump included one. Falls
    back to a CONTENT SIGNATURE (row count + latest valid_start_date over
    SNOMED) rather than a load timestamp: a timestamp records when the file was
    read, not what was in it, and would make two different vocabularies look
    identical if loaded at the same moment.
    """
    global _VOCAB_RELEASE
    if _VOCAB_RELEASE is not None:
        return _VOCAB_RELEASE

    release = "unknown"
    try:
        tables = {t[0].lower() for t in conn.sql("SHOW TABLES").fetchall()}
        for candidate in ("athena_vocabulary", "vocabulary"):
            if candidate in tables:
                row = conn.sql(
                    f"SELECT vocabulary_version FROM {candidate} "
                    "WHERE vocabulary_id = 'SNOMED' LIMIT 1"
                ).fetchone()
                if row and row[0]:
                    _VOCAB_RELEASE = str(row[0])
                    return _VOCAB_RELEASE
    except Exception:
        pass

    try:
        n, latest = conn.sql("""
            SELECT count(*), max(valid_start_date)
            FROM athena_concept WHERE vocabulary_id = 'SNOMED'
        """).fetchone()
        release = f"signature:snomed_n={n},latest_valid_start={latest}"
    except Exception:
        pass

    _VOCAB_RELEASE = release
    return release


def _in_clause(values):
    return ",".join(["?"] * len(values))


# ==========================================================================
# COMPOUND-SPAN SPLITTING
#
# 2026-08-10, built after scripts/score_gold_recall.py's compound-span
# detection measured this concretely on 17751158-DS-19: GLiNER extracts
# "gunshot wound to abdomen" as ONE Procedure/Condition entity, but the gold
# set links "gunshot wound" (56768003) and "abdomen" (818983003) as TWO
# separate SNOMED annotations. A one-entity-one-concept design can only ever
# satisfy one of them, no matter how good Stage 2b/3 get -- this is the
# fix: give Stage 2a a way to emit two atomic entities instead of one
# compound one, when there's good evidence the text actually carries two
# concepts.
#
# find_compound_split() is the detector; src/clinical_pipeline.py owns
# turning a positive result into new entity_id-bearing rows (offset
# recomputation needs map_offsets_to_original() and the note's abbreviation
# expansion log, neither of which this module has -- see clinical_pipeline's
# docstring for that step).
# ==========================================================================

# Stripped from the ends of a candidate split before it is looked up, so a
# split like "gunshot wound" + "to abdomen" resolves as "abdomen" (a real
# SNOMED body-structure name) rather than failing on the bare preposition.
# Deliberately NOT stripped from the middle -- "left renal" and "left lobe"
# must survive intact, and no connector word sits inside either.
_CONNECTOR_WORDS = {
    "to", "of", "in", "at", "on", "or", "and", "with", "without",
    "the", "a", "an",
}
_TOKEN_RE = re.compile(r"\S+")

# ==========================================================================
# 2026-08-11 NEWLINE COMPOUND SPANS -- CANDIDATE ARBITRATION, NOT A NEW
# SPLITTER. scripts/measure_heuristic_and_boundary.py confirmed
# crosses_sentence_boundary=False for all five embedded-newline compound
# spans measured in the 25-note run (PyRuSH treats an internal '\n' as
# line-wrap formatting, not a sentence boundary -- it will not split these).
#
# The design question was whether to add a new "split on \n" rule. Rejected:
# the SAME 5-example sample contains a counter-case -- 'left \nlung base
# \natelectasis' overlaps gold as 'left \nlung base' (ONE concept, 60859004)
# plus 'atelectasis' -- so a blind newline split would fragment a correct
# concept exactly like the case it was meant to fix. Splitting on
# punctuation/whitespace SHAPE rather than resolvability was the same
# mistake this function's own whole-phrase guard already exists to prevent
# for "and"/"or".
#
# find_compound_split() (below) already IS the correct mechanism: it
# tokenizes on any whitespace (embedded newlines included, since a newline
# is whitespace to _TOKEN_RE) and requires every candidate part to
# independently clear Tier 1/2 before accepting a split, so the ontology --
# not a hand-written newline rule -- decides where the real boundary is. It
# was already being called on every entity (src/clinical_pipeline.py's
# split_compound_entities() has no crosses_sentence_boundary gate). What
# actually blocked it on 4 of the 5 measured cases:
#   * 'L \nclavicular fx' and 'L L2 \ntransverse process fx' -- blocked by
#     _MIN_SPLIT_HALF_CHARS excluding the single-letter part 'L', despite
#     'L' being a real, gold-confirmed SNOMED concept (720617006). Fixed via
#     _SHORT_LATERALITY_TOKENS just below.
#   * 'left acetabular and iliac crest \nfracture' -- already reachable by
#     the EXISTING conjunction-trimming logic (no newline-specific code
#     needed); whether it actually splits depends on 'iliac crest fracture'
#     having an exact Tier 1/2 string match in this OMOP dump, a vocabulary-
#     coverage question this change does not by itself resolve.
#   * 'left \nlung base \natelectasis' -- same story: the 2-way partition
#     ("left lung base" + "atelectasis") is already tried by the existing
#     search; whether it fires depends on "left lung base" resolving at
#     Tier 1/2 exactly, not on any newline-specific gap.
#   * 'S1S2\nAbd' -- NOT fixed by this or any whitespace-tokenizing
#     splitter. "S1S2" glues two sub-tokens ("S1", "S2") with no whitespace
#     between them at all, so there is no token boundary to cut at. This is
#     a distinct, unaddressed gap (sub-token fusion, closer in kind to
#     strip_lab_value_suffix()'s problem than to compound-splitting) and is
#     left as a known limitation, not silently declared solved.
# ==========================================================================

# A split part must be at least this long, so a stray single letter or a bare
# punctuation fragment left over after connector-trimming can never itself
# become a "concept" -- there is no legitimate 1-2 character SNOMED FSN this
# would need to match.
#
# 2026-08-11 EXCEPTION, NOT A REVISION OF THE RULE. That "no legitimate 1-2
# character SNOMED FSN" claim was empirically falsified by gold data before
# this exception existed: scripts/score_gold_recall.py's 25-note run links
# the bare single-letter span 'L' to a real, standalone SNOMED concept
# (720617006) in note 10371195-DS-9 ('L \nclavicular fx' -> gold wants 'L'
# + 'clavicular fx' as TWO separate concepts), and the whitespace-only
# tokenizer already treats 'L' as its own token there (the embedded '\n' is
# whitespace like any other) -- this function was already ARCHITECTURALLY
# capable of finding that split via the same Tier-1/2-required-per-part
# arbitration every other compound-split case uses; _MIN_SPLIT_HALF_CHARS
# was the only thing blocking it. Rather than lowering the general floor
# (which WOULD reopen the collision risk the floor exists to prevent -- see
# find_compound_split()'s REVERT NOTICE for a measured case of exactly that
# kind of over-eager matching), _SHORT_LATERALITY_TOKENS is a narrow,
# evidence-backed whitelist: laterality/bilaterality shorthand is a closed,
# well-known set in clinical text, not an open-ended source of accidental
# short-string collisions.
_MIN_SPLIT_HALF_CHARS = 3
_SHORT_LATERALITY_TOKENS = {"l", "r", "b"}  # left / right / bilateral

# 2026-08-10, gap 1 (docs/Stage2_Compound_And_Qualifier_Gaps.md): the large
# gold note's laterality+device+action procedures ("right EVD placement" ->
# 'right' + 'EVD' + 'placement') need a 3-way split, which the original
# binary-only splitter structurally cannot produce. 4 is headroom past the
# largest real case measured so far (3), not an invitation to explore --
# see _partition_token_ranges()'s docstring for why this bound also keeps
# the search cheap.
_MAX_SPLIT_PARTS = 4


def _tokens_with_offsets(text: str) -> list:
    """[(token_text, start, end), ...] for whitespace-delimited tokens,
    offsets relative to `text`."""
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _trim_connectors(tokens: list) -> list:
    """Drops leading/trailing connector-word tokens (see _CONNECTOR_WORDS)
    from a token list. Compares the token stripped of trailing punctuation
    so "vomiting." or "abdomen," still trim correctly at a sentence edge."""
    start, end = 0, len(tokens)
    while start < end and tokens[start][0].strip(".,;:").lower() in _CONNECTOR_WORDS:
        start += 1
    while end > start and tokens[end - 1][0].strip(".,;:").lower() in _CONNECTOR_WORDS:
        end -= 1
    return tokens[start:end]


def _lookup_tier12(conn, text: str, vocabs: list, domains: list = None,
                   include_domains: bool = False, gliner_label: str = None):
    """Tier-1 (exact) then Tier-2 (synonym) lookup ONLY -- deliberately no
    Tier-3 SapBERT fallback and no cosine-similarity guessing.

    WHY NO TIER 3 HERE. A split/growth candidate must resolve with the SAME
    confidence bar as a clean whole-phrase match, or this would be doing
    exactly the thing this entire investigation flagged as the problem: a
    shaky semantic "match" on an arbitrary fragment of a phrase (recall the
    Zosyn->Zodasiran, tylenol->tylosin, SGPT->AST cases). Tier 1/2 only
    keeps false-split/false-growth risk low and keeps this cheap enough to
    try at every candidate boundary (no embedding call per attempt).

    Whitespace (including embedded newlines, e.g. "left \\nbase") is
    collapsed before lookup, since SNOMED concept names never contain one --
    the ORIGINAL text with its real whitespace is preserved separately by
    the caller for offset purposes.

    include_domains=True (2026-08-10, docs/Stage2_Compound_And_Qualifier_
    Gaps.md gap 3 refinement) additionally attaches "domain_ids" to the
    returned candidate: EVERY distinct domain_id among ALL Tier 1/2 matches
    for this text, not just the winning row's single domain_id. WHY THIS
    MATTERS: the top-1 candidate this function normally returns is picked
    via `ORDER BY concept_id ASC LIMIT 1`, which silently collapses
    multiple cross-domain matches into one arbitrary winner -- fine for
    "does this part resolve at all", wrong for domain_override (see
    normalize_entity()'s docstring). Measured concretely: "Left" has
    multiple distinct exact SNOMED concepts across different qualifier
    categories; collapsing to the lowest concept_id's domain meant Stage
    2b's restricted re-lookup only ever saw ONE of them and reported a
    confident-but-frequently-wrong HIGH-tier match instead of flagging the
    genuine ambiguity for Stage 3 to resolve. Only computed when requested
    (one extra query) since callers that just need a yes/no ("is this a
    real concept") -- the whole-phrase guard, part-validation during
    partition search -- don't need it and shouldn't pay for it.
    """
    search_text = re.sub(r"\s+", " ", text).strip().lower()
    if not search_text:
        return None

    # 2026-08-13 (P1.1): with RANKED_TIER12 on, this pulls the top
    # CANDIDATE_LIMIT rows and ranks them instead of trusting the lowest
    # concept_id. With it off, LIMIT 1 as before -- byte-identical behaviour,
    # including the same single-row query cost.
    limit = CANDIDATE_LIMIT if RANKED_TIER12 else 1
    domain_clause = f" AND domain_id IN ({_in_clause(domains)})" if domains else ""
    rows = conn.sql(f"""
        SELECT concept_id, concept_name, domain_id, vocabulary_id
        FROM athena_concept
        WHERE lower(concept_name) = ? AND standard_concept = 'S'
        AND vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause}
        ORDER BY concept_id ASC LIMIT {limit};
    """, params=[search_text, *vocabs, *(domains or [])]).fetchall()
    tier = None
    if rows:
        tier = "1 (Exact)"
    else:
        domain_clause2 = f" AND c.domain_id IN ({_in_clause(domains)})" if domains else ""
        rows = conn.sql(f"""
            SELECT DISTINCT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id
            FROM athena_concept_synonym s
            JOIN athena_concept c ON s.concept_id = c.concept_id
            WHERE lower(s.concept_synonym_name) = ? AND c.standard_concept = 'S'
            AND c.vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause2}
            ORDER BY c.concept_id ASC LIMIT {limit};
        """, params=[search_text, *vocabs, *(domains or [])]).fetchall()
        if rows:
            tier = "2 (Synonym)"

    if not rows:
        return None

    basis = "concept_id_asc"
    if RANKED_TIER12 and len(rows) > 1:
        rows, basis, _tie = _rank_tier12_candidates(
            conn, rows, gliner_label, text, domains=domains)

    candidate = _candidate(rows[0], tier, 1.0)
    candidate["tier12_rank_basis"] = basis
    if include_domains:
        candidate["domain_ids"] = _tier12_domains(conn, search_text, vocabs)
    return candidate


def _tier12_domains(conn, normalized_text: str, vocabs: list) -> list:
    """All DISTINCT domain_ids across EVERY Tier-1 (exact) and Tier-2
    (synonym) match for `normalized_text` (already lowercased/whitespace-
    collapsed by the caller), unrestricted by domain. See
    _lookup_tier12()'s include_domains docstring for why this exists as a
    separate query rather than folded into the top-1 lookup above.
    """
    rows = conn.sql(f"""
        SELECT DISTINCT domain_id FROM athena_concept
        WHERE lower(concept_name) = ? AND standard_concept = 'S'
        AND vocabulary_id IN ({_in_clause(vocabs)});
    """, params=[normalized_text, *vocabs]).fetchall()
    domains = {r[0] for r in rows}
    if not domains:
        rows = conn.sql(f"""
            SELECT DISTINCT c.domain_id
            FROM athena_concept_synonym s
            JOIN athena_concept c ON s.concept_id = c.concept_id
            WHERE lower(s.concept_synonym_name) = ? AND c.standard_concept = 'S'
            AND c.vocabulary_id IN ({_in_clause(vocabs)});
        """, params=[normalized_text, *vocabs]).fetchall()
        domains = {r[0] for r in rows}
    return sorted(domains)


def _partition_token_ranges(n_tokens: int, max_parts: int):
    """Yields every way to cut `n_tokens` tokens into 2..max_parts contiguous,
    non-empty groups, as lists of (start_idx, end_idx) token-index ranges
    (end exclusive).

    ORDERED BY ASCENDING PART COUNT FIRST (all 2-way partitions, then all
    3-way, then all 4-way). find_compound_split() returns the first
    acceptable partition it finds, so this ordering is what guarantees it
    always prefers the SMALLEST number of concepts the evidence supports --
    the n-way generalization of the same principle the whole-phrase guard
    already enforces at k=0.

    Bounded by _MAX_SPLIT_PARTS, so this is at most a few dozen combinations
    for the short phrases (2-6 tokens) real entities are.
    """
    for k in range(1, min(max_parts, n_tokens)):
        for cuts in itertools.combinations(range(1, n_tokens), k):
            bounds = (0,) + cuts + (n_tokens,)
            yield [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def find_compound_split(conn, text: str, gliner_label: str) -> dict:
    """Tries every way of cutting `text` into 2..`_MAX_SPLIT_PARTS`
    contiguous word groups and returns the FIRST partition where EVERY part
    independently resolves at Tier 1/2 (after trimming leading/trailing
    connector words from each part). Tries 2-way partitions before 3-way
    before 4-way and, within a given part count, tries them in
    left-to-right cut-position order. Returns None if no such partition
    exists -- the common case. Most multi-word entities are genuinely one
    concept ("Exploratory laparotomy", "Aspiration pneumonia") and this
    must not fire on those; requiring EVERY part to independently clear
    Tier 1/2 is what keeps it from firing on an arbitrary phrase that just
    happens to have a plausible-looking cut point.

    2026-08-10 REVERT NOTICE. A same-day "allow one deferred (unresolved)
    part per partition" relaxation was tried here and reverted after a
    real, measured regression: on the large gold note it split "chronic
    systolic heart failure"-type phrases using "heart" as a resolved
    anchor -- "heart" turned out to have a spurious exact Tier-1 match
    against an unrelated SNOMED concept (concept_name "Initial", almost
    certainly a vocabulary data-quality artifact, not a real match) --
    which peeled a real anchor word away from its qualifiers and left the
    remainder ("systolic heart", missing "chronic"/"failure") to Tier 3
    with LESS context than the original merged phrase had. Combined
    linked_recall dropped 10.78% -> 9.72% (median 13.01% -> 11.15%, large
    8.35% -> 7.42%). The relaxation WAS intended to fix a real problem (see
    docs/Stage2_Compound_And_Qualifier_Gaps.md gap 1 refinement history for
    the "Right chest tube placement" over-atomization case it targeted),
    but "at least one part resolves" turned out to be too weak an anchor
    requirement -- this vocabulary has enough single-token exact-string
    collisions that one resolved token is not reliable evidence a split
    boundary is real, and there is no DB access in this development
    environment to characterize which collisions are trustworthy. Requiring
    EVERY part to resolve is a strictly stronger, previously-verified-safe
    bar (measured net-positive across all three gold notes) and is
    reinstated here; the over-atomization case it does not solve is a
    smaller, unconfirmed-impact issue left as a known, documented
    limitation rather than trading a measured win for a measured loss.

    WHOLE-PHRASE GUARD (2026-08-10, added after an earlier measured
    regression of its own). Before trying any partition, the WHOLE text is
    looked up at Tier 1/2 first. If it resolves on its own, this returns
    None immediately: the text is a legitimate standalone compound concept,
    and must not be demoted to its generic parts just because those parts
    also happen to be independently valid concepts. That is not
    hypothetical -- it is exactly what happened on the first real run of
    this function, before the guard existed: "aspiration pneumonia"
    (already correctly resolved as its own Tier-3 concept, matching gold)
    was split into "aspiration" + "pneumonia", and "retroperitoneal
    hematoma" (already correct) was split into "retroperitoneal" +
    "hematoma" -- both previously-correct links turned wrong, and
    note-level linked_recall measurably dropped (15/144 -> 13/144) despite
    the split also fixing two other cases.

    Measured against the real cases that motivated this (see this module's
    "COMPOUND-SPAN SPLITTING" section header): "gunshot wound to abdomen"
    -> "gunshot wound" + "abdomen", "left renal pseudoaneurysm" -> "left
    renal" + "pseudoaneurysm", "left lobe consolidation" -> "left lobe" +
    "consolidation", "nausea or vomiting" -> "nausea" + "vomiting" (all
    2-way). None of these has a Tier 1/2 match for the WHOLE phrase, which
    is why the guard above does not block them.

    Returns {"parts": [{"text", "start", "end", "candidate"}, ...]}, offsets
    relative to the start of `text`, parts in left-to-right order, or None.
    """
    vocabs = VOCAB_BY_LABEL.get(gliner_label, DEFAULT_VOCAB)

    if _lookup_tier12(conn, text, vocabs):
        return None

    tokens = _tokens_with_offsets(text)
    if len(tokens) < 2:
        return None

    for ranges in _partition_token_ranges(len(tokens), _MAX_SPLIT_PARTS):
        parts = []
        for (i, j) in ranges:
            group_tokens = _trim_connectors(tokens[i:j])
            if not group_tokens:
                parts = None
                break
            g_start, g_end = group_tokens[0][1], group_tokens[-1][2]
            g_text = text[g_start:g_end]
            if (len(g_text) < _MIN_SPLIT_HALF_CHARS
                    and g_text.strip().lower() not in _SHORT_LATERALITY_TOKENS):
                parts = None
                break
            # 2026-08-12 LAB-VALUE-SUFFIXED PARTS ("Triglyc-97 HDL-34" ->
            # "Triglyc-97" + "HDL-34"). Each part's RAW g_text is tried
            # first (unchanged behavior for every non-lab compound span);
            # only when that fails AND gliner_label is "Lab Test" do we also
            # try strip_lab_value_suffix()'s candidates as alternate lookup
            # keys. This is necessary, not redundant with
            # process_and_normalize_entities()'s own lab-suffix fallback --
            # that fallback only ever runs on an entity AFTER a split has
            # already been accepted and turned into a separate row. Here,
            # the raw text ("Triglyc-97") fails Tier 1/2 on its own, so
            # without this the split is never even attempted and the whole
            # compound span is rejected, staying one unresolved entity.
            # strip_lab_value_suffix() returns a LIST of candidate NAME
            # strings (the value is discarded, not a single string), so
            # every candidate is tried in order and the first Tier 1/2 hit
            # wins -- same "candidate list, ontology arbitrates" pattern
            # process_and_normalize_entities() already uses, not a new one.
            # g_text (the ORIGINAL, unstripped substring) is still what's
            # stored in the returned part's "text"/offsets -- only the
            # LOOKUP key changes, so downstream span offsets stay exact.
            hit = _lookup_tier12(conn, g_text, vocabs, include_domains=True)
            if not hit and gliner_label == "Lab Test":
                for candidate_text in strip_lab_value_suffix(g_text):
                    hit = _lookup_tier12(conn, candidate_text, vocabs, include_domains=True)
                    if hit:
                        break
            if not hit:
                parts = None
                break
            parts.append({"text": g_text, "start": g_start, "end": g_end, "candidate": hit})
        if parts:
            return {"parts": parts}
    return None


# ==========================================================================
# SPAN GROWTH -- the mirror image of compound-span splitting
#
# 2026-08-10, docs/Stage2_Compound_And_Qualifier_Gaps.md gap 2, measured on
# 19442119-DS-15 (the median gold note): gold repeatedly annotates a LONGER,
# QUALIFIED phrase as one specific concept, while Stage 2a's predicted span
# is the SHORTER, unqualified sub-phrase, which then resolves to a more
# generic -- and WRONG -- concept. "congestive heart failure" (42343007) is
# not the same disease-concept as "heart failure" (84114007); "pulmonary
# hypertension" (70995007) is a different disease entirely from systemic
# "hypertension", not a narrower version of it. Compound-splitting cannot
# fix this -- there is nothing to split, the predicted span is already too
# SHORT, not compound. This needs the opposite operation: try WIDENING the
# span by absorbing adjacent words, and prefer the wider phrase whenever it
# resolves at the same Tier 1/2 confidence bar find_compound_split()
# requires.
#
# find_span_growth() is the detector, symmetric in spirit to
# find_compound_split() in the same module; src/clinical_pipeline.py owns
# turning a positive result into a replacement entity (offset recomputation
# needs the same map_offsets_to_original() machinery the splitter uses).
# ==========================================================================

# Real cases measured (docs/Stage2_Compound_And_Qualifier_Gaps.md gap 2) are
# all SINGLE-SIDED (a qualifier immediately to the LEFT of the extracted
# span): "congestive"+heart failure, "acute"+pulmonary edema,
# "valvular"+heart disease, "Mid"+LAD, "occlusion of the"+LAD (3 words),
# "stenosis of the"+midcircumflex (3 words), "acute on chronic
# systolic"+heart failure (4 words). 5 gives headroom past the longest of
# these without inviting a much larger search.
_MAX_GROW_TOKENS = 5


def find_span_growth(conn, entity_text: str, left_context: str, right_context: str,
                     gliner_label: str) -> dict:
    """Tries absorbing 1..`_MAX_GROW_TOKENS` adjacent words from the LEFT or
    RIGHT of `entity_text` (never both sides at once -- see below) and
    returns the FIRST growth whose combined text resolves at Tier 1/2,
    smallest absorption first, left before right.

    `left_context`/`right_context` MUST already be trimmed to the entity's
    own sentence by the caller (mirroring how src/entity_extraction.py's
    build_local_context() is sentence-bounded) -- this function has no
    sentence awareness of its own, so a caller that hands it a whole-note
    context risks absorbing words from an unrelated adjacent statement.

    WHY NO TIER 3 HERE. Same reasoning as find_compound_split(): a grown
    candidate must clear the SAME confidence bar as a clean match, or this
    would be trading a narrow-but-real match for a shaky semantic guess on
    a WIDER phrase -- worse than doing nothing.

    WHY THIS RUNS EVEN WHEN THE UN-GROWN TEXT ALREADY RESOLVES. Unlike
    find_compound_split()'s whole-phrase guard (which exists precisely to
    STOP it from overriding an already-correct match), growth exists BECAUSE
    a narrow match succeeding at Tier 1/2 does not mean it is the BEST
    match -- "heart failure" resolves cleanly on its own, and is still
    wrong when gold wants "congestive heart failure". A grown match is
    preferred over the narrow one whenever it ALSO resolves, since a longer
    phrase matching a real, specific SNOMED concept name is virtually
    always more clinically precise than a shorter substring's generic one.
    The collision risk (a coincidental unrelated concept happening to share
    the grown phrase's exact name) is the same low-probability class of
    risk find_compound_split() already accepts for requiring an exact
    string match, not a new one.

    WHY NOT BOTH SIDES AT ONCE. Every real case measured so far is single-
    sided. Trying combined left+right growth would multiply the search
    (a two-sided pass adds up to _MAX_GROW_TOKENS^2 extra lookups) for a
    pattern not yet observed -- left out deliberately as scope control, not
    an oversight; worth adding if a two-sided case is ever measured.

    Returns {"text", "grow_left_chars", "grow_right_chars", "candidate"}
    (char counts absorbed from left_context's END / right_context's START,
    letting the caller compute absolute offsets without re-tokenizing), or
    None.
    """
    vocabs = VOCAB_BY_LABEL.get(gliner_label, DEFAULT_VOCAB)

    left_tokens = _tokens_with_offsets(left_context)[-_MAX_GROW_TOKENS:]
    right_tokens = _tokens_with_offsets(right_context)[:_MAX_GROW_TOKENS]

    trials = []  # (grown_text, n_words_absorbed, grow_left_chars, grow_right_chars)
    for n in range(1, len(left_tokens) + 1):
        chosen = left_tokens[-n:]
        grow_chars = len(left_context) - chosen[0][1]
        grown = left_context[chosen[0][1]:] + entity_text
        trials.append((grown, n, grow_chars, 0))
    for n in range(1, len(right_tokens) + 1):
        chosen = right_tokens[:n]
        grow_chars = chosen[-1][2]
        grown = entity_text + right_context[:grow_chars]
        trials.append((grown, n, 0, grow_chars))

    trials.sort(key=lambda t: t[1])  # fewest absorbed words first

    for grown_text, _n, gl_chars, gr_chars in trials:
        hit = _lookup_tier12(conn, grown_text, vocabs, include_domains=True)
        if hit:
            return {
                "text": grown_text, "grow_left_chars": gl_chars,
                "grow_right_chars": gr_chars, "candidate": hit,
            }
    return None


# ==========================================================================
# LAB VALUE SUFFIX STRIPPING
#
# 2026-08-10, docs/Stage2_Compound_And_Qualifier_Gaps.md. MIMIC discharge
# summaries report labs in compact flowsheet notation -- "WBC-13.0",
# "Glucose-117", "UREA N-25" -- test name and numeric result glued together
# by a hyphen with no space. GLiNER extracts the WHOLE hyphenated token as
# one Lab Test span (there is no whitespace boundary to stop at), so
# normalize_entity() tries to match "wbc-13.0" against SNOMED/LOINC concept
# names -- which obviously never include the numeric result -- and fails at
# every tier. Measured directly: WBC-13.0, PTT-29.0 and Glucose-117 all "0
# (Failed)" on 17751158-DS-19; ALT-736/AST-956 the same way on
# 19442119-DS-15.
#
# This exact pattern was already known on the ASSERTION side -- see
# src/assertion.py's is_structured_result() docstring, which names the same
# GLUCOSE-109/UREA N-25/CREAT-0.3 examples as a reason lab-panel lines must
# not be treated as narrative prose -- but nothing closed the loop on
# normalization actually being able to link the test name.
#
# 2026-08-11 DELIMITER-LESS CASE ("RR18", "SpO2100") -- CANDIDATE
# ARBITRATION, NOT A SECOND REGEX RULE. An earlier version of this fix tried
# a single `(?<=[A-Za-z])` lookbehind to also split delimiter-less tokens.
# That lookbehind can ONLY ever match immediately after a LETTER, so it is
# structurally incapable of finding the correct boundary for any test name
# that itself ends in a digit: on "SpO2100" it fires at "SpO"|"2100" (the
# position after 'O') and can never reach "SpO2"|"100" (the position after
# '2', a digit) -- it silently returned "SpO", a 3-character string that
# clears _MIN_SPLIT_HALF_CHARS on its own and so was never caught by any
# safety net. It went into Tier 1/2/3 lookup looking like a verified
# stripped name, not a guess.
#
# There is no regex that reliably distinguishes "trailing digit belongs to
# the NAME" (SpO2) from "trailing digit belongs to the VALUE" (RR18) --
# both are the same string shape. So strip_lab_value_suffix() does not try
# to decide; it returns every plausible split as a CANDIDATE LIST and
# process_and_normalize_entities() tries each one against Tier 1/2 in
# order, keeping whichever one the OMOP vocabulary actually confirms. This
# is the same "let the ontology arbitrate instead of hand-writing the
# boundary" principle find_compound_split() already uses for the embedded-
# newline compound-span problem -- see that function's "2026-08-11 NEWLINE
# COMPOUND SPANS" comment block above for the fuller rationale.
# ==========================================================================

# Case A: explicit hyphen delimiter. Unambiguous -- the hyphen's position
# fixes the boundary regardless of what precedes it, so a lazy name is safe
# here (unlike the delimiter-less case below, there is only one place the
# literal "-" can be, so laziness can't land on the wrong one).
#
# 2026-08-12: value alternation extended past pure numerics to cover
# categorical lab results ("Leuks-NEG", "HER2-POS") -- same shape, same
# hyphen delimiter, just a qualitative reading instead of a quantitative
# one. Case-insensitive (re.I) since MIMIC free text mixes "NEG"/"Neg"/
# "negative" inconsistently. Safe to widen: the `value` alternation only
# controls whether a candidate is OFFERED, not whether it is TRUSTED -- the
# `name` half still has to independently clear Tier 1/2 exact/synonym
# lookup downstream (strip_lab_value_suffix's caller, or
# find_compound_split's per-part lookup below), so a false-positive match
# here just wastes one lookup rather than mislabeling anything. Verified
# against real entities from this project's own live batch runs: still
# correctly declines to match non-lab hyphenated text ("well-appearing",
# "darunavir-cobicistat", "T2-L1", "ANCA-NEGATIVE B" -- the trailing " B"
# breaks the end anchor) while now matching "Leuks-NEG", "HER2-POS",
# "Rh-negative".
_LAB_VALUE_HYPHEN_RE = re.compile(
    r"^(?P<name>.+?)\s*-\s*"
    r"(?P<value>-?\d+(?:\.\d+)?|NEG(?:ATIVE)?|POS(?:ITIVE)?|TRACE|NORMAL|ABNORMAL)\s*$",
    re.I,
)

# Case B: explicit whitespace delimiter ("O2 98"). Also unambiguous for the
# same reason -- the class deliberately excludes whitespace, so greedy vs
# lazy makes no difference; it always stops at the real space.
_LAB_VALUE_SPACE_RE = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9/%()]*)\s+(?P<value>[<>]?-?\d+(?:\.\d+)?\*?)\s*$")

# Case C gate: the delimiter-less shape ("RR18", "SpO2100") -- starts with a
# letter, no whitespace/hyphen/other punctuation anywhere, ends in a digit.
# Only reached when Cases A and B both miss, i.e. there is no real
# delimiter character in the text at all.
_DELIMITER_LESS_SHAPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*\d$")

# Known short (<3-char) lab/vitals name tokens -- mirrors
# _SHORT_LATERALITY_TOKENS's architecture: a narrow, evidence-backed
# whitelist rather than lowering _MIN_SPLIT_HALF_CHARS generally (which
# would reopen the short-string collision risk that floor exists to
# prevent). Bounded scope: only matters for CANDIDATES that already passed
# the delimiter-based or delimiter-less split logic above, and every
# candidate still has to clear Tier 1/2 exact/synonym lookup afterward --
# this whitelist only controls whether a short name is worth TRYING, not
# whether it is trusted once tried.
_SHORT_LAB_TOKENS = {"rr", "o2", "ph", "k", "mg", "na", "cl", "ca", "hb"}

# 2026-08-12, used by the LAB VALUE SUFFIX FALLBACK's acceptance rule in
# process_and_normalize_entities() -- explicit rank rather than sorting the
# match_tier strings directly, same reasoning as score_gold_recall.py's own
# TIER_RANK: "0 (Failed)" would otherwise sort ahead of "1 (Exact)" because
# '0' < '1' lexicographically, which is backwards (0 is the WORST tier).
# Lower rank = more trustworthy = preferred.
_LAB_TIER_RANK = {"1 (Exact)": 1, "2 (Synonym)": 2, "3 (Semantic)": 3,
                  "4 (Fuzzy)": 4, "0 (Failed)": 5}


def strip_lab_value_suffix(text: str) -> list:
    """Returns candidate test-name strings for a flowsheet-style lab result
    (e.g. "WBC-13.0" -> ["WBC"], "RR18" -> ["RR"], "SpO2100" -> ["SpO",
    "SpO2"]), or [] if `text` doesn't match a lab-value shape at all, or
    every candidate is too short/unrecognized to be worth trying (reuses
    _MIN_SPLIT_HALF_CHARS's 3-character floor, with the _SHORT_LAB_TOKENS
    exception above for known short lab names).

    Returns a LIST, not a single string, because the delimiter-less case is
    genuinely ambiguous at the string level -- see this section's 2026-08-11
    comment block above for why a single regex cannot resolve it and what
    replaced the attempt that tried. The hyphen and whitespace cases are
    unambiguous and always return at most one candidate.

    Ordered so the more-likely-correct interpretation is tried first for
    the delimiter-less case (name stops at the start of the trailing digit
    run, before trying the name-absorbs-one-more-digit variant), but
    ordering is a minor efficiency detail, not a correctness guarantee --
    the caller only keeps a candidate that Tier 1/2 actually confirms, so a
    wrong candidate earlier in the list can never win over a right one
    later in it.

    The numeric value is discarded, not carried forward -- it is a
    measurement RESULT, not part of the concept name, and this codebase has
    no lab-value-fact table yet to hand it to. That is a real, separate gap
    (the value itself is clinically meaningful and currently just vanishes
    from the entity once stripped), not something this function's caller
    should be expected to solve.
    """
    stripped_text = (text or "").strip()
    if not stripped_text:
        return []

    candidates = []

    m = _LAB_VALUE_HYPHEN_RE.match(stripped_text)
    if m:
        candidates.append(m.group("name").strip())

    if not candidates:
        m = _LAB_VALUE_SPACE_RE.match(stripped_text)
        if m:
            candidates.append(m.group("name").strip())

    if not candidates and _DELIMITER_LESS_SHAPE_RE.match(stripped_text):
        i = len(stripped_text)
        while i > 0 and stripped_text[i - 1].isdigit():
            i -= 1
        # i is the index of the first digit of the TRAILING digit run.
        # i > 0 is guaranteed by the gate regex requiring a leading letter.
        candidates.append(stripped_text[:i])  # value starts at the run: "RR"+"18"
        # Only offered when at least one digit remains for the value --
        # otherwise this "candidate" would just be the whole original
        # string with nothing left to have stripped (e.g. "C5" -> "C" only,
        # no second candidate, since absorbing the '5' leaves no value at
        # all and correctly falls through to rejection below).
        if i + 1 <= len(stripped_text) - 1:
            candidates.append(stripped_text[:i + 1])  # name absorbs one more digit: "SpO2"+"100"

    out = []
    for c in candidates:
        c = c.strip()
        if c and (len(c) >= _MIN_SPLIT_HALF_CHARS or c.lower() in _SHORT_LAB_TOKENS):
            out.append(c)
    return out


def get_sapbert_embedding(text: str) -> list:
    """Generates a 768-dimensional SapBERT vector for a given text."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = sapbert(**tokens)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
    return embedding


_TIER_DEFAULT_MATCH_BASIS = {
    "1 (Exact)": "exact_text",
    "2 (Synonym)": "synonym",
    "3 (Semantic)": "semantic_similarity",
    "4 (Fuzzy)": "fuzzy_edit_distance",
}


def _candidate(row, tier, score, match_basis=None):
    """match_basis (2026-08-14, Stage 3 "Lasix problem" follow-up) records HOW
    a candidate was found, not just how well it scored. Without it, a
    KG-verified brand-alias hit (furosemide for "Lasix", similarity 0.57) and
    a coincidental spelling match (lasalocid, similarity 0.68) are
    STRUCTURALLY IDENTICAL dicts to Stage 3 -- nothing distinguishes "this is
    a certain fact from the terminology graph" from "this merely sounds
    alike", so a small model has no way to weigh the former over the
    latter's higher raw score. Defaults to the tier's usual basis; callers
    that inject alias-expansion candidates (see _tier3_semantic_rows) override
    it to "verified_brand_alias" for exactly those rows.
    """
    return {
        "omop_concept_id": row[0],
        "concept_name": row[1],
        "domain_id": row[2],
        "vocabulary_id": row[3],
        "match_tier": tier,
        "similarity_score": score,
        "match_basis": match_basis or _TIER_DEFAULT_MATCH_BASIS.get(tier, "unknown"),
    }


def _cosine(a, b):
    """Plain cosine similarity between two equal-length vectors. Returns 0.0
    on a zero vector rather than raising -- a degenerate embedding must cost a
    candidate its ranking bonus, not take down normalization."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _concept_class_map(conn, concept_ids):
    """{concept_id: concept_class_id} for the given ids.

    A separate lookup rather than an extra column on every Tier 1/2 SELECT,
    for two reasons: the tuple shape those queries return feeds _candidate()
    and three other call sites that would all need changing, and this query
    only runs when ranking is enabled AND there is more than one candidate to
    rank -- i.e. never on the common single-hit path. Failure returns an empty
    map, which makes every candidate score equally on class and lets the
    remaining criteria decide, rather than breaking the lookup.
    """
    if not concept_ids:
        return {}
    try:
        rows = conn.sql(
            f"SELECT concept_id, concept_class_id FROM athena_concept "
            f"WHERE concept_id IN ({','.join('?' * len(concept_ids))});",
            params=list(concept_ids)).fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}


def _class_rank(concept_class_id, gliner_label):
    """Lower is better. Preferred classes score 0..n by their position in
    TIER12_CLASS_PREFERENCE; unlisted classes score 90; demoted classes 99.
    """
    if concept_class_id is None:
        return 90
    preferred = TIER12_CLASS_PREFERENCE.get(gliner_label) or []
    if concept_class_id in preferred:
        return preferred.index(concept_class_id)
    if concept_class_id in TIER12_CLASS_DEMOTED:
        return 99
    return 90


def _specificity(conn, concept_ids):
    """{concept_id: n_ancestors}. More ancestors = deeper in the SNOMED
    hierarchy = more specific.

    Used only as a late tiebreak, and deliberately preferring the MORE
    specific concept: the DrivenData gold annotations name specific clinical
    findings ("Fracture of clavicle") far more often than their generic
    parents ("Fracture"), so when class, domain and semantic similarity have
    all failed to separate two candidates, the specific one is the better bet.
    Best-effort -- athena_concept_ancestor is 78.4M rows but this queries a
    handful of ids at a time, and any failure degrades to "no specificity
    signal" rather than raising.
    """
    if not concept_ids:
        return {}
    try:
        rows = conn.sql(
            f"SELECT descendant_concept_id, COUNT(*) FROM athena_concept_ancestor "
            f"WHERE descendant_concept_id IN ({','.join('?' * len(concept_ids))}) "
            f"GROUP BY descendant_concept_id;",
            params=list(concept_ids)).fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}


def _rank_tier12_candidates(conn, rows, gliner_label, entity_text, domains=None,
                            vector=None, use_semantic=None):
    """Orders Tier 1/2 result rows by clinical plausibility instead of by
    concept_id, and reports whether the ranking actually separated them.

    Returns (ordered_rows, basis, unresolved_tie).
      ordered_rows    -- the same tuples, reordered
      basis           -- "concept_id_asc" or "ranked_v1", persisted to
                         normalized_entities.tier12_rank_basis so an A/B is
                         gradable per row rather than per run
      unresolved_tie  -- True when the top two could not be separated; the
                         caller turns this into is_ambiguous so Stage 3 sees
                         a real question instead of an arbitrary answer

    RANKING KEY, most significant first. Each criterion is cheap and only
    consulted when the ones above it tie:
      1. concept_class_id preference for this GLiNER label. One SQL lookup,
         no model call, and the strongest signal available -- a "Condition"
         mention resolving to a Qualifier Value concept is almost always the
         collision case, not the intended reading.
      2. Domain agreement with GLINER_LABEL_TO_DOMAIN. Usually already
         enforced by the query's domain filter, but NOT when domain_override
         is in play (compound-split entities) or when the label has no domain
         mapping, so it still discriminates in exactly the cases that are
         hardest.
      3. SapBERT cosine between the ENTITY TEXT and each CONCEPT NAME. Reuses
         the embedding machinery Tier 3 already depends on. Costs at most
         CANDIDATE_LIMIT (3) embeddings, and only for multi-hit Tier 1/2
         lookups, which are the minority.
      4. Specificity (ancestor depth), preferring the more specific concept.
      5. concept_id ASC -- retained as the FINAL tiebreak so the function stays
         deterministic and reproducible. The complaint was never that
         concept_id ordering is unstable; it is that it was the FIRST
         criterion rather than the last.

    Never raises. Any failure (no class map, no embeddings, DB error) degrades
    to the legacy ordering with basis "concept_id_asc", so a ranking problem
    can never be worse than today's behaviour.
    """
    if not rows or len(rows) == 1:
        return rows, "concept_id_asc", False

    try:
        ids = [r[0] for r in rows]
        classes = _concept_class_map(conn, ids)
        depths = _specificity(conn, ids)

        # Criterion 3 is opt-in (TIER12_RANK_SEMANTIC). When off, sims stays
        # empty and key() reads a constant 0.0 for every candidate, so the
        # criterion contributes nothing to the ordering and no embedding is
        # requested -- not for the entity, and not for any concept name.
        semantic = TIER12_RANK_SEMANTIC if use_semantic is None else use_semantic
        sims = {}
        if semantic:
            if vector is None:
                vector = get_sapbert_embedding(entity_text)
            for r in rows:
                try:
                    sims[r[0]] = _cosine(vector, get_sapbert_embedding(r[1]))
                except Exception:
                    sims[r[0]] = 0.0

        wanted_domains = set(domains or GLINER_LABEL_TO_DOMAIN.get(gliner_label) or [])

        def key(r):
            cid, name, domain_id = r[0], r[1], r[2]
            return (
                _class_rank(classes.get(cid), gliner_label),      # asc, lower better
                0 if (not wanted_domains or domain_id in wanted_domains) else 1,
                -sims.get(cid, 0.0),                              # desc
                -depths.get(cid, 0),                              # desc (more specific)
                cid,                                              # asc, final tiebreak
            )

        ordered = sorted(rows, key=key)

        # BASIS IS HONEST ABOUT DEGRADED INPUTS. _concept_class_map() swallows
        # its own errors and returns {} so a vocabulary without concept_class_id
        # cannot break normalization -- but a ranking made without the single
        # strongest criterion is not the same ranking, and labelling it
        # "ranked_v1" would make the A/B compare two different things under one
        # name. Recorded separately so those rows can be excluded from (or
        # examined in) the comparison.
        basis = "ranked_v1" if classes else "ranked_v1_no_class"
        if semantic:
            basis += "_semantic"

        # A tie that survives class, domain and semantics is a real clinical
        # ambiguity, not something an ordering rule should pretend to settle.
        k0, k1 = key(ordered[0]), key(ordered[1])
        unresolved = (k0[0] == k1[0] and k0[1] == k1[1]
                      and abs(k0[2] - k1[2]) < TIER12_TIE_EPSILON)
        return ordered, basis, unresolved
    except Exception:
        return rows, "concept_id_asc", False


def _fuzzy_typo_candidates(conn, search_text, vocabs, domains, exclude_ids=None):
    """Edit-distance supplement to an already-uncertain Tier 3 result. See the
    FUZZY_MAX_EDIT_DISTANCE comment above for why this exists and why it is
    additive rather than its own tier.

    Best-effort: DuckDB's levenshtein() is a core scalar function as of the
    duckdb>=0.9.0 pinned in requirements.txt, but this was written and is not
    exercised against a live database in this environment -- a missing or
    renamed function must degrade to "no fuzzy candidates found" rather than
    breaking normalize_entity() for every call, so any duckdb error here is
    swallowed rather than raised.
    """
    if len(search_text) < FUZZY_MIN_TEXT_LENGTH:
        return []
    domain_clause = f" AND domain_id IN ({_in_clause(domains)})" if domains else ""
    lo, hi = len(search_text) - FUZZY_MAX_EDIT_DISTANCE, len(search_text) + FUZZY_MAX_EDIT_DISTANCE
    try:
        rows = conn.sql(f"""
            WITH scored AS (
                SELECT concept_id, concept_name, domain_id, vocabulary_id,
                       levenshtein(lower(concept_name), ?) AS edit_distance
                FROM athena_concept
                WHERE standard_concept = 'S'
                AND vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause}
                AND length(concept_name) BETWEEN ? AND ?
            )
            SELECT concept_id, concept_name, domain_id, vocabulary_id, edit_distance
            FROM scored
            WHERE edit_distance <= ?
            ORDER BY edit_distance ASC, concept_id ASC
            LIMIT {CANDIDATE_LIMIT};
        """, params=[search_text, *vocabs, *(domains or []), lo, hi,
                     FUZZY_MAX_EDIT_DISTANCE]).fetchall()
    except duckdb.Error:
        return []
    exclude_ids = exclude_ids or set()
    out = []
    for concept_id, name, domain_id, vocab_id, edit_distance in rows:
        if concept_id in exclude_ids:
            continue
        # Score on the same 0-1 scale as Tier 3's cosine similarity so it can
        # sit in the same candidate list, without claiming false precision --
        # this is "how much of the string matched", not a calibrated quantity.
        score = round(1 - (edit_distance / max(len(search_text), len(name))), 4)
        out.append(_candidate((concept_id, name, domain_id, vocab_id), "4 (Fuzzy)", score))
    return out


def _alias_expand_brand_to_generic(conn, search_text):
    """Brand-name -> generic-drug alias resolution (2026-08-13, "Lasix
    problem": SapBERT's embedding space does not reliably place a brand name
    close to its active ingredient, so a brand-only mention can silently drop
    out of Tier 3's top-K even though every other signal -- MoLLM reasoning,
    RxNorm itself -- agrees on the underlying drug).

    There is no direct 1-hop CONCEPT_RELATIONSHIP edge from a Brand Name
    concept to its generic ingredient (verified empirically against this DB:
    a 'Maps to'/'Tradename of' query straight from the brand concept returns
    nothing). The real path is three hops:
        brand -[Brand name of]-> branded product (Branded Drug Comp)
              -[Tradename of]-> generic Clinical Drug Comp (standard, RxNorm)
              -[RxNorm has ing]-> Ingredient (standard, RxNorm)

    Only the final Ingredient concept_ids are returned, not the intermediate
    dose-specific Clinical Drug Comp ones -- a well-established brand like
    Lasix has hundreds of branded SKUs (every dose/form/pack-size
    combination), each fanning out to its own generic Clinical Drug Comp, so
    keeping those would balloon the candidate list back into the exact
    crowding problem Fix 2 exists to prevent. The active ingredient(s) are
    what a bare brand mention ("Lasix") actually needs to resolve to, and
    every branded SKU collapses to the same small handful of ingredient
    concept_ids (usually one; combination drugs have a few).

    Returns a (possibly empty) set of standard concept_ids. Empty whenever
    search_text isn't an exact RxNorm Brand Name -- i.e. free on every
    non-brand entity, which is almost all of them.
    """
    try:
        rows = conn.sql("""
            WITH brand AS (
                SELECT concept_id FROM athena_concept
                WHERE lower(concept_name) = ?
                AND vocabulary_id = 'RxNorm' AND concept_class_id = 'Brand Name'
            ),
            branded_product AS (
                SELECT DISTINCT r.concept_id_2 AS concept_id
                FROM athena_concept_relationship r
                JOIN brand b ON r.concept_id_1 = b.concept_id
                WHERE r.relationship_id = 'Brand name of' AND r.invalid_reason IS NULL
            ),
            generic AS (
                SELECT DISTINCT r.concept_id_2 AS concept_id
                FROM athena_concept_relationship r
                JOIN branded_product bp ON r.concept_id_1 = bp.concept_id
                WHERE r.relationship_id = 'Tradename of' AND r.invalid_reason IS NULL
            ),
            ingredient AS (
                SELECT DISTINCT r.concept_id_2 AS concept_id
                FROM athena_concept_relationship r
                JOIN generic g ON r.concept_id_1 = g.concept_id
                WHERE r.relationship_id = 'RxNorm has ing' AND r.invalid_reason IS NULL
            )
            SELECT DISTINCT c.concept_id
            FROM athena_concept c
            WHERE c.concept_id IN (SELECT concept_id FROM ingredient)
            AND c.standard_concept = 'S';
        """, params=[search_text]).fetchall()
        return {r[0] for r in rows}
    except duckdb.Error:
        return set()


def _collapse_hierarchy_duplicates(conn, cands):
    """Collapses candidates connected by a direct SNOMED 'Is a'/'Subsumes'
    edge into a single entry, keeping whichever has the higher similarity
    score.

    WHY THIS EXISTS (2026-08-13, ALT-measurement example from
    docs/Stage3_Open_Issues.md): Tier 3 can retrieve several *distinct*
    standard concepts that are not accidental duplicates but real SNOMED
    parent/child concepts describing the same clinical idea at slightly
    different specificity (e.g. "ALT - blood measurement" is a parent of both
    "Alanine transaminase activity measurement" and "Alanine aminotransferase
    measurement"). Presenting all of them to MoLLM voting fractures what
    should be one answer into an artificial 2-1 or 1-1-1 split. A plain
    GROUP BY concept_id can't catch this -- the ids are genuinely different --
    so this walks the relationship graph among just the candidate set and
    merges connected components instead.

    Only direct edges within the candidate set are considered (no external
    hierarchy lookups), so this stays cheap: at most CANDIDATE_LIMIT-ish rows
    are ever passed in. Never raises -- any DB error degrades to returning
    cands unchanged.
    """
    if len(cands) < 2:
        return cands
    ids = [c["omop_concept_id"] for c in cands]
    placeholders = ",".join("?" * len(ids))
    try:
        rows = conn.sql(f"""
            SELECT DISTINCT concept_id_1, concept_id_2
            FROM athena_concept_relationship
            WHERE concept_id_1 IN ({placeholders}) AND concept_id_2 IN ({placeholders})
            AND relationship_id IN ('Is a', 'Subsumes') AND invalid_reason IS NULL
            AND concept_id_1 != concept_id_2;
        """, params=[*ids, *ids]).fetchall()
    except duckdb.Error:
        return cands
    if not rows:
        return cands

    parent = {cid: cid for cid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in rows:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    best_for_root = {}
    for c in cands:
        root = find(c["omop_concept_id"])
        current = best_for_root.get(root)
        if current is None or c["similarity_score"] > current["similarity_score"]:
            best_for_root[root] = c

    seen_roots = set()
    out = []
    for c in cands:
        root = find(c["omop_concept_id"])
        if root in seen_roots:
            continue
        seen_roots.add(root)
        out.append(best_for_root[root])
    return out


def _tier3_semantic_rows(conn, vector, vocabs, domains, alias_ids=None):
    """The Tier 3 SapBERT top-K query, plus force-including any alias_ids
    (see _alias_expand_brand_to_generic) regardless of where they land in the
    similarity ranking. Without this, a real cosine-similarity gap between a
    brand name and its own generic ingredient silently drops the correct
    concept out of the top CANDIDATE_LIMIT before Stage 3 ever sees it.

    alias_ids are scored by their own cosine similarity (not pinned to 1.0),
    so Stage 3 still sees the true semantic distance -- only their presence
    in the candidate list is guaranteed, not their rank.
    """
    domain_clause = f" AND domain_id IN ({_in_clause(domains)})" if domains else ""
    alias_ids = list(alias_ids or [])
    try:
        if alias_ids:
            placeholders = ",".join("?" * len(alias_ids))
            rows = conn.sql(f"""
                WITH topk AS (
                    SELECT concept_id, concept_name, domain_id, vocabulary_id,
                           list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
                    FROM athena_concept
                    WHERE embedding IS NOT NULL AND standard_concept = 'S'
                    AND vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause}
                    ORDER BY similarity DESC, concept_id ASC LIMIT {CANDIDATE_LIMIT}
                ),
                alias AS (
                    SELECT concept_id, concept_name, domain_id, vocabulary_id,
                           list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
                    FROM athena_concept
                    WHERE embedding IS NOT NULL AND standard_concept = 'S'
                    AND concept_id IN ({placeholders})
                )
                SELECT * FROM topk
                UNION
                SELECT * FROM alias
                ORDER BY similarity DESC, concept_id ASC;
            """, params=[vector, *vocabs, *(domains or []), vector, *alias_ids]).fetchall()
        else:
            rows = conn.sql(f"""
                SELECT concept_id, concept_name, domain_id, vocabulary_id,
                       list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
                FROM athena_concept
                WHERE embedding IS NOT NULL AND standard_concept = 'S'
                AND vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause}
                ORDER BY similarity DESC, concept_id ASC LIMIT {CANDIDATE_LIMIT};
            """, params=[vector, *vocabs, *(domains or [])]).fetchall()
    except duckdb.Error:
        return []
    return rows


def _tier_queries(conn, search_text, vocabs, domains, entity_text, vector=None,
                  gliner_label=None):
    """Runs tiers 1-3 under a given domain restriction. Returns
    (candidates, tier_name, best_tier3_score) or (None, None, score).

    Factored out of normalize_entity() so the exact same three queries can be
    re-run with the domain filter lifted (see _detect_domain_conflict) without
    duplicating the SQL -- two copies of these queries drifting apart would be
    a silent correctness bug of the same family as the LIMIT-1 issue.
    """
    domain_clause = f" AND domain_id IN ({_in_clause(domains)})" if domains else ""
    domain_clause2 = f" AND c.domain_id IN ({_in_clause(domains)})" if domains else ""

    rows = conn.sql(f"""
        SELECT concept_id, concept_name, domain_id, vocabulary_id
        FROM athena_concept
        WHERE lower(concept_name) = ? AND standard_concept = 'S'
        AND vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause}
        ORDER BY concept_id ASC LIMIT {CANDIDATE_LIMIT};
    """, params=[search_text, *vocabs, *(domains or [])]).fetchall()
    if rows:
        # 2026-08-13 (P1.1): rank before returning, so candidates[0] -- the
        # pick every downstream consumer treats as Stage 2b's answer -- is
        # chosen on clinical plausibility rather than on integer id.
        rows, _basis, _tie = (_rank_tier12_candidates(
            conn, rows, gliner_label, entity_text, domains=domains, vector=vector)
            if RANKED_TIER12 else (rows, "concept_id_asc", False))
        return [_candidate(r, "1 (Exact)", 1.0) for r in rows], "1", None

    rows = conn.sql(f"""
        SELECT DISTINCT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id
        FROM athena_concept_synonym s
        JOIN athena_concept c ON s.concept_id = c.concept_id
        WHERE lower(s.concept_synonym_name) = ? AND c.standard_concept = 'S'
        AND c.vocabulary_id IN ({_in_clause(vocabs)}) {domain_clause2}
        ORDER BY c.concept_id ASC LIMIT {CANDIDATE_LIMIT};
    """, params=[search_text, *vocabs, *(domains or [])]).fetchall()
    if rows:
        rows, _basis, _tie = (_rank_tier12_candidates(
            conn, rows, gliner_label, entity_text, domains=domains, vector=vector)
            if RANKED_TIER12 else (rows, "concept_id_asc", False))
        return [_candidate(r, "2 (Synonym)", 1.0) for r in rows], "2", None

    if vector is None:
        vector = get_sapbert_embedding(entity_text)
    alias_ids = _alias_expand_brand_to_generic(conn, search_text)
    rows = _tier3_semantic_rows(conn, vector, vocabs, domains, alias_ids=alias_ids)
    if not rows:
        return None, None, 0.0
    cands = [_candidate(r, "3 (Semantic)", round(r[4], 4),
                       match_basis="verified_brand_alias" if r[0] in alias_ids else None)
            for r in rows]
    cands = _collapse_hierarchy_duplicates(conn, cands)
    return cands, "3", cands[0]["similarity_score"]


def _detect_domain_conflict(conn, search_text, vocabs, domains, entity_text, vector,
                            gliner_label=None):
    """Re-runs the tiers WITHOUT the domain restriction after a filtered miss.

    WHY THIS EXISTS. The domain filter is a hard WHERE clause on every tier, so
    when GLiNER labels a span `Medication` but the correct concept sits in the
    Condition domain, the query does not return that concept flagged as
    mismatched -- it returns nothing, and the entity is reported "0 (Failed)".
    That silently conflates two situations that need completely different
    handling downstream:

        (a) the concept genuinely is not in the vocabulary, and
        (b) the concept IS in the vocabulary, under a different domain than the
            extraction label predicted -- i.e. GLiNER probably mislabelled it.

    (b) is recoverable and is exactly the disambiguation Stage 3 exists to do,
    but the filter was destroying the evidence for it. It also inflates the
    apparent failure rate: a correctly-extracted-but-mislabelled entity was
    indistinguishable from an unmappable one in any evaluation.

    Only runs after a filtered miss, so it costs nothing on the common path.
    The SapBERT vector is passed in rather than recomputed for the same reason.
    """
    if not domains:
        return None
    cands, tier, _ = _tier_queries(conn, search_text, vocabs, None, entity_text, vector,
                                   gliner_label=gliner_label)
    if not cands:
        return None
    if tier == "3" and cands[0]["similarity_score"] < TIER3_SIMILARITY_FLOOR:
        return None
    return {
        "found_domain": cands[0]["domain_id"],
        "expected_domains": domains,
        "tier": tier,
        "candidates": cands,
    }


def normalize_entity(entity_text: str, conn, gliner_label: str = None,
                     domain_override: list = None) -> dict:
    """Maps an entity to OMOP concepts using the 3-tier approach.

    Returns the top candidate's fields (backwards-compatible keys) plus:
      candidates[]      -- up to CANDIDATE_LIMIT ranked candidates
      ambiguous         -- whether Stage 3 should be asked to disambiguate
      ambiguity_reason  -- why, so a reviewer never has to guess

    gliner_label drives both the vocabulary restriction and the (default)
    domain restriction applied to every tier. Passing None applies neither,
    i.e. the pre-2026-08-07 behavior -- callers should pass the real label
    whenever they have one.

    domain_override (2026-08-10, docs/Stage2_Compound_And_Qualifier_Gaps.md
    gap 3), if given, is used INSTEAD of GLINER_LABEL_TO_DOMAIN.get(gliner_label)
    for the domain restriction. WHY THIS EXISTS: src/clinical_pipeline.py's
    compound-split/span-growth detectors run their OWN unrestricted Tier 1/2
    lookup (src/normalization._lookup_tier12()) to confirm a split/grown
    part resolves at all, and that lookup already knows which OMOP domain
    the match actually lives in. Re-deriving a domain from the (possibly
    now-wrong, e.g. a bare qualifier word relabeled from its Procedure
    parent) gliner_label and hoping GLINER_LABEL_TO_DOMAIN happens to have a
    correct entry for it silently throws that evidence away -- concretely,
    "Left" split off "Left craniectomy" resolves during detection, but
    Stage 2b's un-overridden domain restriction (Procedure) excludes
    whatever domain the qualifier-value concept actually lives in, so the
    split's own confirmed match was previously lost on the very next call.
    Passing the detector's own domain straight through is self-consistent
    BY CONSTRUCTION -- it never requires knowing or guessing the correct
    OMOP domain_id string.

    CAN BE MULTIPLE DOMAINS (2026-08-10 refinement, gap 3 follow-up). The
    detector passes EVERY distinct domain_id its own unrestricted lookup
    found across ALL matching rows (_lookup_tier12()'s include_domains
    option), not just its single top-ranked candidate's domain. This
    matters because a real ambiguity was being silently hidden: "Left" has
    multiple distinct exact SNOMED concepts spread across different
    qualifier categories, and collapsing to just the lowest concept_id's
    domain meant this function's tier-1 query only ever saw ONE of them --
    reporting a confident HIGH-tier match that frequently disagreed with
    gold, instead of the len(candidates)>1 ambiguous=True path that already
    exists a few lines below for exactly this situation. Passing the full
    domain set lets that existing ambiguity-detection machinery do its job
    instead of being starved of the evidence it needs.
    """
    search_text = entity_text.lower().strip()
    vocabs = VOCAB_BY_LABEL.get(gliner_label, DEFAULT_VOCAB)
    domains = domain_override if domain_override is not None else GLINER_LABEL_TO_DOMAIN.get(gliner_label)
    domain_clause = f" AND domain_id IN ({_in_clause(domains)})" if domains else ""

    # Records which tiers were attempted and what each returned. Previously
    # only the WINNING tier survived in match_tier, so "Tier 1 and 2 found
    # nothing and Tier 3 scored 0.71" was indistinguishable from "Tier 3 scored
    # 0.71" with the earlier tiers never tried. Stage 3's deeper-resolution
    # mode is supposed to reason over exactly this search path, and it was
    # being thrown away before it got there.
    trace = []

    # ==========================================
    # TIER 1: Exact Lexical Match
    # ==========================================
    tier1_query = f"""
    SELECT concept_id, concept_name, domain_id, vocabulary_id
    FROM athena_concept
    WHERE lower(concept_name) = ?
    AND standard_concept = 'S'
    AND vocabulary_id IN ({_in_clause(vocabs)})
    {domain_clause}
    ORDER BY concept_id ASC
    LIMIT {CANDIDATE_LIMIT};
    """
    rows = conn.sql(tier1_query, params=[search_text, *vocabs, *(domains or [])]).fetchall()
    trace.append({"tier": "1 (Exact)", "attempted": True, "hits": len(rows)})
    if rows:
        # 2026-08-13 (P1.1). Two changes, both gated on RANKED_TIER12:
        #
        #  (a) candidates[0] is chosen by _rank_tier12_candidates() rather than
        #      by lowest concept_id -- the fix for the 52.71% Tier 1 accuracy
        #      in the 2026-08-13 report S5.3.
        #
        #  (b) `ambiguous` now means "ranking could NOT separate the top two",
        #      not "there was more than one row". That distinction is the point:
        #      the old flag fired on every multi-match, so Stage 3 was handed a
        #      flood of cases the ranking can now settle, while the genuinely
        #      undecidable ones were silently resolved by integer id and never
        #      flagged at all. Exactly inverted from what is useful.
        rows, basis, unresolved = (
            _rank_tier12_candidates(conn, rows, gliner_label, entity_text,
                                    domains=domains)
            if RANKED_TIER12 else (rows, "concept_id_asc", len(rows) > 1))
        cands = [_candidate(r, "1 (Exact)", 1.0) for r in rows]
        ambiguous = unresolved if RANKED_TIER12 else len(cands) > 1
        return _result(
            cands,
            ambiguous=ambiguous,
            reason=("unresolved_exact_tie_after_ranking" if RANKED_TIER12
                    else "multiple_exact_concept_name_matches") if ambiguous else None,
            trace=trace,
            rank_basis=basis,
        )

    # ==========================================
    # TIER 2: Synonym Lexical Match
    # ==========================================
    domain_clause2 = f" AND c.domain_id IN ({_in_clause(domains)})" if domains else ""
    tier2_query = f"""
    SELECT DISTINCT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id
    FROM athena_concept_synonym s
    JOIN athena_concept c ON s.concept_id = c.concept_id
    WHERE lower(s.concept_synonym_name) = ?
    AND c.standard_concept = 'S'
    AND c.vocabulary_id IN ({_in_clause(vocabs)})
    {domain_clause2}
    ORDER BY c.concept_id ASC
    LIMIT {CANDIDATE_LIMIT};
    """
    rows = conn.sql(tier2_query, params=[search_text, *vocabs, *(domains or [])]).fetchall()
    trace.append({"tier": "2 (Synonym)", "attempted": True, "hits": len(rows)})
    if rows:
        rows, basis, unresolved = (
            _rank_tier12_candidates(conn, rows, gliner_label, entity_text,
                                    domains=domains)
            if RANKED_TIER12 else (rows, "concept_id_asc", len(rows) > 1))
        cands = [_candidate(r, "2 (Synonym)", 1.0) for r in rows]
        ambiguous = unresolved if RANKED_TIER12 else len(cands) > 1
        return _result(
            cands,
            ambiguous=ambiguous,
            reason=("unresolved_synonym_tie_after_ranking" if RANKED_TIER12
                    else "multiple_exact_synonym_matches") if ambiguous else None,
            trace=trace,
            rank_basis=basis,
        )

    # ==========================================
    # TIER 3: Semantic Vector Match (SapBERT)
    # ==========================================
    vector = get_sapbert_embedding(entity_text)
    alias_ids = _alias_expand_brand_to_generic(conn, search_text)
    rows = _tier3_semantic_rows(conn, vector, vocabs, domains, alias_ids=alias_ids)
    trace.append({
        "tier": "3 (Semantic)", "attempted": True, "hits": len(rows),
        "top_score": round(rows[0][4], 4) if rows else None,
        "runner_up_score": round(rows[1][4], 4) if len(rows) > 1 else None,
        "floor": TIER3_SIMILARITY_FLOOR,
        "alias_expanded": bool(alias_ids),
    })

    if not rows:
        conflict = _detect_domain_conflict(conn, search_text, vocabs, domains, entity_text,
                                           vector, gliner_label=gliner_label)
        if conflict:
            return _result(conflict["candidates"], ambiguous=True,
                           reason="label_domain_conflict", failed=True, conflict=conflict,
                           trace=trace)
        return _result([], ambiguous=True, reason="no_candidates_at_any_tier", failed=True,
                       trace=trace)

    cands = [_candidate(r, "3 (Semantic)", round(r[4], 4),
                       match_basis="verified_brand_alias" if r[0] in alias_ids else None)
            for r in rows]
    cands = _collapse_hierarchy_duplicates(conn, cands)
    top = cands[0]["similarity_score"]

    # Below the floor, the top hit is reported as a FAILED match but the
    # candidates are still forwarded: "close but not close enough" is exactly
    # the case Stage 3's deeper-resolution mode exists for, and discarding the
    # near-misses would leave it nothing to reason about.
    if top < TIER3_SIMILARITY_FLOOR:
        # A weak in-domain match may still be beaten by a strong out-of-domain
        # one, which is itself the signal that the label was wrong -- so the
        # conflict check runs here too, not only on a total miss.
        conflict = _detect_domain_conflict(conn, search_text, vocabs, domains, entity_text,
                                           vector, gliner_label=gliner_label)
        if conflict:
            return _result(conflict["candidates"], ambiguous=True,
                           reason="label_domain_conflict", failed=True, conflict=conflict,
                           trace=trace)
        cands, fuzzy_added = _merge_fuzzy(conn, search_text, vocabs, domains, cands, trace)
        reason = "tier3_below_similarity_floor_fuzzy_added" if fuzzy_added else \
            "tier3_below_similarity_floor"
        return _result(cands, ambiguous=True, reason=reason, failed=True, trace=trace)

    if len(cands) > 1 and (top - cands[1]["similarity_score"]) < TIER3_AMBIGUITY_MARGIN:
        # Record 3, docs/Stage3_Open_Issues.md Issue 3: three semantically-close
        # misspelled-drug candidates (SPIRILENE/SPIRGETINE/SPIRAPRILAT) crowded
        # out the correctly-spelled concept, which never appeared in the Tier 3
        # list at all. This is exactly the situation the fuzzy supplement
        # targets -- Tier 3 is already telling us it's unsure between several
        # near-tied options, so a near-exact spelling match belongs in that
        # same conversation.
        cands, fuzzy_added = _merge_fuzzy(conn, search_text, vocabs, domains, cands, trace)
        reason = "tier3_top2_margin_below_threshold_fuzzy_added" if fuzzy_added else \
            "tier3_top2_margin_below_threshold"
        return _result(cands, ambiguous=True, reason=reason, trace=trace)

    # 2026-08-13 (Fix 1 follow-up, "Aldactone problem"). A KG-verified brand
    # alias can lose the raw cosine-similarity race to a false-friend string
    # match confident enough to clear BOTH the floor and the margin check
    # above (measured: "Aldactone" -> "propiolactone" at 0.83, margin 0.11 --
    # spironolactone sat at 0.56, present in cands but never reached Stage 3).
    # process_and_normalize_entities() only forwards the full candidate list
    # when ambiguous=True (mapping["candidates"][:1] otherwise, to keep
    # Stage 3's prompt small on genuinely settled matches) -- so an alias hit
    # that isn't candidates[0] must force ambiguous here, or the guarantee
    # Fix 1 exists to provide is silently undone one function later.
    if alias_ids and cands[0]["omop_concept_id"] not in alias_ids \
            and any(c["omop_concept_id"] in alias_ids for c in cands):
        return _result(cands, ambiguous=True, reason="alias_candidate_outranked", trace=trace)

    return _result(cands, ambiguous=False, reason=None, trace=trace)


def _merge_fuzzy(conn, search_text, vocabs, domains, cands, trace):
    """Appends any edit-distance candidates not already present, capped so an
    already-uncertain result cannot grow unboundedly. Mutates nothing;
    returns (merged_candidates, fuzzy_candidates_added)."""
    existing_ids = {c["omop_concept_id"] for c in cands}
    fuzzy = _fuzzy_typo_candidates(conn, search_text, vocabs, domains, exclude_ids=existing_ids)
    trace.append({"tier": "4 (Fuzzy)", "attempted": True, "hits": len(fuzzy)})
    if not fuzzy:
        return cands, []
    merged = (cands + fuzzy)[:CANDIDATE_LIMIT * 2]
    return merged, fuzzy


def _result(candidates: list, ambiguous: bool, reason, failed: bool = False,
            conflict: dict = None, trace: list = None,
            rank_basis: str = None) -> dict:
    """Shapes the return value, keeping the pre-existing top-level keys so
    existing consumers keep working while `candidates` carries the new
    information.

    rank_basis (2026-08-13) records WHICH tiebreak produced candidates[0] --
    "ranked_v1" or "concept_id_asc" -- and is persisted to
    normalized_entities.tier12_rank_basis. Without it the A/B between the two
    orderings would only be measurable per run; with it, a mixed database can
    still be split into the two populations and compared.
    """
    if not candidates or failed:
        top = {
            "match_tier": "0 (Failed)", "concept_id": None, "concept_name": "Unmapped",
            "domain_id": None, "vocab": None,
            "score": candidates[0]["similarity_score"] if candidates else 0.0,
        }
    else:
        c = candidates[0]
        top = {
            "match_tier": c["match_tier"], "concept_id": c["omop_concept_id"],
            "concept_name": c["concept_name"], "domain_id": c["domain_id"],
            "vocab": c["vocabulary_id"], "score": c["similarity_score"],
        }
    top["candidates"] = candidates
    top["ambiguous"] = ambiguous
    top["ambiguity_reason"] = reason
    top["domain_conflict"] = conflict
    top["tier_trace"] = trace
    top["tier12_rank_basis"] = rank_basis
    return top


def compute_confidence_tier(gliner_confidence: float, normalization_ambiguous: bool,
                            expansion_ambiguous: bool, domain_conflict: bool = False,
                            crosses_sentence_boundary: bool = False,
                            short_token_text: str = None,
                            match_tier: str = None) -> tuple:
    """Stage 3's routing tier, and the reasons behind it.

    LOW if ANY of these independent signals fires. They capture genuinely
    different failure modes and none subsumes the others:
      * gliner_confidence (HIGH_GLINER_RISK_FLOOR) -- "the model was
        confident, and confidence is inversely related to accuracy here"
      * normalization             -- "which concept is it"
      * expansion                 -- "was the abbreviation read correctly at all"
      * domain_conflict           -- "was the extraction LABEL even right"
      * crosses_sentence_boundary -- "is this even ONE entity"
      * short_token_text          -- "is this a bare abbreviation Stage 2b's
        exact/synonym/semantic tiers are structurally unlikely to resolve
        correctly" (three sub-signals: short_token, isupper_abbreviation,
        alnum_mix -- see the constants above this function for what each
        catches and the measured hit rate that shaped its bound)
    An entity can have a single clean, unambiguous OMOP match and still deserve
    deeper resolution because the span itself was shaky, or because it came
    from an abbreviation with three plausible readings.

    short_token_text is expected to be the entity's ORIGINAL (pre-expansion)
    surface form -- the heuristics were measured against original_text in
    scripts/measure_heuristic_and_boundary.py, and process_and_normalize_
    entities() passes orig_text here for that reason. Passing None (the
    default) skips all three sub-signals, preserving prior behavior for any
    other caller that hasn't been updated to supply it.

    2026-08-11 TIER-GATED EXEMPTION (not label-gated -- see below). match_tier,
    when it is "1 (Exact)" or "2 (Synonym)", suppresses the short_token/
    isupper_abbreviation/alnum_mix sub-signals entirely. Rationale: a bare
    abbreviation like "WBC"/"RBC"/"MCV" that Stage 2b's own exact or synonym
    vocabulary already resolved cleanly doesn't need Stage 3's deeper
    resolution just for being short -- the match itself is high-precision.

    This is deliberately gated on match_tier, NOT on gliner_label ==
    "Lab Test" -- an earlier version of this fix proposed exempting the
    entire Lab Test label from the short-token penalty, which was rejected:
    it would also exempt genuinely risky short lab abbreviations (K, Na, Ca,
    Cl) that only resolved via a weaker Tier 3 (Semantic)/Tier 4 (Fuzzy)
    match, not just the safe ones that motivated the original proposal. A
    Tier 3/4 "K" match is exactly the kind of shaky resolution this whole
    heuristic exists to catch, so it must still fire. Only Tier 1/2 -- the
    tiers structurally unlikely to be wrong -- get the exemption. match_tier
    of None (unmeasured, or an older caller that hasn't been updated) does
    NOT qualify for the exemption, preserving the pre-existing conservative
    default.

    2026-08-13 SUPERSEDED -- ALL TIER-1/2 EXEMPTIONS REMOVED. The paragraph
    above describes behaviour this function no longer has. It is retained
    because the reasoning it records (why the exemption was gated on
    match_tier rather than on gliner_label == "Lab Test") is still sound and
    still worth not re-litigating; only its premise turned out to be false.

    The premise was that Tier 1/2 are "structurally unlikely to be wrong".
    evaluation/stage2b_cal_eval.py measured them at 52.71% (n=776) and 61.84%
    (n=532) against gold. On 2026-08-12 the weak_match_tier rule below dropped
    its own exemption in response; the other two were left in place and
    described as "harmless no-ops", on the grounds that weak_match_tier
    already forces LOW for the same entities.

    They were harmless for ROUTING and harmful for MEASUREMENT. Because the
    gates sat around the `reasons.append(...)` calls rather than around the
    tier decision, a Tier 1/2 entity's tier_reasons recorded only
    "weak_match_tier" no matter how many other signals were true of it. The
    next iteration -- deciding which of these signals actually predicts
    correctness, and therefore which exemption could be safely restored --
    needs precisely the data those gates were throwing away. Both are now
    removed. Routing behaviour is bit-identical (weak_match_tier already
    forces LOW for every known tier, and `reasons` is OR'd); what changes is
    that tier_reasons is now a complete record rather than a censored one.
    """
    reasons = []
    # 2026-08-11: polarity flipped from "low_gliner_confidence" (was: fire
    # below a floor) to "high_gliner_risk" (fires at/above a floor) -- see
    # HIGH_GLINER_RISK_FLOOR's definition above for the inverted-calibration
    # finding that motivated this.
    #
    # 2026-08-13 TIER-GATED EXEMPTION, same pattern and rationale as the
    # short_token/isupper_abbreviation/alnum_mix trio above: measured via
    # scripts/measure_gliner_risk_vs_match_tier.py over the 28-note re-
    # ingested batch. Within EITHER match_tier bucket, high_gliner_risk adds
    # no real discriminative signal -- confirmed-Tier-1/2 accuracy is 57.7%
    # whether high_gliner_risk fires or not (57.7% vs 56.2%), and in the
    # unconfirmed-Tier-3/4/0 bucket high_gliner_risk entities were actually
    # MORE accurate, not less (33.3% vs 27.5%) -- the opposite of what "risk"
    # should predict. match_tier is doing the real work; gate this signal on
    # it the same way, so a high GLiNER-confidence span that Stage 2b's own
    # exact/synonym vocabulary already confirmed doesn't get pushed to
    # Stage 3 purely for a confidence score that measurement shows isn't
    # predictive once match_tier is known. match_tier of None (unmeasured)
    # does NOT qualify, same conservative default as the trio's gate.
    #
    # 2026-08-13 EXEMPTION REMOVED (docs/2026-08-13_Code_Improvement_
    # Proposals.md P1.2). The Tier-1/2 gate that used to wrap this append is
    # gone. ROUTING IS UNCHANGED -- weak_match_tier below already forces LOW
    # for every known match_tier, and `reasons` is OR'd -- but the RECORD is
    # not: while the gate stood, high_gliner_risk was never appended for a
    # Tier 1/2 entity, so tier_reasons was systematically blank on exactly the
    # population you would have to study to justify ever restoring the
    # exemption. Measuring a signal requires observing it fire. The gate
    # suppressed the observation, not the risk.
    if gliner_confidence is not None and gliner_confidence >= HIGH_GLINER_RISK_FLOOR:
        reasons.append("high_gliner_risk")
    if normalization_ambiguous:
        reasons.append("normalization_ambiguous")
    if expansion_ambiguous:
        reasons.append("expansion_ambiguous")
    # A cross-domain match means the extraction LABEL is in question, not just
    # the concept -- always worth Stage 3's deeper-resolution path, since a
    # mislabelled entity that silently resolves would carry a wrong domain into
    # KG3 and into every guideline lookup made against it thereafter.
    if domain_conflict:
        reasons.append("label_domain_conflict")
    # 2026-08-10: measured concretely on 17751158-DS-19 via
    # scripts/score_gold_recall.py's compound-span detection -- GLiNER
    # extracted "Gunshot wound to abdomen\nPneumonia" as ONE entity spanning
    # two PyRuSH sentences, merging an unrelated diagnosis-list line into a
    # single concept that can never link correctly to either gold annotation
    # it overlaps. See src/entity_extraction.py's sentence_ids_spanned() for
    # why sentence boundaries (not raw "\n") are the right signal -- a
    # line-wrapped continuation of one statement stays inside one sentence,
    # so this does not fire on those.
    if crosses_sentence_boundary:
        reasons.append("entity_span_crosses_sentence_boundary")

    # 2026-08-11 short-token/abbreviation signals. Independent of each other
    # (an entity can trip more than one) and independent of gliner_confidence
    # -- a short abbreviation reported at high confidence is still a short
    # abbreviation.
    #
    # 2026-08-13: Tier-1/2 exemption removed here too, same reasoning as
    # high_gliner_risk above and with the same zero routing impact. These three
    # sub-signals are the ones with concrete measured hit rates behind them
    # (scripts/measure_heuristic_and_boundary.py); censoring them on Tier 1/2
    # made those hit rates unmeasurable on more than half the corpus.
    if short_token_text:
        t = short_token_text.strip()
        if t:
            if len(t) <= SHORT_TOKEN_MAX_LEN:
                reasons.append("short_token")
            if len(t) <= ISUPPER_MAX_LEN and t.isupper() and any(c.isalpha() for c in t):
                reasons.append("isupper_abbreviation")
            if _ALNUM_MIX_RE.search(t):
                reasons.append("alnum_mix")

    # 2026-08-12 weak_match_tier. Independent of every signal above --
    # measured concretely on a real re-ingestion run: Glucose-72 and
    # UreaN-15/UreaN-17 resolved via Tier 3 (SapBERT semantic similarity,
    # not an exact/synonym confirmation) and STILL landed HIGH tier, because
    # nothing above checks which tier actually produced the match -- only
    # whether it was ambiguous. A Tier 3/4 match can be perfectly
    # unambiguous (no close runner-up candidate) and still be wrong; that is
    # exactly what Tier 3/4 mean here (see _lookup_tier12()'s own docstring:
    # "Tier 1/2 only keeps false-split/false-growth risk low" -- the same
    # principle applies to ordinary normalization, not just compound-split
    # candidates). This closes that gap: any match_tier other than Exact/
    # Synonym routes LOW regardless of label, ambiguity, or every other
    # signal -- a semantic guess should never silently outrank Stage 3
    # review just because no second candidate happened to compete with it.
    #
    # 2026-08-13 CORRECTION -- Tier 1/2 exemption removed. This rule (and
    # the high_gliner_risk / short_token exemptions above it) exempted Tier
    # 1 "Exact" and Tier 2 "Synonym" on the assumption that those tiers are
    # "structurally unlikely to be wrong" (see the Tier-gated-exemption
    # comment above this function, 2026-08-11). evaluation/stage2b_cal_eval.py
    # run against the 30-note corpus directly measured that assumption as
    # false: Tier 1 is 52.71% accurate against gold (776 gradable), Tier 2
    # is 61.84% (532 gradable) -- neither clears any reasonable bar for
    # "high-precision, skip review." Dropping the tuple exclusion here means
    # every entity with a KNOWN match_tier (1, 2, 3, or 0) now routes LOW
    # regardless of tier -- there is currently no match_tier value this
    # pipeline's own measurement can call trustworthy enough to silently
    # skip Stage 3. The high_gliner_risk and short_token Tier-1/2
    # exemptions above are left as-is (harmless now: this rule already
    # forces LOW for the same entities via `reasons`, an OR'd list), rather
    # than also rewritten, to keep this a single, auditable line-level
    # change against a specific measurement rather than a broader rewrite
    # under time pressure.
    #
    # match_tier of None (unmeasured, or an older caller that hasn't been
    # updated) does NOT trigger this -- same conservative-default choice as
    # the short-token exemption above, for the same reason: "we don't know"
    # must not be silently treated as either "safe" or "risky".
    if match_tier is not None:
        reasons.append("weak_match_tier")

    return ("LOW" if reasons else "HIGH"), reasons


def process_and_normalize_entities(extracted_entities: list, conn, is_test: bool = False) -> list:
    """Normalizes Stage 2a entities against OMOP and persists results.

    Takes the list of DICTS that entity_extraction.extract_and_store_entities()
    now returns (it previously took positional tuples -- see that module's
    docstring for why the shape changed).

    DEDUP FAN-OUT: normalization is computed once per distinct
    (expanded_text, gliner_label) pair, then that result is written out for
    EVERY entity_id sharing it. The old orchestrator deduplicated entities
    before this function ever saw them, which saved the redundant SapBERT calls
    but also meant only one of several identical mentions got a normalization
    row at all -- so mentions 2..n had no concept mapping, and once Stage 3
    runs per entity_id they would have had nothing to validate. Computing once
    and fanning out keeps the saving without the data loss.
    """
    conn.sql("""
    CREATE TABLE IF NOT EXISTS normalized_entities (
        note_id VARCHAR,
        original_text VARCHAR,
        expanded_text VARCHAR,
        gliner_label VARCHAR,
        gliner_confidence FLOAT,
        omop_concept_id BIGINT,
        omop_concept_name VARCHAR,
        omop_domain VARCHAR,
        omop_vocab VARCHAR,
        match_tier VARCHAR,
        similarity_score FLOAT,
        is_test BOOLEAN DEFAULT FALSE,
        UNIQUE(note_id, original_text, expanded_text, gliner_label)
    );
    """)
    for ddl in [
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS is_test BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS entity_id VARCHAR;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS candidates JSON;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS is_ambiguous BOOLEAN DEFAULT FALSE;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS ambiguity_reason VARCHAR;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS confidence_tier_in VARCHAR;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS tier_reasons JSON;",
        # Records that a good concept was found under a DIFFERENT OMOP domain
        # than the GLiNER label predicted. Lets the evaluation decompose
        # "0 (Failed)" into genuinely-unmappable vs mislabelled-but-mappable,
        # which were previously indistinguishable.
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS domain_conflict JSON;",
        # Which tiers ran and what each returned, including the Tier-3
        # runner-up score. Lets an auditor see the search path, not just its
        # outcome.
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS tier_trace JSON;",
        # docs/Provenance_Schema.md Stage 2b fields that were never written.
        # `matched` is redundant with match_tier != '0 (Failed)' but is what the
        # schema specifies and what downstream filters will actually read --
        # a boolean is far harder to get wrong than string-matching a tier label.
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS matched BOOLEAN;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS normalized_from VARCHAR;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS athena_vocabulary_release VARCHAR;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS domain_id_queried JSON;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS vocab_queried JSON;",
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS sapbert_pooling_method VARCHAR;",
        # 2026-08-13: which Tier 1/2 ranking produced the top-1 pick. NULL on
        # rows written before RANKED_TIER12 existed, "concept_id_asc" when the
        # legacy arbitrary tiebreak ran, "ranked_v1" when the ranker did. This
        # is what makes the A/B in report S5.3 gradable per row rather than
        # per run -- see _rank_tier12_candidates().
        "ALTER TABLE normalized_entities ADD COLUMN IF NOT EXISTS tier12_rank_basis VARCHAR;",
    ] + provenance_alter_statements("normalized_entities"):
        conn.sql(ddl)

    vocab_release = get_athena_vocabulary_release(conn)
    cache = {}
    normalized_results = []
    rows_to_insert = []

    for ent in extracted_entities:
        note_id = ent["note_id"]
        orig_text = ent["original_text"]
        expanded_text = ent["expanded_text"]
        label = ent["entity_label"]
        # See normalize_entity()'s domain_override docstring -- set only on
        # compound-split/span-growth-produced entities
        # (src/clinical_pipeline.py); ordinary Stage 2a entities have no
        # such key, so .get() returns None and behavior is unchanged.
        domain_override = ent.get("domain_override")

        # Normalization is a pure function of (text, label, domain_override)
        # within a note, so the cache is the fan-out mechanism: identical
        # mentions share one computation but each still gets its own row
        # keyed by entity_id. domain_override is part of the key because two
        # entities can share (text, label) yet carry different overrides --
        # e.g. the SAME bare qualifier word split off two different parent
        # labels in one note.
        cache_key = (expanded_text, label,
                    tuple(domain_override) if domain_override else None)
        if cache_key not in cache:
            mapping = normalize_entity(expanded_text, conn, gliner_label=label,
                                       domain_override=domain_override)
            normalized_from = "expanded"

            # ORIGINAL-FORM FALLBACK. Normalisation runs on expanded_text, so a
            # wrong abbreviation expansion poisons it with no way to recover --
            # and wrong expansions are real: measured on note 10000032-DS-21,
            # `NAD` expanded to "nicotinamide adenine dinucleotide" (the
            # alphabetically-first meaning) rather than "no acute distress",
            # and `TTP` to "Thrombotic Thrombocytopenic Purpura" rather than
            # "tenderness to palpation". When the expanded form fails to map,
            # the raw abbreviation is often the better key -- `SOB`, `EGD` and
            # `INR` all resolve directly.
            #
            # Only fires on failure, and only when the two forms actually
            # differ, so it costs nothing on the common path. `normalized_from`
            # records which one produced the result, so a reviewer can see that
            # the expansion was bypassed rather than having to infer it.
            if (mapping["match_tier"] == "0 (Failed)"
                    and orig_text and orig_text.strip().lower() != expanded_text.strip().lower()):
                retry = normalize_entity(orig_text, conn, gliner_label=label,
                                         domain_override=domain_override)
                if retry["match_tier"] != "0 (Failed)":
                    mapping = retry
                    normalized_from = "original_after_expanded_failed"

            # LAB VALUE SUFFIX FALLBACK (2026-08-10, extended 2026-08-11 to
            # try MULTIPLE candidates -- see strip_lab_value_suffix()'s
            # docstring and this module's "2026-08-11 DELIMITER-LESS CASE"
            # comment above for the full rationale). Tries whichever text
            # form(s) actually carry the pattern -- expanded first (matching
            # the two fallbacks above), then original, in case abbreviation
            # expansion altered the hyphen/spacing -- and within each text
            # form, tries every candidate strip_lab_value_suffix() returns
            # in order.
            #
            # 2026-08-12 GATE WIDENED from "== 0 (Failed)" to "not in
            # (1, 2)": a Tier 1/2 exact/synonym match on the cleanly-stripped
            # name should always be preferred over ANY weaker result on the
            # raw noisy text, not only over a total failure.
            #
            # 2026-08-12 ACCEPTANCE RULE, ATTEMPT 1 AND WHY IT REGRESSED.
            # First tried "only adopt if the stripped candidate reaches Tier
            # 1/2" -- reasoning that a Tier 3/4 stripped guess is not a
            # verified improvement. Re-ingestion promptly disproved this:
            # WBC-7, RBC-3, Hgb-9.9, MCV-97, MCHC-32, RDW-12, Glucose-72 and
            # UreaN-15/17 all went from "Tier 3, some plausible concept" to
            # "0 (Failed), Unmapped". Root cause: the RAW text ("WBC-7") was
            # already below TIER3_SIMILARITY_FLOOR and reporting "0 (Failed)"
            # even before today -- the OLD fallback (gated the same way) was
            # ALREADY firing and adopting whatever the stripped text got
            # (its old acceptance bar was simply "not 0 (Failed)"), which is
            # how "WBC-7" ever showed "Leucocyte count" via Tier 3 on "WBC"
            # in the first place. Requiring Tier 1/2 specifically rejected
            # that stripped Tier-3 result and fell back to the ORIGINAL
            # raw-text mapping -- itself "0 (Failed)" -- discarding a useful
            # near-miss candidate instead of improving on it.
            #
            # FIXED via explicit tier-rank comparison: adopt the stripped
            # candidate whenever its tier is STRICTLY BETTER (lower rank)
            # than whatever mapping already has, not only when it reaches
            # Tier 1/2. This restores the old "surface the best available
            # near-miss to Stage 3" behavior for the Failed->Tier3 case,
            # while still fixing the original problem this gate exists for
            # (a Tier 1/2 match on "WBC" beating a Tier 3 guess on "WBC-7")
            # and never adopting a same-or-worse-tier stripped guess just
            # because it happens to be a different string. Stops searching
            # further candidates only once Tier 1/2 is reached -- nothing
            # can beat a confirmed match, so there is no reason to keep
            # trying once one is found; a Tier 3/4 improvement keeps the
            # best seen so far but keeps looking, in case a LATER candidate
            # in the list reaches the confirmed tier instead (see strip_lab_
            # value_suffix()'s own docstring: "a wrong candidate earlier in
            # the list can never win over a right one later in it" -- that
            # guarantee only holds if a merely-better-but-unconfirmed early
            # hit does not stop the search).
            if mapping["match_tier"] not in ("1 (Exact)", "2 (Synonym)") and label == "Lab Test":
                pre_strip_tier = mapping["match_tier"]
                best_mapping = mapping
                best_source = None
                confirmed = False
                for candidate_text, source_name in (
                        (expanded_text, "expanded"), (orig_text, "original")):
                    for stripped in strip_lab_value_suffix(candidate_text or ""):
                        retry = normalize_entity(stripped, conn, gliner_label=label,
                                                 domain_override=domain_override)
                        if _LAB_TIER_RANK.get(retry["match_tier"], 9) < \
                                _LAB_TIER_RANK.get(best_mapping["match_tier"], 9):
                            best_mapping = retry
                            best_source = f"value_stripped_from_{source_name}:{stripped}"
                        if retry["match_tier"] in ("1 (Exact)", "2 (Synonym)"):
                            confirmed = True
                            break
                    if confirmed:
                        break
                if best_source:
                    mapping = best_mapping
                    # Records the winning candidate string AND what tier it
                    # upgraded from, not just which text form it came from --
                    # with multiple candidates now possible per form,
                    # "expanded" alone no longer says whether "RR" or "RR1"
                    # was the one that won, and the upgrade note makes it
                    # visible in an audit whether this reached a confirmed
                    # Tier 1/2 match or only improved on a weaker guess.
                    normalized_from = f"{best_source} (upgraded_from_{pre_strip_tier})"

            mapping = dict(mapping)
            mapping["normalized_from"] = normalized_from
            cache[cache_key] = mapping
        mapping = cache[cache_key]

        conflict = mapping.get("domain_conflict")
        tier, tier_reasons = compute_confidence_tier(
            ent.get("confidence"),
            mapping["ambiguous"],
            ent.get("expansion_ambiguous", False),
            domain_conflict=bool(conflict),
            crosses_sentence_boundary=ent.get("crosses_sentence_boundary", False),
            short_token_text=orig_text,
            match_tier=mapping["match_tier"],
        )

        # Only forward alternatives when there is something to choose between;
        # a confident, unambiguous match ships a single candidate so Stage 3's
        # prompt doesn't spend tokens on options nobody is choosing between.
        forwarded = mapping["candidates"] if mapping["ambiguous"] else mapping["candidates"][:1]

        record = {
            "entity_id": ent["entity_id"],
            "note_id": note_id,
            "original_text": orig_text,
            "expanded_text": expanded_text,
            "gliner_label": label,
            "gliner_confidence": round(ent["confidence"], 4) if ent.get("confidence") is not None else None,
            "omop_concept_id": mapping["concept_id"],
            "omop_concept_name": mapping["concept_name"],
            "omop_domain": mapping["domain_id"],
            "omop_vocab": mapping["vocab"],
            "match_tier": mapping["match_tier"],
            "similarity_score": mapping["score"],
            "candidates": forwarded,
            "is_ambiguous": mapping["ambiguous"],
            "ambiguity_reason": mapping["ambiguity_reason"],
            "confidence_tier_in": tier,
            "tier_reasons": tier_reasons,
            "domain_conflict": conflict,
            "tier_trace": mapping.get("tier_trace"),
            "matched": mapping["match_tier"] != "0 (Failed)",
            "normalized_from": mapping.get("normalized_from", "expanded"),
        }
        normalized_results.append(record)

        rows_to_insert.append((
            note_id, orig_text, expanded_text, label, record["gliner_confidence"],
            mapping["concept_id"], mapping["concept_name"], mapping["domain_id"],
            mapping["vocab"], mapping["match_tier"], mapping["score"], is_test,
            ent["entity_id"], json.dumps(forwarded), mapping["ambiguous"],
            mapping["ambiguity_reason"], tier, json.dumps(tier_reasons),
            json.dumps(conflict) if conflict else None,
            json.dumps(mapping.get("tier_trace")),
            mapping["match_tier"] != "0 (Failed)",
            mapping.get("normalized_from", "expanded"),
            vocab_release,
            json.dumps(GLINER_LABEL_TO_DOMAIN.get(label)),
            json.dumps(VOCAB_BY_LABEL.get(label, DEFAULT_VOCAB)),
            SAPBERT_POOLING,
            mapping.get("tier12_rank_basis"),
            # Hashed over `forwarded` -- the list Stage 3 will actually be
            # shown -- so mollm_decisions.candidates_hash and this column are
            # directly comparable. A mismatch on the same entity_id is exactly
            # the candidate-list drift report S6 could not previously detect.
            *provenance_params(candidates=forwarded),
        ))

    if rows_to_insert:
        conn.executemany(f"""
        INSERT INTO normalized_entities
        (note_id, original_text, expanded_text, gliner_label, gliner_confidence,
         omop_concept_id, omop_concept_name, omop_domain, omop_vocab,
         match_tier, similarity_score, is_test, entity_id, candidates,
         is_ambiguous, ambiguity_reason, confidence_tier_in, tier_reasons,
         domain_conflict, tier_trace, matched, normalized_from,
         athena_vocabulary_release, domain_id_queried, vocab_queried,
         sapbert_pooling_method, tier12_rank_basis, {provenance_column_sql()})
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                {provenance_placeholders()})
        ON CONFLICT (note_id, original_text, expanded_text, gliner_label) DO UPDATE SET
            gliner_confidence = EXCLUDED.gliner_confidence,
            omop_concept_id = EXCLUDED.omop_concept_id,
            omop_concept_name = EXCLUDED.omop_concept_name,
            omop_domain = EXCLUDED.omop_domain,
            omop_vocab = EXCLUDED.omop_vocab,
            match_tier = EXCLUDED.match_tier,
            similarity_score = EXCLUDED.similarity_score,
            is_test = EXCLUDED.is_test,
            entity_id = EXCLUDED.entity_id,
            candidates = EXCLUDED.candidates,
            is_ambiguous = EXCLUDED.is_ambiguous,
            ambiguity_reason = EXCLUDED.ambiguity_reason,
            confidence_tier_in = EXCLUDED.confidence_tier_in,
            tier_reasons = EXCLUDED.tier_reasons,
            domain_conflict = EXCLUDED.domain_conflict,
            tier_trace = EXCLUDED.tier_trace,
            matched = EXCLUDED.matched,
            normalized_from = EXCLUDED.normalized_from,
            athena_vocabulary_release = EXCLUDED.athena_vocabulary_release,
            domain_id_queried = EXCLUDED.domain_id_queried,
            vocab_queried = EXCLUDED.vocab_queried,
            sapbert_pooling_method = EXCLUDED.sapbert_pooling_method,
            tier12_rank_basis = EXCLUDED.tier12_rank_basis,
            -- Provenance is OVERWRITTEN on conflict, deliberately: this row
            -- now describes the latest normalization run, so its created_at/
            -- run_id/code_version/candidates_hash must describe that run too.
            -- Leaving the original values would make the row claim provenance
            -- it no longer has -- worse than no provenance at all.
            created_at = EXCLUDED.created_at,
            run_id = EXCLUDED.run_id,
            code_version = EXCLUDED.code_version,
            candidates_hash = EXCLUDED.candidates_hash;
        """, rows_to_insert)

    return normalized_results
