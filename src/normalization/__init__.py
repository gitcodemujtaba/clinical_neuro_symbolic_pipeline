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


# 2026-08-14: split into src/normalization/ (see docs/2026-08-14_Dead_Code_Audit.md §6 for
# the rationale). This __init__.py re-exports every name the single-file module used to
# expose -- including underscore-prefixed "private" helpers -- because several call sites
# (tests/test_tier12_ranking.py, tests/test_confidence_tier_reasons.py, ad-hoc diagnostic
# scripts) reach into those directly via `import src.normalization as N; N._whatever(...)`.
# `from src.normalization import X` and `import src.normalization as N; N.X` both continue
# to work exactly as before for every one of these names; only the file they physically
# live in changed.

from .constants import (
    PROJECT_DIR,
    DB_PATH,
    GLINER_LABEL_TO_DOMAIN,
    VOCAB_BY_LABEL,
    DEFAULT_VOCAB,
    DOMAIN_TO_GLINER_LABEL,
    TIER3_SIMILARITY_FLOOR,
    CANDIDATE_LIMIT,
    TIER3_AMBIGUITY_MARGIN,
    RANKED_TIER12,
    TIER12_RANK_SEMANTIC,
    TIER12_CLASS_PREFERENCE,
    TIER12_CLASS_DEMOTED,
    TIER12_TIE_EPSILON,
    HIGH_GLINER_RISK_FLOOR,
    SHORT_TOKEN_MAX_LEN,
    ISUPPER_MAX_LEN,
    _ALNUM_MIX_RE,
    FUZZY_MAX_EDIT_DISTANCE,
    FUZZY_MIN_TEXT_LENGTH,
)
from .sapbert_model import (
    MODEL_NAME,
    SAPBERT_POOLING,
    tokenizer,
    sapbert,
    get_sapbert_embedding,
    _cosine,
)
from .vocab_release import (
    _VOCAB_RELEASE,
    get_athena_vocabulary_release,
)
from .text_utils import (
    _in_clause,
    _CONNECTOR_WORDS,
    _TOKEN_RE,
    _tokens_with_offsets,
    _trim_connectors,
)
from .compound_span import (
    _MIN_SPLIT_HALF_CHARS,
    _SHORT_LATERALITY_TOKENS,
    _MAX_SPLIT_PARTS,
    _lookup_tier12,
    _tier12_domains,
    _partition_token_ranges,
    find_compound_split,
    _MAX_GROW_TOKENS,
    find_span_growth,
    _LAB_VALUE_HYPHEN_RE,
    _LAB_VALUE_SPACE_RE,
    _DELIMITER_LESS_SHAPE_RE,
    _SHORT_LAB_TOKENS,
    _LAB_TIER_RANK,
    strip_lab_value_suffix,
)
from .tier_retrieval import (
    _TIER_DEFAULT_MATCH_BASIS,
    _candidate,
    _concept_class_map,
    _class_rank,
    _specificity,
    _rank_tier12_candidates,
    _fuzzy_typo_candidates,
    _alias_expand_brand_to_generic,
    _collapse_hierarchy_duplicates,
    _tier3_semantic_rows,
    _tier3_hybrid_rows,
    _rrf_scores,
    _tier_queries,
    _detect_domain_conflict,
    HYBRID_RETRIEVAL_ENABLED,
    RRF_K,
    RRF_WEIGHT_DENSE,
    RRF_WEIGHT_SPARSE,
    RRF_WEIGHT_PRIOR,
    RRF_POOL_SIZE,
)
from .orchestrator import (
    normalize_entity,
    _merge_fuzzy,
    _result,
    compute_confidence_tier,
    process_and_normalize_entities,
)

