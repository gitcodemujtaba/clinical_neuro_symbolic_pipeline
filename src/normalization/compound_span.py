"""src/normalization/compound_span.py — compound-span splitting, span-growth detection, lab-value-suffix stripping (split from src/normalization.py, 2026-08-14)."""
import re
import itertools

from .constants import *  # noqa: F401,F403
from .text_utils import _in_clause, _tokens_with_offsets, _trim_connectors
from .tier_retrieval import _candidate, _rank_tier12_candidates

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


# 2026-08-18, confirmed root cause of a real production bug: "Metoprolol
# Succinate" (and "Metoprolol Tartrate") were being split into two separate
# entities ("Metoprolol" + "Succinate"), each independently normalized and
# auto-written to KG3 as unrelated facts. GLiNER itself extracts the phrase
# correctly as ONE span (confirmed via direct re-test) -- the split happens
# here, because "metoprolol succinate" as a bare two-word string has no
# exact/synonym Tier 1/2 hit in this RxNorm dump (it only appears combined
# with dose/form, e.g. as part of a longer Clinical Drug name), so the
# WHOLE-PHRASE GUARD above doesn't fire, the partition search runs, and
# "succinate" alone independently resolves as a real chemical/ingredient
# concept -- passing this function's own "every part must resolve" bar even
# though a drug and its salt form are never two separate clinical facts.
# Scoped to Medication-labeled spans only, and to a closed, well-known set of
# salt/ester suffixes -- not a general veto on short resolvable parts, which
# would reopen the over-atomization risk this function's own REVERT NOTICE
# already documents.
_DRUG_SALT_SUFFIXES = {
    "succinate", "tartrate", "hydrochloride", "sulfate", "sulphate",
    "maleate", "fumarate", "besylate", "mesylate", "citrate", "acetate",
    "phosphate", "bitartrate", "hydrobromide", "nitrate", "gluconate",
    "chloride", "sodium", "potassium", "calcium", "magnesium",
}



# 2026-08-10, gap 1 (docs/Stage2_Compound_And_Qualifier_Gaps.md): the large
# gold note's laterality+device+action procedures ("right EVD placement" ->
# 'right' + 'EVD' + 'placement') need a 3-way split, which the original
# binary-only splitter structurally cannot produce. 4 is headroom past the
# largest real case measured so far (3), not an invitation to explore --
# see _partition_token_ranges()'s docstring for why this bound also keeps
# the search cheap.
_MAX_SPLIT_PARTS = 4




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
            # Drug + salt-suffix guard -- see _DRUG_SALT_SUFFIXES' own
            # comment above. Checked here, before the Tier 1/2 lookup below,
            # since a salt suffix is never a meaningless fragment -- it
            # independently resolves as a real chemical/ingredient concept,
            # which is exactly why it would otherwise pass every check here.
            if gliner_label == "Medication" and g_text.strip().lower() in _DRUG_SALT_SUFFIXES:
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



