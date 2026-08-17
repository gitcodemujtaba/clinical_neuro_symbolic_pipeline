"""
src/abbreviation_flywheel.py -- Stage 1's kg2a_abbreviations dictionary
currently disambiguates a multi-meaning abbreviation via a static tiebreak
chain (numeric-context -> OMOP-groundability -> alphabetical-default,
src/preprocessing.py's expand_text_and_track_offsets()) that knows nothing
about which meaning the rest of the pipeline actually confirms, note after
note. This module is the infrastructure for two data-driven tiebreaks that
consult the pipeline's own accumulated output instead:

1. OBSERVED-FREQUENCY PRIORITY. Every ambiguous-expansion entity whose
   Stage 2b normalization reaches a real tier (not '0 (Failed)') is a weak
   observation: "this meaning, in this section, led somewhere real."
   record_ambiguous_expansion_outcome() logs it unconditionally -- the
   ledger (abbreviation_observed_expansions) is a passive analytics/audit
   table, always populated, never itself gating anything.
   compute_frequency_priority() is a SEPARATE, much stricter consumer of
   that ledger: see "2026-08-17 POSTURE INVERSION" below for why it now
   requires explicit allow-listing before it will ever return a real
   ledger-derived answer.

2. CONTEXT-PATTERN RULES. Once dry_run=False and real SME/HITL review flows
   through hitl_review_queue, mine_context_rules() turns confirmed
   resolutions into (abbreviation, meaning, trigger word, pre/post
   position) rules -- a DETERMINISTIC pre-filter, not a model judgment.
   This matters specifically because src/acronym_escalation.py's own
   raw_local_context() fix already proved that showing a model GOOD context
   is not sufficient on its own: 'PDA' still resolved wrong across all
   three models even with clean, uncorrupted context, because the models'
   own prior toward the more textbook-famous reading overrode it. A
   deterministic lock that never asks a model to judge sidesteps that
   exact failure mode -- structurally the same shape as
   src.mollm_tier_gate.tier3_fast_path(), not a prompt improvement.

ONLY MECHANISM 1 EXCLUDES THE KNOWN-BIAS LIST -- DELIBERATELY ASYMMETRIC, not
an oversight. The abbreviations already proven to have SYSTEMATIC (not
random) bias -- LAD/LCX/LMCA/RCA/PDA/OM/PLV
(src.mollm_tier_gate.CORONARY_SEGMENT_TRAP_ABBREVIATIONS) plus RA/BP/NAD/ASA/
ABD/TTP/CTA/VS/PCA/ICA/AVR/RBC/ISS/CTX/HCT/RFA/WBC (confirmed wrong,
repeatedly, in the corpus-scale acronym-escalation grading) plus anything
matching the short-alphanumeric-code shape (S2/T1/V12, ...) -- must never be
auto-reweighted by mechanism 1 (compute_frequency_priority()): aggregating
the pipeline's OWN confident-but-wrong guesses just formalizes the bias, it
doesn't correct it (the exact circularity risk
src.mollm_tier_calibrator.py's prior_confirmation_count ablation already
demonstrated empirically, at a smaller scale, this session).

Mechanism 2 (mine_context_rules()) is different IN KIND, not just in
maturity, and deliberately does NOT apply this exclusion: its input is real
reviewer-confirmed hitl_review_queue rows, not the pipeline's own guess --
independent ground truth is exactly the kind of evidence that CAN correct a
systematic bias rather than reinforce it. This is the intended eventual
path to relaxing the coronary/short-code hard traps for real, so excluding
them here would defeat the whole point of building it. Until real review
data exists, mine_context_rules() simply has nothing to mine for ANY
abbreviation and returns 0 either way -- the asymmetry only starts to
matter once that data exists, but it must be right from day one, not
patched in later once someone notices LAD never gets picked up.

SAFE BY DEFAULT, same discipline as every other feature added this session:
every lookup function returns None on a miss, a missing table, a DB error,
or conn=None -- the caller's existing tiebreak chain is the fallback, never
interrupted by this module's own failure.

2026-08-17 POSTURE INVERSION -- mechanism 1 flipped from a block-list to an
allow-list, after the FIRST real production-data run (50 train-split notes,
Stage 1->2b) gold-checked 7/7 non-excluded abbreviations as WRONG (DM,
IVF, air, CP, SBP, NC, ACS -- see
docs/2026-08-17_Crosswalk_Fix_And_Flywheel_Production_Run.md). Worse: for
several of those, `selection_basis='observed_frequency_priority'` had
already started appearing in the ledger mid-batch -- i.e. the mechanism was
actively re-selecting its own earlier wrong guesses within the SAME run,
the exact circularity failure mode this module's design was meant to guard
against, just far more widespread than the original ~20-abbreviation
_ADDITIONAL_BIAS_ABBREVIATIONS list anticipated. A 7/7 real-data failure
rate means "block the known-bad ones" was the wrong shape of gate entirely:
compute_frequency_priority() now requires an abbreviation to be explicitly
in VERIFIED_ALLOW_LIST before it will ever return a ledger-derived answer,
full stop -- absence of known bias is no longer sufficient, presence of
verified-safe evidence is required. VERIFIED_ALLOW_LIST starts EMPTY: no
abbreviation from this session's ledger has actually been gold-verified
correct yet (the 7 checked were all wrong; the rest were never checked).
Populate it only from real gold verification, the same discipline that
caught this failure in the first place -- never from "wasn't on the old
exclusion list," which is exactly the reasoning that just failed.
_is_bias_excluded() and _ADDITIONAL_BIAS_ABBREVIATIONS are kept, not
removed, as a second, redundant safety layer: even an abbreviation
mistakenly added to VERIFIED_ALLOW_LIST later still can't win if it matches
a known-bias pattern.
"""
import re

MIN_FREQUENCY_PRIORITY_SUPPORT = 3
FREQUENCY_PRIORITY_MARGIN = 0.20  # winning meaning must lead the runner-up by this much
MIN_CONTEXT_RULE_SUPPORT = 5
CONTEXT_WINDOW_CHARS = 60  # generous char slice, tokenized down to whole words below

# Allow-list gate for mechanism 1 (compute_frequency_priority()) -- see the
# module docstring's "2026-08-17 POSTURE INVERSION" section. Starts EMPTY.
# An abbreviation belongs here only after its dominant ledger meaning has
# been checked against real gold annotations and confirmed correct -- never
# added speculatively or because it "seems safe" or "wasn't flagged."
# Lowercase, matching expand_text_and_track_offsets()'s token_lower lookup.
VERIFIED_ALLOW_LIST = set()

# See module docstring's "POSTURE INVERSION" paragraph -- this list is now a
# secondary, redundant safety net (mechanism 1 requires VERIFIED_ALLOW_LIST
# membership first), kept rather than removed as defense in depth. Sourced
# directly from this session's own confirmed-wrong findings, not guessed.
# Lowercase, matching expand_text_and_track_offsets()'s token_lower lookup.
_ADDITIONAL_BIAS_ABBREVIATIONS = {
    "ra", "bp", "nad", "asa", "abd", "ttp", "cta", "vs", "pca", "ica",
    "avr", "rbc", "iss", "ctx", "hct", "rfa", "wbc",
}


def _is_bias_excluded(abbreviation: str) -> bool:
    """True when this abbreviation must never be auto-reweighted by
    mechanism 1 (observed-frequency priority) -- see module docstring.
    Imports from src.mollm_tier_gate lazily (matching this codebase's
    established pattern for cross-stage references, e.g. route_tier()'s own
    late import of src.mollm_tier_calibrator) so Stage 1's import chain
    doesn't pay for src.llm_client's ollama import at module load time for
    every caller, only when this function actually runs.
    """
    from src.mollm_tier_gate import (
        CORONARY_SEGMENT_TRAP_ABBREVIATIONS, SHORT_ALPHANUMERIC_CODE_RE)
    text = (abbreviation or "").strip().lower()
    if not text:
        return True
    if text in CORONARY_SEGMENT_TRAP_ABBREVIATIONS or text in _ADDITIONAL_BIAS_ABBREVIATIONS:
        return True
    return bool(SHORT_ALPHANUMERIC_CODE_RE.match(text))


# ==========================================================================
# Mechanism 1 -- observed-frequency priority
# ==========================================================================

ABBREVIATION_OBSERVATIONS_DDL = """
CREATE TABLE IF NOT EXISTS abbreviation_observed_expansions (
    abbreviation VARCHAR,
    clinical_context VARCHAR,
    expansion VARCHAR,
    omop_domain VARCHAR,
    selection_basis VARCHAR,
    hit_count INTEGER DEFAULT 1,
    last_updated TIMESTAMP DEFAULT now(),
    PRIMARY KEY (abbreviation, clinical_context, expansion)
);
"""


def record_ambiguous_expansion_outcome(conn, abbreviation: str, clinical_context: str,
                                       expansion: str, omop_domain, selection_basis: str) -> None:
    """Logs one weak observation: this ambiguous abbreviation's picked
    expansion, in this note section, reached a real Stage 2b tier (the
    caller is responsible for only calling this when match_tier != '0
    (Failed)' -- this function does not re-check that itself, same
    separation of concerns as src.acronym_escalation.upsert_acronym_prior()).

    Deliberately a SEPARATE table from acronym_priors: that cache only
    records MoLLM-escalation-CONFIRMED resolutions (a model independently
    judged it), a stronger signal than "Stage 1's own tiebreak heuristic
    happened to reach a real concept" -- conflating the two would let this
    module's weaker evidence dilute Phase 4's already-validated cache.

    No-op (never raises) on conn=None or a DB error, same contract as every
    other recorder in this codebase.
    """
    if conn is None or not abbreviation or not expansion:
        return
    try:
        conn.sql(ABBREVIATION_OBSERVATIONS_DDL)
        conn.execute("""
            INSERT INTO abbreviation_observed_expansions
                (abbreviation, clinical_context, expansion, omop_domain,
                 selection_basis, hit_count, last_updated)
            VALUES (?, ?, ?, ?, ?, 1, now())
            ON CONFLICT (abbreviation, clinical_context, expansion) DO UPDATE SET
                hit_count = hit_count + 1,
                last_updated = now()
        """, [abbreviation.lower(), clinical_context or "General", expansion,
              omop_domain, selection_basis])
    except Exception:
        pass


def compute_frequency_priority(conn, abbreviation: str, meanings: list):
    """Returns the meaning with a clearly-dominant observed hit-count share
    among `meanings`, or None when the abbreviation isn't on
    VERIFIED_ALLOW_LIST, is bias-excluded, there isn't enough data, or no
    single meaning clearly dominates (a real, close ambiguity -- same
    conservatism as src.preprocessing._select_by_groundability(), which
    also returns None rather than force a pick when the evidence doesn't
    clearly separate the candidates).

    ALLOW-LIST GATE (checked first, before any DB query): see the module
    docstring's "2026-08-17 POSTURE INVERSION" section. An abbreviation
    must be explicitly verified-safe to ever get a real answer here -- the
    ledger can accumulate arbitrarily much data for anything else and it
    still returns None, by design, until a human promotes it.

    "Clearly dominant" = total hit_count >= MIN_FREQUENCY_PRIORITY_SUPPORT
    AND the winner's share of the total exceeds the runner-up's share by at
    least FREQUENCY_PRIORITY_MARGIN. Both floors exist for the same reason
    MIN_CACHE_HIT_COUNT does in src.acronym_escalation: a small or narrow
    lead is not enough to trust as a production tiebreak.
    """
    if conn is None or not abbreviation or not meanings:
        return None
    if (abbreviation or "").strip().lower() not in VERIFIED_ALLOW_LIST:
        return None
    if _is_bias_excluded(abbreviation):
        return None
    try:
        conn.sql(ABBREVIATION_OBSERVATIONS_DDL)
        rows = conn.execute("""
            SELECT expansion, sum(hit_count) FROM abbreviation_observed_expansions
            WHERE abbreviation = ? GROUP BY expansion
        """, [abbreviation.lower()]).fetchall()
    except Exception:
        return None
    if not rows:
        return None

    counts = {expansion: count for expansion, count in rows if expansion in meanings}
    total = sum(counts.values())
    if total < MIN_FREQUENCY_PRIORITY_SUPPORT or not counts:
        return None

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    winner, winner_count = ranked[0]
    runner_up_count = ranked[1][1] if len(ranked) > 1 else 0
    if (winner_count - runner_up_count) / total < FREQUENCY_PRIORITY_MARGIN:
        return None
    return winner


# ==========================================================================
# Mechanism 2 -- context-pattern rules (mined from real HITL/SME review)
# ==========================================================================

ABBREVIATION_CONTEXT_RULES_DDL = """
CREATE TABLE IF NOT EXISTS abbreviation_context_rules (
    abbreviation VARCHAR,
    meaning VARCHAR,
    trigger_word VARCHAR,
    position VARCHAR,          -- 'pre' or 'post'
    support_count INTEGER,
    score DOUBLE,              -- log-odds of trigger_word given this meaning vs. the rest
    last_updated TIMESTAMP DEFAULT now(),
    PRIMARY KEY (abbreviation, meaning, trigger_word, position)
);
"""

_WORD_RE = re.compile(r"[A-Za-z]+")

# Deliberately excludes near-universal clinical filler words that would
# otherwise dominate every window regardless of meaning ("patient", "was",
# "the", ...) -- a trigger word is only useful if it's DISTINCTIVE to one
# meaning, and a word every note contains can never be that.
_STOP_TRIGGER_WORDS = {
    "the", "a", "an", "is", "was", "were", "with", "and", "or", "of", "to",
    "in", "on", "for", "patient", "pt", "noted", "seen", "at", "by",
}


def context_window(raw_text: str, orig_start: int, orig_end: int, window_chars: int = None):
    """Pre/post tokenized word windows sliced from RAW note text (NOT
    Stage-1-expanded text) around an entity's own character offsets --
    same reasoning as src.acronym_escalation.raw_local_context(): feeding a
    model (or here, a rule miner/matcher) Stage 1's own already-expanded
    text would show it a prior guess dressed up as literal wording, which
    is exactly the circularity bug that function was built to fix. Returns
    (pre_words, post_words), both lists of lowercased words, possibly empty.
    """
    if raw_text is None or orig_start is None or orig_end is None:
        return [], []
    w = window_chars or CONTEXT_WINDOW_CHARS
    pre_slice = raw_text[max(0, orig_start - w):orig_start]
    post_slice = raw_text[orig_end:min(len(raw_text), orig_end + w)]
    pre_words = [m.group(0).lower() for m in _WORD_RE.finditer(pre_slice)][-3:]
    post_words = [m.group(0).lower() for m in _WORD_RE.finditer(post_slice)][:3]
    return pre_words, post_words


def select_by_context_pattern(conn, meanings: list, abbreviation: str,
                              raw_text: str, orig_start: int, orig_end: int):
    """Deterministic pre-filter: if this occurrence's actual pre/post
    context words match a mined trigger rule strongly associated with
    exactly one candidate meaning, return that meaning -- checked BEFORE
    any model call, same "skip the judgment call entirely" shape as
    src.mollm_tier_gate.tier3_fast_path(). Returns None on no match, no
    rules, conn=None, or any DB error (falls through to the existing
    tiebreak chain unchanged).
    """
    if conn is None or not abbreviation or not meanings:
        return None
    pre_words, post_words = context_window(raw_text, orig_start, orig_end)
    if not pre_words and not post_words:
        return None
    try:
        conn.sql(ABBREVIATION_CONTEXT_RULES_DDL)
        rows = conn.execute("""
            SELECT meaning, trigger_word, position, score FROM abbreviation_context_rules
            WHERE abbreviation = ? AND meaning IN ({})
        """.format(",".join("?" * len(meanings))),
            [abbreviation.lower()] + list(meanings)).fetchall()
    except Exception:
        return None
    if not rows:
        return None

    window_words = {"pre": set(pre_words), "post": set(post_words)}
    scores = {}
    for meaning, trigger_word, position, score in rows:
        if trigger_word in window_words.get(position, set()):
            scores[meaning] = scores.get(meaning, 0.0) + score
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] <= ranked[1][1]:
        return None  # a genuine tie between two matched meanings -- don't force a pick
    return ranked[0][0]


def _log_odds(word_count_this: int, total_this: int,
              word_count_other: int, total_other: int, alpha: float = 1.0) -> float:
    """Add-alpha-smoothed log-odds ratio of `word` appearing given THIS
    meaning vs. every OTHER meaning of the same abbreviation, over the raw
    2x2 occurrence/non-occurrence counts (standard Laplace smoothing on
    COUNTS, not on a derived probability -- smoothing a probability directly
    can still land exactly on 0 or 1 when a count equals its total, which
    then hits log(0); smoothing the counts first never can, since alpha>0
    keeps every term strictly positive).
    """
    import math
    a = word_count_this + alpha
    b = (total_this - word_count_this) + alpha
    c = word_count_other + alpha
    d = (total_other - word_count_other) + alpha
    return math.log(a / b) - math.log(c / d)


def mine_context_rules(conn, raw_text_by_note: dict, min_support: int = None) -> int:
    """Reads real reviewer-confirmed resolutions from hitl_review_queue,
    tokenizes each one's pre/post context window, and writes trigger-word
    rules to abbreviation_context_rules. Returns the number of rules
    written. Safe to run repeatedly (idempotent upsert); safe to run with
    zero real review data (returns 0, writes nothing) -- this is expected
    and correct until dry_run=False and real SME review actually
    accumulates, not a bug in this function.

    `raw_text_by_note` ({note_id: raw_text}) is supplied by the caller
    rather than loaded here -- this module has no opinion on WHERE raw note
    text lives (a CSV, same as scripts/score_gold_recall.py's load_gold(),
    or a real notes table in production); it only needs the text once a
    note_id is known. A note_id missing from this dict is skipped, not an
    error -- partial coverage degrades to fewer rules, not a crash.

    METHOD, deliberately simple and auditable rather than a generic
    pattern-mining library: for each (abbreviation, trigger_word, position)
    pair, score = log-odds of that word appearing given THIS meaning vs.
    every OTHER meaning of the same abbreviation (_log_odds() above) -- a
    word that appears near one meaning and never the others is a strong
    signal; a word that appears near all of them equally is not. Requires
    min_support independent examples per abbreviation (summed across its
    meanings) before ANY rule for it is written, and per-word support >= 2
    within that, so a rule is never built from a single confirmed example.
    """
    if conn is None:
        return 0
    min_support = min_support or MIN_CONTEXT_RULE_SUPPORT
    try:
        conn.sql(ABBREVIATION_CONTEXT_RULES_DDL)
        rows = conn.execute("""
            SELECT e.original_text, h.corrected_concept_id, e.orig_start, e.orig_end,
                   e.note_id
            FROM hitl_review_queue h
            JOIN extracted_entities e ON e.entity_id = h.entity_id
            WHERE h.reviewer_decision IN ('APPROVED', 'CORRECTED')
              AND e.expansion_ambiguous = TRUE
        """).fetchall()
    except Exception:
        return 0
    if not rows:
        return 0

    # Group by (abbreviation, meaning=corrected_concept_id) -> list of
    # (pre_words, post_words). concept_id stands in for "meaning" here
    # since that's the durable identifier hitl_review_queue actually
    # records -- a human-readable meaning string can be joined in later by
    # the caller of select_by_context_pattern() if needed for display.
    examples = {}
    for original_text, concept_id, orig_start, orig_end, note_id in rows:
        raw_text = raw_text_by_note.get(note_id)
        if raw_text is None or not original_text or concept_id is None:
            continue
        pre_words, post_words = context_window(raw_text, orig_start, orig_end)
        key = (original_text.lower(), str(concept_id))
        examples.setdefault(key, []).append((pre_words, post_words))

    # abbreviation -> {meaning: [(pre_words, post_words), ...]}
    by_abbrev = {}
    for (abbrev, meaning), occs in examples.items():
        by_abbrev.setdefault(abbrev, {})[meaning] = occs

    n_written = 0
    for abbrev, meanings in by_abbrev.items():
        # Deliberately NO _is_bias_excluded() check here, unlike
        # compute_frequency_priority() above -- see module docstring's
        # "BOTH MECHANISMS EXCLUDE" paragraph for why mechanism 1 excludes
        # the known-bias list (the pipeline's own confident-but-wrong guess
        # would just formalize itself) but mechanism 2 does not: real
        # reviewer-confirmed hitl_review_queue rows aren't the pipeline's
        # own guess, they're independent ground truth, which is exactly the
        # kind of evidence that CAN correct a systematic bias rather than
        # reinforce it. This mechanism is the intended eventual path to
        # relaxing the coronary/short-code hard traps for real, once real
        # data exists -- excluding them here would defeat that purpose.
        total_examples = sum(len(occs) for occs in meanings.values())
        if total_examples < min_support or len(meanings) < 2:
            continue  # not enough data, or only one observed meaning -- nothing to disambiguate

        for meaning, occs in meanings.items():
            total_this = len(occs)
            total_other = total_examples - total_this
            word_counts_this = {"pre": {}, "post": {}}
            for pre_words, post_words in occs:
                for w in set(pre_words) - _STOP_TRIGGER_WORDS:
                    word_counts_this["pre"][w] = word_counts_this["pre"].get(w, 0) + 1
                for w in set(post_words) - _STOP_TRIGGER_WORDS:
                    word_counts_this["post"][w] = word_counts_this["post"].get(w, 0) + 1

            word_counts_other = {"pre": {}, "post": {}}
            for other_meaning, other_occs in meanings.items():
                if other_meaning == meaning:
                    continue
                for pre_words, post_words in other_occs:
                    for w in set(pre_words) - _STOP_TRIGGER_WORDS:
                        word_counts_other["pre"][w] = word_counts_other["pre"].get(w, 0) + 1
                    for w in set(post_words) - _STOP_TRIGGER_WORDS:
                        word_counts_other["post"][w] = word_counts_other["post"].get(w, 0) + 1

            for position in ("pre", "post"):
                for word, count_this in word_counts_this[position].items():
                    if count_this < 2:
                        continue
                    count_other = word_counts_other[position].get(word, 0)
                    score = _log_odds(count_this, total_this, count_other, total_other)
                    if score <= 0:
                        continue  # not actually distinctive of this meaning
                    try:
                        conn.execute("""
                            INSERT INTO abbreviation_context_rules
                                (abbreviation, meaning, trigger_word, position,
                                 support_count, score, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, now())
                            ON CONFLICT (abbreviation, meaning, trigger_word, position)
                            DO UPDATE SET support_count = excluded.support_count,
                                         score = excluded.score, last_updated = now()
                        """, [abbrev, meaning, word, position, count_this, score])
                        n_written += 1
                    except Exception:
                        continue
    return n_written
