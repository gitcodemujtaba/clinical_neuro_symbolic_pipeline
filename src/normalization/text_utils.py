"""src/normalization/text_utils.py — tokenization/SQL-clause helpers shared across tiers (split from src/normalization.py, 2026-08-14)."""
import re

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



