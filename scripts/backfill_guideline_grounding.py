"""
scripts/backfill_guideline_grounding.py

Attempts to resolve missing SNOMED codes (and derive ICD10CM via OMOP's
"Maps to" relationship) for guideline triplet nodes in local_triplets_db2_v6/
where snomed == "N/A". See docs/Guideline_Triplets_KG_Review.md (S3.1) for
the finding this addresses: 974 of 1,969 curated triplet nodes (49.5%) are
currently ungrounded.

WHY THIS IS A SEPARATE SCRIPT FROM src/normalization.py, NOT A CALL INTO IT:
This grounds the curated GUIDELINE knowledge graph -- the deterministic
evidence base Stage 3 cites as ground truth for every note the pipeline ever
processes. An error here doesn't affect one patient note, it propagates to
every future extraction that gets grounded against that guideline concept.
So this applies a stricter acceptance policy than Stage 2b's per-note
normalization:
  - Tier 1 (exact concept-name) and Tier 2 (exact synonym) matches are
    auto-accepted.
  - Tier 3 (SapBERT semantic similarity) matches auto-accept once score >=
    AUTO_ACCEPT_FLOOR (0.70) -- see that constant's comment for the
    2026-08-07 policy change (project owner directive) that superseded the
    original "never auto-accept, always human review" design described
    below this bullet historically. In short: since TIER3_SIMILARITY_FLOOR
    (0.72) already exceeds AUTO_ACCEPT_FLOOR, every Tier-3 candidate that
    clears the candidate-generation floor now also clears the accept floor.
    Below AUTO_ACCEPT_FLOOR, matches still land in a review report for a
    human to sign off on -- this remains a good first candidate for the
    project's HITL mechanism, ahead of Stage 4 existing.
  - Node types that aren't codeable clinical entities (numeric/temporal
    criteria like "Quantitative Threshold" or "Timeframe") are skipped
    entirely rather than forced through concept matching.
  - Medication-typed nodes are matched against RxNorm/RxNorm Extension
    (where OMOP actually codes drugs), not SNOMED.

This script never mutates the original triplet files. In dry-run mode
(default) it only writes a JSON report. With --apply, it writes a
*separate* local_triplets_db2_v6_grounded/ copy containing only the
auto-accepted (Tier 1/2) matches, tagged with a grounding_provenance block
so it's clear which codes were originally curated vs. backfilled.

GLINER PRE-EXTRACTION (2026-08-07 addition):
Some ungrounded node names are compound phrases -- "Serum creatinine
increase by >=0.3 mg/dl within 48 hours", "Systolic blood pressure
threshold for discontinuing nitroglycerin" -- that are very unlikely to hit
an exact Tier 1/2 match as a whole string, so they'd previously go straight
to the noisier Tier 3 semantic fallback (the same tier that produced
`bioplar`->`Bourgvilain`-style garbage on patient notes). Before falling
back to Tier 3 on the full name, this script now runs GLiNER (the same
zero-shot model as Stage 2a) over the name to pull out the core clinical
span -- e.g. "Serum creatinine" -- and tries Tier 1/2 exact/synonym
matching on that span first. GLiNER itself never produces a code; it's
used purely to simplify a compound phrase down to something more likely to
land an exact/synonym hit. Candidate spans are also constrained to a
domain_id allowlist keyed on the node's @type (see DOMAIN_BY_TYPE) so a
short span can't exact-match an unrelated-domain concept (e.g. a
bare word "lung" hitting the anatomy concept "Lung structure" for a node
that is actually an Intervention).

Even with the domain guard, span-based hits are NEVER auto-accepted (second
2026-08-07 addition, after a full corpus run) -- they always land in
needs_review, same as Tier 3. A full run showed ~1/3 of span-based Tier 1/2
hits were still wrong in one of two domain-compatible-but-semantically-wrong
ways: acronym collision (node "PCT (Procalcitonin) level" -> span "PCT" ->
concept "Percent"; "MACE" -> concept "Mace") and enumeration collapse (a
node naming several drugs/tests grounds to whichever single one GLiNER
extracted, e.g. a 7-drug-class list -> just "ivabradine"). Only full-name
Tier 1/2 matches still auto-accept, since compound guideline phrases are
specific enough that both failure modes are far less plausible there.
Every result records `resolved_via` ("full_name" or "gliner_span") and,
for span-based matches, `matched_text` (the specific span that matched,
distinct from the node's original `name`) so a reviewer can sanity-check
that the extracted span still represents the right concept for that node.
A full-corpus validation run confirmed the fix: all 6 span-based
auto_accepts scored 0.92-0.95 on the consistency check and were genuinely
clean matches, while every previously-flagged acronym-collision and
enumeration-collapse case (PCT->Percent, MACE->Mace, the 7-drug-class list,
the "amphotericin B" meaning-inverting case) now correctly lands in
needs_review.

MEDCAT ENTITY REUSE (2026-08-07, third addition, same day):
This fixes a *precision* problem (bad auto-accepts). It does nothing for
the much bigger *recall* problem: on the full 916-node corpus, only 84
nodes (9%) auto-accepted and 448 (49%) got no_match at all -- nothing in
the entire embedding space cleared TIER3_SIMILARITY_FLOOR. See
docs/Guideline_Triplets_KG_Review.md S3.1/S3.7: the upstream MedCAT
entity-linking pass (data/triplets-rules-backup-data/local_medcat_entities_db2_v6/)
already found and linked a SNOMED CUI for many of these spans while
processing the source chunk -- it just didn't survive the "qualification"
threshold for injection into local_grounded_chunks_db2_v6 (qualification
rate varies 21%-78% across chunks per S3.7), so the triplet-extraction LLM
never saw a `[SNOMED: ...]` tag for that phrase and emitted "N/A". That
CUI is still sitting in the *_medcat.json entity list, unused.
Concrete example that motivated this: node "Use point-of-care lung
ultrasound" (Intervention) previously only reached Tier 3 on the full
name, landing on "Point of care ultrasound" at 0.7417 (needs_review). The
matching *_medcat.json file for that chunk contains an entity with
source_value "point-of-care ultrasound", snomed_cui "870384002"
("Point of care diagnostic ultrasonography"), confidence_score 1.0 --
MedCAT already got this right upstream; the triplet extractor just never
carried it forward.
Mechanics: each node's `provenance.source_document` field names its
source chunk; the corresponding <chunk>_medcat.json file's `entities[]`
list is loaded (and cached per source_document, since many nodes share a
chunk). A candidate is any entity whose `source_value`, tokenized and
stopword-stripped, is a *complete subset* of the node name's own tokens
(handles cases like the example above, where "point-of-care ultrasound"
isn't a contiguous substring of "Use point-of-care lung ultrasound"
because "lung" sits in between). Candidates are tried most-specific
(most tokens) and highest-MedCAT-confidence first. This tier is tried
after full-name and GLiNER-span Tier 1/2 both fail, before falling back to
a blind Tier-3 embedding search on the full name -- a validated,
pre-linked CUI is a stronger signal than a nearest-neighbor search over
the whole vocabulary, when it exists.
This is NOT trusted any more than a GLiNER span is: S3.7 documented real
homonym collisions and boilerplate mislinks in this exact MedCAT data even
at confidence_score 1.0 (a journal running-header "Annals of Emergency
Medicine" tagged as a clinical concept; "95% confidence interval"'s
statistical sense of "confidence" tagged as if clinical). So a resolved
MedCAT candidate goes through the exact same two gates as a GLiNER span:
the DOMAIN_BY_TYPE allowlist, then the SPAN_CONSISTENCY_FLOOR SapBERT
check against the full node name before it's allowed to auto_accept;
otherwise it lands in needs_review same as everything else, tagged
`resolved_via: "medcat_entity"` with `medcat_cui`, `medcat_confidence`,
and `matched_text` recorded for audit. Same caveat as SPAN_CONSISTENCY_FLOOR
itself: this has not been run against a live DuckDB/SapBERT in the
environment this was written in -- validate with --limit against the
point-of-care-ultrasound node specifically before trusting a full run.
UPDATE after the first full-corpus run with this tier live (same day):
recall moved as intended -- no_match dropped from 448 to 261 nodes (49% to
28% of the corpus) and auto_accept rose from 35 to 65 -- but the run
surfaced two precision gaps neither existing gate closes, both now fixed
(see MEDCAT_CUI_BLOCKLIST and MEDCAT_CONFIDENCE_FLOOR just below): (1) CUI
773568002, the exact "Annals of Emergency Medicine" journal-header mislink
S3.7 already documented, auto-accepted node "Emergency medicine physicians"
at consistency 0.8639 -- a right-looking answer from a compromised upstream
source, which a node-name-vs-concept consistency check has no way to catch
since it only ever compares against the node name, never against where the
CUI came from; (2) two "Usually diagnostic" nodes auto-accepted onto the
generic concept "Diagnosis" via matched_text "diagnostic" at
medcat_confidence only 0.3165 -- consistency was high because "diagnostic"
and "Diagnosis" are lexically near-identical, independent of whether
MedCAT's own linker was confident in that specific CUI.

Usage:
    python3 scripts/backfill_guideline_grounding.py                # dry run, report only
    python3 scripts/backfill_guideline_grounding.py --apply         # also write grounded copies
    python3 scripts/backfill_guideline_grounding.py --no-gliner     # skip the span-extraction step
    python3 scripts/backfill_guideline_grounding.py --no-medcat     # skip the MedCAT-entity-reuse step

Requires the populated kg2_lexical_store.duckdb (Athena OMOP + SapBERT
embeddings) to already exist -- run scripts/import_athena.py and
scripts/build_concept_embeddings.py first if it doesn't.
"""

import os
import re
import sys
import json
import glob
import argparse
from datetime import datetime, timezone

import duckdb

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    _SAPBERT_AVAILABLE = True
except ImportError:
    _SAPBERT_AVAILABLE = False

try:
    from gliner import GLiNER
    _GLINER_AVAILABLE = True
except ImportError:
    _GLINER_AVAILABLE = False

# Same label set as src/entity_extraction.py, kept in sync deliberately --
# these are guideline-authored phrases, not noisy note text, so a lower
# threshold than Stage 2a's 0.5 is reasonable (0.3): precision matters less
# here since GLiNER's output is only ever tried against Tier 1/2 exact
# matching, never used to auto-accept a code by itself.
GLINER_LABELS = ["Condition", "Symptom", "Medication", "Procedure", "Anatomy", "Lab Test"]
GLINER_MODEL_NAME = "urchade/gliner_medium-v2.1"  # same checkpoint as Stage 2a; swap here if
                                                   # entity_extraction.py moves to a clinical checkpoint
GLINER_THRESHOLD = 0.3

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/home/ec2-user/clinical_neuro_symbolic_pipeline")
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")
# Source data was moved under data/ on 2026-08-07 (was previously at the repo root).
# Points at the *cleaned* corpus (scripts/clean_local_triplets.py output) so grounding
# backfill builds on top of the deduped/canonicalized/citation-classified data rather
# than the raw corpus -- run clean_local_triplets.py first if data/local_triplets_db2_v6_cleaned
# doesn't exist yet.
TRIPLETS_DIR = os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned")
REPORT_DIR = os.path.join(PROJECT_DIR, "grounding_backfill_report")
# Upstream MedCAT entity-linking output, one file per source chunk (see
# "MEDCAT ENTITY REUSE" in the module docstring). A node's
# provenance.source_document field (e.g. "foo.json") maps to
# "foo_medcat.json" in this directory.
MEDCAT_ENTITIES_DIR = os.path.join(PROJECT_DIR, "data", "triplets-rules-backup-data",
                                    "local_medcat_entities_db2_v6")

# Not codeable clinical entities -- never attempt to ground these against a
# concept vocabulary (see Guideline_Triplets_KG_Review.md S3.1 for the full
# @type breakdown of the 974 ungrounded nodes).
SKIP_TYPES = {"Quantitative Threshold", "quantitative_threshold", "Timeframe"}

# OMOP codes medications via RxNorm, not SNOMED -- SNOMED's own Substance
# hierarchy is out of scope for clinical entity linking generally (see
# Proposal_Alignment_Review.md S3.8 on the DrivenData benchmark's exclusion
# of substances). Everything else targets SNOMED specifically (not the full
# multi-vocabulary OMOP space) since that's the code this KG is built around.
VOCAB_BY_TYPE = {
    "Medication": ["RxNorm", "RxNorm Extension"],
}
DEFAULT_VOCAB = ["SNOMED"]

# 2026-08-07 addition, after a review run auto-accepted "Use point-of-care
# lung ultrasound" (type Intervention) onto concept "Lung structure" (domain
# Spec Anatomic Site) -- GLiNER's span extraction had shrunk the node name
# down to "lung", which then hit an exact synonym match on the *anatomy*
# concept rather than anything procedure-like. Tier 1/2 exact/synonym
# matching only constrains vocabulary_id, not domain_id, so a short, generic
# extracted span can land on a technically-exact but semantically unrelated
# domain and still get auto-accepted. This is only a meaningful risk for
# GLiNER-extracted spans (single/few words, chosen for brevity over
# specificity); full guideline node names are compound and specific enough
# that an unrelated-domain exact match is far less plausible, so this
# allowlist is applied ONLY to the span-matching path in ground_node(), not
# to full-name Tier 1/2 matching. Node types not listed here get no domain
# filter (matches pre-fix behavior) -- extend this as new @type values show
# up in the corpus (see the "@type breakdown" note on SKIP_TYPES above).
DOMAIN_BY_TYPE = {
    "Condition": {"Condition"},
    "Finding": {"Condition", "Observation", "Measurement"},
    "Intervention": {"Procedure", "Device", "Measurement"},
    "Acuity": {"Observation", "Measurement"},
    "Medication": {"Drug"},
}

# Floor for even considering a Tier-3 candidate worth reporting. Matches the
# 0.72 threshold documented in Implementation_Methodology.md. Note this is a
# floor for "worth a human's time to look at", not an auto-accept bar -- see
# module docstring.
TIER3_SIMILARITY_FLOOR = 0.72

# Floor for recovering a GLiNER-span Tier 1/2 hit -- or, as of the MedCAT
# entity-reuse addition, a MedCAT-linked-CUI hit -- back to auto_accept (see
# "SPAN CONSISTENCY CHECK" / "MEDCAT ENTITY REUSE" in the module docstring).
# Both paths share this floor because they pose the same question: is this
# already-found exact match/pre-linked CUI actually representative of the
# whole node name, or just lexically present somewhere inside it. This is
# deliberately a
# DIFFERENT, stricter threshold than TIER3_SIMILARITY_FLOOR: Tier 3's floor
# answers "is this candidate worth a human's time", this one answers "is
# this exact/synonym match trustworthy enough to skip human review
# entirely", which is a higher bar. 0.85 is a reasoned starting point, not
# an empirically validated one -- it was chosen without the ability to run
# SapBERT locally (no model/network access in the environment this fix was
# written in). Before trusting a full --apply run, validate it against the
# known cases from the 2026-08-07 full-corpus run: it should score HIGH for
# "Transthoracic Echocardiography (TTE) evaluation" vs. matched concept
# "Transthoracic echocardiography" (should stay auto_accept), and LOW for
# "Procalcitonin (PCT) level" vs. "Percent", and for "ACEi, ARB, ARNi, MRA,
# SGLT2i, ivabradine, vericiguat" vs. "ivabradine" (both should be pushed to
# needs_review). Adjust this constant if real scores don't separate cleanly.
#
# 2026-08-07 addition (sixth, project-owner policy change): lowered from
# 0.85 to AUTO_ACCEPT_FLOOR (0.70), and Tier 3 (previously never
# auto-accepted regardless of score -- see ground_node()) now uses the same
# floor. Rationale given: these triplets ground Stage 3 (MoLLM) as their own
# subject/predicate/object text; the attached SNOMED/ICD10 code is
# enrichment on top of that, not a requirement for the triplet to be usable
# as grounding context. A marginally-wrong code in the 0.70-0.85 band is
# therefore judged lower-stakes than under the original design, where an
# incorrect code was treated as corrupting the deterministic evidence base
# outright. This trades precision for recall versus the original "no
# LLM/embedding guess enters the evidence base without human sign-off"
# policy. One known concrete cost, flagged for visibility rather than
# blocking the change: "Emergency medicine physicians" -> "Emergency
# medicine" (Tier-3 full-name match, score 0.8639) will now auto-accept even
# though this is the same journal-running-header mislink documented in
# MEDCAT_CUI_BLOCKLIST's comment below -- that blocklist only filters the
# MedCAT-entity candidate path, not this independent Tier-3 embedding match
# that reaches the same wrong concept via the full node name directly.
AUTO_ACCEPT_FLOOR = 0.70
SPAN_CONSISTENCY_FLOOR = AUTO_ACCEPT_FLOOR

# 2026-08-07 addition (fourth, after the first full-corpus MedCAT-tier run):
# CUIs documented in Guideline_Triplets_KG_Review.md S3.7 as boilerplate/
# homonym mislinks in the upstream MedCAT entity data itself -- i.e. the
# mislink is in the *source* the MedCAT tier reads from, not something the
# SPAN_CONSISTENCY_FLOOR gate can catch. Concretely: 773568002 ("Emergency
# medicine") is a journal running-header mislink ("Annals of Emergency
# Medicine"), yet it auto-accepted node "Emergency medicine physicians" in
# the first full run with span_consistency_score 0.8639 -- comfortably over
# SPAN_CONSISTENCY_FLOOR, because "Emergency medicine physicians" and
# "Emergency medicine" are genuinely close in embedding space even though
# the CUI that produced the match was extracted from an unrelated citation
# artifact, not real clinical content about this node. A right-looking
# answer reached via a known-bad source is still not trustworthy -- the
# consistency check can only validate the concept against the node name, it
# has no way to know the CUI's provenance was compromised upstream. 225487001
# ("confidence interval") and 246228008 ("grading") are the other two S3.7
# examples (statistical/generic-word homonyms of clinical-sounding SNOMED
# terms) -- included preemptively even though a full run hasn't yet surfaced
# them going to auto_accept, since the failure mode (domain-compatible,
# embeddable-similar, upstream-mislinked) is identical. Extend this set if
# further runs surface other recurring bad CUIs.
MEDCAT_CUI_BLOCKLIST = {"773568002", "225487001", "246228008"}

# 2026-08-07 addition (fifth), same run: two nodes ("Usually diagnostic" x2,
# type Acuity) auto-accepted via MedCAT entity CUI 439401001 ("Diagnosis")
# on matched_text "diagnostic" with medcat_confidence only 0.3165 -- i.e.
# MedCAT's own entity linker was itself unsure this span even referred to
# that concept, but span_consistency_score (0.8552) was high anyway because
# "diagnostic" and "Diagnosis" are trivially close in embedding space as a
# generic word pair, independent of whether the specific CUI is the right
# one. SPAN_CONSISTENCY_FLOOR validates "does this concept represent the
# node name", not "was this CUI a confident link" -- those are different
# questions, and only the MedCAT confidence score answers the second one.
# This floor closes that gap: auto_accept via the MedCAT-entity path now
# requires medcat_confidence >= this value in addition to the consistency
# check. 0.4 is a reasoned starting point (it also correctly keeps a good
# match -- "PSI (Pneumonia Severity Index)" -> "Pneumonia severity index" at
# medcat_confidence 0.4473 -- while excluding the 0.3165 case above), not an
# empirically validated one; a candidate that fails this floor still lands
# in needs_review rather than being discarded, same as any other gate in
# this script.
MEDCAT_CONFIDENCE_FLOOR = 0.4


def load_sapbert():
    print("Loading SapBERT for Tier 3 candidate search...")
    tok = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    return tok, model


def load_gliner():
    print("Loading GLiNER for compound-name span extraction...")
    return GLiNER.from_pretrained(GLINER_MODEL_NAME)


def extract_candidate_spans(gliner_model, name):
    """Runs GLiNER over a (possibly compound) node name and returns candidate
    sub-spans sorted by GLiNER's own confidence, deduplicated by text. Used
    only to generate additional strings to try against Tier 1/2 exact
    matching -- GLiNER's label choice is not trusted or used here, just the
    span boundaries."""
    if gliner_model is None:
        return []
    try:
        entities = gliner_model.predict_entities(name, GLINER_LABELS, threshold=GLINER_THRESHOLD)
    except Exception:
        return []
    entities.sort(key=lambda e: e.get("score", 0), reverse=True)
    seen = set()
    spans = []
    for e in entities:
        text = e.get("text", "").strip()
        key = text.lower()
        if not text or key == name.lower().strip() or key in seen:
            continue  # skip empty, identical-to-full-name, or duplicate spans
        seen.add(key)
        spans.append(text)
    return spans


def get_embedding(tok, model, text):
    tokens = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**tokens)
        return out.last_hidden_state[:, 0, :].squeeze().tolist()


def _cosine_similarity(a, b):
    """Plain-Python cosine similarity between two equal-length embedding
    vectors (already-materialized lists, as returned by get_embedding).
    Used by the span consistency check -- not performance-critical (one
    node-name embedding reused across at most a handful of candidate
    spans), so this deliberately avoids adding a numpy dependency just for
    a single dot product."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately small and generic -- just enough to stop trivial connector
# words from counting as "content" when comparing a MedCAT entity's
# source_value against a node name (see find_medcat_candidates()). Not a
# clinical-domain stopword list; MedCAT source_values are short enough
# (usually 1-4 words) that a bigger list isn't needed.
MEDCAT_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "with", "by", "is", "are", "was", "were", "be", "this", "that",
    "these", "those", "as", "from", "than", "then",
}


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def _medcat_entities_path(source_document, medcat_dir=MEDCAT_ENTITIES_DIR):
    base = os.path.splitext(os.path.basename(source_document))[0]
    return os.path.join(medcat_dir, f"{base}_medcat.json")


def get_medcat_entities(medcat_cache, source_document, medcat_dir=MEDCAT_ENTITIES_DIR):
    """Loads and caches (per source_document, not per node -- many nodes
    share a chunk) the upstream MedCAT entity list for a node's originating
    chunk. Returns [] if the file doesn't exist or source_document is
    missing/unrecognized, so callers can treat "no MedCAT data for this
    node" the same as "no candidates found" rather than needing a separate
    error path."""
    if not source_document:
        return []
    if source_document in medcat_cache:
        return medcat_cache[source_document]
    path = _medcat_entities_path(source_document, medcat_dir)
    entities = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                entities = json.load(f).get("entities", [])
        except Exception:
            entities = []
    medcat_cache[source_document] = entities
    return entities


def find_medcat_candidates(name, entities):
    """Searches a chunk's upstream MedCAT-linked entities for ones whose
    source_value is wholly represented in the node's name, as a candidate
    SNOMED CUI sourced from entity-linking rather than Tier 1/2/3 lookups
    against the OMOP vocabulary directly. See "MEDCAT ENTITY REUSE" in the
    module docstring.

    A candidate's tokens (lowercased, stopword-stripped) must be a complete
    SUBSET of the node name's own tokens -- not a contiguous substring,
    since a MedCAT span like "point-of-care ultrasound" can be split apart
    in the node's own (differently-worded) name, e.g. "Use point-of-care
    LUNG ultrasound". This is deliberately permissive on containment and
    relies on ranking + the caller's domain filter + SapBERT consistency
    check for precision, exactly like a GLiNER span.

    Returns candidates sorted (most tokens, i.e. most specific, first; then
    highest MedCAT confidence_score), deduplicated by CUI so the caller
    tries the strongest candidate for a given concept exactly once."""
    node_tokens = set(_tokenize(name)) - MEDCAT_STOPWORDS
    matches = []
    for e in entities:
        source_value = e.get("source_value") or ""
        cui = e.get("snomed_cui")
        if not source_value or not cui:
            continue
        if cui in MEDCAT_CUI_BLOCKLIST:
            continue
        ent_tokens = set(_tokenize(source_value)) - MEDCAT_STOPWORDS
        if not ent_tokens or not ent_tokens.issubset(node_tokens):
            continue
        matches.append((len(ent_tokens), e.get("confidence_score") or 0.0, cui,
                         source_value, e.get("preferred_name")))
    matches.sort(key=lambda m: (-m[0], -m[1]))
    seen_cuis = set()
    candidates = []
    for m in matches:
        if m[2] in seen_cuis:
            continue
        seen_cuis.add(m[2])
        candidates.append(m)
    return candidates


def _in_clause(vocabs):
    return ",".join(["?"] * len(vocabs))


def tier1_exact(conn, text, vocabs, domains=None):
    domain_clause = f" AND domain_id IN ({_in_clause(domains)})" if domains else ""
    q = f"""
        SELECT concept_id, concept_name, domain_id, vocabulary_id
        FROM athena_concept
        WHERE lower(concept_name) = ?
          AND standard_concept = 'S'
          AND vocabulary_id IN ({_in_clause(vocabs)})
          {domain_clause}
        ORDER BY concept_id ASC
        LIMIT 1;
    """
    params = [text.lower().strip(), *vocabs, *(domains or [])]
    return conn.sql(q, params=params).fetchone()


def tier2_synonym(conn, text, vocabs, domains=None):
    domain_clause = f" AND c.domain_id IN ({_in_clause(domains)})" if domains else ""
    q = f"""
        SELECT c.concept_id, c.concept_name, c.domain_id, c.vocabulary_id
        FROM athena_concept_synonym s
        JOIN athena_concept c ON s.concept_id = c.concept_id
        WHERE lower(s.concept_synonym_name) = ?
          AND c.standard_concept = 'S'
          AND c.vocabulary_id IN ({_in_clause(vocabs)})
          {domain_clause}
        ORDER BY c.concept_id ASC
        LIMIT 1;
    """
    params = [text.lower().strip(), *vocabs, *(domains or [])]
    return conn.sql(q, params=params).fetchone()


def tier3_semantic(conn, tok, model, text, vocabs):
    vec = get_embedding(tok, model, text)
    q = f"""
        SELECT concept_id, concept_name, domain_id, vocabulary_id,
               list_cosine_similarity(embedding, ?::FLOAT[]) AS similarity
        FROM athena_concept
        WHERE embedding IS NOT NULL
          AND standard_concept = 'S'
          AND vocabulary_id IN ({_in_clause(vocabs)})
        ORDER BY similarity DESC, concept_id ASC
        LIMIT 1;
    """
    return conn.sql(q, params=[vec, *vocabs]).fetchone()


def icd10_crosswalk(conn, snomed_concept_id):
    """Follows OMOP 'Maps to' from a resolved concept_id to ICD10CM, if one exists."""
    q = """
        SELECT c2.concept_code, c2.concept_name
        FROM athena_concept_relationship r
        JOIN athena_concept c2 ON r.concept_id_2 = c2.concept_id
        WHERE r.concept_id_1 = ?
          AND r.relationship_id = 'Maps to'
          AND c2.vocabulary_id = 'ICD10CM'
          AND r.invalid_reason IS NULL
        LIMIT 1;
    """
    return conn.sql(q, params=[snomed_concept_id]).fetchone()


def resolve_snomed_cui(conn, cui, domains=None):
    """Resolves a raw SNOMED CUI as reported by MedCAT (e.g. "870384002",
    matched against athena_concept.concept_code) to a standard OMOP
    concept, for the MedCAT-entity-reuse candidate path. Tries the CUI as
    a standard concept directly first; MedCAT/UMLS CUIs aren't guaranteed
    to already be OMOP-standard, so if that fails, follows 'Maps to' to a
    standard concept -- the same crosswalk relationship icd10_crosswalk()
    uses, just walked from a SNOMED code to a standard concept instead of
    from a standard concept to ICD10CM."""
    domain_clause = f" AND domain_id IN ({_in_clause(domains)})" if domains else ""
    q1 = f"""
        SELECT concept_id, concept_name, domain_id, vocabulary_id
        FROM athena_concept
        WHERE concept_code = ?
          AND vocabulary_id = 'SNOMED'
          AND standard_concept = 'S'
          {domain_clause}
        LIMIT 1;
    """
    hit = conn.sql(q1, params=[cui, *(domains or [])]).fetchone()
    if hit:
        return hit

    domain_clause2 = f" AND c2.domain_id IN ({_in_clause(domains)})" if domains else ""
    q2 = f"""
        SELECT c2.concept_id, c2.concept_name, c2.domain_id, c2.vocabulary_id
        FROM athena_concept c1
        JOIN athena_concept_relationship r ON r.concept_id_1 = c1.concept_id
        JOIN athena_concept c2 ON r.concept_id_2 = c2.concept_id
        WHERE c1.concept_code = ?
          AND c1.vocabulary_id = 'SNOMED'
          AND r.relationship_id = 'Maps to'
          AND r.invalid_reason IS NULL
          AND c2.standard_concept = 'S'
          {domain_clause2}
        LIMIT 1;
    """
    return conn.sql(q2, params=[cui, *(domains or [])]).fetchone()


def _exact_hit_result(hit, tier_label, resolved_via, matched_text, status="auto_accept"):
    result = {"status": status, "tier": tier_label, "concept_id": hit[0],
              "concept_name": hit[1], "domain_id": hit[2], "vocab": hit[3], "score": 1.0,
              "resolved_via": resolved_via}
    if resolved_via == "gliner_span":
        result["matched_text"] = matched_text
    return result


def ground_node(conn, tok, model, gliner_model, name, node_type, cache,
                 source_document=None, medcat_cache=None, medcat_dir=MEDCAT_ENTITIES_DIR):
    vocabs = VOCAB_BY_TYPE.get(node_type, DEFAULT_VOCAB)
    # source_document is part of the cache key (unlike everything else this
    # function tries) because the MedCAT-entity-reuse path below is
    # genuinely chunk-dependent -- the same node name string appearing in
    # two different source chunks can have different upstream MedCAT
    # entities available for it, so the two instances aren't guaranteed to
    # resolve the same way. This does cost some cache-hit rate on the small
    # number of duplicate names that legitimately recur across different
    # files, but correctness matters more here than that minor slowdown.
    cache_key = (name.lower().strip(), tuple(vocabs), source_document)
    if cache_key in cache:
        return cache[cache_key]

    # --- Try the full node name first (unchanged behavior) ---
    hit = tier1_exact(conn, name, vocabs)
    if hit:
        result = _exact_hit_result(hit, "1 (Exact)", "full_name", name)
        cache[cache_key] = result
        return result

    hit = tier2_synonym(conn, name, vocabs)
    if hit:
        result = _exact_hit_result(hit, "2 (Synonym)", "full_name", name)
        cache[cache_key] = result
        return result

    # --- Full name didn't get an exact hit -- try GLiNER-extracted spans ---
    # before falling back to Tier 3 on the full (possibly compound) name.
    # Only Tier 1/2 (exact) is tried on spans -- deliberately not Tier 3,
    # since stacking a fuzzy span-extraction on top of a fuzzy similarity
    # search compounds uncertainty rather than reducing it.
    # Span hits are additionally constrained to the domain(s) expected for
    # this node's @type (see DOMAIN_BY_TYPE) -- a short extracted span like
    # "lung" can be an exact synonym match against an unrelated-domain
    # concept (e.g. anatomy) even when the node itself is an Intervention,
    # not a body site. Full-name matching above is not domain-filtered since
    # compound node names are specific enough that this false-positive mode
    # is far less plausible there.
    #
    # 2026-08-07 (second addition, same day) -- SPAN CONSISTENCY CHECK:
    # a full corpus run showed the domain filter above isn't sufficient on
    # its own. ~1/3 of GLiNER-span Tier-1/2 hits were still wrong in one of
    # two domain-compatible-but-semantically-wrong ways:
    #   (a) acronym collision -- a short extracted span coincidentally
    #       exact/synonym-matches an unrelated concept, e.g. node "PCT
    #       (Procalcitonin) level" -> span "PCT" -> concept "Percent"
    #       (Observation domain, same domain Finding nodes commonly land in);
    #       "MACE" (Major Adverse Cardiac Events) -> concept "Mace".
    #   (b) enumeration collapse -- a node naming several distinct
    #       drugs/tests gets grounded to whichever single one GLiNER
    #       happened to extract, silently dropping the rest, e.g. "ACEi,
    #       ARB, ARNi, MRA, SGLT2i, ivabradine, vericiguat" (7 drug classes)
    #       -> "ivabradine" alone; "Azoles or echinocandins over
    #       conventional amphotericin B" -> "amphotericin B", the drug the
    #       recommendation says NOT to prefer.
    # Rather than blanket-downgrading every span hit to needs_review (which
    # would also throw away good matches like "Transthoracic
    # Echocardiography (TTE) evaluation" -> "Transthoracic
    # echocardiography"), a span hit is auto-accepted only if the SapBERT
    # embedding of the FULL node name is still close to the embedding of
    # the matched concept's name (>= SPAN_CONSISTENCY_FLOOR cosine
    # similarity) -- i.e. the exact/synonym match found via the span isn't
    # just lexically present somewhere in the name, it's actually
    # representative of what the whole name means. This is a consistency
    # check on an already-found exact match, not a new fuzzy search, so it
    # doesn't reintroduce the "stacking fuzzy-on-fuzzy" risk that's the
    # reason Tier 3 itself is never tried on spans. If SapBERT isn't
    # available, or the check fails, the hit still falls back to
    # needs_review rather than being discarded -- it's still a real
    # exact/synonym match, just not confidently representative enough to
    # skip human sign-off. See SPAN_CONSISTENCY_FLOOR's own comment: this
    # threshold was set without the ability to run SapBERT in the
    # environment this fix was written in, and should be validated against
    # known good/bad cases before trusting a full --apply run.
    allowed_domains = DOMAIN_BY_TYPE.get(node_type)
    name_embedding = None  # computed lazily, at most once per node
    for span in extract_candidate_spans(gliner_model, name):
        hit = tier1_exact(conn, span, vocabs, domains=allowed_domains)
        tier_label = "1 (Exact, via GLiNER span)"
        if not hit:
            hit = tier2_synonym(conn, span, vocabs, domains=allowed_domains)
            tier_label = "2 (Synonym, via GLiNER span)"
        if not hit:
            continue

        status = "needs_review"
        consistency_score = None
        if _SAPBERT_AVAILABLE and tok is not None and model is not None:
            try:
                if name_embedding is None:
                    name_embedding = get_embedding(tok, model, name)
                concept_embedding = get_embedding(tok, model, hit[1])
                consistency_score = round(_cosine_similarity(name_embedding, concept_embedding), 4)
                if consistency_score >= SPAN_CONSISTENCY_FLOOR:
                    status = "auto_accept"
            except Exception:
                pass  # stay conservative (needs_review) if embedding fails

        result = _exact_hit_result(hit, tier_label, "gliner_span", span, status=status)
        if consistency_score is not None:
            result["span_consistency_score"] = consistency_score
        cache[cache_key] = result
        return result

    # --- Neither the full name nor a GLiNER span landed an exact hit --
    # before giving up on exact matching entirely and falling to Tier 3's
    # blind embedding search, check whether the upstream MedCAT
    # entity-linking pass (run earlier in the pipeline, before triplet
    # extraction) already found and linked a SNOMED CUI for some part of
    # this node's name. See "MEDCAT ENTITY REUSE" in the module docstring
    # for why this candidate exists (many CUIs MedCAT found never survived
    # the injection-qualification threshold, so the triplet extractor never
    # saw them) and why it's gated exactly like a GLiNER span rather than
    # trusted outright (MedCAT's own confidence_score does not protect
    # against homonym collisions / boilerplate mislinks -- see S3.7).
    if medcat_cache is not None:
        entities = get_medcat_entities(medcat_cache, source_document, medcat_dir)
        for _num_tokens, medcat_confidence, cui, matched_text, _medcat_name in \
                find_medcat_candidates(name, entities):
            hit = resolve_snomed_cui(conn, cui, domains=allowed_domains)
            if not hit:
                continue

            status = "needs_review"
            consistency_score = None
            if _SAPBERT_AVAILABLE and tok is not None and model is not None:
                try:
                    if name_embedding is None:
                        name_embedding = get_embedding(tok, model, name)
                    concept_embedding = get_embedding(tok, model, hit[1])
                    consistency_score = round(_cosine_similarity(name_embedding, concept_embedding), 4)
                    # Both gates required -- see MEDCAT_CONFIDENCE_FLOOR's own
                    # comment: consistency alone validates the concept against
                    # the node name, not whether MedCAT's own linker was
                    # confident this span mapped to that CUI in the first
                    # place. A low-confidence link that happens to be a
                    # generic word pair (e.g. "diagnostic"/"Diagnosis") can
                    # pass consistency on lexical similarity alone.
                    if (consistency_score >= SPAN_CONSISTENCY_FLOOR
                            and medcat_confidence >= MEDCAT_CONFIDENCE_FLOOR):
                        status = "auto_accept"
                except Exception:
                    pass  # stay conservative (needs_review) if embedding fails

            result = {"status": status, "tier": "1/2 (via MedCAT entity)",
                      "concept_id": hit[0], "concept_name": hit[1], "domain_id": hit[2],
                      "vocab": hit[3], "score": 1.0, "resolved_via": "medcat_entity",
                      "matched_text": matched_text, "medcat_cui": cui,
                      "medcat_confidence": round(medcat_confidence, 4)}
            if consistency_score is not None:
                result["span_consistency_score"] = consistency_score
            cache[cache_key] = result
            return result

    if not _SAPBERT_AVAILABLE:
        result = {"status": "no_match", "tier": "0 (Failed - SapBERT unavailable)",
                  "concept_id": None, "concept_name": None, "domain_id": None,
                  "vocab": None, "score": 0.0, "resolved_via": "full_name"}
        cache[cache_key] = result
        return result

    # --- Neither the full name nor any extracted span got an exact hit --
    # fall back to Tier 3 semantic search on the full name, same as before. ---
    hit = tier3_semantic(conn, tok, model, name, vocabs)
    if hit and hit[4] >= TIER3_SIMILARITY_FLOOR:
        # 2026-08-07 (sixth addition): previously always needs_review
        # regardless of score -- see AUTO_ACCEPT_FLOOR's comment for the
        # policy change. Since TIER3_SIMILARITY_FLOOR (0.72) already exceeds
        # AUTO_ACCEPT_FLOOR (0.70), this branch is currently unconditional
        # for every candidate that reaches it; the `>=` check is kept
        # explicit rather than collapsed so this still does the right thing
        # if either floor is retuned independently later.
        status = "auto_accept" if hit[4] >= AUTO_ACCEPT_FLOOR else "needs_review"
        result = {"status": status, "tier": "3 (Semantic)", "concept_id": hit[0],
                  "concept_name": hit[1], "domain_id": hit[2], "vocab": hit[3],
                  "score": round(hit[4], 4), "resolved_via": "full_name"}
    else:
        result = {"status": "no_match", "tier": "0 (Failed)", "concept_id": None,
                  "concept_name": None, "domain_id": None, "vocab": None,
                  "score": round(hit[4], 4) if hit else 0.0, "resolved_via": "full_name"}
    cache[cache_key] = result
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Backfill missing SNOMED/ICD10 codes in the curated guideline triplet corpus."
    )
    parser.add_argument("--apply", action="store_true",
                         help="Also write auto-accepted (Tier 1/2 exact) matches into a new "
                              "local_triplets_db2_v6_grounded/ copy. Default is dry-run, report only.")
    parser.add_argument("--triplets-dir", default=TRIPLETS_DIR)
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--no-gliner", action="store_true",
                         help="Skip the GLiNER compound-name span-extraction step; behave exactly "
                              "as the pre-2026-08-07 version (full-name Tier 1/2 then Tier 3 only).")
    parser.add_argument("--no-medcat", action="store_true",
                         help="Skip the MedCAT-entity-reuse step (see module docstring); behave as "
                              "if data/triplets-rules-backup-data/local_medcat_entities_db2_v6 "
                              "doesn't exist.")
    parser.add_argument("--medcat-dir", default=MEDCAT_ENTITIES_DIR,
                         help="Directory containing <chunk>_medcat.json entity files.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Stop after processing this many ungrounded nodes. Useful for a quick "
                              "smoke test before committing to a full run.")
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        sys.exit(f"DuckDB not found at {args.db_path}. Run scripts/import_athena.py and "
                  f"scripts/build_concept_embeddings.py first.")

    conn = duckdb.connect(args.db_path, read_only=True)

    # Tier 3 runs a full-table cosine-similarity scan (no vector index -- see
    # Databases.md's own "Known Configuration & Engineering Open Items" #4).
    # Printing the table size up front makes it obvious if that's why a run
    # feels stuck: a large athena_concept table means every Tier-3 fallback
    # is a genuinely slow, CPU-bound full scan, not a hang.
    try:
        concept_count = conn.sql(
            "SELECT COUNT(*) FROM athena_concept WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        print(f"athena_concept rows with embeddings (Tier-3 scan size per query): {concept_count:,}")
    except Exception:
        pass

    tok = model = None
    if _SAPBERT_AVAILABLE:
        tok, model = load_sapbert()
    else:
        print("WARNING: transformers/torch not available -- Tier 3 semantic matching AND the "
              "GLiNER-span/MedCAT-entity consistency checks will all be skipped. GLiNER-span and "
              "MedCAT-entity exact/synonym hits will all be routed to needs_review (no way to "
              "verify representativeness without SapBERT), and only full-name Tier 1/2 matches "
              "will auto-accept.")

    gliner_model = None
    if not args.no_gliner:
        if _GLINER_AVAILABLE:
            gliner_model = load_gliner()
        else:
            print("WARNING: gliner package not available -- compound node names will go straight "
                  "to Tier 3 on the full name, same as before this feature existed.")

    medcat_cache = None
    if not args.no_medcat:
        if os.path.isdir(args.medcat_dir):
            medcat_cache = {}
        else:
            print(f"WARNING: MedCAT entities directory not found at {args.medcat_dir} -- "
                  f"MedCAT-entity-reuse step will be skipped, same as --no-medcat.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_dir = args.triplets_dir.rstrip("/\\") + "_grounded"
    if args.apply:
        os.makedirs(out_dir, exist_ok=True)

    summary = {"auto_accept": 0, "auto_accept_via_gliner_span": 0, "auto_accept_via_medcat": 0,
               "needs_review": 0, "needs_review_via_gliner_span": 0, "needs_review_via_medcat": 0,
               "no_match": 0, "skipped_type": 0}
    review_rows = []
    cache = {}

    files = sorted(glob.glob(os.path.join(args.triplets_dir, "*.json")))
    if not files:
        sys.exit(f"No triplet files found in {args.triplets_dir}")

    # Pre-scan just to report how much work there is before starting the
    # (potentially slow) grounding loop -- without this, the script prints
    # nothing for its entire runtime, which looks identical to a hang.
    total_ungrounded = 0
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for node in data.get("@graph", []):
            snomed = node.get("snomed")
            if (not snomed or str(snomed).strip().upper() == "N/A") and node.get("@type") not in SKIP_TYPES and node.get("name"):
                total_ungrounded += 1

    print(f"Scanning {len(files)} triplet files in {args.triplets_dir} "
          f"({total_ungrounded} ungrounded nodes to process"
          + (f", capped at --limit {args.limit}" if args.limit else "") + ") ...", flush=True)

    processed = 0
    limit_hit = False
    for file_idx, fp in enumerate(files, 1):
        if limit_hit:
            break
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)

        file_ungrounded = sum(
            1 for n in data.get("@graph", [])
            if (not n.get("snomed") or str(n.get("snomed")).strip().upper() == "N/A")
            and n.get("@type") not in SKIP_TYPES and n.get("name")
        )
        print(f"[{file_idx}/{len(files)}] {os.path.basename(fp)} "
              f"({file_ungrounded} ungrounded node(s)) ...", flush=True)

        changed = False
        for node in data.get("@graph", []):
            if args.limit and processed >= args.limit:
                limit_hit = True
                break
            snomed = node.get("snomed")
            if snomed and str(snomed).strip().upper() != "N/A":
                continue  # already grounded, leave untouched

            node_type = node.get("@type")
            name = node.get("name", "")
            if node_type in SKIP_TYPES or not name:
                summary["skipped_type"] += 1
                continue

            provenance = node.get("provenance")
            # Defensive: at least one node in the corpus has provenance == ""
            # (empty string) instead of the expected {source_document, ...}
            # dict -- a pre-existing data-quality quirk, not something to
            # crash on. Treat anything non-dict the same as "no provenance
            # info", which just means the MedCAT-entity-reuse tier gets no
            # candidates for that node and falls through to Tier 3, same as
            # before this feature existed.
            source_document = provenance.get("source_document") if isinstance(provenance, dict) else None
            result = ground_node(conn, tok, model, gliner_model, name, node_type, cache,
                                  source_document=source_document, medcat_cache=medcat_cache,
                                  medcat_dir=args.medcat_dir)
            summary[result["status"]] += 1
            resolved_via = result.get("resolved_via")
            if resolved_via == "gliner_span":
                if result["status"] == "auto_accept":
                    summary["auto_accept_via_gliner_span"] += 1
                elif result["status"] == "needs_review":
                    summary["needs_review_via_gliner_span"] += 1
            elif resolved_via == "medcat_entity":
                if result["status"] == "auto_accept":
                    summary["auto_accept_via_medcat"] += 1
                elif result["status"] == "needs_review":
                    summary["needs_review_via_medcat"] += 1
            processed += 1
            if processed % 25 == 0 or processed == total_ungrounded:
                print(f"  ... {processed}/{total_ungrounded} processed "
                      f"(auto_accept={summary['auto_accept']}, needs_review={summary['needs_review']}, "
                      f"no_match={summary['no_match']})", flush=True)

            row = {
                "source_file": os.path.basename(fp),
                "node_id": node.get("@id"),
                "name": name,
                "type": node_type,
                **result,
            }

            if result["status"] == "auto_accept":
                icd = icd10_crosswalk(conn, result["concept_id"])
                row["icd10_code"] = icd[0] if icd else None
                row["icd10_name"] = icd[1] if icd else None
                if args.apply:
                    node["snomed"] = str(result["concept_id"])
                    node["snomed_name"] = result["concept_name"]
                    if icd:
                        node["icd10"] = icd[0]
                    node["grounding_provenance"] = {
                        "source": "backfill_guideline_grounding",
                        "tier": result["tier"],
                        "grounded_at": datetime.now(timezone.utc).isoformat(),
                    }
                    changed = True

            review_rows.append(row)

        if args.apply and changed:
            out_path = os.path.join(out_dir, os.path.basename(fp))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"grounding_backfill_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": review_rows}, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("BACKFILL SUMMARY")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFull report: {report_path}")
    if args.apply:
        print(f"Auto-accepted (Tier 1/2) matches written to: {out_dir}")
    print(f"\n{summary['needs_review']} Tier-3 candidates need human sign-off before use -- "
          f"see rows with status=needs_review in the report. These are NOT written by --apply, "
          f"by design (see module docstring).")


if __name__ == "__main__":
    main()
