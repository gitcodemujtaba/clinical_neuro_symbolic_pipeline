"""
src/retrieval.py — Stage 3 grounding retrieval.

Implements docs/MoLLM_Stage3_Retrieval_Design.md S5: given a Stage 2
ValidationRecord, find the guideline rules and ontology context that apply to
it. Kept separate from mollm_ensemble.py for one specific reason -- retrieval
coverage can then be measured across the entire 272-note gold set with ZERO
LLM calls, which is the first experiment that should be run once the
vocabulary is loaded (design doc S9).

THE TWO MEASUREMENTS THAT SHAPED THIS MODULE (both taken directly from the
corpus, not estimated -- see design doc S2):

1. Exact SNOMED-code matching reaches 10.80% of gold annotations; adding exact
   name matching gets the union to 12.48%. The curated corpus holds only 447
   distinct SNOMED codes. So ~87.5% of entities have NO directly-matching
   guideline rule, "nothing retrieved" is the COMMON case rather than an
   error, and hierarchy traversal (Channel B) is load-bearing rather than an
   optimisation.

2. Of the 91 codes carrying more than one node name, 43 (47%) attach
   clinically unrelated names to a single code -- 24484000 (the qualifier
   "Severe") holds "GOLD 3 (severe)", "major bleeding" and "severe AKI";
   272118002 holds both "acute NSTEMI" and "ST-segment elevation myocardial
   infarction", which are clinically opposite and drive different reperfusion
   decisions. A naive code match would therefore hand an NSTEMI patient's
   entity a STEMI reperfusion rule, complete with a verbatim citation. That is
   worse than retrieving nothing: it is an authoritative-looking wrong answer
   in the scenario where being wrong costs most. Hence name_agreement_guard(),
   applied to EVERY code-based match.

WHY OMOP's athena_concept_ancestor RATHER THAN A GRAPH FOR HIERARCHY:
Channel B needs SNOMED IS_A ancestors. DuckDB already has (or will have, once
scripts/import_athena.py runs) athena_concept_ancestor, which is precisely a
materialised transitive closure of the OMOP hierarchy with hop counts --
exactly what the traversal needs, without waiting on the graph import or
paying a Bolt round-trip per entity. DuckDBHierarchy is therefore the default
provider; Neo4jHierarchy remains available for when KG1 is populated and for
guideline-specific traversals the OMOP closure cannot express. Both satisfy
the same tiny interface so the choice is a constructor argument, not a
rewrite.
"""

import json
import glob
import os
import re
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline"
# 2026-08-11 Stage3 Issue1 rule backfill -- see docs/Stage3_Issue1_Rule_Backfill.md.
# Callers that pick from scripts/*.py's own TRIPLETS_CANDIDATES lists (test_stage3_live.py,
# profile_stage3.py, diagnose_guard_suppression.py, measure_channel_b_coverage.py) are
# unaffected by this default; this only matters for a caller that constructs
# GuidelineIndex() with no argument.
DEFAULT_TRIPLETS_DIR = os.path.join(PROJECT_DIR, "data", "local_triplets_db2_v6_cleaned_grounded_rules_added")

# --- name-agreement guard bands (design doc S5.1) --------------------------
NAME_AGREE_STRONG = 0.75
NAME_AGREE_REJECT = 0.45
WEAK_AGREEMENT_PENALTY = 0.6

# --- hierarchy traversal (design doc S5.3) --------------------------------
MAX_HIERARCHY_HOPS = 3
HIERARCHY_DECAY = 0.9

# SNOMED concepts so general that every clinical entity subsumes under them;
# traversing through these makes every entity match every rule and renders
# match_confidence meaningless. `Qualifier value` is listed for a concrete
# reason: it is how the 24484000 "Severe" collision propagates, so barring it
# kills a whole class of false matches at the traversal level rather than
# relying on the name guard alone to catch each one.
HIERARCHY_STOP_CODES = {
    "138875005",  # SNOMED CT Concept (root)
    "404684003",  # Clinical finding
    "64572001",   # Disease
    "71388002",   # Procedure
    "123037004",  # Body structure
    "362981000",  # Qualifier value
    "105590001",  # Substance
    "410607006",  # Organism
    "243796009",  # Situation with explicit context
    "272379006",  # Event
}

# --- evidence budget (design doc S5.5) ------------------------------------
MAX_RULES_PER_RECORD = 5
CITATION_TYPE_RANK = {
    "verbatim": 0,
    "paraphrase_with_recovered_excerpt": 1,
    "paraphrase": 2,
    "pointer_unverifiable": 3,
    None: 4,
}
CITABLE_TYPES = {"verbatim", "paraphrase_with_recovered_excerpt", "paraphrase"}

# --- node @type compatibility ---------------------------------------------
# Maps each GLiNER entity label to the guideline node @type values that
# plausibly describe the same kind of thing. Corpus @type distribution:
# Finding 834, Condition 284, Intervention 263, Acuity 135, Medication 115,
# Quantitative Threshold 63, Timeframe 1.
#
# SOFT SIGNAL, NOT A FILTER — and that distinction is forced by the data.
# The corpus's own @type assignments are internally inconsistent: the cleaning
# report records 41+ groups where one SNOMED code carries conflicting types
# ("Statins" as both Finding and Medication, "Endotracheal intubation" as both
# Intervention and Finding, "ICU admission" as Finding/Condition/Intervention).
# Those are curation artefacts, not clinical distinctions, so a hard type gate
# would suppress correct evidence roughly as often as wrong evidence. A mild
# downweight expresses "this is less likely to be about the same kind of
# thing" without pretending the type labels are authoritative.
#
# `Anatomy` maps to nothing deliberately: the guideline corpus has no
# anatomical node type at all, so every match for an Anatomy entity is a type
# mismatch by construction and penalising it would be meaningless. An empty
# set means "no expectation", scored neutrally.
TYPE_COMPATIBILITY = {
    "Condition": {"Condition", "Finding", "Acuity"},
    "Symptom": {"Finding", "Condition", "Acuity"},
    "Medication": {"Medication", "Intervention"},
    "Procedure": {"Intervention", "Medication"},
    "Lab Test": {"Finding", "Quantitative Threshold"},
    "Anatomy": set(),
}
TYPE_MISMATCH_PENALTY = 0.85

# The 2b->KG bridge. GLINER_LABEL_TO_DOMAIN (normalization.py) bridges the
# GLiNER label to the OMOP domain, and TYPE_COMPATIBILITY above bridges the
# GLiNER label to the guideline @type -- but nothing connected the OMOP domain
# to the guideline @type, so both existing checks descended from the SAME
# source. If GLiNER mislabels a span, it corrupts the domain filter and the
# type-agreement score in the same direction, and one wrong label looks like
# two independent confirmations.
#
# The OMOP domain is derived from the matched CONCEPT, not from the extractor,
# so it fails independently. Agreement between two signals with different
# provenance is worth considerably more than agreement between two views of
# one signal.
DOMAIN_TO_TYPE = {
    "Condition": {"Condition", "Finding", "Acuity"},
    "Observation": {"Finding", "Acuity", "Quantitative Threshold"},
    "Drug": {"Medication", "Intervention"},
    "Procedure": {"Intervention"},
    "Measurement": {"Finding", "Quantitative Threshold"},
    "Spec Anatomic Site": set(),
}

# Predicates that speak to a given GLiNER label. Used only as a final
# tiebreaker among rules of comparable match_confidence -- never to filter,
# because a rule with an "unrelated" predicate can still be the one that
# contradicts the extraction.
PREDICATE_AFFINITY = {
    "Medication": {"REQUIRES_MEDICATION", "NOT_RECOMMENDED_FOR", "RECOMMENDED_FOR",
                   "CONTRAINDICATED_IF", "REQUIRES_DOSAGE", "ADMINISTERED_FOR"},
    "Procedure": {"REQUIRES_INTERVENTION", "REQUIRES_MANAGEMENT", "IS_USED_IN",
                  "NOT_RECOMMENDED_FOR", "RECOMMENDED_FOR"},
    "Lab Test": {"INDICATES", "HAS_QUANTITATIVE_THRESHOLD", "IS_ASSESSED_BY", "DEFINED_BY"},
    "Condition": {"INDICATES", "TRIGGERS_SEVERITY", "REQUIRES_INTERVENTION",
                  "HAS_ETIOLOGY", "DEFINED_BY", "HAS_CRITERION"},
    "Symptom": {"INDICATES", "TRIGGERS_SEVERITY", "PRESENTS_WITH", "HAS_CRITERION"},
    "Anatomy": set(),
}

# Interim boilerplate suppression. docs/Implementation_Checklist.md and
# Guideline_Triplets_KG_Review.md S6 both state boilerplate nodes were tagged
# `quality_flag: likely_boilerplate` during cleaning -- verified against the
# cleaned corpus on 2026-08-08, ZERO nodes and ZERO rules actually carry that
# flag. The cleaning script's patterns target the grounded-chunk text but are
# applied to triplet node names, where the boilerplate surfaces differently
# ("American College of Emergency Physicians" is a real node name under code
# 25876001 and is not in the pattern list at all). Suppressed here at query
# time until the cleaning pass is fixed; harmless to leave in place afterwards.
BOILERPLATE_NAME_PATTERNS = [
    re.compile(r"annals of emergency medicine", re.I),
    re.compile(r"american college of emergency physicians", re.I),
    re.compile(r"key words?/phrases for literature searches", re.I),
    re.compile(r"^study selection", re.I),
    re.compile(r"^clinical policy$", re.I),
    re.compile(r"literature (search|classification)", re.I),
]

_STOPWORDS = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "with",
              "by", "at", "is", "was", "be", "as", "from"}

# POLARITY TOKENS — a mismatch on any of these forces rejection, whatever the
# rest of the overlap looks like.
#
# Found by testing the guard against the real collision cases rather than by
# reasoning about it in the abstract, and it was failing on the worst one:
# entity `NSTEMI` was matching guideline node `ST-segment elevation myocardial
# infarction` at FULL confidence, because NSTEMI's own SNOMED FSN is "Acute
# NON-ST segment elevation myocardial infarction" -- which contains every
# content token of the STEMI name. Token overlap alone therefore scored a
# perfect match between two clinically opposite conditions that drive
# different reperfusion decisions. The single word carrying the whole
# distinction, "non", was being treated as ordinary filler.
#
# These tokens invert meaning rather than qualify it, so any asymmetry is
# disqualifying and no amount of other overlap can compensate.
POLARITY_TOKENS = {"non", "not", "no", "without", "absent", "absence", "negative",
                   "excluding", "excluded", "ruled", "denies", "free"}

# GENERIC CLINICAL TOKENS — carry no discriminating content on their own.
#
# Also found by testing: `CURB-65` matched `HEART Score` because their only
# shared token was "score". Both are risk instruments; neither fact makes them
# the same instrument. An overlap consisting ONLY of these tokens is treated as
# no overlap, rather than being allowed to drag two unrelated concepts into a
# weak match. They are NOT removed from the token sets (that would break
# genuinely-matching pairs like "Stage C HF"), only disallowed as the sole
# basis for a match.
GENERIC_CLINICAL_TOKENS = {
    "score", "scale", "index", "level", "levels", "grade", "class", "classification",
    "severe", "severity", "mild", "moderate", "acute", "chronic", "high", "low",
    "increased", "decreased", "elevated", "reduced", "risk", "patient", "patients",
    "therapy", "treatment", "management", "care", "disease", "disorder", "syndrome",
    "test", "testing", "value", "values", "result", "results", "status", "type",
    "use", "using", "assessment", "evaluation", "criteria", "criterion",
}


# ==========================================================================
# Name agreement
# ==========================================================================

def _tokens(text: str) -> set:
    if not text:
        return set()
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS} or set(words)


def _is_acronym_of(short: str, long_text: str) -> bool:
    """True when `short` is an acronym of `long_text`.

    MEASURED NEED (2026-08-08). Guideline authors write the clinical short
    form; SNOMED stores the formal one. The guard was rejecting these as
    mismatches:
        'White blood cell count'                  vs 'WBC'    (581 annotations)
        'Chronic obstructive pulmonary disease'   vs 'COPD'   (112)
        'Left bundle branch block'                vs 'LBBB'
        'Continuous positive airway pressure ...' vs 'CPAP'
        'ST segment elevation myocardial infarc.' vs 'STEMI'
    The first attempt at fixing this used athena_concept_synonym on the
    assumption that these are registered synonyms. They largely are not --
    OMOP's synonym table for SNOMED carries formal description variants rather
    than clinical acronyms. Rather than assume a second time, this derives the
    relationship from the two strings themselves.

    Deliberately strict, because acronym matching is easy to make promiscuous:
      * the short form must be 2-6 characters, alphabetic, and not longer than
        the number of words available to it;
      * its letters must appear as WORD-INITIALS IN ORDER (subsequence, not
        anagram), so 'DVT' cannot match 'Ventricular Tachycardia Disorder';
      * a leading multi-letter run is allowed to consume one word ('ST' from
        'ST segment'), which is what makes STEMI work.
    """
    short = re.sub(r"[^a-z]", "", (short or "").lower())
    if not (2 <= len(short) <= 6) or not long_text:
        return False
    words = [w for w in re.findall(r"[a-z]+", long_text.lower())
             if w not in _STOPWORDS]
    if len(words) < 2 or len(short) > len(words) + 1:
        return False

    i = 0
    for w in words:
        if i < len(short) and w[0] == short[i]:
            i += 1
            # Allow one word to supply a two-letter prefix ('st' <- 'st'),
            # which is how ST-elevation acronyms are built.
            if i < len(short) and len(w) >= 2 and w[1] == short[i] and w[:2] == short[i - 1:i + 1]:
                i += 1
    return i == len(short)


def _tokens_match(x: str, y: str) -> bool:
    """Token equality with light morphological tolerance.

    Exact match, or one token is a >=5-character prefix of the other. That
    single rule covers the inflectional variants the guard was rejecting
    ("Unresponsive" vs "Unresponsiveness", "smoker" vs "smokers") without
    resorting to a stemmer, which would introduce its own dependency and its
    own errors. The 5-character floor is what stops it degenerating: shorter
    prefixes would make "sep" bridge "sepsis" and "separation".
    """
    if x == y:
        return True
    if len(x) >= 5 and y.startswith(x):
        return True
    if len(y) >= 5 and x.startswith(y):
        return True
    # Shared-stem match for inflectional variants the prefix rule misses,
    # e.g. 'lethargy'/'lethargic' (common prefix 'lethar', neither is a prefix
    # of the other). The RATIO requirement is what keeps this honest:
    # 'hypertension'/'hyperthyroidism' share 6 leading characters too, but only
    # 0.40 of the longer token, so they stay separate.
    n = 0
    for cx, cy in zip(x, y):
        if cx != cy:
            break
        n += 1
    return n >= 4 and n / max(len(x), len(y)) >= 0.6


def token_set_ratio(a: str, b: str) -> float:
    """Jaccard-style overlap over content tokens.

    Deliberately not an edit-distance measure and deliberately no new
    dependency: clinical concept names differ by word choice and word order
    ("Acute Kidney Injury (AKI)" vs "AKI"), not by character-level typos, and
    character distance scores those as dissimilar while scoring the genuinely
    dangerous "acute NSTEMI" / "ST-segment elevation myocardial infarction"
    pair as moderately similar. Token containment gets both cases right.

    Uses the SMALLER token set as denominator so an abbreviation fully
    contained in an expanded name scores 1.0 rather than being penalised for
    brevity.

    Two overrides, both added after testing against the real collisions in the
    corpus (see POLARITY_TOKENS and GENERIC_CLINICAL_TOKENS):
      * asymmetric polarity ("non-ST elevation" vs "ST elevation") returns 0
        regardless of how much else overlaps;
      * an overlap consisting only of generic clinical filler ("score")
        returns 0, because it is not evidence of anything.

    KNOWN LIMITATION: matching is lexical, so morphological variants score 0
    ("sepsis" vs "septic", "cardiac" vs "heart"). This is the right trade-off
    here -- the guard's job is to stop a wrong code assertion from delivering
    wrong evidence, so a false reject costs one missed rule while a false
    accept delivers an authoritative-looking wrong one. Genuine semantic
    relationships between distinct concepts are Channel B's job (hierarchy),
    not this function's. Measured effect on the real corpus: the guard
    suppresses 26.1% of same-code node pairs.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0

    if (ta & POLARITY_TOKENS) != (tb & POLARITY_TOKENS):
        return 0.0

    shared = set()
    for x in ta:
        for y in tb:
            if _tokens_match(x, y):
                shared.add(x)
                break
    if not shared or shared <= GENERIC_CLINICAL_TOKENS:
        return 0.0

    return len(shared) / min(len(ta), len(tb))


def name_agreement_guard(entity_name: str, concept_fsn: str, node_name: str,
                         concept_synonyms=None) -> tuple:
    """(multiplier, status) for a code-based match. See design doc S5.1.

    Scored against BOTH the entity's surface text and the SNOMED FSN, taking
    the better of the two: an entity extracted as "NSTEMI" and a guideline node
    named "acute NSTEMI" agree at FSN level even when the raw surface forms do
    not, while 24484000's "major bleeding" vs "severe AKI" fail against both.
    """
    # Scored against the concept's FULL NAME SET, not just its FSN.
    #
    # MEASURED FAILURE (2026-08-08). Comparing against the FSN alone rejected
    # matches that are plainly the same concept, because guideline authors
    # write the clinical short form while SNOMED stores the formal one:
    #   'White blood cell count'                        vs 'WBC'      (581 ann)
    #   'Chronic obstructive pulmonary disease'         vs 'COPD'     (112 ann)
    #   'Cerebrovascular accident'                      vs 'Stroke'   (173 ann)
    #   'Continuous positive airway pressure ventilat.' vs 'CPAP'
    #   'Left bundle branch block'                      vs 'LBBB'
    # Every one of those is a registered synonym in athena_concept_synonym --
    # 2.79M rows that exist for exactly this purpose and that the guard was
    # ignoring. Using them replaces a fuzzier string threshold with the
    # vocabulary's own authority, which is both more accurate and easier to
    # defend than tuning a similarity cutoff.
    candidates = [entity_name or "", concept_fsn or ""]
    if concept_synonyms:
        candidates.extend(concept_synonyms)
    node = node_name or ""
    score = max(token_set_ratio(c, node) for c in candidates if c) if any(
        c for c in candidates) else 0.0

    # Acronym relationship in either direction, checked only if token overlap
    # already failed -- it is a fallback, not a competing signal.
    if score < NAME_AGREE_STRONG:
        for c in candidates:
            if not c:
                continue
            if _is_acronym_of(node, c) or _is_acronym_of(c, node):
                score = 1.0
                break
    if score >= NAME_AGREE_STRONG:
        return 1.0, "agree"
    if score >= NAME_AGREE_REJECT:
        return WEAK_AGREEMENT_PENALTY, "weak"
    return 0.0, "reject"


def is_boilerplate(name: str) -> bool:
    return any(p.search(name or "") for p in BOILERPLATE_NAME_PATTERNS)


def type_agreement(gliner_label: str, node_type: str) -> tuple:
    """(multiplier, status) for GLiNER label vs guideline node @type.

    Three outcomes rather than two, because "we have no expectation" and "we
    expected something else" are different situations and collapsing them
    would penalise Anatomy entities for a gap in the corpus rather than for
    anything about the match:

      * `agree`   — the node type is one the label plausibly denotes.
      * `neutral` — no expectation registered (unknown label, missing @type,
                    or a label like Anatomy the corpus has no type for).
      * `mismatch`— an expectation exists and this is not it. Downweighted,
                    never rejected: see TYPE_COMPATIBILITY for why the corpus's
                    own types cannot bear a hard filter.

    Surfaced to Stage 3 as well as used in ranking. A guideline node typed
    `Intervention` matched against an entity GLiNER labelled `Medication` is
    exactly the kind of disagreement the MoLLM should see and reason about --
    it may indicate the extraction label is wrong, which is a finding in its
    own right rather than noise to hide.
    """
    expected = TYPE_COMPATIBILITY.get(gliner_label)
    if not expected or not node_type:
        return 1.0, "neutral"
    if node_type in expected:
        return 1.0, "agree"
    return TYPE_MISMATCH_PENALTY, "mismatch"


def combined_type_agreement(gliner_label: str, omop_domain: str, node_type: str) -> tuple:
    """Type agreement from BOTH the extraction label and the matched concept's
    OMOP domain. Returns (multiplier, status).

    Two checks with independent provenance (see DOMAIN_TO_TYPE). The penalty is
    applied ONCE even when both disagree rather than compounding to 0.72,
    because the two signals are correlated in practice -- most of the time a
    wrong label and a wrong domain have the same root cause, and squaring the
    penalty would overstate how much independent evidence there is.

    Status is the more informative of the two outcomes:
      * `agree`        — both agree, or one agrees and the other has no opinion
      * `both_mismatch`— both disagree. The strongest available signal that the
                         retrieved rule is about a different kind of thing, and
                         worth naming separately so Stage 3 can say so plainly.
      * `mismatch`     — one disagrees, the other is neutral or agrees.
      * `neutral`      — neither has an opinion.
    """
    label_mult, label_status = type_agreement(gliner_label, node_type)

    expected = DOMAIN_TO_TYPE.get(omop_domain)
    if not expected or not node_type:
        domain_status = "neutral"
    elif node_type in expected:
        domain_status = "agree"
    else:
        domain_status = "mismatch"

    statuses = {label_status, domain_status}
    if label_status == "mismatch" and domain_status == "mismatch":
        return TYPE_MISMATCH_PENALTY, "both_mismatch"
    if "mismatch" in statuses:
        return TYPE_MISMATCH_PENALTY, "mismatch"
    if "agree" in statuses:
        return 1.0, "agree"
    return 1.0, "neutral"


# ==========================================================================
# Guideline index (Channels A and D source)
# ==========================================================================

class GuidelineIndex:
    """In-memory index over the cleaned JSON-LD guideline corpus.

    File-backed rather than graph-backed on purpose. The corpus is small (1,697
    nodes / 1,119 rules across 76 files) so it loads in well under a second and
    fits comfortably in memory, and this lets retrieval coverage be measured
    over the whole gold set TODAY, before Memgraph ingestion exists. The
    Memgraph-backed implementation can be swapped in behind the same interface
    later without touching mollm_ensemble.py.

    CRITICAL INGESTION RULE (design doc S4.2): nodes are indexed by SNOMED code
    but NEVER merged by it. Guideline_Triplets_KG_Review.md S3.2 recommends
    `MERGE ... ON snomed`; that recommendation is superseded, because merging
    24484000's nodes would permanently fuse "major bleeding" with "severe AKI".
    A code maps to a LIST of distinct nodes here, and the name guard decides
    which of them a given entity may legitimately reach.
    """

    def __init__(self, triplets_dir: str = DEFAULT_TRIPLETS_DIR):
        self.nodes_by_code = defaultdict(list)
        self.nodes_by_name = defaultdict(list)
        self.nodes_by_uid = {}
        self.rules_by_source = defaultdict(list)
        self.rules_by_target = defaultdict(list)
        self.stats = {"files": 0, "nodes": 0, "rules": 0, "grounded_nodes": 0}
        self._load(triplets_dir)

    def _load(self, triplets_dir: str):
        for path in sorted(glob.glob(os.path.join(triplets_dir, "*.json"))):
            fname = os.path.basename(path)
            try:
                graph = json.load(open(path, encoding="utf-8")).get("@graph", [])
            except (json.JSONDecodeError, OSError):
                continue
            self.stats["files"] += 1

            for node in graph:
                # Identity is (file, @id): @id is a within-file provenance
                # handle, not a graph identity.
                uid = (fname, node.get("@id"))
                name = node.get("name") or ""
                code = node.get("snomed")
                record = {
                    "uid": uid,
                    "file": fname,
                    "name": name,
                    "node_type": node.get("@type"),
                    "snomed_code": code if code and code != "N/A" else None,
                    "quality_flag": node.get("quality_flag"),
                    "source_document": (node.get("provenance") or {}).get("source_document"),
                    "section_title": (node.get("provenance") or {}).get("section_title"),
                }
                self.nodes_by_uid[uid] = record
                self.stats["nodes"] += 1
                if record["snomed_code"]:
                    self.nodes_by_code[record["snomed_code"]].append(record)
                    self.stats["grounded_nodes"] += 1
                if name:
                    self.nodes_by_name[name.strip().lower()].append(record)

                for rule in node.get("rules", []):
                    target_uid = (fname, rule.get("target"))
                    r = {
                        "source_uid": uid,
                        "target_uid": target_uid,
                        "predicate": rule.get("predicate"),
                        "rationale": rule.get("rationale"),
                        "citation": rule.get("citation"),
                        "citation_verbatim_excerpt": rule.get("citation_verbatim_excerpt"),
                        "citation_type": rule.get("citation_type"),
                        "quality_flag": rule.get("quality_flag"),
                    }
                    self.rules_by_source[uid].append(r)
                    self.rules_by_target[target_uid].append(r)
                    self.stats["rules"] += 1

    def rules_touching(self, node_uid) -> list:
        """Rules where the node is EITHER endpoint.

        Both directions matter: a rule stating "AKI REQUIRES_INTERVENTION
        dialysis" is equally relevant whether the extracted entity was the AKI
        or the dialysis. Retrieving only outgoing edges would silently halve
        coverage on a corpus where 943 of 1,697 nodes have no outgoing rules at
        all.
        """
        return (
            [dict(r, direction="outgoing") for r in self.rules_by_source.get(node_uid, [])]
            + [dict(r, direction="incoming") for r in self.rules_by_target.get(node_uid, [])]
        )

    def nodes_for_code(self, code: str) -> list:
        return self.nodes_by_code.get(str(code), [])

    def nodes_for_name(self, name: str) -> list:
        return self.nodes_by_name.get((name or "").strip().lower(), [])


# ==========================================================================
# Hierarchy providers (Channel B)
# ==========================================================================

class DuckDBHierarchy:
    """SNOMED ancestors via OMOP's athena_concept_ancestor closure.

    Upward only, and capped. Both constraints are clinical, not technical:
      * Downward traversal (AKI -> its subtypes) would retrieve rules about
        conditions the note never documented. Generalising from what the note
        says is valid inference; specialising into what it does not say is not.
      * Beyond ~3 levels SNOMED converges on near-root concepts that subsume
        most of the corpus, at which point every entity matches every rule.
    """

    def __init__(self, conn):
        self.conn = conn
        self._cache = {}

    def ancestors(self, snomed_code: str, max_hops: int = MAX_HIERARCHY_HOPS) -> list:
        """Returns [(ancestor_snomed_code, hops)], nearest first."""
        key = (snomed_code, max_hops)
        if key in self._cache:
            return self._cache[key]

        query = """
        SELECT anc.concept_code, MIN(a.min_levels_of_separation) AS hops
        FROM athena_concept d
        JOIN athena_concept_ancestor a ON a.descendant_concept_id = d.concept_id
        JOIN athena_concept anc ON anc.concept_id = a.ancestor_concept_id
        WHERE d.concept_code = ? AND d.vocabulary_id = 'SNOMED'
          AND anc.vocabulary_id = 'SNOMED'
          AND a.min_levels_of_separation BETWEEN 1 AND ?
        GROUP BY anc.concept_code
        ORDER BY hops ASC, anc.concept_code ASC
        """
        try:
            rows = self.conn.sql(query, params=[str(snomed_code), max_hops]).fetchall()
        except Exception:
            rows = []

        result = [(str(c), int(h)) for c, h in rows if str(c) not in HIERARCHY_STOP_CODES]
        self._cache[key] = result
        return result


class Neo4jHierarchy:
    """SNOMED ancestors via KG1's IS_A edges, for once the graph is populated.

    Same interface as DuckDBHierarchy so it is a constructor swap. Kept
    because the OMOP closure only expresses the SNOMED hierarchy -- traversals
    that need to mix IS_A with guideline edges require the unified graph.
    """

    def __init__(self, driver):
        self.driver = driver
        self._cache = {}

    def ancestors(self, snomed_code: str, max_hops: int = MAX_HIERARCHY_HOPS) -> list:
        key = (snomed_code, max_hops)
        if key in self._cache:
            return self._cache[key]
        cypher = f"""
        MATCH path = (c:Concept {{snomed_code: $code}})-[:IS_A*1..{max_hops}]->(anc:Concept)
        WHERE NOT anc.snomed_code IN $stop
        RETURN anc.snomed_code AS code, min(length(path)) AS hops
        ORDER BY hops ASC, code ASC
        """
        try:
            with self.driver.session() as session:
                rows = session.run(cypher, code=str(snomed_code),
                                   stop=list(HIERARCHY_STOP_CODES)).data()
            result = [(str(r["code"]), int(r["hops"])) for r in rows]
        except Exception:
            result = []
        self._cache[key] = result
        return result


class NullHierarchy:
    """No-op provider, so retrieval runs (Channels A/C/D only) before the
    vocabulary is imported. Reports itself so a caller can tell "hierarchy
    found nothing" from "hierarchy was never consulted" -- a distinction that
    matters when interpreting a coverage measurement."""

    available = False

    def ancestors(self, snomed_code: str, max_hops: int = MAX_HIERARCHY_HOPS) -> list:
        return []


# ==========================================================================
# Vocabulary retriever (KG2 / DuckDB)
# ==========================================================================

class VocabularyRetriever:
    """OMOP lookups that support retrieval but are not the graph itself."""

    def __init__(self, conn):
        self.conn = conn
        self._fsn_cache = {}
        self._xwalk_cache = {}
        self._fsn_by_code = {}
        self._syn_by_code = {}

    def snomed_code_for_concept(self, omop_concept_id):
        """SNOMED code for an OMOP concept, via crosswalk for non-SNOMED vocabs.

        For SNOMED-vocabulary concepts, concept_code IS the SNOMED code, so no
        crosswalk is needed. Medications are the case that needs one: Stage 2b
        normalises them against RxNorm, which carries no SNOMED code, so
        without this every Medication entity would be structurally unable to
        reach guideline evidence.

        UNVERIFIED against real data: code/data/athena_omop/ is still a
        .gitkeep, so whether 'Maps to' actually links RxNorm to SNOMED in your
        Athena download has not been confirmed. Returns None on a miss and the
        caller degrades to the text-only path -- measure the real hit rate once
        scripts/import_athena.py has run rather than assuming it.
        """
        if omop_concept_id is None:
            return None
        if omop_concept_id in self._xwalk_cache:
            return self._xwalk_cache[omop_concept_id]

        code = None
        try:
            row = self.conn.sql(
                "SELECT concept_code, vocabulary_id FROM athena_concept WHERE concept_id = ?",
                params=[omop_concept_id],
            ).fetchone()
            if row and row[1] == "SNOMED":
                code = str(row[0])
            else:
                xrow = self.conn.sql("""
                    SELECT c2.concept_code
                    FROM athena_concept_relationship r
                    JOIN athena_concept c2 ON c2.concept_id = r.concept_id_2
                    WHERE r.concept_id_1 = ?
                      AND c2.vocabulary_id = 'SNOMED'
                      -- 'Mapped from', NOT 'Maps to'. OMOP direction convention:
                      -- RxNorm is the STANDARD vocabulary for drugs, so a SNOMED
                      -- drug concept 'Maps to' RxNorm and the reverse edge is
                      -- stored as RxNorm 'Mapped from' SNOMED. Querying from the
                      -- RxNorm side (which is what Stage 2b hands us for a
                      -- Medication), 'Maps to' matches almost nothing.
                      -- Measured on the EC2 vocabulary 2026-08-08, RxNorm->SNOMED:
                      --   Mapped from        199,311
                      --   RxNorm - SNOMED eq  22,800
                      --   Value mapped from    1,348
                      --   Maps to                  0  (absent — hence this fix)
                      -- The original list had exactly this backwards and would
                      -- have silently reduced Medication crosswalk coverage by
                      -- ~90% while still appearing to work.
                      AND r.relationship_id IN ('Mapped from', 'RxNorm - SNOMED eq',
                                                'Value mapped from', 'Maps to')
                      AND r.invalid_reason IS NULL
                    ORDER BY c2.concept_id ASC
                    LIMIT 1
                """, params=[omop_concept_id]).fetchone()
                if xrow:
                    code = str(xrow[0])
        except Exception:
            code = None

        self._xwalk_cache[omop_concept_id] = code
        return code

    def fsn_for_snomed_code(self, snomed_code):
        """Concept name for a SNOMED code. Cached; None if not found.

        Needed by Channel B's guard. See channel_b_hierarchy() for why the
        ANCESTOR's name is the right thing to guard against rather than the
        entity's.
        """
        if not snomed_code:
            return None
        if snomed_code in self._fsn_by_code:
            return self._fsn_by_code[snomed_code]
        try:
            row = self.conn.sql("""
                SELECT concept_name FROM athena_concept
                WHERE concept_code = ? AND vocabulary_id = 'SNOMED'
                ORDER BY concept_id ASC LIMIT 1
            """, params=[str(snomed_code)]).fetchone()
            name = row[0] if row else None
        except Exception:
            name = None
        self._fsn_by_code[snomed_code] = name
        return name

    def synonyms_for_snomed_code(self, snomed_code) -> list:
        """All registered synonyms for a SNOMED code, from athena_concept_synonym.

        Capped at 25. Some SNOMED concepts carry dozens of synonyms and the
        guard only needs enough alternative phrasings to recognise a match --
        beyond that it is cost with no discrimination gain. Ordered by length
        so short clinical forms ("WBC", "COPD"), which are what guideline
        authors actually write, are the ones retained.
        """
        if not snomed_code:
            return []
        if snomed_code in self._syn_by_code:
            return self._syn_by_code[snomed_code]
        try:
            rows = self.conn.sql("""
                SELECT DISTINCT s.concept_synonym_name
                FROM athena_concept c
                JOIN athena_concept_synonym s ON s.concept_id = c.concept_id
                WHERE c.concept_code = ? AND c.vocabulary_id = 'SNOMED'
                ORDER BY length(s.concept_synonym_name) ASC
                LIMIT 25
            """, params=[str(snomed_code)]).fetchall()
            syns = [r[0] for r in rows if r[0]]
        except Exception:
            syns = []
        self._syn_by_code[snomed_code] = syns
        return syns

    def concept_context(self, omop_concept_id) -> dict:
        """Channel C: FSN, domain and immediate parents.

        Retrieved for EVERY record regardless of guideline coverage -- this is
        what serves the ~87.5% of entities with no guideline evidence, so they
        still receive real symbolic grounding (ontological rather than
        guideline-based) instead of an empty prompt.
        """
        if omop_concept_id is None:
            return {}
        if omop_concept_id in self._fsn_cache:
            return self._fsn_cache[omop_concept_id]

        ctx = {}
        try:
            row = self.conn.sql("""
                SELECT concept_name, domain_id, vocabulary_id, concept_class_id, concept_code
                FROM athena_concept WHERE concept_id = ?
            """, params=[omop_concept_id]).fetchone()
            if row:
                ctx = {
                    "fsn": row[0], "domain_id": row[1], "vocabulary_id": row[2],
                    "concept_class_id": row[3], "concept_code": str(row[4]),
                }
            parents = self.conn.sql("""
                SELECT p.concept_name, p.concept_code
                FROM athena_concept_ancestor a
                JOIN athena_concept p ON p.concept_id = a.ancestor_concept_id
                WHERE a.descendant_concept_id = ? AND a.min_levels_of_separation = 1
                ORDER BY p.concept_id ASC
                LIMIT 3
            """, params=[omop_concept_id]).fetchall()
            ctx["parents"] = [{"name": n, "code": str(c)} for n, c in parents]
        except Exception:
            ctx = ctx or {}
            ctx.setdefault("parents", [])

        self._fsn_cache[omop_concept_id] = ctx
        return ctx


# ==========================================================================
# The retriever
# ==========================================================================

class GroundingRetriever:
    """Runs the four channels and returns ranked, capped evidence."""

    def __init__(self, guideline_index: GuidelineIndex, vocabulary: VocabularyRetriever,
                 hierarchy=None):
        self.index = guideline_index
        self.vocab = vocabulary
        self.hierarchy = hierarchy or NullHierarchy()

    # ---------------------------------------------------------------- rules

    def _rules_from_nodes(self, nodes, entity_name, concept_fsn, base_confidence,
                          channel, matched_code=None, hops=None, gliner_label=None,
                          omop_domain=None, stats=None, concept_synonyms=None) -> list:
        """`stats` accumulates WHY candidate evidence was discarded.

        Every `continue` below silently removed a rule that a code or name
        match had genuinely found. Without accounting, Stage 3 and any later
        audit see an empty or short evidence list with no way to tell
        "the KG contains nothing about this concept" from "the KG contained
        four rules and the guard rejected all of them". Those need opposite
        interpretations -- the second says the concept IS covered by the
        guidelines but the code assertion was untrustworthy, which is itself
        a finding. Counts are recorded per reason and carried into the
        decision artifact.
        """
        stats = stats if stats is not None else {}
        out = []
        for node in nodes:
            if is_boilerplate(node["name"]):
                stats["suppressed_boilerplate_node"] = stats.get("suppressed_boilerplate_node", 0) + 1
                continue
            multiplier, agreement = name_agreement_guard(
                entity_name, concept_fsn, node["name"], concept_synonyms=concept_synonyms)
            if agreement == "reject":
                stats["suppressed_name_disagreement"] = stats.get("suppressed_name_disagreement", 0) + 1
                stats.setdefault("suppressed_node_names", []).append(node["name"])
                # Keep the rules themselves, not merely a count. Stage 3 can
                # ask to see suppressed evidence (see mollm_ensemble's
                # expand_evidence), and that request has to be answerable from
                # the stored artifact -- re-running retrieval later would not
                # reproduce it faithfully, because the guard bands or the KG
                # may have changed in between.
                for rule in self.index.rules_touching(node["uid"]):
                    other_uid = (rule["target_uid"] if rule["direction"] == "outgoing"
                                 else rule["source_uid"])
                    other = self.index.nodes_by_uid.get(other_uid, {})
                    stats.setdefault("suppressed_rules", []).append({
                        "rule_id": f"{node['file']}::{rule['predicate']}::"
                                   f"{node['uid'][1]}->{other_uid[1]}",
                        "predicate": rule["predicate"],
                        "source_name": node["name"],
                        "target_name": other.get("name"),
                        "rationale": rule["rationale"],
                        "citation": rule["citation"],
                        "citation_verbatim_excerpt": rule["citation_verbatim_excerpt"],
                        "citation_type": rule["citation_type"],
                        "source_document": node["source_document"],
                        "suppression_reason": "name_disagreement",
                        "matched_node_name": node["name"],
                        "matched_node_type": node.get("node_type"),
                        "match_channel": channel,
                    })
                continue
            type_mult, type_status = combined_type_agreement(
                gliner_label, omop_domain, node.get("node_type"))
            multiplier *= type_mult
            # The 108 nodes flagged same_snomed_type_mismatch_not_merged are
            # exactly the code assertions the cleaning pass could not verify,
            # so they are not trusted for code-derived channels. They remain
            # reachable by name (Channel D), where the match does not depend on
            # the disputed code at all.
            if channel in ("A", "B") and node.get("quality_flag") == "same_snomed_type_mismatch_not_merged":
                stats["suppressed_unverified_code_assertion"] = stats.get(
                    "suppressed_unverified_code_assertion", 0) + 1
                continue

            for rule in self.index.rules_touching(node["uid"]):
                other_uid = rule["target_uid"] if rule["direction"] == "outgoing" else rule["source_uid"]
                other = self.index.nodes_by_uid.get(other_uid, {})
                if is_boilerplate(other.get("name", "")):
                    stats["suppressed_boilerplate_target"] = stats.get(
                        "suppressed_boilerplate_target", 0) + 1
                    continue
                out.append({
                    "rule_id": f"{node['file']}::{rule['predicate']}::{node['uid'][1]}->{other_uid[1]}",
                    "predicate": rule["predicate"],
                    "direction": rule["direction"],
                    "source_name": node["name"] if rule["direction"] == "outgoing" else other.get("name"),
                    "target_name": other.get("name") if rule["direction"] == "outgoing" else node["name"],
                    "rationale": rule["rationale"],
                    "citation": rule["citation"],
                    "citation_verbatim_excerpt": rule["citation_verbatim_excerpt"],
                    "citation_type": rule["citation_type"],
                    "source_document": node["source_document"],
                    "section_title": node["section_title"],
                    "match_channel": channel,
                    "match_confidence": round(base_confidence * multiplier, 4),
                    "name_agreement": agreement,
                    "matched_via_code": matched_code,
                    "hierarchy_hops": hops,
                    "matched_node_name": node["name"],
                    "matched_node_type": node.get("node_type"),
                    "other_node_type": other.get("node_type"),
                    "type_agreement": type_status,
                })
        return out

    def channel_a_direct_code(self, snomed_code, entity_name, concept_fsn,
                              gliner_label=None, omop_domain=None, stats=None) -> list:
        if not snomed_code:
            return []
        return self._rules_from_nodes(
            self.index.nodes_for_code(snomed_code), entity_name, concept_fsn,
            base_confidence=1.0, channel="A", matched_code=snomed_code,
            gliner_label=gliner_label, omop_domain=omop_domain, stats=stats,
            concept_synonyms=self.vocab.synonyms_for_snomed_code(snomed_code),
        )

    def channel_b_hierarchy(self, snomed_code, entity_name, concept_fsn,
                            gliner_label=None, omop_domain=None, stats=None) -> list:
        if not snomed_code:
            return []
        out = []
        for anc_code, hops in self.hierarchy.ancestors(snomed_code, MAX_HIERARCHY_HOPS):
            nodes = self.index.nodes_for_code(anc_code)
            if not nodes:
                continue
            # GUARD AGAINST THE ANCESTOR'S NAME, NOT THE ENTITY'S.
            #
            # This was wrong until 2026-08-08 and the error was large. The
            # guard exists to catch an untrustworthy CODE ASSERTION -- a
            # guideline node claiming a code whose concept it does not actually
            # describe (the 24484000 "Severe" / NSTEMI-vs-STEMI class). For a
            # direct match that means comparing the node's name against the
            # entity's concept, because they are supposed to be the same thing.
            #
            # For a HIERARCHY match they are NOT supposed to be the same thing.
            # The whole point of Channel B is to reach a rule stated more
            # generally than the entity: `Stage 2 AKI` matching a node named
            # `Acute Kidney Injury` is a correct generalisation, not a
            # collision. Guarding it against the entity's own name rejected
            # precisely the generalisations the channel exists to provide --
            # measured over the gold set, hierarchy contributed +17.93pp of raw
            # coverage but only +0.64pp after that mistaken guard, i.e. ~96% of
            # its value was being discarded.
            #
            # The entity->ancestor link needs no guarding at all: it comes from
            # SNOMED's own IS_A hierarchy and is already deterministic. What
            # still needs checking is node->ancestor, so the ancestor's own
            # name is what the node is compared against.
            anc_fsn = self.vocab.fsn_for_snomed_code(anc_code) or ""
            anc_syns = self.vocab.synonyms_for_snomed_code(anc_code)
            out.extend(self._rules_from_nodes(
                nodes, anc_fsn, anc_fsn,
                base_confidence=HIERARCHY_DECAY ** hops,
                channel="B", matched_code=anc_code, hops=hops,
                gliner_label=gliner_label, omop_domain=omop_domain, stats=stats,
                concept_synonyms=anc_syns,
            ))
        return out

    def channel_d_name(self, entity_text, expanded_text, gliner_label=None,
                       omop_domain=None, stats=None) -> list:
        """Exact normalised name match against guideline nodes.

        Constrained to LOW-tier records by retrieve() rather than run by
        default: measured standalone contribution is 0.78% of gold annotations,
        which does not justify running it for every entity. It exists to reach
        the 982 ungrounded nodes that are structurally unreachable by code, and
        as the fallback when a Medication has no RxNorm->SNOMED crosswalk.

        Exact (normalised) matching, not SapBERT, in this implementation:
        ungrounded node names average 5.3 words ("Absence of volume overload or
        alternative diagnosis") against 2 for grounded ones, so compound
        guideline phrasings do not embed close to short clinical spans, and a
        vector pass here would add cost and a calibration burden for very
        little reach. The SapBERT variant is a deliberate later step, not an
        oversight.
        """
        out = []
        for text in filter(None, {(entity_text or "").strip().lower(),
                                  (expanded_text or "").strip().lower()}):
            nodes = self.index.nodes_for_name(text)
            out.extend(self._rules_from_nodes(
                nodes, text, text, base_confidence=0.8, channel="D",
                gliner_label=gliner_label, omop_domain=omop_domain, stats=stats,
            ))
        return out

    # ------------------------------------------------------------- ranking

    @staticmethod
    def rank_and_cap(rules: list, gliner_label: str, cap: int = MAX_RULES_PER_RECORD) -> list:
        """Deduplicate, sort, cap. Sort order is deliberate and ordered:

        1. match_confidence  -- how well the rule's node matches THIS entity.
           Primary, because a perfectly-cited rule about the wrong concept is
           worse than useless.
        2. citation_type     -- tiebreaker only. Preferring checkable evidence
           is right, but not at the cost of relevance.
        3. predicate affinity -- final tiebreaker.
        """
        best = {}
        for r in rules:
            key = (r["predicate"], r["source_name"], r["target_name"])
            if key not in best or r["match_confidence"] > best[key]["match_confidence"]:
                best[key] = r

        affinity = PREDICATE_AFFINITY.get(gliner_label, set())
        ranked = sorted(
            best.values(),
            key=lambda r: (
                -r["match_confidence"],
                CITATION_TYPE_RANK.get(r["citation_type"], 4),
                # Type agreement ranks BELOW citation quality but ABOVE
                # predicate affinity: a node whose @type matches the entity's
                # label is more likely to be about the same thing, but the
                # corpus's types are too inconsistent to outrank whether the
                # evidence is checkable at all.
                {"agree": 0, "neutral": 1, "mismatch": 2,
                 "both_mismatch": 3}.get(r.get("type_agreement"), 1),
                0 if r["predicate"] in affinity else 1,
                r["rule_id"],
            ),
        )

        # A pointer_unverifiable rule as the ONLY evidence invites an
        # uncheckable citation, so it is dropped in that case -- but kept when
        # checkable rules accompany it, where it adds context without being
        # the sole thing the model can cite.
        capped = ranked[:cap]
        if len(capped) == 1 and capped[0]["citation_type"] not in CITABLE_TYPES:
            return [], ranked, {"dropped_uncitable_sole_rule": 1,
                                "deduped": len(rules) - len(ranked), "dropped_by_cap": 0}
        return capped, ranked, {"dropped_by_cap": max(0, len(ranked) - len(capped)),
                                "deduped": len(rules) - len(ranked),
                                "dropped_uncitable_sole_rule": 0}

    # ------------------------------------------------------------ entrypoint

    def retrieve(self, record: dict) -> dict:
        """Full retrieval for one ValidationRecord.

        ASSERTION GATING (design doc S6.4b): guideline rules describe what to do
        when a finding IS PRESENT. Applying them to a negated or family-history
        mention is a category error, and with ~14.2% of spans negated it is a
        high-frequency one. So retrieval is skipped symbolically here, BEFORE
        the model is involved, rather than being left for the LLM to notice.
        """
        candidates = record.get("candidates") or []
        primary = candidates[0] if candidates else {}
        entity_text = record.get("expanded_text") or record.get("original_text") or ""
        gliner_label = record.get("gliner_label")
        omop_domain = primary.get("domain_id")
        stats = {}

        concept_ctx = self.vocab.concept_context(primary.get("omop_concept_id"))
        concept_fsn = concept_ctx.get("fsn", "")

        assertion = record.get("assertion_status", "PRESENT")
        experiencer = record.get("experiencer", "PATIENT")

        result = {
            "concept_context": concept_ctx,
            "candidate_contexts": [
                self.vocab.concept_context(c.get("omop_concept_id")) for c in candidates
            ] if record.get("confidence_tier_in") == "LOW" else [],
            "rules": [],
            "channels_run": [],
            "snomed_code": None,
            "retrieval_skipped_reason": None,
            "hierarchy_available": getattr(self.hierarchy, "available", True),
            # Full accounting of evidence that was FOUND and then discarded,
            # with reasons. Lives in the provenance artifact rather than the
            # prompt: DuckDB JSON has no size limit, the 8,192-token prompt
            # does. Stage 3 gets a one-line summary (see mollm_ensemble's
            # _format_evidence); an auditor gets the whole picture.
            "suppression": {},
            "channels_skipped": [],
        }

        if assertion == "ABSENT" or experiencer != "PATIENT":
            result["retrieval_skipped_reason"] = (
                f"assertion_status={assertion}, experiencer={experiencer}: "
                "no asserted patient finding for a guideline to bear on"
            )
            result["channels_run"] = ["C"]
            return result

        snomed_code = self.vocab.snomed_code_for_concept(primary.get("omop_concept_id"))
        result["snomed_code"] = snomed_code

        pooled = []
        pooled.extend(self.channel_a_direct_code(snomed_code, entity_text, concept_fsn,
                                                 gliner_label=gliner_label,
                                                 omop_domain=omop_domain, stats=stats))
        result["channels_run"].extend(["A", "C"])

        pooled.extend(self.channel_b_hierarchy(snomed_code, entity_text, concept_fsn,
                                               gliner_label=gliner_label,
                                                 omop_domain=omop_domain, stats=stats))
        if getattr(self.hierarchy, "available", True):
            result["channels_run"].append("B")

        if not getattr(self.hierarchy, "available", True):
            result["channels_skipped"].append(
                {"channel": "B", "reason": "no hierarchy provider configured"})

        if record.get("confidence_tier_in") == "LOW" and not pooled:
            pooled.extend(self.channel_d_name(record.get("original_text"), entity_text,
                                              gliner_label=gliner_label,
                                                 omop_domain=omop_domain, stats=stats))
            result["channels_run"].append("D")

        else:
            result["channels_skipped"].append({
                "channel": "D",
                "reason": ("not run: tier is HIGH" if record.get("confidence_tier_in") != "LOW"
                           else "not run: channels A/B already returned evidence"),
            })

        result["rules"], all_ranked, cap_stats = self.rank_and_cap(pooled, gliner_label)
        stats.update(cap_stats)
        result["suppression"] = stats
        # The COMPLETE ranked set, not just the five shown. Persisted so a
        # follow-up request for more evidence is served from the stored record
        # rather than by re-running retrieval, and so an auditor can see every
        # rule that was available at decision time. Bounded at 100 purely to
        # stop a pathological hub concept writing an unbounded blob; the cap is
        # recorded when it bites so the truncation is never silent.
        result["all_ranked_rules"] = all_ranked[:100]
        if len(all_ranked) > 100:
            result["all_ranked_rules_truncated_from"] = len(all_ranked)
        result["rules_pooled_before_cap"] = len(pooled)
        return result
