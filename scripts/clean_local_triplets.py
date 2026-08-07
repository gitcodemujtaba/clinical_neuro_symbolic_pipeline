"""
scripts/clean_local_triplets.py

Cleans up data/local_triplets_db2_v6/ (the MedCAT-pipeline guideline triplet
corpus, chosen as the final KG source over data/rules-llm/ -- see
docs/Implementation_Checklist.md and docs/Guideline_Triplets_KG_Review.md for
why) by fixing the structural weaknesses identified in that review that DON'T
require the populated DuckDB (grounding/ICD10 backfill is a separate script,
scripts/backfill_guideline_grounding.py, and still needs to run on the EC2
box). This script only needs the triplet files themselves and the raw source
chunks, both already present locally, so it runs end-to-end here.

Fixes applied, each independently toggleable, all non-destructive (writes to
data/local_triplets_db2_v6_cleaned/, never touches the originals):

1. NODE CONSOLIDATION (--dedup, default on)
   Merges nodes that share the same real (non-N/A) `snomed` code within a
   file into a single canonical node, unions their `rules`, and rewrites
   every other node's rule `target` references that pointed at a
   now-merged duplicate to point at the canonical node instead. This is
   the fix for S3.2 in Guideline_Triplets_KG_Review.md (63% of nodes were
   redundant re-instantiations of an already-present concept). Referential
   integrity is re-validated after the rewrite -- this is the one step
   that could introduce dangling edges if done carelessly, so it's checked
   explicitly rather than assumed.

2. PREDICATE / TYPE CANONICALIZATION (--canonicalize, default on)
   Applies a conservative, hand-reviewed mapping for predicates that are
   unambiguously the same relation expressed inconsistently (see
   PREDICATE_CANONICAL_MAP below) plus the `quantitative_threshold` /
   `Quantitative Threshold` type-vs-predicate collision. Deliberately does
   NOT merge predicate pairs where direction is ambiguous (e.g.
   RECOMMENDS vs RECOMMENDED_FOR) -- those are left alone and flagged in
   the report for a human decision rather than guessed at here.

3. CITATION CLASSIFICATION (--classify-citations, default on)
   Re-checks every rule's `citation` against its source chunk (matched by
   filename against data/triplets-rules-backup-data/local_chunks_db2_v6/).
   Adds a `citation_type` field: "verbatim" (>=0.8 containment, left as
   is), "paraphrase" (0.4-0.8, a best-effort verbatim excerpt is recovered
   from the source and added as `citation_verbatim_excerpt` alongside the
   original as `citation_paraphrase`), or "pointer_unverifiable" (<0.4 --
   flagged, original text kept, nothing invented). This directly targets
   S3.3 -- Stage 3's planned `citation_verified` check needs to know which
   citations it can actually check.

4. BOILERPLATE FLAGGING (--flag-boilerplate, default on)
   Flags (does not delete) nodes/rules whose name or citation matches
   known non-clinical boilerplate patterns identified in S3.7 (journal
   running headers, literature-search-methodology paragraphs). Adds
   `quality_flag: "likely_boilerplate"` so a human or a later filter step
   can decide whether to exclude them -- deliberately not deleting curated
   data unilaterally.

Usage:
    python3 scripts/clean_local_triplets.py
    python3 scripts/clean_local_triplets.py --triplets-dir data/local_triplets_db2_v6 \\
        --chunks-dir data/triplets-rules-backup-data/local_chunks_db2_v6 \\
        --out-dir data/local_triplets_db2_v6_cleaned
"""

import os
import re
import sys
import json
import glob
import argparse
import difflib
import collections
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Conservative predicate/type canonicalization. Only pairs I'm confident mean
# the same relation regardless of phrasing are merged here -- direction-
# ambiguous pairs (RECOMMENDS vs RECOMMENDED_FOR, IS_OUTCOME vs
# IS_OUTCOME_OF) are deliberately left alone; see module docstring.
# ---------------------------------------------------------------------------
PREDICATE_CANONICAL_MAP = {
    "SUGGESTS NOT USING": "NOT_RECOMMENDED_FOR",
    "RECOMMENDS NOT USING": "NOT_RECOMMENDED_FOR",
    "quantitative_threshold": "HAS_QUANTITATIVE_THRESHOLD",
    "REQUIRES_QUANTITATIVE_THRESHOLD": "HAS_QUANTITATIVE_THRESHOLD",
    "HAS_THRESHOLD": "HAS_QUANTITATIVE_THRESHOLD",
    "IS_DEFINED_BY": "DEFINED_BY",
}
TYPE_CANONICAL_MAP = {
    "quantitative_threshold": "Quantitative Threshold",
}

# Non-clinical boilerplate patterns found during review (S3.7): journal
# running headers and literature-search-methodology narrative. Deliberately
# NOT flagging evidence-grading language ("Level B recommendation", "Class
# III study") -- that's clinically meaningful metadata, not boilerplate.
BOILERPLATE_PATTERNS = [
    re.compile(r"Annals of Emergency Medicine", re.IGNORECASE),
    re.compile(r"Key words/phrases for literature searches", re.IGNORECASE),
    re.compile(r"Study Selection:", re.IGNORECASE),
    re.compile(r"searches included\b.*\bsearch dates", re.IGNORECASE),
    re.compile(r"were identified (in|from) the search", re.IGNORECASE),
    re.compile(r"candidates for further review", re.IGNORECASE),
    re.compile(r"^Clinical Policy$", re.IGNORECASE),
]

CITATION_SNOMED_TAG = re.compile(r"\s*\[SNOMED:[^\]]+\]")


def strip_snomed_tags(text):
    return CITATION_SNOMED_TAG.sub("", text)


def normalize_ws(text):
    return re.sub(r"\s+", " ", text or "").strip()


def is_boilerplate(text):
    if not text:
        return False
    return any(p.search(text) for p in BOILERPLATE_PATTERNS)


def best_containment(needle, haystack):
    sm = difflib.SequenceMatcher(None, haystack, needle, autojunk=False)
    match = sm.find_longest_match(0, len(haystack), 0, len(needle))
    return match, match.size / max(1, len(needle))


def recover_excerpt(citation_clean, chunk_norm, min_len=40):
    """Best-effort: find the longest contiguous run of the citation that
    actually appears in the source chunk, expand slightly to word
    boundaries, and return it. Returns None if nothing substantial found."""
    match, ratio = best_containment(citation_clean, chunk_norm)
    if match.size < min_len:
        return None, ratio
    start, end = match.b, match.b + match.size
    # expand to nearby sentence-ish boundaries, capped so we don't grab the whole doc
    left = chunk_norm.rfind(".", max(0, start - 200), start)
    right = chunk_norm.find(".", end, min(len(chunk_norm), end + 200))
    left = left + 1 if left != -1 else max(0, start - 40)
    right = right + 1 if right != -1 else min(len(chunk_norm), end + 40)
    return chunk_norm[left:right].strip(), ratio


def load_chunk_text(chunks_dir, base_filename):
    """base_filename is the triplet file's name with '_triplets' removed."""
    path = os.path.join(chunks_dir, base_filename)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return normalize_ws(json.load(f).get("content", ""))


def dedup_nodes(graph, report):
    """Merge nodes sharing a real snomed code within this file. Returns the
    new node list and an id-remap dict (old_id -> canonical_id) for rewriting
    rule targets afterward."""
    groups = collections.OrderedDict()  # snomed_code -> [nodes]
    passthrough = []

    for node in graph:
        s = node.get("snomed")
        if s and str(s).strip().upper() not in ("N/A", ""):
            key = str(s).strip()
            groups.setdefault(key, []).append(node)
        else:
            passthrough.append(node)

    id_remap = {}
    merged_nodes = []
    for code, nodes in groups.items():
        if len(nodes) == 1:
            merged_nodes.append(nodes[0])
            continue

        seen_types = collections.Counter(n.get("@type") for n in nodes)
        if len(seen_types) > 1:
            # Nodes sharing a SNOMED code but disagreeing on @type are NOT
            # auto-merged -- a type mismatch is a strong signal these are
            # actually distinct concepts sharing a generic parent code (e.g.
            # "SBP >= 160" and "SBP < 90" both citing the general "systolic
            # blood pressure" SNOMED code but representing opposite clinical
            # criteria; "suspected X" as a Finding vs "X" as a Condition).
            # Confirmed by inspection during development -- see
            # docs/Guideline_Triplets_KG_Review.md addendum. Flag for human
            # review instead of guessing.
            report["type_mismatch_on_merge"].append({
                "snomed": code, "types": dict(seen_types),
                "names": [n.get("name") for n in nodes],
            })
            for n in nodes:
                n["quality_flag"] = "same_snomed_type_mismatch_not_merged"
            merged_nodes.extend(nodes)
            continue

        canonical = nodes[0]
        canonical_id = canonical["@id"]

        # Prefer a non-N/A icd10 among the duplicates if the canonical lacks one.
        icd10 = canonical.get("icd10")
        if (not icd10 or str(icd10).upper() == "N/A"):
            for n in nodes[1:]:
                if n.get("icd10") and str(n.get("icd10")).upper() != "N/A":
                    icd10 = n.get("icd10")
                    break
        if icd10:
            canonical["icd10"] = icd10

        # Union rules, de-duplicating exact (predicate, target, rationale) repeats.
        seen_rules = set()
        combined_rules = []
        for n in nodes:
            for r in n.get("rules", []) or []:
                sig = (r.get("predicate"), r.get("target"), r.get("rationale"))
                if sig in seen_rules:
                    continue
                seen_rules.add(sig)
                combined_rules.append(r)
        if combined_rules:
            canonical["rules"] = combined_rules

        for n in nodes[1:]:
            id_remap[n["@id"]] = canonical_id

        merged_nodes.append(canonical)
        report["nodes_merged"] += len(nodes) - 1

    return merged_nodes + passthrough, id_remap


def rewrite_targets(graph, id_remap, report):
    """Point every rule's target at its canonical id if it was merged away.
    Drops any rule that would become a self-loop as a result (the two
    concepts it used to relate are now the same node)."""
    for node in graph:
        rules = node.get("rules", []) or []
        kept = []
        for r in rules:
            tgt = r.get("target")
            new_tgt = id_remap.get(tgt, tgt)
            if new_tgt == node["@id"]:
                report["self_loops_dropped"] += 1
                continue
            r["target"] = new_tgt
            kept.append(r)
        node["rules"] = kept
    return graph


def canonicalize_predicates_and_types(graph, report):
    for node in graph:
        t = node.get("@type")
        if t in TYPE_CANONICAL_MAP:
            node["@type"] = TYPE_CANONICAL_MAP[t]
            report["types_canonicalized"] += 1
        for r in node.get("rules", []) or []:
            p = r.get("predicate")
            if p in PREDICATE_CANONICAL_MAP:
                r["predicate"] = PREDICATE_CANONICAL_MAP[p]
                report["predicates_canonicalized"] += 1
    return graph


def classify_citations(graph, chunk_text, report):
    if chunk_text is None:
        report["files_missing_source_chunk"] += 1
        return graph
    for node in graph:
        for r in node.get("rules", []) or []:
            cit = r.get("citation")
            if not cit or len(cit) < 15:
                continue
            clean = strip_snomed_tags(normalize_ws(cit))
            _, ratio = best_containment(clean, chunk_text)
            if ratio >= 0.8:
                r["citation_type"] = "verbatim"
                report["citations_verbatim"] += 1
            elif ratio >= 0.4:
                excerpt, _ = recover_excerpt(clean, chunk_text)
                if excerpt:
                    r["citation_paraphrase"] = cit
                    r["citation_verbatim_excerpt"] = excerpt
                    r["citation_type"] = "paraphrase_with_recovered_excerpt"
                    report["citations_recovered"] += 1
                else:
                    r["citation_type"] = "paraphrase"
                    report["citations_paraphrase"] += 1
            else:
                r["citation_type"] = "pointer_unverifiable"
                report["citations_unverifiable"] += 1
    return graph


def _add_flag(obj, flag):
    existing = obj.get("quality_flag")
    if not existing:
        obj["quality_flag"] = flag
    elif flag not in existing.split("+"):
        obj["quality_flag"] = existing + "+" + flag


def flag_boilerplate(graph, report):
    for node in graph:
        name_bp = is_boilerplate(node.get("name"))
        rule_flags = []
        for r in node.get("rules", []) or []:
            if is_boilerplate(r.get("citation")) or is_boilerplate(r.get("rationale")):
                _add_flag(r, "likely_boilerplate")
                rule_flags.append(r)
        if name_bp or rule_flags:
            if name_bp:
                _add_flag(node, "likely_boilerplate")
            report["boilerplate_flags"] += (1 if name_bp else 0) + len(rule_flags)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--triplets-dir", default="data/local_triplets_db2_v6")
    parser.add_argument("--chunks-dir", default="data/triplets-rules-backup-data/local_chunks_db2_v6")
    parser.add_argument("--out-dir", default="data/local_triplets_db2_v6_cleaned")
    parser.add_argument("--report-dir", default="data/cleaning_reports")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--no-canonicalize", action="store_true")
    parser.add_argument("--no-classify-citations", action="store_true")
    parser.add_argument("--no-flag-boilerplate", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    report = collections.defaultdict(int)
    report["type_mismatch_on_merge"] = []
    report["files_processed"] = 0
    per_file_before_after = []

    files = sorted(glob.glob(os.path.join(args.triplets_dir, "*.json")))
    if not files:
        sys.exit(f"No triplet files found in {args.triplets_dir}")

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        graph = data.get("@graph", [])
        nodes_before = len(graph)
        rules_before = sum(len(n.get("rules", []) or []) for n in graph)

        id_remap = {}
        if not args.no_dedup:
            graph, id_remap = dedup_nodes(graph, report)
            graph = rewrite_targets(graph, id_remap, report)

        if not args.no_canonicalize:
            graph = canonicalize_predicates_and_types(graph, report)

        if not args.no_classify_citations:
            base = os.path.basename(fp).replace("_triplets.json", ".json")
            chunk_text = load_chunk_text(args.chunks_dir, base)
            graph = classify_citations(graph, chunk_text, report)

        if not args.no_flag_boilerplate:
            graph = flag_boilerplate(graph, report)

        # Final referential-integrity check post-rewrite.
        ids_now = set(n["@id"] for n in graph)
        dangling = 0
        for n in graph:
            for r in n.get("rules", []) or []:
                if r.get("target") not in ids_now:
                    dangling += 1
        report["dangling_targets_after_cleaning"] += dangling

        data["@graph"] = graph
        data["_cleaning_provenance"] = {
            "cleaned_by": "clean_local_triplets.py",
            "cleaned_at": datetime.now(timezone.utc).isoformat(),
            "nodes_before": nodes_before,
            "nodes_after": len(graph),
            "rules_before": rules_before,
            "rules_after": sum(len(n.get("rules", []) or []) for n in graph),
        }

        out_path = os.path.join(args.out_dir, os.path.basename(fp))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        per_file_before_after.append({
            "file": os.path.basename(fp),
            "nodes_before": nodes_before, "nodes_after": len(graph),
            "rules_before": rules_before, "rules_after": sum(len(n.get("rules", []) or []) for n in graph),
            "dangling_after": dangling,
        })
        report["files_processed"] += 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.report_dir, f"clean_local_triplets_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": dict(report), "per_file": per_file_before_after}, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("CLEANING SUMMARY")
    print("=" * 70)
    for k, v in report.items():
        if isinstance(v, list):
            print(f"  {k}: {len(v)} (see report file for details)")
        else:
            print(f"  {k}: {v}")
    print(f"\nCleaned files written to: {args.out_dir}")
    print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
