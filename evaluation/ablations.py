"""
evaluation/ablations.py — Stage 3 (MoLLM) value-addition ablations.

Answers one question with five separate measurements, because "does MoLLM
help" is not one question: it bundles together (a) whether adding ANY
verification layer on top of Stage 2b helps, (b) whether the KG-grounded
evidence specifically is doing the work or just the ensemble+citation-guard
machinery would do the same with an empty graph, (c) whether two models beat
one, (d) whether the citation-verification guard specifically is earning its
keep, and (e) what the accuracy gain actually costs in human review volume.
Collapsing these into a single "with vs without Stage 3" number would answer
a question nobody is asking and could not support Objective 2's specific
claim (KG-grounded prompting), only a weaker one (LLM ensembles help).

THE FIVE ABLATIONS
  1. stage3_value_added()      -- Stage 2b baseline accuracy vs. Stage 3's
                                   AUTOMATICALLY-COMMITTED final accuracy
                                   (AUTO_VALIDATED / MOLLM_RESOLVED only --
                                   HITL_REQUIRED has no automatic answer to
                                   grade), plus how much of the corpus Stage 3
                                   could decide on its own vs. deferred.
  2. grounded_vs_ungrounded()  -- splits Stage 3 decisions by whether
                                   guideline retrieval found ANY rule, and
                                   compares citation-failure rate, HITL rate
                                   and accuracy between the two groups. This
                                   is the one that actually tests the
                                   "neuro-symbolic" claim rather than "LLM
                                   ensemble" -- see the WARNING below.
  3. ensemble_vs_single()      -- each model's own (ungated) verdict graded
                                   independently against gold, next to the
                                   two-model agreement-gated ensemble, on
                                   resolution-mode decisions only (the only
                                   mode with a gold label to grade against --
                                   same scoping cal_eval.py uses and for the
                                   same reason).
  4. citation_guard_value()    -- for every decision HITL'd specifically for
                                   queue_reason == "citation_verification_failed",
                                   replays mollm_ensemble.route() with
                                   citation_verified forced True (everything
                                   else identical) to ask "what would have
                                   happened without this guard", then grades
                                   the counterfactual answer. Reuses the real
                                   route() function rather than
                                   re-deriving its logic, so this cannot drift
                                   from what production actually does.
  5. error_catching_matrix()   -- the actual cost-benefit number: crossing
                                   {Stage 2b baseline was right/wrong} with
                                   {Stage 3 auto-admitted / deferred to HITL}
                                   gives catch rate (wrong answers correctly
                                   caught) against false-deflection rate
                                   (right answers needlessly sent to a
                                   human) -- the two numbers Objective 5's
                                   deflection-rate metric actually trades off.

WARNING WORTH READING BEFORE QUOTING ANY NUMBER FROM THIS SCRIPT.
As of 2026-08-11 (docs/Stage3_Open_Issues.md, Issue 1), guideline retrieval
returns zero rules for the large majority of entities -- only 41 high-
frequency guideline nodes were rule-less by design and even after backfill,
end-to-end coverage against normalized_entities has not been measured. That
means grounded_vs_ungrounded()'s "grounded" bucket may currently be small
enough that its numbers are noise, and any overall stage3_value_added()
number computed today is measuring mostly the ungrounded bucket -- i.e.
"ensemble + citation guard", not "KG-grounded reasoning". Report both
buckets' N alongside their numbers, always, and do not average over them.

SAMPLE SIZE. Same caveat as cal_eval.py and score_gold_recall.py: as of
2026-08-11 only a handful of notes have gone through Stage 3. Every number
below is for validating this script's methodology and re-running as the
corpus grows, not a production result.

GRADABILITY. Only resolution-mode decisions have a candidate list a SNOMED
code can be read off of. Contradiction / non_asserted_check decisions never
reassign the concept -- Stage 3 either endorses Stage 2b's existing pick
(routes it AUTO_VALIDATED/MOLLM_RESOLVED) or defers it (HITL_REQUIRED) -- so
their "final concept" is always Stage 2b's own pick, graded the same way
scripts/score_gold_recall.py grades it. This lets ablations 1 and 5 include
all three modes (there is always a baseline-vs-final comparison to make),
while ablations 2-4 note where they are scoped to resolution mode only.

Run:
  python3 evaluation/ablations.py
  python3 evaluation/ablations.py --note-ids 10000032-DS-21 --out ablation_report.json
"""

import argparse
import collections
import json
import os
import re
import sys

import duckdb

PROJECT_DIR = "/home/ec2-user/clinical_neuro_symbolic_pipeline_reorder"
DB_PATH = os.path.join(PROJECT_DIR, "db", "kg2_lexical_store.duckdb")

sys.path.insert(0, PROJECT_DIR)

from src.retrieval import VocabularyRetriever  # noqa: E402
from src.mollm_ensemble import (  # noqa: E402
    route as mollm_route,
    INGESTION_AUTO, INGESTION_RESOLVED, INGESTION_HITL,
)
from scripts.score_gold_recall import (  # noqa: E402
    load_gold, overlaps, _first_existing, GOLD_CANDIDATES,
)

RESOLVED_RE = re.compile(r"^RESOLVED_TO_CANDIDATE_(\d+)$")
AUTOMATIC_ROUTES = {INGESTION_AUTO, INGESTION_RESOLVED}


# ==========================================================================
# Data loading
# ==========================================================================

def load_decisions(conn, note_ids):
    """Every mollm_decisions row (all three modes) for these notes, joined to
    its Stage 2a/2b source row on the SAME safe composite key
    scripts/score_gold_recall.py and evaluation/cal_eval.py use --
    (note_id, original_text, expanded_text, gliner_label), never entity_id.
    See score_gold_recall.py's module docstring "KNOWN DB CAVEAT" for why:
    normalized_entities is unique on this composite key, not entity_id, so a
    duplicate-text entity's row can have been silently overwritten by a later
    mention sharing the same key, and joining on entity_id alone would read a
    stale/wrong row without any error.
    """
    rows = conn.execute("""
        SELECT d.mollm_call_id, d.entity_id, d.note_id, d.mode,
               d.ensemble_agreement, d.composite_confidence,
               d.citation_verified, d.mollm_routing_decision, d.queue_reason,
               d.models, d.retrieved_context,
               e.orig_start, e.orig_end, e.entity_label,
               n.candidates, n.omop_concept_id, n.omop_vocab, n.match_tier
        FROM mollm_decisions d
        JOIN extracted_entities e ON e.entity_id = d.entity_id
        JOIN normalized_entities n
          ON n.entity_id = e.entity_id
        -- 2026-09-01: was joined on (note_id, original_text, expanded_text,
        -- gliner_label) to work around a real DB defect where
        -- normalized_entities didn't reliably carry entity_id (see
        -- scripts/fix_normalized_entities_dedup_key.py) -- now fixed, and
        -- the old composite-key join would double-count rows since the fix
        -- landed (two entity_ids sharing that tuple now correctly have two
        -- separate normalized_entities rows).
        WHERE d.is_test = TRUE
          AND d.note_id IN ({})
    """.format(",".join("?" * len(note_ids))), note_ids).fetchall()

    cols = ["mollm_call_id", "entity_id", "note_id", "mode",
            "ensemble_agreement", "composite_confidence",
            "citation_verified", "mollm_routing_decision", "queue_reason",
            "models", "retrieved_context",
            "orig_start", "orig_end", "entity_label",
            "candidates", "stage2b_omop_concept_id", "stage2b_omop_vocab",
            "stage2b_match_tier"]
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        d["models"] = _json(d["models"], [])
        d["retrieved_context"] = _json(d["retrieved_context"], {})
        d["candidates"] = _json(d["candidates"], [])
        d["has_evidence"] = bool(d["retrieved_context"].get("rules"))
        out.append(d)
    return out


def _json(v, default):
    if not v:
        return default
    try:
        return json.loads(v) if isinstance(v, str) else v
    except (ValueError, TypeError):
        return default


# ==========================================================================
# Grading primitives -- shared by every ablation below
# ==========================================================================

def gold_concept_ids(row, gold_by_note):
    """SNOMED concept_ids of every gold annotation overlapping this entity's
    span, or None if nothing overlaps (Stage 2a extracted something outside
    the annotated set -- not gradable, and not a Stage 3 failure of any
    kind)."""
    gold = gold_by_note.get(row["note_id"], [])
    overlapping = [g for g in gold
                   if overlaps(row["orig_start"], row["orig_end"], g["start"], g["end"])]
    if not overlapping:
        return None
    return {g["concept_id"] for g in overlapping}


def grade_stage2b_baseline(row, gold_ids, vocab):
    """Was Stage 2b's OWN top-1 pick (before Stage 3 ever ran) correct?
    Same check scripts/score_gold_recall.py does for its combined linked
    recall figure, just re-run here per-decision so it can be crossed against
    Stage 3's routing choice. Returns True/False, or None if the pick can't
    be crosswalked to a SNOMED code at all (e.g. an unresolved RxNorm
    medication -- an uncrosswalked miss, not a wrong-concept miss; see
    score_gold_recall.py's "uncrosswalked" bucket for the same distinction)."""
    if not row["stage2b_omop_concept_id"]:
        return None
    code = vocab.snomed_code_for_concept(row["stage2b_omop_concept_id"])
    if code is None:
        return None
    return code in gold_ids


def official_verdict(row):
    """The ensemble's single verdict, or None on disagreement. Mirrors
    mollm_ensemble.route()'s own reading of model_results[0] -- disagreement
    routes to HITL before any single verdict is treated as "the" answer, so
    there is nothing to grade as an ensemble opinion when this is None."""
    if not row["ensemble_agreement"]:
        return None
    if not row["models"]:
        return None
    return row["models"][0].get("verdict")


def stage3_final_concept(row, vocab, routing_override=None):
    """The SNOMED code Stage 3 would commit to AUTOMATICALLY, i.e. only under
    a routing decision in {AUTO_VALIDATED, MOLLM_RESOLVED} -- HITL_REQUIRED
    has, by construction, no automatic answer, only a deferral. Returns
    (code_or_None, reason_string).

    routing_override lets a caller ask "what if routing had been X instead"
    without needing a second copy of this function -- used by
    citation_guard_value() for its counterfactual. Defaults to the row's
    actual recorded routing.
    """
    routing = routing_override if routing_override is not None else row["mollm_routing_decision"]
    if routing not in AUTOMATIC_ROUTES:
        return None, "deferred_to_hitl"

    if row["mode"] != "resolution":
        # Contradiction / non_asserted_check verdicts never reassign the
        # concept. An AUTO/RESOLVED routing here means Stage 3 ENDORSED
        # Stage 2b's existing pick, not that it chose a different one.
        if not row["stage2b_omop_concept_id"]:
            return None, "stage2b_pick_uncrosswalkable"
        code = vocab.snomed_code_for_concept(row["stage2b_omop_concept_id"])
        return code, ("stage2b_pick_uncrosswalkable" if code is None else "endorsed_stage2b_pick")

    verdict = official_verdict(row)
    if verdict is None:
        # Should not occur when routing is AUTO/RESOLVED (both require
        # ensemble_agreement per route()'s first safety rule) -- guarded
        # anyway rather than assumed.
        return None, "model_disagreement"

    m = RESOLVED_RE.match(verdict)
    if not m:
        # NONE_CORRECT / INSUFFICIENT_EVIDENCE cannot reach AUTO/RESOLVED
        # per route() -- if this fires, routing_override created a
        # combination production never produces, which is fine for a
        # counterfactual but worth surfacing rather than crashing on.
        return None, f"non_resolution_verdict:{verdict}"

    idx = int(m.group(1)) - 1
    candidates = row["candidates"]
    if idx < 0 or idx >= len(candidates):
        return None, "candidate_index_out_of_range"
    code = vocab.snomed_code_for_concept(candidates[idx].get("omop_concept_id"))
    return code, ("candidate_uncrosswalkable" if code is None else "resolved_to_candidate")


# ==========================================================================
# Ablation 1 -- overall value added
# ==========================================================================

def stage3_value_added(decisions, gold_by_note, vocab):
    """Stage 2b baseline accuracy vs. Stage 3's automatically-committed final
    accuracy, on the subset where BOTH are gradable (overlapping gold span,
    both picks crosswalkable). Also reports how much of the corpus Stage 3
    could decide on its own (coverage) vs. deferred -- accuracy on an
    ever-shrinking automatically-decided slice is not free, it trades
    directly against ablation 5's false-deflection cost.
    """
    baseline_correct = stage3_correct = n_comparable = 0
    n_deferred = n_ungradable = 0
    by_mode = collections.Counter()

    for row in decisions:
        gids = gold_concept_ids(row, gold_by_note)
        if gids is None:
            n_ungradable += 1
            continue
        b = grade_stage2b_baseline(row, gids, vocab)
        final_code, reason = stage3_final_concept(row, vocab)
        if reason == "deferred_to_hitl":
            n_deferred += 1
            continue
        if b is None or final_code is None:
            n_ungradable += 1
            continue
        n_comparable += 1
        baseline_correct += int(b)
        stage3_correct += int(final_code in gids)
        by_mode[row["mode"]] += 1

    return {
        "n_comparable": n_comparable,
        "n_deferred_to_hitl": n_deferred,
        "n_ungradable": n_ungradable,
        "stage2b_baseline_accuracy": (baseline_correct / n_comparable) if n_comparable else None,
        "stage3_automatic_accuracy": (stage3_correct / n_comparable) if n_comparable else None,
        "automatic_coverage": (n_comparable / (n_comparable + n_deferred))
                              if (n_comparable + n_deferred) else None,
        "by_mode": dict(by_mode),
    }


# ==========================================================================
# Ablation 2 -- grounded vs ungrounded
# ==========================================================================

def grounded_vs_ungrounded(decisions, gold_by_note, vocab):
    """Splits every decision by has_evidence (retrieved_context.rules
    non-empty) and reports citation-failure rate, HITL rate, and accuracy
    where gradable, per group. See module WARNING: do not average these two
    rows together, and always report N -- the whole point is that they may
    currently answer different questions."""
    groups = {True: [], False: []}
    for row in decisions:
        groups[row["has_evidence"]].append(row)

    out = {}
    for grounded, rows in groups.items():
        key = "grounded" if grounded else "ungrounded"
        n = len(rows)
        n_citation_fail = sum(1 for r in rows if r["citation_verified"] is False)
        n_hitl = sum(1 for r in rows if r["mollm_routing_decision"] == INGESTION_HITL)

        correct = comparable = 0
        for r in rows:
            gids = gold_concept_ids(r, gold_by_note)
            if gids is None:
                continue
            code, reason = stage3_final_concept(r, vocab)
            if reason == "deferred_to_hitl" or code is None:
                continue
            comparable += 1
            correct += int(code in gids)

        out[key] = {
            "n_decisions": n,
            "citation_failure_rate": (n_citation_fail / n) if n else None,
            "hitl_rate": (n_hitl / n) if n else None,
            "n_gradable_automatic": comparable,
            "automatic_accuracy": (correct / comparable) if comparable else None,
        }
    return out


# ==========================================================================
# Ablation 3 -- ensemble vs single model (resolution mode only -- the only
# mode with a gold label to grade against, same scoping as cal_eval.py)
# ==========================================================================

def _grade_single_verdict(verdict, candidates, gids, vocab):
    """Grades ONE model's own verdict in isolation, no agreement requirement
    -- used to ask "how good is this model alone", separate from what the
    gated ensemble decided."""
    if verdict == "NONE_CORRECT":
        candidate_codes = [vocab.snomed_code_for_concept(c.get("omop_concept_id")) for c in candidates]
        any_right = any(c in gids for c in candidate_codes if c)
        return not any_right, "none_correct"
    m = RESOLVED_RE.match(verdict or "")
    if not m:
        return None, f"non_resolution_verdict:{verdict}"
    idx = int(m.group(1)) - 1
    if idx < 0 or idx >= len(candidates):
        return None, "candidate_index_out_of_range"
    code = vocab.snomed_code_for_concept(candidates[idx].get("omop_concept_id"))
    if code is None:
        return None, "candidate_uncrosswalkable"
    return code in gids, "resolved_to_candidate"


def ensemble_vs_single(decisions, gold_by_note, vocab):
    """Per-model standalone accuracy (each model's verdict graded alone) next
    to the two-model agreement-gated ensemble's accuracy, on resolution-mode
    decisions with a gradable gold span. Answers: does requiring agreement
    between two models actually buy anything over trusting either one, or
    does it just throw away cases where a lone model happened to be right?
    """
    per_model = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    ensemble = {"n": 0, "correct": 0}
    agreement_count = 0
    n_resolution = 0

    for row in decisions:
        if row["mode"] != "resolution":
            continue
        n_resolution += 1
        gids = gold_concept_ids(row, gold_by_note)
        if gids is None:
            continue
        candidates = row["candidates"]

        for m in row["models"]:
            name = m.get("model") or "?"
            ok, _ = _grade_single_verdict(m.get("verdict"), candidates, gids, vocab)
            if ok is None:
                continue
            per_model[name]["n"] += 1
            per_model[name]["correct"] += int(ok)

        if row["ensemble_agreement"]:
            agreement_count += 1
            verdict = official_verdict(row)
            ok, _ = _grade_single_verdict(verdict, candidates, gids, vocab)
            if ok is not None:
                ensemble["n"] += 1
                ensemble["correct"] += int(ok)

    return {
        "n_resolution_mode_decisions": n_resolution,
        "agreement_rate": (agreement_count / n_resolution) if n_resolution else None,
        "per_model_accuracy": {
            name: {"n": v["n"], "correct": v["correct"],
                   "accuracy": (v["correct"] / v["n"]) if v["n"] else None}
            for name, v in per_model.items()
        },
        "ensemble_accuracy": {
            "n": ensemble["n"], "correct": ensemble["correct"],
            "accuracy": (ensemble["correct"] / ensemble["n"]) if ensemble["n"] else None,
        },
    }


# ==========================================================================
# Ablation 4 -- citation guard value (counterfactual, reuses real route())
# ==========================================================================

def citation_guard_value(decisions, gold_by_note, vocab):
    """For every decision HITL'd specifically because the citation guard
    failed it (queue_reason == "citation_verification_failed" -- i.e. every
    OTHER safety rule and threshold had already passed), replays the real
    mollm_ensemble.route() with citation_verified forced True to get the
    counterfactual routing, then grades what that counterfactual answer would
    have been. This reuses production's own route() rather than
    re-implementing its threshold logic a second time, so it cannot silently
    drift from what the pipeline actually does.

    catches = counterfactually-admitted decisions that would have been WRONG
    -- the guard's measured value.
    costs   = counterfactually-admitted decisions that would have been RIGHT
    -- cases where the guard deferred a decision that didn't need deferring
    (a real cost, in human review volume, not a free safety margin).
    """
    candidates_for_guard = [r for r in decisions if r["queue_reason"] == "citation_verification_failed"]

    catches = costs = ungradable = 0
    examples = []
    for row in candidates_for_guard:
        ensemble = {"ensemble_agreement": row["ensemble_agreement"],
                    "composite_confidence": row["composite_confidence"]}
        counterfactual = mollm_route(ensemble, {"citation_verified": True}, row["models"])
        code, reason = stage3_final_concept(
            row, vocab, routing_override=counterfactual["mollm_routing_decision"])

        gids = gold_concept_ids(row, gold_by_note)
        if gids is None or code is None:
            ungradable += 1
            continue

        would_be_correct = code in gids
        if would_be_correct:
            costs += 1
        else:
            catches += 1
            if len(examples) < 10:
                examples.append({
                    "note_id": row["note_id"], "entity_id": row["entity_id"],
                    "counterfactual_routing": counterfactual["mollm_routing_decision"],
                })

    n = len(candidates_for_guard)
    return {
        "n_citation_guard_hitl_decisions": n,
        "n_gradable": catches + costs,
        "catches_wrong_answer_prevented": catches,
        "costs_unnecessary_deferral": costs,
        "n_ungradable": ungradable,
        "catch_rate_of_gradable": (catches / (catches + costs)) if (catches + costs) else None,
        "example_catches": examples,
    }


# ==========================================================================
# Ablation 5 -- error-catching / false-deflection confusion matrix
# ==========================================================================

def error_catching_matrix(decisions, gold_by_note, vocab):
    """The cost-benefit number: crosses {Stage 2b baseline right/wrong}
    against {Stage 3 auto-admitted / deferred to HITL}, across ALL modes
    (this only needs the Stage 2b baseline grade, which every mode has --
    unlike ablations 2-4 it does not need Stage 3's own re-pick).

    catch_rate           = P(deferred to HITL | Stage 2b baseline was WRONG)
                            -- recall of catching bad answers before they
                            ship. Higher is better.
    false_deflection_rate = P(deferred to HITL | Stage 2b baseline was RIGHT)
                            -- cost: correct answers needlessly sent to a
                            human. Lower is better. This is exactly
                            Evaluation_Criteria.md's "false deflection rate"
                            metric, computed here per-decision rather than
                            from the Stage 5 re-audit sample it was
                            originally scoped for -- a useful early read, not
                            a replacement for that human-audited number.
    """
    tp_catch = fn_missed_wrong = fp_unnecessary_hitl = tn_correctly_auto = 0
    n_ungradable = 0

    for row in decisions:
        gids = gold_concept_ids(row, gold_by_note)
        if gids is None:
            n_ungradable += 1
            continue
        baseline_ok = grade_stage2b_baseline(row, gids, vocab)
        if baseline_ok is None:
            n_ungradable += 1
            continue
        deferred = row["mollm_routing_decision"] == INGESTION_HITL

        if not baseline_ok and deferred:
            tp_catch += 1
        elif not baseline_ok and not deferred:
            fn_missed_wrong += 1
        elif baseline_ok and deferred:
            fp_unnecessary_hitl += 1
        else:
            tn_correctly_auto += 1

    n_wrong = tp_catch + fn_missed_wrong
    n_right = fp_unnecessary_hitl + tn_correctly_auto
    return {
        "n_ungradable": n_ungradable,
        "stage2b_wrong_caught_by_hitl": tp_catch,
        "stage2b_wrong_missed_auto_admitted": fn_missed_wrong,
        "stage2b_right_unnecessarily_deferred": fp_unnecessary_hitl,
        "stage2b_right_correctly_auto_admitted": tn_correctly_auto,
        "catch_rate": (tp_catch / n_wrong) if n_wrong else None,
        "false_deflection_rate": (fp_unnecessary_hitl / n_right) if n_right else None,
    }


# ==========================================================================
# Reporting
# ==========================================================================

def print_report(report):
    print("=" * 78)
    print("STAGE 3 (MoLLM) VALUE-ADDITION ABLATIONS")
    print("=" * 78)
    print("\nSee module docstring WARNING before quoting any number below --")
    print("grounded/ungrounded split sizes matter more than either accuracy.\n")

    v = report["1_stage3_value_added"]
    print("--- 1. Overall value added (Stage 2b baseline vs. Stage 3 automatic) ---")
    print(f"  comparable N: {v['n_comparable']}   deferred to HITL: {v['n_deferred_to_hitl']}"
          f"   ungradable: {v['n_ungradable']}")
    if v["n_comparable"]:
        print(f"  Stage 2b baseline accuracy : {v['stage2b_baseline_accuracy']*100:.1f}%")
        print(f"  Stage 3 automatic accuracy : {v['stage3_automatic_accuracy']*100:.1f}%")
        print(f"  automatic coverage         : {v['automatic_coverage']*100:.1f}% "
              f"(rest deferred to a human)")
    print(f"  by mode: {v['by_mode']}")

    g = report["2_grounded_vs_ungrounded"]
    print("\n--- 2. Grounded vs. ungrounded (tests the KG-specific contribution) ---")
    for key in ("grounded", "ungrounded"):
        r = g[key]
        acc = f"{r['automatic_accuracy']*100:.1f}%" if r["automatic_accuracy"] is not None else "-"
        cf = f"{r['citation_failure_rate']*100:.1f}%" if r["citation_failure_rate"] is not None else "-"
        hr = f"{r['hitl_rate']*100:.1f}%" if r["hitl_rate"] is not None else "-"
        print(f"  {key:<11} n={r['n_decisions']:>4}  citation-fail={cf:>7}  "
              f"hitl-rate={hr:>7}  automatic-accuracy={acc:>7} (n_gradable={r['n_gradable_automatic']})")

    e = report["3_ensemble_vs_single"]
    print(f"\n--- 3. Ensemble vs. single model (resolution mode, n={e['n_resolution_mode_decisions']}, "
          f"agreement rate={e['agreement_rate']*100:.1f}%)" if e["agreement_rate"] is not None
          else f"\n--- 3. Ensemble vs. single model (n=0) ---")
    for name, v2 in e["per_model_accuracy"].items():
        acc = f"{v2['accuracy']*100:.1f}%" if v2["accuracy"] is not None else "-"
        print(f"  {name:<20} n={v2['n']:>4}  accuracy={acc}")
    ens = e["ensemble_accuracy"]
    acc = f"{ens['accuracy']*100:.1f}%" if ens["accuracy"] is not None else "-"
    print(f"  {'ENSEMBLE (agreeing)':<20} n={ens['n']:>4}  accuracy={acc}")

    c = report["4_citation_guard_value"]
    print(f"\n--- 4. Citation-guard value (counterfactual: guard disabled) ---")
    print(f"  n HITL'd for citation failure: {c['n_citation_guard_hitl_decisions']}  "
          f"gradable: {c['n_gradable']}")
    print(f"  would-have-been WRONG if admitted (guard's catches): {c['catches_wrong_answer_prevented']}")
    print(f"  would-have-been RIGHT if admitted (guard's cost)   : {c['costs_unnecessary_deferral']}")
    if c["catch_rate_of_gradable"] is not None:
        print(f"  catch rate among gradable: {c['catch_rate_of_gradable']*100:.1f}%")

    m = report["5_error_catching_matrix"]
    print(f"\n--- 5. Error-catching / false-deflection matrix (all modes) ---")
    print(f"  Stage2b wrong, caught by HITL      : {m['stage2b_wrong_caught_by_hitl']}")
    print(f"  Stage2b wrong, missed (auto-admit)  : {m['stage2b_wrong_missed_auto_admitted']}")
    print(f"  Stage2b right, unnecessarily deferred: {m['stage2b_right_unnecessarily_deferred']}")
    print(f"  Stage2b right, correctly auto-admit : {m['stage2b_right_correctly_auto_admitted']}")
    if m["catch_rate"] is not None:
        print(f"  catch rate            : {m['catch_rate']*100:.1f}%")
    if m["false_deflection_rate"] is not None:
        print(f"  false deflection rate : {m['false_deflection_rate']*100:.1f}%")

    n_total = (v["n_comparable"] + v["n_deferred_to_hitl"] + v["n_ungradable"])
    print(f"\nSAMPLE SIZE CAVEAT: {n_total} total Stage 3 decisions loaded. See module "
          f"docstring -- methodology validation only at this scale, not a result.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note-ids", default=None,
                     help="Comma-separated note_ids. Default: every note_id with "
                          "is_test=TRUE rows in mollm_decisions.")
    ap.add_argument("--gold", default=None, help="path to train_annotations.csv")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default=None, help="write full JSON report here")
    args = ap.parse_args()

    gold_path = args.gold or _first_existing(GOLD_CANDIDATES, "gold annotations CSV")
    conn = duckdb.connect(args.db, read_only=True)
    try:
        if args.note_ids:
            note_ids = [n.strip() for n in args.note_ids.split(",") if n.strip()]
        else:
            # 2026-08-31 FIX: excludes the locked test split by default --
            # see evaluation/splits.py; this script's default previously had
            # no such guard.
            from evaluation.splits import load_split
            all_notes = {r[0] for r in conn.execute(
                "SELECT DISTINCT note_id FROM mollm_decisions WHERE is_test = TRUE"
            ).fetchall()}
            note_ids = sorted(all_notes - load_split("test"))
        if not note_ids:
            raise SystemExit("No is_test=TRUE rows in mollm_decisions. "
                             "Run scripts/test_stage3_live.py --store first.")

        print(f"gold:  {gold_path}")
        print(f"db:    {args.db}")
        print(f"notes: {note_ids}")

        gold_rows = load_gold(gold_path, note_ids)
        gold_by_note = collections.defaultdict(list)
        for g in gold_rows:
            gold_by_note[g["note_id"]].append(g)

        decisions = load_decisions(conn, note_ids)
        if not decisions:
            raise SystemExit(f"No mollm_decisions rows found for {note_ids}.")

        vocab = VocabularyRetriever(conn)

        report = {
            "n_decisions_loaded": len(decisions),
            "1_stage3_value_added": stage3_value_added(decisions, gold_by_note, vocab),
            "2_grounded_vs_ungrounded": grounded_vs_ungrounded(decisions, gold_by_note, vocab),
            "3_ensemble_vs_single": ensemble_vs_single(decisions, gold_by_note, vocab),
            "4_citation_guard_value": citation_guard_value(decisions, gold_by_note, vocab),
            "5_error_catching_matrix": error_catching_matrix(decisions, gold_by_note, vocab),
        }
    finally:
        conn.close()

    print_report(report)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
