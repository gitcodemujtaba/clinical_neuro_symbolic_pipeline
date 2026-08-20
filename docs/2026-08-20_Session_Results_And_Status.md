# 2026-08-20 — Session Results & Current Status

Companion to `docs/Implementation_Methodology.md` (the architecture
reference) — this doc is the point-in-time results snapshot. See that
doc's own header for how the two relate, and `docs/2026-08-19_Lab_Procedure_Vs_Observable_Entity_Finding.md`
for the full investigation this session's Tier 2 work builds on.

## 1. Tier 2 (`TIER_2_AUTO_RESOLVED`) — root cause fixed, held out of AUTO pending re-validation

Measured baseline going into this session: ~20% precision on
`TIER_2_AUTO_RESOLVED` (3/3 unanimous re-rank to a candidate other than
#1). Two real, distinct root causes found and fixed, both verified against
live data, not assumed:

* **Lab-Test Procedure-vs-Observable-Entity confusion.** SNOMED carries
  parallel Procedure-class and Observable-Entity-class concepts for the
  same lab test (e.g. "MCH" / "MCHC" siblings); Tier 3 ranking sometimes
  preferred the wrong one. Fixed in `tier_retrieval.py`'s
  `_prefer_lab_procedure_over_observable()` (now tags the winning
  candidate's `match_basis` so downstream logic can trust it
  deterministically) plus a new `_lab_procedure_fast_path()` in
  `mollm_tier_gate.py` that only fires when `is_ambiguous` is False (the
  guard that was initially missing and briefly caused MCH to resolve
  confidently to MCHC's concept — caught and fixed before shipping).
* **`CONDITION_VS_OBSERVATION_PRIOR` was permanently dead code.** It
  required a `concept_class_id` field on candidates that
  `_candidate()`'s real production path never populates (only an idealized
  test fixture had it) — meaning this tiebreak rule had *never fired in
  production*, on any decision, ever. Relaxed to a `domain_id`-only check
  in `_condition_vs_observation_duplicate()`; added a regression test
  using a `concept_class_id`-free candidate pair specifically so this
  can't silently regress back to dead code again.

**Decision, not yet reversed:** rather than trust these fixes recover
Tier 2's precision without direct re-measurement, `TIER_2_AUTO_RESOLVED`
stays excluded from `AUTO_TIERS` — it now routes to HITL review
(`queue_reason="tier2_auto_resolved_pending_revalidation"`) instead of
writing automatically, even though it's still detected/labeled as the same
structurally-unanimous signal. This was an explicit, conservative choice
(a third-party suggestion to kill Tier 2 outright was reviewed and
declined in favor of this narrower gate). **Open**: the fresh-note
re-evaluation that would justify re-including it in `AUTO_TIERS` is
pending the fresh25 batch below.

## 2. Calibrator retrained — new baseline adopted

`ConsensusCalibrator` retrained on a larger pool (`scripts/retrain_calibrator_full_corpus.py`,
note-disjoint train/val split). **Val AUROC improved 0.74 → 0.845.**
Adopted into production (`models/consensus_calibrator_v1.pkl`; the prior
version is kept alongside as `.bak_2026-08-20`, not deleted). The script's
own adoption gate (`--save` only overwrites the production model if the
new AUROC beats the current baseline) held here — this is a genuine
improvement, not a lateral retrain.

**Caveat carried forward, not yet closed:** this is still an internal
note-disjoint validation, not a genuinely fresh-note test. The fresh25
batch below is also the vehicle for closing that gap.

## 3. Three 8B-model hard-case-resolution architectures — built, measured, shelved

All three target cases the 3B ensemble itself can't unanimously resolve.
Each was iterated on live user feedback (remove deterministic KG bypass,
widen KG search via both SNOMED code and name, add a gold-verified
worked example to the prompt, give the model independent concept-type
judgment rather than trusting a possibly-wrong upstream label, fix a real
sampling bias before trusting any precision number).

| Architecture | Approach | Result |
|---|---|---|
| `src/tier4_kg_escalation.py` | From-scratch candidate re-evaluation with KG grounding | 27.8% (5/18) — poor. 58% of sampled hard cases had only ONE candidate at all: a retrieval-bottleneck problem, not a reasoning one. |
| `src/tier2b_llm_candidate_generation.py` | Stage 2b augmentation: generate-then-verify a new candidate against real vocabulary, independent type judgment | 10.7% recall recovery (3/28) — a narrow win concentrated in one category. |
| `src/tier4_arbiter_8b.py` | End-of-pipeline arbiter shown the 3B ensemble's own verdicts + reasoning + KG evidence | **38.0% → 51.0% precision** on a properly note-diversified N=100 sample (100 distinct notes, max 3 entities/note) — 14 fixes to 1 regression. Clearly the strongest of the three. |

**Explicit decision: all three stay shelved, unwired from `AUTO_TIERS` and
the production routing path.** Confirmed at the end of this session that
none are imported into `mollm_tier_gate.route_tier()` or any production
script. The user's own framing for this decision: solid initial telemetry
still needs a statistically airtight, cross-note-validated case before
touching the active gate — double down on already-validated components
(Tier 1B calibrator, Tier 3 allow-list/brand aliases, a real fresh-note
Tier 2 re-evaluation) instead of shipping a promising-but-narrowly-tested
new gate.

## 4. IoU benchmark-metric fidelity fix

Checked the actual DrivenData SNOMED-CT benchmark page
(https://www.drivendata.org/benchmarks/310/benchmark-snomed-ct/page/983/)
directly against `evaluation/iou_metrics.py`'s implementation. Found two
real bugs, both fixed:

* **Wrong "class" semantics.** The module was pooling all spans into one
  fake "ALL" bucket, on the mistaken premise that this project's gold has
  no per-class field. It does — the benchmark's own "class" IS the SNOMED
  concept ID, and gold's `concept_id` column already carries it. The old
  number was also being computed at Stage 2a, before any concept is even
  resolved — structurally impossible to be the real metric, since the
  benchmark's IoU is concept-gated ("the predicted concept ID must match
  exactly; relationships between concepts are not taken into account for
  scoring"). Fixed via a new `benchmark_char_iou()`, computed at Stage 2b
  using each entity's resolved concept as its class; Stage 2a's old number
  is now honestly labeled `span_only_char_iou`, a concept-blind diagnostic.
* **Cross-note character collision.** Character sets were keyed by raw
  integer offset — scoring multiple notes at once could spuriously overlap
  two unrelated notes' spans sharing a numeric offset range. Fixed by
  keying on `(note_id, offset)`; verified with a synthetic test showing two
  notes with identical offsets no longer collide.

No numeric baseline exists yet for the corrected `benchmark_char_iou` —
the old number was measuring something else entirely, so there's nothing
valid to compare it against. Wired into
`ui/pages/4_📊_Evaluation_Metrics.py`'s Stage 2b section; full test suite
(77/77) still passes.

## 5. Troubleshooting UI — gold-vs-prediction span diff

`ui/pages/3_🔍_Troubleshooting.py`'s note highlighter extended per explicit
spec: green = our entities, blue = abbreviations (both pre-existing), new
gold/amber highlight for gold annotations, and — where our span and a gold
span overlap but disagree on exact start/end — the shared character range
renders grey with each side's non-overlapping extension kept in its own
color, rather than one flat highlight hiding the boundary mismatch.
SNOMED codes now also shown in our own prediction tooltips (previously
concept names only). Verified via `streamlit.testing.v1.AppTest` (no
exceptions) and a standalone test of the offset-splitting logic against
exact-match, ours-wider, gold-wider, and staggered-overlap cases.

## 6. Fresh25 validation batch — in progress, not yet graded

25 genuinely fresh notes (outside every calibrator train/val note) queued
for Stage 1→2a→2b, chained into Stage 3, specifically to close two open
items above: a real fresh-note `TIER_2_AUTO_RESOLVED` re-evaluation, and a
real fresh-note calibrator check against the 0.845 internal-validation
number. **Status as of this doc: still running** (note 1 of 25 — the
corpus's known longest/slowest note — done; note 2 in progress). Grading
against gold has not happened yet; no numbers from this batch are reported
anywhere in this doc or elsewhere, deliberately, until it actually
completes.

## Open items carried forward

* Fresh-note `TIER_2_AUTO_RESOLVED` precision (blocks re-including it in
  `AUTO_TIERS`).
* Fresh-note `TIER_1B_CALIBRATED_AUTO_VALIDATED` precision against 0.845.
* No corrected `benchmark_char_iou` baseline yet — first real measurement
  should happen against the fresh25 batch once it's graded, not against
  calibrator-training notes.
* The 8B arbiter (`tier4_arbiter_8b.py`) remains the strongest unwired
  candidate for a future escalation tier, contingent on a larger
  cross-note validation the user has not yet requested.
