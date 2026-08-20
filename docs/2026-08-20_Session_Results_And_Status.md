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

## 6. Fresh25 validation batch — complete and graded

25 genuinely fresh notes (outside every calibrator train/val note),
Stage 1→2a→2b→3, 3,519 entities processed (0 errors), 390.5 minutes total.
Graded against gold via `evaluation/grade_fresh25_by_tier.py`
(clean-span, SNOMED-crosswalked, same methodology as every other
tier-grading script this project uses):

| Tier | n decisions | Clean-span precision |
|---|---|---|
| Tier 1 (unanimous) | 1,248 | 544/690 = **78.8%** |
| Tier 1B (calibrator) | 197 | 100/108 = **92.6%** |
| Tier 2 (unanimous re-rank) | 94 | 11/68 = **16.2%** |
| Tier 3 (fast path) | 491 | 323/364 = 88.7% |
| Combined `AUTO_TIERS` (1+1B+3) | 1,936 | 967/1,162 = **83.2%** |

**Tier 2: the fix from §1 did not recover precision at scale (16.2% vs.
the ~20% baseline it was meant to fix) — confirming the decision to hold
it out of `AUTO_TIERS` was correct, not overcautious.** Root cause dug
into directly: every single `TIER_2_AUTO_RESOLVED` decision in the DB
(259/259, all entity types) is flagged `is_ambiguous=True` by retrieval.
This is structural, not incidental — Tier 2 requires all 3 models to
unanimously re-rank *away* from retrieval's own top candidate, which only
happens when that top candidate already looks shaky. So "3/3 unanimous"
in this specific population is more likely to reflect the 3 models
sharing a common bias than independently verifying the same correct
answer. Confirmed concretely for the lab-panel cases (MCV/MCHC/RDW):
retrieval's `lab_procedure_preferred` tag on candidate #1 is present and
correct, the fast path correctly declines because `is_ambiguous=True`,
and the LLM ensemble's fallback judgment is what lands on the wrong
candidate. **This tier likely needs calibrator-style scoring, not a
deterministic fast path, to ever safely re-enter `AUTO_TIERS`** — a
concrete next step, not yet built.

**Tier 1B calibrator: validated, holds up well on fresh notes (92.6%)** —
a real improvement over the earlier 5-note check (84.2%) that had
triggered the 0.65→0.72 threshold raise. The retrained calibrator
(0.845 AUROC), the two hard traps, and the 0.72 threshold are working
together as intended on a larger, more diverse fresh sample. This closes
the open item from §2 cleanly and positively.

**Tier 1: lower than prior checkpoints (78.8% vs. 84–88% at earlier
checkpoints) — partially explained, partially a new finding.** Broken
down by entity type: `Medication` is 18/18 (100%) wrong, fully explained
by the already-documented gold-annotation-schema gap (gold codes
medications to Administration-of-X procedure concepts, not the drug
substance — see the medication crosswalk finding in prior session
memory, not new). `Lab Test` is 64/94 (68%) wrong — **new, and not
caused by the `lab_procedure_preferred` mechanism** (only 1/120 Lab Test
Tier-1 candidates used that match basis; the rest were plain
`semantic_similarity`). Direct lookup of gold's own target concepts shows
this is a genuine SNOMED near-duplicate-concept problem: gold's targets
are themselves Procedure/Measurement-class concepts, just a *different*
specific one than what SapBERT matched (e.g. gold wants "Blood calcium
measurement" [312472004], we matched "Calcium measurement" [71878006];
for ALT, we matched a UK-SNOMED-extension code instead of the US-core
concept gold uses). A previously unquantified semantic-retrieval
precision ceiling for common labs with multiple close SNOMED synonyms —
distinct from anything this session's fixes targeted, and not yet
addressed.

## 7. Follow-up fixes after the fresh25 grading

Three pieces of work directly targeting the fresh25 findings above:

**Tier 2 calibrator escape hatch, built and shadow-validated (not yet
useful).** `route_tier()`'s unanimous-re-rank branch now consults a
fitted `ConsensusCalibrator` the same way the split-vote path already
does, landing promotions in a new, structurally separate
`TIER_2B_CALIBRATED_AUTO_RESOLVED` tier (not `TIER_1B`, since Tier 2's
feature distribution -- 100% `is_ambiguous`, zero vote disagreement -- is
materially different from what the calibrator was fit on). Retroactively
shadow-validated against the fresh25 batch's 94 stored Tier 2 decisions
(`scripts/shadow_validate_tier2b.py`, no pipeline re-run needed): every
single decision scored 0.0-0.1, nowhere near the 0.72 threshold -- zero
would-be promotions. This is the correct, safe outcome given how the
calibrator was trained; the mechanism is verified to never wrongly
promote with the current model, but needs its own Tier-2-shaped training
data (which doesn't exist in sufficient volume yet) before it can
actually help.

**Lab Test near-duplicate-concept ceiling, fixed.** Gold-mined 12 new
curated aliases into `_LAB_TEST_ALIASES` (calcium, alt, na, urean, ph,
creat, phos, inr, rdw, mcv, mchc, total co2 -- each 100% consistent in
gold, n=41-491). Also found and fixed a deeper gap: force-inclusion into
the candidate pool alone isn't sufficient (checked raw SapBERT similarity
directly -- the correct alias concept often doesn't rank #1, sometimes
not even top-5), and `tier3_fast_path()`'s deterministic bypass had never
actually checked for `match_basis == "verified_lab_test_alias"` despite
the module's own docstring claiming it did -- confirmed this was already
silently affecting the *existing* `hct` alias in production (`HCT-32`
entities were landing in `TIER_5_TRUE_AMBIGUITY`, not any auto-write
tier). Added a dedicated fast-path branch for it.

**KG search-loop escalation for Tier 4/5, built, smoke-tested, NOT
adopted.** A genuine multi-round loop (`src/kg_search_loop.py`) letting
an 8B model request KG searches (relationships/ancestor/name-collision/
free-text) round by round before committing to a verdict, rather than a
one-shot pre-fetch. Found and fixed a real bug during the smoke test (the
schema let the model decline to decide even on the forced-final round).
Even after the fix, 7/9 smoke-tested entities still declined to commit to
a verdict after 2-4 search rounds ("I need more evidence, but I'm not
sure what specific search would be helpful"). Verdict: a 3-8B-class model
doesn't reliably have the planning/executive function to drive its own
search strategy -- the existing one-shot pre-fetch design
(`src.tier4_kg_escalation`, specifically the arbiter architecture at
51.0% precision) remains the reference design. Not wired in; the planned
50-entity graded batch validation was cancelled once the smoke test's
pattern made the direction clear.

## 8. Extraction recall gap — root-caused and partially closed

Checked against the leaderboard comparison the user provided (DriveData
SNOMED-CT benchmark public rankings): this pipeline's macro/support-
weighted char IoU (0.1431 / 0.2400) currently ranks last and second-to-
last respectively against the 5 public entries shown. Root-caused the
single largest contributing factor: **Stage 2a extraction recall is only
46.1%** across the full 140-note DB, and char IoU requires actual
character overlap, so roughly half of gold's characters are
mathematically unreachable before linking quality even enters the
picture.

Corpus-wide miss analysis (140 notes, 38,689 gold annotations) found:
- Every missed gold concept's SNOMED domain is already inside what the
  6 GLiNER labels target -- no missing entity-type category, this is a
  pure extraction-quality gap.
- **100% of misses are true zero-shot blind spots** (GLiNER proposes
  nothing overlapping at any confidence, not just below threshold) --
  rules out threshold tuning as a lever entirely.
- 97.4% of misses are short (<=25 chars), 71% single-word. The
  most-repeated missed texts are dominated by bare CBC/chemistry-panel
  abbreviations with no attached value ("Creat trended up", not
  "Creat-1.2"): Creat, Hgb, RBC, Na, Hct, Cl, MCH, MCHC, RDW, HCO3, WBC,
  UreaN, Phos, Calcium, AnGap -- 2,097 misses from these 15 terms alone.

**Fix built**: `src/lab_abbrev_coldstart.py`, same architecture as
`src.physexam_shorthand` (gold-mined text->concept injection, skips
anything GLiNER already found), reusing `_LAB_TEST_ALIASES`/
`tier3_fast_path`'s existing mechanism for concept resolution rather than
needing a new bypass. 21 case-sensitive terms, each independently
verified against a live DB consistency check (caught and fixed one real
copy-paste bug -- HCO3 was initially mapped to the wrong concept).

**Measured impact (read-only simulation against real note text + gold,
not yet a re-processed/re-measured corpus)**: extraction recall
**46.4% -> 51.9% (+5.5 points)**, recovering 2,140 of 20,754 originally-
missed gold spans. Verified end-to-end on the single worst-affected note
(`10513485-DS-7`): 101/101 target-term misses recovered, 100%.

**Second fix, same session**: `src/narrative_state_word_coldstart.py` --
a small, gold-consistency-screened set of common single-word state
descriptors (`alert`, `improved`, `baseline`, `warm`, `clinic`, all
>=95% consistent to one concept, case-merged). Explicitly screened OUT
several tempting but genuinely polysemous candidates (`pain` 76.7%
across 4 concepts, `stable` 88.4%, `negative` 64.5% split ~65/28 across
two different concepts, `procedure`/`support`/`tender`/`masses`/`wound`
all below bar) rather than force-injecting a false-confidence majority.
Reuses `src.physexam_shorthand`'s concept-resolution bypass via a newly
generalized `_cold_start_mapping()` in `orchestrator.py`.

**Combined measured impact**: extraction recall **46.4% -> 53.2%
(+6.8 points total)** -- the narrative-word fix adds 493 additionally-
recovered spans on top of the lab-abbreviation fix's 2,140.

## 9. Proposal alignment gap analysis, and manual-baseline benchmark (Gap 3)

Mapped the current implementation against the original COM748 proposal's
5 objectives and Methodology section. Three real gaps found: (1)
guideline-derived KG injection (Objective 2 / Methodology §1) -- data
prepared (76 processed triplet files) but `scripts/init_memgraph_
guidelines.py` never actually implemented, the production tier gate
does not consume it; (2) KG embeddings (Objective 4, TransE/RotatE/
CompGCN from repurposed classification data) -- confirmed via direct
code search, the only trace anywhere in the codebase is a single
comment in `src/kg3_query.py` acknowledging it as unbuilt future work;
(3) manual-annotation-baseline benchmarking (Objective 5 / Methodology
§4) -- never measured, no baseline existed anywhere in this project
before this session. Two apparent gaps are NOT real gaps and should be
written up as closed, deliberate outcomes: the acronym-escalation
cold-start (built, corpus-scale tested, found insufficient at 34-36%
precision, correctly kept off) and the dry-run-only KG3 write gate
(deliberately withheld pending precision, the correct call).

**Gap 3 closed**: real, source-verified manual-annotation-speed
benchmark. Source: Wei, Q., Franklin, A., Cohen, T., & Xu, H. (2018).
"Clinical text annotation -- what factors are associated with the cost
of time?" *AMIA Annual Symposium Proceedings*. 9 clinician annotators
tagging problems/treatments/tests (directly comparable to this
project's Condition/Medication/Procedure/Lab Test categories) on the
2010 i2b2/VA clinical NLP dataset, 6,663 sentences -- measured
40.47-92.22 words/minute. This project's own corpus: 272 notes, mean
1,554.1 words/note (computed directly).

| | Manual (range) | This pipeline | Ratio |
|---|---|---|---|
| Extraction+linking only (Stage 1->2a->2b) | 16.85-38.41 min/note | 5.71 min/note | 2.95x-6.73x faster |
| Full pipeline incl. tier-gate (Stage 1->2a->2b->3) | 16.85-38.41 min/note | 21.33 min/note | 0.79x-1.80x -- barely faster, SLOWER than the fastest manual annotator |

**Honest finding, not flattering**: once Stage 3's LLM-ensemble
validation is counted, the raw speed advantage over manual annotation
largely disappears (up to 18 LLM calls/entity is genuinely more
compute-intensive than a human reading and clicking once). The correct
framing for the paper is NOT a velocity claim -- it's that 55-57% of
entities never require a human at all, cutting required expert review
volume by over half. That claim is well-supported; a raw-speed claim
for the full pipeline is not.

**Three supplementary citations checked, one has a real misattribution
problem worth flagging if it surfaces elsewhere**: arXiv:2405.02664
(MedPromptExtract) confirmed verbatim ("9.6 man-hours for 48
annotations... 12 minutes per document"); a ResearchGate Nov 2024
pharmacovigilance-NLP productivity study confirmed (4.689 min vs 0.729
min/document, 84% gain, 98.24% accuracy); general industry
medical-annotation cost premium (3-5x general NLP annotation cost)
directionally confirmed across independent sources. One claim -- "124
words/hour... when SNOMED CT ontology linking is required" -- traced to
a REAL number (TLT8 proceedings) but the WRONG task: that figure is
from general syntactic treebank (dependency-parsing) annotation, with
no connection to clinical NER or SNOMED linking. Not used.

## 10. Real bug found and fixed during the recall-fix backfill

While running Stage 3 tier-gating on the newly-injected lab-abbreviation/
narrative-state-word cold-start entities (§8), found a genuine, live bug:
`_collapse_hierarchy_duplicates()` (the step that merges SNOMED parent/
child candidate pairs so they don't fracture ensemble voting into an
artificial split) picks whichever candidate has the higher **raw**
similarity score when two candidates share an "Is a"/"Subsumes" edge. A
curated alias candidate is scored by its own real cosine similarity, not
pinned to 1.0 -- so this silently discarded a gold-verified alias in
favor of a same-hierarchy sibling that merely scored higher on plain
semantic similarity, with no awareness that one candidate was curated
ground truth and the other a guess.

Confirmed live on "HCO3": the curated concept (4194291, "Blood
bicarbonate measurement") lost to its own SNOMED parent (4227915,
"Bicarbonate measurement", 0.8857 similarity) purely because the
non-curated sibling scored higher -- candidate[0] ended up being the
*wrong* concept even though the correct one had been successfully
force-included into the pool.

**Fixed**: a candidate carrying a curated match_basis
(`verified_lab_test_alias`/`verified_brand_alias`/`lab_procedure_preferred`)
now wins its hierarchy root unconditionally, regardless of raw score.
Verified directly (`normalize_entity('HCO3', ...)` now correctly returns
the curated concept as candidate #1). 7 new regression tests. All 2,899
already-written cold-start entities were re-normalized under the fix
(`scripts/refix_coldstart_hierarchy_collapse.py`), and the 100 stale
tier-gate decisions the partially-completed Stage 3 pass had already
written under the buggy candidates were cleared and are being re-graded.

**Why this matters beyond the immediate fix**: this is the second time
this session a downstream ranking/dedup step turned out to have no
awareness of curated match_basis tags (`tier3_fast_path()`'s missing
`verified_lab_test_alias` branch, §1, was the first). Worth a broader
audit of every function that ranks or collapses candidates for the same
blind spot, rather than assuming this is the last instance.

## Open items carried forward

* **Tier 2's calibrator escape hatch needs its own training data.** The
  mechanism (§7) is built and verified safe (never wrongly promotes with
  the current model), but genuinely inert until a calibrator variant is
  fit specifically on Tier-2-shaped (unanimous re-rank, is_ambiguous)
  examples with real gold labels -- not enough volume exists yet.
* No corrected `benchmark_char_iou` baseline yet — should be computed
  against this now-graded fresh25 batch next, not against
  calibrator-training notes.
* **The 8B arbiter (`tier4_arbiter_8b.py`) is confirmed the reference
  design for future escalation-tier work** (51.0% precision) after the
  KG search-loop alternative was tried and found inferior (§7) —
  contingent on a larger cross-note validation the user has not yet
  requested.
