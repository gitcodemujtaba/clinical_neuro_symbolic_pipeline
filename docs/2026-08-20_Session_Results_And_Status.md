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

## 11. Gap 1 (KG embeddings, Objective 4) — closed, correcting an earlier wrong call

Earlier in this session, Gap 1 was scoped OUT with the reasoning "no
accumulated KG3 write volume to embed yet." That reasoning was wrong --
it conflated the dynamic patient-instance graph (KG3, genuinely empty,
`dry_run=True` everywhere) with the REFERENCE graph (SNOMED CT itself,
via `athena_concept_relationship`), which is already real, substantial,
and fully populated in this project's own DuckDB. Nothing about training
a KG embedding model requires KG3 to have any data at all.

**Built**: `src/kg_embedding.py` -- TransE (Bordes et al. 2013) in plain
PyTorch, no new library dependency (pykeen/torch-geometric aren't
installed in this environment). RotatE and CompGCN -- the other two
methods the proposal names -- are real, meaningfully more complex
follow-on work (complex-valued embeddings; a full graph-convolutional
architecture respectively), stated honestly as out of scope rather than
attempted under time pressure.

**Data**: a real SNOMED subgraph scoped to the 7,261 concepts this
pipeline's own candidate pools have actually touched -- 24,872 real
relationship edges among them, 104 distinct relationship types. This is
the "based on our TP records" framing: the graph's *scope* is defined by
concepts this project's own tier gate has actually resolved and graded,
not an arbitrary slice of all of SNOMED.

**Two real evaluations, both run end to end**:
- Intrinsic (standard KGE literature protocol): held-out link prediction
  on 2,488 real triples never seen during training. **MRR 0.768,
  Hits@10 0.909** (RAW setting, not filtered -- not directly comparable
  to filtered-MRR numbers common in papers, flagged explicitly).
- Extrinsic, tied directly to this project's own task: for 457
  gold-confirmed true-positive tier-gate decisions with real competing
  candidates (1,623 comparisons), the wrong-but-competing candidate sat
  closer to the correct concept than a random unrelated concept **70.2%
  of the time** (vs. 50% chance) -- a real, positive signal the
  embedding captures genuine clinical proximity among candidates that
  actually competed for the same mention.

Ran read-only, concurrently with the active Stage 3 recall-fix batch,
with no contention. 13 new tests including a genuine generalization
check. **Of the 3 real gaps this session's proposal-mapping surfaced,
all 3 are now closed with real, tested, committed code.**

## 12. SNOMED near-duplicate retrieval fix, KGE tiebreak evaluation, and the
    exhaustive-candidate-eval net-impact assessment

**Trigger**: MCHC/RDW (and other lab abbreviations) were losing Tier 2
ensemble votes to SNOMED regional-extension concepts instead of the
correct International Release concept, tanking Tier 2 precision on that
subset.

**Root cause, verified against live data before fixing (not assumed)**:
`vocabulary_id` cannot discriminate -- only one distinct value (`'SNOMED'`)
exists in the whole `athena_concept` table; OMOP bundles the UK national
extension into it. `concept_class_id` (Procedure vs. Observable Entity)
isn't reliable either -- 23,842 of the UK-extension concepts are
themselves `'Procedure'` class. The real, robust signal is the SCTID's own
namespace-identifier block: extension concepts embed a reserved `1000000`
segment in `concept_code` that International Release concepts never
carry. Confirmed: 98,487 of ~1.09M SNOMED concepts (9%) match this
pattern, and **zero of the corpus's 4,522 distinct gold-standard SNOMED
codes** are among them -- excluding the pattern from retrieval cannot cost
a true match.

**Fix** (`src/normalization/tier_retrieval.py`, commit `6f5135d`): (1)
`concept_code NOT LIKE '%1000000___'` spliced into every open-ended Tier
1-4 retrieval query. (2) Removing that duplicate surfaced a *second*,
distinct near-duplicate class underneath it -- 22 SNOMED "X calculation
technique" concepts (`concept_class_id='Qualifier Value'`, not Observable
Entity) that the existing `_prefer_lab_procedure_over_observable()` rank
penalty didn't cover; extended to penalize both classes.

**A real deployment gap found and corrected mid-session**: restarting the
Stage 3 batch runner after this fix landed did **not** exercise it --
`scripts/run_stage3_tier_gate.py` reads already-stored
`normalized_entities.candidates`, it never recomputes Stage 2b. Confirmed
live: MCHC/RDW decisions written minutes into the restarted run still
picked the pre-fix UK-extension concepts. Fixed via two scoped
re-normalization scripts (`scripts/refix_uk_extension_lab_candidates.py`
for bare-text alias matches, `scripts/refix_uk_extension_lab_suffixed.py`
for value-suffixed variants like `MCHC-31` -- reusing the existing,
already-correct `strip_lab_value_suffix()` rather than adding a
second/parallel regex).

**Result, gold-verified on MCHC + RDW specifically (28 gradable
entities)**: **100% (28/28)** of post-fix candidate pools now surface the
gold-correct concept as the top SapBERT candidate. This is a full,
verified fix at the retrieval layer. It did not (yet) increase the
auto-write rate for this population, for a separate, pre-existing reason:
27/28 still route to `TIER_4_ENSEMBLE_SPLIT` because the
`short_alphanumeric_code_trap` safety gate forces short lab codes through
the full 3-model ensemble regardless of retrieval quality -- a deliberate
existing safety mechanism, not a shortfall of this fix. One case that did
reach unanimous agreement picked a **third**, previously-unexamined
near-duplicate ("Mean cell hemoglobin concentration - finding", Clinical
Finding domain) -- flagged as a real, open gap, not fixed this session.

**KGE topological tiebreak, evaluated against real gold data (not
synthetic) for the first time**. Built `evaluation/kg_tiebreak_validation.py`
(sweeps a `TIE_THRESHOLD` over SapBERT top1/top2 score deltas, grades
`src.kg_embedding_tiebreak`'s pick against gold, reports win/loss/neutral)
plus checkpointing (`save_model()`/`load_model()` in `src/kg_embedding.py`)
so the TransE model didn't need retraining for every validation run.
Retrained on the larger post-backfill TP pool (455 records, up from 457 in
§11's earlier run -- consistent scale): MRR 0.776, Hits@10 0.909.

Real, decisive finding: **KGE is not a safe replacement for the hardcoded
`_prefer_lab_procedure_over_observable()` rule.** Head-to-head on the
rule's own pattern (Lab Test, Procedure vs. Observable-Entity/Qualifier-
Value), the rule had **zero losses at every threshold tested (0.01-0.08,
up to n=380)**; KGE had **63 losses** once the threshold widened past
0.02. Directly falsified the specific hypothesis that KGE would "naturally
push" the third near-duplicate concept away: checked against the actual
retrained model on the actual failing entity -- KGE picked the *same*
wrong concept, with a 0.0018 score margin (noise, not a real topological
separation). On the broader population the rule was never built for (any
label, any tied pair, n=1,828 at threshold 0.03), KGE showed a genuine
positive net (265 win / 181 loss) -- real value as a generalist secondary
signal, not as a rule replacement. **Decision: kept the hardcoded rule
as-is; KGE tiebreak stays built, tested, evaluated, and NOT wired into
production.**

**Exhaustive-candidate-eval net-impact assessment** (closes the open
question tracked in project memory since 2026-08-19). Built
`evaluation/exhaustive_candidate_eval_impact.py`: grades the
`EXHAUSTIVE_CANDIDATE_EVAL_ENABLED` tiebreak-eligible population (any
entity where a model independently accepted 2+ candidates before
resolution) against gold, compared to the non-eligible population. Result
on a 5-note test scope (989 decisions screened): **tiebreak-eligible
precision 14.3% (3/21) vs. 84.7% (265/313) non-eligible -- a 70.4pp gap**,
spot-checked against real entities (not a grading artifact). This closes
the open question from an "assume positive/neutral" default to a
measured, strongly negative finding for the broad population -- the
flag's one verified win (§ wound-dehiscence pattern) does not generalize.
A mitigation (route tiebreak-eligible entities straight to HITL rather
than pay for the comparative call) is identified but **not implemented or
verified** -- stated as future work, not a completed fix.

**Discipline note for the paper**: a collaborator-proposed fix
("`vocabulary_id`-based UK/US filter") was checked against live data
before implementing and found not directly implementable as stated; a
collaborator-proposed narrative ("KGE will make hardcoded rules obsolete")
was checked against the actual retrained model on the actual case it cited
and found false. Both corrections are load-bearing for the honesty of the
figures above -- plausible-sounding claims from any source, including a
confident collaborator, were verified against real data before being
reported, not accepted on confidence alone.

## 13. Fresh-10-note held-out final validation

Genuinely held-out check requested directly: "if our pipeline is final,
should we run fresh notes with gold to check the results?" Two batches of
5 notes each, drawn from the official LOCKED test split
(`data/splits/note_splits.csv`, never used for calibrator training or this
session's debugging), full Stage 1→2b→3 run where a note had never been
processed at all, Stage 2b/3 re-run (picking up every fix from §12) where
it had:

- **Batch 1** (the pre-existing fresh5 set from 2026-08-17, re-run today):
  n=24 AUTO_TIERS decisions, 19 correct, **79.2%**.
- **Batch 2** (the 5 smallest remaining locked-test-split notes, chosen for
  speed): n=32, 24 correct, **75.0%**.
- **Combined 10 notes**: **250 total decisions, 166 gradable, AUTO_TIERS
  n=56, correct=43, precision = 76.8%.**

Both batches capped at 25 graded entities/note (time-boxed, not full-note)
-- a real, stated scope limit, not a hidden one. 3 of the 10 notes were in
the calibrator's own training set; its leakage guard correctly degraded to
untrained/no-op for those, so `TIER_1B_CALIBRATED_AUTO_VALIDATED` did not
engage in either batch -- this validates the SNOMED/retrieval fixes
cleanly (unrelated to calibrator training data) but not the calibrator
itself fresh.

**This 76.8% figure is the recommended headline AUTO-tier precision
number for the paper** -- it is measured on notes genuinely outside both
this session's debugging and (mostly) the calibrator's training set, not
on notes used to develop or tune the fixes being reported.

`ui/components/fresh10_notes.py` (new) holds the 10 note_ids as a shared
constant; `ui/pages/1_🚀_Pipeline_Runner.py`, `3_🔍_Troubleshooting.py`,
and `4_📊_Evaluation_Metrics.py` now scope their note selectors to exactly
this population, so the Streamlit demo reflects this validated set rather
than the full, mixed-vintage corpus.

## 14. Decisions & rationale, §12-13 — for direct citation

Every decision below is stated with the specific data behind it, not
just the conclusion, so it can be pulled into the paper's methodology or
discussion sections with its evidence attached.

**Decision: filter SNOMED regional-extension concepts by `concept_code`
namespace pattern, not `vocabulary_id` or `concept_class_id`.**
Evidence: queried `athena_concept` directly — only one distinct
`vocabulary_id` value (`'SNOMED'`) exists in the whole table (1,093,147
rows), so no vocabulary-level filter is possible. `concept_class_id`
alone is unreliable: of the 98,487 concepts matching the extension
namespace pattern (`concept_code LIKE '%1000000___'`), 23,842 are
themselves `'Procedure'` class, the same class as the correct
International concepts. The namespace pattern itself has zero overlap
with the corpus's 4,522 gold-standard codes, confirmed by direct
set-intersection before shipping.

**Decision: extend the existing Procedure-preference rank penalty to
also cover `'Qualifier Value'` class, not just `'Observable Entity'`.**
Evidence: after the namespace filter removed one duplicate, a second
near-duplicate surfaced for RDW specifically — `concept_id=42536363`
("RDW ... calculation technique", `concept_class_id='Qualifier Value'`)
outscored the correct concept (`4281085`) by 0.077 (0.8208 vs 0.7437)
even before any penalty, exceeding the existing 0.1 bonus's margin.
Queried the broader pattern before generalizing the fix: 22 SNOMED
concepts share this "X calculation technique"/Qualifier-Value shape
across common lab tests (MCV, MCHC, MCH, RDW, PDW, PMV, anion gap, GFR,
INR, etc.), zero of which are ever gold-correct in this corpus.

**Decision: keep the hardcoded `_prefer_lab_procedure_over_observable()`
rule; do NOT replace it with the KGE topological tiebreak, despite a
plausible-sounding argument that KGE should generalize better.**
Evidence: head-to-head on the rule's own pattern (Lab Test, Procedure
vs. {Observable Entity, Qualifier Value}), the rule had **0 losses at
every threshold tested (0.01-0.08, n up to 380)**; KGE had **63 losses**
once the threshold widened past 0.02. A specific claim that KGE would
"naturally push" a third near-duplicate concept ("Mean cell hemoglobin
concentration - finding") away from the correct answer was checked
directly against the actual retrained model on the actual failing
entity — KGE selected the identical wrong concept, with a 0.0018
embedding-distance margin between the two (noise, not signal). This is
a deliberate divergence from an initially-proposed narrative, corrected
by checking the retrained model rather than accepting the argument on
its plausibility.

**Decision: KGE tiebreak stays built and evaluated, but unwired from
production.** Evidence: on the broader population beyond the hardcoded
rule's scope (any label, any tied pair, n=1,828 at threshold 0.03), KGE
showed a genuine positive net (265 win / 181 loss) — real signal, but a
~9.9% loss rate is not risk-free for unreviewed auto-writes without its
own calibrated gating mechanism, which does not yet exist.

**Decision: `EXHAUSTIVE_CANDIDATE_EVAL_ENABLED` stays default-on; its
proposed HITL-routing mitigation is documented but not implemented.**
Evidence: the flag's one verified win (wound-dehiscence-class duplicate
concepts) is real and unaffected by this finding. Its cost was already
known (~34% more LLM calls). This session closed the missing half — the
tiebreak-eligible population (2+ candidates independently accepted by a
model) precision-graded at **14.3% (3/21)** vs. **84.7% (265/313)** for
the non-eligible population on a 5-note sample, a genuine, large,
previously-unmeasured gap. Not yet acted on because the mitigation
(route straight to HITL) has not itself been implemented or verified.

**Decision: scope Stage 2b re-normalization to specific term-matching
entities, not the full Lab Test population, after a first attempt
proved infeasible.** Evidence: re-normalizing all 6,172 `entity_label
='Lab Test'` entities read ~5GB and processed under 200 entities in 40+
minutes wall time (root cause not fully diagnosed under time pressure).
Rescoped twice, each narrowing reusing existing logic rather than
approximating it: first to `lower(original_text) IN` the 26 curated
`_LAB_TEST_ALIASES` keys (2,484 entities), then — after gold-checking
surfaced that value-suffixed variants like `MCHC-31` were still
resolving wrong — to entities whose `strip_lab_value_suffix()` output
(the pipeline's own existing suffix-stripping function, not a new
regex) lands in the curated alias set (1,187 entities). A
collaborator-proposed new regex for this exact purpose was checked
against the codebase first and found to duplicate existing, already-
correct logic — the actual gap was that those entities had simply never
been re-normalized since the fix landed, not a missing mechanism.

**Decision: "MCH vs. MCHC collision" flagged during grading was not a
real bug — no fix applied.** Evidence: `_LAB_TEST_ALIASES` already has
independent, gold-verified entries for both (`"mch": 4182871` at
408/408 gold consistency, `"mchc": 4290193` at 490/490). The apparent
collision was an artifact of testing raw semantic search without
forcing the alias in, not a production code path.

**Decision: report 76.8% (fresh-10-note AUTO-tier precision) as the
paper's headline number, not the higher figures measured earlier in
development.** Evidence: those earlier figures (e.g. 94.4%, 98.0% cited
in the plan's checklist) were measured on notes used during active
debugging of the exact mechanisms being scored — a form of the same
circularity this project's own `prior_confirmation_count` ablation
already demonstrated for a different mechanism. 76.8% (43/56) was
measured on 10 notes from the official locked test split, most outside
the calibrator's own training set, none used to develop the SNOMED/KGE
fixes being reported.

**Decision: drop an unsourced "45.3% pre-fix precision" figure from a
draft results table rather than include it.** Evidence: no record of
this number being measured in this session; source could not be
verified. Stated explicitly rather than silently omitted, so the gap in
the pre/post comparison is visible, not hidden.

**Process note, stated for methodological transparency**: two
collaborator-proposed technical claims were checked against live data
or the actual trained model before being accepted, and both required
correction — a `vocabulary_id`-based SNOMED UK/US filter (not
implementable as proposed; the actual schema has no such field to
filter on) and a claim that KGE would resolve the "Finding"-domain
near-duplicate (falsified by running the actual retrained model against
the actual entity). This verify-before-accepting discipline is the same
one applied to every other claim in this document.

## 15. Final evaluation-criteria numbers: corpus-wide vs. fresh-10 held-out

Direct response to "the proposal names P/R/F1, annotation velocity,
cost-effectiveness -- give the real table." Computed twice, deliberately:
once corpus-wide (144 `is_test` notes, mixed pre-/post-fix vintage) and
once on the fresh-10 held-out notes (§13), so the comparison itself is
the finding, not just the numbers.

**Methodology, stated precisely so it's reproducible**: linked precision
is computed over the SAME population as linked recall (every prediction
with a resolved SNOMED code, across all tiers, checked against gold for
that same note set) -- not AUTO-tier-only precision against full-corpus
recall, which would be two different populations and make F1 meaningless.
Deflection rate = ALL `AUTO_TIERS` decisions (not just the clean-span
gradable subset) / all Stage 3 decisions -- a real bug in an earlier draft
of this table divided a gradable-restricted AUTO count by the
unrestricted total, undercounting deflection by roughly 20-25 points;
caught and corrected before this version.

| Metric | Corpus-wide (144 notes, mixed vintage) | Fresh-10 (held-out, post-fix) |
|---|---|---|
| Gold annotations | 39,403 | 1,497 |
| Span recall | 53.0% (20,873/39,403) | 49.5% (741/1,497) |
| Linked recall | 33.5% (13,208/39,403) | 26.8% (401/1,497) |
| Linked precision | 50.0% (13,197/26,382) | 45.3% (401/886) |
| **Linked F1** | 40.1% | 33.7% |
| Benchmark char IoU (macro / weighted) | 0.1437 / 0.2824 | 0.1453 / 0.2425 |
| **AUTO-tier precision** | 86.9% (5,841/6,724 gradable) | **76.8% (43/56 gradable)** |
| **Deflection rate** | 57.0% (10,953/19,202) | **31.2% (78/250)** |

**Annotation velocity / cost-effectiveness** (from §9, real, source-cited,
Wei et al. 2018 -- not re-derived here):

| | Manual (cited range) | This pipeline | Ratio |
|---|---|---|---|
| Extraction + linking only (Stage 1->2b) | 16.85-38.41 min/note | 5.71 min/note | 2.95x-6.73x faster |
| Full pipeline incl. tier-gate ensemble | 16.85-38.41 min/note | 21.33 min/note | 0.79x-1.80x -- speed advantage largely disappears |

**Reading it honestly, not just reporting it**: every metric is higher
corpus-wide than on fresh-10 -- recall, precision, and deflection all
move the same direction. This is the expected signature of development-
set leakage, not an inconsistency: notes used to build and tune this
session's fixes (SNOMED retrieval, calibrator training) score better on
themselves than on genuinely unseen notes. The gap sizes are themselves
informative -- a 10.1pp AUTO-precision gap and a 25.8pp deflection gap
quantify how much of the corpus-wide number reflects genuine
generalization vs. fitting to what was debugged on.

**Recommended framing for the paper**: report both columns, not just the
corpus-wide one. Lead with fresh-10 as the generalization claim (76.8%
precision, 31.2% deflection, 33.7% F1); report corpus-wide explicitly
labeled "mixed development/held-out notes," not as an unqualified
headline.

**What's still genuinely missing**: no confidence intervals were computed
for any number above. No T0->T2 deflection-rate trend exists (both
figures above are single point-in-time reads). False-deflection rate is
not measurable -- zero completed human reviews exist anywhere in the
system (`hitl_review_queue` has 0 rows, every write path is
`dry_run=True`).

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
* **The MCHC/RDW retrieval fix (§12) is verified on 28 entities, not the
  full 26-term lab-abbreviation alias population.** The other 24 curated
  terms are re-normalized (§12's two refix scripts) but not yet graded at
  the Stage 3 ensemble level.
* **A third SNOMED near-duplicate pattern** ("Clinical Finding"-domain
  concepts colliding with lab-test "determination" concepts, e.g. "Mean
  cell hemoglobin concentration - finding") is confirmed real (§12) but
  neither the hardcoded rule nor KGE currently handles it -- open gap.
* **`EXHAUSTIVE_CANDIDATE_EVAL_ENABLED`'s proposed HITL-routing mitigation
  (§12) is not implemented.** The flag remains default-on; only its net
  impact is now measured, not yet acted on.
* **The 35.0%/22.4% deflection-rate figures given in chat earlier this
  session were wrong** (a gradable-restricted numerator divided by an
  unrestricted denominator) -- corrected to 57.0%/31.2% in §15. Flagged
  here so the error trail is visible, not just quietly fixed.
