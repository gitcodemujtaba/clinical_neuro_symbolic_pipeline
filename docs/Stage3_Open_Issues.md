# Stage 3 — Open Issues After First Live Run

**Date:** 2026-08-10
**Status:** Stage 3's core logic is written and proven end-to-end against both live vLLM endpoints. Issues 1–4 concern *quality of the result*; the "Implementation completeness" section near the end lists what is still genuinely unbuilt (pipeline integration, the untested expansion path, production writes, unit tests for the safety rules).

## Run that produced this

```
python3 scripts/test_stage3_live.py --limit 10 --store
```

Note `10000032-DS-21`, 10 records, ensemble = BioMistral-7B-AWQ (:8000) + OpenBioLLM-Llama3-8B-AWQ (:8001).
All 20 model calls used `guided_json_schema`; no fallback to unguided decoding.
Artifacts persisted to `mollm_decisions` with `is_test = TRUE`.

**Outcome: 10/10 routed to `HITL_REQUIRED`.** Every safety rule behaved as designed. The problem is that the system had nothing to be right about.

---

## Issue 1 — Zero guideline rules retrieved in 10/10 records

**Evidence.** Every record printed `rules retrieved: 0 (pooled 0)`, with empty `suppression`. Four records had retrieval skipped entirely and correctly by the assertion gate (`assertion_status=ABSENT, experiencer=PATIENT`); the other six ran channels A/B/C/D and still found nothing.

**Not a loading problem.** The guideline KG loads fine — `1314 nodes, 871 rules` from `data/local_triplets_db2_v6_cleaned_grounded`. The corpus is present and grounded. What is missing is *rules attached to the nodes these entities resolve to*.

**Why it matters.** Objective 2 is KG-grounded validation. With no evidence in the prompt, the ensemble is exercising unaided clinical judgment and the "grounded" claim is untested. Every other number in this document is measured against a system running without its central input.

**Known fix, already scoped.** 41 high-frequency guideline nodes carry no rules: `Physical examination` (645 gold annotations), `Hypertension` (551), `Clinical evaluation` (345), `Cardiac dysfunction` (120), `GERD` (102), `Anemia` (99). Attaching rules to these was measured at **+3.25pp** coverage. This is curation work, not code.

**Also worth checking:** whether the 10 entities in this note fall outside the 8.53% guarded coverage by bad luck or by systematic gap. `measure_channel_b_coverage.py` reports coverage against the gold set; end-to-end coverage against `normalized_entities` has never been measured (pending task #2).

---

## Issue 2 — Citation fabrication in 6/10 records

**Status 2026-08-11: hypothesis tested and rejected — see result below.**
Title/evidence below describe the original 2026-08-10 run; current fabrication
rate on the zero-rule subset is 5/7 (see "Result, 2026-08-11" below).

**Evidence.** Records 3, 4, 6, 8, 9, 10 cited rule ids (`RULE_ID_1`, `1`, `197684`) when the prompt contained `EVIDENCE: NO GUIDELINE EVIDENCE RETRIEVED.` `verify_citations()` caught every one as `fabricated_rule_id_not_in_evidence`.

**The guard works.** This is the hallucination-detection mechanism doing exactly its job, and it is the *only* thing that stopped record 3 (see Issue 3) from being auto-validated.

**Hypothesis worth testing.** `verdict_schema()` lists `cited_evidence` in `required`, so guided decoding forces the model to emit the array. An empty array is legal, but a required field invites filling. **Test:** make `cited_evidence` optional when zero rules were retrieved, re-run the same 10 records, compare fabrication rates.

Either result is publishable: if fabrication drops, structured-output design induces hallucination (a finding about guided decoding, not about these models); if it doesn't, the models fabricate regardless and the guard's necessity is reinforced.

**Do not** constrain `cited_evidence` to known rule ids. `docs/MoLLM_Stage3_Retrieval_Design.md` deliberately leaves it free so fabrication remains *detectable*; making it structurally impossible would remove the signal rather than the problem.

**Implemented 2026-08-11.** `verdict_schema()` (`src/llm_client.py`) now takes
`require_citation: bool`; when `False`, `cited_evidence` stays a legal
property (so a model that has real evidence to offer still can) but is
dropped from `required`. `_query_one()` (`src/mollm_ensemble.py`) takes the
same flag and `validate_record()` sets it to `bool(retrieval.get("rules"))`
for both the first-round call and the post-expansion call (retrieval can go
from empty to non-empty after `expand_evidence()`, so it's recomputed, not
carried over). `verify_citations()` is untouched — a model that still
volunteers a citation with zero rules retrieved is caught exactly as before.
py_compile clean on both files; a stubbed check confirms
`verdict_schema(..., require_citation=False)` produces
`required=['verdict','reasoning','confidence']` with `cited_evidence` still
present in `properties`. Not yet run against live vLLM — no model or DB
access in this environment. **To close the experiment, on EC2:**

```
python3 -c "
import duckdb
conn = duckdb.connect('db/kg2_lexical_store.duckdb')
conn.sql(\"DELETE FROM mollm_decisions WHERE is_test = TRUE\")
"
git pull   # or however the src/ changes get onto the instance
python3 scripts/test_stage3_live.py --limit 10 --store
python3 scripts/diagnose_citation_quotes.py
```

Note the corpus changed between the original run and this one (Issue 1 is
now fixed — see `docs/Stage3_Issue1_Rule_Backfill.md`), so not all 10 records
still retrieve 0 rules; `Worsening ABD distension and pain`, `Paracentesis`
and `gum bleeding` now have real evidence and are not part of this
comparison. The zero-rule subset to compare against is `edema`, `dysuria`,
`melena`, `easy bruising`, `spirnolactone`, `food poisoning`, `orthopnea` (7
of 10) — before the schema change, 5 of those 7 fabricated a citation
(`edema`, `dysuria`, `melena`, `easy bruising`, `spirnolactone`;
`food poisoning` and `orthopnea` correctly returned `citations=0
verified=True`).

**Result, 2026-08-11 — no drop. Guard-necessity outcome.** Re-ran the same
10 records on EC2 with `require_citation=False` deployed (`src/llm_client.py`,
`src/mollm_ensemble.py` uploaded via scp, stale `is_test` rows cleared
first — confirmed via `diagnose_citation_quotes.py`, not just the live-test
summary). Fabrication is unchanged at 5/7, the same five records, the same
placeholder rule_ids (`RULE_ID_1` for `spirnolactone`, `'1'` for `edema`/
`dysuria`/`melena`/`easy bruising`). `orthopnea` and `food poisoning` are
still clean at `citations=0`.

Two earlier re-runs of this same test (before the fix was actually deployed
— `git pull` reported "Already up to date" because the local edits had never
been pushed anywhere) produced byte-identical reasoning text to this run.
That is expected and not a sign this run also failed to deploy:
`TEMPERATURE = 0.0` (`src/llm_client.py`) makes decoding greedy, and under
guided JSON-schema decoding, removing a key from `required` does not stop
the model from choosing to emit that key — it only stops the grammar from
*forcing* it to appear. `cited_evidence` was already a legal property before
this change; the schema was never structurally coercing the model into
filling it. The five fabricating records still choose to volunteer a
citation because that is their preferred continuation, required or not.

**Conclusion.** The required-field hypothesis is rejected. Citation
fabrication under zero retrieved evidence is a genuine property of these two
models' behavior, not an artifact of `verdict_schema()`'s design. This
reinforces `verify_citations()`'s necessity as a symbolic check rather than
removing the need for one — consistent with Issue 3's finding that
confidence and agreement both failed to catch a wrong answer and only the
citation guard held. The `require_citation` plumbing is left in place
(harmless, and `cited_evidence` remains available to any model that does
have something legitimate to cite after an expansion round) but this
specific fix does not reduce HITL volume.

---

## Issue 3 — Ensemble agreement and confidence both failed on a wrong answer

**The case.** Record 3, `spirnolactone` — a misspelling of spironolactone. Stage 2's candidates were SPIRILENE, SPIRGETINE, SPIRAPRILAT: lexically close, pharmacologically unrelated. The correct concept was **not among the candidates at all**, so the only correct verdict was `NONE_CORRECT`, which was in the allowed vocabulary.

Both models returned `RESOLVED_TO_CANDIDATE_3` (SPIRAPRILAT — spirapril, an ACE inhibitor). The note itself settles the true answer: "self-discontinuing lasix and spirnolactone", with abdominal distension and paracentesis, is standard furosemide-plus-spironolactone management of cirrhotic ascites.

BioMistral's reasoning is the revealing part — it identified the drug correctly ("Spirinolactone, which is a diuretic") and then picked SPIRAPRILAT anyway. It knew the answer and still chose from the menu.

**What failed:**

| Signal | Value | Verdict |
|---|---|---|
| `ensemble_agreement` | `True` | Failed — correlated error across different base families |
| `composite_confidence` | 0.916 | Failed — above `AUTO_VALIDATE_THRESHOLD` (0.85) |
| `raw_confidence_label` | HIGH / HIGH | Failed — both self-reported high |
| `citation_verified` | `False` | **Held** — forced `HITL_REQUIRED` |

**Why this matters for the dissertation.** Architectural diversity (Mistral vs Llama-3) did not prevent correlated failure, because the error originated in the *candidate list*, not the models. This is direct evidence for the proposal's argument that LLM consensus is insufficient and symbolic verification is load-bearing.

**Secondary finding — Stage 2 gap.** No edit-distance or typo-tolerant retrieval channel. Misspellings are routine in clinical text and SapBERT's semantic match returns lexical neighbours that are pharmacologically wrong. Worth considering a fuzzy channel, or at minimum measuring how often this occurs.

**Prompt issue.** `NONE_CORRECT` was available and unused in the one case that demanded it. The resolution prompt should make rejecting all candidates a first-class option rather than a listed one.

**Implemented 2026-08-11 — both.**

*Prompt.* `build_prompt()`'s resolution-mode TASK text (`src/mollm_ensemble.py`)
now says explicitly that the candidates came from lexical/semantic similarity,
not a correctness guarantee, and names the exact failure mode observed here:
the nearest-SPELLED match and the nearest-MEANING match can be different
concepts, and the model should return `NONE_CORRECT` rather than pick the
closest-looking wrong one. Previously `NONE_CORRECT` only appeared inside the
`Allowed verdicts:` enum line, with nothing telling the model when to reach
for it.

*Fuzzy channel.* `normalize_entity()` (`src/normalization.py`) now runs a
`levenshtein()`-based supplement (new `_fuzzy_typo_candidates()` /
`_merge_fuzzy()`) whenever Tier 3 is already uncertain — below
`TIER3_SIMILARITY_FLOOR` or margin-ambiguous between its top two — the exact
two states record 3 was in. Deliberately not a fourth tier that runs
unconditionally: edit distance on short clinical tokens is noisy, so it only
widens an already-ambiguous candidate list, capped at
`CANDIDATE_LIMIT * 2` (6), tagged `"4 (Fuzzy)"`, never overrides a confident
Tier 1-3 match. `ambiguity_reason` gets a `_fuzzy_added` suffix when it fires,
so this stays auditable in `tier_trace` rather than silently changing which
candidates show up. py_compile clean; not run against a live DuckDB in this
environment, and `levenshtein()`'s availability on the deployed DuckDB build
(`duckdb>=0.9.0` per requirements.txt) has not been directly confirmed — the
lookup is wrapped in `except duckdb.Error: return []` so a missing/renamed
function degrades to "no fuzzy candidates" rather than breaking
normalization. **To verify on EC2** (read-only, does not touch
`normalized_entities` or re-run any pipeline):

```
python3 -c "
import sys; sys.path.insert(0, 'src')
import duckdb
from normalization import normalize_entity, DB_PATH

conn = duckdb.connect(DB_PATH, read_only=True)
print('levenshtein available:', conn.sql(\"SELECT levenshtein('spirnolactone','spironolactone')\").fetchone())

result = normalize_entity('spirnolactone', conn, gliner_label='Medication')
print('match_tier:', result['match_tier'], 'ambiguous:', result['ambiguous'],
      'reason:', result['ambiguity_reason'])
for c in result['candidates']:
    print(' ', c['match_tier'], c['concept_name'], c['omop_concept_id'], c['similarity_score'])
"
```

Success looks like: the `levenshtein` sanity check returns a small integer
(1 or 2, not an error), `ambiguity_reason` ends in `_fuzzy_added`, and
`Spironolactone` (or its correct OMOP concept name) appears in the printed
candidates tagged `4 (Fuzzy)`. If `levenshtein` errors, that confirms the
function-availability risk above and the fix needs a different DuckDB
built-in (`damerau_levenshtein`/`jaro_winkler_similarity` are the fallbacks
to try). If it succeeds but no `4 (Fuzzy)` row appears, `spironolactone` is
either outside the 2-edit budget from `spirnolactone` (it isn't — one
character) or not marked `standard_concept = 'S'` in this vocabulary, which
would be worth checking directly.

**Confirmed end-to-end on EC2, 2026-08-11.** `levenshtein('spirnolactone',
'spironolactone')` returned `1`. Direct `normalize_entity()` call showed
`spironolactone` (OMOP 970250, score 0.9286) as a `4 (Fuzzy)` candidate
alongside the three wrong Tier-3 ones, `ambiguity_reason =
tier3_top2_margin_below_threshold_fuzzy_added`.

Getting this to actually reach Stage 3 needed re-running Stage 2 for the
note first — `test_stage3_live.py` reads whatever is already persisted in
`normalized_entities`, it does not recompute it, so the fix only shows up
once `normalize_entity()` runs again and overwrites that row. Two wrong
turns finding the right command, worth recording so the next person doesn't
repeat them: `scripts/test_pipeline_e2e.py --note-ids 10000032-DS-21`
defaults to reading `data/raw_notes/gold_notes.csv` (a 272-row gold-only
extract), and `10000032-DS-21` isn't a gold note, so it silently normalized
0 entities — needed `--input data/raw_notes/discharge.csv` to force the
full corpus. Second, re-running Stage 2 on the full note surfaces 131
entities in file order, so `test_stage3_live.py --limit 10` no longer reaches
`spirnolactone` (it's the 15th entity, not the 3rd, once every entity in the
note is scored rather than the original curated 10-record test slice) —
needed `--tier LOW --limit 20` to reach it.

With that, **both models correctly resolved `RESOLVED_TO_CANDIDATE_4`
(spironolactone)**, composite confidence 0.9635 — higher than the original
wrong-but-confident 0.916 that opened this issue. Record still routes to
`HITL_REQUIRED`, but now for the Issue 2 reason (`citation_verification_failed`
— BioMistral cited a fabricated rule_id, `C10000000...`, despite `require_citation=False`
applying here since 0 rules were retrieved) rather than a wrong resolution.
That's the citation guard correctly catching a *different*, independent
problem on an otherwise-correct record — good evidence the two fixes are
doing separate, real work rather than one masking the other.

**Both halves of Issue 3 are confirmed fixed.** The prompt fix's specific
claim (return `NONE_CORRECT` rather than the closest-spelled wrong drug)
was never actually exercised by this record, because the fuzzy channel
made the correct concept available to resolve TO — a better outcome than
`NONE_CORRECT` would have been. The prompt fix's effect is still visible
elsewhere in this run: `edema` (a separate resolution-mode record, no
relation to spirnolactone) returned `NONE_CORRECT` from BioMistral with
`model_disagreement` against OpenBioLLM's `RESOLVED_TO_CANDIDATE_1`, where
prior runs had both models agree on a wrong resolution — consistent with,
though not conclusive proof of (n=1), the prompt change doing its intended
job.

---

## Issue 4 — Confidence signals are not usable as calibrated quantities

**Logprob confidence is flat.** All 20 model responses fell between 0.80 and 0.95 regardless of correctness. The confidently-wrong record 3 (0.916) scored higher than several correct verdicts. `AUTO_VALIDATE_THRESHOLD = 0.85` would admit almost everything, including the wrong answers.

**Self-reported confidence is a per-model constant.** BioMistral returned HIGH on 10/10. OpenBioLLM returned LOW on 7/10, including records where it agreed with BioMistral. This is model-level bias, not a per-record signal — and it vindicates the design decision in `src/llm_client.py` to record `raw_confidence_label` but never route on it. Directly reportable as a calibration finding (`docs/Evaluation_Criteria.md` already plans an ECE/reliability analysis).

**Consequence.** `AUTO_VALIDATE_THRESHOLD` and `MOLLM_RESOLVE_THRESHOLD` are currently unjustifiable numbers. They need a labelled calibration set — but calibrating now would fit them to a system that is about to change once Issue 1 is fixed. **Sequence: fix retrieval, then calibrate.**

**Implemented 2026-08-11 — the tooling, not a threshold fit.** `evaluation/cal_eval.py`
was a 0-line stub (`Implementation_Checklist.md`); it now computes ECE, a
reliability table, a threshold-coverage/precision sweep, and the per-model
`raw_confidence_label` breakdown this issue calls for — but scoped honestly
to what this project's data can actually grade, which is narrower than "all
Stage 3 decisions":

Only **resolution**-mode decisions (`RESOLVED_TO_CANDIDATE_N` /
`NONE_CORRECT`) are graded, by crosswalking the chosen candidate's OMOP
concept to SNOMED (reusing `VocabularyRetriever.snomed_code_for_concept()`,
the same crosswalk `scripts/score_gold_recall.py` uses for Stage 2b's own
pick) and checking it against `train_annotations.csv`'s gold concept_id for
any overlapping span. **Contradiction and non_asserted_check verdicts
(`SUPPORTED`/`CONTRADICTED`/`INSUFFICIENT_EVIDENCE`) are NOT graded** — the
SNOMED challenge gold set has no label for "was this guideline-compliance
judgment correct," because that was never what its annotators were asked.
That needs the human "periodic re-audit sample"
`docs/Evaluation_Criteria.md` already names for this exact reason, and
`cal_eval.py` reports how many such decisions exist per run rather than
silently only covering the gradable subset. Model-disagreement records and
records whose entity has no overlapping gold span are also excluded and
counted separately, not folded into "incorrect."

Joins `normalized_entities` on the same composite key
(`note_id, original_text, expanded_text, gliner_label`)
`score_gold_recall.py` uses rather than `entity_id`, for the same reason
documented in that script's "KNOWN DB CAVEAT" section — `entity_id` joins
can silently return an overwritten row for duplicate-text mentions.

py_compile clean; logic-tested (stubbed `duckdb` module + fake
`VocabularyRetriever`, no live DB in this environment) against: correct and
incorrect `RESOLVED_TO_CANDIDATE_N`, both directions of `NONE_CORRECT`
(correctly rejecting all candidates, and wrongly rejecting a candidate that
was actually right — the `spirnolactone` shape), model disagreement,
out-of-range candidate index, no overlapping gold span, ECE computing ~0.0
on a synthetic well-calibrated set. Not run against the real database or
gold CSV. **To run on EC2:**

```
python3 evaluation/cal_eval.py
python3 evaluation/cal_eval.py --note-ids 10000032-DS-21 --out cal_report.json
```

**Sample-size caveat, stated up front in the script's own docstring and
report footer:** as of 2026-08-11 only a handful of notes have gone through
Stage 3, so whatever ECE/threshold numbers this prints are for validating
the script's methodology and for re-running as the corpus grows — not a
production threshold fit off a single-digit-to-low-double-digit gradable
sample. Real recalibration of `AUTO_VALIDATE_THRESHOLD` /
`MOLLM_RESOLVE_THRESHOLD` should wait for the validation slice
`docs/Evaluation_Criteria.md` describes.

**First real numbers, 2026-08-11 — note `17751158-DS-19` (the smallest gold
note), `--limit 30`.** 11 resolution-mode decisions, 6 gradable (5 excluded
as `model_disagreement`), 17 contradiction-mode decisions correctly excluded
entirely (no gold label for guideline-compliance correctness). n=6 is far
too small to fit a threshold on, but the direction is exactly what this
issue predicted, now with a number attached:

- **Accuracy: 1/6 = 16.7%.**
- **ECE = 0.7529.** All 6 decisions fell in the `[0.9, 1.0)` confidence bin
  (mean confidence 0.9196) against 16.7% actual accuracy — as flat and
  overconfident as the original 10-record run suggested, just quantified.
- **Threshold sweep: coverage stays 100% and precision stays 16.7% at every
  threshold from 0.05 to 0.90.** All 6 decisions cluster too tightly to be
  separated by any threshold in that range. At the current
  `AUTO_VALIDATE_THRESHOLD = 0.85`, all 6 would auto-validate — 5 of them
  wrongly.
- **Per-model raw_confidence_label: both models said HIGH on all 6 gradable
  records, 1/6 correct each** — no LOW labels appeared in this particular
  gradable subset, but the pattern from the original run (self-reported
  confidence not tracking correctness) holds as far as this sample shows.

Also surfaced a small pre-existing inaccuracy in `test_stage3_live.py`,
unrelated to `cal_eval.py` itself: its final `"N artifact(s) written"` line
counts every processed record, but the loop `continue`s past
`store_decision()` when a record errored (JSON parse failure), so 2 of the
30 records this run reported as "written" were never actually persisted.
`cal_eval.py`'s own count (11 + 17 = 28) is the accurate one, caught by its
"every decision must be accounted for" bookkeeping.

---

## Smaller observations

- **Record 4 (`edema`, ABSENT).** The assertion gate correctly skipped retrieval, but the record still went to resolution mode, and BioMistral reasoned that "abdominal distension is consistent with edema." The gate suppressed the evidence but not the question. Consider whether non-asserted records should reach resolution mode at all.
- **Record 2 (`Paracentesis` → `Centesis`).** A tier-2 synonym match scored 1.0 and was marked HIGH, assigning the generic parent (`Centesis`) rather than the specific procedure. BioMistral noticed. Worth checking whether a distinct `Paracentesis` concept exists in the vocabulary and why it lost.
- **Record 9 (`easy bruising`).** BioMistral asserted it "is not a valid clinical concept in the ontology." It is. Confabulation with no evidence to anchor against.

---

## Verified 2026-08-10 — determinism and throughput

Measured with `scripts/profile_stage3.py --limit 5 --runs 2`.

**Determinism: confirmed, at every strictness.** Routing decisions, verdicts, `composite_confidence`, logprobs *and* the reasoning text were bit-identical across repeated runs (5/5 on each). No low-order logprob drift from batched GPU reduction order, which had been the expected failure mode. The "deterministic, traceable" claim in `src/llm_client.py` is now measured rather than asserted, and the ECE/reliability analysis can assume stable inputs.

*Caveat:* measured at one request at a time. Reproducibility under concurrent load is a different question and should be re-checked if the batch runner introduces parallelism.

**Throughput: 5.32s median per record** (both models, sequential), ≈677 records/hour.

| | median | min | max |
|---|---|---|---|
| per record (both models) | 5.32s | 4.70s | 11.95s |
| BioMistral per call | 3.36s | 2.96s | 9.14s |
| OpenBioLLM per call | 2.12s | 1.74s | 2.71s |

The smaller model is the slower one. This is output-token-bound, not size-bound: BioMistral pads `reasoning` with restatements — the same behaviour behind the truncation risk that `parse_json_response()` now salvages.

**Accepted as adequate for this stage.** Two levers exist if that changes:

1. *Concurrent ensemble calls.* `validate_record()` queries the models sequentially, but they are separate vLLM processes with independent GPU allocations and can overlap. Per-record cost would fall from `sum` (5.32s) to roughly `max` (3.36s), ~37%, and would take BioMistral's variance off the critical path.
2. *Cap reasoning length.* Constrain `reasoning` in `verdict_schema()` or lower `MAX_OUTPUT_TOKENS` from 800. Cuts the latency tail and the truncation rate together; `reasoning` informs human review and never the routing decision, so brevity costs nothing.

Note the corpus projection printed by the script is currently meaningless — `normalized_entities` holds 80 rows (one note). Re-run it once more notes are processed.

## Implementation completeness — what is NOT yet done

The four issues above are quality problems. These are genuine implementation gaps, and they should not be confused with them.

- **Stage 3 is not integrated into the pipeline.** `scripts/test_pipeline_e2e.py` contains no reference to `mollm_ensemble`. Stage 3 currently runs only from `scripts/test_stage3_live.py` — a diagnostic tool, one note at a time, gated by `--limit`. There is no batch runner, no resume-after-failure, no cross-note progress tracking. Stage 4 cannot consume Stage 3's output until this exists.
- **`expand_evidence()` has never executed.** No record in the 10-record run printed an `expansion round:` line. The second-round MORE_RULES / SUPPRESSED_RULES escalation is implemented and documented but entirely unexercised. Expected, given Issue 1 — with zero rules retrieved there is nothing to expand — but it means the path is unverified and will need deliberate testing once retrieval produces evidence.
- **`store_decision()` has only ever run with `is_test=TRUE`.** The production write path is unproven, including whether `ON CONFLICT` behaves correctly on re-runs.
- ~~**No unit tests for `mollm_ensemble.py`.**~~ **DONE 2026-08-10.** `tests/test_stage3_safety_rules.py` — 31 tests over `verify_citations()`, `combine()` and `route()`, passing in 0.08s with no models, database or network. Each safety-rule test pairs its condition with a confidence high enough to auto-validate, so a regression that reorders the checks fails there rather than in production. `TestRecord3Regression` encodes the `spirnolactone`/SPIRAPRILAT near-miss, including the counterfactual proving it *would* have auto-validated without the citation check. Verified by mutation: disabling the citation check broke 2 tests, faking a 1.0 confidence broke 1, OR-aggregation broke 3, and letting `CONTRADICTED` auto-validate broke 1.

## Suggested order

1. **Issue 1** — attach rules to the 41 rule-less nodes. Binding constraint; everything else is measured against a crippled system until this is done.
2. **Issue 2** — the `cited_evidence` required-field experiment. Cheap, fast, self-contained finding.
3. **Issue 3** — prompt revision for `NONE_CORRECT`; investigate a typo-tolerant Stage 2 channel.
4. **Issue 4** — build the labelled calibration set and fit the thresholds. Last, because it depends on the others.

## Environment notes for whoever picks this up

- vLLM runs in a **separate Python 3.11 venv** (`/home/ec2-user/vllm-env`); the pipeline stays on 3.9 because scispaCy 0.5.3 pins spacy < 3.8. They communicate over HTTP only.
- Servers do **not** survive an instance stop/start. Every session begins with `bash scripts/start_vllm.sh`.
- MedGemma was dropped: Gemma-3 cannot run on a Tesla T4 under any dtype. See the header of `src/llm_client.py`.
- `--gpu-memory-utilization` fractions are **disjoint shares that must fit in currently-free memory**, and the two servers must start **sequentially**. Both facts cost a debugging session; see the header of `scripts/start_vllm.sh`.
