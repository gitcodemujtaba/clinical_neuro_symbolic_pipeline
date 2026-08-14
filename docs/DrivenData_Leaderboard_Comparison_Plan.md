# Comparing Against the DrivenData SNOMED CT Entity Linking Benchmark Leaderboard

Prepared 2026-08-11 in response to: "can we compare the benchmark against
https://www.drivendata.org/benchmarks/310/benchmark-snomed-ct/leaderboard/ for
at least 25 notes?"

## Current leaderboard (fetched 2026-08-11)

| Rank | Entrant | Model | Macro char IoU | Support-weighted char IoU |
|---|---|---|---|---|
| 1 | adityakadam | Dictionary Backbone with Abstention-Trained Neural Residual | 0.4657 | 0.6165 |
| 2 | STLabs | Residual Linker | 0.4636 | 0.6256 |
| 3 | Reference | SNOMED CT Entity Linking Super Dictionary | 0.4582 | 0.6234 |
| 4 | MITEL-UNIUD | A Two-Stage Pipeline for Linking Clinical Notes to SNOMED CT | 0.4085 | 0.5621 |
| 5 | Reference | FAISS + Qwen3 Instruct | 0.2321 | 0.2160 |

## Two important caveats before treating this as apples-to-apples

**1. Two different "SNOMED CT" datasets are in play.** The leaderboard above
is the *live* DrivenData Benchmark (`benchmarks/310`), scored against a
hidden ~25-note test set nobody outside DrivenData has access to. What this
project actually has locally (`code/data/snomed-ct-entity-linking-challenge-1.2.0/`)
is the *archived* PhysioNet SNOMED CT Entity Linking Challenge — 272 notes,
75,491 annotations, with its own official split preserved in
`train_annotations.csv`'s `annotation_type` column: 204 `train` notes, 68
`test` notes, plus 1,065 supplementary `proposed_ACCEPTED` annotations. These
are related (same annotation effort, same concept scope) but **not the same
test set** — a score computed on the 68 archived-test notes is methodologically
analogous to the leaderboard, not a literal leaderboard submission.
`Implementation_Checklist.md` already flags this distinction and says the
dissertation should state it explicitly.

**2. The scoring metric is already a faithful reimplementation, but the
system it would score today is incomplete.** `scripts/score_gold_recall.py`
re-derives `official_character_iou()` (macro + support-weighted character IoU)
directly from the benchmark's own `scoring.py`, verified against synthetic
cases — a number from it means the same thing as a leaderboard number. But
Stage 3 (the MoLLM guideline-grounding gate — Objective 2, the actual
neuro-symbolic contribution) is **not yet wired into
`scripts/test_pipeline_e2e.py`**. Predictions currently come only from
Stage 1 → 2a (GLiNER-BioMed extraction) → 2b (SapBERT/OMOP normalization).
A 25-note run today is a legitimate baseline number, but it is scoring the
extraction+normalization backbone alone, not the full pipeline the proposal
describes.

## What's already proven at small scale

`score_gold_recall.py` has only ever been run on 3 notes (small/median/large
by gold-annotation count), for Stage 3 diagnostic purposes, not as a
benchmark claim. No run has covered 25+ notes yet.

## Why I can't execute this from here

Running `test_pipeline_e2e.py` on 25 notes requires the populated
`db/kg2_lexical_store.duckdb` (OMOP/Athena concepts, SapBERT embeddings,
grounded guideline KG) and the GLiNER-BioMed model, all of which live on the
EC2 instance referenced throughout the code (`PROJECT_DIR =
"/home/ec2-user/clinical_neuro_symbolic_pipeline"`) — not in this sandboxed
workspace, which only has the raw CSVs. I can prepare everything needed to
run it, but the actual execution needs to happen on that box (or wherever the
populated DB now lives).

## Ready-to-run: full official test split (68 notes, exceeds the 25-note minimum)

All 68 `annotation_type=test` notes from `train_annotations.csv` — the
complete archived-Challenge test split, not a sample of it. 23,131 gold
annotations total (112 to 670 per note). Using the full split rather than a
25-note subset removes any question of whether a smaller sample happened to
favor or disadvantage the pipeline.

```
17739994-DS-31,10960609-DS-16,16345916-DS-17,10043750-DS-6,13227028-DS-12,18752997-DS-9,12962702-DS-14,15531886-DS-12,10371195-DS-9,16867779-DS-14,12991484-DS-15,13164440-DS-18,14631997-DS-13,11161110-DS-13,12314513-DS-16,11180362-DS-10,14719866-DS-37,12128814-DS-15,11576109-DS-15,18059545-DS-17,12247014-DS-9,11783215-DS-18,18570237-DS-10,13016981-DS-27,17348831-DS-20,10848570-DS-12,18046197-DS-13,12626414-DS-32,14102739-DS-16,10839721-DS-9,12884747-DS-9,14280440-DS-8,17692355-DS-9,11296936-DS-66,11997336-DS-3,16867446-DS-13,12970259-DS-4,18454624-DS-16,10965697-DS-12,10912090-DS-33,18847983-DS-6,11515132-DS-10,16393593-DS-5,13602608-DS-6,15259244-DS-19,16991646-DS-11,10794068-DS-18,14975962-DS-19,12304719-DS-18,19297319-DS-11,11134545-DS-21,11215929-DS-4,14055839-DS-15,12545016-DS-17,11021124-DS-15,12549331-DS-3,11838076-DS-20,11652327-DS-14,12612379-DS-23,10860165-DS-24,13272597-DS-24,11649745-DS-4,12018901-DS-68,12050253-DS-20,15853461-DS-4,11392990-DS-6,15807359-DS-28,11532659-DS-11
```

Commands to run on the EC2 box (or current pipeline host) — note this is 68
notes of real length (mean ~10,257 chars, up to 24,858), so `test_pipeline_e2e.py`
will take considerably longer than the previous 3-note runs; no throughput
number exists yet for Stage 1/2a/2b at this scale (only Stage 3's per-record
vLLM throughput has been measured, and Stage 3 isn't in this path):

```bash
python3 scripts/test_pipeline_e2e.py --note-ids 17739994-DS-31,10960609-DS-16,16345916-DS-17,10043750-DS-6,13227028-DS-12,18752997-DS-9,12962702-DS-14,15531886-DS-12,10371195-DS-9,16867779-DS-14,12991484-DS-15,13164440-DS-18,14631997-DS-13,11161110-DS-13,12314513-DS-16,11180362-DS-10,14719866-DS-37,12128814-DS-15,11576109-DS-15,18059545-DS-17,12247014-DS-9,11783215-DS-18,18570237-DS-10,13016981-DS-27,17348831-DS-20,10848570-DS-12,18046197-DS-13,12626414-DS-32,14102739-DS-16,10839721-DS-9,12884747-DS-9,14280440-DS-8,17692355-DS-9,11296936-DS-66,11997336-DS-3,16867446-DS-13,12970259-DS-4,18454624-DS-16,10965697-DS-12,10912090-DS-33,18847983-DS-6,11515132-DS-10,16393593-DS-5,13602608-DS-6,15259244-DS-19,16991646-DS-11,10794068-DS-18,14975962-DS-19,12304719-DS-18,19297319-DS-11,11134545-DS-21,11215929-DS-4,14055839-DS-15,12545016-DS-17,11021124-DS-15,12549331-DS-3,11838076-DS-20,11652327-DS-14,12612379-DS-23,10860165-DS-24,13272597-DS-24,11649745-DS-4,12018901-DS-68,12050253-DS-20,15853461-DS-4,11392990-DS-6,15807359-DS-28,11532659-DS-11

python3 scripts/score_gold_recall.py --note-ids 17739994-DS-31,10960609-DS-16,16345916-DS-17,10043750-DS-6,13227028-DS-12,18752997-DS-9,12962702-DS-14,15531886-DS-12,10371195-DS-9,16867779-DS-14,12991484-DS-15,13164440-DS-18,14631997-DS-13,11161110-DS-13,12314513-DS-16,11180362-DS-10,14719866-DS-37,12128814-DS-15,11576109-DS-15,18059545-DS-17,12247014-DS-9,11783215-DS-18,18570237-DS-10,13016981-DS-27,17348831-DS-20,10848570-DS-12,18046197-DS-13,12626414-DS-32,14102739-DS-16,10839721-DS-9,12884747-DS-9,14280440-DS-8,17692355-DS-9,11296936-DS-66,11997336-DS-3,16867446-DS-13,12970259-DS-4,18454624-DS-16,10965697-DS-12,10912090-DS-33,18847983-DS-6,11515132-DS-10,16393593-DS-5,13602608-DS-6,15259244-DS-19,16991646-DS-11,10794068-DS-18,14975962-DS-19,12304719-DS-18,19297319-DS-11,11134545-DS-21,11215929-DS-4,14055839-DS-15,12545016-DS-17,11021124-DS-15,12549331-DS-3,11838076-DS-20,11652327-DS-14,12612379-DS-23,10860165-DS-24,13272597-DS-24,11649745-DS-4,12018901-DS-68,12050253-DS-20,15853461-DS-4,11392990-DS-6,15807359-DS-28,11532659-DS-11 --out benchmark_68note_report.json
```

If a faster first check is preferred before committing to all 68, a
stratified 25-note subset spanning the same annotation-count range (112–670)
is available on request.

## Reading the result once it exists

Compare `in-scope (excl. Medication)` macro/support-weighted IoU from
`print_official_metrics()`'s output against the leaderboard table above. Two
things to state alongside any number:

- This is Stage 1→2a→2b only (no Stage 3 grounding yet) — a baseline, not
  the full-system claim the proposal makes for Objective 2.
- This is scored on the archived Challenge's 68-note official test split,
  not the live Benchmark's hidden test set — analogous methodology and
  metric, not a literal same-test-set comparison.
