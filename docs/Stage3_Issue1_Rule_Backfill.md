# Stage 3, Issue 1 — Rule Backfill for High-Frequency Rule-Less Nodes

**Date:** 2026-08-10/11
**Status:** Confirmed on EC2. **Read the correction section first** — it
supersedes the corpus this doc originally worked from.

---

## Stage 2b determinism — confirmed fixed, not a live bug (2026-08-11)

`Proposal_Alignment_Review.md` §7 and `Implementation_Checklist.md` both
flagged Stage 2b normalization as non-deterministic (no `ORDER BY` tiebreak,
dated 2026-08-07). Re-reading `src/normalization.py` in full shows the fix
already landed 2026-08-08 (module docstring, plus explicit `ORDER BY
concept_id ASC` / `ORDER BY similarity DESC, concept_id ASC` tiebreaks on
every tier query in `normalize_entity()`, `_tier_queries()`, and
`_lookup_tier12()`). The docs were just never updated.

That raised a real question: the two `test_stage3_live.py --store` runs
against the new corpus (in the LLM-facing confirmation below) had shown
`Worsening ABD distension and pain` and `Paracentesis` picking up different
`rules retrieved` counts across runs, which looked like live non-determinism.
Investigation: `mollm_decisions.mollm_call_id` is a random `uuid4()` with no
timestamp column and an `ON CONFLICT DO NOTHING` that never fires in
practice — every test run (including ones from before the corpus fix was
deployed) leaves rows behind forever, indistinguishable by time.

Verification: cleared all `is_test=TRUE` rows, then ran
`test_stage3_live.py --limit 10 --store` twice back-to-back on note
`10000032-DS-21`. Every one of the 10 entities returned identical
`snomed_code` and identical retrieved rule IDs in both runs — including
`Worsening ABD distension and pain` (same 2 rules: `INDICATES: Abdominal
pain -> Ectopic pregnancy`, `HAS_ETIOLOGY: Abdominal pain -> Vascular
issue`, both runs) and `Paracentesis` (same 1 rule: `IS_PART_OF: Removal ->
Intravascular access devices`, both runs). Retrieval is deterministic.
Docs corrected accordingly.

Side finding from the same paired run, reinforcing the earlier citation
diagnosis rather than changing it: for `gum bleeding`, BioMistral cited the
hyphen rule_id (`non-st-elevation...`) and OpenBioLLM cited the en-dash
rule_id (`non–st-elevation...`) **identically in both runs** — so that
earlier en-dash observation was a consistent per-model quoting quirk, not
one-off sampling noise (the retraction of the "encoding bug" hypothesis
still stands; it's not a bug at all, just each model reproducing whichever
literal string happened to be in its prompt).

Also newly visible in this run: when 0 rules are retrieved but the routing
mode is `resolution` (candidate-selection), both models still emit a
`cited_evidence` entry with a placeholder rule_id (`'1'`, `RULE_ID_1`) that
`verify_citations()` correctly flags `fabricated_rule_id_not_in_evidence`.
This is the same failure mode as Issue 2 ("make cited_evidence optional
when 0 rules retrieved") — not a new issue, but this run makes the pattern
unambiguous across `edema`, `dysuria`, `melena`, `easy bruising`,
`spirnolactone`.

---

## Confirmed on EC2 (2026-08-11)

`scripts/check_stage3_prerequisites.py`: no blocking issues — all four
required tables populated, Channel B hierarchy resolves on all 5 probe codes,
RxNorm→SNOMED crosswalk has 199,311+22,800+... links, 445/447 guideline codes
resolve in the vocabulary.

`scripts/measure_channel_b_coverage.py` against
`local_triplets_db2_v6_cleaned_grounded_rules_added`: **guideline KG loaded as
1,700 nodes (949 grounded), 1,162 rules** — matches the local build exactly
(1,700/1,162/949), confirming the deployed corpus is the one this doc
describes, not a stale copy.

| | Before (Issue 1 baseline) | After this pass |
|---|---|---|
| Guarded guideline coverage | 8.53% | **28.10%** |
| Direct-code (Channel A), guarded | 10.65% (~10.80% raw) | 10.65% (unchanged — expected, Channel A doesn't depend on rule content) |
| Hierarchy (Channel B), guarded | not separately reported at 8.53% baseline | 17.45% (13,176 annotations) |
| `node_has_no_rules` (of the 142 direct-code-matched concepts) | the ~41-node problem this whole pass targeted | **13 concepts, 62 annotations (0.08% of gold)** |

+19.57pp is well beyond the doc's own +3.25pp estimate for attaching rules to
the 41 named nodes alone — the rest of the lift is the corpus-completeness fix
(merging in the 25 never-grounded files, §"Merge pass" above), which hadn't
been measured as a coverage number anywhere before this.

`scripts/diagnose_guard_suppression.py`'s remaining `node_has_no_rules` sample
(`Emphysema` 17, `Myocarditis` 5, `Bronchitis` 8, `Valvular heart disease` 5,
`Systemic mycoses` 5, `Acute bronchitis` 1) matches this doc's documented
"14 remaining, gold_freq ≤ 17" list exactly — nothing unexpected surfaced.

**One `name_reject` pair worth a second look, not yet acted on:** `Thrombolytic
therapy` (SNOMED FSN) rejected against guideline node `Fibrinolytic therapy` —
these are standard synonyms in clinical usage (both mean clot-dissolving
therapy), so this may be a false rejection from `name_agreement_guard` rather
than a genuine collision, unlike the `Problem`/`Viral disease` pairs in the
same sample which look like correct rejections. Low volume (1 annotation) so
not urgent, but cheap to fix if `_is_acronym_of`/`token_set_ratio` gets a
synonym-pair pass later.

**Next step:** run the LLM-facing checks — `test_stage3_live.py --dry-run`
first (confirms prompts now carry non-empty `EVIDENCE:` sections), then the
live `--limit 10 --store` run to compare against Issue 1's original "10/10
records, 0 rules retrieved, 10/10 HITL_REQUIRED" result.

### LLM-facing confirmation (2026-08-11) — retrieval fixed, a new gate is now binding

`test_stage3_live.py --dry-run --limit 10` then `--limit 10 --store` on the
same note (`10000032-DS-21`) as Issue 1's original diagnostic.

**Retrieval side: fixed, exactly as designed.** Record 1 (`Worsening ABD
distension and pain` → Abdominal pain) retrieved the two rules curated in this
pass directly. Record 2 (`Paracentesis` → Centesis) retrieved the `Removal →
Intravascular access devices` rule via 3-hop hierarchy, unprompted by anything
specific to this pass — the ranking machinery found it on its own. Record 8
(`gum bleeding`) retrieved 5 rules from the NSTE-ACS antiplatelet
bleeding-risk chain, which was already in the base 51-file grounded corpus and
had simply never fired in a live run before. Both models' reasoning text now
engages with real guideline content instead of unaided judgment — a
qualitative improvement in the audit trail even before routing is considered.
`ABSENT`-assertion records (4, 5, 6, 9, 10) correctly skipped retrieval, no
regression there.

**Routing: still 10/10 HITL_REQUIRED** — same raw count as the Issue 1
baseline, but for a materially different reason on records 1 and 2. Both
models agreed (`SUPPORTED`, composite confidence 0.88–0.91) and cited the
*correct* rule_id — not fabricated, unlike records 3/4/9/10's
`fabricated_rule_id_not_in_evidence` pattern (pre-existing, Issue 2 territory,
untouched by this pass). Instead both failed a **different** check:
`quote_not_found_in_source`.

**Root cause, confirmed 2026-08-11 via `scripts/diagnose_citation_quotes.py`
against the raw `mollm_decisions.models` payload — this is model-specific, not
a shared weakness:**

- **BioMistral quotes verbatim, reliably.** On both `Worsening ABD
  distension...` citations, its `quote` matched the source `citation_verbatim_excerpt`
  at containment 1.0 (`verified=True`). Same on `Paracentesis`
  (containment 1.0, modulo one dropped space — vLLM tokenization artifact, not
  a paraphrase).
- **OpenBioLLM does not.** On the same two `Worsening ABD distension...`
  citations, containment was 0.404 and 0.2656 — both far under the 0.8
  threshold. On `Paracentesis` it was 0.3535. In every case the failure has an
  identifiable cause: **OpenBioLLM is quoting the `rationale:` line from the
  prompt, not the `quotable source:` line.** Its `Paracentesis` "quote" —
  `"Prompt removal of intravascular access devices that are a possible source
  of infection is a best-practice recommendation for adults with sepsis or
  septic shock, once alternative vascular access has been established."` — is
  this doc's own curated `rationale` field, copied close to verbatim. The
  `rationale` is written as *my* paraphrase of the guideline, deliberately
  separate from the verbatim `citation`/`citation_verbatim_excerpt` fields —
  so when OpenBioLLM cites it as if it were the quotable source, it is
  citing a real, non-fabricated idea, sourced from the wrong prompt field, and
  `verify_citations()` (correctly) can't tell that apart from a bad paraphrase
  because both fail containment against `citation_verbatim_excerpt` the same
  way.
- Because `verify_citations()` ANDs every model's checks together
  (`citation["citation_verified"] = citation["citation_verified"] and
  other["citation_verified"]`), **BioMistral's perfect citation doesn't save
  the record** — one model quoting the wrong field is enough to force
  `HITL_REQUIRED` even when both models agree on the verdict and one of them
  cited flawlessly. This is arguably correct conservative behavior (the
  ensemble's citation guarantee should be as strong as its weakest member,
  not its best), but it means the citation-fidelity bottleneck is really an
  **OpenBioLLM-specific instruction-following gap**, not a symmetric ensemble
  problem — worth a prompt fix (make the rationale/quotable-source distinction
  more forceful, e.g. drop `rationale` from what's shown to the model, or
  visually set the quotable text apart) rather than a `verify_citations()`
  change.
- **The zero-evidence fabrication pattern (Issue 2) reproduces exactly as
  documented, on both the original and this larger run.** Records 3/4/6/9/10
  and the `melena`/`easy bruising`/`edema`/`gum bleeding`-with-0-rules variants
  all show BioMistral filling `cited_evidence` with a bogus placeholder
  (`"1"`, `"RULE_ID_1"`) or, on `dysuria`, the entity's own OMOP `concept_id`
  (`197684`) misused as if it were a guideline rule_id — a subtly different
  and slightly more dangerous fabrication than a bare placeholder, since 197684
  is a real identifier, just the wrong *kind* of identifier. This is Issue 2's
  own hypothesis (`cited_evidence` being `required` in the schema invites
  filling it even with nothing to cite) reproduced with concrete examples
  rather than the aggregate count originally reported.
- **Note for record 8 (`gum bleeding`) and record 1 (`Worsening ABD
  distension...`):** the earlier hypothesis in this doc about an en-dash/
  hyphen encoding bug does NOT hold up against the raw data — BioMistral
  reproduces the exact en-dash filename correctly in its `rule_id` on the
  `Worsening ABD distension...` record; on `gum bleeding` it was BioMistral
  that used the plain hyphen and OpenBioLLM that used the correct en-dash, the
  reverse of what would be expected from a systematic per-model normalization
  bug. Most likely just token-sampling noise on one call, not a reproducible
  defect — deprioritized unless it recurs.

**Also visible in this data, unrelated to citations:** `Worsening ABD
distension and pain` and `Paracentesis` each appear twice in `mollm_decisions`
(the note was run through `--store` twice) with **different retrieval
results between the two runs** — one run found 2/1 rules, the other found 0
for the same entity text on the same note. This is very likely the
already-documented Stage 2b non-determinism bug (`Proposal_Alignment_Review.md`
§7: `normalize_entity()` has no `ORDER BY` tiebreak, so which OMOP concept a
span resolves to — and therefore which `snomed_code` Channel A/B search on —
can differ run to run). Worth confirming with
`scripts/diagnose_citation_quotes.py`'s query extended to show `retrieved_context.snomed_code`
per call_id; if confirmed, this pass's coverage gain is real but sits on top
of a pre-existing reliability bug that should be fixed before trusting
repeated measurements.

**Net assessment:** this pass did what it set out to do — the KG-grounding
step of Objective 2 is now real for these entities, and BioMistral's
containment=1.0 citations prove the mechanism works end-to-end when a model
quotes correctly. What it surfaced is two *specific*, well-evidenced follow-on
problems rather than a vague "citation fidelity" concern: an OpenBioLLM
instruction-following gap (quoting the wrong prompt field), and the
zero-evidence fabrication pattern Issue 2 already named, now reproduced with
concrete examples including one case (`dysuria`/197684) worth flagging on its
own. Both are sharper, more actionable versions of the same dissertation point
Issue 3 already made about `spirnolactone` — symbolic verification is
load-bearing, and it is now shown catching two distinct, nameable failure
modes rather than one aggregate "6/10 records" number.

---

## Correction (2026-08-11) — found the real grounded corpus

Everything below the divider was written against
`data/local_triplets_db2_v6_cleaned/` because
`data/local_triplets_db2_v6_cleaned_grounded/` did not exist anywhere visible
in this workspace at the time. It turned out to exist at
**`code/data/local_triplets_db2_v6_cleaned_grounded/`** (51 files, 1,183 nodes,
772 rules) — a different, smaller, already-partly-EC2-processed corpus, and the
one Issue 1's "41 nodes" / "1314 nodes, 871 rules" figures actually refer to
(not an exact match on node/rule counts, likely because the corpus moved on
between when Issue 1 was measured and this sync, but close).

Reproducing the same reachable-rules-by-code method against this real corpus:

- **42 rule-less codes with gold_freq > 0** (versus 53 on the wrong corpus) —
  within 1 of the doc's stated 41, and all 6 named examples (`Physical
  examination` 645, `Hypertension` 551, `Clinical evaluation` 345, `Cardiac
  dysfunction` 120, `GERD` 102, `Anemia` 99) match exactly. This is a
  materially better reproduction of Issue 1's own numbers than the first pass.
- Of the 11 source files touched in the first pass, **8 exist in the real
  corpus and 3 do not**: `emergency_severity_index_...chapter_4...part2`,
  `non–st-elevation_acute_coronary_syndromes_chunk_3...`, and
  `acute_heart_failure_syndromes_critical_question_3_nitroglyceri...` were
  never processed by grounding backfill at all. The rules added for those
  three files' nodes (`Abdominal pain`, `Tachycardia`, `Trauma`, `Fracture`,
  `Severe flank pain`, `Ovarian torsion`, `Testicular torsion`, `Audible
  stridor`, `Nonischemic ECG`, `High-flow oxygen`, the new `Endotracheal
  intubation` node) are **not part of this corrected pass** — not because the
  curation was wrong, but because those documents don't exist in the corpus
  Stage 3 actually loads, so the rules would never be reachable regardless of
  content.
- Redid the pass against the real corpus for the 8 files that do exist, and
  added 3 more targets that only became visible once measuring against the
  right corpus: `Cardiovascular Cluster` (113257007, gold_freq 4),
  `Vascular access` (27550009, gold_freq 2), `Acute decompensated heart
  failure` (195111005, gold_freq 3 — weakly grounded, see report; marked
  `citation_type: paraphrase` rather than `verbatim` to be honest about that).
- **Result: 42 → 12 remaining rule-less high-frequency codes**, all at
  gold_freq ≤ 17 (`Emphysema`, `Nonischemic cardiomyopathy`, `Bronchitis`,
  `Myocarditis`, `Valvular heart disease`, `Systemic mycoses`, `Acute systolic
  heart failure`, `Bronchiolitis`, `Lung Transplantation`, `Hypertension`
  [a second, separate 64715009 code from the main 38341003 one], `AATD`,
  `Acute bronchitis`) — documented, not curated, same reasoning as before
  (diminishing returns for the volume they'd recover).
- New output: **`code/data/local_triplets_db2_v6_cleaned_grounded_rules_added/`**
  (30 rules across 8 files, 2 new nodes), plus
  `code/data/rule_backfill_report_20260811_v2_grounded.json`. Same
  non-destructive convention, same verification (0 dangling targets, valid
  JSON, spot-checked through the real `src.retrieval` classes — all 6 checks,
  including the 3 new targets, returned real evidence).

**The original `data/local_triplets_db2_v6_cleaned_rules_added/` output (below)
is superseded.** It's not wrong on its own terms — the rules in it are still
genuine, source-grounded content — but it was built against a corpus that
isn't the one Stage 3 actually loads, so it shouldn't be pointed at in
production. Left in place rather than deleted; treat
`local_triplets_db2_v6_cleaned_grounded_rules_added/` as the one to use.

### New finding: 25 of 76 source files (33%) never went through grounding backfill at all

This is a more foundational gap than "node has no rules" — these documents
have **no grounded nodes whatsoever** in the corpus Stage 3 loads, so any
entity whose only guideline coverage lives in one of them is structurally
unreachable via Channel A/B regardless of rule content. All 7
`emergency_severity_index` files, both `acute_heart_failure` critical-question
files 2 and 3, 5 `community-acquired_pneumonia` files, both `non–st-elevation`
critical-question chunks 3 and 4, all 4 `reperfusion_therapy_for_stemi` files,
2 `ssc-adult-guidelines` files (infection part 1, initial resuscitation),
`kdigo-2012-aki-guideline` chunk 4 (AKI Definition — the chunk with the richest
rule content in the whole corpus, per the earlier Stage 2 investigation), and
2 NPSG files. Full list in the report JSON.

**Quantified:** 100 SNOMED codes are reachable *only* through one of these 25
files (not duplicated elsewhere in the 51 grounded files) — comprising
**1,993 gold annotations, 2.6% of the full 75,491-annotation gold set**.
Grounding these 25 files would very likely move the needle more than the rest
of this Issue-1 pass combined, since it recovers whole documents rather than
individual nodes within already-grounded ones. Worth running
`scripts/backfill_guideline_grounding.py --apply` against these 25 specifically
on EC2 before the next coverage measurement — this wasn't visible without the
real corpus, so it isn't in the original Issue 1 write-up either.

### Merge pass (2026-08-11) — the 25 files didn't need EC2 to add value

Checked before waiting on a DuckDB-backed backfill run: **44% of the 514 nodes
across those 25 files already carry real SNOMED codes** from the original
extraction — they just never went through the merge/backfill step, not that
the codes are missing. That's immediately usable without EC2: Channel D (exact
name match) doesn't even need a code, and Channel A directly benefits from the
44% that already have one.

Copied all 25 files as-is into
`code/data/local_triplets_db2_v6_cleaned_grounded_rules_added/` (now a full
76-file corpus, 1,700 nodes, 1,149 rules after the merge). Re-ran the
reachable-rules-by-code measurement against the merged set and found 14 more
high-frequency rule-less nodes that the earlier 51-file measurement couldn't
even see, because the entities weren't in the corpus at all: `Abdominal pain`
(276 gold), `Nonischemic ECG` (156), `Tachycardia` (95), `Trauma` (63),
`Fracture` (60), `Endotracheal intubation` (50), `High-flow oxygen` (27),
`Severe flank pain` (18), plus `Ovarian torsion`, `Femur fracture`,
`Testicular torsion`, and `Audible stridor` at lower frequency.

Curated real rules for all of these the same way as the rest of this pass —
grounded in the actual `emergency_severity_index_...chapter_4...part2`,
`non–st-elevation_...chunk_3`, and `acute_heart_failure_...chunk_3_nitroglyceri`
chunk text (these are the same rules originally drafted in the "wrong corpus"
first pass; reused after confirming the underlying file content is
byte-identical). 13 more rules, 1 more new node (`Endotracheal intubation`,
same reasoning as before — its existing instances are all
`same_snomed_type_mismatch_not_merged`-flagged).

**Final state: 26 → 14 remaining rule-less high-frequency codes**, all at
gold_freq ≤ 17 (`Emphysema`, `Nonischemic cardiomyopathy`, `Bronchitis`,
`Myocarditis`, `Valvular heart disease`, `Systemic mycoses`, `Acute systolic
heart failure`, `Bronchiolitis`, `Triage acuity level`, `Lung Transplantation`,
`Hypertension` [64715009], `Gastrointestinal issue`, `AATD`, `Acute
bronchitis`) — same documented-not-curated reasoning as before.

**Verification, full 76-file merged corpus:**
- 0 invalid JSON files, 0 dangling rule targets (checked directly, not just
  via the script's own self-check).
- Loaded through the real `src.retrieval.GuidelineIndex`: 1,700 nodes, 1,149
  rules.
- Ran `GroundingRetriever.retrieve()` end-to-end for all 8 of the newly-merged
  high-value entities (`Abdominal pain`, `Nonischemic ECG`, `Tachycardia`,
  `Trauma`, `Fracture`, `Endotracheal intubation`, `High-flow oxygen`,
  `Severe flank pain`) — all 8 returned real evidence.

Report: `code/data/rule_backfill_report_20260811_v3_merge.json`.

**Still not done:** the 289 nodes (56%) in those 25 files that are still `N/A`
would need either the real `scripts/backfill_guideline_grounding.py` run
against a populated DuckDB on EC2, or further manual curation like this pass —
not attempted here since it's a different, larger task (grounding missing
codes, not attaching rules to already-grounded nodes). Not run against real
gold-annotation scoring for the same reason as the rest of this doc.

---

## Original pass (2026-08-10, against the wrong corpus — kept for record)

## What this is

`Stage3_Open_Issues.md` Issue 1 named 41 high-frequency guideline nodes that carry
no rules — including `Physical examination` (645 gold annotations), `Hypertension`
(551), `Clinical evaluation` (345), `Cardiac dysfunction` (120), `GERD` (102),
`Anemia` (99) — and called attaching rules to them "curation work, not code,"
measured at **+3.25pp** coverage. This is that curation pass.

**Environment constraint carried over from Stage 2's gaps doc: no EC2/live-DuckDB
access exists here**, so the "41 nodes" figure could not be reproduced exactly —
that number came from `data/local_triplets_db2_v6_cleaned_grounded/` (the
post-grounding-backfill corpus, EC2-only). Only the pre-grounding
`data/local_triplets_db2_v6_cleaned/` is available locally. Reproducing the same
method (rules reachable through Channel A/B, i.e. excluding nodes flagged
`same_snomed_type_mismatch_not_merged`) against gold-annotation frequency from
`data/evaluaiton-dataset/.../train_annotations.csv` (which **is** available
locally) found **53 rule-less codes with gold_freq > 0** on this corpus. All 6
named examples matched the doc's own gold-frequency numbers exactly
(645/551/345/120/102/99), which is good evidence the two corpora overlap enough
for this list to be a legitimate stand-in.

## What was done

Curated real rules, grounded only in the actual guideline chunk text (
`data/triplets-rules-backup-data/local_chunks_db2_v6/`, the pre-extraction source
documents — not written from memory or general clinical knowledge), for the
**27 highest-frequency** rule-less codes (gold_freq ≥ 13, plus `Cardiac
dysfunction` at 120 which is explicitly named in the doc). Together these 27
account for the large majority of the 53-code list's total gold-annotation
volume.

**Method, per node:**
1. Located every node instance for that SNOMED code across the corpus and
   picked one **not** carrying `quality_flag: same_snomed_type_mismatch_not_merged`
   (that flag makes Channel A/B skip the node entirely — see `retrieval.py`'s
   `_rules_from_nodes`), so the new rule is actually reachable, not just present.
2. Pulled the real source chunk for that node's `provenance.source_document`.
3. Wrote 1–2 rule triples (predicate/rationale/citation/`citation_verbatim_excerpt`)
   using text copied directly from that chunk — every `citation` field below is a
   real quotation, not a paraphrase invented for this pass, unless marked
   otherwise.
4. Linked to an **existing node in the same file** as the rule target wherever
   one fit (referential integrity requires same-file resolution — see
   `GuidelineIndex._load`).
5. Minted 3 new, cleanly-typed nodes only where every existing instance of that
   code was flagged: `Endotracheal intubation` (112798008), `Pulmonary
   hypertension in COPD` (70995007), and a `Levosimendan` (N/A) node needed as a
   second target for `Cardiac dysfunction`. All three are tagged
   `"curated_by": "manual_curation_2026-08-10_stage3_issue1"` for audit.
6. Reused existing corpus predicates (`INDICATES`, `IS_PART_OF`,
   `REQUIRES_MANAGEMENT`, `TRIGGERS_SEVERITY`, `NOT_RECOMMENDED_FOR`, etc.) where
   the fit was right; minted one new predicate, `IS_DIFFERENTIAL_DIAGNOSIS_OF`,
   for the four COPD-differential-diagnosis nodes, since nothing in the existing
   38-predicate vocabulary expressed that relation.

**Result:** 39 new rules attached + 1 rule on a newly-minted node (40 total),
across 11 source files, 3 new nodes. Every added rule carries a
`"curated_by"` provenance tag distinguishing it from the original extraction
pipeline's output.

## Non-destructive output

Per the same convention `clean_local_triplets.py` and
`backfill_guideline_grounding.py` already use: originals in
`data/local_triplets_db2_v6_cleaned/` were never touched. Output is a full copy
at **`data/local_triplets_db2_v6_cleaned_rules_added/`**, plus
`data/rule_backfill_report_20260810.json` (per-file rule/node counts).

## Verification performed

- All 76 files in the output directory parse as valid JSON; node count
  1697→1700, rule count 1119→1159 (exactly accounts for the 40 additions).
- **Zero dangling rule targets** — checked programmatically (every rule's
  `target` resolves to a real `@id` in the same file), same check
  `clean_local_triplets.py` runs after its own edits.
- Loaded the output directory through the **real, unmodified**
  `src.retrieval.GuidelineIndex` (not a reimplementation) and confirmed all 28
  spot-checked codes now show ≥1 reachable rule through `rules_touching()`,
  correctly excluding flagged nodes.
- Ran the actual `src.retrieval.GroundingRetriever.retrieve()` end-to-end for 5
  of the entities (`Physical examination`, `Hypertension`, `Cardiac
  dysfunction`, `GERD`, `Endotracheal intubation`) with a stub vocabulary
  provider. All 5 returned real evidence. Two of them (`Hypertension`, `Cardiac
  dysfunction`) exercised the corpus's existing same-code-different-concept
  collision guard as a side effect — e.g. `Cardiac dysfunction`'s code
  (80891009) is also carried by an unrelated `HEART score 0 to 3` node, and
  `name_agreement_guard` correctly suppressed that one while surfacing the two
  genuinely-relevant rules. This is the guard working as designed, not a defect
  in this pass.

**Not done / explicitly out of scope for this pass:**
- The remaining ~26 lower-frequency codes from the 53-code list (gold_freq < 13:
  `Bacteremia`'s neighbors down through `Acute bronchitis` at freq 1) are
  documented but not curated — diminishing returns for the volume they'd
  recover.
- **Not run on EC2 / against real gold-annotation scoring.** This pass only
  confirms the rules are *structurally reachable*; it doesn't re-measure
  `measure_channel_b_coverage.py` or `score_gold_recall.py`-style linked_recall,
  because those need the live vLLM + DuckDB environment this sandbox doesn't
  have.

## To confirm on EC2

**Superseded — use the corrected corpus instead:**

```
python3 scripts/diagnose_guard_suppression.py --triplets code/data/local_triplets_db2_v6_cleaned_grounded_rules_added
python3 scripts/measure_channel_b_coverage.py --triplets code/data/local_triplets_db2_v6_cleaned_grounded_rules_added
python3 scripts/test_stage3_live.py --limit 10 --store
```

(Original commands below pointed at the wrong-corpus output; kept for record.)

```
python3 scripts/diagnose_guard_suppression.py --triplets data/local_triplets_db2_v6_cleaned_rules_added
python3 scripts/measure_channel_b_coverage.py --triplets data/local_triplets_db2_v6_cleaned_rules_added
python3 scripts/test_stage3_live.py --limit 10 --store
```

Compare `rules retrieved: 0` counts against the Issue 1 baseline (10/10 records,
0 rules each). Expect at least the records touching `Physical examination`,
`Hypertension`, `Clinical evaluation`, `Cardiac dysfunction`, `GERD`, or
`Anemia` to now retrieve real evidence. If this holds up, re-point
`src/retrieval.py`'s `DEFAULT_TRIPLETS_DIR` (or whatever config wraps it) at
`local_triplets_db2_v6_cleaned_rules_added` — after first re-running
`scripts/backfill_guideline_grounding.py` against it, since the EC2-only
`_grounded` corpus this doc's original 41-node count was measured against still
has SNOMED/ICD10 grounding for nodes this pass never saw.
