# GLiNER Models — Technical Implementation Reference

Deep-dive companion to `docs/Implementation_Methodology.md`'s Stage 2a
section. This pipeline uses **two** independent models from the `gliner`
library, both zero-shot, both running over the same `expanded_text` and
sharing one character-offset coordinate system, but doing genuinely
different jobs:

- **Part 1 — GLiNER-BioMed** (`Ihor/gliner-biomed-large-v1.0`,
  `src/entity_extraction.py`) — zero-shot **span extraction**: the six
  clinical entity types this pipeline's every later stage is built
  around.
- **Part 2 — GLiNER-relex** (`knowledgator/gliner-relex-large-v1.0`,
  `src/extraction.py`) — zero-shot joint **relation extraction**: typed
  edges between two entity spans (e.g. `[Medication] -[treated with]->
  [Condition]`), run independently alongside (not on top of) the
  GLiNER-BioMed pass.

Every claim in both parts is grounded in the real, current source
(`src/entity_extraction.py`, 940 lines; `src/extraction.py`, 386 lines)
and real measured numbers — including a live run of
`scripts/measure_relation_coverage.py` performed while writing Part 2,
not a number quoted from memory or an old log.

---

# Part 1 — GLiNER-BioMed (Zero-Shot Span Extraction)

## 1. What GLiNER-BioMed is and why it replaced the original model

**Model**: `Ihor/gliner-biomed-large-v1.0`
(`GLINER_MODEL_NAME`, `src/entity_extraction.py:79`), loaded via
`GLiNER.from_pretrained()` — a zero-shot named-entity-recognition model:
given a text and an arbitrary list of label strings (not a fixed,
fine-tuned label set), it predicts spans matching each label without
task-specific fine-tuning.

**2026-08-07 swap, recorded in the module's own docstring**
(`src/entity_extraction.py:4-10`): the pipeline originally used
`urchade/gliner_medium-v2.1`, a general-domain checkpoint. It was
replaced with GLiNER-BioMed — a suite of GLiNER checkpoints trained
specifically on biomedical NER benchmarks, including clinical
narratives, by DS4DH / University of Geneva — because a general-domain
model would understate the architecture's real accuracy on the intended
clinical benchmark comparison. This is a zero-shot *model* swap, not a
fine-tuning exercise: the six clinical labels below (§2) were not part of
any task-specific training this project ran — the model brings its own
biomedical pretraining, this project only chooses the label vocabulary
and calibrates a threshold against it.

## 2. Labels and inference configuration

```python
CLINICAL_LABELS = [
    "Condition", "Symptom", "Medication",
    "Procedure", "Anatomy", "Lab Test",
]
```

Six clinical span types (`src/entity_extraction.py:144-151`), passed
directly to `model.predict_entities(text, CLINICAL_LABELS, ...)` —
GLiNER's zero-shot interface accepts arbitrary label strings at call
time; there is no separate fine-tuning step tying the model to this
specific set.

**Flat, not nested, NER** (`FLAT_NER = True`,
`src/entity_extraction.py:112-125`). GLiNER supports both flat
(non-overlapping) and nested (overlapping) span extraction, and this
codebase had — as a bug, not a deliberate choice — two Stage 2a modules
disagreeing about which to use (`entity_extraction.py` defaulted to
flat, `src/extraction.py` passed `flat_ner=False`). Resolved to flat
project-wide, for a structural reason specific to how this pipeline
consumes entities downstream: `extracted_entities` is the canonical
table, and Stage 4 writes one KG3 `:PatientObservation` per entity — a
nested span ("kidney" inside "acute kidney injury") would create two
observations for one clinical fact, and the relation-linking logic in
`src/extraction.py` would have several equally valid overlap targets
with no deduplication policy to resolve them. Nested extraction is
defensible in the abstract; adopting it silently via a library default
without a dedup policy was judged worse than choosing flat deliberately.

## 3. GPU inference, with an explicit fallback

```python
_GLINER_DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
try:
    model = GLiNER.from_pretrained(GLINER_MODEL_NAME, map_location=_GLINER_DEVICE)
except Exception:
    _GLINER_DEVICE = "cpu"
    model = GLiNER.from_pretrained(GLINER_MODEL_NAME, map_location="cpu")
```

(`src/entity_extraction.py:127-142`.) `GLiNER.from_pretrained()`'s own
default is `map_location="cpu"` — silently CPU-only unless told
otherwise. This was caught live: `py-spy` profiling against a running
Stage 1-2b batch showed a single long note's chunked extraction pass
taking **10+ minutes of pure CPU torch inference**, on a box that
already has a real GPU (Tesla T4) the Ollama MoLLM ensemble was already
using at 100% utilization per model (confirmed via `ollama ps`). The
explicit `try/except` fallback exists because GPU memory is shared with
those three concurrently-loaded Ollama models (~7.5GB already in use) —
a CUDA OOM on GLiNER load must degrade to CPU, not crash extraction
outright.

## 4. Confidence threshold — calibrated against real corpus data, not left at the library default

```python
EXTRACTION_THRESHOLD = 0.35
RETAIN_SUBTHRESHOLD = True
SUBTHRESHOLD_FLOOR = 0.35
```

This is one of the more counter-intuitive, evidence-driven decisions in
the whole pipeline, worth walking through in full.

**The library default is 0.5.** A corpus-wide threshold sweep
(`evaluation/cal_eval.py`'s `threshold_sweep()`, run against **21,653**
stored Stage 2a predictions) measured `precision_if_admitted` — of the
spans that would be admitted at a given threshold, what fraction are
actually correct — across the confidence range, and found GLiNER's own
confidence score is **inverted** against real correctness in this
corpus: precision falls *monotonically* from **65.0% at 0.35 down to
50.0% at 0.95**. A model that was well-calibrated would show precision
rising with its own stated confidence; this one shows the opposite. The
sweep also showed this isn't a tradeoff — dropping the threshold to 0.35
strictly dominates on both axes at once: coverage rises from 82.4% to
100% *and* precision rises from 62.4% to 65.0%, simultaneously. There was
no principled reason left to keep the higher threshold.

**Sub-threshold retention, not a simple threshold drop.** Extraction
actually runs at `SUBTHRESHOLD_FLOOR = 0.35` (the same value as
`EXTRACTION_THRESHOLD` currently, but a structurally distinct constant —
see below), and *every* span down to that floor is stored, with anything
below `EXTRACTION_THRESHOLD` flagged `below_threshold=TRUE`
(`src/entity_extraction.py:922`) rather than discarded at extraction
time. The reasoning, from the code's own comment
(`src/entity_extraction.py:94-108`): spans scoring in the 0.35-0.50 band
are exactly the population where a downstream KG-grounded second opinion
(Stage 3) has the most to contribute, and discarding them before Stage 3
ever sees them makes one of the project's own claims — "Stage 3
recovered N entities the extractor nearly discarded" — impossible to
measure. `below_threshold` rows must never reach normalization, KG3, or
evaluation counts by default (`src/clinical_pipeline.py` filters on the
flag before calling Stage 2b) — the flag, not a second table or a
deletion, is the entire enforcement mechanism, following this codebase's
consistent "flag and keep, don't silently drop" convention (the same
pattern used for `superseded_by_split`, `possibly_truncated`, and
`domain_conflict` elsewhere in the pipeline).

**What's still open, stated honestly in the code**: nothing has been
measured *below* 0.35 — `SUBTHRESHOLD_FLOOR` is the actual extraction
floor, so the sweep only has data down to that point. Whether pushing
the floor even lower would keep helping or start flooding the pipeline
with noise is explicitly flagged as unexamined, not assumed either way.

## 5. Sliding-window chunking — the real truncation gap, closed

This is the largest, most technically involved addition to GLiNER's raw
usage in this pipeline.

### 5.1 The problem, confirmed directly against the checkpoint

GLiNER's own word-level tokenizer silently truncates any input over
`model.config.max_len` — **2048 word-tokens for this checkpoint**,
confirmed live rather than assumed. This corpus's notes run up to
24,858 characters (mean ~10,257), long enough to exceed that ceiling —
and every entity past the truncation point is simply never predicted,
with only a `possibly_truncated` flag (added first, chunking deliberately
deferred to a later pass) marking that it happened, not recovering what
was lost.

### 5.2 The pre-check — decide chunked-vs-single before paying for inference

```python
all_words = list(model.data_processor.words_splitter(expanded_text))
if len(all_words) > CHUNK_WORD_BUDGET:
    raw_entities, possibly_truncated, gliner_input_token_count = (
        _extract_entities_chunked(expanded_text, sentences, floor))
else:
    ...
    raw_entities = model.predict_entities(
        expanded_text, CLINICAL_LABELS, threshold=floor, flat_ner=FLAT_NER)
```

(`src/entity_extraction.py:772-789`.) The word count is computed with
**GLiNER's own tokenizer** (`model.data_processor.words_splitter`), not
an approximation via `.split()` or a character-count heuristic — this
matters because the whole point of the pre-check is to match the
model's *real* truncation math, not guess at a characters-to-tokens
conversion ratio. The tokenizer call itself is cheap (no model
inference), so this pre-check is nearly free, deciding chunked-vs-single
*before* ever calling the expensive `predict_entities()` — a deliberate
choice over "try the full call, catch the truncation warning, retry
chunked," which would pay for one wasted full forward pass on every long
note as a matter of course (exactly what the pre-2026-08-16 version did).

`CHUNK_WORD_BUDGET = 1800` — not 2048 — deliberately leaves margin under
the real ceiling for `CHUNK_OVERLAP_WORDS` plus any per-chunk variance
between the tokenizer's word count and whatever internal accounting
GLiNER applies once a chunk is actually run; the margin absorbs that
uncertainty rather than testing the ceiling exactly.

### 5.3 Sentence-boundary-snapped windows, not a raw cut

`_build_chunks()` (`src/entity_extraction.py:258-319`) splits a note
into overlapping `(start_char, end_char)` windows, **always snapped to
sentence boundaries** — never a raw character or word-count cut. This
mirrors the same reasoning `build_local_context()` uses elsewhere in the
same file: a raw cut risks splitting a span mid-entity, or worse,
separating a finding from the negation/qualifier cue that governs its
meaning ("She denies chest pain" cut between "denies" and "chest pain"
would silently flip the entity's real assertion status if a chunk
boundary landed there).

Algorithm, walked through:
1. Compute each sentence's word count using the same tokenizer as the
   pre-check.
2. Greedily accumulate whole sentences into a chunk until adding the
   next sentence would exceed `CHUNK_WORD_BUDGET` (a single sentence
   that alone exceeds the budget is still included whole — an
   unavoidable rare edge case, explicitly logged via
   `possibly_truncated` rather than silently absorbed, not pretended
   away).
3. Back up from the chunk's end by approximately `CHUNK_OVERLAP_WORDS`
   (128), snapped to the nearest sentence boundary, so the next chunk
   starts with that overlapping tail — an entity sitting near a chunk
   boundary is guaranteed to appear complete, with its own
   sentence-level context intact, in at least one chunk.
4. The function is pure (no model/DB access, takes pre-tokenized
   sentences/words as plain data) specifically so it's directly
   unit-testable with synthetic data — see §8.

### 5.4 Running chunks and merging back to global coordinates

`_extract_entities_chunked()` (`src/entity_extraction.py:322-371`) runs
`model.predict_entities()` once per chunk, and is a **drop-in
replacement** for a single whole-note call's return shape — nothing
downstream of the call site (assertion detection, offset mapping back to
the original note, storage) needs to know chunking happened at all. Each
chunk-local span offset is shifted by that chunk's own `chunk_start` to
recover the note's global expanded-text coordinates.

**Deduplication**: an entity extracted identically (same start/end/label)
by two consecutive chunks' overlap region collapses to one row, keeping
the higher-confidence score of the two. This is deliberately *exact-match
only* — the function does not attempt to merge or reconcile two
*different* boundary-adjacent predictions (e.g. a span one chunk sees
starting one word earlier than the other) — that's a harder problem left
out of scope, with downstream normalization's own cache-key
deduplication serving as the existing safety net for near-duplicate
spans that slip through.

### 5.5 Measured result, on the corpus's own longest note

Directly measured against note `11532659-DS-11` (24,858 characters, the
exact note the original truncation-detection comment cited as the
motivating case):

| | Single-call (pre-fix) | Chunked (post-fix) |
|---|---|---|
| Entities found | 124 (confirmed truncated) | **399** |
| Time | 35.8s | 85.6s |

**282 real clinical entities recovered** that the single-pass call
silently dropped — spot-checked, not just counted: pleural effusions, a
right-upper-lobe pneumonia finding, a full lab panel, and several
procedures were among the recovered entities, all past the original
truncation point. The cost (85.6s vs. 35.8s on this one, unusually long
note) was judged acceptable within this project's 2-5 minute/note
end-to-end latency budget.

## 6. Post-extraction false-positive filters — three real, evidence-grounded patterns

GLiNER-BioMed has no concept of "this is document structure, not
clinical content," and no "this text is an administrative placeholder"
signal — three concrete false-positive patterns were found live, sized
against the real 272-note corpus (not assumed to generalize from a
single example), and fixed with narrowly-scoped filters rather than
broad heuristics. All three run *after* extraction, on the entity's
original-text span, immediately before an entity is added to
`processed_entities` (`src/entity_extraction.py:824-829`).

### 6.1 Credential-citation false positive

A physician-credential citation — `"Fax to ___, MD"`, `"Last Verified
___ by ___, MD"` — gets extracted as a `Condition` span and Tier 1
exact-matches straight to the SNOMED concept **"Muscular dystrophy"**
(confirmed live on note `10302979-DS-5`). `_is_credential_citation()`
(`src/entity_extraction.py:200-210`) fires only when the entity text is
exactly `"MD"` **and** immediately preceded, skipping whitespace only,
by a comma — the credential-citation shape specifically, not a bare
"MD" anywhere. Measured against the corpus: the pattern `", MD"` appears
in only 6/272 notes, and every sampled occurrence (7 checked) was
unambiguously a credential citation, never a diagnosis — a broader
`"Fax to"`/`"Attn:"` line-level filter was considered and rejected as
both too rare to justify (1/272 notes) and riskier (a line-based filter
could suppress real clinical content sharing a line with routing text).

### 6.2 Section-header leak

GLiNER has no signal that a line is a *document-structure* header rather
than clinical content. Confirmed live on note `10513485-DS-7`: the
header **"Major Surgical or Invasive Procedure:"** was split word-by-word
into three separate false entities ("Major", "Surgical", "Invasive
Procedure"). Re-testing confirmed this is not a one-off — GLiNER
extracts section headers (whole or fragmented, depending on surrounding
text) across every variant tried, and this pattern is corpus-wide, not
rare: "Major Surgical or Invasive Procedure" alone appears in **272/272
(100%)** of notes, and every standard MIMIC section header checked
(History of Present Illness, Past Medical History, Physical Exam, ...)
appears in 86-100% of notes.

`_is_section_header_text()` (`src/entity_extraction.py:232-237`) reuses
`src.preprocessing.segment_sections()`'s **existing** header-span
provenance (`header_start`, and `start` marking where the body begins)
rather than a hardcoded list of known header strings — the underlying
`SECTION_HEADER_RE` is already a generic "Capitalized Words:" pattern,
so this filter generalizes to any header the note actually has, not just
the one case caught live.

### 6.3 Structured-placeholder false positive ("None")

`"None"` is MIMIC's standard placeholder for an empty section
(`"Major Surgical or Invasive Procedure:\nNone"`,
`"Pending Results:\nNone"`). GLiNER extracts it as a `Condition` entity
(confirmed live, score 0.731). Measured against the corpus: a standalone
`"None"` line appears in **93/272 notes (34%)**, and every sampled
occurrence (10 checked) was a section placeholder, never a genuine
clinical statement. `_is_placeholder_none()`
(`src/entity_extraction.py:251-255`) is scoped to the exact
standalone-token shape (the whole entity text, stripped, case-insensitive)
rather than substring matching — a real sentence merely containing the
word "none" is unaffected, since GLiNER would have to extract "none"
alone as its own labeled span for this filter to fire at all.

## 7. Offset reconciliation — mapping back to the original note text

GLiNER runs on **expanded** text (Stage 1's abbreviation-expanded
version of the note, e.g. "MS" → "multiple sclerosis"), but every
downstream consumer (HITL review, KG3 provenance, evaluation against
gold) needs offsets into the **original** raw note. `map_offsets_to_original()`
(`src/entity_extraction.py:374-397`, internally called "the Time
Machine") walks the list of Stage 1 expansions and, for each one,
computes the character-count shift it introduced, adjusting the
GLiNER-reported span's start/end back through every expansion that
occurred before or around it. This is the single mechanism every
extracted entity's `orig_start`/`orig_end` (the columns everything else
in the pipeline keys off) depends on.

## 8. Testing & reproducibility

- **`tests/test_entity_chunking.py`** — 15 checks, all passing
  (`python3 -m pytest tests/test_entity_chunking.py -v`). Uses an
  AST-extraction technique (parses `_build_chunks()`'s source directly
  out of `src/entity_extraction.py` rather than importing the module)
  specifically because importing `entity_extraction.py` normally loads
  the real, multi-second-cost GLiNER-BioMed model at import time — a
  cost pure chunk-boundary-math tests have no reason to pay. Covers:
  single-chunk-when-under-budget, sentence-boundary snapping, overlap
  sizing, and the single-sentence-exceeds-budget edge case.
- **To exercise the real model**: any `scripts/run_stage3_*.py` or
  pipeline-runner invocation loads `Ihor/gliner-biomed-large-v1.0` at
  import time (`src/entity_extraction.py:127-142`) — expect a real,
  multi-second model load and multi-minute-scale inference cost on long
  notes, per §5.5's measured figures.

## 9. Known blind spot — compensated by a separate module, not by GLiNER itself

Corpus-wide sizing (`src/physexam_shorthand.py`'s own header comment)
found GLiNER-BioMed **never proposes physical-exam telegraphic
shorthand as a candidate at any confidence** — not a threshold problem,
since these spans are never in the prediction list at all.
`"Abd: S/NT/ND"`, bare section-header abbreviations (`HEENT`, `NAD`),
and similar compressed clinical shorthand account for 1,788 gold
annotations in Physical-Exam sections corpus-wide (2.4% of all 75,491
gold annotations), 98.9% of them short (≤25 char) abbreviation-shaped
spans. This is a real, measured gap in GLiNER's own training
distribution for this specific shorthand style — not something a
threshold change can fix, since lowering `EXTRACTION_THRESHOLD` further
only helps spans the model proposes at *some* confidence.

This is deliberately **not** solved inside `entity_extraction.py` — a
separate module, `src/physexam_shorthand.py`, directly injects entities
from an evidence-mined (mined from real gold annotations, not guessed)
text→concept dictionary, applied after GLiNER's own extraction and
skipping any span GLiNER already covers. Full detail is out of this
document's scope (it is not a GLiNER mechanism), but it's the reason
this specific, measured gap doesn't appear as an open problem elsewhere
in this pipeline's numbers.

## 10. Honest limitations

- **Confidence is not well-calibrated** — the threshold-sweep finding
  (§4) that precision falls as GLiNER's own confidence rises is a real,
  measured property of this model on this corpus; it means GLiNER's raw
  score cannot be used as a standalone trust signal anywhere downstream
  without a corpus-specific calibration step (which is exactly why
  Stage 3's own `ConsensusCalibrator` exists as a separate mechanism —
  see `docs/ConsensusCalibrator_Technical_Reference.md`).
- **Nothing is measured below `SUBTHRESHOLD_FLOOR=0.35`** — whether an
  even lower floor would help or flood the pipeline with noise is
  genuinely unknown, not assumed either direction.
- **Chunk-boundary dedup is exact-match only** (§5.4) — two chunks
  proposing *slightly* different boundaries for the same real entity are
  not merged by this mechanism; downstream normalization's cache-key
  dedup is the only safety net for that case, and it hasn't been
  specifically audited for this scenario.
- **The physical-exam shorthand blind spot (§9) is corpus-specific and
  evidence-mined** — the dictionary in `physexam_shorthand.py` covers
  what this project's own gold corpus contains, not a general solution
  to telegraphic clinical shorthand.

---

# Part 2 — GLiNER-relex (Zero-Shot Relation Extraction)

## 11. What GLiNER-relex is, and the model it replaced

**Model**: `knowledgator/gliner-relex-large-v1.0`
(`RELEX_MODEL_NAME`, `src/extraction.py:88`) — a **joint zero-shot
NER+RE model** on the same `gliner` library GLiNER-BioMed uses. "Joint"
matters here: GLiNER-relex runs its own internal (general-domain, not
clinically tuned) entity detection as part of producing relations — it
does not, and per its model card cannot, accept externally-supplied
spans from GLiNER-BioMed as fixed relation endpoints. This is the source
of the real, ongoing trade-off documented in §14 below.

**Module docstring** (`src/extraction.py:1-72`) records a genuine
three-model journey, worth preserving as real project history rather
than compressing away:

1. **Clinical-T5** was originally slated for this role, but removed from
   the live pipeline before this project's evaluation began — it was
   pretrained on MIMIC-III/IV, which is a direct contamination risk
   against this project's own MIMIC-IV-derived evaluation notes. Kept
   only as an external baseline reference, never run in the live
   pipeline.
2. **GLiREL** (`jackboyla/glirel-large-v0`) was tried next, and
   abandoned after a concrete, reproduced failure: it scored near-zero
   even on **its own README example** (documented score 0.9923, actual
   reproduced score 0.0028). Root-caused, not just observed: this
   project's `transformers==4.57.6`/`torch==2.8.0` are far newer than
   what that single, unmaintained "v0" checkpoint was ever validated
   against, and DeBERTa-v3 (GLiREL's backbone) is known to drift
   numerically across `transformers` versions. Downgrading the
   library versions was considered and rejected — too high a risk of
   breaking GLiNER-BioMed/SapBERT, which both depend on the current
   versions elsewhere in the same pipeline.
3. **GLiNER-relex** — the current, live choice — replaced it: actively
   maintained, and reuses the same `gliner` library dependency
   `entity_extraction.py` already needed, adding no new library surface.

**MedGemma 4B was deliberately not used here**, despite being already
committed to Stage 3's MoLLM ensemble. Reasoning stated plainly in the
code: a model that both *generates* a relation in Stage 2a and then
*votes on validating it* in Stage 3 would not be an independent check —
which is the entire basis of the ensemble's consensus-validation claim.
Reusing it here would have quietly undermined that claim.

## 12. Relation vocabulary and plausibility constraints

```python
RELATION_LABELS = [
    "treated with", "indicates", "causes", "located in", "measured by",
]
```

(`src/extraction.py:104-110`.) Five relation labels, passed to
GLiNER-relex at inference time — zero-shot, so this list can be extended
without retraining. `"treated with"`'s head/tail direction was chosen to
match `docs/Implementation_Methodology.md`'s own
`[Medication]-[:TREATED_WITH]->[Condition]` Stage 2a example verbatim.

GLiNER-relex's model card documents no built-in `allowed_head`/
`allowed_tail` type-constraint mechanism, so this project enforces
plausibility as a **post-hoc filter** instead — the same pattern Stage 2b
already applies to concept candidates, applied here to relation pairs:

```python
RELATION_CONSTRAINTS = {
    "treated with": {"allowed_head": ["Medication", "Procedure"], "allowed_tail": ["Condition", "Symptom"]},
    "indicates":    {"allowed_head": ["Lab Test", "Symptom"],     "allowed_tail": ["Condition"]},
    "causes":       {"allowed_head": ["Medication", "Condition"], "allowed_tail": ["Symptom", "Condition"]},
    "located in":   {"allowed_head": ["Condition", "Symptom", "Procedure"], "allowed_tail": ["Anatomy"]},
    "measured by":  {"allowed_head": ["Condition", "Symptom"],    "allowed_tail": ["Lab Test"]},
}
```

`_passes_label_constraint()` (`src/extraction.py:138-148`) rejects a
prediction whose head/tail labels fall outside the allowed set for that
relation type. Deliberately permissive on the unknown case: a relation
label with **no** matching `RELATION_CONSTRAINTS` entry passes through
**unfiltered**, not rejected — so extending `RELATION_LABELS` in the
future without immediately adding a constraint entry doesn't silently
drop every prediction of the new type.

**Confidence floors**, set per the model card's own recommendations
(`src/extraction.py:124-127`): `ENTITY_CONFIDENCE_FLOOR=0.5`,
`ADJACENCY_THRESHOLD=0.5`, `RELATION_CONFIDENCE_FLOOR=0.7` — the
relation floor is meaningfully higher than the entity floor, reflecting
that a wrong *relation* claim is a stronger, more specific error than a
wrong span.

## 13. FLAT_NER shared with GLiNER-BioMed — a real, measured, counter-intuitive result

Originally, `extraction.py` passed `flat_ner=False` (nested) while
`entity_extraction.py` used the library's flat default — a **library-
default accident**, not a considered decision, per the module's own
2026-08-09 docstring entry. Both now import one shared `FLAT_NER`
constant from `entity_extraction.py` (§Part 1 §2's reasoning: one KG3
observation per entity, no overlap-target ambiguity).

**The prediction going in was that this would REDUCE relation yield** —
nested spans give GLiNER-relex more candidate endpoints to relate, so
forcing flat seemed like it could only cost coverage. Measured directly
on note `10000032-DS-21`, it did the **opposite**:

| | Before (nested relex) | After (flat, shared) |
|---|---|---|
| Relations found | 3 | 3 |
| Unlinked endpoints | 2 | 0 |
| Endpoint linking rate | 33% | **100%** |

Same relations found either way — the difference is entirely in the
**linking** step (§14), not the extraction step: when both models
produce flat spans, they agree on span boundaries, so `_overlap_ratio()`
clears `MIN_ENDPOINT_OVERLAP` that it previously missed. Both unresolved
endpoints under the old (nested) setting were relex spans like "right
upper quadrant" with no flat counterpart in `extracted_entities` to link
to. This is recorded in the code specifically because **the prediction
was confidently wrong in a way only real measurement caught** — boundary
*consistency* between the two independent models turned out to matter
more than the extra candidate endpoints nesting nominally provided, the
opposite of the trade-off assumed when the nested setting was first
chosen.

## 14. Endpoint linking — offset overlap, not text matching

GLiNER-relex predicts relations between spans it finds via its own
internal (general-domain) entity detection — it cannot be handed
GLiNER-BioMed's canonical spans directly. An earlier version of this
module stored only the head/tail **text and label** for a relation,
discarding both GLiNER-relex's own character offsets and the entity list
its internal pass returns — which made connecting a relation's endpoint
back to a specific, canonical `extracted_entities` row genuinely hard: a
fuzzy text-similarity match, described at the time as "a bigger, separate
piece of work" and deferred.

**Fixed, 2026-08-08**: that difficulty was self-inflicted, not inherent
— both models run over the exact same `expanded_text`, so their spans
already live in one shared character-offset coordinate system. Linking a
relation endpoint to a canonical entity is a **character-overlap test**,
not a fuzzy heuristic:

```python
def _overlap_ratio(a_start, a_end, b_start, b_end) -> float:
    """Character overlap as a fraction of the SHORTER span."""
    overlap = min(a_end, b_end) - max(a_start, b_start)
    if overlap <= 0:
        return 0.0
    return overlap / max(1, min(a_end - a_start, b_end - b_start))
```

(`src/extraction.py:171-182`.) The **shorter-span denominator** is
deliberate: a canonical entity `"chest pain"` fully contained inside a
wider relex endpoint `"severe chest pain"` should score 1.0, because
both refer to the same real mention — using the union (a Jaccard-style
ratio) would penalize exactly the nesting disagreement the two
independently-trained models are expected to have.

`_link_endpoint()` (`src/extraction.py:185-206`) picks the
`extracted_entities` row with the highest overlap ratio, requiring
`>= MIN_ENDPOINT_OVERLAP = 0.50` (calibration target, per
`docs/MoLLM_Stage3_Retrieval_Design.md` §8) to count as `"linked"`; ties
break deterministically (larger overlap, then lower `entity_id`) so
linking is reproducible across runs, the same determinism discipline
Stage 2b's own retrieval `ORDER BY` fix exists to guarantee elsewhere in
this codebase. Below the threshold, the endpoint is recorded as
`"unresolved_low_overlap"` — **not guessed at**. This honesty is
structural, not incidental: `extracted_relations` stores
`head_link_status`/`tail_link_status` per row, and any consumer
(§15's Channel E included) must check `== "linked"` before trusting an
endpoint's identity — an unresolved endpoint is explicitly passed
through to Stage 3 as ungrounded free text rather than silently dropped
or silently force-matched.

**The trade-off that remains real, not fully closed by the fix above**:
GLiNER-relex still does its own general-domain entity typing internally,
so a relation's `head_entity_label`/`tail_entity_label` as GLiNER-relex
sees them may genuinely differ from what GLiNER-BioMed assigned the same
span. `entity_extraction.py` remains the single source of truth for
`extracted_entities` — GLiNER-relex's own typing is stored
(`head_entity_label`/`tail_entity_label` on `extracted_relations`) but
never overrides the canonical entity's own `entity_label`.

## 15. Where extracted relations actually get used: Channel E of guideline-evidence retrieval

Unlike a diagnostic-only mechanism, GLiNER-relex's output is genuinely
**wired into production retrieval** — `src/retrieval.py`'s
`channel_e_relation()` (`src/retrieval.py:1120-1159`), one of the
guideline-evidence-matching channels available to Stage 3 (behind
`GUIDELINE_EVIDENCE_ENABLED`, off by default corpus-wide — see
`docs/MoLLM_Prompts_And_Reasoning_Technical_Reference.md`).

**The idea**: for an entity with a `linked` GLiNER-relex relation to
another Stage-2-normalized entity, look for a guideline-KG edge that
connects a node reachable from *this* entity's SNOMED code to a node
reachable from the *related* entity's code, via a predicate compatible
with the relex relation label. A guideline rule confirmed from **both**
ends this way is treated as stronger evidence than either entity matched
in isolation — Channels A/B (the other retrieval channels) only ever
look at one entity at a time.

**The relex-label → guideline-predicate bridge**
(`RELEX_LABEL_TO_PREDICATES`, `src/retrieval.py:147-153`):

```python
RELEX_LABEL_TO_PREDICATES = {
    "treated with": {"REQUIRES_MEDICATION", "REQUIRES_INTERVENTION"},
    "indicates":    {"INDICATES", "TRIGGERS_SEVERITY"},
    "causes":       set(),
    "located in":   set(),
    "measured by":  set(),
}
```

Only 2 of the 5 relation labels are mapped to real guideline predicates
— `"causes"`, `"located in"`, `"measured by"` are **deliberately left
empty**, not guessed at, because (as the code's own comment states)
nobody had enumerated the other ~45+ real predicate names in the
guideline corpus to check for a plausible match before this document was
written. `scripts/measure_relation_coverage.py` exists specifically to
close that gap by inspection rather than guesswork.

### 15.1 Real, live-measured numbers (not quoted from an old log)

Run directly against the current production DB while writing this
document (`python3 scripts/measure_relation_coverage.py`, read-only,
safe alongside a concurrent write-locked batch):

**Guideline predicate coverage** — the guideline KG has **52 distinct
predicates across 1,162 rules total**. The top 4 by volume:

| Predicate | Rule count | Mapped? |
|---|---|---|
| `INDICATES` | 469 | ✅ (`"indicates"`) |
| `REQUIRES_INTERVENTION` | 274 | ✅ (`"treated with"`) |
| `TRIGGERS_SEVERITY` | 133 | ✅ (`"indicates"`) |
| `REQUIRES_MEDICATION` | 71 | ✅ (`"treated with"`) |

These four **already-mapped** predicates account for **947 of 1,162
rules (81.5%)** — so although only 4 of 52 distinct predicate *names*
are mapped, the current mapping already covers the large majority of
guideline rule *volume*. The 48 unmapped predicates are individually
small (the largest, `HAS_QUANTITATIVE_THRESHOLD`, is 34 rules; most are
single digits), which is real, useful context for prioritizing whether
completing the mapping is worth the effort — a real, measured answer to
a question the code's own comment left open, not a re-guess.

**Channel E's plausible reach** (81 notes measured, every note with any
stored relation):

| Stage | Count | % of prior stage |
|---|---|---|
| Entities (denominator, for scale) | 17,738 | — |
| Relations extracted | 300 | — |
| Relations with **both** endpoints linked | 146 | 48.7% of relations |
| ...of those, **both** ends SNOMED-anchored (Channel E's actual reach) | **34** | 23.3% of linked relations |

So **34 of 300 extracted relations (11.3%)** clear the bar Channel E
requires *before* the predicate mapping is even checked — a small,
genuinely measured reach, not a large or a negligible one. This is the
honest, current answer to the question the module's own comment posed
("if it's a small percentage, Channel E is a real but minor addition");
34/300 supports "real but minor" as the fair characterization, with the
predicate-mapping completion (§ above) as the more promising next lever
if this channel's reach is ever prioritized further.

## 16. Storage and provenance

`extracted_relations` (`src/extraction.py:227-259`) stores, per relation:
both endpoints' text/label, the relation label, a stable
`relation_id` (`make_relation_id()`, hashed over spans so it survives a
re-run unchanged — the same pattern as `entity_extraction.py`'s
`make_entity_id()`), both endpoints' `entity_id` (when linked),
expanded- **and** original-text offsets for both endpoints (mapped back
through the same `map_offsets_to_original()` "Time Machine" §Part 1 §7
uses, so a relation can always be located in the text a clinician
actually wrote — necessary for the HITL reviewer to inspect it), each
endpoint's `link_status`/`overlap_ratio`, and `relex_model_version` for
the same reason `entity_extraction.py` stamps `gliner_model_version` —
so a row can always be traced to which checkpoint produced it.

A high `unresolved_count` for a note is printed rather than only
recorded silently per row (`src/extraction.py:342-348`) — the reasoning
stated directly in the code: a high unresolved rate is the signal that
the two models are disagreeing about span boundaries more than expected,
and that `MIN_ENDPOINT_OVERLAP=0.50` may need revisiting; that signal
would be invisible if only ever inspected one row at a time.

## 17. Honest limitations (Part 2)

- **No dedicated unit test file exists for `src/extraction.py`** as of
  this writing (confirmed: no `test_extraction*.py` or
  `test_relex*.py` under `tests/`) — unlike GLiNER-BioMed's chunking
  logic (Part 1 §8), the relation-linking math (`_overlap_ratio()`,
  `_link_endpoint()`) has no standalone pure-logic test coverage today.
  This is a real, stated gap, not glossed over.
- **48 of 52 real guideline predicates remain unmapped** to any relex
  label (§15.1) — `"causes"`, `"located in"`, `"measured by"` retrieve
  nothing from Channel E today, by design (empty set, not a guess), but
  also therefore contribute zero reach regardless of how often
  GLiNER-relex predicts them.
- **Channel E's measured reach is small** (34/300 relations, 11.3%,
  §15.1) — real, positive, but a minor contribution relative to Channels
  A-D, and Channel E itself sits behind `GUIDELINE_EVIDENCE_ENABLED`,
  which is off by default corpus-wide.
- **GLiNER-relex's internal entity typing can diverge from GLiNER-
  BioMed's** (§14) — a structural trade-off from using a joint NER+RE
  model that cannot accept externally-supplied spans, not fully closed,
  only worked around via offset-overlap linking plus treating
  GLiNER-BioMed as the sole source of truth for canonical entity labels.
