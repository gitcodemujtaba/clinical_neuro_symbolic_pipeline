# RotatE Knowledge-Graph Embedding — Technical Reference

*2026-08-31. Companion to `docs/TransE_KG_Embedding_Technical_Reference.md`. Closes the second of the proposal's two named-but-unbuilt KGE methods (Objective 4) — CompGCN remains deliberately deferred, see §5.*

## 1. Why not the SNOMED vocabulary graph again

TransE trained on the Athena/OMOP relationship graph restricted to this pipeline's own touched concepts (7,269 concepts, ~24,900 edges). Repeating that exact choice for RotatE was explicitly considered and rejected in favor of **repurposed/curated project data** instead. Two real, already-produced artifacts were found and used, plus a third discovered mid-build:

- **`guideline`** — the curated clinical-guideline graph, a second, separate population living on the same Memgraph instance as KG3 (`:GuidelineNode` nodes, real predicate types like `INDICATES`/`REQUIRES_INTERVENTION`/`TRIGGERS_SEVERITY`). Only edges where both endpoints ground to a real SNOMED code are usable; of those, only edges whose SNOMED code further crosswalks to a real, standard OMOP concept are trainable (see §2).
- **`gold`** — this project's own gold-confirmed candidate-competition signal, reusing `gather_tp_records()` (`scripts/build_kg_embeddings.py`, imported not duplicated) and flattening each record into `(correct_concept_id, PREFERRED_OVER, wrong_concept_id)` triples.
- **`combined`** — simple concatenation of the two above.
- **`snomed_is_a`** — added after directly verifying, live, that a fourth real data source exists: the separate KG1 Neo4j instance (`bolt://localhost:7687`, distinct from Memgraph/KG3) holds a fully populated SNOMED IS_A hierarchy (386,110 `:SnomedConcept` nodes, 641,727 real `IS_A` edges). This is the same *category* of data as the vocabulary graph TransE already used (raw ontology structure, not repurposed/curated project output) — kept as an explicit fourth arm to test whether a much larger, purer single-relation-type graph outperforms the two much smaller curated arms, deliberately NOT folded into `combined`.

All four configs are run through **identical** training/evaluation code (`train_rotate()`, `evaluate_link_prediction()`, `evaluate_against_tp_records()`, and the same `evaluation/kg_tiebreak_validation.py` sweep) — only the training triples differ. This is the actual controlled ablation.

## 2. RotatE math and the packing trick

Standard RotatE (Sun et al. 2019): entities as complex vectors, relations as unit-modulus rotations via a learned phase vector (`r = cos(θ) + i·sin(θ)`). Score: `-‖h∘r − t‖` (complex Hadamard product = per-dimension rotation, then L2 norm of the residual).

**Compatibility trick**: entity embeddings are stored as one real-valued `nn.Embedding(n_entities, 2*dim)` — first half real, second half imaginary. This is not a hack invented for this codebase; reference implementations (OpenKE, pykeen) store RotatE this way internally too. It's what lets `evaluate_link_prediction()`, `evaluate_against_tp_records()` (`src/kg_embedding.py`, imported unchanged), and `src/kg_embedding_tiebreak.py`'s two functions work with **zero edits** — all of them only ever call `.score(h,r,t)` or `.entity_emb(idx)`, never assuming anything about the internal packing.

**Geometric caveat, stated up front.** Raw Euclidean L2 over the packed `[re;im]` vector is a *weaker* proxy for "topical closeness" in RotatE than in TransE. TransE's translational geometry makes raw closeness a direct proxy for "connected by something small" (`h+r≈t` implies `h` and `t` are only `r` apart). RotatE's rotational geometry doesn't share that property — two entities connected by a real, well-fit relation can still sit far apart in raw packed coordinates if that relation's phase rotation is large. This is a real, testable hypothesis, not asserted as fact — and turned out to matter (see §4).

**A concrete algebraic consequence, found while writing this module's unit tests.** `src/kg_embedding.py`'s own TransE test checks that an untrained, transitively-implied edge (`0 IS_A 2`, from training edges `0 IS_A 1` and `1 IS_A 2`) scores higher than its reversed, clearly-wrong counterpart. This is a mathematical certainty for TransE once training converges near-exactly: if `e1=e0+r` and `e2=e1+r`, then `score(0,r,2) = -‖r‖` while `score(2,r,0) = -‖3r‖`, always favoring the implied direction. **RotatE has no equivalent guarantee.** Composing the same rotation twice gives `e2 = e0∘r(2θ)`; the implied-vs-reversed score gap becomes `2sin(θ/2)` vs `2sin(3θ/2)` — not monotonic in `θ`, so which one wins depends on where the trained phase angle happens to land. Copying TransE's test verbatim for RotatE would have been checking the wrong property for this architecture; `tests/test_kg_embedding_rotate.py` instead checks the thing RotatE's own margin-ranking loss actually optimizes (a real training triple scores higher than a corrupted one).

## 3. Five deliberate deviations from the TransE code this was adapted from

1. **No unit-norm entity clamp.** Official RotatE leaves entity magnitude meaningful; TransE's `_normalize_entities()` renormalization step is deliberately not carried over.
2. **Entity-norm diagnostic logged every 10 epochs** (mean/max L2 norm), specifically because of (1) — unbounded norm growth under plain hinge loss with no normalization is a known real instability mode; this is an up-front diagnostic, not a reaction to having observed it.
3. **`relation_phase` initialized uniformly in `[-π, π]`** (not Xavier) — avoids a slow-start failure mode where most rotations begin near-identity.
4. **Loss/margin kept identical to TransE** (plain hinge, `margin=1.0`), not RotatE's literature self-adversarial sigmoid loss (`gamma` 9-24). The right call for a controlled TransE-vs-RotatE comparison in this codebase, but it means the resulting MRR is **not comparable to published RotatE benchmarks** — same disclosure discipline as the TransE doc's own RAW-vs-filtered-MRR caveat.
5. **L2 norm order (`p=2`)**, matching TransE's own choice, not the paper's L1 option.

## 4. Results — all four configs, run this session

All four RotatE configs trained end-to-end (`scripts/build_kg_embeddings_rotate.py --config all`), plus a fresh same-session re-run of `evaluation/kg_tiebreak_validation.py` against TransE's existing checkpoint for a fair, same-population comparison. All numbers below are real, from that one run, on the current live corpus (4,843 gradable tied-pair entities) — nothing rounded up or cherry-picked.

### 4.1 Training data yield (real attrition, not idealized counts)

| Config | Raw source size | Real trainable triples | Entities | Relation types |
|---|---|---|---|---|
| `guideline` | 1,144 total Memgraph edges, 355 with both endpoints SNOMED-grounded | **263** (92 more dropped: SNOMED code fails to crosswalk to a standard OMOP concept) | 154 | 26 |
| `gold` | 452 TP records | **1,593** | 145 | 1 (`PREFERRED_OVER`) |
| `combined` | guideline + gold | **1,856** | 297 | 27 |
| `snomed_is_a` | 641,727 raw Neo4j IS_A edges | **530,515** (~17% dropped to crosswalk failure) | 319,557 | 1 (`IS_A`) |
| *TransE (for comparison)* | *SNOMED relationship subgraph, touched concepts* | *24,922* | *7,269* | *104* |

### 4.2 Both evaluations, all four configs plus TransE

| Config | Link-prediction MRR / Hits@10 | Extrinsic: usable records | Extrinsic: frac. correct-closer-than-random |
|---|---|---|---|
| `guideline` | 0.375 / 0.481 (n=27 held-out) | **0 / 452** — near-zero vocabulary overlap with the TP-record population | n/a |
| `gold` | 0.500 / 0.963 (n=160) | 452 / 452 | **72.3%** |
| `combined` | 0.467 / 0.914 (n=186) | 452 / 452 | **84.2%** (best of all 5) |
| `snomed_is_a` | 0.028 / 0.076 (n=2000)† | 422 / 452 | 67.4% |
| *TransE* | *0.776 / 0.909 (n=2000)* | *455 / 455* | *68.9%* |

†`snomed_is_a`'s MRR/Hits@10 is **not comparable** to the other rows — it ranks each held-out triple's true tail against a 319,557-entity candidate pool, vs. hundreds/thousands for every other config (TransE included, at 7,269). A near-zero MRR here reflects the much harder ranking problem, not a categorically worse embedding.

**First real signal, and it's a genuinely positive one for RotatE in isolation**: on the aggregate "does the embedding space separate a real competing-wrong-candidate from an unrelated concept" question, `gold` and `combined` both **beat TransE outright** (72.3% and 84.2% vs. TransE's 68.9%). Taken alone, this would suggest RotatE's rotational geometry captures this project's own disambiguation signal (`gold`'s `PREFERRED_OVER` triples) better than TransE's translational geometry does.

### 4.3 The decisive test: real per-entity tiebreak win/loss, all four configs plus TransE

This is the test that actually matters for whether any of this is usable — not the aggregate signal above, but whether picking a winner by embedding distance helps or hurts on real, individual gold-graded decisions. Full population, `TIE_THRESHOLD=0.03` (SapBERT top1/top2 score gap):

| Config | Resolved | Win | Loss | Net | Win rate of resolved |
|---|---|---|---|---|---|
| `guideline` | 0 | 0 | 0 | 0 | n/a |
| `gold` | 1,076 | 97 | **760** | **−663** | 9.0% |
| `combined` | 1,078 | 97 | **921** | **−824** | 9.0% |
| `snomed_is_a` | 1,543 | 22 | **500** | **−478** | 1.4% |
| *TransE* | *1,791* | *228* | *263* | *−35* | *12.7%* |

**And head-to-head against the existing hardcoded `_prefer_lab_procedure_over_observable()` rule** (§ below explains what it is), on exactly the subset where the rule applies:

| Config | n (rule-applicable) | KGE win | KGE loss | Rule win | Rule loss |
|---|---|---|---|---|---|
| `guideline` | 295 | 0 | 0 | 111 | **0** |
| `gold` | 295 | 0 | **172** | 111 | **0** |
| `combined` | 295 | 0 | **172** | 111 | **0** |
| `snomed_is_a` | 295 | 2 | **0** | 111 | **0** |
| *TransE* | *295* | *110* | *105* | *111* | **0** |

**The full, honest finding, all four configs considered together:**

1. **The aggregate embedding-separation signal (§4.2) does not predict per-entity tiebreak safety (§4.3).** `combined` has the best aggregate score of all five checkpoints (84.2%) and the worst full-population tiebreak net (−824). This is the concrete confirmation of the geometric caveat stated in §2 up front, not a surprise discovered after the fact: raw packed-vector L2 distance is a weaker proxy for "which candidate is actually correct" in RotatE than the aggregate statistic suggests.
2. **`guideline` is simply too small to be useful** — it never resolves a single tied pair in the entire gradable population.
3. **`gold`/`combined` are net-harmful as a tiebreak** — they lose 8–9x more often than they win, and lose 172 times against the hardcoded rule's own subset (where the rule itself has zero losses).
4. **`snomed_is_a` has a distinct, interesting failure profile**: it is nearly **inert** on the hardcoded rule's own narrow subset (0 losses there, but only 2 wins — it essentially doesn't engage with that specific WBC/RBC-style Procedure-vs-Observable-Entity pattern) while being **the single worst performer on the broader population** (1.4% win rate of resolved cases, the lowest of every config or method tried). Bigger and purer (single-relation-type) does not translate into a better tiebreak signal — if anything, it's worse.
5. **RotatE, in every usable configuration, is worse than TransE at this specific task.** TransE's full-population net (−35) and rule-subset losses (105) are both smaller than every RotatE config's corresponding numbers. Combined with TransE's own well-known weakness here (still losing to the rule, 105 vs. 0), **the complete finding across both of the proposal's remaining named KGE methods is: neither beats a 3-line hardcoded class-preference rule, and RotatE is the weaker of the two.**

**A side finding, found while gathering the TransE comparison row, outside this plan's original scope but too material to omit**: TransE's own real-world numbers have **drifted since `docs/TransE_KG_Embedding_Technical_Reference.md`/`docs/Implementation_Methodology.md` were written** — those docs report a net-*positive* full-population result (265 win / 181 loss) and 63 rule-subset losses; the fresh same-session re-run above shows 228 win / 263 loss (net-negative) and 105 rule-subset losses, on a corpus that has grown since those numbers were recorded. The conclusion (don't wire TransE in as a replacement for the rule) is unchanged, but the "real value as a generalist secondary signal" framing in `Implementation_Methodology.md` no longer matches the current data and should be corrected (see §6).

## 5. Sequencing — CompGCN deferred

RotatE (this plan, 4-config ablation) is now complete. CompGCN is a separate, later decision — a substantially bigger lift (message-passing layers, a composition operator, more training infrastructure) than either TransE or RotatE, deliberately not bundled into this plan. If pursued, the same data-source question this plan answers applies again, already resolved by these findings rather than needing to be re-litigated: repurposed/curated project data (gold, guideline) doesn't automatically beat raw ontology data (`snomed_is_a`) or vice versa — both failed the same real test, for different reasons (too small; net-harmful despite good aggregate signal). A future CompGCN attempt should budget for the same two-track evaluation (aggregate signal AND per-entity tiebreak win/loss) rather than trusting the aggregate number alone, given §4.3's finding that the two can point in opposite directions.

## 6. Where else a KGE signal could plausibly help — not built, proposed for a separate follow-up

Given neither TransE nor RotatE works as a hard, greedy re-rank, three lower-risk integration points were considered and are recorded here as backlog, not started:

1. **`ConsensusCalibrator` feature, not a gate** (recommended first, if any of this is pursued further) — feed a KGE tiebreak distance as one more input to the existing logistic-regression calibrator (`src/mollm_tier_calibrator.py`), the same pattern already proven for `kg3_confirmation_count`. Bounded risk: the calibrator can learn to down-weight a noisy signal toward zero rather than being forced to trust it outright, which directly addresses §4.3's finding that `gold`/`combined` have real aggregate signal that a greedy pick can't safely extract. Testable with the existing `evaluation/tier_gate_cal_eval.py` feature-ablation harness — no new validation infrastructure needed.
2. **HITL-queue triage/prioritization signal**, not an auto-decision — where a KGE pick disagrees with SapBERT's top-1, that disagreement correlates with something real (§4.3's non-trivial win+loss counts, not pure noise); could rank human-review order without needing to win a precision fight, since it wouldn't gate a write.
3. **Acronym-escalation note-internal-consistency check** (Stage 1) — `src/acronym_escalation.py`'s existing MoLLM-based expansion picker was measured (2026-08-17) to fail at 34.3%–36.1% precision from a textbook-prior bias (e.g. `LAD`→"left anterior descending artery" regardless of context). A KGE check of which candidate expansion's concept sits closer, in embedding space, to OTHER already-resolved entities in the same note is a structurally different signal (relational context, not a text prior) untried against this specific failure mode.

None of these three is built. Each needs its own honest validation batch before use — this session's 0-for-3 record on "fuse a second ranking signal into an existing decision" (BM25+dense hybrid, guideline-evidence injection, KGE tiebreak) is a real pattern, not bad luck, and argues for treating a positive result as the thing to prove, not assume.
