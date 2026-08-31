# docs/Evaluation_Criteria.md

## Evaluation Data & Leakage Control
* The evaluation uses the DrivenData SNOMED CT Entity Linking Challenge dataset, consisting of 272 discharge notes drawn from MIMIC-IV-Note and annotated by SNOMED International[cite: 7].
* **Test Set:** Approximately 70 notes are locked away and used exclusively for final benchmark evaluation at T0, T1, and T2[cite: 7].
* **Validation Set:** A slice of the approximately 200 training notes is used solely to calibrate the MoLLM's confidence thresholds against empirical accuracy[cite: 7].
* **Active Learning Stream:** The unannotated bulk of MIMIC-IV-Note feeds KG 3 growth[cite: 7]. The 272 DrivenData note IDs are explicitly excluded from this stream to prevent contamination[cite: 7].
* **Baseline Asymmetry:** It is noted that Clinical-T5 may have prior exposure to the raw text of the DrivenData notes through broad pretraining on MIMIC-IV[cite: 7].

## Extraction & Normalization Accuracy
* **Span-level:** Measured via mean Intersection-over-Union at the character level to match the original DrivenData scoring convention[cite: 7].
* **Concept-level:** Precision, recall, and F1 are computed on (span, concept_id) pairs together[cite: 7].
* Results are broken down by entity type and vocabulary, with confidence intervals reported alongside every subgroup[cite: 7].

## Statistical Comparison Method
* Bootstrap confidence intervals are resampled at the note level[cite: 7].
* A paired comparison design is used, where both systems are scored on identical notes[cite: 7].

## MoLLM Gate Validation
* Contradiction-check precision and recall are measured against extractions with known guideline conflicts[cite: 7].
* Resolution accuracy on MOLLM_RESOLVED triples is measured via a periodic re-audit sample[cite: 7].
* Expected Calibration Error (or a reliability diagram) is computed on the validation slice before setting production thresholds, and re-checked at T0, T1, and T2[cite: 7].

## Active Learning / Effort Metrics
* **Deflection rate:** The proportion of extractions reaching KG 3 without human-in-the-loop (HITL) intervention is tracked at each checkpoint[cite: 7].
* **False deflection rate:** The proportion of the Stage 5 re-audit sample that should have gone to HITL but did not is explicitly tracked as a patient-safety metric[cite: 7].

## Success Criteria
* The accuracy claim is supported if the pipeline's concept-level F1 at T2 meets or exceeds Clinical-T5's on the locked test set with non-overlapping bootstrap confidence intervals[cite: 7].
* The effort-reduction claim is supported via a T0 to T2 deflection-rate trend that is statistically distinguishable from flat, alongside a false-deflection rate that stays within a pre-set acceptable bound[cite: 7].

---

## Implementation status (editorial note, added 2026-08-30 — not part of the cited proposal above)

What actually exists today, against each section above, stated plainly:

- **Confidence intervals — partial.** The proposal calls for bootstrap CIs
  resampled at the note level. What's built instead, as of 2026-08-30, is
  a simpler Wilson score interval on the headline binomial proportions
  (AUTO-tier precision, Linked precision/recall, calibrator promotion
  precision) — see `docs/Code_Reference_Stages_And_Metrics.md` §14. Wilson
  treats each graded entity as independent, which is not what "resampled
  at the note level" means and likely understates true uncertainty for
  populations dominated by a few large notes. **Bootstrap CIs are not yet
  built.**
- **T0/T1/T2 checkpoints — not implemented.** Every result in
  `docs/FINAL_RESULTS_Single_Source_Of_Truth.md` is a single point-in-time
  measurement; no trend across checkpoints exists.
- **False deflection rate — real number exists as of 2026-08-31, via a
  gold-substituted proxy, not the proposal's actual metric.** Zero real
  human reviews still exist in production (`hitl_review_queue` is
  populated but unreviewed), so the proposal's independent-re-audit
  design remains unbuildable as specified. What's real instead: a wrong
  AUTO-tier decision (checked against gold, not a human reviewer) is
  exactly what "should have gone to HITL but did not" means, so
  `1 - auto_tier_precision` gives a genuine, Wilson-CI'd number —
  7.9%-23.2% depending on population (`docs/Code_Reference_Stages_And_
  Metrics.md` §15). No "pre-set acceptable bound" was ever defined
  anywhere in this project to check these numbers against — see
  `docs/FINAL_RESULTS_Single_Source_Of_Truth.md`'s Known Limitations.
- **Deflection rate — real, measured.** See §3 above; corpus-wide,
  fresh-10, and fresh-5 numbers all exist and are compared side by side in
  `docs/FINAL_RESULTS_Single_Source_Of_Truth.md` §10.3.
- **Concept-level precision/recall/F1 — real, measured** (as "Linked
  precision/recall/F1" in this project's own naming), same doc, same
  section. No comparison against Clinical-T5 exists — that baseline was
  never run.
- **ECE — real, measured**, but for the superseded single-pass ensemble
  gate (`src/mollm_calibrator.py`), not the current two-step CoT tier gate
  (`src/mollm_tier_gate.py`) — see `docs/ConsensusCalibrator_Technical_Reference.md`'s
  own warning not to confuse the two modules.