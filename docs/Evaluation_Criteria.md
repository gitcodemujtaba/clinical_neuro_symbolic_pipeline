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