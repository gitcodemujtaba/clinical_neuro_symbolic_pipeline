# dormant/

Files moved here rather than deleted, per `docs/2026-08-14_Dead_Code_Audit.md`. Each was a
one-off diagnostic/hypothesis-test script with no remaining callers or documentation
references, and (unlike the `measure_*`/`diagnose_*` scripts still in `scripts/`) not cited
anywhere as provenance for a tuned constant. Kept instead of deleted in case a finding needs
re-verifying later. Moved with `git mv`, so `git log --follow` on any file here still shows
its full history before the move.

- `scripts/diagnose_glirel.py` — GLiREL diagnostic; conclusion already recorded in
  `src/extraction.py`'s module docstring (GLiREL was abandoned in favor of GLiNER-relex).
- `scripts/check_pluralization_gap.py` — one-time read-only check of whether plural surface
  forms fail Tier 1 exact-match.
- `scripts/test_hyphen_preprocessing_hypothesis.py` — one-time empirical test settling a
  specific hyphen-preprocessing question.

**Not here**: the four zero-byte scaffold files (`evaluation/eval_suite.py`,
`scripts/init_memgraph_snomed.py`, `scripts/init_memgraph_guidelines.py`,
`tests/test_pipeline_integration.py`) that the original audit draft nearly flagged as dead —
they're actually open, tracked TODO items in `docs/Implementation_Checklist.md`, not abandoned
work, and were deliberately left in place.
