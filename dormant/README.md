# dormant/

Files moved here rather than deleted. Kept instead of deleted in case a finding needs
re-verifying later. Moved with `git mv`, so `git log --follow` on any file here still shows
its full history before the move.

**2026-08-17 batch** (re-established this folder after the 2026-08-14 batch was fully
superseded and removed in commit `3500a64` — see that commit and
`docs/2026-08-14_Dead_Code_Audit.md` for the earlier round). Found via a full import/doc-citation
audit across `src/`, `scripts/`, `evaluation/`, and `ui/` — see
`docs/2026-08-17_Crosswalk_Fix_And_Flywheel_Production_Run.md` for the full audit findings,
including everything that was checked and deliberately kept (most of `evaluation/` and
`scripts/` have zero Python-level imports but are standalone tools cited extensively across
docs, which is not the same as being unused).

- `src/mollm_wholenote_ensemble.py` — self-labeled in its own docstring: "EXPERIMENTAL. Not
  Objective 2, not Objective 3, not wired into src/hitl_queue.py or any production write
  path." Zero Python-level importers and zero references anywhere in `docs/*.md` or the
  active project plan.
- `scripts/analysis/grade_wholenote_results.py` — the grader for the above file's output;
  same zero-reference status, moved as a pair since it has no purpose without it.
- `scripts/inspect_duckdb.py` — generic ad-hoc DB inspection utility, zero references
  anywhere (docs, plan, or other scripts).

**Left in place despite zero imports** (contrast case, so this doesn't get re-flagged later):
`scripts/verify_stage2_fixes.py` (tied to a specific 2026-08-11 fix, has real audit-trail
value in its own docstring even without external citations — borderline, deliberately not
moved this round) and the entire `evaluation/*.py` / `scripts/analysis/*.py` grading/measurement
toolset (zero imports is expected for standalone report scripts; nearly all of them are cited
by name across multiple dated docs as evidence for specific findings).

**Separately flagged, not an archiving matter**: 7 files in `scripts/analysis/` hardcode the
*old* pre-rename path (`/home/ec2-user/clinical_neuro_symbolic_pipeline`, missing the
`_reorder` suffix), which still exists as a stale sibling copy — running them today would
silently read/write against that stale copy instead of erroring. Not fixed as part of this
move; flagged in the same doc above.
