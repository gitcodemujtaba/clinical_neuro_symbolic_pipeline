# Archive — Historical Documentation

Moved here 2026-08-27, `git mv` (full history preserved, nothing
deleted). These are point-in-time investigation logs and design docs
whose content has been superseded by the current documentation set in
`docs/` — mostly, every real finding and decision in these files was
mined into `docs/Implementation_Decisions_Log.md`, so nothing here is
"lost," just out of the way of the current, accurate reference set.

**Dated investigation logs** (2026-08-13 through 2026-08-19) — each
documents one session's real findings at the time; superseded as a
*current reference* by `docs/Implementation_Decisions_Log.md` (which
mined every real decision + evidence from all of them into one
document) and `docs/FINAL_RESULTS_Single_Source_Of_Truth.md` (current
numbers). Still useful for the full, unabridged narrative behind any
one specific finding.

**Design/proposal/audit docs, superseded by a current rewrite**:
- `Implementation_Checklist.md` → superseded by `docs/Implementation_Methodology.md`
- `Databases.md` → superseded by the README's Databases section + `Implementation_Methodology.md`
- `Provenance_Schema.md` → superseded by `docs/Provenance_Fields_Technical_Reference.md`
- `MoLLM_Stage3_Retrieval_Design.md` → describes the superseded BioMistral/OpenBioLLM ensemble design; superseded by `docs/MoLLM_Prompts_And_Reasoning_Technical_Reference.md`
- `MoLLM_Redesign_Proposal.md` → the adopted mechanisms are documented in `Implementation_Methodology.md` / the Decisions Log
- `Proposal_Alignment_Review.md` → the 3 gaps it identified are closed; captured in the Decisions Log
- `Guideline_Triplets_KG_Review.md` / `Rules_LLM_Triplets_Review.md` → the contradiction between these two was found and resolved (see Decisions Log §7); superseded by `docs/MoLLM_Prompts_And_Reasoning_Technical_Reference.md` §8.2
- `Stage1_2_Completeness_Audit.md`, `Stage2_Compound_And_Qualifier_Gaps.md`, `Stage3_Issue1_Rule_Backfill.md`, `Stage3_Open_Issues.md` → all findings mined into the Decisions Log; current open items (if any remain) are tracked in `docs/2026-08-20_Session_Results_And_Status.md`'s "Open items carried forward" section instead

For the current, accurate documentation set, see `docs/`'s own README
index (in the main repo `README.md`) rather than anything in this
folder.
