# Pipeline Refinement Commit Notes

Living review log for AI-assisted refinement commits.

Use this file after Adam starts the longer iteration/refinement run. The instruction file is `debug/pipeline_refinement_instructions.md`; review both markdown files before each change slice. Once iteration begins, do not edit the instruction file unless Adam explicitly asks. Update this notes file after each AI commit and whenever Adam asks for running notes.

## How To Use

For each commit, append a new entry at the top of `Commit Log` or directly below the most recent entry.

Each entry should let Adam quickly review what changed after an unattended run:

- what was changed
- why it was changed
- acceptance criteria used
- what was expected to run and what was confirmed not to run
- tests and smoke checks run
- smoke timeout decisions and log inspection when a smoke is extended
- storage/cache impact
- resume and force-restart impact
- CLI impact
- any risks, follow-ups, or rollback notes

## Entry Template

```markdown
## YYYY-MM-DD HH:MM - <short_sha> - ai: <commit subject>

Status: accepted | needs follow-up | reverted

Summary:
-

Acceptance Criteria:
-

Expected To Run:
-

Confirmed Not Run:
-

Files/Modules Changed:
-

Validation:
- Pytest:
- Smoke (20 min max unless Adam approves longer):
- Smoke extension to 1 hour:
- Logs inspected:
- Not run:

Resume / Force-Restart Impact:
- Resume behavior:
- Force-restart cleanup:
- Partial-output handling:

Storage/Cache Impact:
- Created:
- Cleaned:
- Persisted:
- Size check:

CLI Impact:
-

Retired Code/Tests:
-

Risks And Follow-Ups:
-

Rollback Notes:
-
```

## Commit Log

## 2026-05-01 01:41 - pending - ai: disconnect analysis CLI selector

Status: accepted

Summary:
- Removed the retired `analysis` stage from the top-level stage-sequence CLI dispatch map.
- Removed `analyse` and `analyze` aliases so analysis is no longer directly runnable through `axon_recon stages ...`.
- Added regression tests that retired analysis selectors fail fast as unsupported stage tokens.

Acceptance Criteria:
- `analysis`, `analyse`, and `analyze` are rejected by `_parse_stage_list_tokens`.
- Active stage and phase selectors still pass the focused CLI test suite.
- The old `stages.analysis` package is not deleted in this slice; only top-level dispatch is disconnected.

Expected To Run:
- Active CLI selectors for preprocess, spikesort, reconstruct, and still-wired direct templates selectors.

Confirmed Not Run:
- `analysis` cannot be selected through the top-level stage-sequence CLI.

Files/Modules Changed:
- `src/axon_recon/pipeline/cli.py`
- `src/axon_recon/pipeline/tests/test_cli_stage_sequence.py`

Validation:
- Pytest: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_cli_stage_sequence.py -q` passed.
- Smoke (20 min max unless Adam approves longer): not run; focused CLI tests covered parser and mocked dispatch behavior.
- Smoke extension to 1 hour: not needed.
- Logs inspected: none.
- Not run: real-data CLI smoke.

Resume / Force-Restart Impact:
- Resume behavior: unchanged.
- Force-restart cleanup: unchanged.
- Partial-output handling: unchanged.

Storage/Cache Impact:
- Created: none.
- Cleaned: none.
- Persisted: none.
- Size check: not applicable.

CLI Impact:
- `analysis`, `analyse`, and `analyze` are no longer supported selectors in `axon_recon stages` / `axon_recon stage`.

Retired Code/Tests:
- Top-level analysis CLI dispatch entry and aliases removed.
- Analysis package and runtime helpers remain for later deletion after a broader reference audit.

Risks And Follow-Ups:
- `run_analysis_from_runtime` and analysis-stage tests still exist and should be removed when the analysis module is fully retired.
- Direct templates selectors remain wired and should be moved/removed in a later templates retirement slice.

Rollback Notes:
- Restore the analysis import, aliases, and `_STAGE_HANDLERS["analysis"]` entry if the old analysis selector is temporarily needed again.

## 2026-05-01 01:39 - fd782c0 - ai: limit all selector to active stages

Status: accepted

Summary:
- Updated the CLI `all` selector so it expands only to active v2 stages: preprocess, spikesort, reconstruct.
- Added an explicit regression test that `templates` and `analysis` are excluded from the canonical `all` order.

Acceptance Criteria:
- `axon_recon stages all` dispatch order is preprocess, spikesort, reconstruct.
- Direct legacy templates/analysis selectors are not removed in this slice and remain a separate retirement follow-up.
- Focused CLI selector tests pass.

Expected To Run:
- `preprocess`, `spikesort`, and `reconstruct` for the `all` selector.

Confirmed Not Run:
- `templates` and `analysis` are no longer included by `all`.

Files/Modules Changed:
- `src/axon_recon/pipeline/cli.py`
- `src/axon_recon/pipeline/tests/test_cli_stage_sequence.py`

Validation:
- Pytest: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_cli_stage_sequence.py -q` passed.
- Smoke (20 min max unless Adam approves longer): not run; parser/dispatch contract change was covered by focused mocked CLI tests.
- Smoke extension to 1 hour: not needed.
- Logs inspected: none.
- Not run: real-data CLI smoke.

Resume / Force-Restart Impact:
- Resume behavior: unchanged.
- Force-restart cleanup: unchanged.
- Partial-output handling: unchanged.

Storage/Cache Impact:
- Created: none.
- Cleaned: none.
- Persisted: none.
- Size check: not applicable.

CLI Impact:
- `all` is now the active-stage pipeline only; direct templates/analysis selectors still exist until later retirement slices.

Retired Code/Tests:
- None; this only removes retired stages from canonical `all` dispatch.

Risks And Follow-Ups:
- Direct `templates.*` and `analysis` CLI handlers remain wired and should be removed in later retirement slices after auditing callers.

Rollback Notes:
- Restore `templates` and `analysis` in `_CANONICAL_STAGE_ORDER` if `all` must temporarily include retired stages again.
