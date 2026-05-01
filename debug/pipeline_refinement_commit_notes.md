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

## 2026-05-01 02:11 - pending - ai: retire direct templates runtime wrappers

Status: accepted

Summary:
- Removed direct `run_templates*_from_runtime` entry points from `pipeline.runner` now that direct templates CLI dispatch is retired.
- Deleted the unused `stages/templates/cli.py` module.
- Deleted pipeline-level tests that only protected direct templates runtime wrapper behavior.

Acceptance Criteria:
- No direct `run_templates*_from_runtime` or `_run_templates_substage_from_runtime` symbols remain under `src/axon_recon`.
- No `register_templates_subparser` or `stages.templates.cli` references remain under `src/axon_recon`.
- Reconstruct embedded template phase wrappers continue to pass focused tests.
- Pipeline tests collect without stale direct templates wrapper imports.

Expected To Run:
- Reconstruct-owned template-related phases through `reconstruct.<phase>` selectors/wrappers.

Confirmed Not Run:
- Direct templates runtime wrappers and direct templates CLI module are removed.

Files/Modules Changed:
- `src/axon_recon/pipeline/runner.py`
- `src/axon_recon/pipeline/stages/templates/cli.py`
- `src/axon_recon/pipeline/tests/test_templates_target_status.py`

Validation:
- Pytest: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_reconstruct_target_status.py src/axon_recon/pipeline/tests/test_cli_stage_sequence.py -q` passed.
- Pytest collection: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest --collect-only src/axon_recon/pipeline -q` passed collection.
- Reference audit: no `run_templates*_from_runtime`, `_run_templates_substage_from_runtime`, `register_templates_subparser`, or `stages.templates.cli` matches remain under `src/axon_recon/**`.
- Diagnostics: no VS Code/Pylance errors in `pipeline/runner.py` or `pipeline/cli.py`.
- Smoke (20 min max unless Adam approves longer): not run; this removed retired direct runtime entry points while preserving mocked reconstruct dispatch coverage.
- Smoke extension to 1 hour: not needed.
- Logs inspected: none.
- Not run: real-data smoke.

Resume / Force-Restart Impact:
- Resume behavior: unchanged for active reconstruct path; retired direct templates runtime resume behavior is removed.
- Force-restart cleanup: unchanged for active reconstruct path; retired direct templates runtime force-restart behavior is removed.
- Partial-output handling: unchanged for active reconstruct path.

Storage/Cache Impact:
- Created: none.
- Cleaned: removed direct templates runtime wrapper code/tests.
- Persisted: none.
- Size check: not applicable.

CLI Impact:
- Direct templates CLI module is deleted; top-level direct templates selectors were already unsupported.

Retired Code/Tests:
- Retired direct templates runtime wrappers, unused templates CLI module, and direct-wrapper target-status tests.

Risks And Follow-Ups:
- `stages.templates` core/config/models/tests remain because reconstruct still depends on them; next templates work should migrate those imports into reconstruct before deleting the package.

Rollback Notes:
- Restore the deleted wrappers, `stages/templates/cli.py`, and `test_templates_target_status.py` if direct templates runtime dispatch must be temporarily recovered.

## 2026-05-01 02:04 - 505f662 - ai: rename reconstruct template cli handlers

Status: accepted

Summary:
- Renamed reconstruct CLI handler functions for embedded template phases from `_run_templates_*_from_args` to reconstruct-owned `_run_reconstruct_*_from_args` names.
- Updated the top-level CLI imports and handler registry to use the reconstruct-owned handler names directly.
- Kept accepted CLI selectors unchanged.

Acceptance Criteria:
- `src/axon_recon/pipeline/stages/reconstruct/cli.py` no longer exposes `_run_templates_*_from_args` handlers.
- `src/axon_recon/pipeline/cli.py` no longer imports `_run_reconstruct_templates_*` aliases for preferred reconstruct phase selectors.
- Existing CLI selector tests still pass.

Expected To Run:
- Preferred reconstruct phase selectors such as `reconstruct.analyzers`, `reconstruct.build_templates`, and `reconstruct.reports`.

Confirmed Not Run:
- No direct templates CLI selector support was reintroduced.

Files/Modules Changed:
- `src/axon_recon/pipeline/cli.py`
- `src/axon_recon/pipeline/stages/reconstruct/cli.py`

Validation:
- Pytest: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_cli_stage_sequence.py -q` passed.
- Diagnostics: no VS Code/Pylance errors in changed files.
- Reference audit: no `_run_templates_` matches remain in `stages/reconstruct/cli.py`; no `_run_reconstruct_templates_` aliases remain in `pipeline/cli.py`.
- Smoke (20 min max unless Adam approves longer): not run; this was a CLI handler rename with mocked selector coverage.
- Smoke extension to 1 hour: not needed.
- Logs inspected: none.
- Not run: real-data smoke.

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
- Accepted selector contract is unchanged; internal handler names now match reconstruct ownership.

Retired Code/Tests:
- Retired legacy reconstruct CLI handler names that implied direct templates ownership.

Risks And Follow-Ups:
- Runtime wrapper/function names still include `templates` while the underlying templates package remains to be migrated.

Rollback Notes:
- Restore the old handler function names and import aliases if an external import unexpectedly relied on private CLI helpers.

## 2026-05-01 02:01 - 43151f3 - ai: align reconstruct template phase names

Status: accepted

Summary:
- Changed reconstruct embedded-template runtime wrappers to report preferred `reconstruct.<phase>` stage names instead of legacy `reconstruct.templates_*` names.
- Added target-status coverage for the embedded template wrappers so result/progress stage names match the CLI selectors.
- Left direct templates runtime wrappers untouched for now because templates migration/deletion is not complete.

Acceptance Criteria:
- `run_reconstruct_templates_*_from_runtime` wrappers return aggregate stages such as `reconstruct.analyzers`, `reconstruct.build_templates`, and `reconstruct.reports`.
- `pipeline.runner` no longer contains `reconstruct.templates_` stage-name strings.
- Existing preferred CLI selector tests still pass.

Expected To Run:
- Preferred reconstruct template-related phase selectors and runtime wrappers.

Confirmed Not Run:
- Legacy `reconstruct.templates_*` stage names are no longer emitted by reconstruct runtime wrappers.

Files/Modules Changed:
- `src/axon_recon/pipeline/runner.py`
- `src/axon_recon/pipeline/tests/test_reconstruct_target_status.py`

Validation:
- Pytest: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_reconstruct_target_status.py src/axon_recon/pipeline/tests/test_cli_stage_sequence.py -q` passed.
- Diagnostics: no VS Code/Pylance errors in changed files.
- Reference audit: no `reconstruct.templates_` matches remain in `src/axon_recon/pipeline/runner.py`.
- Smoke (20 min max unless Adam approves longer): not run; this was a mocked runtime stage-name alignment slice.
- Smoke extension to 1 hour: not needed.
- Logs inspected: none.
- Not run: real-data smoke.

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
- Runtime aggregate names now match preferred reconstruct CLI selectors for embedded template phases.

Retired Code/Tests:
- Retired legacy `reconstruct.templates_*` emitted stage names from reconstruct runtime wrappers.

Risks And Follow-Ups:
- Function names still include `templates` because the underlying templates package has not been migrated into reconstruct yet.
- Direct `run_templates_*_from_runtime` wrappers and tests remain until templates retirement is complete.

Rollback Notes:
- Restore the previous `reconstruct.templates_*` stage-name strings if downstream log parsing temporarily depends on them.

## 2026-05-01 01:55 - d3f41e3 - ai: remove retired analysis runtime config

Status: accepted

Summary:
- Removed the dead top-level `stages.analysis` subtree from `debug/debug.runtime.yml` after deleting the analysis package.
- Left nested template/reconstruct `analysis:` knobs intact because they are not the retired stage selector.
- Confirmed the runtime config loads and no longer contains an active `analysis` stage key.

Acceptance Criteria:
- `debug/debug.runtime.yml` has no top-level `stages.analysis` block.
- The debug runtime still loads through the project config loader.
- Retired `analysis` selector remains unsupported.

Expected To Run:
- Active debug runtime stages remain `preprocess`, `spikesort`, `templates`, and `reconstruct` in config, with CLI `all` still limited to preprocess/spikesort/reconstruct.

Confirmed Not Run:
- No analysis stage runtime can be selected or configured from the debug runtime.

Files/Modules Changed:
- `debug/debug.runtime.yml`
- `debug/pipeline_refinement_commit_notes.md`

Validation:
- Config load: `RuntimeConfig.load('debug/debug.runtime.yml')` succeeded; `analysis in stages` printed `False` and `templates in stages` printed `True`.
- Pytest: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_config.py src/axon_recon/pipeline/tests/test_cli_stage_sequence.py -q` passed.
- CLI rejection: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m axon_recon.pipeline.cli stages analysis --config debug/debug.runtime.yml` failed as expected with `Unsupported stage token: analysis`.
- Smoke (20 min max unless Adam approves longer): not run; this was a retired config-block removal with no active stage execution path.
- Smoke extension to 1 hour: not needed.
- Logs inspected: none.
- Not run: real-data smoke.

Resume / Force-Restart Impact:
- Resume behavior: unchanged for active stages; retired analysis resume config is removed.
- Force-restart cleanup: unchanged for active stages; retired analysis force-restart config is removed.
- Partial-output handling: unchanged for active stages.

Storage/Cache Impact:
- Created: none in the repository.
- Cleaned: removed dead runtime YAML config only.
- Persisted: none.
- Size check: not applicable.

CLI Impact:
- No new selectors; `analysis` remains unsupported.

Retired Code/Tests:
- Removed the retired stage's debug runtime config block.

Risks And Follow-Ups:
- `stages.templates` remains in runtime config because reconstruct still depends on templates config and code; templates retirement needs migration first.
- Obsolete analysis docs under `docs/ai_slop` still reference old v1/v2 analysis concepts and should be considered during later doc/v1 cleanup.

Rollback Notes:
- Restore the removed `stages.analysis` YAML block from the prior commit if the deleted analysis runtime is temporarily restored.

## 2026-05-01 01:53 - b953428 - ai: delete retired analysis stage

Status: accepted

Summary:
- Deleted the retired `stages.analysis` package, its stage-local tests, and the pipeline-level analysis target status tests.
- Removed analysis runtime wiring from `pipeline.runner`, including `run_analysis_from_runtime`, analysis publish helpers, and cross-well result attachment helpers.
- Confirmed active CLI dispatch still rejects `analysis` and continues to expose only preprocess, spikesort, and reconstruct selectors.

Acceptance Criteria:
- No analysis-stage imports, runtime entry points, or result models remain under `src/axon_recon`.
- Retired `analysis` is rejected as a stage selector.
- Pipeline tests collect without stale analysis import failures.

Expected To Run:
- Active preprocess, spikesort, and reconstruct stage/phase selectors.

Confirmed Not Run:
- Analysis stage runtime and direct analysis CLI dispatch are removed.

Files/Modules Changed:
- `src/axon_recon/pipeline/runner.py`
- `src/axon_recon/pipeline/stages/analysis/`
- `src/axon_recon/pipeline/tests/test_analysis_target_status.py`

Validation:
- Pytest: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_cli_stage_sequence.py -q` passed.
- Pytest collection: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest --collect-only src/axon_recon/pipeline -q` passed collection.
- CLI rejection: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m axon_recon.pipeline.cli stages analysis --config debug/debug.runtime.yml` failed as expected with `Unsupported stage token: analysis`.
- Reference audit: no matches for `stages.analysis`, `run_analysis_from_runtime`, `run_analysis(`, `AnalysisInputs`, `AnalysisResult`, `parse_analysis`, or `generate_cross_well` under `src/axon_recon/**`.
- Smoke (20 min max unless Adam approves longer): not run; this was a deletion/import-surface slice with no real-data execution path left for analysis.
- Smoke extension to 1 hour: not needed.
- Logs inspected: none.
- Not run: real-data smoke.

Resume / Force-Restart Impact:
- Resume behavior: unchanged for active stages; retired analysis resume behavior is removed.
- Force-restart cleanup: unchanged for active stages; retired analysis force-restart behavior is removed.
- Partial-output handling: unchanged for active stages.

Storage/Cache Impact:
- Created: none in the repository.
- Cleaned: removed retired analysis code/tests only.
- Persisted: none.
- Size check: not applicable.

CLI Impact:
- `analysis` remains unsupported in `axon_recon stages` / `axon_recon stage` and now has no backing runtime package.

Retired Code/Tests:
- Deleted analysis API/config/core/cross-well/runner/models package files.
- Deleted analysis stage tests and pipeline analysis target-status tests.

Risks And Follow-Ups:
- `debug/debug.runtime.yml` still contains a `stages.analysis` block and should be cleaned in a later config slice.
- Any future analysis rewrite should be introduced as a new active design rather than reviving the deleted v2 analysis package.

Rollback Notes:
- Restore the deleted `stages.analysis` package, runner imports/helpers, and analysis target-status tests if the old analysis runtime must be temporarily recovered.

## 2026-05-01 01:45 - ac8251e - ai: retire direct templates CLI selectors

Status: accepted

Summary:
- Removed direct top-level `templates` stage and phase selectors from the stage-sequence CLI.
- Removed direct imports from `stages.templates.cli` in the top-level CLI.
- Converted direct templates selector tests into rejection tests now that reconstruct owns the active template phase entry points.

Acceptance Criteria:
- `templates`, `template`, `templates.*`, and legacy `template.*` selectors fail fast as unsupported top-level stage tokens.
- Preferred `reconstruct.<phase>` selectors remain available for active template-related reconstruct work.
- Focused CLI selector tests pass.

Expected To Run:
- Active selectors under preprocess, spikesort, and reconstruct.
- Reconstruct template-related selectors via `reconstruct.analyzers`, `reconstruct.build_templates`, and related preferred tokens.

Confirmed Not Run:
- Direct top-level `templates` selectors are no longer accepted by `axon_recon stages` / `axon_recon stage`.

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
- Direct `templates` / `template` selectors retired from the top-level stage-sequence CLI.
- Active template-related work should be invoked through reconstruct selectors.

Retired Code/Tests:
- Removed direct top-level CLI imports and handlers for `stages.templates.cli`.
- Removed tests that preserved direct templates CLI dispatch.

Risks And Follow-Ups:
- The `stages.templates` package and runtime helpers still exist and should be deleted after a broader reference audit confirms reconstruct covers active workflows.
- The debug runtime still contains a `stages.templates` block; later slices should remove or migrate remaining active config references.

Rollback Notes:
- Restore `stages.templates.cli` imports, template aliases, and direct `_STAGE_HANDLERS` entries if direct templates selectors are temporarily needed again.

## 2026-05-01 01:43 - 4ff446e - ai: add reconstruct phase selectors

Status: accepted

Summary:
- Added preferred `reconstruct.<phase>` CLI selectors for embedded template phases that already appear in the reconstruct runtime phase sequence.
- Mapped older `reconstruct.templates_*`, `recon.templates_*`, and `reconstruction.templates_*` forms to the preferred selectors.
- Added mixed selector coverage for `axon_recon stages spikesort reconstruct.analyzers`.

Acceptance Criteria:
- Preferred reconstruct phase tokens such as `reconstruct.analyzers`, `reconstruct.build_templates`, and `reconstruct.plot_templates` parse and dispatch.
- Legacy reconstruct template-prefixed tokens still map to the same behavior for now.
- A mixed full-stage plus reconstruct-phase selector runs in the requested order.

Expected To Run:
- `spikesort` followed by `reconstruct.analyzers` in the mixed selector test.
- Preferred reconstruct phase handlers for existing embedded template operations.

Confirmed Not Run:
- Full `reconstruct` does not run when only `reconstruct.analyzers` is selected.
- Retired `analysis` selector remains unsupported.

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
- Preferred reconstruct phase selectors now match runtime phase names for embedded template phases.
- Older reconstruct template-prefixed selectors remain as aliases until the direct templates stage is fully retired.

Retired Code/Tests:
- None; this is an additive selector-alignment slice.

Risks And Follow-Ups:
- Direct top-level `templates.*` selectors still exist and should be retired after confirming reconstruct selectors cover active workflows.

Rollback Notes:
- Remove the new `reconstruct.<phase>` handler entries and restore old reconstruct template-prefixed handler keys if needed.

## 2026-05-01 01:41 - faffe4e - ai: disconnect analysis CLI selector

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
