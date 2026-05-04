# Agent Guardrails Commit Notes

Living review log for AI-assisted work governed by the guardrail documents in this directory.

The guardrail documents are:

- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/logging_agent_guardrails.md`
- `debug/parallelism_agent_guardrails.md`
- `debug/container_mpi4py_NERSC_optimization_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`
- `debug/optimization_simplificaiton_guardrails.md`
- `debug/first_version_pipeline_guardrails.md` temporary first-version cleanup guardrail

Once agentic development begins, treat those guardrail documents as locked. Use this file for mutable running notes unless Adam explicitly asks to change a guardrail document.

## How To Use

After each AI commit, append a new entry at the top of `Commit Log` or directly below the most recent entry.

Every entry should include:

- what changed
- why it changed
- guardrail documents consulted
- acceptance criteria used
- focused tests run
- real-data smoke command and scope
- logs and artifacts inspected
- what was expected to run and confirmed not to run
- storage/cache impact
- CLI/debug-flag impact
- logging/parallelism impact when relevant
- container/NERSC/MPI impact when relevant
- residual risk, follow-ups, and rollback notes

Commits should be frequent, coherent, and prefixed with `ai:`. Do not push unless Adam explicitly asks.

## Entry Template

```markdown
## YYYY-MM-DD HH:MM - <short_sha> - ai: <commit subject>

Status: accepted | needs follow-up | reverted

Summary:
-

Guardrails Consulted:
-

Acceptance Criteria:
-

Expected To Run:
-

Confirmed Not Run:
-

Validation:
- Focused tests:
- Real-data smoke:
- Logs inspected:
- Artifacts inspected:
- Not run:

CLI / Debug Flag Impact:
-

Logging / Parallelism Impact:
-

Storage / Cache Impact:
- Created:
- Modified:
- Removed:

Container / NERSC / MPI Impact:
-

Resume / Force-Restart Impact:
-

Residual Risk And Follow-Ups:
-

Rollback Notes:
-
```

## Commit Log

## 2026-05-04 - pending - ai: honor spikesort and reconstruct debug limits

Status: accepted

Summary:
- Propagated shared CLI debug limit overrides through direct spikesort phase wrappers and runtime selectors.
- Moved spikesort and reconstruct dataset/well debug target limiting ahead of scratch input materialization.
- Carried applied debug-limit metadata into spikesort, reconstruct, and reconstruct-template phase summaries/log starts.
- Fixed direct spikesort sort segment limiting in the legacy sorter path.
- Fixed active debug runtime resource class names that blocked direct phase smokes during config validation.
- Fixed reconstruct non-unit phase result handling, template report unit limiting, configured template-root lookup, shared-root force-restart template-cache preservation, and clear-template-cache output-root selection.

Guardrails Consulted:
- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/logging_agent_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`
- `debug/parallelism_agent_guardrails.md`
- `/memories/repo/pipeline-debug-limits.md`
- `/memories/repo/templates-artifact-layout.md`

Acceptance Criteria:
- Direct `stages spikesort.<phase>` and `stages reconstruct.<phase>` selectors receive the same dataset, well, segment, and unit debug limits as full stages.
- Dataset/well target limits are applied before scratch input materialization or inspection.
- Enabled active phases write summaries/logs that expose the applied debug limits.
- Direct selected phases run only the requested phase while respecting existing force-restart/resume boundaries.

Expected To Run:
- Active spikesort phases: `bootstrap_concat_binary`, `sort`, `bombcell_label`, `cleanup_concat_binary`.
- Active reconstruct phases: `analyzers`, `build_templates`, `plot_templates`, `report_templates`, `generate_gtrs`, all downstream report/plot phases, and `clear_templates_cache`.

Confirmed Not Run:
- Full dataset/well scope.
- Disabled merge/template optional phases in the active runtime config.
- Push to remote.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_cli_stage_sequence.py src/axon_recon/pipeline/tests/test_spikesort_target_status.py src/axon_recon/pipeline/tests/test_reconstruct_target_status.py` passed with `189 passed`.
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/test_runner.py src/axon_recon/pipeline/stages/spikesort/tests/test_spikesort_config.py src/axon_recon/pipeline/stages/reconstruct/tests/test_runner.py src/axon_recon/pipeline/stages/reconstruct/tests/test_clear_templates_cache.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py` passed with `247 passed`.
- Real-data smoke: active spikesort phases passed under `--limit-datasets 1 --limit-wells 1 --limit-segments 2 --force-restart`.
- Real-data smoke: `reconstruct.analyzers`, `reconstruct.build_templates`, `reconstruct.plot_templates`, and `reconstruct.report_templates` passed under one dataset, one well, two segments, and small unit scopes.
- Real-data smoke: `reconstruct.generate_gtrs` passed with candidate units `0,2,3,4,5,6,7,8,9,10`; unit 8 succeeded and nine units failed with data-level axon_velocity branch/channel errors.
- Real-data smoke: downstream phases `plot_recons`, `plot_branch_propagations`, `plot_branch_velocities`, `plot_unit_summary`, `report_recons`, `report_recon_grid`, `report_full_chip_layout`, and `report_summaries` passed using unit 8.
- Real-data smoke: `reconstruct.clear_templates_cache` first exposed the wrong template root, then passed after the fix and cleared `recon_outputs/cache`.
- Diagnostics: VS Code diagnostics reported no errors for `debug/debug.runtime.yml` and this notes file; focused pytest covered modified source/test files.

CLI / Debug Flag Impact:
- Direct spikesort wrappers now forward `--limit-segments`, `--limit-datasets`, and `--limit-wells-per-dataset` to runtime selection.
- Direct reconstruct wrappers now forward unit/segment/dataset/well limits into both reconstruction and template input construction.
- Spikesort and reconstruct target limits now run through early target selection before scratch materialization.

Logging / Parallelism Impact:
- Stage/phase logs now include applied debug-limit context for the active spikesort/reconstruct/template phases.
- No resource-profile or worker-count semantics were intentionally changed.
- `generate_gtrs` still emits a process-pool parent-death-signal initializer warning in the container and falls back to in-process execution; this did not block the smoke.

Storage / Cache Impact:
- Created/updated limited real-data scratch/output artifacts for `Media_Density_T5_02182026_AR/260224/M08073/AxonTracking/000031/well000` under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch`.
- `reconstruct.generate_gtrs --force-restart` now preserves shared `recon_outputs/cache/templates` when templates and reconstruction share the output root.
- `reconstruct.clear_templates_cache` now clears the configured templates cache under `recon_outputs/cache` instead of legacy `template_outputs/cache`.

Container / NERSC / MPI Impact:
- Container smokes rebuilt the local `axon-recon:local` image from this source.
- No MPI/NERSC-specific changes were made.

Resume / Force-Restart Impact:
- Reconstruct force restart no longer deletes required shared-root template cache before `generate_gtrs` reads it.
- Unit-scoped downstream reconstruct phases reused the successful unit 8 GTR without clearing upstream artifacts.

Residual Risk And Follow-Ups:
- Several candidate units failed graph tracking due to data-level axon_velocity errors such as `No branches found`, `No branches left after cleaning`, and `Not enough channels selected to compute velocity`; unit 8 verified the downstream success path.
- Full dataset/well scope intentionally remains untested under the smoke guardrail.

Rollback Notes:
- Revert the runtime selector, direct wrapper, input-model/config, summary/log metadata, reconstruct template-root/cache, runtime YAML, and focused-test edits from this slice to restore previous spikesort/reconstruct direct phase behavior.

## 2026-05-04 - pending - ai: honor preprocess direct phase debug flags

Status: accepted

Summary:
- Added canonical `--limit-wells` as an alias to the existing per-dataset well limit behavior.
- Propagated preprocess CLI debug limits through direct `stages preprocess.<phase>` argument handlers, phase orchestrators, and public runtime wrappers.
- Applied preprocess target limits during target selection for all preprocess phases, not only copy-to-scratch materialization, so non-copy phases no longer inspect every configured scratch input before limiting.
- Carried target debug limits into `PreprocessInputs` and wrote `applied_debug_limits` into every preprocess phase summary.
- Added wrapper coverage for every current direct preprocess phase module.

Guardrails Consulted:
- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`
- `debug/logging_agent_guardrails.md`
- `debug/parallelism_agent_guardrails.md`
- `/memories/repo/pipeline-debug-limits.md`

Acceptance Criteria:
- Full preprocess and direct preprocess phase selectors receive the same `--limit-segments`, `--limit-datasets`, and well-limit values.
- `--limit-wells` and `--limit-wells-per-dataset` resolve to the same per-dataset well limiting behavior.
- Direct preprocess phases apply dataset/well limits before scratch input inspection or materialization.
- `preprocess.preprocess_segments` receives `--limit-segments` before segment work and writes only the limited segment manifest.
- Phase summaries record applied debug limits for auditability.

Expected To Run:
- Unit coverage for all current direct preprocess phase wrappers: `copy_src_to_scratch`, `save_rec_metadata`, `prepare_raw_binaries`, `wipe_src_scratch`, `preprocess_segments`, `plot_segment_traces`, `plot_segment_channel_layouts`, `concat_segments`, `plot_concat_traces`, `plot_concat_channel_layout`, and `plot_raster_threshold`.
- Real-data direct smokes for active heavy phases: `copy_src_to_scratch`, `save_rec_metadata`, and `preprocess_segments` on one dataset, one well, and two segments.

Confirmed Not Run:
- Full dataset/well scope.
- Optional disabled plotting, concat, report, cleanup, and wipe runtime phases in real data.
- Downstream spikesort/reconstruct stages.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_cli_stage_sequence.py src/axon_recon/pipeline/tests/test_parallel_fanout.py src/axon_recon/pipeline/tests/test_preprocess_target_status.py src/axon_recon/pipeline/stages/preprocess/tests/test_preprocess_config.py src/axon_recon/pipeline/stages/preprocess/tests/test_runner.py -q` passed.
- Real-data smoke: `axon-recon-container --gpus all stages preprocess.copy_src_to_scratch --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells 1 --limit-segments 2 --limit-units 15 --force-restart` passed.
- Real-data smoke: `axon-recon-container --gpus all stages preprocess.save_rec_metadata --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells 1 --limit-segments 2 --limit-units 15 --force-restart` passed.
- Real-data smoke: `axon-recon-container --gpus all stages preprocess.preprocess_segments --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells 1 --limit-segments 2 --limit-units 15 --force-restart` passed.
- Logs inspected: copy/materialization and non-copy direct phases now log `Applying execution target dataset limit before scratch materialization: 13 -> 1 dataset(s)` before touching only `dataset_000:data.raw.h5`; segment phase logs `segment_count=2`.
- Artifacts inspected: `context/copy_src_to_scratch_summary.json`, `context/recording_metadata_summary.json`, and `context/segment_recordings_summary.json` all include `applied_debug_limits` with dataset 1, wells-per-dataset 1, and segments 2; `preprocessed_segments/manifest.json` has `segment_count: 2`.
- Not run: real-data smokes for disabled optional direct phases; wrapper tests cover their debug-limit propagation.

CLI / Debug Flag Impact:
- Direct preprocess phase selectors now honor `--limit-segments`, `--limit-datasets`, `--limit-wells`, and `--limit-wells-per-dataset` through the same runtime override path as full preprocess.
- `--limit-units` remains parsed by the shared CLI but is not used by preprocess phases because preprocess has no unit scope.

Logging / Parallelism Impact:
- Target-limit logs now appear before non-copy direct phases inspect existing scratch inputs.
- Phase summaries now expose applied debug limits.
- No changes to worker-count or resource telemetry semantics.

Storage / Cache Impact:
- Created/updated limited real-data scratch/output artifacts for `Media_Density_T5_02182026_AR/260224/M08073/AxonTracking/000031/well000` under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch`.
- No full-scope scratch materialization was performed.

Container / NERSC / MPI Impact:
- Container smokes rebuilt the local `axon-recon:local` image from this source.
- No MPI/NERSC-specific changes were made.

Resume / Force-Restart Impact:
- Direct phase smokes used `--force-restart`; direct-phase force restart remained phase-scoped and did not delete source H5 data.

Residual Risk And Follow-Ups:
- Direct real-data smokes were limited to the active preprocess phases. Optional disabled phases were validated at wrapper/dispatch level but not run against real data in this slice.
- Direct phase terminal labels still show duplicated stage/phase text such as `preprocess.preprocess_segments.preprocess_segments`; this is cosmetic logging debt, not a debug-limit blocker.

Rollback Notes:
- Revert the CLI, preprocess orchestrator, runner, input-model, config, runner-summary, and focused-test edits from this slice to restore previous direct phase debug-limit behavior.

## 2026-05-04 - pending - ai: tighten preprocess phase semantics and smoke limits

Status: accepted

Summary:
- Made preprocess phase names canonical for configured sequences and direct selected phases, including `preprocess.<phase>` selectors.
- Full-stage preprocess now records explicit skipped summaries, logs, and timeline events for disabled phases listed in `phase_sequence` instead of silently filtering them out.
- Fixed resume behavior so a complete phase payload returned from disk does not fall through and rerun the phase core.
- Added `phase_statuses` to preprocess summaries.
- Moved preprocess dataset/well debug limits ahead of scratch input materialization when copy-to-scratch is active.
- Made observability environment capture robust when container UIDs do not have passwd entries.

Guardrails Consulted:
- `debug/stage_and_phase_behavior_guardrails.md`
- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/logging_agent_guardrails.md`
- `debug/parallelism_agent_guardrails.md`
- `debug/container_mpi4py_NERSC_optimization_guardrails.md`
- `debug/first_version_pipeline_guardrails.md`

Acceptance Criteria:
- Disabled phases in `phase_sequence` are visible as skipped, not silently ignored.
- Omitted phases still do not run just because config blocks exist.
- Direct phase selectors canonicalize consistently and reject invalid phase names.
- Resume-complete phase artifacts prevent rerun of the corresponding core.
- CLI dataset/well debug limits constrain scratch input materialization before heavy filesystem work.
- Observability artifacts are written successfully inside containers where `getpass.getuser()` cannot resolve the UID.

Expected To Run:
- `copy_src_to_scratch`, `save_rec_metadata`, and `preprocess_segments` for one dataset, one well, and two segments in the real-data smoke.

Confirmed Not Run:
- Disabled plotting, concat, report, cleanup, and wipe phases in the active runtime config.
- Scratch materialization for datasets beyond the single limited target.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_parallel_fanout.py src/axon_recon/pipeline/tests/test_preprocess_target_status.py src/axon_recon/pipeline/stages/preprocess/tests/test_preprocess_config.py src/axon_recon/pipeline/stages/preprocess/tests/test_runner.py -q` passed.
- Real-data smoke: `axon-recon-container --gpus all stages preprocess --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells-per-dataset 1 --limit-segments 2 --force-restart` passed with `targets_total: 1`, `targets_succeeded: 1`, `targets_failed: 0`.
- Logs inspected: final smoke showed `Applying execution target dataset limit before scratch materialization: 13 -> 1 dataset(s)`, `Selected wells: 1`, and `Starting preprocess_segments ... segment_count=2`.
- Artifacts inspected: `preprocess_summary.json` has `phase_statuses` for the three active phases as `success`; `preprocessed_segments/manifest.json` has `segment_count: 2`; `run_metadata/environment.json` was written with fallback user `uid:1010`.
- Not run: full dataset/well scope and downstream spikesort/reconstruct stages.

CLI / Debug Flag Impact:
- Existing preprocess debug limit flags now apply before scratch input materialization when preprocessing uses scratch input copies.
- No new CLI flags added in this slice.

Logging / Parallelism Impact:
- Added explicit phase start/completion/skipped semantics and `phase_statuses` summary reporting.
- Preserved declared `max_threads` versus observed raw thread telemetry semantics.

Storage / Cache Impact:
- Created/updated limited real-data scratch/output artifacts for `Media_Density_T5_02182026_AR/260224/M08073/AxonTracking/000031/well000` under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch`.
- No full-scope scratch materialization was performed.

Container / NERSC / MPI Impact:
- Container smoke rebuilt the local image from the modified source.
- No MPI changes were made.
- Observability no longer depends on passwd user lookup inside the container.

Resume / Force-Restart Impact:
- Resume-complete phase payloads now prevent phase-core reruns.
- Real-data smoke used `--force-restart`, which cleared only the limited target preprocess output directory.

Residual Risk And Follow-Ups:
- Broader YAML knob/legacy alias cleanup remains separate first-version work and was not started in this slice.
- Full-scope behavior intentionally not exercised under the smoke guardrail.

Rollback Notes:
- Revert the preprocess runner/config/runner/test edits from this slice to restore previous phase filtering, target-selection, and observability user behavior.

## 2026-05-03 - pending - ai: add agent guardrail documents

Status: accepted

Summary:
- Added locked guardrail documents for CLI debug flags, logging, parallelism, container/mpi4py/NERSC preparation, stage/phase behavior, and optimization/simplification.
- Added a separate temporary first-version pipeline guardrail for minimizing active YAML knobs, removing undesired fallback code, deleting unused aliases, and eliminating unused legacy code before rollout.
- Added this mutable commit-notes file for future AI implementation slices.

Guardrails Consulted:
- Source notes under `debug/ai_notes`.
- Existing refinement and container commit-note templates.
- Repo memories for debug limits, console/progress logging, resource telemetry, and process lifecycle.

Acceptance Criteria:
- Each requested guardrail file exists in `debug/`.
- Each guardrail emphasizes frequent `ai:` commits, real-data smoke tests, CLI debug flags, limited smoke scope, expanded scope for logging/parallelism, acceptance criteria, and locked-doc treatment.
- A separate first-version pipeline guardrail exists because those cleanup priorities may be less true after rollout.
- A separate commit-notes Markdown file exists in `debug/`.

Validation:
- Focused tests: not run; docs-only change.
- Real-data smoke: not run; docs-only change.
- Logs inspected: not applicable.
- Artifacts inspected: created Markdown files.

Residual Risk And Follow-Ups:
- Future implementation passes should update this file after every AI commit and leave the guardrail documents locked unless Adam asks for changes.
