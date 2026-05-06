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

## 2026-05-06 - pending - ai: align downstream reconstruct resource workers

Status: accepted

Summary:
- Routed direct downstream reconstruct phase selectors through selected-phase resource-class parallelism instead of deriving worker count from every enabled reconstruct phase.
- Added direct-phase worker allocation logging for reconstruct substages so resource class, `n_jobs`, and source are visible before the phase gate.
- Reclassified `report_summaries` in the debug runtime from `template_build` to `plot_report_grid` so summary deck/report work reserves report RAM and `plot_slots`.
- Added focused tests for direct phase resource-class selection, downstream phase worker allocation, and plot-slot gating for report phases.

Guardrails Consulted:
- `debug/parallelism_agent_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`
- `debug/logging_agent_guardrails.md`

Acceptance Criteria:
- Direct reconstruct phase selectors derive worker count from the selected phase `resource_class.cpu_cores`.
- Downstream unit phases that already parallelize units receive capped `inputs.n_jobs` values aligned with their live resource gate.
- Plot/report phases reserve `plot_slots` and heavy report RAM where configured.
- Logs make resource class, worker count, gate acquisition, and phase-tune advisory details visible.

Expected To Run:
- Focused unit tests for direct reconstruct runtime dispatch, downstream worker allocation, generate-GTR batching, and resource-budget slot gating.
- Import/syntax lint for touched Python files.
- Limited real-data host smoke covering `generate_gtrs`, `plot_recons`, and `report_summaries`.

Confirmed Not Run:
- Full repository test suite.
- Container image rebuild.
- Multi-well phase-tune calibration.
- Remote push.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_reconstruct_target_status.py::test_run_reconstruct_direct_phase_parallelism_uses_selected_resource_class src/axon_recon/pipeline/stages/reconstruct/tests/test_runner.py::test_reconstruct_phase_worker_allocation_uses_resource_class_cpu_for_downstream_phases src/axon_recon/pipeline/stages/reconstruct/tests/test_runner.py::test_run_reconstruct_generate_gtrs_batches_logs_unified_progress src/axon_recon/pipeline/tests/test_resource_budget.py::test_phase_budget_limits_plot_slots_for_report_phases src/axon_recon/pipeline/tests/test_resource_budget.py::test_phase_budget_limits_cpu_and_ram_capacity -q` passed.
- Lint: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m ruff check --select I,F src/axon_recon/pipeline/runner.py src/axon_recon/pipeline/tests/test_reconstruct_target_status.py src/axon_recon/pipeline/tests/test_resource_budget.py src/axon_recon/pipeline/stages/reconstruct/tests/test_runner.py` passed.
- Real-data smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/axon-reconstructor stages reconstruct.generate_gtrs reconstruct.plot_recons reconstruct.report_summaries --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells 1 --limit-units 4 --phase-tune --force-restart` succeeded.
- Logs inspected: `/tmp/axon_recon_downstream_parallelism_fix_smoke.log` showed `generate_gtrs` with `n_jobs=2`, `derived_unit_workers=2`, `unit_procs=2`, `plot_recons` with `resource_class=plot_unit`, `plot_slots=1`, and `max_threads=2`, and `report_summaries` with `resource_class=plot_report_grid`, `plot_slots=1`, and `max_threads=4`.
- Artifacts inspected: phase-tune output updated under the configured `resource_tuning` output root.

CLI / Debug Flag Impact:
- No CLI flag changes.
- Direct phase selectors continue to run only the requested phase and now show selected-phase worker allocation in logs.

Logging / Parallelism Impact:
- Direct reconstruct substages log `reconstruct phase worker allocation` with phase, resource class, `n_jobs`, and source.
- Direct downstream reconstruct phases now align `pipeline_thread_count`/resource usage `max_threads` with selected phase resource-class CPU rather than the maximum across all enabled reconstruct phases.

Storage / Cache Impact:
- Created/modified smoke outputs, per-phase summaries, and phase-tune artifacts under the configured scratch output during validation.
- No cache layout changes.

Container / NERSC / MPI Impact:
- No Dockerfile, container wrapper, NERSC, or MPI changes.

Resume / Force-Restart Impact:
- Smoke used `--force-restart` for the selected downstream phases only.
- Resume behavior is unchanged.

Residual Risk And Follow-Ups:
- `generate_gtrs` still recorded expected per-unit data-quality failures for units with too few selected channels while the target completed successfully.
- Multi-well calibration remains a separate tuning pass.

Rollback Notes:
- Revert the direct reconstruct phase resource-class selection change, direct worker allocation log, debug runtime `report_summaries` class change, and added tests to restore previous direct-phase behavior.

## 2026-05-06 - pending - ai: inline phase tune recommendations

Status: accepted

Summary:
- Moved per-resource-class phase-tune recommendation details into the existing `Phase resource usage` block for each phase observation.
- Kept aggregate `resource_tuning_summary.json` and `resource_tuning_report.md` artifact generation at the end of `--phase-tune` runs.
- Stopped emitting the duplicated post-stage resource/profile recommendation console logs after `stages: completed ...`.

Guardrails Consulted:
- `debug/logging_agent_guardrails.md`
- `debug/parallelism_agent_guardrails.md`

Acceptance Criteria:
- Per-phase `--phase-tune` logs show resource tuning recommendation details next to the measured `phase_tune_*` metrics.
- End-of-run phase tuning still writes aggregate observations, summary JSON, and Markdown report artifacts.
- `--phase-tune` remains explicit and does not change normal stage/phase logging.

Expected To Run:
- Focused resource-usage, phase-tuning, CLI, and phase-chain tests.
- Import/syntax lint for edited Python files.
- Limited real-data host smoke for `reconstruct.build_templates` with one dataset, one well, and one unit.

Confirmed Not Run:
- Full repository test suite.
- Container image rebuild.
- Multi-well calibration.
- Remote push.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_resource_usage.py src/axon_recon/pipeline/tests/test_phase_tuning.py src/axon_recon/pipeline/tests/test_pipeline_logging.py::test_phase_chain_logs_resource_class_and_writes_resource_usage src/axon_recon/pipeline/tests/test_cli_stage_sequence.py::test_phase_tune_runs_stage_then_emits_recommendations src/axon_recon/pipeline/tests/test_cli_stage_sequence.py::test_phase_tune_rejects_unlimited_scope_before_running_stage -q` passed.
- Lint: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m ruff check --select I,F src/axon_recon/pipeline/resource_usage.py src/axon_recon/pipeline/phase_tuning.py src/axon_recon/pipeline/cli.py src/axon_recon/pipeline/execution/phase_chain.py src/axon_recon/pipeline/tests/test_resource_usage.py` passed.
- Diagnostics: VS Code diagnostics reported no errors for edited Python files.
- Real-data smoke: `axon-reconstructor stages reconstruct.build_templates --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells 1 --phase-tune --limit-units 1` succeeded.
- Logs inspected: `/tmp/axon_recon_phase_tune_inline_host.log` showed `phase_tune_recommendation:` nested in the `Phase resource usage: reconstruct.build_templates.build_templates` block and no `Resource tuning recommendation:` or `Resource profile tuning recommendation:` records after stage completion.

CLI / Debug Flag Impact:
- No CLI flag changes.
- `--phase-tune` still controls both external system-tool sampling and recommendation log placement.

Logging / Parallelism Impact:
- Per-phase resource usage records include a structured `phase_tune_recommendation` payload when phase tuning is active and resource config is available.
- Parallelism, resource gating, and worker allocation behavior are unchanged.

Storage / Cache Impact:
- Created/modified phase-tune raw logs and aggregate tuning artifacts under the configured run output during validation.
- No source artifact path changes.

Container / NERSC / MPI Impact:
- No Dockerfile or container runtime changes.
- Cached local container image was not rebuilt during this slice, so `axon-recon-container --no-build` still reflected the previously built image until rebuilt.

Resume / Force-Restart Impact:
- None.

Residual Risk And Follow-Ups:
- Inline recommendations are one-observation advisories; the Markdown report remains the better place to review aggregate multi-run recommendations.

Rollback Notes:
- Revert the phase-chain inline recommendation call, resource usage formatter extension, and phase-tuning end-of-run log quieting to restore the previous separate post-stage recommendation logs.

## 2026-05-05 - 9791241 - ai: add phase tune system telemetry

Status: accepted

Summary:
- Kept `--phase-tune` as an explicit calibration mode and added opt-in per-phase system-tool sampling around phase windows.
- Added `pidstat` and `iostat` sidecar collection to phase resource usage records, with raw tool logs under `resource_tuning/raw/...` and summarized CPU/RAM/process-IO/device-IO fields in the recommendation report.
- Installed lightweight accounting tools (`time`, `sysstat`, `procps`) in the cached container runtime dependency layer.
- Left normal stage/phase runs unchanged unless `--phase-tune` is explicitly requested.

Guardrails Consulted:
- `debug/parallelism_agent_guardrails.md`
- `debug/container_mpi4py_NERSC_optimization_guardrails.md`
- `debug/logging_agent_guardrails.md`

Acceptance Criteria:
- Individual phase runs with `--phase-tune` collect per-phase CPU, RAM, process disk IO, and device IO metrics from established tools when available.
- Whole-stage runs with `--phase-tune` continue to emit one observation per phase window.
- Missing system tools degrade to the existing Python monitor with warnings instead of failing the phase.
- Runtime YAML remains advisory-only and is not rewritten.

Expected To Run:
- Focused phase-tuning/resource-usage tests.
- Focused CLI and phase-chain resource usage tests.
- Container image build and tool availability check.
- Limited real-data/container `reconstruct.build_templates` phase-tune smoke.

Confirmed Not Run:
- Full repository test suite.
- Multi-well phase-tune calibration.
- Deep profilers such as Memray, Scalene, or perf as part of default `--phase-tune`.
- Remote push.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_resource_usage.py src/axon_recon/pipeline/tests/test_phase_tuning.py src/axon_recon/pipeline/tests/test_pipeline_logging.py::test_phase_chain_logs_resource_class_and_writes_resource_usage src/axon_recon/pipeline/tests/test_cli_stage_sequence.py::test_phase_tune_runs_stage_then_emits_recommendations src/axon_recon/pipeline/tests/test_cli_stage_sequence.py::test_phase_tune_rejects_unlimited_scope_before_running_stage -q` passed.
- Lint: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m ruff check --select I,F src/axon_recon/pipeline/resource_usage.py src/axon_recon/pipeline/phase_tuning.py src/axon_recon/pipeline/cli.py src/axon_recon/pipeline/execution/phase_chain.py src/axon_recon/pipeline/tests/test_resource_usage.py src/axon_recon/pipeline/tests/test_phase_tuning.py` passed.
- Container build: `containers/axon-recon/build_local_image.sh --image axon-recon:local` succeeded.
- Container tool check: rebuilt image contains `/usr/bin/time`, `pidstat`, `iostat`, `sar`, and `ps`.
- Container monitor smoke: in-image `start_phase_resource_monitor(...)` produced `phase_tune_tools=['pidstat', 'iostat']` with one sample and no warnings.
- Real-data smoke: `axon-recon-container --no-build stages reconstruct.build_templates --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells 1 --phase-tune --limit-units 1` succeeded.
- Logs inspected: phase resource usage included `phase_tune_avg_cpu_pct`, `phase_tune_peak_cpu_pct`, `phase_tune_peak_rss_gb`, process read/write rates, device read/write rates, await, util, and raw log directory.
- Artifacts inspected: `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/resource_tuning/resource_tuning_report.md` included the new phase-tune metrics.

CLI / Debug Flag Impact:
- `--phase-tune` still uses the existing CLI surface.
- `resources.tuning.system_tools_enabled`, `resources.tuning.system_tool_interval_s`, and `resources.tuning.write_tool_logs` can tune the system-tool sampler behavior.

Logging / Parallelism Impact:
- Per-phase resource records gain `phase_tune_*` fields only when tuning data is present.
- Existing resource gating and worker allocation behavior is unchanged.

Storage / Cache Impact:
- Created raw phase-tune logs under `resource_tuning/raw/<run>/<stage>/<phase>/...` when `write_tool_logs` is true.
- Added `memray_*.bin` to `.gitignore` for local profiling outputs.

Container / NERSC / MPI Impact:
- Local Docker image now installs `time`, `sysstat`, and `procps` before the source copy so the layer is cached across repo edits.
- No MPI behavior changed.
- Shifter/NERSC images need to be rebuilt/pushed before relying on in-image `pidstat`/`iostat` telemetry there.

Resume / Force-Restart Impact:
- None. Phase-tune measurement wraps whatever phase execution path is selected and does not change resume/force-restart semantics.

Residual Risk And Follow-Ups:
- Concurrent multi-well full-stage tuning can still make process-level attribution less precise because overlapping phase windows share the same Python parent process; individual phase calibration remains the intended high-confidence workflow.
- `perf`, Memray, and Scalene remain separate deep-profiling tools, not default phase-tune dependencies.

Rollback Notes:
- Revert the resource usage sidecar monitor, CLI phase-tune monitor configuration, and Docker observability package layer to return to Python-only resource usage reporting.

## 2026-05-05 - pending - ai: route build templates through canonical orchestrator

Status: accepted

Summary:
- Moved `build_templates` phase orchestration into `src/axon_recon/pipeline/stages/reconstruct/phases/build_templates.py` so the phase flow can be audited in one ordered file.
- Routed templates API, templates runner phase dispatch, and reconstruct direct-phase dispatch through the canonical phase module.
- Removed duplicate cached-analyzer bootstrap and payload-root routing logic from the broad templates runner instead of leaving a compatibility implementation behind.
- Added `reconstruct/phases/__init__.py` so the new phase module is included by package discovery.

Guardrails Consulted:
- `debug/stage_and_phase_behavior_guardrails.md`
- `debug/optimization_simplificaiton_guardrails.md`
- `debug/first_version_pipeline_guardrails.md`
- `debug/parallelism_agent_guardrails.md`

Acceptance Criteria:
- The build_templates phase reads as an ordered route: resolve context, force-restart cleanup, payload readiness check, cached-analyzer bootstrap when needed, core build from payloads, summary write.
- Public/direct build_templates phase routes enter the canonical phase module.
- The broad templates runner no longer carries duplicate build_templates cached-analyzer orchestration.
- Existing lazy cached-analyzer behavior and disk payload behavior remain covered by focused tests.

Expected To Run:
- Focused templates runner tests.
- Focused reconstruct stage dispatch tests.
- Import/stale-import lint checks for touched files.

Confirmed Not Run:
- Full repository test suite.
- Real-data smoke.
- Full CLI/container run.
- Remote push.

Validation:
- Diagnostics: VS Code diagnostics reported no errors for touched source and test files.
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py src/axon_recon/pipeline/stages/reconstruct/tests/test_runner.py -q` passed.
- Import/stale-import lint: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m ruff check --select I ...` passed for the new phase module, templates API, and touched test imports.
- Stale-import lint: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m ruff check --select F401 ...` passed for the new phase module, templates runner, templates API, and touched test file.
- Not run: broad Ruff over the large touched test file still reports unrelated pre-existing line-length/indentation/local-variable findings outside this slice.

CLI / Debug Flag Impact:
- No CLI flag changes.
- Existing external selectors are still routed; this slice did not remove CLI/config aliases beyond eliminating duplicate build_templates implementation code.

Logging / Parallelism Impact:
- Build_templates logs remain at the same phase points and now include `lazy_load_analyzers` in the settings line.
- No resource class or worker allocation behavior changed.

Storage / Cache Impact:
- Created `src/axon_recon/pipeline/stages/reconstruct/phases/__init__.py`.
- Created `src/axon_recon/pipeline/stages/reconstruct/phases/build_templates.py`.
- Modified source payload materialization ownership only by moving orchestration code; artifact paths and cleanup behavior remain unchanged.

Container / NERSC / MPI Impact:
- None.

Resume / Force-Restart Impact:
- Force-restart source-payload cleanup remains scoped to the build_templates payload root.
- Existing source payload reuse behavior is unchanged.

Residual Risk And Follow-Ups:
- Broader legacy CLI/config aliases such as `templates_build_templates` and nested `per_unit_processing.build_templates` remain for a separate explicit cleanup slice.
- A limited real-data `reconstruct.build_templates` smoke was not run after this routing-only refactor.

Rollback Notes:
- Revert the new `reconstruct/phases` module additions and restore the build_templates functions/imports in `templates/runner.py` if the canonical phase route needs to be backed out.

## 2026-05-05 - 945159a - ai: improve axon recon container cache

Status: accepted

Summary:
- Split the axon-recon Docker build so heavyweight plugin/runtime dependency installs run before the full repository copy.
- Copied only `containers/axon-recon/install_maxwell_hdf5_plugin.py` before the expensive install layer, then copied the full repo later for the local package and optional sibling package installs.
- Kept entrypoint/smoke script chmod and symlink creation after the final copy so overwritten script metadata is refreshed.

Guardrails Consulted:
- `debug/optimization_simplificaiton_guardrails.md`
- `debug/container_mpi4py_NERSC_optimization_guardrails.md`

Acceptance Criteria:
- Ordinary source changes no longer invalidate the plugin/runtime dependency install layer.
- Final image behavior stays equivalent for local axon_reconstructor, UnitMatchPy, and SLAy installs.
- Container scripts still resolve through the existing build helper.

Expected To Run:
- Docker build helper dry run.

Confirmed Not Run:
- Full Docker image build.
- Runtime real-data smoke.
- Remote push.

Validation:
- Diagnostics: VS Code diagnostics reported no errors for `containers/axon-recon/Dockerfile`.
- Dry run: `containers/axon-recon/build_local_image.sh --dry-run --no-unitmatch --no-slay` resolved the expected `docker build` command.
- Not run: full Docker build, because this slice specifically reduces build invalidation and the dry run was enough to validate wrapper wiring.

CLI / Debug Flag Impact:
- No CLI flag changes.

Logging / Parallelism Impact:
- None.

Storage / Cache Impact:
- Modified Docker layer cache behavior only; no runtime artifacts were created or removed.

Container / NERSC / MPI Impact:
- Container build cache should now reuse plugin/runtime install layers when repository source files change.
- No NERSC/MPI runtime behavior changed.

Resume / Force-Restart Impact:
- None.

Residual Risk And Follow-Ups:
- A full image build was not run in this slice; remaining risk is shell/install ordering in the final image build.

Rollback Notes:
- Revert `containers/axon-recon/Dockerfile` to restore the previous single post-copy install layer.

## 2026-05-04 - pending - ai: show spikeinterface progress bars

Status: accepted

Summary:
- Removed spikesort process-wide stdout/stderr redirection from the external-debug-output suppression path so SpikeInterface/tqdm bars can reach the terminal while logger-level noise suppression remains in place.
- Added explicit `progress_bar` controls for local SpikeInterface sorting and reconstruct template analyzer policies, defaulting to visible progress and propagating into SpikeInterface global job kwargs and analyzer `compute()` calls.
- Kept `verbose` separate from progress bars: `verbose` still controls sorter chatter, while `progress_bar` controls tqdm visibility.
- Updated focused tests to assert terminal streams remain live and progress kwargs are passed even when verbose logging is false.

Guardrails Consulted:
- `debug/logging_agent_guardrails.md`
- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`

Acceptance Criteria:
- SpikeInterface progress bars are not swallowed by pipeline logging/debug-output suppression.
- Local SpikeInterface sort/analyzer paths use an explicit progress-bar knob instead of coupling progress to verbose logging.
- Reconstruct template analyzer computes can show SpikeInterface progress bars through analyzer policy config.
- Phase lifecycle/resource logs remain visible around active progress bars.
- No process-global stdout/stderr redirection is used in the spikesort SI sort/debug suppression path.

Expected To Run:
- Focused host tests for spikesort local SI/config and reconstruct template config/SI extraction.
- Focused container tests for the same paths plus pipeline logging and preprocess progress-adjacent tests.
- Real-data TTY smokes for a SpikeInterface binary-save path and the local SpikeInterface sort path.

Confirmed Not Run:
- Full dataset/well scope.
- Full repository test suite.
- Remote push.

Validation:
- Diagnostics: VS Code diagnostics reported no errors for modified source/test files.
- Host focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/test_local_spikeinterface.py src/axon_recon/pipeline/stages/spikesort/tests/test_spikesort_config.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_config.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_spikeinterface_extract.py -q` passed.
- Environment API check: Pylance snippet confirmed SpikeInterface `0.103.3`, `set_global_job_kwargs(**job_kwargs)`, and `SortingAnalyzer.compute(..., **kwargs)` support the propagated progress kwargs.
- Container focused tests: `docker run --rm -v "$PWD":/work -w /work axon-recon:test python -m pytest src/axon_recon/pipeline/stages/spikesort/tests/test_local_spikeinterface.py src/axon_recon/pipeline/stages/spikesort/tests/test_spikesort_config.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_config.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_spikeinterface_extract.py -q` passed.
- Container logging/preprocess tests: `docker run --rm -v "$PWD":/work -w /work axon-recon:test python -m pytest src/axon_recon/pipeline/tests/test_pipeline_logging.py src/axon_recon/pipeline/stages/preprocess/tests/test_prepare_raw_binaries_core.py src/axon_recon/pipeline/stages/preprocess/tests/test_runner.py -q` passed.
- Real-data TTY smoke: `script -qefc '/home/adamm/miniconda3/envs/axon_recon/bin/axon-recon-container --gpus all stages spikesort.bootstrap_concat_binary --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells 1 --limit-segments 1 --limit-units 15 --force-restart --phase-tune' /tmp/axon-recon-si-bootstrap-progress-smoke.txt` passed and showed `write_binary_recording (workers: 6 processes): 100%|...| 127/127`.
- Real-data TTY smoke: `script -qefc '/home/adamm/miniconda3/envs/axon_recon/bin/axon-recon-container --gpus all stages spikesort.sort --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells 1 --limit-segments 1 --limit-units 15 --force-restart --phase-tune' /tmp/axon-recon-si-sort-progress-smoke.txt` passed and showed `SpikeInterface global job kwargs: {'n_jobs': 8, 'chunk_duration': '1s', 'progress_bar': True}`, Kilosort/tqdm progress, and `estimate_sparsity (workers: 8 processes): 100%|...| 127/127`.
- Real-data TTY smoke: `script -qefc '/home/adamm/miniconda3/envs/axon_recon/bin/axon-recon-container --gpus all stages preprocess.preprocess_segments --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells 1 --limit-segments 1 --limit-units 15 --force-restart --phase-tune' /tmp/axon-recon-si-progress-smoke.txt` passed for the lazy preprocess path and preserved pipeline progress/lifecycle/resource logs.
- Logs inspected: TTY transcripts in `/tmp/axon-recon-si-bootstrap-progress-smoke.txt`, `/tmp/axon-recon-si-sort-progress-smoke.txt`, and `/tmp/axon-recon-si-progress-smoke.txt`.

CLI / Debug Flag Impact:
- No new CLI flags.
- Added config-level `progress_bar` control for spikesort local SpikeInterface sorting and reconstruct template analyzer policies.
- Existing `debug_outputs=false` no longer hides terminal stdout/stderr in spikesort sort paths; it still suppresses configured noisy logger stream handlers.

Logging / Parallelism Impact:
- Progress bars and logs now coexist in real TTY validation.
- Local SI sort global job kwargs now include `progress_bar=True` by default, independent of `verbose`.
- Merge/bombcell analyzer extension job kwargs inherit the same progress setting through `_merge_analyzer_compute_job_kwargs()`.

Storage / Cache Impact:
- Created/updated limited-scope smoke outputs under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs`.
- `--phase-tune` temporarily wrote a resource recommendation into `debug/debug.runtime.yml`; that validation noise was restored before commit.

Container / NERSC / MPI Impact:
- Container validation used the existing `axon-recon:test` image.
- No NERSC/MPI-specific behavior was changed.

Resume / Force-Restart Impact:
- No resume semantics changed.
- Real-data smokes used `--force-restart` to force progress-producing work in a one-dataset/one-well/one-segment scope.

Residual Risk And Follow-Ups:
- Full production-scope Kilosort/template analyzer progress behavior was not run; the limited TTY smokes and focused tests cover the progress/logging mechanics.

Rollback Notes:
- Revert the `progress_bar` config/input propagation and restore spikesort debug-output stdout/stderr redirection if the operator wants the prior no-terminal-output behavior.

## 2026-05-04 - pending - ai: log resource gate waits everywhere

Status: accepted

Summary:
- Promoted resource gate wait warnings to structured `phase_resource_gate_waiting` events emitted from `ResourceBudgetManager.phase_budget()`.
- Added wait payload details to the warning record, including slot demand, available slots at first wait, keyed resource demand/limits, first wait snapshot, `well_workers`, and planned target count.
- Kept the warning at the shared resource gate layer so preprocess direct phases, spikesort/reconstruct phase chains, and reconstruct template phase chains all get the same wait signal when workers queue for slots.
- Added focused tests for the low-level budget manager warning and for JSONL emission through the shared phase-chain path.

Guardrails Consulted:
- `debug/logging_agent_guardrails.md`
- `debug/parallelism_agent_guardrails.md`
- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`

Acceptance Criteria:
- Any resource-gated phase that has to wait for a slot emits a warning log.
- The warning is structured with event `phase_resource_gate_waiting` for JSONL filtering.
- The human log still includes a readable wait warning with phase, target, resource class, worker count, slot demand, available slots, and keyed-resource state.
- Existing `phase_resource_gate` acquisition and `phase_resource_usage` logs remain unchanged after acquisition.

Expected To Run:
- Focused resource budget and pipeline logging tests.
- Containerized focused and broader pipeline tests.
- Limited real-data `preprocess.save_rec_metadata --phase-tune` smoke with enough datasets to force h5 read slot waits.

Confirmed Not Run:
- Full dataset/well scope.
- Runtime YAML mutation.
- Push to remote.

Validation:
- Diagnostics: VS Code diagnostics reported no errors for modified source/test files.
- Focused host tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_resource_budget.py src/axon_recon/pipeline/tests/test_pipeline_logging.py -q` passed with 13 tests.
- Container focused tests: `docker run --rm -v /home/adamm/dev/pkgs/axon_reconstructor:/workspace -w /workspace --entrypoint python axon-recon:test -m pytest src/axon_recon/pipeline/tests/test_resource_budget.py src/axon_recon/pipeline/tests/test_pipeline_logging.py -q` passed with 13 tests after one transient terminal SIGINT rerun.
- Container broad tests: `docker run --rm -v /home/adamm/dev/pkgs/axon_reconstructor:/workspace -w /workspace --entrypoint python axon-recon:test -m pytest src/axon_recon/pipeline/tests --ignore=src/axon_recon/pipeline/tests/test_progress.py -q` passed.
- Real-data smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/axon-recon-container --gpus all stages preprocess.save_rec_metadata --config debug/debug.runtime.yml --limit-segments 2 --limit-datasets 12 --limit-wells-per-dataset 1 --limit-units 15 --force-restart --phase-tune` passed for run `debug.runtime-20260504T071712Z`.
- Logs inspected: `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/logs/pipeline.jsonl` contains six `phase_resource_gate_waiting` warning events with `well_workers=12`, `target_count=12`, blocked `h5_read_slots`, and scratch input H5 keyed requests.
- Logs inspected: `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/logs/pipeline.log` contains matching human-readable `Phase resource gate waiting for slots` warning lines.

CLI / Debug Flag Impact:
- No CLI flag changes.

Logging / Parallelism Impact:
- Added structured wait-warning events for workers blocked before resource acquisition.
- Runtime gate behavior and slot accounting are unchanged.

Storage / Cache Impact:
- Updated tuning/log artifacts under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs` during smoke validation.
- Rebuilt local `axon-recon:local` and dev-flavored `axon-recon:test` images.

Container / NERSC / MPI Impact:
- Container validation used rebuilt local images.
- No NERSC/MPI-specific behavior was changed.

Resume / Force-Restart Impact:
- No resume semantics were changed.
- The smoke used `--force-restart` to regenerate metadata and wait observations for the limited scope.

Residual Risk And Follow-Ups:
- The real-data smoke directly exercised preprocess `save_rec_metadata`; shared phase-chain coverage is synthetic but uses the same `ResourceBudgetManager` path used by spikesort/reconstruct/template phases.

Rollback Notes:
- Revert the structured warning payload/event changes in `resource_budget.py` and the focused tests from this slice to return to plain warning-only wait logs.

## 2026-05-04 - pending - ai: tune scratch h5 read paths

Status: accepted

Summary:
- Treated `save_rec_metadata.metadata_source: scratch_h5` as a scratch-input request, alongside the older scratch aliases, while preserving `source_h5_path` as the original source provenance path.
- Added phase-specific H5 read path resolution for preprocess phases so resource gates and resource usage logs use the file each phase actually reads.
- Added `phase_read_h5_path` to phase resource JSONL payloads and tuning observations.
- Updated phase tuning disk measurement and utilization logic to prefer `phase_read_h5_path` over original `source_h5_path`; scratch read rows remain visible even when scratch inputs share the output device benchmark.

Guardrails Consulted:
- `debug/logging_agent_guardrails.md`
- `debug/parallelism_agent_guardrails.md`
- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`

Acceptance Criteria:
- With `metadata_source: scratch_h5` and existing scratch inputs, `save_rec_metadata` reads the scratch H5 while the original source path remains recorded as provenance.
- Preprocess H5 read resource keys point at the actual phase read path, not always the original source path.
- `--phase-tune` observations, disk measurements, and bandwidth pressure use scratch input paths when scratch inputs are selected and available.
- Reports no longer surface NAS source H5 measurements for scratch-selected metadata reads.

Expected To Run:
- Focused phase tuning and preprocess runner tests.
- Containerized focused and broader pipeline tests.
- Limited real-data `preprocess.save_rec_metadata --phase-tune` smoke using `debug/debug.runtime.yml`.

Confirmed Not Run:
- Full dataset/well scope.
- Runtime YAML mutation.
- Push to remote.

Validation:
- Diagnostics: VS Code diagnostics reported no errors for modified source/test files.
- Focused host tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_phase_tuning.py src/axon_recon/pipeline/stages/preprocess/tests/test_runner.py -q` passed with 48 tests.
- Container focused tests: `docker run --rm -v /home/adamm/dev/pkgs/axon_reconstructor:/workspace -w /workspace --entrypoint python axon-recon:test -m pytest src/axon_recon/pipeline/tests/test_phase_tuning.py -q` passed with 9 tests.
- Container focused tests: `docker run --rm -v /home/adamm/dev/pkgs/axon_reconstructor:/workspace -w /workspace --entrypoint python axon-recon:test -m pytest src/axon_recon/pipeline/stages/preprocess/tests/test_runner.py -q` passed with 39 tests.
- Container broad tests: `docker run --rm -v /home/adamm/dev/pkgs/axon_reconstructor:/workspace -w /workspace --entrypoint python axon-recon:test -m pytest src/axon_recon/pipeline/tests --ignore=src/axon_recon/pipeline/tests/test_progress.py -q` passed.
- Real-data smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/axon-recon-container --gpus all stages preprocess.save_rec_metadata --config debug/debug.runtime.yml --limit-segments 2 --limit-datasets 2 --limit-wells-per-dataset 1 --limit-units 15 --force-restart --phase-tune` passed for run `debug.runtime-20260504T062937Z`.
- Logs inspected: `resource_usage_observations.jsonl` contains original NAS `source_h5_path`, scratch `phase_read_h5_path`, and scratch keyed `source_h5_path` resource requests for `save_rec_metadata`.
- Artifacts inspected: `resource_tuning_report.md` includes a `phase_read_h5` disk measurement and `phase_read_h5` disk utilization row under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/inputs/...`.

CLI / Debug Flag Impact:
- No CLI flag changes.

Logging / Parallelism Impact:
- Added `phase_read_h5_path` to phase resource gate/usage payloads.
- Resource gate concurrency behavior is unchanged; keyed H5 accounting now targets the actual read file for preprocess phases.

Storage / Cache Impact:
- Updated tuning artifacts under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/resource_tuning` during smoke validation.
- Rebuilt local `axon-recon:local` and dev-flavored `axon-recon:test` images.

Container / NERSC / MPI Impact:
- Container validation used the rebuilt local images.
- No NERSC/MPI-specific behavior was changed.

Resume / Force-Restart Impact:
- No resume semantics were changed.
- The smoke used `--force-restart` to regenerate metadata and tuning observations for the selected limited scope.

Residual Risk And Follow-Ups:
- The scratch H5 disk measurement row may reuse the run-output device benchmark when scratch inputs and outputs share the same disk; the report calls this out in the measurement note.
- A combined focused container pytest command was intermittently interrupted by SIGINT from the terminal session; the same test files passed when rerun separately in the container.

Rollback Notes:
- Revert the metadata source alias handling, preprocess phase read-path resource context, `phase_read_h5_path` log field, phase tuning read-path preference, and focused tests from this slice to restore source-path-only tuning behavior.

## 2026-05-04 - pending - ai: include queued resource demand in tuning

Status: accepted

Summary:
- Added structured resource gate acquisition metadata from `ResourceBudgetManager.phase_budget()`, including slot demands, keyed requests, keyed limits, acquisition availability, and actual wait time when a worker blocks.
- Attached resource gate metadata to shared phase-chain and preprocess `phase_resource_usage` records and added `phase_resource_gate` log events.
- Updated `--phase-tune` to compute requested slot demand from `gate wait + active phase` intervals, while preserving active-only demand as a diagnostic.
- Updated reports/logs so active profile IO recommendations now call out requested demand including queued workers and resource gate wait totals.

Guardrails Consulted:
- `debug/logging_agent_guardrails.md`
- `debug/parallelism_agent_guardrails.md`
- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`

Acceptance Criteria:
- Waiting workers are represented in tuning observations through resource gate wait metadata.
- Peak profile IO demand used for recommendations includes queued/requested intervals, not only active post-acquisition intervals.
- Reports expose both requested and active slot demand so queueing effects can be distinguished from actual disk throughput.
- Existing resource gates continue to enforce the same profile and keyed limits.

Expected To Run:
- Focused resource budget and phase tuning tests.
- Containerized focused and broad pipeline tests.
- Limited real-data `preprocess.preprocess_segments --phase-tune` smoke.

Confirmed Not Run:
- Full dataset/well scope.
- Runtime YAML mutation.
- Push to remote.

Validation:
- Focused container tests: `docker run --rm -v /home/adamm/dev/pkgs/axon_reconstructor:/workspace -w /workspace axon-recon:test python -m pytest src/axon_recon/pipeline/tests/test_resource_budget.py src/axon_recon/pipeline/tests/test_phase_tuning.py -q` passed.
- Container broad tests: `docker run --rm -v /home/adamm/dev/pkgs/axon_reconstructor:/workspace -w /workspace axon-recon:test python -m pytest src/axon_recon/pipeline/tests --ignore=src/axon_recon/pipeline/tests/test_progress.py -q` passed.
- Real-data smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/axon-recon-container --gpus all stages preprocess.preprocess_segments --config debug/debug.runtime.yml --limit-segments 2 --limit-datasets 6 --limit-wells-per-dataset 1 --limit-units 15 --force-restart --phase-tune` passed for run `debug.runtime-20260504T054709Z`.
- Logs inspected: `pipeline.jsonl` includes `phase_resource_gate` events and `resource_gate` payloads on `phase_resource_usage` events; `phase_tuning_profile_recommendation` logs requested demand and gate wait stats.
- Artifacts inspected: `resource_tuning_report.md` includes `max_requested_*_slot_demand`, `max_active_*_slot_demand`, and resource gate wait counts/totals.
- Diagnostics: VS Code diagnostics reported no errors for modified source/test files.

CLI / Debug Flag Impact:
- No CLI flag changes.

Logging / Parallelism Impact:
- Added `phase_resource_gate` events and `resource_gate` payloads to phase resource usage logs.
- Runtime concurrency behavior is unchanged; the tuning calculation now includes queued/requested demand when gate waits occur.

Storage / Cache Impact:
- Updated tuning artifacts under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/resource_tuning` during the smoke.
- Rebuilt container images as part of validation.

Container / NERSC / MPI Impact:
- Rebuilt local `axon-recon:local` and refreshed disposable `axon-recon:test` for container validation.
- No NERSC/MPI-specific behavior was changed.

Resume / Force-Restart Impact:
- No resume or force-restart semantics were changed.

Residual Risk And Follow-Ups:
- The limited smoke did not observe an actual gate wait with the current profile and target shape; synthetic tests cover queued demand where active-only demand stays lower than requested demand.

Rollback Notes:
- Revert the resource gate metadata payloads, phase log attachments, requested-demand overlap calculation, report/log field additions, and focused tests from this slice to return to active-only demand accounting.

## 2026-05-04 - pending - ai: log profile io tuning reasons

Status: accepted

Summary:
- Expanded active profile IO tuning notes so flat recommendations explain the limiting reason, including utilization thresholds, peak recommended slot demand, current profile slots, and why increasing slots would not change the observed run.
- Emitted profile tuning notes and warnings as structured log events instead of only writing them to the report.
- Emitted per-path disk bandwidth utilization records to logs so measured capacity and observed read/write utilization are visible in `pipeline.log` and `pipeline.jsonl`.

Guardrails Consulted:
- `debug/logging_agent_guardrails.md`
- `debug/parallelism_agent_guardrails.md`
- `debug/cli_debug_flags_agent_guardrails.md`

Acceptance Criteria:
- The report explains why underused bandwidth can still produce flat profile slot recommendations.
- `pipeline.jsonl` includes structured events for profile recommendation notes/warnings and disk bandwidth utilization.
- `pipeline.log` includes the same human-readable notes for terminal/file auditability.
- No runtime gating behavior or YAML mutation is changed.

Expected To Run:
- Focused phase tuning unit tests.
- Container focused and broad pipeline tests.
- Limited real-data `preprocess.preprocess_segments --phase-tune` smoke matching the confusing report shape.

Confirmed Not Run:
- Full dataset/well scope.
- Runtime YAML mutation.
- Push to remote.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_phase_tuning.py -q` passed with six focused tests.
- Container focused tests: `docker run --rm -w /opt/axon_reconstructor axon-recon:test python -m pytest src/axon_recon/pipeline/tests/test_phase_tuning.py` passed with `6 passed`.
- Container broad tests: `docker run --rm -w /opt/axon_reconstructor axon-recon:test python -m pytest src/axon_recon/pipeline/tests --ignore=src/axon_recon/pipeline/tests/test_progress.py` passed with `325 passed`.
- Real-data smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/axon-recon-container --gpus all stages preprocess.preprocess_segments --config debug/debug.runtime.yml --limit-segments 2 --limit-datasets 6 --limit-wells-per-dataset 1 --limit-units 15 --force-restart --phase-tune` passed.
- Logs inspected: `pipeline.jsonl` has `phase_tuning_disk_bandwidth_utilization` and `phase_tuning_profile_recommendation_note` events for run `debug.runtime-20260504T052146Z`; `pipeline.log` includes the same note text.
- Artifacts inspected: `resource_tuning_report.md` now includes the explicit note that peak demand `2` did not exceed current profile slots `3`, so increasing slots would not change the run.
- Diagnostics: VS Code diagnostics reported no errors for modified tuning source/test files.

CLI / Debug Flag Impact:
- No CLI flag changes.

Logging / Parallelism Impact:
- Added `phase_tuning_disk_bandwidth_utilization`, `phase_tuning_profile_recommendation_note`, and `phase_tuning_profile_recommendation_warning` structured events.
- Runtime resource gates and worker allocation behavior are unchanged.

Storage / Cache Impact:
- Updated tuning artifacts under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/resource_tuning` during the smoke.
- No cache deletion or runtime YAML mutation was performed.

Container / NERSC / MPI Impact:
- Rebuilt local `axon-recon:local` and refreshed disposable `axon-recon:test`.
- No NERSC/MPI-specific behavior was changed.

Resume / Force-Restart Impact:
- No force-restart semantics were changed.

Residual Risk And Follow-Ups:
- The profile recommendation still correctly stays flat when observed slot demand does not exceed current profile slots; this entry only improves explanation/log visibility.

Rollback Notes:
- Revert the profile-note wording, added log events, and focused test from this slice to restore the previous shorter profile recommendation logs.

## 2026-05-04 - pending - ai: tune profile io by bandwidth

Status: accepted

Summary:
- Changed active profile IO slot recommendations from overlap-only sizing to bandwidth-utilization-aware sizing.
- Added safe advisory disk bandwidth sampling during `--phase-tune`: temporary read/write sampling under the tuning output directory and source-H5 read sampling when source files are available.
- Compared observed aggregate phase read/write rates against measured capacity and used that utilization to recommend profile slot increases for underuse or decreases for saturation.
- Preserved slot-overlap metrics as diagnostics so recommendations still show observed concurrent demand.
- Added disk bandwidth measurement and utilization sections to the tuning report and summary JSON.

Guardrails Consulted:
- `debug/parallelism_agent_guardrails.md`
- `debug/logging_agent_guardrails.md`
- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`

Acceptance Criteria:
- `--phase-tune` measures relevant disk capacity without mutating runtime YAML.
- Profile `h5_read_slots` / `disk_heavy_slots` recommendations are based on observed bandwidth utilization when measurements are available.
- Underused disk bandwidth can recommend increasing slots only when the selected run had more concurrent slot demand than the current profile allowed.
- Saturated disk bandwidth can recommend decreasing slots when observed concurrent demand is reducible.
- Slot-overlap metrics remain visible but are not the sole profile recommendation driver.

Expected To Run:
- Focused phase tuning unit tests for underused-bandwidth increase and saturated-bandwidth decrease cases.
- Containerized focused and broad pipeline tests.
- Limited real-data `preprocess.preprocess_segments --phase-tune` smoke.

Confirmed Not Run:
- Full dataset/well scope.
- Runtime YAML mutation.
- Destructive disk benchmarking outside the tuning artifact directory.
- Push to remote.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_phase_tuning.py` passed with `5 passed`.
- Container focused tests: `docker run --rm -w /opt/axon_reconstructor axon-recon:test python -m pytest src/axon_recon/pipeline/tests/test_phase_tuning.py` passed with `5 passed` after rebuilding the container from this source.
- Container broad tests: `docker run --rm -w /opt/axon_reconstructor axon-recon:test python -m pytest src/axon_recon/pipeline/tests --ignore=src/axon_recon/pipeline/tests/test_progress.py` passed with `324 passed`.
- Real-data smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/axon-recon-container stages preprocess.preprocess_segments --config debug/debug.runtime.yml --phase-tune --limit-datasets 1 --limit-wells-per-dataset 1 --limit-segments 2 --force-restart` passed.
- Logs inspected: latest smoke emitted `phase_tuning_profile_recommendation` with bandwidth utilization fields for run `debug.runtime-20260504T050354Z`.
- Artifacts inspected: `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/resource_tuning/resource_tuning_report.md` includes `Disk Bandwidth Measurements` and `Disk Bandwidth Utilization`; summary JSON includes `disk_bandwidth_measurements` and bandwidth utilization values.
- Diagnostics: VS Code diagnostics reported no errors for modified tuning source/test files.

CLI / Debug Flag Impact:
- No new CLI flags were added.
- Existing `--phase-tune` now performs advisory disk bandwidth sampling by default; it remains configurable through `resources.tuning` keys.

Logging / Parallelism Impact:
- Profile recommendation logs now include `h5_read_utilization` and `disk_heavy_utilization`.
- Runtime resource gates are unchanged; this is advisory tuning only.

Storage / Cache Impact:
- Creates and removes a temporary benchmark file under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/resource_tuning` during tuning.
- Reads a bounded sample from source H5 files when available.
- Updated tuning artifacts under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/resource_tuning`.

Container / NERSC / MPI Impact:
- Rebuilt local `axon-recon:local` from this source and refreshed disposable `axon-recon:test` for container tests.
- No NERSC/MPI-specific behavior was changed.

Resume / Force-Restart Impact:
- No force-restart semantics were changed.

Residual Risk And Follow-Ups:
- Disk capacity sampling is a bounded benchmark and can be affected by OS cache, current storage load, NAS behavior, and sample size.
- The smoke run observed very low process-level disk read counters for `preprocess_segments`, so H5 read utilization remained near zero despite source-H5 read capacity being measured.
- Representative multi-target tuning is still needed before accepting profile slot changes for large runs.

Rollback Notes:
- Revert the disk benchmark helpers, bandwidth pressure calculation, profile recommendation changes, report additions, and focused tests from this slice to restore overlap-only profile IO advice.

## 2026-05-04 - pending - ai: recommend profile io slots

Status: accepted

Summary:
- Extended `--phase-tune` so IO slot advice now includes active resource profile capacity recommendations, not only per-phase resource class slot demand.
- Estimated observed concurrent H5/disk-heavy slot demand from phase resource usage timestamps and wall times.
- Added report and structured-log output for active profile `h5_read_slots` and `disk_heavy_slots` recommendations.
- Kept profile recommendations advisory-only and scoped to the selected tuning run.

Guardrails Consulted:
- `debug/parallelism_agent_guardrails.md`
- `debug/logging_agent_guardrails.md`
- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`

Acceptance Criteria:
- Phase-level IO recommendations still classify whether each observed phase should consume H5 or disk-heavy slots.
- Active profile recommendations can increase slots when observed/recommended concurrent phase demand exceeds the current profile.
- Active profile recommendations can decrease slots only with enough representative observations and nonzero slot-consuming demand.
- Non-IO selected phases do not recommend zeroing global profile IO slots.
- The human-readable report and structured logs expose profile recommendations separately from phase recommendations.

Expected To Run:
- Focused phase tuning unit tests for resource class IO advice and active profile IO capacity advice.
- A small real-data `--phase-tune` smoke to confirm CLI artifact/log emission.

Confirmed Not Run:
- Full dataset/well scope.
- Runtime YAML mutation.
- Destructive disk benchmarking.
- Push to remote.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_phase_tuning.py` passed with `5 passed`.
- Container focused tests: `docker run --rm -w /opt/axon_reconstructor axon-recon:test python -m pytest src/axon_recon/pipeline/tests/test_phase_tuning.py` passed with `5 passed` after rebuilding the container from this source.
- Real-data smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/axon-recon-container stages preprocess.save_rec_metadata --config debug/debug.runtime.yml --phase-tune --limit-datasets 1 --limit-wells-per-dataset 1 --limit-segments 2` passed.
- Logs inspected: latest smoke emitted `phase_tuning_profile_recommendation` with `profile=lab_server_safe h5_read_slots=3->3 disk_heavy_slots=3->3`.
- Artifacts inspected: `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/resource_tuning/resource_tuning_report.md` includes `## Active Profile IO Slots`; summary JSON includes `active_profile_recommendation`.
- Diagnostics: VS Code diagnostics reported no errors for modified tuning source/test files.

CLI / Debug Flag Impact:
- No new CLI flags were added.
- Existing `--phase-tune` output now includes active profile IO slot advice for the selected limited scope.

Logging / Parallelism Impact:
- Added `phase_tuning_profile_recommendation` structured log event.
- Profile slot recommendations use estimated overlap from phase completion timestamps and wall times; they do not change runtime gating.

Storage / Cache Impact:
- Updated phase tuning artifacts under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/resource_tuning` during the smoke.
- No cache deletion or runtime YAML mutation was performed.

Container / NERSC / MPI Impact:
- Rebuilt local `axon-recon:local` from this source and refreshed disposable `axon-recon:test` for container tests.
- No NERSC/MPI-specific behavior was changed.

Resume / Force-Restart Impact:
- No force-restart semantics were changed.

Residual Risk And Follow-Ups:
- Profile IO recommendations are scoped to observed overlap in the selected tuning run; representative multi-target tuning is needed before lowering active profile slots.
- Disk throughput remains process-observed read/write rate, not a standalone disk capability benchmark.

Rollback Notes:
- Revert the profile recommendation helpers, report/log additions, and focused tests from this slice to restore phase-only IO recommendations.

## 2026-05-04 - pending - ai: add phase resource tuning

Status: accepted

Summary:
- Added `--phase-tune` and `--confirm-full-scope` to stage-sequence CLI runs so selected phases can emit advisory resource-class tuning outputs after a successful limited run.
- Added advisory phase tuning artifacts from actual `phase_resource_usage` records, including RAM, planned CPU, observed native thread diagnostics, and observed read/write throughput.
- Added `source_h5_path` to structured log context for source-keyed tuning analysis.
- Wrapped direct spikesort and reconstruct phase selectors in the shared one-phase resource chain so direct enabled phases emit standardized resource telemetry like full-stage phases.
- Kept tuning advisory-only; runtime YAML is not modified automatically.

Guardrails Consulted:
- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/logging_agent_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`
- `debug/parallelism_agent_guardrails.md`
- `debug/optimization_simplificaiton_guardrails.md`
- `debug/first_version_pipeline_guardrails.md`

Acceptance Criteria:
- Enabled phase selectors across preprocess, spikesort, and reconstruct emit resource usage logs when run directly or as part of a stage chain.
- `--phase-tune` runs the selected limited scope first, then bases recommendations on actual current-run resource telemetry.
- Full-scope phase tuning is refused unless `--confirm-full-scope` is provided.
- CPU/RAM recommendations are conservative and do not raise CPU for small measurement jitter around an already-covered one-core phase.
- Disk recommendations use observed phase read/write throughput and clearly avoid destructive benchmarking.

Expected To Run:
- Direct phase resource telemetry wrappers for direct spikesort phase selectors and direct reconstruct phase selectors.
- Advisory artifact generation for selected stages/phases with emitted `phase_resource_usage` records.
- Container tests and a limited real-data phase-tune smoke.

Confirmed Not Run:
- Full dataset/well scope without explicit confirmation.
- Automatic runtime YAML mutation.
- Destructive disk benchmarking.
- Push to remote.

Validation:
- Focused tests: `docker run --rm -w /opt/axon_reconstructor axon-recon:test python -m pytest src/axon_recon/pipeline/tests/test_phase_tuning.py src/axon_recon/pipeline/tests/test_cli_stage_sequence.py src/axon_recon/pipeline/tests/test_spikesort_target_status.py::test_run_spikesort_sort_from_runtime_wraps_direct_phase_in_resource_chain src/axon_recon/pipeline/tests/test_reconstruct_target_status.py::test_run_reconstruct_direct_phase_wraps_resource_chain` passed with `132 passed`.
- Broad tests: `docker run --rm -w /opt/axon_reconstructor axon-recon:test python -m pytest src/axon_recon/pipeline/tests --ignore=src/axon_recon/pipeline/tests/test_progress.py` passed with `322 passed`.
- Real-data smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/axon-recon-container stages preprocess.save_rec_metadata --config debug/debug.runtime.yml --phase-tune --limit-datasets 1 --limit-wells-per-dataset 1 --limit-segments 2` passed.
- Logs inspected: latest structured log recorded `phase_resource_usage` for `preprocess.save_rec_metadata.save_rec_metadata` and `phase_tuning_recommendation` with `cpu_cores=1->1`.
- Artifacts inspected: `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/resource_tuning/resource_usage_observations.jsonl`, `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/resource_tuning/resource_tuning_summary.json`, and `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/resource_tuning/resource_tuning_report.md` for run `debug.runtime-20260504T035459Z`.
- Diagnostics: VS Code diagnostics reported no errors for modified source/test files.
- Not run: full pipeline real-data phase tuning; earlier host all-stage smoke was intentionally not treated as validation because host Kilosort dependencies were missing and the run was interrupted.

CLI / Debug Flag Impact:
- Stage sequence parsers now accept `--phase-tune` and `--confirm-full-scope`.
- `--phase-tune` requires an explicit scope limit or full-scope confirmation before stages run.
- Existing debug limit flags continue to define the phase-tuning observation scope.

Logging / Parallelism Impact:
- Direct spikesort/reconstruct phase selectors now pass through `run_phase_chain` with phase resource classes and pipeline thread counts.
- Phase tuning consumes structured `phase_resource_usage` records and writes `phase_tuning_started`, `phase_tuning_recommendation`, and `phase_tuning_completed` log events.
- Source H5 path is now part of log context for keyed IO/resource analysis.

Storage / Cache Impact:
- Created/updated tuning artifacts under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/resource_tuning`.
- Updated structured pipeline logs under `/mnt/disk15tb/adamm/scratch/axon_recon_scratch/outputs/logs`.
- No cache deletion or runtime YAML mutation was performed.

Container / NERSC / MPI Impact:
- Rebuilt local `axon-recon:local` from this source.
- Recreated disposable `axon-recon:test` by installing pytest on top of the rebuilt local image for containerized tests.
- No NERSC/MPI-specific behavior was changed.

Resume / Force-Restart Impact:
- No force-restart semantics were intentionally changed.
- Phase tuning observes the completed selected run and can be repeated; artifacts are overwritten with the latest tuning summary/report.

Residual Risk And Follow-Ups:
- Full-suite collection still has an unrelated existing `src/axon_recon/pipeline/tests/test_progress.py` `TabError`, so broad pipeline tests were run with that file ignored.
- Preprocess direct phase console messages still display the stage-qualified phase name as `preprocess.save_rec_metadata.save_rec_metadata`; the tuning report de-duplicates that heading.
- Current disk IO guidance is based on process-level read/write counters, not a standalone disk capability benchmark.

Rollback Notes:
- Revert the phase tuning module, CLI flags, logging context field, direct resource-chain wrappers, and associated tests from this slice to restore previous telemetry/tuning behavior.

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
