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

## 2026-05-07 13:01 - pending - ai: integrate task allocation previews

Status: accepted

Summary:
- Integrated task allocation plans at the shared runtime target distribution boundary when `resources.task_allocation.enabled=true`.
- Added per-target task-slot context so workers can observe their assigned slot, while preserving existing distribution behavior when no allocation plan is attached.
- Added `--alloc` to `axon-recon stage/stages` so selected stages print allocation details and return without invoking stage handlers; the container wrapper forwards this flag unchanged.

Guardrails Consulted:
- `debug/parallelism_agent_guardrails.md`
- `debug/container_mpi4py_NERSC_optimization_guardrails.md`
- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/logging_agent_guardrails.md`
- `debug/first_version_pipeline_guardrails.md`

Acceptance Criteria:
- Task allocation remains opt-in and disabled configs behave as before.
- Runtime target distribution clamps active workers to available task slots and exposes the current task slot to target-local workers.
- `--alloc` computes selected targets, resource classes, resolved parallelism, and allocation details without running stage work.
- `axon-recon-container ... --alloc` forwards the flag to the pipeline command.

Expected To Run:
- Focused unit tests for CPU allocation, target distribution, runtime logging context, stage-sequence CLI parsing, and container wrapper argument forwarding.
- Read-only/safe dry preview smoke commands only.

Confirmed Not Run:
- No preprocess, spikesort, reconstruct, analyzer, MPI, or real container stage work was launched by the agent.
- `--alloc` smoke returned before any stage handler execution.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_cpu_allocation.py src/axon_recon/pipeline/tests/test_distributor.py src/axon_recon/pipeline/tests/test_logging_context.py src/axon_recon/pipeline/tests/test_cli_stage_sequence.py src/axon_recon/pipeline/tests/test_container_cli.py` -> 182 passed.
- Broader stage target-status tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_preprocess_target_status.py src/axon_recon/pipeline/tests/test_spikesort_target_status.py src/axon_recon/pipeline/tests/test_reconstruct_target_status.py` -> 103 passed.
- Pipeline test directory excluding the known malformed progress test: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests --ignore=src/axon_recon/pipeline/tests/test_progress.py` -> 390 passed.
- Existing focused test gap: including `src/axon_recon/pipeline/tests/test_progress.py` currently fails at collection with a pre-existing `TabError` on line 76; not modified in this slice.
- Real-data smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m axon_recon.pipeline.cli stages reconstruct.report_templates --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells 1 --alloc` -> printed allocation preview and completed without stage work.
- Container wrapper smoke: `axon-recon-container --no-build --dry-run stages reconstruct.report_templates --config debug/debug.runtime.yml --alloc` -> resolved Docker command ending in `axon-recon:local stages reconstruct.report_templates --config debug/debug.runtime.yml --alloc`.
- Diagnostics: VS Code `get_errors` on touched source and test files -> no errors.
- Logs inspected: focused pytest output, allocation preview smoke output, and container dry-run output.
- Artifacts inspected: none.
- Not run: full pipeline test suite, any real stage command, any non-dry-run container command, and MPI/Slurm commands.

CLI / Debug Flag Impact:
- Added `--alloc` to `stage` and `stages` commands.
- The flag supports existing stage selectors, debug limit flags, unit filters, force flags, and `--target-datasets`, then returns after printing a preview.
- `--phase-tune` monitoring is disabled for allocation-only previews.

Logging / Parallelism Impact:
- When a task allocation plan is attached, runtime topology logging includes backend, bind mode, CPUs per task, task count, slots, and CPU capacity.
- Target logs include the assigned task slot id and CPU set.
- Distribution now accepts optional task slots and schedules at most one active target per slot.

Storage / Cache Impact:
- Created: none.
- Modified: `src/axon_recon/pipeline/cli.py`, `src/axon_recon/pipeline/cpu_allocation.py`, `src/axon_recon/pipeline/execution/context.py`, `src/axon_recon/pipeline/execution/distributor.py`, `src/axon_recon/pipeline/runner.py`, focused tests, and this commit log.
- Removed: none.

Container / NERSC / MPI Impact:
- Container wrapper argument handling remains pass-through; added focused coverage for `--alloc` forwarding.
- No MPI or Slurm behavior was added in this slice.
- Local-affinity plans are computed from current process-visible topology, preserving NERSC-shaped task-slot concepts without requiring MPI.

Resume / Force-Restart Impact:
- `--alloc` accepts force flags for preview parity but does not run or restart stage work.
- Runtime execution remains resume/force-restart driven by the underlying stage handlers when `--alloc` is absent.

Residual Risk And Follow-Ups:
- `--alloc` currently reports planned CPU slots but does not yet enforce OS CPU affinity or nested thread environment variables inside workers.
- Direct preview target selection can still log existing scratch input reuse while computing targets; it does not materialize copy work.
- The pre-existing `test_progress.py` indentation error should be fixed separately if that test file is needed in the validation set.

Rollback Notes:
- Revert the commit to remove the `--alloc` CLI path, task-slot context propagation, runtime plan attachment, and focused tests.

## 2026-05-07 00:45 - pending - ai: add task allocation plan builder

Status: accepted

Summary:
- Added pure task-allocation planning dataclasses and slot-building logic in `src/axon_recon/pipeline/cpu_allocation.py` without wiring the plan into live stage execution yet.
- The plan builder consumes the parsed `TaskAllocationConfig`, detected `CpuTopology`, optional `StageParallelism`, and optional RAM or `/dev/shm` capacity inputs.
- The implementation currently supports `backend=local_affinity`, returns no plan when task allocation is disabled, computes capacity from CPU units first, applies reserve and resource-cap clamps, and builds concrete `TaskSlot` CPU sets for later integration.

Guardrails Consulted:
- `debug/parallelism_agent_guardrails.md`
- `debug/container_mpi4py_NERSC_optimization_guardrails.md`
- `debug/first_version_pipeline_guardrails.md`

Acceptance Criteria:
- Disabled task allocation returns no plan.
- The 24-core lab-style topology with `cpus_per_task=4`, `use_hyperthreads=false` yields 6 slots.
- `reserve_cpus=2` reduces the same topology to 5 slots.
- Explicit `tasks_per_node` is clamped to available capacity under the current clear-policy choice.
- RAM and `/dev/shm` per-task limits can reduce effective task count below CPU capacity.

Expected To Run:
- Focused CPU topology and task-allocation-plan unit tests only.

Confirmed Not Run:
- No preprocess, spikesort, reconstruct, analyzer, containerized pipeline, or MPI work was launched by the agent.
- No runtime integration into target distribution or worker affinity was attempted in this slice.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_cpu_allocation.py` → 11 passed.
- Diagnostics: VS Code `get_errors` on `src/axon_recon/pipeline/cpu_allocation.py` and `src/axon_recon/pipeline/tests/test_cpu_allocation.py` → no errors.
- Real-data smoke: not run.
- Logs inspected: focused pytest output for the plan builder slice.
- Artifacts inspected: none.
- Not run: broader pipeline tests and any stage command that could interfere with Adam’s active analyzer work.

CLI / Debug Flag Impact:
- None in this slice. The existing `systopo` command remains unchanged.

Logging / Parallelism Impact:
- No runtime parallelism behavior change yet.
- Added `TaskSlot` and `TaskAllocationPlan` dataclasses for the next integration slice.

Storage / Cache Impact:
- Created: none.
- Modified: `src/axon_recon/pipeline/cpu_allocation.py`, `src/axon_recon/pipeline/tests/test_cpu_allocation.py`, `debug/agent_guardrails_commit_notes.md`.
- Removed: none.

Container / NERSC / MPI Impact:
- No runtime container, MPI, or NERSC behavior changes.
- The plan builder now accepts optional RAM and `/dev/shm` capacities so the later container/local-affinity slice can clamp tasks without changing the planner interface.

Resume / Force-Restart Impact:
- None.

Residual Risk And Follow-Ups:
- The current implementation clamps explicit `tasks_per_node` to capacity rather than failing; if Adam wants strict validation later, that should be a deliberate policy switch.
- `bind=logical_cpus` currently treats `reserve_cpus` and `cpus_per_task` as logical-CPU units, while physical-core bindings treat them as core units; that distinction should be kept explicit in later logging.
- The next slice should integrate the plan at the target distribution boundary and clamp effective `well_workers` to the number of planned slots.

Rollback Notes:
- Revert the commit to remove the pure plan builder and its focused tests.

## 2026-05-07 00:25 - pending - ai: add systopo cpu topology command

Status: accepted

Summary:
- Added a read-only CPU topology detector in `src/axon_recon/pipeline/cpu_allocation.py` that derives visible CPUs from process affinity, reads per-CPU topology from sysfs when available, and falls back to logical-CPU-only grouping with a warning when sysfs data is unavailable.
- Added `systopo` to the main CLI so `axon-reconstructor systopo` prints a user-reviewable topology report without requiring a runtime config or starting any pipeline stage work.
- Confirmed the existing container wrapper already forwards config-free commands, added coverage for `axon-recon-container systopo`, and added `axon-recon` as a package script alias in project metadata for future installs.

Guardrails Consulted:
- `debug/parallelism_agent_guardrails.md`
- `debug/container_mpi4py_NERSC_optimization_guardrails.md`
- `debug/first_version_pipeline_guardrails.md`

Acceptance Criteria:
- Unit tests cover sysfs topology detection for 1 socket, 24 cores, 2 threads per core.
- Unit tests cover cpuset-restricted visibility and missing-sysfs fallback.
- A direct CLI command prints topology information for user review without requiring config.
- The container wrapper accepts and forwards the config-free `systopo` command shape.

Expected To Run:
- Focused topology detector tests, focused CLI/container tests, and read-only topology command checks only.

Confirmed Not Run:
- No preprocess, spikesort, reconstruct, analyzer, containerized pipeline stage, or MPI work was launched by the agent.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_cpu_allocation.py src/axon_recon/pipeline/tests/test_cli_stage_sequence.py src/axon_recon/pipeline/tests/test_container_cli.py` → 161 passed.
- Diagnostics: VS Code `get_errors` on touched topology and CLI files → no errors.
- Host command smoke: `axon-reconstructor systopo | head -n 20` → printed live visible CPU topology for the current machine.
- Container wrapper smoke: `axon-recon-container --no-build --dry-run systopo` → resolved a config-free `docker run ... axon-recon:local systopo` command.
- Logs inspected: focused pytest output and the live host `systopo` sample output.
- Artifacts inspected: none.
- Not run: editable reinstall or container execution, to avoid unnecessary churn while Adam had an analyzer phase running.

CLI / Debug Flag Impact:
- Added `systopo` as a top-level read-only CLI command.
- Added `axon-recon` as a package script alias in `pyproject.toml` for future installs.
- Created a local workstation symlink `/home/adamm/miniconda3/bin/axon-recon -> /home/adamm/miniconda3/bin/axon-reconstructor` so the short alias works immediately without reinstalling during the active analyzer run.

Logging / Parallelism Impact:
- No stage parallelism behavior changes yet.
- `systopo` currently prints a topology report and inherits the standard pipeline start/completion log lines.

Storage / Cache Impact:
- Created: `src/axon_recon/pipeline/cpu_allocation.py`, `src/axon_recon/pipeline/tests/test_cpu_allocation.py`.
- Modified: `src/axon_recon/pipeline/cli.py`, `src/axon_recon/pipeline/tests/test_cli_stage_sequence.py`, `src/axon_recon/pipeline/tests/test_container_cli.py`, `pyproject.toml`, `debug/agent_guardrails_commit_notes.md`.
- Removed: none.

Container / NERSC / MPI Impact:
- No MPI or NERSC runtime behavior changes.
- The container wrapper now has focused coverage for a config-free read-only topology command, which is useful for container visibility checks before later affinity work.

Resume / Force-Restart Impact:
- None.

Residual Risk And Follow-Ups:
- The live topology report shows kernel-provided core IDs, which may not be contiguous or numerically ordered by visible CPU range.
- The next slice should turn this topology object into a task-slot allocation plan without changing stage execution when task allocation is disabled.

Rollback Notes:
- Revert the commit to remove the detector, the `systopo` command, and the metadata alias.

## 2026-05-06 21:45 - pending - ai: add task allocation config schema

Status: accepted

Summary:
- Added a typed `resources.task_allocation` schema to the central pipeline resource parser.
- Kept the new schema fully additive and default-disabled so existing runtime YAML keeps current behavior.
- Added focused parser coverage for defaults, explicit `local_affinity` config, and invalid enum or numeric values.

Guardrails Consulted:
- `debug/parallelism_agent_guardrails.md`
- `debug/container_mpi4py_NERSC_optimization_guardrails.md`
- `debug/first_version_pipeline_guardrails.md`

Acceptance Criteria:
- Existing runtime YAML remains valid with no behavior change when `resources.task_allocation` is absent.
- Invalid `resources.task_allocation` enum and numeric values fail clearly during parsing.
- Focused tests cover defaults, explicit config, and invalid values.

Expected To Run:
- Central resource parser updates and focused parser tests only.

Confirmed Not Run:
- No preprocess, spikesort, reconstruct, analyzer, container, MPI, or real-data runtime work was launched by the agent.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/tests/test_resources.py` → 20 passed.
- Diagnostics: VS Code `get_errors` on `src/axon_recon/pipeline/resources.py` and `src/axon_recon/pipeline/tests/test_resources.py` → no errors.
- Real-data smoke: not run.
- Logs inspected: pytest output for the resource parser slice.
- Artifacts inspected: none beyond repository source and test code.
- Not run: container smoke, broader pipeline tests, and any command that could interfere with the user’s active analyzer work.

CLI / Debug Flag Impact:
- None yet. This slice adds typed runtime config parsing only.

Logging / Parallelism Impact:
- None yet at runtime. This slice only introduces the parsed schema needed for later local-affinity and scheduler-shaped allocation work.

Storage / Cache Impact:
- Created: none.
- Modified: `src/axon_recon/pipeline/resources.py`, `src/axon_recon/pipeline/tests/test_resources.py`, `debug/agent_guardrails_commit_notes.md`.
- Removed: none.

Container / NERSC / MPI Impact:
- No runtime impact. The new schema recognizes future `local_affinity`, `mpi`, and `slurm` backends without enabling any of them.

Resume / Force-Restart Impact:
- None.

Residual Risk And Follow-Ups:
- The schema currently defaults to a disabled allocation block and does not yet validate cross-field semantics such as `enabled=true` plus missing capacity decisions.
- The next slice should use this schema for topology detection and plan construction without changing existing non-allocation execution.

Rollback Notes:
- Revert the commit to remove the additive schema and its parser tests.

## 2026-05-06 21:15 - pending - ai: document NERSC shaped local affinity plan

Status: accepted

Summary:
- Added `debug/nersc_shaped_local_affinity_plan.md`, a sequential planning note for evolving local pipeline parallelism toward NERSC-shaped task allocation.
- The plan keeps local CPU affinity as the first backend and defers MPI/Slurm until the task allocation abstraction is stable.
- The note captures config shape, CPU topology detection, task slot planning, target-distribution integration, worker affinity, nested thread env, logging, phase-tune metadata, container smoke, and later MPI/Slurm backends.

Guardrails Consulted:
- `debug/parallelism_agent_guardrails.md`
- `debug/container_mpi4py_NERSC_optimization_guardrails.md`
- `debug/optimization_simplificaiton_guardrails.md`

Acceptance Criteria:
- Debug folder gains a markdown roadmap for sequential implementation slices.
- Existing locked guardrail docs remain unchanged.
- The plan preserves non-MPI default behavior and marks NERSC validation as deferred.

Expected To Run:
- Documentation-only update.

Confirmed Not Run:
- No pipeline stages, real-data smoke, container builds, or MPI commands were launched by the agent for this slice.

Validation:
- Focused tests: not run; documentation-only change.
- Real-data smoke: not run.
- Logs inspected: not applicable.
- Artifacts inspected: existing debug guardrail notes for local style and acceptance vocabulary.
- Not run: full test suite and container smoke.

CLI / Debug Flag Impact:
- None. The document proposes future CLI override vocabulary but does not implement it.

Logging / Parallelism Impact:
- None in code. The document proposes future task-allocation logs and phase-tune metadata.

Storage / Cache Impact:
- Created: `debug/nersc_shaped_local_affinity_plan.md`.
- Modified: `debug/agent_guardrails_commit_notes.md`.
- Removed: none.

Container / NERSC / MPI Impact:
- No runtime impact. The document recommends local CPU affinity first, MPI later, and Slurm/NERSC last after real validation.

Resume / Force-Restart Impact:
- None.

Residual Risk And Follow-Ups:
- The first implementation slice should verify whether current well target distribution uses threads or processes before applying per-worker affinity, because process-wide affinity is unsafe for independent thread workers.
- Follow-up implementation should begin with config parsing and topology detection only.

Rollback Notes:
- Revert the commit to remove the planning note and its commit-log entry.

## 2026-05-06 12:05 - pending - ai: fix plot templates v2 sizing and latency scale

Status: accepted

Summary:
- Fixed `plot_templates_v2` marker sizing so the raw per-channel metric range maps onto the configured visible marker-size range, and corrected the config parser to honor `render.marker_min_size` / `render.marker_max_size` from runtime YAML.
- Corrected v2 latency coloring to use the persisted effective per-unit sampling rate when available, which fixes the 10x delay-scale inflation on upsampled templates, and reversed latency coloring so smaller delays use the yellow end of the colormap.
- Restored the v1-style scale-circle concept to v2 and enabled it in the active `debug/debug.runtime.yml` block used for the current smoke path.

Guardrails Consulted:
- `debug/stage_and_phase_behavior_guardrails.md`
- `debug/logging_agent_guardrails.md`
- `debug/parallelism_agent_guardrails.md`

Acceptance Criteria:
- `plot_templates_v2` honors the configured marker size range from the active runtime YAML.
- Latency colors and colorbar direction in v2 match the intended semantics for upsampled templates.
- v2 renders include a scale circle using the same conceptual reference as v1 without reintroducing overlap logic.

Expected To Run:
- Only the direct `plot_templates_v2` config/render/phase slice and its focused unit tests.
- No analyzer/template rebuilds, no reconstruct/GTR phases, and no real-data heavy compute.

Confirmed Not Run:
- No analyzer, build_templates, or generate_gtrs work.
- No real-data CLI smoke was launched from the agent.
- No overlap-resolution loop or non-overlapping size pass was added to v2.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_config.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_render.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py -k "plot_templates_v2_phase_block or render_template_circles_plot_v2 or plot_templates_v2_phase_writes_direct_outputs"` → 5 passed, 191 deselected.
- Focused tests: targeted narrow v2 render/config slice also passed earlier during iteration.
- Lint: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m ruff check src/axon_recon/pipeline/stages/reconstruct/templates/models/inputs.py src/axon_recon/pipeline/stages/reconstruct/templates/config.py src/axon_recon/pipeline/stages/reconstruct/templates/core/render.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_config.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_render.py --ignore I001,F821,F841,E501,UP035,UP015,UP037,UP028,UP034,B009,B010,E731,E101` → passed.
- Real-data smoke: not run by the agent; Adam indicated he would run smoke tests separately.
- Logs inspected: pytest failures/success output for the config/render/phase v2 slice.
- Artifacts inspected: rendered v2 png/svg test outputs and unit summary metadata lookup behavior.
- Not run: broad reconstruct CLI and full test suite.

CLI / Debug Flag Impact:
- No CLI interface changes.
- Active debug runtime now explicitly enables the v2 scale circle and reversed latency colorbar for smoke testing.

Logging / Parallelism Impact:
- No logging flow or parallelism behavior changes.

Storage / Cache Impact:
- Created: none in repository code beyond test temp artifacts.
- Modified: v2 per-unit plot rendering behavior and the active debug runtime YAML block.
- Removed: none.

Container / NERSC / MPI Impact:
- None.

Resume / Force-Restart Impact:
- None; this slice only affects v2 template plotting and its runtime config.

Residual Risk And Follow-Ups:
- Real-data smoke is still the right follow-up to confirm the latency scale now visually matches the v1 reference unit on the current run.
- The v2 latency colormap now auto-reverses for latency, which matches the requested behavior but may be worth making explicitly user-toggleable in future if both directions are needed.

Rollback Notes:
- Revert the commit to restore the prior v2 sizing/latency behavior and remove the runtime scale-circle/colorbar tweaks.

## 2026-05-06 11:47 - pending - ai: add analyzer unit manifests for build resume

Status: accepted

Summary:
- Added analyzer source-unit manifest artifacts under `templates_outputs/context/analyzer_source_units/*.json` during the reconstruct analyzers phase.
- Updated reconstruct build_templates to load or backfill those manifests from cached analyzers, skip absent unit/source pairs before launching payload workers, and reuse complete source payload artifacts.
- Added build artifact resume in the core unit builder so completed unit template artifacts are counted as reused without rebuilding after cancellation.

Guardrails Consulted:
- `debug/stage_and_phase_behavior_guardrails.md`
- `debug/logging_agent_guardrails.md`
- `debug/parallelism_agent_guardrails.md`

Acceptance Criteria:
- Direct `reconstruct.analyzers` writes per-source unit membership artifacts without broad analyzer/template recompute.
- Direct `reconstruct.build_templates` rerun without `--force-restart` backfills missing manifest artifacts, fills missing source payloads, skips known-absent unit/source payload workers, and reuses completed unit outputs.
- Process-pool lazy payload materialization logs preflight skip/reuse outcomes before submitting real worker jobs.

Expected To Run:
- Analyzer manifest write/backfill for selected cached analyzer sources.
- Missing source payload materialization only for unit/source pairs present in the source manifest.
- Core unit template build only for units without complete materialized template artifacts and per-unit summary.

Confirmed Not Run:
- `extract_template_segments` support was not reintroduced.
- `compute_template_similarity` behavior was not changed.
- Legacy source payload paths were not restored.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py -k "analyzers_phase_logs_settings_and_writes_run_stats or build_templates_phase_loads_cached_analyzers_when_payloads_missing or build_templates_phase_lazy_loads_cached_analyzers_per_unit or build_templates_phase_uses_unit_manifests_for_lazy_dispatch or build_templates_phase_resumes_partial_source_payloads or build_templates_phase_reuses_completed_unit_artifacts or build_templates_phase_uses_unit_workers or cached_analyzer_process_materialization_logs_each_unit_source_future or parallel_lazy_materialization_filters_labels"` → 9 passed, 53 deselected.
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_io.py` → 67 passed.
- Lint: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m ruff check src/axon_recon/pipeline/stages/reconstruct/phases/analyzers.py src/axon_recon/pipeline/stages/reconstruct/phases/build_templates.py src/axon_recon/pipeline/stages/reconstruct/templates/core/build_templates.py src/axon_recon/pipeline/stages/reconstruct/templates/source_units.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py --ignore I001,F821,F841,E501,UP035,UP015,UP037,UP028,B009,B010,E731,E101` → passed.
- Diagnostics: VS Code `get_errors` on touched Python files → no errors.
- Real-data smoke: not run; this change is covered by focused runner/IO tests to avoid launching real analyzer/template compute in the current workspace.
- Logs inspected: unit test logs asserting analyzer manifest writes and preflight `skipped_unit_absent_preflight` logs.
- Artifacts inspected: test-created manifest JSON and materialized source/unit cache artifacts.
- Not run: full real-data reconstruct CLI.

CLI / Debug Flag Impact:
- No CLI flags or YAML schema changes.

Logging / Parallelism Impact:
- Added analyzer/build_templates logs when source unit manifests are written or backfilled.
- Lazy build_templates process scheduling performs manifest/cache preflight in the parent process, reducing submitted unit/source payload worker jobs for known-absent or already-materialized payloads.

Storage / Cache Impact:
- Created: additive `context/analyzer_source_units/<urlquoted_source_name>.json` artifacts with source name, source kind, unit ids, and unit counts.
- Modified: build_templates source payload cache reuse now validates per-unit source payload artifacts before scheduling payload work.
- Removed: none.

Container / NERSC / MPI Impact:
- No container, MPI, or scheduler changes.

Resume / Force-Restart Impact:
- Rerunning analyzers/build_templates without `--force-restart` creates missing source-unit manifest artifacts and missing source payloads while preserving existing complete work.
- `--force-restart` continues to clear build_templates-owned materialized template/source payload outputs before rebuilding.

Residual Risk And Follow-Ups:
- Manifest backfill still loads cached analyzers once per missing source manifest; the intended steady state avoids this after the additive artifacts exist.
- Real-data smoke remains useful before large production runs because unit tests cannot exercise SpikeInterface disk analyzer loading cost end to end.

Rollback Notes:
- Revert the commit to stop producing/consuming source-unit manifests; existing additive JSON artifacts can be left in place because older code ignores them.

## 2026-05-06 - pending - ai: add lightweight plot templates v2

Status: accepted

Summary:
- Added a separate `plot_templates_v2` template phase that loads current `cache/templates/{merged,full}` artifacts directly and renders one minimal circle plot per selected unit without calling the legacy overlap/layout adjustment renderer path.
- Added typed v2 knobs for output path/DPI, channel scope, marker sizing/coloring, explicit plot limits/padding, fixed title/unit/coordinate text positions, direct colorbar axes, and scale-bar placement.
- Wired `reconstruct.plot_templates_v2` through config parsing, reconstruct/template phase dispatch, direct runtime execution, and the top-level CLI stage selector.
- Filled the debug runtime `plot_templates_v2` block while leaving it out of the full reconstruct `phase_sequence` so it can be tested as an explicit direct phase.
- Added focused coverage for v2 config parsing, renderer output without overlap sizing, phase summary/unit-summary writes, reconstruct phase dispatch, debug-runtime parsing, and CLI selector parsing.

Guardrails Consulted:
- `debug/optimization_simplificaiton_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`
- `debug/cli_debug_flags_agent_guardrails.md`
- `debug/logging_agent_guardrails.md`

Acceptance Criteria:
- `stages reconstruct.plot_templates_v2` resolves to a direct reconstruct template phase.
- V2 plotting has no overlap checks, no iterative zoom-out loop, and no final non-overlapping circle sizing pass.
- V2 plot decoration is controlled by explicit inclusion and position knobs rather than automatic overlap logic.
- The phase writes per-unit output paths and a phase summary without running real-data CLI during this slice.

Expected To Run:
- Focused pytest coverage for v2 render/config/phase/CLI dispatch.
- Syntax compile and focused Ruff checks on touched Python files.
- VS Code diagnostics for touched Python/YAML files.

Confirmed Not Run:
- Real-data CLI or container smoke; Adam planned to test in the CLI after implementation.
- Full repository test suite.
- Container image rebuild.
- Remote push.

Validation:
- Syntax: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m py_compile src/axon_recon/pipeline/stages/reconstruct/templates/core/render.py src/axon_recon/pipeline/stages/reconstruct/phases/plot_templates_v2.py src/axon_recon/pipeline/stages/reconstruct/templates/config.py src/axon_recon/pipeline/stages/reconstruct/templates/models/inputs.py src/axon_recon/pipeline/stages/reconstruct/runner.py src/axon_recon/pipeline/runner.py src/axon_recon/pipeline/cli.py src/axon_recon/pipeline/stages/reconstruct/cli.py` passed.
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_render.py::test_render_template_circles_plot_v2_writes_outputs_without_overlap_pass src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_config.py::test_load_templates_config_parses_plot_templates_v2_phase_block src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py::test_run_reconstruct_templates_plot_templates_v2_phase_writes_direct_outputs src/axon_recon/pipeline/stages/reconstruct/tests/test_runner.py::test_reconstruct_phase_resolver_handles_templates_phases src/axon_recon/pipeline/stages/reconstruct/tests/test_config.py::test_load_config_reconstruct_populates_templates_inputs_from_debug_runtime src/axon_recon/pipeline/tests/test_cli_stage_sequence.py::test_parse_stage_list_tokens_supports_reconstruct_phase_tokens` passed, 33 tests.
- Lint: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m ruff check --select F,I --ignore I001,F821,F841 ...` passed for touched Python files. `F821`/`F841` were ignored for known pre-existing debt in the large render test/helper files.
- Diagnostics: VS Code diagnostics reported no errors for touched Python/YAML files.
- Real-data smoke: not run by request; Adam will test via CLI.
- Logs inspected: none for this slice.
- Artifacts inspected: no real output artifacts inspected or modified.
- Not run: no real-data reconstruct, no cache clear, no full suite.

CLI / Debug Flag Impact:
- New direct selector: `reconstruct.plot_templates_v2` plus `recon.*`, `reconstruction.*`, and `templates_*` aliases.
- Existing debug/target flags flow through the shared reconstruct runtime path for the new phase.
- `debug/debug.runtime.yml` now contains a `plot_templates_v2` knob block and keeps the full reconstruct sequence from running it unless selected explicitly.

Logging / Parallelism Impact:
- Adds start, per-unit render, summary, and run-stats logs for `templates.plot_templates_v2`.
- No new inner plot parallelism; v2 renders units sequentially inside the direct phase handler.

Storage / Cache Impact:
- Created: per-unit `template_circles_v2.{png,svg}` paths when requested and `context/plot_templates_v2_summary.json`.
- Modified: per-unit `unit_templates_summary.json` gains `template_circles_v2` output metadata when the phase runs.
- Removed: none.

Container / NERSC / MPI Impact:
- No Dockerfile, container wrapper, NERSC, or MPI changes.

Resume / Force-Restart Impact:
- V2 skips existing requested PNG/SVG outputs unless `force_restart`, `force_replot`, or `force_replot_per_unit` is set.
- No cache deletion or template rebuild behavior changed.

Residual Risk And Follow-Ups:
- V2 intentionally trades automatic label/colorbar/scale-bar overlap protection for simpler fixed-position controls, so visual tuning is now a YAML responsibility.
- A one-unit real-data CLI smoke is still needed to inspect the actual formatting and timing.

Rollback Notes:
- Revert the v2 phase/config/CLI wiring and remove the `plot_templates_v2` block from the debug runtime to return to the previous plot-template surface.

## 2026-05-06 - pending - ai: restrict plot templates resources

Status: accepted

Summary:
- Removed legacy template artifact resolution fallbacks from reconstruct phase environment setup, templates runner materialized roots, and clear-template-cache root selection.
- Downstream reconstruct phases now require the configured/current `cache/templates/{merged,full}` layout, so missing-template errors point at `recon_outputs/cache/templates/merged` for the active runtime.
- Forced `templates.plot_templates` unit rendering to run sequentially inside a phase lease, ignoring parallel unit worker overrides that made recent full-well plotting much slower.
- Tightened the debug runtime to one plot slot, `plot_unit.ram_gb: 48`, and `plot_templates.resources.unit_procs: 1` so plot work does not overlap with template-build/analyzer-heavy phases on the lab-server-safe profile.
- Added focused tests for current-cache resolution, legacy-root rejection, clear-cache root selection, sequential plot planning, and RAM-based plot/build exclusion.

Guardrails Consulted:
- `debug/parallelism_agent_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`
- `debug/logging_agent_guardrails.md`

Acceptance Criteria:
- GTR/downstream reconstruct template discovery no longer falls through to `template_outputs`, `templates_outputs`, `stg4_templates_outputs`, `templates`, `merged_units`, or `full_channels_templates` layouts.
- Future plot-template phases are serialized at the profile slot level and within the per-unit plot loop.
- The resource budget blocks a `plot_unit` lease while a `template_build` lease holds RAM under the debug lab-server-safe budget.
- Validation uses focused pytests and static checks only; no real-data reconstruct run starts while the user's build_templates run is active.

Expected To Run:
- Focused Ruff import/undefined-name checks for touched Python files.
- Focused pytest coverage for template path resolution, clear-cache root selection, plot execution planning, and resource gating.

Confirmed Not Run:
- Real-data CLI runs or smokes.
- Full repository test suite.
- Cache-clearing phase against real outputs.
- Container image rebuild.
- Remote push.

Validation:
- Lint: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m ruff check --select F,I --ignore F401 src/axon_recon/pipeline/stages/reconstruct/runner.py src/axon_recon/pipeline/stages/reconstruct/templates/runner.py src/axon_recon/pipeline/stages/reconstruct/phases/plot_templates.py src/axon_recon/pipeline/stages/reconstruct/phases/clear_templates_cache.py src/axon_recon/pipeline/stages/reconstruct/core/clear_templates_cache.py src/axon_recon/pipeline/stages/reconstruct/tests/test_runner.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py src/axon_recon/pipeline/stages/reconstruct/tests/test_clear_templates_cache.py src/axon_recon/pipeline/tests/test_resource_budget.py` passed. `F401` was ignored because the large templates runner has pre-existing unused-import debt unrelated to this change.
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/reconstruct/tests/test_runner.py::test_resolve_templates_dirs_supports_cached_templates_layout src/axon_recon/pipeline/stages/reconstruct/tests/test_runner.py::test_resolve_templates_dirs_prefers_configured_templates_output_root src/axon_recon/pipeline/stages/reconstruct/tests/test_runner.py::test_resolve_templates_dirs_ignores_legacy_template_roots src/axon_recon/pipeline/stages/reconstruct/tests/test_runner.py::test_prepare_reconstruct_environment_filters_units_by_templates_unit_labels src/axon_recon/pipeline/stages/reconstruct/tests/test_clear_templates_cache.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py::test_resolve_plot_templates_execution_plan_is_sequential src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py::test_resolve_plot_templates_execution_plan_ignores_parallel_resource_overrides src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py::test_run_reconstruct_templates_plot_batches_runs_units_sequentially src/axon_recon/pipeline/tests/test_resource_budget.py::test_phase_budget_blocks_plot_unit_when_template_build_holds_ram -q` passed.
- Diagnostics: VS Code diagnostics reported no errors for touched Python/YAML files.
- Logs inspected: existing scratch logs showed recent `plot_templates` run `debug.runtime-20260506T075401Z` using `worker_count=2`, progressing in pairs roughly every 18 minutes, while earlier sequential plot logs were closer to three minutes per unit. Latest resource-gate logs included CPU/RAM/plot demands, but `plot_unit.ram_gb=24` still allowed overlap with several 8 GB template builds.
- Artifacts inspected: no output artifacts modified.
- Not run: no real-data smoke by explicit user request.

CLI / Debug Flag Impact:
- No CLI flag changes.
- `debug/debug.runtime.yml` now uses one plot slot, a 48 GB `plot_unit` reservation, and one `plot_templates` unit proc.

Logging / Parallelism Impact:
- Plot-template execution-plan logs will report `plot_unit_workers=1`, `unit_procs=1`, and `parallel=false` even if higher unit worker overrides are present.
- No new log fields or logger names.

Storage / Cache Impact:
- Created: none.
- Modified: debug runtime config and tests only.
- Removed: no files; legacy root discovery paths were removed from code.

Container / NERSC / MPI Impact:
- No Dockerfile, container wrapper, NERSC, or MPI changes.

Resume / Force-Restart Impact:
- Resume now fails fast on the configured current cache path when templates are missing instead of probing legacy roots.
- No force-restart/cache-clearing command was run.

Residual Risk And Follow-Ups:
- Sequential plot rendering avoids the observed matplotlib/thread contention but a 185-unit plot phase can still be long if every unit must be replotted.
- Existing `artifact_lookup_roots` analyzer/source fallbacks remain unchanged; this change only removes legacy materialized-template output layout fallbacks.
- A real-data plot smoke was intentionally deferred until the active build_templates run is safe to leave alone.

Rollback Notes:
- Revert the resolver/root-candidate changes and debug runtime resource edits to restore legacy probing and parallel unit plotting.

## 2026-05-06 - pending - ai: stream unit source materialization logs

Status: accepted

Summary:
- Changed lazy cached-analyzer materialization process work from one future per unit to one future per unit/source payload while preserving the same max worker count.
- Parent process now logs each unit/source payload materialization result as its future completes, instead of waiting for all sources for a unit to finish.
- Aggregated unit/source results back into the existing per-unit materialization summaries and source payload summary structure.
- Kept process submissions source-major across selected units so the pool fans out across units for each segment/source.

Guardrails Consulted:
- `debug/parallelism_agent_guardrails.md`
- `debug/logging_agent_guardrails.md`

Acceptance Criteria:
- Detailed unit/source payload logs appear during process-mode materialization, before the per-unit completion summaries.
- The phase still caps materialization concurrency at the phase `n_jobs`/unit-worker count.
- Summary payload counts and downstream build_templates unit execution remain unchanged in shape.
- Focused tests cover the source-future process helper and existing lazy materialization behavior.
- A process-mode real-data smoke succeeds and shows source-level logs before unit-level summaries.

Expected To Run:
- Focused Ruff import/syntax checks for touched Python files.
- Focused tests for lazy materialization, source-future logging, and label-filtered parallel materialization.
- Narrow real-data process-mode `reconstruct.build_templates` smoke with one dataset, one well, four limited units, and two segment sources.

Confirmed Not Run:
- Full repository test suite.
- Container image rebuild.
- Multi-well logging smoke.
- Remote push.

Validation:
- Lint: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m ruff check --select I,F src/axon_recon/pipeline/stages/reconstruct/phases/build_templates.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py` passed.
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py::test_run_reconstruct_templates_build_templates_phase_lazy_loads_cached_analyzers_per_unit src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py::test_cached_analyzer_process_materialization_logs_each_unit_source_future src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py::test_run_reconstruct_templates_build_templates_phase_parallel_lazy_materialization_filters_labels -q` passed.
- Diagnostics: VS Code diagnostics reported no errors for touched Python files.
- Real-data smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/axon-reconstructor stages reconstruct.build_templates --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells 1 --limit-units 4 --limit-segments 2 --force-restart` passed with `targets_failed: 0`.
- Smoke log inspected: `/tmp/axon_recon_unit_source_streaming_log_smoke.log` showed `lazy materialization start: requested_units=3 source_count=2 worker_count=3 executor=process`, unit/source materialization logs for units 1, 3, and 4 before the `lazy-materialized cached analyzer payloads` unit summaries, and downstream `build_templates unit execution start: requested_units=3 worker_count=3 executor=process`.

CLI / Debug Flag Impact:
- No CLI flag or runtime YAML changes.

Logging / Parallelism Impact:
- Per unit/source logs now stream from the parent process as each process-pool future completes.
- The configured worker cap is unchanged; only the task granularity within the pool changed.
- Detailed logs remain controlled by `build_templates.emit_unit_source_materialization_log`.

Storage / Cache Impact:
- No new output artifacts.
- Source payload files are still written to the same `cache/source_payloads` paths.

Container / NERSC / MPI Impact:
- No Dockerfile, container wrapper, NERSC, or MPI changes.

Resume / Force-Restart Impact:
- Resume and force-restart semantics are unchanged.

Residual Risk And Follow-Ups:
- Unit/source task granularity may interleave source payload writes differently, but each task writes a unique unit/source payload directory.
- Real smoke still emitted the existing SpikeInterface margin warnings; the phase completed successfully.

Rollback Notes:
- Revert the unit/source job split, source-future process helper aggregation, and focused process-helper test to restore per-unit process futures and delayed source logging.

## 2026-05-05 - pending - ai: add unit source materialization logs

Status: accepted

Summary:
- Added `build_templates.emit_unit_source_materialization_log` to the templates phase config and runtime parser.
- Enabled the option in `debug/debug.runtime.yml` for the active debug runtime.
- Added opt-in INFO logs for each unit/source payload materialization result, including status, lazy-load mode, duration, channel count, and waveform count.
- Emitted those logs from both lazy unit-scoped materialization and non-lazy source-scoped materialization paths.
- Added a one-retry guard for `force_restart` source-payload cleanup when generated cache deletion hits an `ENOTEMPTY` directory race.

Guardrails Consulted:
- `debug/logging_agent_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`

Acceptance Criteria:
- The runtime YAML can turn detailed unit/source materialization logs on or off.
- The debug runtime has the detailed logs enabled for current investigation.
- Log lines identify the unit, segment/source name, materialization status, duration, channel count, and waveform count.
- Focused tests cover config parsing, log emission, cleanup retry, and existing parallel lazy materialization behavior.
- A narrow real-data logging smoke succeeds and shows the new unit/source log lines.

Expected To Run:
- Focused Ruff import/syntax checks for touched Python files.
- Focused tests for templates config parsing and build_templates materialization behavior.
- Narrow real-data `reconstruct.build_templates` smoke with one dataset, one well, two units, and two segments.

Confirmed Not Run:
- Full repository test suite.
- Container image rebuild.
- Multi-well logging smoke.
- Remote push.

Validation:
- Lint: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m ruff check --select I,F src/axon_recon/pipeline/stages/reconstruct/templates/models/inputs.py src/axon_recon/pipeline/stages/reconstruct/templates/config.py src/axon_recon/pipeline/stages/reconstruct/phases/build_templates.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_config.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py` passed.
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_config.py::test_load_templates_config_parses_phased_templates_blocks src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py::test_run_reconstruct_templates_build_templates_phase_lazy_loads_cached_analyzers_per_unit src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py::test_build_templates_force_restart_retries_non_empty_source_payload_cleanup src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py::test_run_reconstruct_templates_build_templates_phase_parallel_lazy_materialization_filters_labels -q` passed.
- Diagnostics: VS Code diagnostics reported no errors for touched Python files.
- Real-data smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/axon-reconstructor stages reconstruct.build_templates --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells 1 --limit-units 2 --limit-segments 2 --force-restart` passed with `targets_failed: 0`.
- Smoke log inspected: `/tmp/axon_recon_unit_source_materialization_log_smoke.log` showed `emit_unit_source_materialization_log=True`, `lazy materialization start: requested_units=1 source_count=2 worker_count=1 executor=serial`, and per unit/source logs for `unit=1 source=000_rec0000` and `unit=1 source=001_rec0001` with channel and waveform counts.

CLI / Debug Flag Impact:
- No CLI flag changes.
- Added a runtime YAML option under `stages.reconstruct.phases.build_templates`.

Logging / Parallelism Impact:
- Adds opt-in detailed materialization logs at normal INFO level.
- Lazy process-worker behavior remains parent-logged after each unit result is collected, keeping logs visible through the existing logging stack.
- No worker-count or slot-allocation behavior changed.

Storage / Cache Impact:
- No new output artifacts.
- `force_restart` cleanup now retries once when deleting generated `cache/source_payloads` hits an `ENOTEMPTY` race.
- Validation regenerated selected source payload and build-template cache artifacts under the configured scratch output root.

Container / NERSC / MPI Impact:
- No Dockerfile, container wrapper, NERSC, or MPI changes.

Resume / Force-Restart Impact:
- Force-restart source-payload cleanup is more robust for generated payload cache directories.
- Resume behavior without force-restart is unchanged.

Residual Risk And Follow-Ups:
- Detailed unit/source logging can be noisy on full-scope runs; leave it enabled only while investigating unit-segment performance.
- The real smoke emitted the existing SpikeInterface margin warning; the phase completed successfully.

Rollback Notes:
- Revert the config field/parser/runtime setting, unit/source materialization log helpers, cleanup retry, and focused tests to return to previous per-unit-only materialization logging.

## 2026-05-05 - pending - ai: parallelize lazy template payload materialization

Status: accepted

Summary:
- Parallelized `build_templates` lazy cached-analyzer source payload materialization across selected units using the phase `n_jobs`/unit-worker budget.
- Kept each materialization worker unit-scoped, loading cached analyzer sources lazily and writing only that unit's payload artifacts.
- Applied the existing unit-label filter before materializing explicit unit selections, so rejected units do not get source payloads.
- Applied the same label-filtered unit scope to downstream reconstruct unit discovery, including GTR generation, plots, reports, and full-chip layout summaries.

Guardrails Consulted:
- `debug/parallelism_agent_guardrails.md`
- `debug/stage_and_phase_behavior_guardrails.md`
- `debug/logging_agent_guardrails.md`

Acceptance Criteria:
- Lazy source payload materialization uses the same unit worker budget as the build_templates phase.
- Only label-allowed units are materialized and passed to downstream reconstruct phases.
- Materialization worker counts and executor choice are visible in logs.
- Existing serial behavior remains available when only one unit/worker is selected or process workers fail.

Expected To Run:
- Focused tests for lazy cached-analyzer materialization, label-filtered materialization, and downstream reconstruct unit selection.
- Import/syntax lint for touched Python files.
- Limited real-data host smoke for `reconstruct.build_templates` with one dataset, one well, and a few units.

Confirmed Not Run:
- Full repository test suite.
- Container image rebuild.
- Multi-well parallelism smoke.
- Remote push.

Validation:
- Focused tests: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m pytest src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py::test_run_reconstruct_templates_build_templates_phase_lazy_loads_cached_analyzers_per_unit src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py::test_run_reconstruct_templates_build_templates_phase_parallel_lazy_materialization_filters_labels src/axon_recon/pipeline/stages/reconstruct/tests/test_runner.py::test_prepare_reconstruct_environment_filters_units_by_templates_unit_labels -q` passed.
- Lint: `/home/adamm/miniconda3/envs/axon_recon/bin/python -m ruff check --select I,F src/axon_recon/pipeline/stages/reconstruct/phases/build_templates.py src/axon_recon/pipeline/stages/reconstruct/phases/report_full_chip_layout.py src/axon_recon/pipeline/stages/reconstruct/runner.py src/axon_recon/pipeline/stages/reconstruct/templates/tests/test_runner.py src/axon_recon/pipeline/stages/reconstruct/tests/test_runner.py` passed.
- Diagnostics: VS Code diagnostics reported no errors for touched Python files.
- Real-data smoke: `/home/adamm/miniconda3/envs/axon_recon/bin/axon-reconstructor stages reconstruct.build_templates --config debug/debug.runtime.yml --limit-datasets 1 --limit-wells 1 --limit-units 4 --phase-tune --force-restart` succeeded.
- Logs inspected: `/tmp/axon_recon_build_templates_parallel_materialization_smoke.log` showed label filtering kept 3/4 units, `templates.build_templates lazy materialization start: requested_units=3 source_count=19 worker_count=3 executor=process`, per-unit lazy materialization completion for units 1, 3, and 4, `build_templates unit execution start: requested_units=3 worker_count=3 executor=process`, and `targets_failed: 0`.

CLI / Debug Flag Impact:
- No CLI flag changes.
- Existing `--limit-units`, unit-label filter config, and phase resource-derived `n_jobs` now affect lazy payload materialization as well as downstream reconstruct work.

Logging / Parallelism Impact:
- Added a lazy materialization execution-plan log with requested units, source count, worker count, and executor.
- Per-unit lazy materialization completion logs remain unit-scoped.
- Downstream reconstruct unit selection logs label-filter counts before phase work begins.

Storage / Cache Impact:
- Source payload caches under `cache/source_payloads` are now only written for selected label-allowed units.
- Validation regenerated selected build-template caches and phase-tune artifacts under the configured scratch output root.

Container / NERSC / MPI Impact:
- No Dockerfile, container wrapper, NERSC, or MPI changes.
- Uses local `ProcessPoolExecutor` with Linux parent-death signal initializer, matching existing process-worker patterns.

Resume / Force-Restart Impact:
- Existing `force_restart` source-payload cleanup remains scoped to the build_templates payload root.
- Unit-scoped resume behavior is unchanged except that label-rejected units are no longer materialized for this phase.

Residual Risk And Follow-Ups:
- Parallel lazy materialization can increase concurrent read pressure on cached analyzer folders; the phase resource class currently controls worker count but does not add a separate disk slot.
- SpikeInterface emitted existing filter margin warnings during smoke; the phase completed successfully.

Rollback Notes:
- Revert the build_templates lazy materialization job pool, downstream reconstruct label-filter application, full-chip unit-list handoff, and focused tests to restore serial lazy materialization and previous downstream unit discovery.

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
