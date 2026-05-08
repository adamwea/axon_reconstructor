# Parallelism Agent Guardrails

Status: guardrail document. Once agentic development begins, treat this file as locked. Do not edit it unless Adam explicitly asks for guardrail changes.

This document defines the expected behavior for well, dataset, segment, unit, resource-slot, and future MPI parallelism in the active `axon_recon` pipeline.

## Operating Contract

- Commit frequently after each coherent accepted slice, using an `ai:` prefix in the commit subject.
- Update `debug/agent_guardrails_commit_notes.md` after every AI commit.
- Run focused tests plus real-data smoke tests when touching parallelism, resource budgeting, keyed resources, worker allocation, logging, progress, subprocess behavior, or stage dispatch.
- Supported task-allocation CLI overrides: `--task-profile <name>` (switches active machine profile), `--task-backend {local_affinity,mpi,none}` (overrides backend), `--cpus-per-task <int>` (overrides slot size). These flags override the YAML for the current run only; they never mutate the YAML file.
- **No per-phase parallelism knobs in runtime YAML.** CPU control lives only in `resources.profiles.<name>.task_allocation`, `resources.phase_budgets`, or CLI flags. The following YAML keys are forbidden and are silently ignored or warned by the validator: `phases.<X>.n_jobs`, `phases.<X>.outputs.segment_save_n_jobs`, `phases.<X>.outputs.concat_save_n_jobs`, `phases.<X>.analyzer.n_jobs`, `phases.<X>.am_kwargs.n_jobs`, `phases.<X>.um_kwargs.n_jobs`.
- Start with 1 dataset, 1 well, 2 segments, and a few units when validating ordinary behavior.
- Use at least 2 wells when validating well-worker concurrency.
- Use up to 2 datasets with 2 wells per dataset when dataset/well interaction, keyed H5 limits, or log interleaving matters.
- Do not run full-scope tests unless Adam explicitly requests them.

## Core Parallelism Model

- Stages run in the selected order.
- Within a stage, independent wells may run concurrently when resources allow.
- Phase budgets (under `resources.phase_budgets`) describe expected CPU, RAM, GPU, disk, analyzer, plot, and H5 demand per phase; they are machine-agnostic and shared across all profiles.
- The active machine profile (`resources.profiles.<name>`) owns three things: `capacity` (hardware resource totals), `task_allocation` (MPI/process binding settings), and `keyed_resource_limits` (per-key concurrency caps such as `source_h5_path`).
- The runtime derives safe stage fanout from phase budget demand and active profile capacity.
- Live phase gates must account for CPU and RAM as occupied resources, not only discrete slots, so wells in different phases share the remaining active profile budget.
- For unit-focused phases, `resource_class.cpu_cores` is the per-well unit-worker budget while `resource_class.ram_gb` is the per-well RAM budget for the phase.
- Total active unit workers across wells must stay within the unoccupied CPU budget through the same resource gate that limits phase entry.
- Discrete resources must be gated during phase execution, not just estimated during planning.
- Unit and segment workers must be resolved explicitly and logged clearly.

Each phase resource class declares a `nested_shape` that describes how inner CPUs are spent within a task slot:
- `si_njobs` — SpikeInterface `n_jobs` fanout (parallel SI compute across recordings/segments).
- `segment_workers` — a `ThreadPoolExecutor` with one worker per segment; each worker holds one segment analyzer.
- `unit_workers` — a `ThreadPoolExecutor` with one worker per unit; workers read from completed segment-level outputs.
- `serial` — single-threaded; always 1 inner worker regardless of slot size.

The per-phase `cpus_per_task` field is OPTIONAL and acts as a cap on the inner thread count. When absent, the phase inherits `task_allocation.cpus_per_task` from the active profile. A phase with `nested_shape: serial` always runs 1 thread, ignoring `cpus_per_task` entirely.

Parallelism must be understandable from logs. Every phase that launches work should make effective worker counts, resource class, and any active keyed resources visible in logs or phase summaries.

### Templates Phases

Template artifacts are produced in two phases. `extract_partial_templates` (`nested_shape: segment_workers`) reads each segment analyzer once and writes per-(unit, segment) partial templates under `cache/source_payloads/<source>/<unit>/`. `build_templates` (`nested_shape: unit_workers`) reads the partials and produces the merged per-unit template under `cache/templates/merged/<unit>/`. `build_templates` MUST NOT reopen segment analyzers; if partial payloads are missing, it must error and direct the operator to run `templates.extract_partial_templates` first.

## Resource And Slot Guardrails

Expected resource concepts:

- CPU cores
- RAM GB
- GPU sort slots
- H5 read slots
- disk-heavy slots
- plot slots
- analyzer slots
- keyed resources such as `source_h5_path`

Rules:

- Do not globally serialize a stage to solve one phase's resource problem.
- Gate only the phase/resource that needs gating.
- CPU and RAM demands are live resource demands: if other wells already occupy CPU/RAM in earlier or later phases, a new well must wait before entering a phase whose declared `cpu_cores` or `ram_gb` would overcommit the active profile.
- Unit-focused reconstruct phases should derive their effective `n_jobs` or unit worker count from the active phase resource class CPU demand unless an explicitly supported phase override narrows it further.
- `spikesort.sort` may be single-well gated when GPU/Kilosort resources require it.
- Other spikesort phases should retain well parallelism when their resources allow it.
- Per-source H5 contention must use keyed resource limits such as `source_h5_path`, not broad dataset-wide guesses.
- Read/write/disk-heavy phases must respect disk and H5 slots.
- Plot/report phases must respect plot slots and memory-heavy resource classes.
- Analyzer phases must respect analyzer slots and inner `n_jobs` settings.
- SpikeInterface `n_jobs` for any analyzer/sorter/save/concat call is computed by `resolve_inner_worker_count(...)` from the active task slot and phase budget. Direct reads of `inputs.n_jobs` from phase code are forbidden.

## Thread And Process Telemetry

Use precise language in logs and summaries:

- `max_threads` is pipeline-declared current-process concurrency.
- `observed_process_max_threads` is raw process/native-library observation.
- Child process counts and child memory must be included when configured.
- Native library thread pools should not be mistaken for pipeline-owned workers.
- Warnings should compare pipeline-owned concurrency against planned resources and report raw observations as diagnostic context.
- Inner worker count is derived from `task_slot.cpu_count` (or the active MPI rank's CPU affinity, treated identically) and optionally clamped by `phase_budgets[<stage.phase>].cpus_per_task`. Phases never read `inputs.n_jobs` directly to decide fanout.

## Required Tests And Acceptance Criteria

### Inner worker derivation tests

- With slot.cpu_count=10 and `phase_cpus_per_task=None`, inner = 10 (inherits).
- With slot.cpu_count=10 and `phase_cpus_per_task=4`, inner = 4 (clamps).
- With no active slot, inner falls back to `yaml_n_jobs_override` or 1.
- `nested_shape: "serial"` always returns 1 regardless of slot or clamp.

## Shared-State Guardrails

- Avoid shared mutable state across well workers.
- Do not mutate global stdout/stderr, logging handlers, environment variables, process working directory, or global library settings from worker threads.
- If a global mutation is unavoidable, scope it to startup or the main thread and document the reason.
- Use locks, keyed gates, or central queue/listener patterns for shared files and logging sinks.
- Only one owner should write global summaries unless writes are explicitly partitioned and merged.

## MPI Preparation Guardrails

Future MPI work should partition independent targets by rank before local fanout.

Rules:

- MPI mode must be optional.
- Non-MPI behavior must remain the default and must not require `mpi4py` at runtime unless the selected mode uses it.
- Rank metadata must appear in logs and summaries when MPI mode is active.
- Rank 0 should own global summaries unless the implementation provides safe rank-scoped summaries plus a merge step.
- Avoid nested `ProcessPoolExecutor`, broad subprocess spawning, and nested container launches inside MPI ranks unless tested on the target HPC system.
- Add fake-MPI adapter tests before requiring real `srun` or `mpirun` validation.

## Resource Tuning Mode

The pipeline should support an advisory resource tuning/calibration mode for estimating realistic `phase_resource_class` values before large runs.

Calibration should measure representative limited executions and recommend CPU/RAM/slot estimates with safety margins. It must not silently rewrite runtime YAML.

Required calibration behavior:

- Can run a whole stage with limits.
- Can run a single phase when upstream artifacts already exist.
- Runs external system-tool samplers only when `--phase-tune` is explicitly requested; normal stage/phase runs must not start those tools implicitly.
- Refuses full-scope calibration unless explicitly confirmed.
- Records stage, phase, resource class, dataset, recording, well, source H5 path, applied limits, wall time, memory, child process usage, raw thread observations, pipeline thread counts, disk I/O, GPU metrics when available, `pidstat`/`iostat` phase-tune metrics when available, status, and exception type when failed.
- Writes machine-readable observations and a human-readable recommendation report.
- Reports recommendations as advisory, not automatic enforcement.
- Includes inner parallelism settings in the report so recommendations are not treated as universal.

## Required Tests And Acceptance Criteria

### Worker allocation unit tests

Acceptance criteria:

- Resource-derived stage fanout matches the active profile and enabled phase resource classes.
- Explicit worker overrides are honored where supported.
- Keyed resources reduce only the matching contended work.
- Pipeline-declared thread counts and raw observed thread counts are reported separately.

### Minimal real-data smoke

Example:

```bash
axon-recon-container stages preprocess \
  --config debug/debug.runtime.yml \
  --limit-datasets 1 \
  --limit-wells 1 \
  --limit-segments 2 \
  --force-restart
```

Acceptance criteria:

- Effective worker counts are logged.
- Phase resource classes and resource-usage summaries are logged.
- No unselected target work starts.
- Phase summaries contain resource usage.

### Well-worker concurrency smoke

Example:

```bash
axon-recon-container stages preprocess \
  --config debug/debug.runtime.yml \
  --limit-datasets 1 \
  --limit-wells 2 \
  --limit-segments 2 \
  --force-restart
```

Acceptance criteria:

- Two wells can make progress concurrently when resource gates allow it.
- Completion logs appear for both wells.
- Resource gates do not globally serialize unrelated phases.
- File logs and JSONL events preserve correct well context.

### Keyed H5 contention smoke

Use 2 wells from the same source H5 when validating H5 contention.

Acceptance criteria:

- At most the configured number of workers enter same-file H5 gated phases concurrently.
- Other non-H5 phases continue when resources allow.
- Logs identify the active `source_h5_path` key.
- There are no HDF5 contention failures.

### Sort-gate smoke

Acceptance criteria:

- `spikesort.sort` respects GPU/sort-slot gating.
- Non-sort phases do not inherit unnecessary global serialization.
- Logs show when a worker waits for the sort slot and when it acquires/releases it.

### Resource tuning smoke

Acceptance criteria:

- A limited stage calibration records observations for every phase executed.
- A direct phase calibration fails clearly if required upstream artifacts are missing.
- Recommendations include observations, safety factors, current class values, proposed values, and caveats.
- Runtime YAML is not mutated by default.
