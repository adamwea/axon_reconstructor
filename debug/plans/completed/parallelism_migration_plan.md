# Parallelism Migration Plan — Toward A Unified, Scheduler-Shaped Model

Status: implementation plan. Hand off to a less-expensive coding agent one slice at a time. Each slice is self-contained: it has a precise file:line edit list, an acceptance check, and a single small commit (`claude:` prefixed when Claude is doing the work; older `ai:`-prefixed commits remain unchanged). Do not start a slice until the previous slice's acceptance check has passed.

This plan supersedes the layered "current vs legacy" parallelism model. Its goal is to make every stage and phase derive worker counts from one source of truth — the active task allocation — and to delete the stage-by-stage legacy knobs.

---

## 0. Goal And End State

### Mental model (target)

```
Outer layer:   one well = one task. The task gets a fixed CPU slot
               (task_slot) bound by local_affinity or owned by an
               MPI rank. The slot has cpus_per_task CPUs.

Inner layer:   each phase declares a `nested_shape` that says how
               to spend those task-local CPUs:
                  - si_njobs        -> SpikeInterface n_jobs
                  - segment_workers -> per-well ThreadPool over segments
                  - unit_workers    -> per-well ThreadPool over units
                  - serial          -> 1 thread, no fanout
               The default inner thread count equals task_slot.cpu_count.
               A phase MAY opt to clamp by setting cpus_per_task in its
               budget (e.g., a memory-heavy phase that should not consume
               the full slot). If unset, the phase inherits the slot.

Profiles:      one profile per machine (lab_server, perlmutter_cpu,
               perlmutter_gpu, ...). A profile owns the task_allocation
               block, capacity description, and keyed_resource_limits.
               Phase budgets are machine-agnostic and live at the top of
               the resources block, shared across profiles.
```

### What is being deleted

- `StageParallelism.max_stage_workers`, `well_workers` (free-form YAML knobs), `divide_stage_workers_by_wells`, `unit_workers` (free-form YAML knob)
- `_unit_workers_after_well_worker_clamp(...)` in `pipeline/runner.py`
- `_warn_legacy_stage_parallelism_keys(...)` and the legacy fallback branch of `resolve_stage_parallelism()` (config.py:405-424) — the new resolver only takes profiles + phase budgets
- `resources.profiles` as a separate sibling of `resources.task_allocation` — `task_allocation` (and `capacity`, `keyed_resource_limits`) move into `resources.profiles[<name>]`. Phase budgets stay at top level (machine-agnostic).
- `resources.phase_resource_classes` is renamed to `resources.phase_budgets` (still top-level, not nested under profiles).
- `resources.keyed_resource_limits` at top level — it becomes `resources.profiles[<name>].keyed_resource_limits`
- The legacy YAML keys `max_simultaneous_well_reads_per_dataset`, `max_simultaneous_well_reads_per_h5_file` (back-compat warnings deleted; replaced by per-phase `keyed_resources` declarations)
- The `inputs.n_jobs`-driven `ThreadPoolExecutor(max_workers=int(inputs.n_jobs))` patterns inside phase core modules (12 sites listed in §3 Discovery)

### What stays

- `cpu_allocation.py` — `TaskSlot`, `TaskAllocationPlan`, `task_slot_affinity_context`, `apply_thread_env_context`, `detect_cpu_topology`. Add helpers (§Slice 5).
- `mpi_adapter.py` — unchanged. MPI ranks remain the outer task layer when backend=mpi.
- `distribute_targets(...)` — unchanged signature. Still consumes `task_slots` and `mpi_context`.
- The `phase_resource_class` *concept* (CPU/RAM/slot demands per phase) — only the location and key-name change.

### Guardrails Maintenance

The contract document `debug/guardrails/parallelism_agent_guardrails.md` is currently locked but explicitly being updated as part of this migration. **Each slice that changes the contract must update the guardrails doc in the same commit.** Per-slice "Guardrails update" bullets list the specific section/wording to revise. The end state is a guardrails doc whose vocabulary matches the new model exactly: nested_shape, phase_budgets, profiles[<name>], task_slot.cpu_count, no well_workers/max_stage_workers/divide_stage_workers_by_wells.

### MPI Vs Local Affinity Parity

Both backends must work end-to-end. The plan exercises three layers of MPI validation:

1. **Unit-level FakeMPI** (Smoke H below) — rank partitioning logic without a real launcher.
2. **Local mpirun smoke** (Smoke I below) — uses the lab server's OpenMPI 4.1.2. Catches local mpi4py / launcher / env_var issues before NERSC.
3. **NERSC-deferred** — Cray MPICH + Shifter + multi-node remain marked deferred until tested at NERSC (per existing `container_mpi4py_NERSC_optimization_guardrails.md`).

Slices that change distribution behavior (5, 6, 7, 8) MUST add an "MPI parity" acceptance bullet that re-runs Smoke I and confirms per-phase `effective=` matches the local_affinity baseline.

---

## 1. End-State Schema (Sketched For Implementation)

This is the YAML the runtime parses after the migration is complete. Every slice below builds toward this exact structure.

```yaml
resources:
  active_profile: lab_server                         # selects which profile is used at runtime

  # ----- Per-machine: profiles describe the host and how tasks are bound to it.
  profiles:
    lab_server:
      capacity:                                      # describes the host (was profiles.<name>)
        cpu_cores: 36
        ram_gb: 50
        gpu_sort_slots: 1
        h5_read_slots: 6
        disk_heavy_slots: 6
        plot_slots: 1
        analyzer_slots: 2

      task_allocation:                               # was resources.task_allocation
        backend: local_affinity                      # local_affinity | mpi | slurm | none
        task_unit: well                              # currently fixed; future: dataset, segment
        cpus_per_task: 10                            # per-task CPU budget — phases inherit this by default
        tasks_per_node: auto
        bind: physical_cores
        use_hyperthreads: false
        reserve_cpus: 0
        set_thread_env: true
        nested_thread_policy: match_cpus_per_task
        ram_gb_per_task: null
        shm_gb_per_task: null

      keyed_resource_limits:                         # was top-level keyed_resource_limits
        source_h5_path:
          description: One well-worker per source H5 file.
          max_concurrent: 1

    perlmutter_cpu:
      capacity: { cpu_cores: 128, ram_gb: 256 }
      task_allocation:
        backend: mpi
        task_unit: well
        cpus_per_task: 16
        bind: physical_cores
        set_thread_env: true
        nested_thread_policy: match_cpus_per_task
      keyed_resource_limits: {}

  # ----- Machine-agnostic: phase budgets describe per-phase shape and demands.
  # Shared across profiles. Numeric demands (ram_gb, slot_demands) describe
  # the phase, not the host; the host's `capacity` declares what's available.
  phase_budgets:
    # Key shape: "<stage>.<phase>". Phases that do not appear default to
    # nested_shape=serial. cpus_per_task is OPTIONAL: when unset, the phase
    # inherits the task slot's CPU count (i.e., profile.task_allocation.cpus_per_task).
    # Set cpus_per_task ONLY to clamp a phase that should use fewer threads.

    preprocess.copy_src_to_scratch:
      nested_shape: serial
      slot_demands: { h5_read_slots: 1 }
      keyed_resources: { source_h5_path: 1 }

    preprocess.save_rec_metadata:
      nested_shape: serial
      slot_demands: { h5_read_slots: 1 }
      keyed_resources: { source_h5_path: 1 }

    preprocess.preprocess_segments:
      nested_shape: si_njobs                         # inherits cpus_per_task from slot
      ram_gb_per_task: 24
      slot_demands: { h5_read_slots: 1, disk_heavy_slots: 1 }
      keyed_resources: { source_h5_path: 1 }

    preprocess.plot_segment_traces:
      nested_shape: segment_workers
      ram_gb_per_task: 4

    preprocess.plot_segment_channel_layouts:
      nested_shape: segment_workers
      ram_gb_per_task: 2

    spikesort.bootstrap_concat_binary:
      nested_shape: si_njobs
      ram_gb_per_task: 24
      slot_demands: { disk_heavy_slots: 1 }
      keyed_resources: { source_h5_path: 1 }

    spikesort.sort:
      nested_shape: si_njobs                         # Kilosort consumes via SI job_kwargs
      ram_gb_per_task: 32
      slot_demands: { gpu_sort_slots: 1, disk_heavy_slots: 1 }

    reconstruct.analyzers:
      nested_shape: si_njobs
      ram_gb_per_task: 25
      slot_demands: { analyzer_slots: 1, disk_heavy_slots: 1 }
      keyed_resources: { source_h5_path: 1 }

    # build_templates is split into two phases — see Slice 2 for the code change.
    reconstruct.extract_partial_templates:
      nested_shape: segment_workers                  # one worker per segment, each owns a segment analyzer
      ram_gb_per_task: 24

    reconstruct.build_templates:
      nested_shape: unit_workers                     # one worker per unit, merges that unit's partial templates
      ram_gb_per_task: 8

    reconstruct.generate_gtrs:
      nested_shape: unit_workers
      ram_gb_per_task: 8

    reconstruct.plot_recons:
      nested_shape: unit_workers
      cpus_per_task: 4                               # explicit clamp: matplotlib RAM scales per worker
      ram_gb_per_task: 8
      slot_demands: { plot_slots: 1 }

    # ... one entry per active phase (full list in §Slice 1.D)
```

Two rules:

1. `task_allocation.cpus_per_task` (profile-level) is the slot size every well-task gets. The slot is allocated once per stage.
2. Per-phase `cpus_per_task` is OPTIONAL and acts as a *cap on inner thread count*. Inner threads = `phase.cpus_per_task` if set, else `task_slot.cpu_count`. A phase budget with `nested_shape: serial` is always 1 thread regardless.

---

## 2. Discovery Targets (Read These First)

A coding agent picking up this plan must ground in these files before editing. Read them. Run grep, do not assume.

```text
src/axon_recon/pipeline/resources.py                        # config dataclasses + parser (parse_resources_config)
src/axon_recon/pipeline/cpu_allocation.py                   # TaskSlot, TaskAllocationPlan, build_task_allocation_plan
src/axon_recon/pipeline/mpi_adapter.py                      # MPIContext, partition_targets_by_mpi_rank
src/axon_recon/pipeline/config.py                           # resolve_stage_parallelism, StageParallelism users
src/axon_recon/pipeline/execution/context.py                # StageParallelism dataclass (line 25)
src/axon_recon/pipeline/execution/distributor.py            # distribute_targets
src/axon_recon/pipeline/runner.py                           # _attach_task_allocation_plan, _distribute_runtime_targets
src/axon_recon/pipeline/stages/preprocess/runner.py         # 3.7K lines; grep before reading
src/axon_recon/pipeline/stages/spikesort/runner.py          # 10.7K lines; grep before reading
src/axon_recon/pipeline/stages/reconstruct/runner.py        # 1.8K lines
debug/debug.runtime.yml                                     # current YAML state
debug/guardrails/parallelism_agent_guardrails.md                       # MUST follow guardrails
debug/plans/active/nersc_shaped_local_affinity_plan.md                   # original design plan
```

### Direct ThreadPoolExecutor inventory (12 sites — replace these in §Slice 5)

| File | Line | Phase | Current source of `max_workers` |
|---|---|---|---|
| stages/preprocess/core/preprocess_segments.py | 323 | preprocess_segments | `n_jobs` parameter (passed from inputs.n_jobs) |
| stages/preprocess/core/plot_segment_traces.py | 360 | plot_segment_traces (channel scoring) | `inputs.plot_n_jobs` |
| stages/preprocess/core/plot_segment_traces.py | 460 | plot_segment_traces (segment plots) | `inputs.plot_n_jobs` |
| stages/reconstruct/core/plot_recons.py | 143 | plot_recons | `int(max(1, int(inputs.n_jobs)))` |
| stages/reconstruct/core/plot_branch_propagations.py | 418 | plot_branch_propagations | `inputs.n_jobs` |
| stages/reconstruct/core/plot_branch_velocities.py | 483 | plot_branch_velocities | `inputs.n_jobs` |
| stages/reconstruct/core/plot_unit_summary.py | 520 | plot_unit_summary | `inputs.n_jobs` |
| stages/reconstruct/core/generate_gtrs.py | 378 | generate_gtrs | `inputs.n_jobs` (line 134) |
| stages/reconstruct/templates/core/build_templates.py | 941 | build_templates | `int(worker_count)` (line 867 from inputs.n_jobs) |
| stages/reconstruct/templates/core/build_templates.py | 959 | build_templates (fallback) | `int(worker_count)` |
| stages/reconstruct/phases/compute_template_similarity.py | 107 | compute_template_similarity | `worker_count` (from inputs.n_jobs) |
| stages/reconstruct/templates/runner.py | 3012 | (legacy templates path) | `worker_count` |

### Legacy plumbing call sites to delete (§Slice 8)

- `pipeline/runner.py:339-344` — `_unit_workers_after_well_worker_clamp` (definition)
- `pipeline/runner.py:389-392` — only call site of the helper above
- `pipeline/runner.py:312-321` — `read_cap`, `keyed_read_cap`, `constrain_stage_parallelism_to_read_groups`
- `pipeline/config.py:78-105` — `_warn_legacy_stage_parallelism_keys`
- `pipeline/config.py:405-424` — legacy `else` branch of `resolve_stage_parallelism`
- `pipeline/config.py:425-447` — `read_cap_raw` / `unit_workers` legacy YAML keys
- `pipeline/resources.py:24-34` — `_LEGACY_RESOURCE_DEFAULT_KEYS`, `_LEGACY_SOURCE_H5_LIMIT_KEYS`, `_WARNED_LEGACY_*` flags
- `pipeline/resources.py:400-433` — `_warn_legacy_*` helpers and `_legacy_source_h5_limit_value`
- `pipeline/execution/context.py:27-33` — `max_stage_workers`, `well_workers`, `divide_stage_workers_by_wells`, `unit_workers_source` fields on `StageParallelism`

### Test files that exercise these paths (run them before AND after each slice)

```bash
conda run -n axon_recon python -m pytest \
  src/axon_recon/pipeline/tests/test_cpu_allocation.py \
  src/axon_recon/pipeline/tests/test_parallel_fanout.py \
  src/axon_recon/pipeline/tests/test_progress.py \
  src/axon_recon/pipeline/tests/test_preprocess_target_status.py \
  src/axon_recon/pipeline/tests/test_spikesort_target_status.py \
  src/axon_recon/pipeline/tests/test_reconstruct_target_status.py \
  -q
```

Save the green count before starting. Each slice must keep that count green.

---

## 3. Smoke Test Matrix (used by every slice's acceptance checks)

CLI flag shape is taken from `debug/mpirun.sh` (verified working):
- `--target-dataset 11,12` — comma-separated dataset IDs (use **two distinct datasets** to validate well concurrency across H5 files)
- `--limit-wells 1` — well count per selected dataset
- `--limit-segments 2` — segment limit per well
- `--limit-units 3` — unit limit per well (for unit-fanout phases)
- `--task-backend mpi` — overrides profile's task_allocation.backend at the CLI

Re-confirm the dataset IDs by running `axon-recon stages preprocess --help` once before the first smoke; substitute as needed.

```bash
# Smoke A: preprocess (local_affinity), 2 datasets × 1 well each — well-worker concurrency on
axon-recon stages preprocess \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --limit-segments 2 --force-restart

# Smoke B: spikesort.bootstrap_concat_binary + spikesort.sort
axon-recon stages spikesort.bootstrap_concat_binary spikesort.sort \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --limit-segments 2

# Smoke C: reconstruct enabled phases (analyzers, extract_partial_templates [post-Slice 2], build_templates, generate_gtrs, plot_recons, plot_unit_summary, report_recons, report_summaries)
axon-recon stages reconstruct \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --limit-segments 2 --limit-units 3

# Smoke D: reconstruct.analyzers in isolation (validates si_njobs nested_shape)
axon-recon stages reconstruct.analyzers \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --limit-segments 2

# Smoke E: reconstruct.extract_partial_templates in isolation (validates segment_workers, post-Slice 2)
axon-recon stages reconstruct.extract_partial_templates \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --limit-segments 2 --limit-units 3

# Smoke F: reconstruct.generate_gtrs in isolation (validates unit_workers nested_shape)
axon-recon stages reconstruct.generate_gtrs \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --limit-segments 2 --limit-units 3

# Smoke G: 2 wells from the SAME dataset (same source H5), keyed gate must serialize them
axon-recon stages preprocess.preprocess_segments \
  --config debug/debug.runtime.yml \
  --target-dataset 11 --limit-wells 2 --limit-segments 2 --force-restart

# Smoke H: FakeMPI rank partitioning unit test (no real launcher; pure Python)
conda run -n axon_recon python -m pytest -q \
  src/axon_recon/pipeline/tests/test_parallel_fanout.py \
  src/axon_recon/pipeline/tests/test_cpu_allocation.py \
  -k "fake_mpi or partition_targets or rank"

# Smoke I: real local mpirun -np 2 (OpenMPI 4.1.2 on lab server)
# Two ranks each get 10 cores via --map-by ppr:2:node:pe=10. Each rank runs one well.
/usr/bin/mpirun -np 2 \
  --map-by ppr:2:node:pe=10 \
  --bind-to core \
  --report-bindings \
  axon-recon stages preprocess \
  --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --limit-segments 2 \
  --task-backend mpi --force-restart
```

Every slice references this matrix by letter (Smoke A, Smoke B, ..., Smoke I).

### MPI parity rule

Slices 5, 6, 7, 8 must run **both** Smoke A (local_affinity) and Smoke I (mpirun) and confirm the per-phase `effective=` log line matches across backends. Differences indicate the helper or distributor doesn't read the rank's affinity correctly.

---

## 4. Migration Slices

Each slice = one PR / one commit. Format:

```
claude: <imperative summary> (slice <N>)
```

Update `debug/commit_log.md` after each commit.

### Slice 1 — Reshape `phase_resource_classes` into per-phase budgets with `nested_shape`

**Goal**: Add `nested_shape` to the per-phase config and make every active phase have an entry. No fanout-behavior change yet — the new field is read but the code still uses `inputs.n_jobs` for inner workers.

**A. Edit `src/axon_recon/pipeline/resources.py`**:
- Add `nested_shape: str = "serial"` and `cpus_per_task: int | None = None` to `PhaseResourceClassConfig` (line 79).
- Validate `nested_shape ∈ {"si_njobs", "segment_workers", "unit_workers", "serial"}` in `_parse_phase_resource_class` (line 301).
- Default `cpus_per_task=None` means inherit from profile-level `task_allocation.cpus_per_task`.
- Keep all legacy fields (`cpu_cores`, `ram_gb`, `keyed_resources`, slot demands) unchanged — they are still read.

**B. Edit `debug/debug.runtime.yml`**:
- For every entry under `resources.phase_resource_classes` (lines 66–151), add `nested_shape: <one of si_njobs|segment_workers|unit_workers|serial>` per the table in §1. Add `cpus_per_task: <int>` only when overriding the existing `cpu_cores` field; otherwise leave absent (inherit).
- Do **not** rename `phase_resource_classes` yet — that happens in Slice 3.

**C. YAML coverage check (no code edits)**: every phase the user runs must have a `resource_class:` line in the YAML `phases` block AND a matching key under `phase_resource_classes`. Verify with:
```bash
grep -n "resource_class:" debug/debug.runtime.yml
```
If any active phase lacks a class assignment, add one to the YAML (not to the runner). Any missing class definition under `phase_resource_classes` should be added there too.

**D. Phase-to-nested_shape mapping (canonical assignment)**:

`cpus_per_task` is *only* listed for phases that should clamp (use fewer than `task_allocation.cpus_per_task`). All other phases inherit the slot. Phases with `nested_shape: serial` always run 1 thread regardless.

| Stage | Phase | nested_shape | cpus_per_task clamp |
|---|---|---|---|
| preprocess | copy_src_to_scratch | serial | — |
| preprocess | save_rec_metadata | serial | — |
| preprocess | prepare_raw_binaries | si_njobs | (inherit) |
| preprocess | preprocess_segments | si_njobs | (inherit) |
| preprocess | concat_segments | si_njobs | (inherit) |
| preprocess | plot_segment_traces | segment_workers | (inherit) |
| preprocess | plot_segment_channel_layouts | segment_workers | (inherit) |
| preprocess | plot_concat_traces | serial | — |
| preprocess | plot_concat_channel_layout | serial | — |
| preprocess | plot_raster_threshold | serial | — |
| preprocess | report_preprocessing | serial | — |
| preprocess | cleanup_preprocessing_outputs | serial | — |
| preprocess | wipe_src_scratch | serial | — |
| spikesort | bootstrap_concat_binary | si_njobs | (inherit) |
| spikesort | sort | si_njobs | (inherit) |
| spikesort | summarize_sort | serial | — |
| spikesort | bombcell_label | si_njobs | (inherit) |
| spikesort | merge_SLAy / merge_si_auto / merge_unitmatch | si_njobs | (inherit) |
| spikesort | cleanup_concat_binary | serial | — |
| reconstruct | resolve_sources | serial | — |
| reconstruct | analyzers | si_njobs | (inherit) |
| reconstruct | extract_partial_templates | segment_workers | (inherit) |
| reconstruct | build_templates | unit_workers | (inherit) |
| reconstruct | compute_template_similarity | unit_workers | (inherit) |
| reconstruct | plot_templates | unit_workers | 4 *(matplotlib RAM)* |
| reconstruct | report_templates / templates_reports | serial | — |
| reconstruct | generate_gtrs | unit_workers | (inherit) |
| reconstruct | plot_recons | unit_workers | 4 *(matplotlib RAM)* |
| reconstruct | plot_branch_propagations | unit_workers | 4 *(matplotlib RAM)* |
| reconstruct | plot_branch_velocities | unit_workers | 4 *(matplotlib RAM)* |
| reconstruct | plot_unit_summary | unit_workers | 4 *(matplotlib RAM)* |
| reconstruct | report_recons | serial | — |
| reconstruct | report_recon_grid | serial | — |
| reconstruct | report_full_chip_layout | serial | — |
| reconstruct | report_summaries | serial | — |
| reconstruct | clear_templates_cache | serial | — |

The `extract_partial_templates` phase does not exist yet — Slice 2 splits it out of `build_templates`. Treat its YAML entry as additive: until Slice 2 lands, leave the YAML key absent.

**E. Add tests**: Extend `pipeline/tests/test_resources_parsing.py` (create if absent) to cover:
- A YAML block with `nested_shape: si_njobs` parses into `PhaseResourceClassConfig.nested_shape == "si_njobs"`.
- An invalid `nested_shape: lolwut` raises `ValueError` with the field name in the message.
- A YAML block without `nested_shape` defaults to `"serial"`.
- A YAML block with `cpus_per_task: 6` parses to `PhaseResourceClassConfig.cpus_per_task == 6`; absent field parses to `None`.

**Guardrails update** (`debug/guardrails/parallelism_agent_guardrails.md`):
- In `## Core Parallelism Model`, add a paragraph introducing `nested_shape` and its four values (`si_njobs`, `segment_workers`, `unit_workers`, `serial`).
- Add a sentence stating that `cpus_per_task` at the per-phase level is an OPTIONAL clamp; the default is to inherit `task_allocation.cpus_per_task` from the active profile.

**Acceptance**:
- `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/tests/test_resources_parsing.py -q` passes including the new tests.
- `conda run -n axon_recon python -c "from axon_recon.pipeline.resources import PhaseResourceClassConfig as P; assert 'nested_shape' in P.__dataclass_fields__"` succeeds.
- Smokes A and C from §3 still complete with the same observable runtime behavior as before this slice (no regressions).
- `git diff --stat` touches only `resources.py`, `debug/debug.runtime.yml`, `debug/guardrails/parallelism_agent_guardrails.md`, and `tests/test_resources_parsing.py`. No code under `pipeline/stages/*` is modified — Slice 1 is config-only.

**Commit**: `claude: add nested_shape to phase resource classes (slice 1)`

---

### Slice 2 — Split `build_templates` into `extract_partial_templates` + `build_templates`

**Goal**: Decompose the existing `build_templates` phase (in the reconstruct stage's templates subtree) into two phases that match how the work actually decomposes — and which directly enables segment_workers and unit_workers to be exercised cleanly in later slices.

The current code path (referenced as `reconstruct.build_templates` in the canonical phase sequence) has two halves inside `src/axon_recon/pipeline/stages/reconstruct/templates/core/build_templates.py`:

1. *Partial template extraction*: per (unit, segment), pull a per-unit slice out of that segment's analyzer and write a partial template artifact. Each work item touches one segment analyzer; segments are independent. Natural fanout: **one worker per segment**. Each segment worker iterates over the unit list internally.
2. *Per-unit template merge*: for each unit, gather its partial templates (one per segment) and merge them into a final per-unit template. Work items are units; segments are read-only. Natural fanout: **one worker per unit**.

**A. Inventory before splitting**:
```bash
grep -n "def \|ThreadPoolExecutor\|partial_template\|merged_template\|merge_templates_for_unit" \
  src/axon_recon/pipeline/stages/reconstruct/templates/core/build_templates.py
grep -rn "build_templates_phase\|run_build_templates_phase\|run_reconstruct_build_templates_phase" \
  src/axon_recon/pipeline/stages/reconstruct/
grep -rn "DEFAULT_INTERNAL_RECONSTRUCTION_PHASE_SEQUENCE\|build_templates" \
  src/axon_recon/pipeline/stages/reconstruct/runner.py
```
Find the natural seam between (1) and (2). The existing `worker_count` ThreadPoolExecutor call at `build_templates.py:941` and `:959` is most likely doing the per-unit-per-segment loop today; verify and confirm where the artifacts land (`cache/templates/partial/<unit_id>/<segment_id>/...` versus `cache/templates/merged/<unit_id>/...`) before splitting.

**B. Code refactor**:
- Move the partial-extraction logic into a new module `src/axon_recon/pipeline/stages/reconstruct/templates/core/extract_partial_templates.py` exposing `run_extract_partial_templates_phase(*, inputs, ...)`. It iterates segments, opens one segment analyzer per worker, and writes per-(unit, segment) partial template artifacts.
- Reduce `build_templates.py`'s `run_build_templates_phase` to the merge half: read the partial template artifacts produced by extract, fan out per-unit, write the merged per-unit template artifacts. Delete any code that re-opens segment analyzers from this module — it must depend purely on the partial outputs.
- Add a new module-level runner shim in the reconstruct stage (alongside the existing `run_reconstruct_templates_*_phase` functions): `run_reconstruct_extract_partial_templates_phase(inputs)` that delegates to the new core function, mirroring the pattern in `reconstruct/runner.py:940`.

**C. Phase plumbing**:
- Insert `"templates_extract_partial_templates"` (or the canonical name the agent picks — match existing prefix conventions) into `DEFAULT_INTERNAL_RECONSTRUCTION_PHASE_SEQUENCE` in `reconstruct/runner.py:69`, **immediately before** `templates_build_templates`.
- Update `_normalize_reconstruct_stage_phase_name`, `_reconstruct_stage_phase_enabled`, `_reconstruct_stage_phase_runner` (search for these around `reconstruct/runner.py` near the existing templates phase resolver) to handle the new name + alias `"templates.extract_partial_templates"`.
- Add a `ReconstructionExtractPartialTemplatesPhaseConfig` dataclass alongside the existing per-phase configs in `reconstruct/models/inputs.py` (search for `ReconstructionPhasesConfig`). Default `enabled: bool = True`.
- Mirror the templates inputs side: add a corresponding `extract_partial_templates_phase` field on the `TemplatesPhasesConfig` if templates phases live there too — match wherever `build_templates_phase` is currently declared.

**D. CLI sub-phase handler**:
- Register `"reconstruct.extract_partial_templates"` in `_STAGE_HANDLERS` in `pipeline/cli.py` (around line 203, alongside existing `reconstruct.build_templates`). Add `_run_reconstruct_extract_partial_templates_from_args` and `run_reconstruct_extract_partial_templates_from_runtime` following the existing pattern for `build_templates`.
- Add aliases as appropriate.

**E. Artifact contract**: Decide and document the on-disk shape:
- partial: `<well_out_dir>/template_outputs/cache/templates/partial/<unit_id>/<segment_index>/{partial_template.npy,partial_channel_locations.npy}` (or whatever the existing extract step already writes — keep it).
- merged: `<well_out_dir>/template_outputs/cache/templates/merged/<unit_id>/{merged_template.npy,merged_channel_locations.npy}` (existing).
- The new `build_templates` phase reads from `partial/`. Write a one-line `partial_summary.json` that `build_templates` can use to discover which units have complete partial sets.

**F. Tests**:
- Migrate any existing tests for `build_templates` that exercise the extract-then-merge sequence into two tests: one for `extract_partial_templates`, one for `build_templates` (merge half). Use a tmp_path with synthetic segment analyzers + 2 units + 2 segments.
- Add `test_reconstruct_extract_partial_templates_runs_before_build_templates`: monkeypatch both phases to record call order; run `run_reconstruct_stage` with both enabled; assert order.
- Add `test_build_templates_consumes_partial_outputs`: after running extract, manually call build with a missing partial file for one unit and assert the unit is skipped or errors clearly (no analyzer fallback).

**G. YAML**:
- In `debug/debug.runtime.yml`, find the existing `phases.build_templates:` block in the `reconstruct` stage and add a sibling `phases.extract_partial_templates:` block above it. Both `enabled: true`. The new block has `resource_class: template_build` (or whatever `build_templates` was using; they share resource demands).
- Add a new entry to `resources.phase_resource_classes`: `extract_partial_templates` with `nested_shape: segment_workers` (Slice 1's table includes it but instructed to skip until this slice — add it now).
- Verify the existing `build_templates` `phase_resource_class` entry has `nested_shape: unit_workers`. If Slice 1 set it to `segment_workers` for the unsplit phase, fix it to `unit_workers` here.

**Guardrails update** (`debug/guardrails/parallelism_agent_guardrails.md`):
- In `## Core Parallelism Model` (or a new "## Templates Phases" subsection), add a note: "Template artifacts are produced in two phases. `extract_partial_templates` (segment_workers) reads each segment analyzer once and writes per-(unit, segment) partial templates. `build_templates` (unit_workers) reads the partials and produces the merged per-unit template. `build_templates` MUST NOT reopen segment analyzers."

**Acceptance**:
- `conda run -n axon_recon python -c "from axon_recon.pipeline.stages.reconstruct.runner import run_reconstruct_extract_partial_templates_phase, run_reconstruct_build_templates_phase"` succeeds.
- `conda run -n axon_recon python -m pytest src/axon_recon/pipeline/stages/reconstruct/tests/ -q` passes (including new tests).
- A direct phase run works: `axon-recon stages reconstruct.extract_partial_templates --config debug/debug.runtime.yml --target-dataset 11 --limit-wells 1 --limit-segments 2 --limit-units 2 --force-restart` produces partial template artifacts.
- A subsequent `axon-recon stages reconstruct.build_templates ...` consumes them and produces merged templates without re-opening segment analyzers (verify by grepping logs for "opening analyzer" or equivalent — should not appear in build_templates).
- The full `reconstruct` stage runs end-to-end with both phases in sequence (Smoke C from §3).

**Commit**: `claude: split build_templates into extract_partial_templates and build_templates (slice 2)`

---

### Slice 3 — Move `task_allocation`, `keyed_resource_limits` INSIDE `profiles[<name>]`; rename `phase_resource_classes` → top-level `phase_budgets`

**Goal**: Reorganize the `resources:` block. Per-machine settings (`task_allocation`, `capacity`, `keyed_resource_limits`) move into `resources.profiles[<name>]`. Phase budgets stay top-level (machine-agnostic) but rename for clarity. Maintain back-compat parsing for one release cycle (warn + remap).

**A. Edit `src/axon_recon/pipeline/resources.py`**:
- Add a new `Profile` dataclass:
  ```python
  @dataclass(frozen=True)
  class Profile:
      capacity: ResourceProfileConfig                          # was profiles.<name>
      task_allocation: TaskAllocationConfig
      keyed_resource_limits: dict[str, KeyedResourceLimitConfig]
  ```
- Change `ResourcesConfig` to:
  ```python
  @dataclass(frozen=True)
  class ResourcesConfig:
      active_profile: str | None
      profiles: dict[str, Profile]
      phase_budgets: dict[str, PhaseResourceClassConfig]   # was phase_resource_classes; STAYS top-level
      defaults: dict[str, Any]
  ```
- Rewrite `parse_resources_config` to:
  1. Read `resources.profiles.<name>.{capacity,task_allocation,keyed_resource_limits}` first; read top-level `resources.phase_budgets`.
  2. If nested profile keys are absent and legacy top-level `task_allocation` / `keyed_resource_limits` are present, build the new shape from them AND log a single deprecation warning naming the keys to migrate. Do not fail.
  3. If `phase_budgets` is absent and legacy `phase_resource_classes` is present, remap with the same one-time warning.
  4. After Slice 11, remove the back-compat paths and require the new shape.
- Update helper functions:
  - `get_active_resource_profile(resources_config) -> Profile | None`
  - `get_phase_budget_config(resources_config, *, stage, phase) -> PhaseResourceClassConfig | None`
  - `get_keyed_resource_limit_config(resources_config, resource_name)` — read from active profile's nested dict.
  - `get_max_phase_resource_demands(...)` — read from top-level `phase_budgets`.

**B. Edit `debug/debug.runtime.yml`**:
- Restructure to match §1 schema: profiles only own capacity / task_allocation / keyed_resource_limits; phase_budgets is top-level.
- Verify:
  ```bash
  conda run -n axon_recon python -c "
  from axon_recon.runtime_config import load_runtime_config
  from axon_recon.pipeline.resources import parse_resources_config
  rc = load_runtime_config('debug/debug.runtime.yml')
  cfg = parse_resources_config(runtime_config=rc)
  print('active=', cfg.active_profile, 'profiles=', list(cfg.profiles), 'budgets=', sorted(cfg.phase_budgets)[:3])
  "
  ```
  Expect: `active= lab_server profiles= ['lab_server'] budgets= [...]`.

**C. Update every callsite of legacy lookups**:
```bash
grep -rn "resources_config\.profiles\b\|resources_config\.keyed_resource_limits\b\|resources_config\.phase_resource_classes\b\|resources_config\.task_allocation\b" src/axon_recon/
```
For each hit:
- `resources_config.task_allocation` → `resources_config.profiles[resources_config.active_profile].task_allocation`
- `resources_config.keyed_resource_limits` → `resources_config.profiles[active].keyed_resource_limits`
- `resources_config.phase_resource_classes` → `resources_config.phase_budgets`
Or route via a helper `get_active_profile(resources_config) -> Profile` (preferred for readability).

**D. Tests**:
- Extend `test_resources_parsing.py` with two fixtures: legacy shape (top-level task_allocation + phase_resource_classes) and new shape. Both must parse to equivalent `ResourcesConfig`.
- Test that the deprecation warning fires once for legacy YAML and not for new-shape YAML.

**Guardrails update** (`debug/guardrails/parallelism_agent_guardrails.md`):
- Replace every reference to `phase_resource_classes` with `phase_budgets`.
- In `## Resource And Slot Guardrails`, replace the description of `resources.profiles.<name>` (capacity-only) with the new shape: each profile owns capacity + task_allocation + keyed_resource_limits; phase budgets live at the top of the resources block (machine-agnostic).

**Acceptance**:
- All existing tests green.
- Smokes A, C complete with **identical effective task allocation** before/after (diff the `task allocation backend=...` log line — byte-identical except timestamps).
- `git grep -nE "resources_config\.(phase_resource_classes|keyed_resource_limits|task_allocation)\b"` returns 0 hits in non-test source files.

**Commit**: `claude: collapse profiles, rename phase budgets (slice 3)`

---

### Slice 4 — Per-machine selectable profiles and CLI override

**Goal**: Add a second profile (`perlmutter_cpu`) to the YAML and validate that `--task-profile <name>` switches between them. Phase budgets are shared across profiles — only the host description, task_allocation, and keyed_resource_limits differ.

**A. Add CLI flag** in `src/axon_recon/pipeline/cli.py` (next to the existing `--task-backend`/`--cpus-per-task` overrides — find them with `grep -n "task-backend\|cpus-per-task" src/axon_recon/pipeline/cli.py`):
- `--task-profile <name>` overrides `resources.active_profile`.
- The override must thread through `_attach_task_allocation_plan(...)` (currently runner.py:347) into `parse_resources_config`. Add an `active_profile_override: str | None` parameter to the resolver pathway.

**B. Add `perlmutter_cpu` profile** to `debug/debug.runtime.yml` under `resources.profiles`:
```yaml
profiles:
  lab_server:
    ...
  perlmutter_cpu:
    capacity:
      cpu_cores: 128
      ram_gb: 256
      gpu_sort_slots: 0
      h5_read_slots: 8
      disk_heavy_slots: 16
      plot_slots: 4
      analyzer_slots: 4
    task_allocation:
      backend: mpi
      task_unit: well
      cpus_per_task: 16
      bind: physical_cores
      use_hyperthreads: false
      reserve_cpus: 0
      set_thread_env: true
      nested_thread_policy: match_cpus_per_task
    keyed_resource_limits:
      source_h5_path:
        max_concurrent: 1
```

`phase_budgets` is unchanged — the same top-level entries serve both profiles. The bigger machine simply gives each task more CPUs (cpus_per_task=16), and each phase that inherits the slot will fan out wider automatically.

**C. Tests**:
- Add `test_active_profile_cli_override` covering `--task-profile perlmutter_cpu` switching.
- Test that an invalid profile name raises `ValueError("resources.active_profile references an undefined profile: 'foo'")`.

**Guardrails update** (`debug/guardrails/parallelism_agent_guardrails.md`):
- Under `## Operating Contract`, add a bullet listing the supported task-allocation CLI overrides: `--task-profile <name>`, `--task-backend {local_affinity,mpi,none}`, `--cpus-per-task <int>`. State that CLI flags override the YAML for the current run only (no YAML mutation).

**Acceptance**:
- Smoke A with `--task-profile lab_server` runs; first task-allocation log line shows `cpus_per_task=10 tasks_per_node=<N>` (the canonical `profile=<name>` field is added in Slice 10; for now just verify the values match `lab_server`'s task_allocation block).
- Smoke A with `--task-profile perlmutter_cpu` *dry-runs only* (the user does not want MPI fired locally without explicit setup): wrap in `--dry-run` if available, or use a unit test that calls `_attach_task_allocation_plan` directly with the override and asserts `plan.backend == "mpi"`.
- `--task-profile nonexistent_machine` fails with the expected error.

**Commit**: `claude: support per-machine selectable profiles via CLI (slice 4)`

---

### Slice 5 — Pipe `task_slot.cpu_count` and `phase_budget` into the inner fanout

**Goal**: Replace the 12 `ThreadPoolExecutor(max_workers=int(inputs.n_jobs))`-style sites with a single helper that derives the inner worker count from the active task slot and (optional) phase budget. After this slice, `inputs.n_jobs` becomes a *derived* value; the YAML key `n_jobs` becomes a *cap*, not a *source*.

**A. Add helper in `src/axon_recon/pipeline/cpu_allocation.py`**:
```python
def resolve_inner_worker_count(
    *,
    nested_shape: str,                 # "si_njobs" | "segment_workers" | "unit_workers" | "serial"
    phase_cpus_per_task: int | None,   # OPTIONAL clamp; None means inherit slot
    yaml_n_jobs_override: int | None,  # legacy YAML cap (e.g., plot_n_jobs: 1)
    work_item_count: int | None,       # number of segments / units / etc; None means unknown
) -> int:
    """Compute the per-well inner worker count from the active task slot.

    Rules:
        - If nested_shape == "serial", return 1 unconditionally.
        - Base = current_task_slot().cpu_count, or 1 if no slot is active.
        - If phase_cpus_per_task is a positive int, base = min(base, phase_cpus_per_task).
        - If yaml_n_jobs_override is a positive int, base = min(base, override).
        - If work_item_count is a positive int, base = min(base, work_item_count).
        - Result must be >= 1.
    """
```
Note: phase RAM is *not* a clamp here; it is enforced by the existing live-resource gate at phase-entry. Inner thread count is purely a CPU question.

Add tests in `tests/test_cpu_allocation.py` covering each clamp.

**B. Rewrite the 12 sites listed in §2 Discovery to use the helper**:
For each site, replace the `inputs.n_jobs`-derived `max_workers` with `resolve_inner_worker_count(...)`. The phase-budget data must be looked up at the call site via a small accessor; add to `cpu_allocation.py`:
```python
def current_phase_budget(stage: str, phase: str) -> PhaseResourceClassConfig | None:
    """Returns the active phase_budget entry from the resources config, or None if not configured."""
```
Implementation: the per-target worker context (set in `runner.py:_distribute_runtime_targets` around line 664) already knows the stage; extend it to stash a reference to the parsed `phase_budgets` dict into a `ContextVar`, similar to `_CURRENT_TASK_SLOT`. Add `phase_budgets_context(...)` and `current_phase_budget(...)` to mirror the existing `task_slot_context` pattern.

**C. Pass the phase name into per-phase entry points**: For each phase function (e.g., `run_preprocess_segments_phase`, `run_reconstruct_generate_gtrs_phase`), the caller (the per-stage runner) already knows the canonical phase name. Wrap the phase invocation in a tiny scope helper that calls `current_phase_budget(stage, phase)` and threads the result down to the call sites. The existing per-stage runner already iterates over named phases — minimal change.

**D. Treat per-phase YAML knobs as transitional caps**:
- The YAML `n_jobs:` / `segment_save_n_jobs:` / `concat_save_n_jobs:` / `analyzer.n_jobs:` keys still in the YAML continue to be respected as *caps* (passed as `yaml_n_jobs_override` to the helper) for this slice ONLY. Slice 7 deletes both the YAML keys and the parsers.
- For phases that have `nested_shape: serial`, the runner must continue to set `inputs.n_jobs=1` for any SI calls those phases happen to make.
- For phases without a `cpus_per_task` clamp in the budget, the inner worker count comes from `task_slot.cpu_count` directly — no override needed.

**E. Tests**:
- Unit: with slot.cpu_count=10 and `phase_cpus_per_task=None`, inner = 10 (inherits).
- Unit: with slot.cpu_count=10 and `phase_cpus_per_task=4`, inner = 4 (clamps).
- Unit: with no active slot, inner falls back to `yaml_n_jobs_override` or 1.
- Unit: `nested_shape: "serial"` always returns 1.
- Integration (in `tests/test_parallel_fanout.py`): run `reconstruct.extract_partial_templates` end-to-end with monkeypatched `ThreadPoolExecutor` to capture the requested `max_workers` and assert it equals `task_slot.cpu_count` (since the phase has no clamp).
- Integration: run `reconstruct.plot_recons` and assert captured `max_workers` equals 4 (the explicit clamp).

**Guardrails update** (`debug/guardrails/parallelism_agent_guardrails.md`):
- Under `## Thread And Process Telemetry`, add: "Inner worker count is derived from `task_slot.cpu_count` (or the active MPI rank's CPU affinity, treated identically) and optionally clamped by `phase_budgets[<stage.phase>].cpus_per_task`. Phases never read `inputs.n_jobs` directly to decide fanout."
- Under `## Required Tests And Acceptance Criteria`, add a new subsection `### Inner worker derivation tests` mirroring the unit tests added in 5.E.

**Acceptance**:
- All existing tests green.
- Smoke E (`reconstruct.extract_partial_templates`, segment_workers) logs:
  ```
  inner workers stage=reconstruct phase=extract_partial_templates nested_shape=segment_workers slot_cpus=10 phase_cap=none effective=10 work_items=<N>
  ```
- Smoke F (`reconstruct.generate_gtrs`, unit_workers, no clamp) shows `effective=10`.
- A direct run of `reconstruct.plot_recons` shows `effective=4` (the explicit clamp).
- Smoke D (`reconstruct.analyzers`) shows SI job_kwargs use `n_jobs=10` (inherited from slot=10, since the phase has no clamp).
- **MPI parity**: Smokes H (FakeMPI test) and I (real `mpirun -np 2`) both pass. Under Smoke I each rank logs its own MPI rank context AND its own `inner workers ... slot_cpus=10` line — the per-rank slot_cpus must equal the rank's `--map-by ... pe=10` allocation. Total wells distributed across the two ranks equals 2.

**Commit**: `claude: derive inner worker count from task slot and phase budget (slice 5)`

---

### Slice 6 — Replace `inputs.n_jobs` with the helper inside SI calls

**Goal**: Get SpikeInterface calls to receive `n_jobs` from the helper rather than `inputs.n_jobs` directly.

**A. Find every SI call site**:
```bash
grep -rn "n_jobs.*inputs\.n_jobs\|job_kwargs.*n_jobs\|\"n_jobs\":\s*int(inputs\.n_jobs" src/axon_recon/pipeline/stages/ | head -100
```

For each phase that uses `nested_shape: si_njobs` (preprocess.preprocess_segments, preprocess.concat_segments, spikesort.bootstrap_concat_binary, spikesort.sort, spikesort.bombcell_label, spikesort merges, reconstruct.analyzers), replace
```python
"n_jobs": int(max(1, int(inputs.n_jobs)))
```
with
```python
budget = current_phase_budget("<stage>", "<phase>")
"n_jobs": resolve_inner_worker_count(
    nested_shape="si_njobs",
    phase_cpus_per_task=getattr(budget, "cpus_per_task", None) if budget else None,
    yaml_n_jobs_override=getattr(inputs, "n_jobs", None),
    work_item_count=None,
)
```

**B. Behavior summary** with the new opt-in clamp model:
- `reconstruct.analyzers` has no `cpus_per_task` clamp in budgets — SI gets `n_jobs = task_slot.cpu_count = 10`.
- `reconstruct.plot_recons` has `cpus_per_task: 4` clamp — its inner ThreadPoolExecutor caps at 4 (and any SI calls it makes also cap at 4, though plot phases typically don't do SI work).
- `preprocess.preprocess_segments` has no clamp — SI gets `n_jobs=10`.

This is correct nersc-shaped behavior: a task does not get more CPUs than `--cpus-per-task`. Phases that need more must run with a larger task allocation. Document this in `parallelism_agent_guardrails.md` (§Operating Contract).

**C. Tests**:
- Integration: monkeypatch `current_task_slot` to return slot with `cpu_count=10`; with `cpus_per_task` unset on the phase budget, assert SI `n_jobs` value is 10.
- Integration: with phase budget `cpus_per_task=4`, slot `cpu_count=10`, assert SI sees `n_jobs=4`.

**Guardrails update** (`debug/guardrails/parallelism_agent_guardrails.md`):
- Under `## Resource And Slot Guardrails`, add: "SpikeInterface `n_jobs` for any analyzer/sorter/save/concat call is computed by `resolve_inner_worker_count(...)` from the active task slot and phase budget. Direct reads of `inputs.n_jobs` from phase code are forbidden."

**Acceptance**:
- Smoke A logs `phase=preprocess_segments n_jobs=10` (inherit; profile cpus_per_task=10).
- Smoke D logs `phase=analyzers n_jobs=10` (inherit).
- Smoke B logs `phase=sort n_jobs=10` (inherit).
- A direct run of `reconstruct.plot_recons` logs `phase=plot_recons n_jobs=4` (clamped by phase budget).
- **MPI parity**: Smoke I shows the same `n_jobs=` values as the equivalent local_affinity smoke (Smoke A/B/D) when the rank-bound CPU count matches the local slot CPU count.

**Commit**: `claude: route si n_jobs through phase budget helper (slice 6)`

---

### Slice 7 — Strip per-phase YAML parallelism knobs

**Goal**: Remove every per-phase parallelism YAML knob. After this slice, the *only* ways to control phase parallelism are:

1. `resources.profiles.<name>.task_allocation.cpus_per_task` (slot size)
2. `resources.phase_budgets.<stage.phase>.cpus_per_task` (per-phase clamp)
3. CLI overrides (`--cpus-per-task`, `--task-profile`, `--task-backend`, etc.)

The user explicitly directed: any per-phase YAML knob that touches resource control must be removed. Phase-level YAML still owns *behavioral* knobs (which sub-step to run, output paths, plot styling, thresholds, etc.), but nothing about CPU/threads/parallelism.

**A. YAML knob inventory** (verified against `debug/debug.runtime.yml` at the time of writing — re-grep before editing):

| Location in YAML | Knob | Currently consumed by |
|---|---|---|
| `phases.preprocess_segments.outputs.segment_save_n_jobs` | `null` | `preprocess/runner.py:998,1019,3081,3565` |
| `phases.prepare_raw_binaries.outputs.segment_save_n_jobs` | `null` | `preprocess/runner.py:3048` |
| `phases.plot_segment_traces.n_jobs` | `1` | (plot_n_jobs flow in `plot_segment_traces.py`) |
| `phases.concat_segments.outputs.concat_save_n_jobs` | `null` | `preprocess/runner.py:997,1023,3138,3564` |
| `phases.bootstrap_concat_binary.n_jobs` | `null` | spikesort runner |
| `phases.bombcell_label.analyzer.n_jobs` | `null` | spikesort runner |
| `phases.merge_SLAy.am_kwargs.n_jobs` (similar for merge_si_auto, merge_unitmatch) | `null` | spikesort runner; `legacy_runner.py:537,538` |
| Any phase block with explicit `n_jobs:` (e.g., reconstruct phases at YAML lines 830, 974) | varies | reconstruct runner |

Re-run this grep at the start of the slice to catch any keys this table missed:
```bash
grep -nE "^\s+(n_jobs|plot_n_jobs|segment_save_n_jobs|concat_save_n_jobs|max_workers|workers):" debug/debug.runtime.yml
grep -rnE "(n_jobs|segment_save_n_jobs|concat_save_n_jobs):\s*(null|[0-9]+)" debug/debug.runtime.yml debug/*.yml
```

**B. YAML edits**: Delete every key from the table above plus anything the grep finds. Comments referencing those knobs (e.g., `# inherit stage-level n_jobs`) also go.

**C. Code edits — preprocess models** (`pipeline/stages/preprocess/models/inputs.py`):
- Remove `concat_save_n_jobs: int | None = None` and `segment_save_n_jobs: int | None = None` fields from the dataclasses at lines 101-102 and 344-345.
- Remove their constructor arguments and any `replace(...)` callers that set them.

**D. Code edits — preprocess runner** (`pipeline/stages/preprocess/runner.py`):
For each line in this list, replace the per-phase override with a call to `resolve_inner_worker_count(...)` keyed by the canonical phase name:
- Line 997: `concat_segments` → `resolve_inner_worker_count(nested_shape="si_njobs", phase_cpus_per_task=current_phase_budget("preprocess","concat_segments").cpus_per_task, yaml_n_jobs_override=None, work_item_count=None)`
- Line 998: `preprocess_segments` → same shape, keyed by `"preprocess.preprocess_segments"`
- Line 1019: `prepare_raw_binaries` → same shape, keyed by `"preprocess.prepare_raw_binaries"`
- Line 1023: `concat_segments` (duplicate path) → same as 997
- Lines 1925, 1931, 3048, 3081, 3138, 3564, 3565: every reference to `concat_save_n_jobs` or `segment_save_n_jobs` from `inputs.phases.*.outputs.*` becomes a helper call OR is deleted if it was just propagating the value.

After this, `grep -rn "segment_save_n_jobs\|concat_save_n_jobs" src/axon_recon/pipeline/` returns 0 hits.

**E. Code edits — spikesort runner** (`pipeline/stages/spikesort/runner.py` — 10K lines, grep first):
```bash
grep -n "phases\.bootstrap_concat_binary\.n_jobs\|phases\.bombcell_label\.analyzer\.n_jobs\|am_kwargs\[.n_jobs.\]\|um_kwargs\[.n_jobs.\]" src/axon_recon/pipeline/stages/spikesort/runner.py
```
Each hit gets replaced by `resolve_inner_worker_count(...)` keyed by the canonical phase name. The `am_kwargs` / `um_kwargs` dicts continue to exist (they hold non-parallelism merge config like dry_run, apply_merges) — only the `n_jobs` key inside them is dropped.

**F. Code edits — reconstruct runner** (`pipeline/stages/reconstruct/runner.py`):
```bash
grep -n "inputs\.phases\.[a-z_]*\.n_jobs\|phase_n_jobs\|plot_n_jobs" src/axon_recon/pipeline/stages/reconstruct/runner.py
```
Each per-phase n_jobs lookup becomes a helper call.

**G. Phase-config dataclasses**: Delete every field on per-phase config dataclasses that holds a parallelism knob:
- `concat_save_n_jobs`, `segment_save_n_jobs` (preprocess)
- `n_jobs` field on any `<Phase>OutputsConfig` or analogous nested config
- Any `plot_n_jobs` field
- The `n_jobs` slot of `am_kwargs` / `um_kwargs` if those are typed dataclasses (they are dicts, so the literal "n_jobs" key gets stripped)
Use `grep -rn "n_jobs:\s*int" src/axon_recon/pipeline/stages/*/models/` to find them.

**H. Tests**: 
- Delete tests that asserted specific `n_jobs` propagation from YAML (they now test obsolete behavior).
- Add `test_per_phase_yaml_n_jobs_no_longer_recognized`: load a YAML containing `phases.preprocess_segments.outputs.segment_save_n_jobs: 8` and assert (a) parsing produces no warning AND no error (the key is silently ignored — extra keys generally are), and (b) the runtime n_jobs is determined by the phase budget, not by the YAML knob.
- Update `test_resources_parsing` if it had any n_jobs assertions.

**I. Documentation**: Update `debug/guardrails/parallelism_agent_guardrails.md` (Operating Contract section) to add: "No per-phase parallelism knobs in runtime YAML. CPU control lives only in `resources.profiles.<name>.task_allocation` and `resources.phase_budgets`."

**Guardrails update** (`debug/guardrails/parallelism_agent_guardrails.md`):
- Under `## Operating Contract`, add a hard rule: "No per-phase parallelism knobs in runtime YAML. CPU control lives only in `resources.profiles.<name>.task_allocation`, `resources.phase_budgets`, or CLI flags. `phases.<X>.n_jobs`, `phases.<X>.outputs.segment_save_n_jobs`, `phases.<X>.outputs.concat_save_n_jobs`, `phases.<X>.analyzer.n_jobs`, `phases.<X>.am_kwargs.n_jobs`, `phases.<X>.um_kwargs.n_jobs` are forbidden — the YAML loader silently ignores or the validator warns."

**Acceptance**:
- `grep -rnE "^\s+(n_jobs|plot_n_jobs|segment_save_n_jobs|concat_save_n_jobs):" debug/*.yml` returns 0 hits.
- `grep -rn "segment_save_n_jobs\|concat_save_n_jobs" src/axon_recon/pipeline/` returns 0 hits in non-legacy code (legacy_runner.py may keep them since it's deprecated).
- Smokes A–F all complete with the same effective parallelism as after Slice 6 (since per-phase knobs were `null`/inherit anyway in the current YAML, removing them changes nothing observable except the `phase parallelism` log no longer mentions YAML overrides).
- A YAML with `phases.preprocess_segments.outputs.segment_save_n_jobs: 8` parses and runs; the value is ignored (or the schema validator warns); the actual SI `n_jobs` matches the phase budget / slot.
- **MPI parity**: Smoke I produces the same per-phase `effective=` and `n_jobs=` values as before this slice.

**Commit**: `claude: strip per-phase yaml parallelism knobs (slice 7)`

---

### Slice 8 — Delete legacy plumbing

**Goal**: Remove every legacy code path enumerated in §2.

**A. Delete from `pipeline/runner.py`**:
- Lines 339-344: `_unit_workers_after_well_worker_clamp`.
- Lines 312-321: `read_cap`, `keyed_read_cap`, `constrain_stage_parallelism_to_read_groups` call. The keyed read cap is now expressed per-phase in `phase_budgets[<phase>].keyed_resources` and is enforced by the existing keyed-resource gate (already wired via `parse_resources_config().keyed_resource_limits`).
- The `unit_workers` and `set_thread_env`/`nested_thread_policy` re-mapping in `_attach_task_allocation_plan` (lines 388-398) — leave only `effective_well_workers = min(parallelism.well_workers, len(plan.slots))` and `task_allocation_plan=plan`.

**B. Delete from `pipeline/config.py`**:
- Lines 78-105: `_warn_legacy_stage_parallelism_keys` and the `_WARNED_LEGACY_STAGE_PARALLELISM_KEYS` set.
- The `else:` branch of `resolve_stage_parallelism` (lines 405-424). Now the only path is the resource-driven path. Add a clear `raise ValueError(...)` if `active_profile is None` or `phase_budgets` is empty.
- Lines 425-447: read-cap / unit_workers YAML key parsing. Replaced by phase budget keyed_resources.

**C. Delete from `pipeline/execution/context.py`**:
Reduce `StageParallelism` to only the fields downstream code actually uses. Final shape:
```python
@dataclass(frozen=True)
class StageParallelism:
    well_workers: int                                  # outer task fanout (matches len(task_slots))
    max_simultaneous_well_reads_per_dataset: int | None  # derived from keyed_resource_limits.source_h5_path
    task_allocation_plan: object | None                # the TaskAllocationPlan
    set_thread_env: bool = False
    nested_thread_policy: str = "preserve_existing"
    use_hyperthreads: bool = False
```
Drop `max_workers`, `max_stage_workers`, `unit_workers`, `unit_workers_source`, `divide_stage_workers_by_wells`.

**D. Delete from `pipeline/resources.py`**:
- Lines 24-34: `_LEGACY_RESOURCE_DEFAULT_KEYS`, `_LEGACY_SOURCE_H5_LIMIT_KEYS`, `_WARNED_LEGACY_*` flags.
- Lines 400-433: `_warn_legacy_*` helpers and `_legacy_source_h5_limit_value`.
- Lines 444-484 in `parse_resources_config`: the `legacy_defaults` and `legacy_source_h5_limit` remap branches.

**E. Delete from YAML and warn-once paths**:
- Remove `resources.defaults` legacy fallback in `get_resource_default` (resources.py:679-696). The top-level `resources.defaults: { chunk_duration: 1s }` block stays where it is — `ResourcesConfig.defaults` is unchanged by Slice 3. What is removed here is the *legacy fallback* that falls through to `resources.<key>` when the key is absent from `defaults`.

**F. Delete legacy log lines** anywhere they reference removed fields:
```bash
grep -rn "max_stage_workers\|divide_stage_workers_by_wells\|unit_workers_source" src/axon_recon/ | grep -v tests
```
Every hit must go.

**G. Update tests** that referenced legacy fields. Tests that exercised the legacy fallback path are deletable.

**Guardrails update** (`debug/guardrails/parallelism_agent_guardrails.md`):
- Sweep the entire doc for `well_workers`, `max_stage_workers`, `divide_stage_workers_by_wells`, `unit_workers` (the YAML knob), and remove or rephrase. Inner worker count is now exclusively `task_slot.cpu_count` (clamped by phase budget); outer worker count is `len(task_slots)` derived from `task_allocation`.
- Under `## Required Tests And Acceptance Criteria` → `### Worker allocation unit tests`, replace acceptance criteria that mentioned legacy knobs with criteria phrased in the new vocabulary.

**Acceptance**:
- All existing non-deleted tests green.
- `grep -rn "max_stage_workers\|divide_stage_workers_by_wells" src/axon_recon/` returns 0 hits.
- `grep -rn "_unit_workers_after_well_worker_clamp" src/axon_recon/` returns 0 hits.
- `grep -rn "_WARNED_LEGACY" src/axon_recon/` returns 0 hits.
- Smokes A through F run unchanged. Log lines no longer contain `max_stage_workers=`.
- **MPI parity**: Smoke I still works end-to-end; rank metadata still appears in logs; per-rank slot_cpus and effective= match Slice 7's values exactly.

**Commit**: `claude: delete legacy stage parallelism plumbing (slice 8)`

---

### Slice 9 — Per-phase keyed-resource gates

**Goal**: Verify that `keyed_resources` declared at the phase budget level (e.g., `source_h5_path: 1` on `preprocess.preprocess_segments`) actually limit concurrency. The gate should already exist in code — this slice tests and tightens it.

**A. Find existing keyed-gate code**:
```bash
grep -rn "keyed_resources\|KeyedResourceLimit\|source_h5_path\|read_group" src/axon_recon/pipeline/execution/
grep -rn "keyed_resources\|KeyedResourceLimit" src/axon_recon/pipeline/runner.py
```
The keyed gate lives in `pipeline/execution/distributor.py` (`_distribute_targets_with_read_group_cap`) and is wired through `max_simultaneous_well_reads_per_dataset` in runner.py.

**B. Verify**: With 2 wells from the *same* source H5, `preprocess.preprocess_segments` should run them serially. Smoke G:
```bash
# Pick 2 wells from the same dataset (same source H5)
conda run -n axon_recon python -m axon_recon stages preprocess.preprocess_segments \
  --config debug/debug.runtime.yml \
  --limit-datasets 1 --limit-wells 2 --limit-segments 2 --force-restart 2>&1 | \
  grep -E "(target task allocation|preprocess_segments|H5 read|keyed)"
```
Expected log signature:
```
target ... well=well000 ... task_slot=0
target ... well=well001 ... waiting on keyed=source_h5_path
... well000 completes ...
target ... well=well001 ... task_slot=0  (or 1)
```

**C. If 2 wells run concurrently for the *same* source H5**, the keyed gate is broken. Investigate `max_simultaneous_well_reads_per_dataset` derivation; ensure it pulls from the active profile's `keyed_resource_limits.source_h5_path.max_concurrent`.

**D. Tests**:
- Add `test_keyed_h5_serializes_same_source` in `tests/test_parallel_fanout.py`: 2 targets with the same `source_h5_path`, `keyed_resource_limits.source_h5_path.max_concurrent=1`, monkeypatched worker fn that records concurrent entries; assert `max_concurrent_observed == 1`.

**Guardrails update** (`debug/guardrails/parallelism_agent_guardrails.md`):
- Refresh `### Keyed H5 contention smoke` (under `## Required Tests And Acceptance Criteria`) to refer to phase-budget keyed_resources declarations rather than the legacy `max_simultaneous_well_reads_per_h5_file` YAML key.

**Acceptance**:
- Smoke G shows the second well waits for the first.
- New test passes.
- Two wells from *different* H5 files in Smoke A still run concurrently (gate is per-key, not global).

**Commit**: `claude: lock keyed resource gates to phase budget declarations (slice 9)`

---

### Slice 10 — Logging cleanup: one canonical line per stage start, one per target start

**Goal**: After all the field renames, the old log lines are no longer self-consistent. Standardize on:

```
Task allocation backend=<bck> profile=<name> task_unit=well visible_cpus=<set> physical_cores=<n> cpus_per_task=<m> tasks_per_node=<k> bind=<b> use_hyperthreads=<h> set_thread_env=<e> policy=<p>

target[<dataset_index>:<stream_id>] task_slot=<s> cpus=<set> stage=<stage> phase=<phase> nested_shape=<shape> effective=<n> thread_env=<vars>
```

Remove or rename every other parallelism-related INFO line that referenced the removed fields.

**A. Files to touch**: search for the old log strings.
```bash
grep -rn "stage_workers=\|max_stage_workers=\|well_workers=" src/axon_recon/pipeline/
```

**B. Add a new canonical phase-level log** at the top of each phase entry (in the per-stage runner), using the helper from Slice 5:
```python
LOGGER.info(
    "phase parallelism stage=%s phase=%s nested_shape=%s slot_cpus=%d phase_cap=%s effective=%d",
    stage_name, phase_name, nested_shape, slot_cpus, str(phase_cap), effective_workers,
    extra={
        "event": "phase_parallelism",
        "stage": stage_name,
        "phase": phase_name,
        "nested_shape": nested_shape,
        "slot_cpus": int(slot_cpus),
        "phase_cap": int(phase_cap) if phase_cap is not None else None,
        "effective_workers": int(effective_workers),
    },
)
```

**Guardrails update** (`debug/guardrails/parallelism_agent_guardrails.md`):
- Update `## Thread And Process Telemetry` log-line examples to match the canonical format from this slice (`Task allocation backend=... profile=... cpus_per_task=...`, `target[<n>:<id>] task_slot=... cpus=... phase=... effective=...`, `phase parallelism stage=... phase=... effective=...`).
- Under `## MPI Preparation Guardrails`, add: "When MPI is active, every log line above includes `mpi_rank` and `mpi_size` fields."

**Acceptance**:
- Smokes A through F show exactly one `Task allocation backend=...` line per stage and exactly one `target[...]` line per well.
- Each phase shows exactly one `phase parallelism stage=... phase=... effective=...` line.
- No log line contains `max_stage_workers=`.
- **MPI parity**: Smoke I logs include `mpi_rank=0|1 mpi_size=2` on every parallelism line.

**Commit**: `claude: standardize parallelism log lines on the new vocabulary (slice 10)`

---

### Slice 11 — Remove back-compat YAML parser and lock the new schema

**Goal**: Delete the legacy YAML migration path added in Slice 3. After this slice, runtime YAML must use the new schema.

**A. Edit `pipeline/resources.py`**:
- In `parse_resources_config`, remove the back-compat branch that reads top-level `task_allocation`, `keyed_resource_limits`, `phase_resource_classes`, `profiles`. Now only `profiles[<name>].{capacity,task_allocation,...}` is recognized.
- Delete the corresponding deprecation warning helper.

**B. Edit `debug/debug.runtime.yml`** to be the canonical example. Add a header comment:
```yaml
# Resources are organized by profile. Each profile owns its own
# task_allocation, capacity, keyed_resource_limits, and phase_budgets.
# Top-level resources.task_allocation/profiles/keyed_resource_limits/
# phase_resource_classes are no longer recognized.
```

**C. Migration test**: Add `test_legacy_yaml_shape_now_rejected` that loads a YAML with the old shape and asserts a clear error message.

**Guardrails update** (`debug/guardrails/parallelism_agent_guardrails.md`):
- Add a final paragraph at the top of the doc (or under `## Operating Contract`): "The legacy `resources.phase_resource_classes`, top-level `resources.task_allocation`, and top-level `resources.keyed_resource_limits` keys are no longer recognized. Runtime YAML must use the per-profile shape (capacity + task_allocation + keyed_resource_limits inside `resources.profiles.<name>`) and top-level `resources.phase_budgets`."
- Final consistency pass: re-read the entire guardrails doc end-to-end and ensure no stale terminology remains.

**Acceptance**:
- All smokes (A–I) green.
- A YAML with the old shape fails with a parse-time error mentioning the new key path.
- `grep -rn "phase_resource_classes" debug/` returns no hits.
- `grep -nE "well_workers|max_stage_workers|divide_stage_workers_by_wells" debug/guardrails/parallelism_agent_guardrails.md` returns no hits.

**Commit**: `claude: lock new resources schema, remove back-compat parser (slice 11)`

---

## 5. Validation Matrix (run AFTER each slice)

| Slice | Must pass | Smokes |
|---|---|---|
| 1 | test_resources_parsing | (config-only; no behavior smoke required) |
| 2 | test_extract_partial_templates + test_build_templates_consumes_partial_outputs | Smokes C, E (extract_partial_templates appears in order log before build_templates) |
| 3 | test_resources_parsing (legacy + new shape) | Smoke A (identical task allocation block before/after) |
| 4 | test_active_profile_cli_override | Smoke A with `--task-profile lab_server` |
| 5 | test_cpu_allocation + test_parallel_fanout | Smokes A, D, E, F (local) **+ Smoke H + Smoke I (MPI parity)** |
| 6 | new SI integration tests | Smokes A, B, D + plot_recons direct + Smoke I |
| 7 | test_per_phase_yaml_n_jobs_no_longer_recognized | Smokes A–F + Smoke I |
| 8 | full test suite (no regressions) | Smokes A–F + Smoke I |
| 9 | test_keyed_h5_serializes_same_source | Smoke G |
| 10 | full test suite | Smokes A–F + Smoke I (MPI logs include mpi_rank/mpi_size) |
| 11 | test_legacy_yaml_shape_now_rejected + full test suite | Smokes A–I all green |

**Smoke I requirement**: any slice that touches `cpu_allocation.py`, `mpi_adapter.py`, `runner.py:_distribute_runtime_targets`, `distributor.py`, or any phase's worker derivation MUST run Smoke I. Slices 5, 6, 7, 8, 10 are mandatory; others recommended.

---

## 6. Cleanup Checklist (post-Slice 11)

After all slices land, verify:

```bash
# (a) No legacy field references
git grep -nE "max_stage_workers|divide_stage_workers_by_wells|_unit_workers_after_well_worker_clamp|_WARNED_LEGACY|legacy_max_workers" src/

# (b) No legacy YAML keys in any test fixtures
git grep -nE "phase_resource_classes:" -- '*.yml' '*.yaml'

# (c) StageParallelism has only the fields kept in Slice 8.D
git grep -n "class StageParallelism" -A 20 src/

# (d) phase_budgets is top-level, every active phase has an entry
conda run -n axon_recon python -c "
from axon_recon.pipeline.config import load_pipeline_runtime_bundle
from axon_recon.pipeline.resources import parse_resources_config
b = load_pipeline_runtime_bundle(config_path='debug/debug.runtime.yml')
cfg = parse_resources_config(runtime_config=b.runtime_config)
for stage in ('preprocess','spikesort','reconstruct'):
    print(stage)
    for k, v in sorted(cfg.phase_budgets.items()):
        if k.startswith(stage+'.'):
            clamp = v.cpus_per_task if v.cpus_per_task is not None else 'inherit'
            print(f'  {k:50s} shape={v.nested_shape:18s} cpus_per_task={clamp}')
"
```
Each command should produce the expected (empty / matching) result. Specifically (d) should list `reconstruct.extract_partial_templates` and `reconstruct.build_templates` as separate entries with shapes `segment_workers` and `unit_workers` respectively.

```bash
# (e) Guardrails doc has no stale vocabulary
grep -nE "well_workers|max_stage_workers|divide_stage_workers_by_wells|phase_resource_classes" \
  debug/guardrails/parallelism_agent_guardrails.md

# (f) Both backends still work end-to-end
# Local affinity:
axon-recon stages preprocess --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --limit-segments 2 --force-restart
# MPI:
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core --report-bindings \
  axon-recon stages preprocess --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --limit-segments 2 --task-backend mpi --force-restart
```
(e) must be empty. (f) must produce the same per-phase `effective=` values across both runs (modulo timestamps).

---

## 7. Risks And Out-Of-Scope

### Risks

- **Native libraries already imported**: NumPy/MKL/OpenBLAS read `OMP_NUM_THREADS` etc. at *import time*. The per-target apply_thread_env_context already addresses this for child processes and late-created pools, but a stage that imports torch/numpy at the top of the runner cannot retroactively shrink an existing thread pool. Document this in `parallelism_agent_guardrails.md` as a known caveat.
- **ThreadPoolExecutor shares process affinity**: Per-target affinity (`os.sched_setaffinity(0, slot.cpus)`) inside a thread mutates the *whole process* affinity. The current code is single-process so this works; if a future slice introduces ProcessPoolExecutor for outer wells, revisit.
- **Phase budget `cpus_per_task` is a *clamp*, not a *target***: setting cpus_per_task=20 on a phase whose slot has only 10 still gives 10, not 20. Phases that need more than the profile's `cpus_per_task` cannot get more without bumping the profile.
- **MPI rank topology vs local affinity**: When `backend=mpi`, each rank's `detect_cpu_topology()` reflects the rank's slurm-bound affinity. The per-task cpus_per_task must match `--cpus-per-task` from `srun`. Validate at NERSC during real-data smoke.

### Out of scope (do not include in these slices)

- Changing the public `axon-recon-container` wrapper interface.
- Adding new backends beyond `local_affinity`, `mpi`, `slurm`, `none`.
- Phase-tune integration with the new schema (already partially done in slice 9 of `nersc_shaped_local_affinity_plan.md`).
- Removing `inputs.n_jobs` from inputs dataclasses entirely. Leave it as a derived value.
- Multi-node MPI rank partitioning logic — rank partitioning of targets already lives in `mpi_adapter.partition_targets_by_mpi_rank` and is unchanged by this migration.

---

## 8. Definition Of Done

The migration is complete when:

1. The 11 slices above are merged in order, each with passing acceptance checks.
2. `debug/debug.runtime.yml` is restructured into the new schema: per-machine `profiles.<name>.{capacity,task_allocation,keyed_resource_limits}` and top-level `phase_budgets`, with both `lab_server` and `perlmutter_cpu` profiles defined.
3. `reconstruct.build_templates` is split: `extract_partial_templates` (segment_workers) runs first and produces partial templates; `build_templates` (unit_workers) consumes them and produces merged templates.
4. **No per-phase parallelism knobs in YAML.** `grep -rnE "(n_jobs|plot_n_jobs|segment_save_n_jobs|concat_save_n_jobs|max_workers|workers):" debug/*.yml` is empty. CPU control lives only in `resources.profiles.<name>.task_allocation`, `resources.phase_budgets`, and CLI flags.
5. Every phase the user runs (preprocess preprocess_segments + plot subset; spikesort bootstrap_concat_binary + sort; reconstruct analyzers + extract_partial_templates + build_templates + generate_gtrs + plot_recons + plot_unit_summary + report_recons + report_summaries) logs a single `phase parallelism stage=... phase=... nested_shape=... effective=...` line whose `effective=` matches the table in §Slice 1.D.
6. The full test suite, including new tests added by each slice, passes.
7. `git grep -nE "max_stage_workers|_unit_workers_after_well_worker_clamp|_WARNED_LEGACY"` is empty across `src/`.
8. The canonical task-allocation log line uses `profile=<name>` and references the active profile, not bare task_allocation config.
9. `cpus_per_task` appears under at most ~5 phase entries (the explicit clamps for memory-heavy plot phases). Every other phase inherits the slot — confirmed by the cleanup-check (d) command.
10. `debug/guardrails/parallelism_agent_guardrails.md` is fully consistent with the new model. No legacy vocabulary remains (cleanup-check (e) is empty). The doc's vocabulary, examples, and acceptance criteria all match the new schema.
11. **Both backends validated end-to-end.** Smoke A (local_affinity) and Smoke I (real `mpirun -np 2`) both run preprocess to completion with matching per-phase `effective=` values. Smoke H (FakeMPI partition tests) passes in the unit test suite.

If any acceptance check fails, the slice does not land. Each slice is a single commit; do not bundle slices.
