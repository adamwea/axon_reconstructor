# NERSC-Shaped Local Affinity Plan

Status: planning note. This is not an implementation contract yet. It describes sequential slices for making local pipeline parallelism look like NERSC-style task allocation while keeping non-MPI behavior as the default.

## Goal

Make local execution behave like a scheduler-managed job:

- a selected well target maps to a task
- each task receives a bounded CPU allocation
- nested numerical libraries and phase workers respect that allocation
- logs expose the effective task, CPU, and thread plan
- the same config vocabulary can later map to `srun`, `sbatch`, `mpirun`, or Shifter without rewriting stage logic

The near-term backend should be local CPU affinity, not MPI. MPI and Slurm should be later backends behind the same allocation abstraction.

## Current Lab Server Shape

Observed local server characteristics:

- OpenMPI is installed on the host: Open MPI 4.1.2.
- Visible CPUs: 48 logical CPUs, `0-47`.
- Physical CPU shape: 1 socket, 24 physical cores, 2 hardware threads per core.
- NUMA shape: 1 NUMA node, CPUs `0-47`.
- RAM: about 62 GiB total, about 52 GiB available during the check.
- `/dev/shm`: 32 GiB.

This is a good single-node prototype target because it has enough cores to test real task fanout but does not require NUMA placement decisions yet.

## Target Mental Model

Use scheduler-like language in config and CLI:

```yaml
resources:
  task_allocation:
    enabled: true
    backend: local_affinity
    task_unit: well
    cpus_per_task: 4
    tasks_per_node: auto
    bind: physical_cores
    use_hyperthreads: false
    reserve_cpus: 0
    set_thread_env: true
    nested_thread_policy: match_cpus_per_task
    ram_gb_per_task: null
    shm_gb_per_task: null
```

Local derivation on the lab server with 24 physical cores:

```text
cpus_per_task=4, use_hyperthreads=false -> 6 well tasks
cpus_per_task=6, use_hyperthreads=false -> 4 well tasks
cpus_per_task=8, use_hyperthreads=false -> 3 well tasks
```

Later NERSC mapping:

```bash
srun --ntasks-per-node=<tasks_per_node> --cpus-per-task=<cpus_per_task> --cpu-bind=cores ...
```

## Sequential Change Slices

### 1. Add Config Schema Only

Add a typed config object for `resources.task_allocation`.

Fields:

- `enabled: bool = false`
- `backend: local_affinity | none | mpi | slurm`
- `task_unit: well`
- `cpus_per_task: int | auto`
- `tasks_per_node: int | auto`
- `bind: physical_cores | logical_cpus | none`
- `use_hyperthreads: bool`
- `reserve_cpus: int`
- `set_thread_env: bool`
- `nested_thread_policy: match_cpus_per_task | force_1 | preserve_existing`
- `ram_gb_per_task: float | null`
- `shm_gb_per_task: float | null`

Acceptance criteria:

- Existing runtime YAML remains valid with no behavior change.
- Invalid values fail clearly during runtime config parsing.
- Tests cover defaults, explicit local affinity config, and invalid enum/int values.

### 2. Detect Visible CPU Topology

Add a small topology detector, ideally in a new module such as `src/axon_recon/pipeline/cpu_allocation.py`.

Primary data sources:

- `os.sched_getaffinity(0)` for CPUs visible to the current process or container
- `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`
- `/sys/devices/system/cpu/cpu*/topology/core_id`
- `/sys/devices/system/cpu/cpu*/topology/physical_package_id`

Fallback behavior:

- If `/sys` topology is unavailable, use visible logical CPUs as independent slots and log a warning.
- Never assume host `lscpu` output equals container-visible CPUs.

Acceptance criteria:

- Unit tests mock `/sys` topology for 1 socket, 24 cores, 2 threads per core.
- Unit tests cover cpuset-restricted visibility.
- Unit tests cover missing `/sys` fallback.

### 3. Build A Task Allocation Plan

Define plain dataclasses:

```python
CpuTopology
TaskSlot
TaskAllocationPlan
```

Plan construction should take:

- parsed task allocation config
- detected topology
- target count
- current stage parallelism
- optional resource profile capacity

Planning rules:

- If `enabled=false`, return no plan and keep current behavior.
- If `tasks_per_node=auto`, compute task count from CPU capacity first.
- If RAM or `/dev/shm` per-task limits are configured, reduce tasks by those capacities too.
- If `reserve_cpus > 0`, remove those cores from the allocatable pool before building slots.
- If `use_hyperthreads=false`, use one logical CPU per physical core.
- If `use_hyperthreads=true`, include sibling logical CPUs in each slot.

Acceptance criteria:

- The lab-server topology with `cpus_per_task=4`, `use_hyperthreads=false` yields 6 slots.
- `reserve_cpus=2`, `cpus_per_task=4` yields 5 slots on the same topology.
- Requested `tasks_per_node` greater than capacity clamps or fails according to a clear policy.

### 4. Integrate At Target Distribution Boundary

Do not wire this through every stage or phase manually. Integrate where target workers are distributed.

Preferred hook:

- `_distribute_runtime_targets(...)` or the lower-level `distribute_targets(...)`

Behavior:

- Resolve the allocation plan once per stage invocation.
- Clamp effective `well_workers` to the number of available task slots.
- Assign one task slot to each active target worker.
- Keep phase resource gates intact; task allocation controls CPU placement and worker fanout, not phase admission by itself.

Acceptance criteria:

- Preprocess, spikesort, and reconstruct inherit the behavior without stage-specific plumbing.
- Existing behavior is unchanged when task allocation is disabled.
- Existing resource gates continue to limit CPU/RAM/slot pressure.

### 5. Apply Worker CPU Affinity

Inside each target worker, before phase work starts:

```python
os.sched_setaffinity(0, slot.cpus)
```

If affinity cannot be set:

- log a warning with the requested CPU set and exception
- continue only if config allows soft affinity failure
- fail clearly if strict binding is later added and enabled

Acceptance criteria:

- Unit tests mock affinity application.
- A local smoke log shows each well target assigned a stable CPU set.
- Worker-level affinity never mutates the parent process affinity unexpectedly.

### 6. Set Nested Thread Environment

Before heavy work starts in a target worker, optionally set:

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
NUMEXPR_NUM_THREADS
VECLIB_MAXIMUM_THREADS
NUMBA_NUM_THREADS
```

Policies:

- `match_cpus_per_task`: set each value to the slot CPU count or physical core count, depending on `bind`.
- `force_1`: force native library pools to one thread when outer well fanout is high.
- `preserve_existing`: log existing values and leave them unchanged.

Important detail:

- Some libraries read thread env vars at import time. This may still help subprocesses and late-created pools, but phase-specific explicit `n_jobs` should remain the main pipeline-owned control.

Acceptance criteria:

- Logs record effective env policy and values.
- Existing phase resource-class CPU derivation remains visible.
- Phase-tune keeps distinguishing pipeline-owned thread counts from native observed threads.

### 7. Add Scheduler-Shaped CLI Overrides

Add optional overrides, probably on the stage-sequence parser and direct stage parsers only after config behavior is tested:

```bash
--task-backend local_affinity
--tasks-per-node 6
--cpus-per-task 4
--bind physical_cores
--use-hyperthreads
--reserve-cpus 2
```

These should map onto the same config object as runtime YAML. They should not become separate stage-specific mechanisms.

Acceptance criteria:

- CLI overrides are reflected in the resolved task allocation log.
- CLI overrides do not mutate runtime YAML.
- `axon-recon-container --dry-run ...` forwards them unchanged.

### 8. Add Logs And Structured Metadata

At stage start:

```text
Task allocation backend=local_affinity task_unit=well visible_cpus=0-47 physical_cores=24 cpus_per_task=4 tasks_per_node=6 bind=physical_cores use_hyperthreads=false
```

At target start:

```text
target[0:well000] task_slot=2 cpus=8-11 thread_env=OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
```

Structured summaries should eventually include:

- allocation backend
- task id / slot id
- CPU set
- CPUs per task
- effective thread env policy
- affinity apply status

Acceptance criteria:

- Logs are enough to reconstruct worker placement after a run.
- JSONL or summary metadata can be used by phase-tune reports later.

### 9. Feed Phase-Tune With Allocation Context

Extend phase-tune observations with allocation metadata:

- `task_allocation_backend`
- `task_slot_id`
- `task_cpu_set`
- `cpus_per_task`
- `tasks_per_node`
- `thread_env_policy`

Use this context in recommendations so CPU guidance is not confused with native thread observations.

Acceptance criteria:

- Reports explain whether CPU estimates were measured under local affinity.
- Recommendations can distinguish per-task CPU demand from whole-node fanout.

### 10. Container Readiness Slice

Validate local affinity inside `axon-recon-container` before MPI.

Checks:

- container-visible `os.sched_getaffinity(0)` matches Docker CPU constraints when configured
- `/sys` topology is visible enough for physical-core grouping
- `--cpuset-cpus` or Docker CPU caps interact predictably with task allocation
- `/dev/shm` sizing remains part of task capacity planning

Acceptance criteria:

- Container smoke with 1 dataset, 2 wells shows two assigned CPU sets.
- CPU-only stages run without OpenMPI inside the image.
- No nested Docker or MPI launch is required.

### 11. Add MPI Backend Later

Only after local affinity is stable, add `backend: mpi`.

Rules:

- Importing the package must not require MPI.
- MPI rank/size detection lives behind a small adapter.
- Rank partitioning happens before local target fanout.
- Rank 0 owns global summaries unless rank summaries and merge are tested.
- Add fake-MPI tests before requiring real `mpirun` validation.

Possible local command shape:

```bash
mpirun -np 6 --map-by ppr:6:node:pe=4 --bind-to core axon-recon ...
```

Acceptance criteria:

- Fake MPI tests prove deterministic non-overlapping target partitioning.
- Non-MPI and local-affinity behavior remain unchanged.

### 12. Add Slurm / NERSC Backend Last

Translate the same allocation config into Slurm job shape:

```bash
srun --ntasks-per-node=6 --cpus-per-task=4 --cpu-bind=cores ...
```

Rules:

- Keep Shifter image and mount rules separate from target allocation logic.
- Treat NERSC validation as deferred until real Perlmutter runs complete.
- Stage-split jobs remain preferable for production: CPU preprocess, GPU sort, CPU reconstruct.

Acceptance criteria:

- Slurm scripts derive task counts and CPU binding from the same config vocabulary.
- Logs include rank/task metadata when Slurm or MPI is active.
- NERSC-only assumptions are documented as deferred until tested at NERSC.

## Initial Smoke Plan

Start with local affinity disabled to confirm no behavior change:

```bash
axon-recon-container stages preprocess \
  --config debug/debug.runtime.yml \
  --limit-datasets 1 \
  --limit-wells 1 \
  --limit-segments 2
```

Then enable local affinity and use two wells:

```bash
axon-recon-container stages preprocess \
  --config debug/debug.runtime.yml \
  --limit-datasets 1 \
  --limit-wells 2 \
  --limit-segments 2
```

Expected observations:

- two wells can run concurrently when resource gates allow
- each target logs a task slot and CPU set
- nested thread env is visible
- no unselected dataset or well starts
- phase resource usage still records normal CPU/RAM/disk metrics

## Non-Goals For The First Slice

- Do not require OpenMPI or `mpi4py`.
- Do not change default parallelism behavior.
- Do not remove existing resource gates.
- Do not globally serialize stages.
- Do not claim NERSC readiness from local Docker tests.
- Do not rewrite runtime YAML automatically from phase-tune observations.

## Main Risks

- Native libraries may have already initialized thread pools before worker env vars are set.
- `ThreadPoolExecutor` workers share process-wide affinity and environment; strict per-worker CPU affinity may require process-backed target workers or a carefully scoped worker model.
- Docker-visible CPU topology can differ from host topology.
- Analyzer phases may be constrained by RAM or `/dev/shm` before CPU.
- Small debug runs are not enough evidence to lower resource-class RAM estimates.

## Preferred First Implementation Decision

Start with local affinity as an opt-in backend and keep it process-safe. If the current target distribution uses threads for well workers, first evaluate whether per-target affinity can be applied safely. If not, the first real implementation slice should add process-backed target workers for affinity-enabled mode only, while leaving existing thread-backed behavior as the default.