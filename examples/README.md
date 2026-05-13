# `examples/`

Reference launch wrappers and sbatch templates for running axon-recon.

## Files

| File | Use case |
|---|---|
| `containrun.sh` | Lab-server Docker run (single host) |
| `localrun.sh` | Lab-server bare-metal single-rank smoke |
| `mpirun.sh` | Lab-server multi-rank via host OpenMPI `mpirun` |
| `smoketest_sort_and_recon.sh` | Lab-server end-to-end sort + recon smoke (per-phase invocations) |
| `perlmutter_preprocess.sbatch.example` | NERSC Perlmutter CPU sbatch template (placeholders for account/image) |
| `perlmutter_spikesort.sbatch.example` | NERSC Perlmutter GPU `sort`-only sbatch template (single rank for sort phase) |
| `perlmutter_spikesort.sbatch` | Concrete NERSC sbatch: full spikesort stage, 4 GPU nodes, m2043_g, regular QoS |
| `perlmutter_reconstruct.sbatch` | Concrete NERSC sbatch: reconstruct stage, 4 CPU nodes, m2043, regular QoS |
| `perlmutter_pipeline_chain.sh` | Orchestrator: detects incomplete spikesort, submits spikesort + reconstruct with `--dependency=afterok` |
| `detect_incomplete_spikesort.py` | Thin wrapper over `axon_recon.pipeline.status` that prints incomplete-spikesort dataset indices as CSV (kept for the chain script — for general per-stage status use `axon-recon status` instead) |
| `example.data.yml` | Data config schema reference (placeholder paths only) |

## Where the pipeline is

For a quick per-stage / per-dataset / per-well completeness rollup, run:

```bash
axon-recon status --config dev/debug_NERSC/debug.runtime.yml
```

The default output is one table per stage (preprocess, spikesort, reconstruct, analysis) showing how many wells in each dataset have the stage's well-level "done" marker on disk, plus the list of missing wells per dataset.

For per-phase detail within each stage (which phase's summary json is present for each well), add `-v`:

```bash
axon-recon status --config dev/debug_NERSC/debug.runtime.yml -v
```

Filter scope with `--target-dataset 0 2 8` (or `0,2,8`) and `--stage spikesort reconstruct`.

The four lab-server `*.sh` wrappers (`containrun.sh`, `localrun.sh`, `mpirun.sh`, `smoketest_sort_and_recon.sh`) accept `RUNTIME_CFG=<path>` so the same script works against `dev/debug_local/`, `dev/debug_NERSC/`, or any custom config.

The two `*.sbatch.example` files are generic templates with `<NERSC_ACCOUNT>` / `<registry>` placeholders. The two `*.sbatch` files (no `.example`) are concrete shapes for adamm/m2043 that the chain script submits directly.

## Spikesort → reconstruct chain

`perlmutter_pipeline_chain.sh` is the recommended way to drive a multi-day NERSC run without babysitting interactive allocs:

```bash
cd /global/u2/a/adammwea/dev/pkgs/axon_recon
examples/perlmutter_pipeline_chain.sh
```

What it does:
1. Runs `detect_incomplete_spikesort.py` against `dev/debug_NERSC/debug.runtime.yml`. The detector walks every included `(dataset, well)` pair and checks for `<well>/spikesort_outputs/merge_SLAy/merge_stage_summary.json`. The merge_SLAy phase is the last enabled phase of the spikesort stage; if its summary exists, every upstream phase ran for that well too.
2. If any datasets are missing one or more wells, submits `perlmutter_spikesort.sbatch` targeting just those indices (4 GPU nodes, regular QoS, 4 h wall).
3. Submits `perlmutter_reconstruct.sbatch` targeting `RECONSTRUCT_TARGETS` (default `0,1,2,3,4,5,6,7,8`, i.e. everything except the last 4 datasets), with `--dependency=afterok:<spikesort jobid>` if step 2 fired.

Overrides:
- `DRY_RUN=1` — print the per-dataset completeness table, the sbatch directives for both jobs, the resulting srun commands, and the dataset targets, without actually calling `sbatch`. Run this first to confirm the chain looks right.
- `RECONSTRUCT_TARGETS="0-12"` — pass a different reconstruct dataset list (the script defaults to "all except last 4" because the last 4 are typically already reconstructed in interactive smoke runs).
- `RUNTIME_CFG=dev/debug_local/debug.runtime.yml` — point at a different runtime yml. The data yml is resolved relative to the runtime yml as usual.

Monitor with:
```bash
squeue -u $USER -o '%.10i %.9P %.2t %.10M %.10L %.20R'
```

The dependent reconstruct job sits in `(Dependency)` state until spikesort completes with exit 0. If spikesort fails, the reconstruct is auto-cancelled.

## Slurm × Shifter × profile — how CPUs are counted

NERSC compute nodes expose **logical CPUs** to Slurm by default (2 × physical on SMT-enabled hardware). The pipeline reads its worker count from the **cgroup cpuset** Slurm gave the rank, then filters by `use_hyperthreads` in the active resource profile. Three layers, two of them imposing limits:

```
salloc / sbatch ──┐
                  │ allocation envelope (what the job can claim)
srun -c / --threads-per-core ─→ cgroup cpuset (hard kernel limit per rank)
                                       │ inherited via fork/exec
                              shifter container
                                       │ sched_getaffinity == cpuset
                              detect_cpu_topology()
                                       │ filtered by use_hyperthreads
                          resolve_inner_worker_count → spikeinterface n_jobs
```

**Rule 1**: the cgroup cpuset is the outermost limit. Nothing inside the container can see more CPUs than Slurm put in the cgroup.

**Rule 2**: `use_hyperthreads` filters within the cgroup. It can never expand beyond it.

**Rule 3**: when `--threads-per-core=1` is on srun, SMT siblings aren't in the cgroup at all → `use_hyperthreads` becomes a no-op (true or false yields the same number).

## Truth table — 4-rank spikesort on a Perlmutter GPU node (128 logical / 64 physical)

| srun shape | profile `use_hyperthreads` | cgroup per rank | container counts | workers/rank | node use |
|---|---|---|---|---|---|
| `-c 16` | false | 16 logical (8 phys + 8 SMT) | 8 physical | **8** | 32/64 cores — ❌ half capacity |
| `-c 16` | true | 16 logical | 16 logical | 16 | 32 phys with SMT contention — 🟡 ok-ish |
| `-c 16 --threads-per-core=1` | false | 16 logical = 16 phys | 16 physical | **16** | 64/64 cores, SMT siblings reserved — ✅ |
| `-c 16 --threads-per-core=1` | true | 16 logical = 16 phys | 16 (== phys) | **16** | identical to row above (profile no-ops) — ✅ |
| `-c 32` | false | 32 logical (16 phys + 16 SMT) | 16 physical | **16** | 64 phys workers + SMT slack for BLAS — ✅ |
| `-c 32` | true | 32 logical | 32 logical | 32 | 64 phys with SMT contention — 🟡 |
| `-c 32 --threads-per-core=1` | either | doesn't fit on 64-core node | — | — | srun rejects ❌ |

## Default for axon-recon on Perlmutter

**`-c 16 --threads-per-core=1` with `use_hyperthreads: false`** (rows 3 and 4 — equivalent results, profile setting documents intent).

Why this default:
- **No silent capacity loss.** Row 1 (current pre-flag state) gave 8 workers when 16 were intended. The flag locks the cgroup to physical cores so srun's `-c N` and the profile's "I want N physical cores per task" mean the same thing.
- **No SMT contention within a rank.** Rows 2 and 6 fit 16+ workers on N physical cores; the cache + execution-unit pressure costs more than it gains for axon-recon's I/O-and-memory-heavy passes.
- **SMT siblings reserved on the node, not contended.** Row 3/4 keeps the SMT pairs in the kernel's idle list, available if the OS or library threading inside a worker needs a brief siblings-of-the-cache pop. No noisy-neighbor risk *between* ranks on the same socket.
- **Intent declared at both layers.** Setting `use_hyperthreads: false` in the profile AND `--threads-per-core=1` on srun makes both the configuration file and the launcher tell the same story. Reading either in isolation produces the right mental model. (Row 5 is a legitimate alternative — slightly higher compute throughput from SMT slack — but the profile would then be technically inconsistent with the cgroup view, which makes debugging harder.)

## How to apply

**Inside an alloc, srun:**

```bash
srun --cpu-bind=cores --threads-per-core=1 --module=gpu -N 1 -n 4 -c 16 \
  shifter --image=adammwea/axon-recon:pipeline-v2 \
  axon-recon stages spikesort --config dev/debug_NERSC/debug.runtime.yml
```

`--hint=nomultithread` is the more readable Slurm-canonical alias for `--threads-per-core=1`. Identical effect.

**In sbatch, both `#SBATCH` directives and the srun line need to agree:**

```
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=16

srun --cpu-bind=cores --threads-per-core=1 -c ${SLURM_CPUS_PER_TASK} ...
```

See `perlmutter_preprocess.sbatch.example` and `perlmutter_spikesort.sbatch.example` for the canonical shapes.

## Known gap (tracked separately)

The pipeline does not currently warn when `srun -c N` produces a cgroup with fewer physical cores than the active profile's `cpus_per_task` implies. Row 1 (the pre-flag misconfiguration) is silent — only visible via tqdm's "workers: N" prints. Worth fixing with a startup-time consistency check.
