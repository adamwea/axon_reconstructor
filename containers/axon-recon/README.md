# axon_recon Container

This image runs the same `axon-recon` CLI as the host environment, with all pipeline stage selectors passed through unchanged. It is a full-pipeline execution environment, not a special-purpose sort image.

## Run modes

The pipeline supports six invocation shapes spanning local host vs. container, single-rank vs. multi-rank, and lab server vs. NERSC Perlmutter. The CLI tail (`stages …`, `--config`, `--target-dataset`, `--limit-wells`, `--task-backend …`, etc.) is identical in every mode; only the launcher prefix changes.

### Mode 1 — Local host, single-process

One-liner: run `axon-recon` directly on the lab server using the host conda env. No container, no MPI.

```bash
axon-recon stages preprocess --config debug/debug.runtime.yml
```

- **Stages**: any (`preprocess`, `spikesort`, `reconstruct`, `analysis`, the `stages …` aggregate, sub-phases like `spikesort.sort`).
- **CPU/GPU**: CPU stages run unconstrained on host CPUs; GPU stages use whatever CUDA visibility the host conda env sees.
- **When to use**: development on the lab server, quick smokes, debugging where iteration speed matters more than reproducibility.
- **See also**: top-level `README.md` in the repo root.

### Mode 2 — Local host + `mpirun` (multi-rank, no container)

One-liner: drive `axon-recon` from the host's OpenMPI `mpirun -np N`, with explicit `--task-backend mpi` so the rank partition fires.

```bash
/usr/bin/mpirun -np 2 \
  --map-by ppr:2:node:pe=10 \
  --bind-to core \
  --report-bindings \
  axon-recon stages preprocess \
    --config debug/debug.runtime.yml \
    --target-dataset 11,12 --limit-wells 1 \
    --task-backend mpi --force-restart
```

- **Stages**: CPU stages only (`preprocess`, `reconstruct`, `analysis`). GPU sort would need `--mpi-ranks 1` semantics (see Mode 1 for sort).
- **CPU/GPU**: host OpenMPI binds ranks to CPU subsets; per-rank thread env follows `resources.task_allocation.set_thread_env`.
- **When to use**: validated end-to-end for preprocess after commit `82ed42c`; preferred when you want to avoid the container's caching/build overhead for fast iteration on CPU stages.
- **See also**: `debug/mpirun.sh` (working example), `debug/plans/active/nersc_shaped_local_affinity_plan.md` slices 1–8 (the local-affinity machinery the per-rank thread env rides on).

### Mode 3 — Local container, single rank

One-liner: forward the same CLI tail through `axon-recon-container`, which manages image build/update, mounts, cache paths, and Docker execution.

```bash
axon-recon-container stages preprocess --config debug/debug.runtime.yml --limit-segments 2 --limit-datasets 2 --limit-wells-per-dataset 1
axon-recon-container --gpus all stages spikesort --config debug/debug.runtime.yml --limit-segments 2 --limit-datasets 2 --limit-wells-per-dataset 1
axon-recon-container stages reconstruct --config debug/debug.runtime.yml --limit-segments 2 --limit-datasets 2 --limit-wells-per-dataset 1 --limit-units 5
```

- **Stages**: any. Sort uses `engine: local_spikeinterface` automatically (`mea_analysis` is blocked inside the container by default).
- **CPU/GPU**: pass `--gpus all` (or set `AXON_RECON_CONTAINER_GPUS=all`) for CUDA-backed sort. CPU-only stages don't need the flag.
- **When to use**: today's default for one-shot stage runs that need the pinned container env (Kilosort4, SpikeInterface 0.104.x, UnitMatchPy, SLAy). Default behaviour on the lab server.
- **See also**: `debug/guardrails/container_mpi4py_NERSC_optimization_guardrails.md` (Container Contract section).

### Mode 4 — Local container + multi-rank inside one container (`--mpi-ranks`)

One-liner: launch ONE container, run `mpirun -np N --allow-run-as-root --bind-to none axon-recon stages …` inside it. The wrapper owns the rank count via `--mpi-ranks N` (or `-n N` short alias). Default (no flag) is byte-for-byte identical to Mode 3.

```bash
axon-recon-container --mpi-ranks 2 stages preprocess \
    --config debug/debug.runtime.yml \
    --target-dataset 11,12 --limit-wells 1 --limit-segments 2 \
    --task-backend mpi --force-restart

axon-recon-container --dry-run --mpi-ranks 2 stages preprocess --config debug/debug.runtime.yml
```

- **Stages**: CPU stages with `N > 1`. For GPU `spikesort.sort` use `--mpi-ranks 1` — the runner raises `SpikesortGpuOversubscriptionError` before any CUDA allocation when `ranks > NVML physical GPU count`, see `container_shifter_shape_plan.md` §6.
- **CPU/GPU**: per-rank `CUDA_VISIBLE_DEVICES` partitioning happens inside the image at `axon_recon.pipeline.mpi_adapter.apply_per_rank_cuda_visible_devices()`, called from the top of `cli.py` *before* any torch/cupy/kilosort import.
- **Important**: host `mpirun -np N axon-recon-container …` is **unsupported** — it spawns N separate containers, each with its own MPI_COMM_WORLD of size 1, and "runs double" (every rank processes every target, triggering `FileExistsError` at `kilosort4.py:127`). The wrapper owns the rank count.
- **When to use**: CPU stages where you want multi-rank parallelism AND the container's pinned env. The local emulation of the NERSC `srun -n N shifter` shape.
- **See also**: `debug/plans/active/container_shifter_shape_plan.md` (the Option-B plan), `debug/guardrails/container_mpi_strategy_note.md` (Option-A/B/C analysis).

### Mode 5 — NERSC interactive (Shifter)

One-liner: allocate an interactive Perlmutter node with the image attached, then drive `axon-recon` through `srun shifter`. Same CLI tail as Mode 4, with `srun` substituting for `mpirun -np`.

```bash
salloc --nodes=1 --time=01:00:00 --constraint=cpu --image=<registry>/<image>:<tag>
# Once allocated:
srun -n 6 --cpu-bind=cores shifter axon-recon stages preprocess \
  --config /global/cfs/<path>/debug.runtime.yml \
  --task-backend slurm \
  --tasks-per-node 6 --cpus-per-task 4 --bind physical_cores
```

- **Stages**: CPU stages multi-rank (`--cpus-per-task ≥ 4`). For GPU sort, request `--constraint=gpu --module=gpu,cuda-mpich` and use `-n 1` per the GPU contention rule from Mode 4.
- **CPU/GPU**: Shifter swaps Cray MPICH at runtime via `--module=gpu,cuda-mpich`; the local OpenMPI in the image is replaced transparently. The pipeline's `mpi_adapter` detects `SLURM_PROCID`/`SLURM_NTASKS` and partitions targets accordingly.
- **Status**: **NERSC validation deferred** per `debug/guardrails/container_mpi4py_NERSC_optimization_guardrails.md`. No part of this repo has been smoked on Perlmutter yet.
- **See also**: `debug/guardrails/container_mpi4py_NERSC_optimization_guardrails.md` (Shifter And Perlmutter Rules).

### Mode 6 — NERSC sbatch (Shifter, multi-rank)

One-liner: full `#SBATCH` job script that pulls the image, mounts CFS/scratch paths, and runs `srun shifter axon-recon stages … --task-backend slurm`. Stage-split is the recommended production shape (one job per CPU/GPU class).

```bash
#!/bin/bash
#SBATCH --job-name=axon-preprocess
#SBATCH --image=<registry>/<image>:<tag>
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=4
#SBATCH --volume="/global/cfs/<path/to/data>:/data:ro"
#SBATCH --volume="/global/cfs/<path/to/scratch>:/scratch:rw"
srun --cpu-bind=cores shifter axon-recon stages preprocess \
  --config /data/debug.runtime.yml \
  --task-backend slurm \
  --tasks-per-node ${SLURM_NTASKS_PER_NODE} \
  --cpus-per-task ${SLURM_CPUS_PER_TASK} \
  --bind physical_cores
```

For GPU sort, swap `--constraint=cpu` for `--constraint=gpu --module=gpu,cuda-mpich`, set `--ntasks-per-node=1`, and export `MPICH_GPU_SUPPORT_ENABLED=1` before `srun`.

- **Stages**: stage-split production runs. CPU preprocess (multi-rank) → GPU sort (single-rank) → CPU reconstruct (multi-rank) → analysis.
- **CPU/GPU**: same as Mode 5 modulo the launcher; the sort job's GPU contention rule still requires `--ntasks-per-node ≤ physical GPU count`.
- **Status**: **NERSC validation deferred**. The .example scripts ship documentation only.
- **See also**: `debug/perlmutter_preprocess.sbatch.example`, `debug/perlmutter_spikesort.sbatch.example`, `debug/plans/active/nersc_shaped_local_affinity_plan.md` slice 12.

### Quick reference

| Machine context | Scale | Mode | Launcher prefix |
|---|---|---|---|
| Lab server, host conda env | single process | 1 | `axon-recon …` |
| Lab server, host conda env | multi-rank | 2 | `/usr/bin/mpirun -np N axon-recon … --task-backend mpi` |
| Lab server, container | single rank | 3 | `axon-recon-container …` |
| Lab server, container | multi-rank | 4 | `axon-recon-container --mpi-ranks N … --task-backend mpi` |
| NERSC Perlmutter, Shifter | interactive | 5 | `srun -n N shifter axon-recon … --task-backend slurm` (after `salloc --image=…`) |
| NERSC Perlmutter, Shifter | sbatch (production) | 6 | `srun shifter axon-recon … --task-backend slurm` (inside an `#SBATCH` script) |

Modes 1–4 are validated on the lab server. Modes 5–6 are documentation-only until smoked on Perlmutter.

## Local Build

Basic image build from the repo root:

```bash
docker build -f containers/axon-recon/Dockerfile -t axon-recon:local .
```

Full dependency builds should use the helper so sibling checkouts are copied into a temporary context and installed as packages:

```bash
containers/axon-recon/build_local_image.sh --image axon-recon:local
```

The default repo-root build installs `axon_recon`, the active runtime Python dependency set, `spikeinterface==0.104.3`, `mpi4py`, and the system OpenMPI 4.x toolchain (`openmpi-bin`, `libopenmpi-dev`). It does not bake local data, scratch outputs, credentials, or sibling workspace paths into the image. The helper detects sibling `../UnitMatch/UnitMatchPy` and `../SLAy` checkouts when present, copies them under `external/` in a temporary context, and passes build args so imports are normal installed-package imports.

Sibling package installs intentionally use package builds with `--no-deps` plus small compatibility runtime specs. UnitMatch and SLAy currently declare conflicting NumPy/Pandas/Torch dependency ranges, while the pipeline only needs them importable through the code paths it calls. Revisit those dependency pins after real merge-stage data smokes.

The image downloads the MaxWell HDF5 compression plugin from the configured `MAXWELL_HDF5_PLUGIN_URL` build arg into `/usr/local/lib/plugin/libcompression.so` and sets `HDF5_PLUGIN_PATH=/usr/local/lib/plugin`. If the download URL stops returning a Linux shared object, the build fails with a message that the container needs an update; the entrypoint also fails early if the plugin is missing at runtime.

The image also installs lightweight Linux accounting tools used by `--phase-tune`: `time`, `sysstat` (`pidstat`, `iostat`, `sar`), and `procps`. Normal stage runs do not start those samplers; the pipeline only launches them around phase windows when `--phase-tune` is explicitly requested.

The Kilosort base image already includes conda at `/home/miniconda3`, and this image uses that single base environment. The live host `axon_recon` conda environment is not copied verbatim into the image because doing so would duplicate a large environment, may bake host-specific paths, and can disturb the Kilosort4 CUDA base stack. Instead, the image mirrors the repo `environment.yml` runtime intent plus container-only additions such as Kilosort4, UnitMatchPy, SLAy, and `mpi4py`; intentional version deviations are kept in the Dockerfile specs.

Current DockerHub tags pushed from this branch:

```text
adammwea/axon-recon:pipeline-v2
adammwea/axon-recon:20260512-pipeline-v2
```

If this repo has moved beyond the digest behind those tags, rebuild and push a fresh tag before relying on newer in-image CLI behavior in Shifter/NERSC.

## Rebuild + redeploy after code changes (NERSC one-liner)

After editing code in this repo, refresh the Shifter image used by sbatch jobs with:

```bash
containers/axon-recon/rebuild_shifter.sh
```

That wrapper picks the best container CLI on this host (`podman-hpc` on NERSC, `podman` or `docker` elsewhere), builds via `build_local_image.sh`, pushes to Docker Hub, then asks `shifterimg` to refresh the cached image so the next `shifter --image=...` invocation picks up the new digest. Pass a tag override as the first positional arg to publish under a different name (e.g. `containers/axon-recon/rebuild_shifter.sh adammwea/axon-recon:20260513-pipeline-v2`).

Prereqs (one-time):

```bash
podman-hpc login docker.io      # or `podman login docker.io` / `docker login`
```

Env overrides:
- `AXON_RECON_CONTAINER_CLI=podman` (or similar) — force a specific CLI.
- `AXON_RECON_REBUILD_SKIP_PUSH=1` — build only, no push / shifterimg refresh. Useful for iteration before you're ready to ship.
- `AXON_RECON_REBUILD_SKIP_SHIFTERIMG=1` — skip the `shifterimg pull` step (e.g. on the lab server).

## Wrapper Details

The wrapper mounts the repo at the same absolute path and sets writable cache locations under `/tmp/axon-recon-cache`. When the forwarded CLI args include `--config`, the wrapper inspects that runtime config and its `data:` YAML with a lightweight scanner, then mounts configured output roots and scratch roots read-write and the common raw H5 root read-only. Disable this with `--no-config-mounts` and add manual mounts with repeated `--mount host_path:container_path[:mode]` flags when needed.

```bash
tools/axon-recon-container --dry-run stages --help
tools/axon-recon-container --dry-run --mpi-ranks 2 stages preprocess --config debug/debug.runtime.yml
```

For CUDA-backed spikesort runs under Docker, request GPU passthrough explicitly with `--gpus all` or set `AXON_RECON_CONTAINER_GPUS=all`. Without Docker GPU passthrough, PyTorch inside the container cannot see a CUDA device and Kilosort will log `GPU usage: N/A` and `GPU memory: N/A`. This image also installs the NVML Python binding (`nvidia-ml-py`, imported by Kilosort as `pynvml`) so that, once CUDA is visible, Kilosort can report GPU utilization percentage in addition to GPU memory.

On POSIX hosts, the wrapper defaults to your current UID:GID so files written through mounted output directories stay owned by the invoking user. Use `--current-user` to make that explicit, `--user UID:GID` to override it, or `--user 0:0` if you intentionally want root inside the container.

```bash
tools/axon-recon-container --current-user stages --help
```

The equivalent generic override is `--user UID:GID`, or `AXON_RECON_CONTAINER_USER=UID:GID`.

Inside this image, `spikesort.phases.sort.engine: mea_analysis` is blocked by default because the legacy MEA_Analysis path can launch a nested Docker container. Use `engine: local_spikeinterface` for container and HPC runs. For intentional local debugging of nested container behavior, set `AXON_RECON_ALLOW_CONTAINER_MEA_ANALYSIS=1`.

## Smoke Checks

Inside a built image:

```bash
axon-recon-smoke-cli
python /opt/axon_recon/containers/axon-recon/smoke_imports.py
```

For a host-side syntax check that does not require all container-only packages to be installed locally:

```bash
python containers/axon-recon/smoke_imports.py --allow-missing
```
