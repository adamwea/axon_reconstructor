# Container + MPI Strategy Note

Status: design note. Not a plan. The goal is to capture the problem and the option space so the cleanup plan can reference it. We are deferring the work — the lab-server flow uses host `axon-recon` + `mpirun` for non-GPU stages, and single-rank `axon-recon-container` for `spikesort.sort`. NERSC + multi-rank GPU is out-of-scope for now.

## Problem

`mpirun -np N axon-recon-container …` spawns N independent docker containers. Each container is its own PID/IPC/network namespace. The MPI runtime mpirun set up in the host (PMIx sockets, MPI_COMM_WORLD with N ranks, env vars like `OMPI_*`, `PMIX_*`) does not penetrate the docker namespace by default. So inside each container, `mpi4py` initializes a fresh `MPI_COMM_WORLD` of size 1 and `axon_recon.pipeline.mpi_adapter` thinks it's the only rank.

Result: all N containers process every target. With `--target-dataset 11,12 --limit-wells 1` we want each rank to handle one target; instead both containers race on the same outputs and trip `FileExistsError` at `kilosort4.py:127` (`Folder {output_folder} already exists`). The user labelled this "runs double" in `debug/mpirun.sh` and abandoned that path early.

What works today:
- `mpirun -np 2 axon-recon stages preprocess … --task-backend mpi` (host binary, no container) — preprocess and bootstrap_concat_binary validated end-to-end after `82ed42c`.
- `axon-recon-container --no-build --gpus all stages spikesort.sort … --force-restart` (single container, no mpirun) — sort validated end-to-end on GPU; both targets serialized through one rank.

What doesn't:
- `mpirun -np N axon-recon-container …` for N > 1.

## Why this is hard

To get MPI to span containers requires three things:

1. **Shared IPC/network**: containers must `--ipc=host --network=host` (or use a CNI overlay that supports MPI). Already partially achievable; the container_cli accepts `--ipc host` in the runtime YAML's `container_caps` block.
2. **PMIx wire-up**: mpirun communicates with each rank via PMIx (or PMI-1) sockets in `/tmp` or via env. Containers that don't share `/tmp` (and most do not) lose the wire-up. Mounting `/tmp` is a kludge; the right answer is `--pid=host --ipc=host --net=host` plus `OMPI_MCA_*` env passthrough so `ompi_proxy` finds the parent runtime.
3. **OMPI version parity**: the host's OpenMPI and the container's OpenMPI must be compatible (same major version, same wire protocol). Spikeinterface's `kilosort4-base` image ships its own OpenMPI; it might not match the host's. If they diverge, you get "PMIx_Init failed" or silent rank-0-only behavior.

The path through is well-trodden in HPC (Singularity/Apptainer is the usual answer), but the lab-server flow uses Docker, not Singularity, and the host OpenMPI version isn't pinned to the container's.

## Option space

### Option A — Pass MPI through (host → container)

Set up the container_cli to add `--ipc=host --network=host --pid=host` and forward `OMPI_*`, `PMIX_*`, `OMPI_MCA_*` env vars when invoked under mpirun. Mount the host's `/var/run/openmpi` (or wherever PMIx sockets live).

- Pro: single binary path; mpirun launches the container directly, axon-recon inside sees the parent rank correctly.
- Con: depends on host/container OMPI parity. Kilosort4 image bumps OpenMPI when upstream rebuilds; we'd be coupled to that.
- Con: `--pid=host --ipc=host --net=host` weakens the container's isolation; unfriendly to multi-tenant or NERSC where container privileges are restricted.
- Con: SystemV semaphore limits and shm_size interact in non-obvious ways once MPI shared memory is on.

### Option B — Run mpirun *inside* a single container

Launch one `docker run` with `--gpus all` and entrypoint `mpirun -np N axon-recon stages …`. The container ships its own mpirun and OpenMPI; axon-recon inside spawns ranks under the in-container mpirun.

- Pro: zero host-container coupling. Container is self-contained.
- Pro: the entrypoint is a single docker invocation; user doesn't need host mpirun.
- Pro: works the same on lab-server and NERSC (NERSC just runs `shifter` or `podman` with the same entrypoint).
- Con: needs `mpirun` and OpenMPI in the image. The current `kilosort4-base` ships OpenMPI 4.x, so this is essentially free. Verify at build time.
- Con: ranks all share the container's CPU set; we lose the host's `--map-by ppr:N:node:pe=K` topology binding. Inside the container we'd pass `--bind-to none` and let axon-recon's task allocation handle CPU pinning via `local_affinity` backend.
- Con: GPU partitioning across ranks is the user's responsibility (CUDA_VISIBLE_DEVICES per rank).

### Option C — Single-rank container, multi-rank host (current state)

Use the container only for stages that need its bundled software (spikesort.sort needs kilosort4). Use host `axon-recon` + `mpirun` for everything else (preprocess, reconstruct, summarize_sort, bombcell_label/merges once those are repaired).

- Pro: no MPI plumbing needed across the host/container boundary. Clean separation.
- Pro: validated end-to-end today. We have proof.
- Pro: the only "lost parallelism" is intra-sort (one well at a time per container invocation); kilosort4 is GPU-bound anyway and the GPU is the bottleneck.
- Con: workflow has two modes ("which binary do I use for which stage?"). Documentation burden.
- Con: doesn't generalize to NERSC where everything runs under shifter/podman.

## Recommendation

**Short term: Option C.** It's already working. The `axon-recon-container` wrapper documents which stages it's appropriate for (spikesort.sort, anything else needing the kilosort4 conda env). For everything else, use `mpirun -np N axon-recon …` directly. The cleanup plan does not need to address container+mpi.

**Medium term: Option B.** The entrypoint is a single docker invocation; ranks live inside. This generalizes to NERSC (shifter exec) and avoids host/container OMPI coupling. To implement:

1. Verify `mpirun` + `mpi4py` + OpenMPI 4.x present in the kilosort4-base image (`docker run kilosort4-base which mpirun && mpirun --version`).
2. Add a flag to `axon-recon-container` (e.g. `--mpi-ranks N`) that, when set, prepends `mpirun -np N --bind-to none` to the inner command and keeps `--task-backend mpi` on axon-recon.
3. Verify GPU partitioning: by default each rank sees all GPUs. axon-recon already supports `cuda_visible_devices` per task slot (resources.profiles.<name>.task_allocation.cuda_visible_devices, if present); confirm and document.
4. Smoke: `axon-recon-container --no-build --gpus all --mpi-ranks 2 stages spikesort.sort --config debug/debug.runtime.yml --target-dataset 11,12 --limit-wells 1 --limit-segments 2`. Expect each target to land on a distinct rank.

Estimated work: 1 day, mostly testing.

**Long term: Option A.** If the lab moves to NERSC heavily and we want a single binary to drive both lab and HPC, Option A becomes appealing because it lets HPC schedulers (SLURM, PBS) treat the container as just another MPI rank. But it's the most coupling-prone of the three and we should not pursue it before Option B is in place.

## Open questions for whenever this is picked up

- Do we ever need cross-node MPI? If the lab-server is the only target, single-node Option B is enough. If NERSC multi-node is in scope, Option A or a Singularity port becomes mandatory.
- Is `cuda_visible_devices` per-task already wired in `axon_recon.pipeline.cpu_allocation`? Last I checked there was a slot-level field but no rank-aware partitioning of GPUs. Worth grepping before Option B work begins.
- The lab server has 1 GPU. With 2 ranks and 1 GPU, we'd need MPS (multi-process service) or one rank running CPU-only. For 2 datasets × 1 well, serializing through one rank is faster than splitting one GPU two ways. Document the bound.

## What this note is NOT

It is not a slice in the cleanup plan. It is reference material for a future planner to read before designing the actual container+MPI work. The cleanup plan's "out of scope" §6 already flags this; this note expands on that flag.
