# Container, mpi4py, And NERSC Optimization Guardrails

Status: guardrail document. Once agentic development begins, treat this file as locked. Do not edit it unless Adam explicitly asks for guardrail changes.

This document defines the acceptance contract for preparing the container, wrapper, MPI hooks, and parallel execution model for NERSC Perlmutter. Local work should make the code Shifter-ready without claiming NERSC validation until it has actually run at NERSC.

## Operating Contract

- Commit frequently after each coherent accepted slice, using an `ai:` prefix in the commit subject.
- Update `debug/commit_log.md` after every AI commit.
- Keep implementation slices small: container dependency, wrapper behavior, MPI adapter, rank partitioning, Shifter docs, or smoke validation.
- Validate normal host CLI and container CLI separately.
- Use real-data smoke tests with CLI debug flags. Do not run full-scope data tests unless Adam explicitly requests them.
- Record what was locally validated and what remains NERSC-deferred.

## Container Contract

The container is a full-pipeline execution environment for every supported active stage and phase. It is not a special-purpose image for only sorting or reconstruction.

Required behavior:

- The normal CLI and container CLI accept the same stage and phase selectors.
- `axon-recon-container` forwards pipeline arguments unchanged after resolving image, mounts, cache paths, user mapping, and Docker execution details.
- CPU-only selectors must run without requiring GPU resources.
- Kilosort-backed `spikesort.sort` may require GPU resources.
- The image must not require nested Docker for the HPC-compatible sort path.
- `engine: local_spikeinterface` is the intended container/HPC sort engine.
- Legacy MEA_Analysis/Docker sorting must be blocked or clearly guarded inside the container unless Adam explicitly enables it for debugging.

## Shifter And Perlmutter Rules

- Keep images compact. Treat images approaching the practical Shifter import risk zone as a stop condition for review.
- Do not bake raw data, scratch outputs, caches, credentials, SSH keys, or local absolute user paths into the image.
- Shifter images are read-only at runtime and root may be squashed. All caches and outputs must go to writable mounts.
- Test local Docker as a non-root user before declaring a slice Shifter-ready.
- Use mounted CFS/scratch paths for data and outputs, not image paths.
- Use `#SBATCH --image=...` in Slurm scripts for Shifter jobs.
- Use absolute NERSC paths in `#SBATCH --volume` lines; do not rely on environment variable expansion there.
- Use `/global/cfs/...` for Community File System paths.
- GPU jobs need the Shifter GPU module and CUDA-aware MPI setup when MPI/GPU support is used.

## mpi4py And MPI Guardrails

MPI support must be additive and explicit.

Rules:

- Non-MPI execution remains the default.
- Importing the package should not require a working MPI launcher unless MPI mode is selected.
- MPI rank/size detection should be isolated behind a small adapter.
- Target partitioning should happen at rank level before local thread/process fanout.
- Rank 0 owns global summaries unless there is a tested rank-summary merge step.
- Rank metadata must appear in logs, JSONL events, and summaries when MPI mode is active.
- Avoid nested process pools, broad subprocess use, and nested container launches inside MPI ranks unless proven safe on Perlmutter.
- Add fake-MPI tests before requiring `srun` validation.

NERSC-specific validation that cannot be done locally must stay marked as deferred until actually tested:

- Shifter image pull/import.
- `srun shifter ...` execution.
- Cray MPICH / Shifter MPI swapping.
- CUDA-aware MPI with `MPICH_GPU_SUPPORT_ENABLED=1`.
- Multi-node target partitioning.
- GPU visibility for Kilosort-backed sorting.

## Wrapper And Mount Guardrails

- The wrapper should auto-mount runtime config data roots, scratch roots, and output roots where possible.
- Config and repo mounts should be read-only unless a development mode intentionally mounts writable source.
- Output, scratch, cache, Matplotlib, Python, Numba, and temporary directories must be writable.
- The wrapper should run with host UID:GID by default on POSIX so mounted outputs remain user-owned.
- The wrapper must fail clearly if required host paths from config cannot be mounted.
- Dry-run mode must print resolved image, mounts, user, environment, and forwarded command.
- New stage/phase selectors added to the normal CLI must work through the wrapper without wrapper code changes.

## CPU/GPU Selector Guardrails

- `preprocess` is CPU-capable unless future profiling proves otherwise.
- `reconstruct` is CPU-capable unless future profiling proves otherwise.
- `spikesort.sort` with Kilosort is GPU-backed.
- `stages all` includes sorting and may require GPU resources if run as one job.
- Production workflows at NERSC should prefer stage-split jobs once artifact/resume boundaries are validated: CPU preprocess, GPU spikesort, CPU reconstruct.
- The wrapper or job docs should fail clearly when a GPU-required selector is launched without visible GPU support.
- CPU-only selectors should not be rejected just because the image contains CUDA-capable packages.

## Required Tests And Acceptance Criteria

### Local container import smoke

Acceptance criteria:

- Container import smoke passes for `axon_recon`, `spikeinterface`, Kilosort/SpikeInterface sorter support, `UnitMatchPy` package discovery or supported import path, `slay` through the pipeline-supported path, and `mpi4py`.
- Import smoke records package versions.
- Any intentionally deferred or unavailable import is documented with a reason.

### Container CLI parity smoke

Example:

```bash
axon-recon-container --dry-run stages preprocess \
  --config debug/debug.runtime.yml \
  --limit-datasets 1 \
  --limit-wells 1 \
  --limit-segments 2
```

Acceptance criteria:

- Forwarded command matches the normal CLI shape.
- Runtime config mounts include data, scratch, output, and cache paths with correct read/write intent.
- Debug flags are forwarded unchanged.
- The wrapper does not parse or rewrite stage/phase semantics.

### Non-root container smoke

Acceptance criteria:

- Container helper smoke passes with host UID:GID.
- HOME/cache/temp paths are writable.
- Created output/log files are owned by the host user.
- No writes are attempted inside read-only package/image paths.

### CPU selector smoke

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

- Runs without GPU flags.
- Logs show CPU-capable resource classes.
- Outputs and logs land on mounted writable paths.

### GPU sort smoke

Acceptance criteria:

- Kilosort-backed sort runs through `local_spikeinterface` in the current container environment.
- No nested Docker path is invoked.
- GPU visibility and sorter version are logged.
- Sort outputs are usable by downstream summarize, bombcell label, merge, and reconstruct resolvers.

### Fake MPI adapter tests

Acceptance criteria:

- Rank/size detection can be simulated without launching MPI.
- Target partitioning is deterministic and non-overlapping across ranks.
- Rank 0 summary ownership is enforced.
- Non-MPI behavior is unchanged when MPI mode is disabled.

### NERSC validation checklist

Acceptance criteria before claiming NERSC-ready:

- Image can be pulled/imported by Shifter.
- CPU-only selector runs on a CPU node with writable mounts.
- GPU sort selector runs on a GPU node with expected CUDA visibility.
- `mpi4py` works under NERSC's expected MPI/Shifter setup.
- Multi-rank target partitioning works with `srun`.
- Logs and summaries include rank metadata.
