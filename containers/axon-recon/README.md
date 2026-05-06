# axon_reconstructor Container

This image is intended to run the same `axon-reconstructor` CLI as the host environment, with all pipeline stage selectors passed through unchanged.

## Simple Wrapper UX

After installing this package, or from this repo with `tools/axon-recon-container`, use the container wrapper with the same arguments you would pass to `axon-reconstructor`:

```bash
axon-recon-container stages reconstruct --config debug/debug.runtime.yml
```

With the default `axon-recon:local` image, the wrapper builds the image if it is missing and rebuilds it when the source fingerprint no longer matches the image label. Use `--no-build` to skip that check, `--rebuild` to force a rebuild, or `--build` when you intentionally want to build/update a non-default `--image` tag.

Real-data smoke commands for the active debug runtime:

```bash
axon-recon-container stages preprocess --config debug/debug.runtime.yml --limit-segments 2 --limit-datasets 2 --limit-wells-per-dataset 1
axon-recon-container --gpus all stages spikesort --config debug/debug.runtime.yml --limit-segments 2 --limit-datasets 2 --limit-wells-per-dataset 1
axon-recon-container stages reconstruct --config debug/debug.runtime.yml --limit-segments 2 --limit-datasets 2 --limit-wells-per-dataset 1 --limit-units 5
```

Those `--limit-*` flags are normal pipeline CLI flags. The wrapper does not interpret stage names or limits; it only handles image build/update, mounts, cache paths, and Docker execution. For spikesort, `--limit-segments` limits the preprocessed segment manifest consumed by `spikesort.bootstrap_concat_binary` before the bootstrapped binary recording is materialized, so downstream sort phases read the smaller concatenated recording.

## Local Build

Basic image build from the repo root:

```bash
docker build -f containers/axon-recon/Dockerfile -t axon-recon:local .
```

Full dependency builds should use the helper so sibling checkouts are copied into a temporary context and installed as packages:

```bash
containers/axon-recon/build_local_image.sh --image axon-recon:local
```

The default repo-root build installs `axon_reconstructor`, the active runtime Python dependency set, `spikeinterface==0.103.2`, and `mpi4py`. It does not bake local data, scratch outputs, credentials, or sibling workspace paths into the image. The helper detects sibling `../UnitMatch/UnitMatchPy` and `../SLAy` checkouts when present, copies them under `external/` in a temporary context, and passes build args so imports are normal installed-package imports.

Sibling package installs intentionally use package builds with `--no-deps` plus small compatibility runtime specs. UnitMatch and SLAy currently declare conflicting NumPy/Pandas/Torch dependency ranges, while the pipeline only needs them importable through the code paths it calls. Revisit those dependency pins after real merge-stage data smokes.

The image downloads the MaxWell HDF5 compression plugin from the configured `MAXWELL_HDF5_PLUGIN_URL` build arg into `/usr/local/lib/plugin/libcompression.so` and sets `HDF5_PLUGIN_PATH=/usr/local/lib/plugin`. If the download URL stops returning a Linux shared object, the build fails with a message that the container needs an update; the entrypoint also fails early if the plugin is missing at runtime.

The image also installs lightweight Linux accounting tools used by `--phase-tune`: `time`, `sysstat` (`pidstat`, `iostat`, `sar`), and `procps`. Normal stage runs do not start those samplers; the pipeline only launches them around phase windows when `--phase-tune` is explicitly requested.

The Kilosort base image already includes conda at `/home/miniconda3`, and this image uses that single base environment. The live host `axon_recon` conda environment is not copied verbatim into the image because doing so would duplicate a large environment, may bake host-specific paths, and can disturb the Kilosort4 CUDA base stack. Instead, the image mirrors the repo `environment.yml` runtime intent plus container-only additions such as Kilosort4, UnitMatchPy, SLAy, and `mpi4py`; intentional version deviations are kept in the Dockerfile specs.

Current DockerHub tags pushed from this branch:

```text
adammwea/axon-recon:pipeline-v2
adammwea/axon-recon:20260501-pipeline-v2
```

If this repo has moved beyond the digest behind those tags, rebuild and push a fresh tag before relying on newer in-image CLI behavior in Shifter/NERSC.

## Smoke Checks

Inside a built image:

```bash
axon-recon-smoke-cli
python /opt/axon_reconstructor/containers/axon-recon/smoke_imports.py
```

For a host-side syntax check that does not require all container-only packages to be installed locally:

```bash
python containers/axon-recon/smoke_imports.py --allow-missing
```

## Local Wrapper

From the repo root, the host wrapper mirrors the normal CLI shape:

```bash
tools/axon-recon-container stages reconstruct --config debug/debug.runtime.yml
```

Dry-run the resolved build and Docker commands:

```bash
tools/axon-recon-container --dry-run stages --help
```

The wrapper mounts the repo at the same absolute path and sets writable cache locations under `/tmp/axon-recon-cache`. When the forwarded CLI args include `--config`, the wrapper inspects that runtime config and its `data:` YAML with a lightweight scanner, then mounts configured output roots and scratch roots read-write and the common raw H5 root read-only. Disable this with `--no-config-mounts` and add manual mounts with repeated `--mount host_path:container_path[:mode]` flags when needed.

For CUDA-backed spikesort runs under Docker, request GPU passthrough explicitly with `--gpus all` or set `AXON_RECON_CONTAINER_GPUS=all`. Without Docker GPU passthrough, PyTorch inside the container cannot see a CUDA device and Kilosort will log `GPU usage: N/A` and `GPU memory: N/A`. This image also installs the NVML Python binding (`nvidia-ml-py`, imported by Kilosort as `pynvml`) so that, once CUDA is visible, Kilosort can report GPU utilization percentage in addition to GPU memory.

On POSIX hosts, the wrapper now defaults to your current UID:GID so files written through mounted output directories stay owned by the invoking user. Use `--current-user` to make that explicit, `--user UID:GID` to override it, or `--user 0:0` if you intentionally want root inside the container:

```bash
tools/axon-recon-container --current-user stages --help
```

The equivalent generic override is `--user UID:GID`, or `AXON_RECON_CONTAINER_USER=UID:GID`.

Inside this image, `spikesort.phases.sort.engine: mea_analysis` is blocked by default because the legacy MEA_Analysis path can launch a nested Docker container. Use `engine: local_spikeinterface` for container and HPC runs. For intentional local debugging of nested container behavior, set `AXON_RECON_ALLOW_CONTAINER_MEA_ANALYSIS=1`.

## Shifter Shape

Later at NERSC, import the pushed image with:

```bash
shifterimg -v pull docker:<registry>/<image>:<tag>
```

CPU-only stage example:

```bash
#SBATCH --image=docker:<registry>/<image>:<tag>
#SBATCH --constraint=cpu

srun shifter axon-reconstructor stages preprocess reconstruct --config /mounted/path/debug.runtime.yml
```

Kilosort-backed spikesort example:

```bash
#SBATCH --image=docker:<registry>/<image>:<tag>
#SBATCH --constraint=gpu
#SBATCH --module=gpu,cuda-mpich

export MPICH_GPU_SUPPORT_ENABLED=1
srun shifter axon-reconstructor stages spikesort --config /mounted/path/debug.runtime.yml
```

Use absolute paths in Slurm volume directives, keep large data on mounted filesystems, and treat CUDA-aware `mpi4py` validation as NERSC-deferred until tested on Perlmutter.