# axon_reconstructor Container

This image is intended to run the same `axon-reconstructor` CLI as the host environment, with all pipeline stage selectors passed through unchanged.

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
tools/axon-recon-container --image axon-recon:local stages reconstruct --config debug/debug.runtime.yml
```

Dry-run the resolved Docker command:

```bash
tools/axon-recon-container --dry-run --image axon-recon:local stages --help
```

The wrapper mounts the repo at the same absolute path and sets writable cache locations under `/tmp/axon-recon-cache`. When the forwarded CLI args include `--config`, the wrapper inspects that runtime config and its `data:` YAML with a lightweight scanner, then mounts configured output roots and scratch roots read-write and the common raw H5 root read-only. Disable this with `--no-config-mounts` and add manual mounts with repeated `--mount host_path:container_path[:mode]` flags when needed.

For Shifter-style/root-squash smoke tests and to avoid root-owned host outputs, run Docker as your current UID/GID:

```bash
tools/axon-recon-container --image axon-recon:local --current-user stages --help
```

The equivalent generic option is `--user UID:GID`, or `AXON_RECON_CONTAINER_USER=UID:GID`.

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