#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# User-editable config
# ----------------------------

# Directory containing this config file
CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Raw data file to test
RAW_H5="$HOME/symlinks/ben-shalom_nas/raw_data/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR/241004/M08029/AxonTracking/000007/data.raw.h5"

# Keep MEA_Analysis outputs on pscratch for now
# NOTE: This must be an *output root*.
# MEA_Analysis will append <project>/<date>/<chip>/<scan>/<run>/<well00x>/... beneath it.
OUT_ROOT="$HOME/symlinks/pscratch/mea_outputs"

# Repos (adjust if yours differ)
MEA_REPO="$HOME/dev/pkgs/MEA_Analysis"
AXON_REPO="$HOME/dev/pkgs/axon_reconstructor"

# Sorter name expected by Mandar pipeline
SORTER="kilosort4"

# CPU workers (defaults to SLURM_CPUS_PER_TASK if present)
N_JOBS="${SLURM_CPUS_PER_TASK:-8}"

# Scratch (fast local on compute nodes). On interactive nodes this is usually set.
SCRATCH_DIR="${SLURM_TMPDIR:-}"

# Stage-back behavior when using scratch
STAGE_BACK="sorter"          # none|sorter|all
STAGE_BACK_MODE="copy"       # copy|move

# Shifter image to use for GPU spikesorting (recommended on Perlmutter).
# IMPORTANT: On Perlmutter, Shifter typically expects fully-qualified image URIs like:
#   docker:<repo>:<tag>
# If you omit the `docker:` prefix, some configurations may not apply the container you expect.
SHIFTER_IMAGE="${SHIFTER_IMAGE:-docker:adammwea/benshalomlab_spikesorter_shifter:v4}"

# Shifter environment hygiene:
# Slurm exports your host PATH into the container by default, so an activated conda env
# can accidentally override the image's python. These knobs let scripts force a container
# PATH and/or a specific python executable.
SHIFTER_CONTAINER_PATH="${SHIFTER_CONTAINER_PATH:-/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
SHIFTER_PYTHON="${SHIFTER_PYTHON:-python3}"

# GPU pinning.
# If this script accidentally runs on the host (not in a Slurm GPU step), some nodes expose all GPUs
# (e.g. "0,1,2,3"). Prefer the first device by default to avoid surprising multi-GPU behavior.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == *","* ]]; then
	CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES%%,*}"
else
	CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0}"
fi

# If running inside the Shifter image, you can have /entrypoint.sh auto-update MEA_Analysis
# to a specific branch before running the driver.
MEA_ANALYSIS_REPO_URL="https://github.com/roybens/MEA_Analysis.git"
MEA_ANALYSIS_BRANCH="dev_branch_aw_2"

# For lab server runs (optional; only used by lab scripts)
DOCKER_IMAGE=""  # e.g. "ghcr.io/<org>/<image>:<tag>"

# Maxwell HDF5 compression plugin
# Needed to read Maxwell-compressed .raw.h5 files outside the Shifter/Docker image.
# If not set, we attempt to auto-detect common locations in your workspace.
if [[ -z "${MAXWELL_HDF5_PLUGIN_DIR:-}" ]]; then
	for candidate in \
		"$CONFIG_DIR/vendor/maxwell_hdf5_plugin/Linux" \
		"$HOME/dev/pkgs/benshalomlab_spikesorter_shifter/maxwell_hdf5_plugin/Linux" \
		"$HOME/dev/pkgs/axon_reconstructor_dev_branch/environments/maxwell_hdf5_plugin/Linux"; do
		if [[ -d "$candidate" ]]; then
			MAXWELL_HDF5_PLUGIN_DIR="$candidate"
			break
		fi
	done
fi
