#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# User-editable config
# ----------------------------

# Raw data file to test
RAW_H5="$HOME/symlinks/ben-shalom_nas/raw_data/B6J_DensityTest_10012024_AR/B6J_DensityTest_10012024_AR/241004/M08029/AxonTracking/000007/data.raw.h5"

# Keep MEA_Analysis outputs on pscratch for now
OUT_ROOT="$HOME/symlinks/pscratch/mea_outputs/B6J_DensityTest_10012024_AR/241004/M08029/AxonTracking/000007"

# Repos (adjust if yours differ)
MEA_REPO="$HOME/dev/pkgs/MEA_Analysis"
AXON_REPO="$HOME/dev/pkgs/axon_reconstructor"

# Sorter name expected by Mandar pipeline
SORTER="kilosort4"

# CPU workers (defaults to SLURM_CPUS_PER_TASK if present)
N_JOBS="${SLURM_CPUS_PER_TASK:-32}"

# Scratch (fast local on compute nodes). On interactive nodes this is usually set.
SCRATCH_DIR="${SLURM_TMPDIR:-}"

# Stage-back behavior when using scratch
STAGE_BACK="sorter"          # none|sorter|all
STAGE_BACK_MODE="copy"       # copy|move

# GPU pinning (single GPU interactive node => 0)
CUDA_VISIBLE_DEVICES_VALUE="0"

# For lab server runs (optional; only used by lab scripts)
DOCKER_IMAGE=""  # e.g. "ghcr.io/<org>/<image>:<tag>"
