#!/usr/bin/env bash
set -euo pipefail

# GPU-node smoke test (inside Shifter):
# - runs on a GPU interactive allocation
# - runs the pipeline with --skip-spikesorting (so no Kilosort)
# - intentionally does NOT set HDF5_PLUGIN_PATH (expects container defaults)
#
# Usage:
#   salloc -A <acct> -C gpu -q interactive -t 00:30:00 -N 1 --gpus=1 --cpus-per-task=16
#   bash tools/smoke_tests/perlmutter/interactive_gpu_node/07_gpu_node_smoketest_no_sort_container_plugin_default.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_shared/00_config.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_shared/_nersc_shifter_helpers.sh"

if [[ ! -f "$RAW_H5" ]]; then
  echo "ERROR: RAW_H5 not found: $RAW_H5" >&2
  exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" && ! -x "/entrypoint.sh" ]]; then
  echo "ERROR: This script is intended to run inside an interactive GPU allocation (SLURM_JOB_ID unset)." >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"

# Ensure we are not relying on the host-style plugin override.
unset HDF5_PLUGIN_PATH || true
unset MAXWELL_HDF5_PLUGIN_DIR || true

export MEA_ANALYSIS_REPO_URL
export MEA_ANALYSIS_BRANCH
export MEA_ANALYSIS_AUTO_UPDATE=1
export MEA_ANALYSIS_AUTO_RUN=1

export MEA_ANALYSIS_DRIVER_CONSOLE_LEVEL="${MEA_ANALYSIS_DRIVER_CONSOLE_LEVEL:-DEBUG}"

RUN_TAG="gpu_node_smoketest_no_sort_container_plugin_default"
export MEA_ANALYSIS_RUN_BANNER="SMOKE TEST: ${RUN_TAG} (GPU node, inside Shifter, skip spikesorting, no HDF5_PLUGIN_PATH override)"
export MEA_ANALYSIS_SUBPROCESS_LOG_DIR="${MEA_ANALYSIS_SUBPROCESS_LOG_DIR:-${OUT_ROOT}/subprocess_logs/${RUN_TAG}}"
export MEA_ANALYSIS_SUBPROCESS_TEE_CONSOLE="${MEA_ANALYSIS_SUBPROCESS_TEE_CONSOLE:-1}"

DRIVER_ARGS=(
  "$RAW_H5"
  --output-dir "$OUT_ROOT"
  --skip-spikesorting
  --n-jobs "$N_JOBS"
  --debug
)

DRIVER_SCRIPT_REL="IPNAnalysis/run_pipeline_driver.py"

if ! ensure_shifter_available; then
  echo "ERROR: 'shifter' command not found in PATH (try: module load shifter)." >&2
  exit 4
fi

shifter_mod_args=()
if [[ -n "${SHIFTER_MODULES:-}" ]]; then
  shifter_mod_args+=("--module=${SHIFTER_MODULES}")
fi

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SHIFTER_IMAGE:-}" ]]; then
  CMD=(
    srun --ntasks=1 --cpus-per-task="$N_JOBS" --gpus=1 \
      shifter "${shifter_mod_args[@]}" --image="$SHIFTER_IMAGE" --env="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}" \
      /bin/bash -lc "cd \"$MEA_REPO\" && python3 -u \"$DRIVER_SCRIPT_REL\" ${DRIVER_ARGS[*]}"
  )
else
  echo "ERROR: SHIFTER_IMAGE not set or not in a Slurm allocation." >&2
  exit 3
fi

echo "GPU-node smoke test (no sorting, container plugin defaults):"
echo "  ${CMD[*]}"

"${CMD[@]}"

echo
echo "Done. Outputs rooted at: $OUT_ROOT"
