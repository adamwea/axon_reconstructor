#!/usr/bin/env bash
set -euo pipefail

# Fast GPU gate check (inside Shifter): verifies the environment can see a CUDA GPU.
# This is meant to fail fast before you spend time on preprocessing or Kilosort.
#
# Usage:
#   salloc -A <acct> -C gpu -q interactive -t 00:10:00 -N 1 --gpus=1 --cpus-per-task=4
#   bash scripts/nersc_perlmutter/08_gpu_require_gpu_dry_check.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_config.sh"

if [[ ! -f "$RAW_H5" ]]; then
  echo "ERROR: RAW_H5 not found: $RAW_H5" >&2
  exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" && ! -x "/entrypoint.sh" ]]; then
  echo "ERROR: Run inside an interactive GPU allocation (SLURM_JOB_ID unset)." >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"

export MEA_ANALYSIS_REPO_URL
export MEA_ANALYSIS_BRANCH
export MEA_ANALYSIS_AUTO_UPDATE=1
export MEA_ANALYSIS_AUTO_RUN=1

export MEA_ANALYSIS_DRIVER_CONSOLE_LEVEL="${MEA_ANALYSIS_DRIVER_CONSOLE_LEVEL:-DEBUG}"

RUN_TAG="gpu_require_gpu_dry_check"
export MEA_ANALYSIS_RUN_BANNER="GPU CHECK: ${RUN_TAG} (inside Shifter, driver --require-gpu --dry)"
export MEA_ANALYSIS_SUBPROCESS_LOG_DIR="${MEA_ANALYSIS_SUBPROCESS_LOG_DIR:-${OUT_ROOT}/subprocess_logs/${RUN_TAG}}"
export MEA_ANALYSIS_SUBPROCESS_TEE_CONSOLE="${MEA_ANALYSIS_SUBPROCESS_TEE_CONSOLE:-1}"

DRIVER_ARGS=(
  "$RAW_H5"
  --output-dir "$OUT_ROOT"
  --require-gpu
  --dry
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES_VALUE"
  --n-jobs "$N_JOBS"
  --debug
)

DRIVER_SCRIPT_REL="IPNAnalysis/run_pipeline_driver.py"

# Prevent host conda from hijacking python inside Shifter.
CONTAINER_PATH="${SHIFTER_CONTAINER_PATH:-/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
SHIFTER_PY="${SHIFTER_PYTHON:-python3}"

if [[ -n "${SLURM_JOB_ID:-}" && -n "${SHIFTER_IMAGE:-}" ]]; then
  # Run inside Shifter; assume $MEA_REPO is visible inside the container (home is mounted).
  CMD=(
    srun --export=NONE,PATH="$CONTAINER_PATH",HOME="$HOME",CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
      --ntasks=1 --cpus-per-task=1 --gpus=1 --image="$SHIFTER_IMAGE" --chdir="$MEA_REPO"
    "$SHIFTER_PY" -u "$DRIVER_SCRIPT_REL" "${DRIVER_ARGS[@]}"
  )
else
  echo "ERROR: SHIFTER_IMAGE not set or not in a Slurm allocation." >&2
  exit 3
fi

echo "GPU dry gate check (require-gpu + dry):"
echo "  ${CMD[*]}"

"${CMD[@]}"

echo
echo "Done. If this passed, CUDA is visible inside the container."
