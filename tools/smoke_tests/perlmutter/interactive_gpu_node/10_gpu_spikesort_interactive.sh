#!/usr/bin/env bash
set -euo pipefail

# Run this *inside* an interactive GPU allocation on Perlmutter.
# Example allocation (adjust account/queue/time as needed):
#   salloc -A <acct> -C gpu -q interactive -t 02:00:00 -N 1 --gpus=1 --cpus-per-task=32
# Then:
#   bash tools/smoke_tests/perlmutter/interactive_gpu_node/10_gpu_spikesort_interactive.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_shared/00_config.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_shared/_nersc_shifter_helpers.sh"

if [[ ! -f "$RAW_H5" ]]; then
  echo "ERROR: RAW_H5 not found: $RAW_H5" >&2
  exit 1
fi

# Basic sanity check: this script is intended for an interactive GPU allocation.
# When run on a login node, SLURM vars are typically unset and GPUs unavailable.
if [[ -z "${SLURM_JOB_ID:-}" && -z "${SLURM_CLUSTER_NAME:-}" && ! -x "/entrypoint.sh" ]]; then
  echo "WARNING: SLURM_JOB_ID not set; are you on a login node? This script should be run inside an interactive GPU allocation." >&2
fi

if ! command -v nvidia-smi >/dev/null 2>&1 && ! command -v rocminfo >/dev/null 2>&1 && [[ ! -x "/entrypoint.sh" ]]; then
  echo "ERROR: No GPU tools detected (nvidia-smi/rocminfo missing). Run this inside a GPU allocation (e.g., salloc -C gpu --gpus=1 ...)." >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"

if [[ -z "${SCRATCH_DIR}" ]]; then
  echo "WARNING: SLURM_TMPDIR not set; running without --scratch-dir (slower)." >&2
fi

cd "$MEA_REPO"

DRIVER_SCRIPT_REL="IPNAnalysis/run_pipeline_driver.py"

# Prevent host conda from hijacking python inside Shifter.
CONTAINER_PATH="${SHIFTER_CONTAINER_PATH:-/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
SHIFTER_PY="${SHIFTER_PYTHON:-python3}"

if ! ensure_shifter_available; then
  echo "ERROR: 'shifter' command not found in PATH (try: module load shifter)." >&2
  exit 4
fi

shifter_mod_args=()
if [[ -n "${SHIFTER_MODULES:-}" ]]; then
  shifter_mod_args+=("--module=${SHIFTER_MODULES}")
fi

# When launching via `srun shifter ...`, we must be able to find the host-side `shifter`
# executable. Do NOT clobber PATH to a container-only PATH for the `shifter` process itself;
# instead, pass the container PATH into the image via `--env=PATH=...`.
SHIFTER_BIN="$(command -v shifter || true)"
if [[ -z "$SHIFTER_BIN" ]]; then
  echo "ERROR: 'shifter' resolved earlier but is not in PATH now." >&2
  exit 4
fi

# Minimal PATH for the host process that runs `shifter`.
HOST_SHIFTER_PATH="$(dirname "$SHIFTER_BIN"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Make the run debuggable: persist per-well stdout/stderr and stamp logs.
RUN_TAG="gpu_spikesort_interactive"
export MEA_ANALYSIS_RUN_BANNER="SPIKESORT: ${RUN_TAG} (GPU node, inside Shifter)"
export MEA_ANALYSIS_DRIVER_CONSOLE_LEVEL="INFO"
export MEA_ANALYSIS_SUBPROCESS_LOG_DIR="$OUT_ROOT/subprocess_logs/$RUN_TAG"
export MEA_ANALYSIS_SUBPROCESS_TEE_CONSOLE="1"

export MEA_ANALYSIS_REPO_URL
export MEA_ANALYSIS_BRANCH
export MEA_ANALYSIS_AUTO_UPDATE=1
export MEA_ANALYSIS_AUTO_RUN=1

DRIVER_ARGS=(
  "$RAW_H5"
  --output-dir "$OUT_ROOT"
  --sorter "$SORTER"
  --require-gpu
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES_VALUE"
  --n-jobs "$N_JOBS"
)

# Quick preflight to prove we're using a CUDA-capable torch *inside the container*.
# This intentionally uses a clean PATH so an activated host conda env can't mask the image's Python.
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SHIFTER_IMAGE:-}" ]]; then
  echo "Preflight: python/torch inside Shifter (clean PATH)" >&2
  srun --export=NONE,PATH="$HOST_SHIFTER_PATH",HOME="$HOME",CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
    --ntasks=1 --cpus-per-task=1 --gpus=1 --chdir="$MEA_REPO" \
    shifter "${shifter_mod_args[@]}" --image="$SHIFTER_IMAGE" \
    --env="PATH=$CONTAINER_PATH" \
    "$SHIFTER_PY" - <<'PY'
import os
import sys

print("python:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print("device[0]:", torch.cuda.get_device_name(0))
except Exception as e:
    print("ERROR: torch import/check failed:", repr(e))
    raise

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
fi

if [[ -x "/entrypoint.sh" && -d "/MEA_Analysis" ]]; then
  # If you happen to already be inside an image that provides an entrypoint, use it.
  CMD=(/entrypoint.sh "${DRIVER_ARGS[@]}")
elif [[ -n "${SLURM_JOB_ID:-}" && -n "${SHIFTER_IMAGE:-}" ]]; then
  # Preferred Perlmutter flow: run the driver inside Shifter.
  CMD=(
    srun --export=ALL,PATH="$HOST_SHIFTER_PATH",HOME="$HOME",CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
      --ntasks=1 --cpus-per-task="$N_JOBS" --gpus=1 --chdir="$MEA_REPO" \
      shifter "${shifter_mod_args[@]}" --image="$SHIFTER_IMAGE" \
      --env="PATH=$CONTAINER_PATH" \
      "$SHIFTER_PY" -u "$DRIVER_SCRIPT_REL" "${DRIVER_ARGS[@]}"
  )
else
  # Fallback: run directly on the host Python environment.
  CMD=(python3 -u "$DRIVER_SCRIPT_REL" "${DRIVER_ARGS[@]}")
fi

if [[ -n "${SCRATCH_DIR}" ]]; then
  CMD+=(
    --scratch-dir "$SCRATCH_DIR"
    --stage-back "$STAGE_BACK"
    --stage-back-mode "$STAGE_BACK_MODE"
  )
fi

echo "Running MEA_Analysis spikesorting on GPU node:"
echo "  ${CMD[*]}"

"${CMD[@]}"

echo
echo "Done. Outputs rooted at: $OUT_ROOT"
echo "Next: run 20_cpu_postprocess_skip_sort.sh (CPU node/login)"
