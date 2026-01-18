#!/usr/bin/env bash
set -euo pipefail

# GPU readiness check (inside Shifter): verifies CUDA visibility + torch works.
# Writes a timestamped log under OUT_ROOT/gpu_checks/.
#
# Usage:
#   salloc -A <acct> -C gpu -q interactive -t 00:10:00 -N 1 --gpus=1 --cpus-per-task=4
#   bash tools/smoke_tests/perlmutter/interactive_gpu_node/09_gpu_readiness_check.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_shared/00_config.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_shared/_nersc_shifter_helpers.sh"

if [[ -z "${SLURM_JOB_ID:-}" && ! -x "/entrypoint.sh" ]]; then
  echo "ERROR: Run inside an interactive GPU allocation (SLURM_JOB_ID unset)." >&2
  exit 2
fi

mkdir -p "$OUT_ROOT/gpu_checks"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_ROOT/gpu_checks/gpu_readiness_${TS}.log"

RUN_TAG="gpu_readiness_check"
BANNER="GPU CHECK: ${RUN_TAG} (inside Shifter)"

echo "=== ${BANNER} ===" | tee "$LOG_FILE"

echo "Host: $(hostname)" | tee -a "$LOG_FILE"
echo "Time: ${TS}" | tee -a "$LOG_FILE"
echo "SHIFTER_IMAGE=${SHIFTER_IMAGE:-}" | tee -a "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" | tee -a "$LOG_FILE"
echo | tee -a "$LOG_FILE"

if ! ensure_shifter_available; then
  echo "WARNING: 'shifter' command not found in PATH (module load shifter may be required)." | tee -a "$LOG_FILE" >&2
fi

shifter_mod_args=()
if [[ -n "${SHIFTER_MODULES:-}" ]]; then
  shifter_mod_args+=("--module=${SHIFTER_MODULES}")
fi

container_script="$(cat <<'SH'
set -euo pipefail

echo "=== container basics ==="
uname -a || true
cat /etc/os-release || true

echo
echo "=== python/torch/h5py/cuda ==="
python3 - <<'PY'
import os
import sys

print("python:", sys.executable)
print("python version:", sys.version)

import h5py
import torch

print("h5py:", h5py.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"device[{i}]:", torch.cuda.get_device_name(i))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
SH
)"

if [[ -x "/entrypoint.sh" ]]; then
  echo "Running torch/CUDA check inside current container..." | tee -a "$LOG_FILE"
  /bin/bash -lc "$container_script" 2>&1 | tee -a "$LOG_FILE"
elif [[ -n "${SLURM_JOB_ID:-}" && -n "${SHIFTER_IMAGE:-}" ]]; then
  echo "Running torch/CUDA check via srun + shifter (NERSC-doc style)..." | tee -a "$LOG_FILE"
  srun --ntasks=1 --cpus-per-task=1 --gpus=1 \
    shifter "${shifter_mod_args[@]}" --image="$SHIFTER_IMAGE" --env="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}" \
    /bin/bash -lc "$container_script" 2>&1 | tee -a "$LOG_FILE"
else
  echo "ERROR: Not inside Shifter and SHIFTER_IMAGE not set." | tee -a "$LOG_FILE" >&2
  exit 3
fi

echo | tee -a "$LOG_FILE"
echo "Wrote: $LOG_FILE" | tee -a "$LOG_FILE"
