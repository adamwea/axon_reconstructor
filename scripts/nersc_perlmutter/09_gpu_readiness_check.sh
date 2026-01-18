#!/usr/bin/env bash
set -euo pipefail

# GPU readiness check (inside Shifter): verifies CUDA visibility + torch works.
# Writes a timestamped log under OUT_ROOT/gpu_checks/.
#
# Usage:
#   salloc -A <acct> -C gpu -q interactive -t 00:10:00 -N 1 --gpus=1 --cpus-per-task=4
#   bash scripts/nersc_perlmutter/09_gpu_readiness_check.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_config.sh"

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

torch_check_py="$(cat <<'PY'
import os
import sys
import traceback

print("python:", sys.executable)

try:
  import torch
except Exception as e:
  print("ERROR: torch import failed:", repr(e))
  traceback.print_exc()
  raise

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
  for i in range(torch.cuda.device_count()):
    print(f"device[{i}]:", torch.cuda.get_device_name(i))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
)"

# When you have an activated host conda env, it can override `python3` even inside Shifter
# because Slurm exports your PATH into the container.
# Use a container-focused PATH and also srun --export=NONE to isolate from host env.
CONTAINER_PATH="${SHIFTER_CONTAINER_PATH:-/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

import_check_py="$(cat <<'PY'
import sys

print("python:", sys.executable)
import h5py
import torch
print("h5py:", h5py.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
PY
)"

probe_sh="$(cat <<'SH'
set -eu

echo "PATH=$PATH"
command -v python3 || true

try_py() {
  py="$1"
  if [ -x "$py" ]; then
  echo ""
  echo "### trying: $py"
  "$py" - <<'PY'
import os
import sys
import importlib

print("python:", sys.executable)

for mod in ("torch", "h5py"):
  try:
    m = importlib.import_module(mod)
    print(f"{mod}:", getattr(m, "__version__", "<no __version__>"))
  except Exception as e:
    print(f"ERROR importing {mod}:", repr(e))

try:
  import torch
  print("cuda available:", torch.cuda.is_available())
  print("device count:", torch.cuda.device_count())
  if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("device[0]:", torch.cuda.get_device_name(0))
except Exception as e:
  print("ERROR torch cuda check:", repr(e))

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
  fi
}

# Try common python locations (conda images, system python, etc.)
try_py /opt/conda/bin/python3
try_py /opt/conda/bin/python
try_py /usr/local/bin/python3
try_py /usr/bin/python3
try_py /bin/python3

echo ""
echo "### trying: python3 from PATH"
python3 - <<'PY'
import sys
print("python:", sys.executable)
try:
  import torch
  print("torch:", torch.__version__)
except Exception as e:
  print("ERROR importing torch:", repr(e))
PY
SH
)"

run_srun() {
  # Run a command, tee output, but do not fail the whole script.
  # Usage: run_srun "label" <command...>
  local label="$1"; shift
  echo "--- ${label} ---" | tee -a "$LOG_FILE"
  set +e
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "[exit_code] ${label}: ${rc}" | tee -a "$LOG_FILE"
  echo | tee -a "$LOG_FILE"
  return 0
}

IN_SHIFTER=0
if [[ -x "/entrypoint.sh" ]]; then
  IN_SHIFTER=1
fi

if [[ "$IN_SHIFTER" -eq 1 ]]; then
  echo "Running torch/CUDA check inside current container..." | tee -a "$LOG_FILE"
  PATH="$CONTAINER_PATH" python3 -u -c "$torch_check_py" 2>&1 | tee -a "$LOG_FILE"
elif [[ -n "${SLURM_JOB_ID:-}" && -n "${SHIFTER_IMAGE:-}" ]]; then
  run_srun "torch/CUDA via srun --image (current PATH)" \
    srun --ntasks=1 --cpus-per-task=1 --gpus=1 --image="$SHIFTER_IMAGE" \
      python3 -u -c "$torch_check_py"

  run_srun "torch/CUDA via srun --image (CONTAINER_PATH)" \
    srun --export=ALL,PATH="$CONTAINER_PATH",HOME="$HOME",CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
      --ntasks=1 --cpus-per-task=1 --gpus=1 --image="$SHIFTER_IMAGE" \
      python3 -u -c "$torch_check_py"

  run_srun "torch/CUDA via srun --export=NONE (isolated)" \
    srun --export=NONE,PATH="$CONTAINER_PATH",HOME="$HOME",CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
      --ntasks=1 --cpus-per-task=1 --gpus=1 --image="$SHIFTER_IMAGE" \
      /bin/sh -c "$probe_sh"
else
  echo "ERROR: Not inside Shifter and SHIFTER_IMAGE not set." | tee -a "$LOG_FILE" >&2
  exit 3
fi

echo | tee -a "$LOG_FILE"
echo "Wrote: $LOG_FILE" | tee -a "$LOG_FILE"
