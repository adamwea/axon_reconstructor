#!/usr/bin/env bash
set -euo pipefail

# Minimal Shifter inventory (Perlmutter), following NERSC docs style.
# - Run via `srun ... shifter --image=...` inside an interactive allocation
#
# Usage (inside interactive GPU allocation recommended):
#   bash tools/smoke_tests/perlmutter/interactive_gpu_node/11_shifter_image_inventory.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_shared/00_config.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: Run inside an interactive allocation (SLURM_JOB_ID unset)." >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_ROOT/gpu_checks"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/shifter_image_inventory_${TS}.log"

{
  echo "=== IMAGE INVENTORY (minimal) ==="
  echo "Host: $(hostname)"
  echo "Time: ${TS}"
  echo "SHIFTER_IMAGE=${SHIFTER_IMAGE:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo
  echo "[host] which shifter:"; command -v shifter || true
} | tee "$LOG_FILE"

if ! command -v shifter >/dev/null 2>&1; then
  {
    echo "ERROR: 'shifter' not found in PATH."
    echo "On Perlmutter this often means you need to load the shifter module first:"
    echo "  module load shifter"
  } | tee -a "$LOG_FILE" >&2
  exit 3
fi

container_script=$'set -e\n'
container_script+=$'echo "[container] uname:"; uname -a || true\n'
container_script+=$'echo "[container] os-release:"; (cat /etc/os-release || true)\n'
container_script+=$'echo "[container] PATH=$PATH"\n'
container_script+=$'echo "[container] which python3:"; command -v python3 || true\n'
container_script+=$'echo "[container] python3 -V:"; python3 -V || true\n'
container_script+=$'python3 - <<"PY"\n'
container_script+=$'import os, sys\n'
container_script+=$'print("python:", sys.executable)\n'
container_script+=$'for mod in ("torch", "h5py"):\n'
container_script+=$'  try:\n'
container_script+=$'    m = __import__(mod)\n'
container_script+=$'    print(f"{mod}:", getattr(m, "__version__", "<no version>"))\n'
container_script+=$'  except Exception as e:\n'
container_script+=$'    print(f"{mod} import error:", repr(e))\n'
container_script+=$'try:\n'
container_script+=$'  import torch\n'
container_script+=$'  print("torch.cuda.is_available():", torch.cuda.is_available())\n'
container_script+=$'  print("torch.cuda.device_count():", torch.cuda.device_count())\n'
container_script+=$'except Exception as e:\n'
container_script+=$'  print("torch cuda check error:", repr(e))\n'
container_script+=$'print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))\n'
container_script+=$'PY\n'

{
  echo "Running: srun ... shifter --image ... (Slurm step)"
  echo "  srun --ntasks=1 --gpus=1 shifter --image=... --env=CUDA_VISIBLE_DEVICES=... /bin/bash -lc '<script>'"
  echo
} | tee -a "$LOG_FILE"

set +e
srun --ntasks=1 --gpus=1 shifter --image="$SHIFTER_IMAGE" --env="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}" /bin/bash -lc "$container_script" 2>&1 | tee -a "$LOG_FILE"
rc_srun=${PIPESTATUS[0]}
set -e
echo "Exit code (srun): ${rc_srun}" | tee -a "$LOG_FILE"

echo "Wrote: $LOG_FILE" | tee -a "$LOG_FILE"

exit 0
