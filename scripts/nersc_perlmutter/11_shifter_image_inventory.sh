#!/usr/bin/env bash
set -euo pipefail

# Inspect what is actually inside the Shifter image on Perlmutter.
# This helps detect: stale Shifter cache, wrong tag, PATH/conda leakage.
#
# Usage (inside interactive GPU allocation recommended):
#   bash scripts/nersc_perlmutter/11_shifter_image_inventory.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_config.sh"

if [[ -z "${SLURM_JOB_ID:-}" && ! -x "/entrypoint.sh" ]]; then
  echo "ERROR: Run inside an interactive allocation (SLURM_JOB_ID unset)." >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_ROOT/gpu_checks"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/shifter_image_inventory_${TS}.log"

BANNER="IMAGE INVENTORY: shifter_image_inventory (inside Shifter)"
{
  echo "=== ${BANNER} ==="
  echo "Host: $(hostname)"
  echo "Time: ${TS}"
  echo "SHIFTER_IMAGE=${SHIFTER_IMAGE:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo
} | tee "$LOG_FILE"

# Try hard to avoid host env leaking into the container.
# NOTE: With --export=NONE, PATH may be empty, so we set PATH explicitly.
CONTAINER_PATH="${SHIFTER_CONTAINER_PATH:-/usr/local/bin:/usr/bin:/bin}"

cmd=(
  srun --export=NONE,PATH="$CONTAINER_PATH",HOME="$HOME",CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" \
    --ntasks=1 --cpus-per-task=1 --gpus=1 --image="$SHIFTER_IMAGE" \
    /bin/sh -c
)

# Everything below executes inside the container.
inner_script=$'set -eu\n\n'
inner_script+=$'echo "[container] uname:"; uname -a || true\n'
inner_script+=$'echo "[container] os-release:"; (cat /etc/os-release || true)\n'
inner_script+=$'echo "[container] PATH=$PATH"\n'
inner_script+=$'echo "[container] ls -ld /opt/conda /entrypoint.sh (if present):"; (ls -ld /opt/conda /entrypoint.sh 2>/dev/null || true)\n'
inner_script+=$'echo "[container] ls -ld /opt/maxwell_hdf5_plugin (if present):"; (ls -ld /opt/maxwell_hdf5_plugin 2>/dev/null || true)\n'
inner_script+=$'echo "[container] which python3:"; command -v python3 || true\n'
inner_script+=$'echo "[container] python3 -V:"; python3 -V || true\n'
inner_script+=$'echo "[container] sys.executable:"; python3 -c "import sys; print(sys.executable)" || true\n'
inner_script+=$'echo "[container] pip -V:"; (python3 -m pip -V || true)\n'
inner_script+=$'echo "[container] pip show torch/h5py:"; (python3 -m pip show torch h5py || true)\n'
inner_script+=$'echo "[container] import torch/h5py:"; python3 - <<"PY" || true\n'
inner_script+=$'import os, sys\n'
inner_script+=$'print("python:", sys.executable)\n'
inner_script+=$'try:\n'
inner_script+=$'  import torch\n'
inner_script+=$'  print("torch:", torch.__version__)\n'
inner_script+=$'  print("cuda available:", torch.cuda.is_available())\n'
inner_script+=$'  print("device count:", torch.cuda.device_count())\n'
inner_script+=$'  if torch.cuda.is_available() and torch.cuda.device_count() > 0:\n'
inner_script+=$'    print("device[0]:", torch.cuda.get_device_name(0))\n'
inner_script+=$'except Exception as e:\n'
inner_script+=$'  print("ERROR importing torch:", repr(e))\n'
inner_script+=$'try:\n'
inner_script+=$'  import h5py\n'
inner_script+=$'  print("h5py:", h5py.__version__)\n'
inner_script+=$'except Exception as e:\n'
inner_script+=$'  print("ERROR importing h5py:", repr(e))\n'
inner_script+=$'print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))\n'
inner_script+=$'PY\n'

{
  echo "Running inside Shifter (isolated):"
  echo "  ${cmd[*]} '<script>'"
  echo
} | tee -a "$LOG_FILE"

set +e
"${cmd[@]}" "$inner_script" 2>&1 | tee -a "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e

echo | tee -a "$LOG_FILE"
echo "Exit code: $rc" | tee -a "$LOG_FILE"
echo "Wrote: $LOG_FILE" | tee -a "$LOG_FILE"

exit "$rc"
