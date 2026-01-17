#!/usr/bin/env bash
set -euo pipefail

# Run this *inside* an interactive GPU allocation on Perlmutter.
# Example allocation (adjust account/queue/time as needed):
#   salloc -A <acct> -C gpu -q interactive -t 02:00:00 -N 1 --gpus=1 --cpus-per-task=32
# Then:
#   ./scripts/nersc_perlmutter/10_gpu_spikesort_interactive.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_config.sh"

if [[ ! -f "$RAW_H5" ]]; then
  echo "ERROR: RAW_H5 not found: $RAW_H5" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

if [[ -z "${SCRATCH_DIR}" ]]; then
  echo "WARNING: SLURM_TMPDIR not set; running without --scratch-dir (slower)." >&2
fi

cd "$MEA_REPO"

CMD=(
  python3 IPNAnalysis/run_pipeline_driver.py "$RAW_H5"
  --output-dir "$OUT_ROOT"
  --sorter "$SORTER"
  --require-gpu
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES_VALUE"
  --n-jobs "$N_JOBS"
)

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
