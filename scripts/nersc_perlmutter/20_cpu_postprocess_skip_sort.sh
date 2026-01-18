#!/usr/bin/env bash
set -euo pipefail

# Run this on a CPU node (batch or interactive) *after* sorter_output exists under OUT_ROOT.
# Example interactive CPU allocation:
#   salloc -A <acct> -C cpu -q interactive -t 02:00:00 -N 1 --cpus-per-task=32
# Then:
#   ./scripts/nersc_perlmutter/20_cpu_postprocess_skip_sort.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_config.sh"

if [[ ! -f "$RAW_H5" ]]; then
  echo "ERROR: RAW_H5 not found: $RAW_H5" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

cd "$MEA_REPO"

DRIVER_SCRIPT_REL="IPNAnalysis/run_pipeline_driver.py"

export MEA_ANALYSIS_REPO_URL
export MEA_ANALYSIS_BRANCH
export MEA_ANALYSIS_AUTO_UPDATE=1
export MEA_ANALYSIS_AUTO_RUN=1

DRIVER_ARGS=(
  "$RAW_H5"
  --output-dir "$OUT_ROOT"
  --skip-spikesorting
  --n-jobs "$N_JOBS"
)

if [[ -x "/entrypoint.sh" && -d "/MEA_Analysis" ]]; then
  CMD=(/entrypoint.sh "${DRIVER_ARGS[@]}")
elif [[ -n "${SLURM_JOB_ID:-}" && -n "${SHIFTER_IMAGE:-}" ]]; then
  CMD=(
    srun --ntasks=1 --cpus-per-task="$N_JOBS" --image="$SHIFTER_IMAGE" --chdir="$MEA_REPO"
    python3 -u "$DRIVER_SCRIPT_REL" "${DRIVER_ARGS[@]}"
  )
else
  CMD=(python3 -u "$DRIVER_SCRIPT_REL" "${DRIVER_ARGS[@]}")
fi

echo "Running MEA_Analysis post-processing (skip spikesorting):"
echo "  ${CMD[*]}"

"${CMD[@]}"

echo
echo "Done. Outputs rooted at: $OUT_ROOT"
echo "Next: run 30_run_axon_reconstructor.sh"
