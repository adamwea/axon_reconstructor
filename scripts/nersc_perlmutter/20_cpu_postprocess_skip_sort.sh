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

CMD=(
  python3 IPNAnalysis/run_pipeline_driver.py "$RAW_H5"
  --output-dir "$OUT_ROOT"
  --skip-spikesorting
  --n-jobs "$N_JOBS"
)

echo "Running MEA_Analysis post-processing (skip spikesorting):"
echo "  ${CMD[*]}"

"${CMD[@]}"

echo
echo "Done. Outputs rooted at: $OUT_ROOT"
echo "Next: run 30_run_axon_reconstructor.sh"
