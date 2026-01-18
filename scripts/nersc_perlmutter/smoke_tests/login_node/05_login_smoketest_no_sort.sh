#!/usr/bin/env bash
set -euo pipefail

# Login-node smoke test: verifies path contract, well enumeration, and that the driver/routine
# can import its dependencies.
#
# This intentionally does NOT run Kilosort (no GPU on login nodes, and heavy compute is disallowed).
#
# Usage:
#   bash scripts/nersc_perlmutter/smoke_tests/login_node/05_login_smoketest_no_sort.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_shared/00_config.sh"

if [[ ! -f "$RAW_H5" ]]; then
  echo "ERROR: RAW_H5 not found: $RAW_H5" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

# Maxwell-compressed .raw.h5 requires the vendor HDF5 decompression plugin.
# In Shifter/Docker this is configured already; on the host we try to point HDF5 at the plugin.
if [[ ! -x "/entrypoint.sh" ]]; then
  if [[ -n "${MAXWELL_HDF5_PLUGIN_DIR:-}" && -d "${MAXWELL_HDF5_PLUGIN_DIR}" ]]; then
    export HDF5_PLUGIN_PATH="${MAXWELL_HDF5_PLUGIN_DIR}"
    echo "Using Maxwell HDF5 plugin via HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH"
  else
    echo "WARNING: Maxwell HDF5 plugin not found; reading $RAW_H5 may fail." >&2
    echo "Set MAXWELL_HDF5_PLUGIN_DIR to .../maxwell_hdf5_plugin/Linux or run inside the Shifter image." >&2
  fi
fi

# Ask the driver to emit DEBUG logs to the console during smoke tests.
export MEA_ANALYSIS_DRIVER_CONSOLE_LEVEL="${MEA_ANALYSIS_DRIVER_CONSOLE_LEVEL:-DEBUG}"

RUN_TAG="login_smoketest_no_sort"
export MEA_ANALYSIS_RUN_BANNER="SMOKE TEST: ${RUN_TAG} (login node, skip spikesorting)"

# Persist per-well routine stdout/stderr to disk; the driver log file only contains driver messages.
export MEA_ANALYSIS_SUBPROCESS_LOG_DIR="${MEA_ANALYSIS_SUBPROCESS_LOG_DIR:-${OUT_ROOT}/subprocess_logs/${RUN_TAG}}"
export MEA_ANALYSIS_SUBPROCESS_TEE_CONSOLE="${MEA_ANALYSIS_SUBPROCESS_TEE_CONSOLE:-1}"

cd "$MEA_REPO"

# This path is for debugging only: avoid --require-gpu and avoid spikesorting.
CMD=(
  python3 IPNAnalysis/run_pipeline_driver.py "$RAW_H5"
  --output-dir "$OUT_ROOT"
  --skip-spikesorting
  --n-jobs "$N_JOBS"
  --debug
)

echo "Login-node smoke test (no sorting):"
echo "  ${CMD[*]}"

"${CMD[@]}"

echo
echo "Done. Outputs rooted at: $OUT_ROOT"