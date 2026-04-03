#!/usr/bin/env bash
set -euo pipefail

# Login-node smoke test with --force-restart: validates a clean re-run path.
#
# Usage:
#   bash tools/smoke_tests/perlmutter/login_node/06_login_smoketest_force_restart.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_shared/00_config.sh"

if [[ -z "${RAW_H5:-}" ]]; then
  echo "ERROR: RAW_H5 is not set. Set RAW_H5 or configure tools/smoke_tests/perlmutter/smoke_tests.local.toml." >&2
  exit 2
fi

if [[ ! -f "$RAW_H5" ]]; then
  echo "ERROR: RAW_H5 not found on disk: $RAW_H5" >&2
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

export MEA_ANALYSIS_DRIVER_CONSOLE_LEVEL="${MEA_ANALYSIS_DRIVER_CONSOLE_LEVEL:-DEBUG}"

RUN_TAG="login_smoketest_force_restart"
export MEA_ANALYSIS_RUN_BANNER="SMOKE TEST: ${RUN_TAG} (login node, skip spikesorting, --force-restart)"
export MEA_ANALYSIS_SUBPROCESS_LOG_DIR="${MEA_ANALYSIS_SUBPROCESS_LOG_DIR:-${OUT_ROOT}/subprocess_logs/${RUN_TAG}}"
export MEA_ANALYSIS_SUBPROCESS_TEE_CONSOLE="${MEA_ANALYSIS_SUBPROCESS_TEE_CONSOLE:-1}"

cd "$MEA_REPO"

CMD=(
  python3 IPNAnalysis/run_pipeline_driver.py "$RAW_H5"
  --output-dir "$OUT_ROOT"
  --skip-spikesorting
  --force-restart
  --n-jobs "$N_JOBS"
  --debug
)

echo "Login-node smoke test (force restart, no sorting):"
echo "  ${CMD[*]}"

"${CMD[@]}"

echo
echo "Done. Outputs rooted at: $OUT_ROOT"
