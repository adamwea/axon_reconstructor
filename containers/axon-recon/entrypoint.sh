#!/usr/bin/env bash
set -euo pipefail

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/axon-recon-cache/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/axon-recon-cache/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/axon-recon-cache/numba}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/axon-recon-cache/pycache}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$PYTHONPYCACHEPREFIX"

if [[ $# -gt 0 && "${1}" == "axon-reconstructor" ]]; then
  exec "$@"
fi

exec axon-reconstructor "$@"