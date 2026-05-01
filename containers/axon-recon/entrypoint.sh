#!/usr/bin/env bash
set -euo pipefail

export AXON_RECON_CACHE_ROOT="${AXON_RECON_CACHE_ROOT:-/tmp/axon-recon-cache}"
export HOME="${HOME:-${AXON_RECON_CACHE_ROOT}/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/axon-recon-cache/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/axon-recon-cache/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/axon-recon-cache/numba}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/axon-recon-cache/pycache}"
cache_dirs=("$HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$PYTHONPYCACHEPREFIX")
if ! mkdir -p "${cache_dirs[@]}"; then
  echo "axon-recon-entrypoint: cannot create runtime cache directories under ${AXON_RECON_CACHE_ROOT}" >&2
  exit 70
fi
for cache_dir in "${cache_dirs[@]}"; do
  if [[ ! -w "$cache_dir" ]]; then
    echo "axon-recon-entrypoint: runtime cache directory is not writable: ${cache_dir}" >&2
    exit 70
  fi
done

if [[ $# -gt 0 && "${1}" == "axon-reconstructor" ]]; then
  exec "$@"
fi
if [[ $# -gt 0 && "${1}" != -* ]] && command -v "${1}" >/dev/null 2>&1; then
  exec "$@"
fi

exec axon-reconstructor "$@"