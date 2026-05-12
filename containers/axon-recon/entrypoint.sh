#!/usr/bin/env bash
set -euo pipefail

export AXON_RECON_CACHE_ROOT="${AXON_RECON_CACHE_ROOT:-/tmp/axon-recon-cache}"
export HOME="${HOME:-${AXON_RECON_CACHE_ROOT}/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/axon-recon-cache/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/axon-recon-cache/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/axon-recon-cache/numba}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/axon-recon-cache/pycache}"
export HDF5_PLUGIN_PATH="${HDF5_PLUGIN_PATH:-/usr/local/lib/plugin}"
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

maxwell_plugin_found=0
IFS=':' read -r -a hdf5_plugin_dirs <<<"$HDF5_PLUGIN_PATH"
for hdf5_plugin_dir in "${hdf5_plugin_dirs[@]}"; do
  if [[ -f "${hdf5_plugin_dir}/libcompression.so" ]]; then
    maxwell_plugin_found=1
    break
  fi
done
if [[ "$maxwell_plugin_found" -ne 1 ]]; then
  echo "axon-recon-entrypoint: MaxWell HDF5 compression plugin libcompression.so was not found in HDF5_PLUGIN_PATH=${HDF5_PLUGIN_PATH}. Rebuild/update the axon-recon container image; the plugin download URL may have changed." >&2
  exit 70
fi

if [[ $# -gt 0 && ( "${1}" == "axon-recon" || "${1}" == "mpirun" ) ]]; then
  exec "$@"
fi
if [[ $# -gt 0 && "${1}" != -* ]] && command -v "${1}" >/dev/null 2>&1; then
  exec "$@"
fi

exec axon-recon "$@"