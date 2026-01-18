#!/usr/bin/env bash
set -euo pipefail

# Small helper functions for NERSC/Perlmutter shells.
# Intended to be sourced by scripts under scripts/nersc_perlmutter/smoke_tests/.

nersc_init_module_cmd() {
  # "module" is often a shell function provided by Lmod.
  # In non-interactive shells it may not be initialized.
  if command -v module >/dev/null 2>&1; then
    return 0
  fi

  # Common Lmod init locations.
  if [[ -r /usr/share/lmod/lmod/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
  elif [[ -r /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi

  command -v module >/dev/null 2>&1
}

nersc_try_module_load() {
  local mod="$1"
  if ! nersc_init_module_cmd; then
    return 1
  fi

  # module is a shell function; "module -t avail" can be slow/noisy.
  # Best effort only.
  module load "$mod" >/dev/null 2>&1 || return 1
  return 0
}

ensure_shifter_available() {
  if command -v shifter >/dev/null 2>&1; then
    return 0
  fi

  # On Perlmutter, shifter is typically available via modules.
  nersc_try_module_load shifter || true

  command -v shifter >/dev/null 2>&1
}
