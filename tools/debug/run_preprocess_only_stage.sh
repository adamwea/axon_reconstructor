#!/usr/bin/env bash
set -euo pipefail

# Runs only stage-1 preprocessing via the canonical runner entrypoint.
# Usage:
#   bash tools/debug/run_preprocess_only_stage.sh [--config <path>] [extra stage args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${REPO_ROOT}/tools/debug/debug.config.yml"
if [[ "${1:-}" == "--config" ]]; then
  CONFIG_PATH="${2:?missing config path after --config}"
  shift 2
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 2
fi

_cfg_python_executable_path() {
  awk '
    /^[[:space:]]*python:[[:space:]]*$/ {in_python=1; next}
    in_python && /^[^[:space:]]/ {in_python=0}
    in_python && /^[[:space:]]+executable_path:[[:space:]]*/ {
      sub(/^[[:space:]]+executable_path:[[:space:]]*/, "")
      gsub(/^"|"$/, "")
      gsub(/^\x27|\x27$/, "")
      print
      exit
    }
  ' "$CONFIG_PATH"
}

_cfg_conda_env_name() {
  awk '
    /^[[:space:]]*python:[[:space:]]*$/ {in_python=1; next}
    in_python && /^[^[:space:]]/ {in_python=0}
    in_python && /^[[:space:]]+conda_env_name:[[:space:]]*/ {
      sub(/^[[:space:]]+conda_env_name:[[:space:]]*/, "")
      print
      exit
    }
  ' "$CONFIG_PATH" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
}

CFG_PYTHON_PATH_RAW="${PYTHON_BIN:-$(_cfg_python_executable_path || true)}"
CFG_PYTHON_PATH="${CFG_PYTHON_PATH_RAW//\$\{HOME\}/$HOME}"
CFG_CONDA_ENV="${CONDA_ENV_NAME:-$(_cfg_conda_env_name || true)}"

if [[ -n "$CFG_PYTHON_PATH" ]] && [[ -x "$CFG_PYTHON_PATH" ]]; then
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
  exec "$CFG_PYTHON_PATH" -m axon_reconstructor.cli stage preprocess --config "$CONFIG_PATH" "$@"
fi

if [[ -n "$CFG_CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
  exec conda run -n "$CFG_CONDA_ENV" python -m axon_reconstructor.cli stage preprocess --config "$CONFIG_PATH" "$@"
fi

if command -v axon-reconstructor >/dev/null 2>&1; then
  exec axon-reconstructor stage preprocess --config "$CONFIG_PATH" "$@"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Neither axon-reconstructor nor ${PYTHON_BIN} was found in PATH." >&2
  exit 127
fi

# Fallback for editable/clone workflows where console scripts are not installed.
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
exec "$PYTHON_BIN" -m axon_reconstructor.cli stage preprocess --config "$CONFIG_PATH" "$@"
