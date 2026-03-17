#!/usr/bin/env bash
set -euo pipefail

# Runs spikesorting in isolated mode by default (sorting only, no merge/analyzer/reports)
# via the canonical runner entrypoint.
# Usage:
#   bash tools/debug/run_mea_phase_sorting.sh [--config <path>] [extra stage args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CALLER_PWD="$(pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${REPO_ROOT}/tools/debug/debug.config.yml"
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Run spikesorting stage only.

Default behavior:
- target_phase=sorting
- run_analyzer=false
- run_reports=false

Usage:
  bash tools/debug/run_mea_phase_sorting.sh [--config <path>] [extra stage args...]

Examples:
  bash tools/debug/run_mea_phase_sorting.sh
  bash tools/debug/run_mea_phase_sorting.sh --force-restart
  bash tools/debug/run_mea_phase_sorting.sh --resume-from sorting
  bash tools/debug/run_mea_phase_sorting.sh --stage-kwargs '{"run_analyzer": true, "run_reports": true}'
EOF
  exit 0
fi

if [[ "${1:-}" == "--config" ]]; then
  CONFIG_PATH="${2:?missing config path after --config}"
  shift 2
fi

if [[ "${CONFIG_PATH}" != /* ]]; then
  if [[ -f "${CALLER_PWD}/${CONFIG_PATH}" ]]; then
    CONFIG_PATH="${CALLER_PWD}/${CONFIG_PATH}"
  elif [[ -f "${REPO_ROOT}/${CONFIG_PATH}" ]]; then
    CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH}"
  fi
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

DEFAULT_STAGE_KWARGS='{"target_phase":"sorting","run_analyzer":false,"run_reports":false}'
INJECT_DEFAULT_STAGE_KWARGS=1
for arg in "$@"; do
  if [[ "$arg" == "--stage-kwargs" ]] || [[ "$arg" == "--stage-kwargs-file" ]]; then
    INJECT_DEFAULT_STAGE_KWARGS=0
    break
  fi
done

run_with_python() {
  local py_cmd="$1"
  shift
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
  if [[ "$INJECT_DEFAULT_STAGE_KWARGS" -eq 1 ]]; then
    exec $py_cmd -m axon_reconstructor.cli stage spikesort --config "$CONFIG_PATH" --stage-kwargs "$DEFAULT_STAGE_KWARGS" "$@"
  fi
  exec $py_cmd -m axon_reconstructor.cli stage spikesort --config "$CONFIG_PATH" "$@"
}

if [[ -n "$CFG_PYTHON_PATH" ]] && [[ -x "$CFG_PYTHON_PATH" ]]; then
  run_with_python "$CFG_PYTHON_PATH" "$@"
fi

if [[ -n "$CFG_CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
  if [[ "$INJECT_DEFAULT_STAGE_KWARGS" -eq 1 ]]; then
    exec conda run -n "$CFG_CONDA_ENV" python -m axon_reconstructor.cli stage spikesort --config "$CONFIG_PATH" --stage-kwargs "$DEFAULT_STAGE_KWARGS" "$@"
  fi
  exec conda run -n "$CFG_CONDA_ENV" python -m axon_reconstructor.cli stage spikesort --config "$CONFIG_PATH" "$@"
fi

if command -v axon-reconstructor >/dev/null 2>&1; then
  if [[ "$INJECT_DEFAULT_STAGE_KWARGS" -eq 1 ]]; then
    exec axon-reconstructor stage spikesort --config "$CONFIG_PATH" --stage-kwargs "$DEFAULT_STAGE_KWARGS" "$@"
  fi
  exec axon-reconstructor stage spikesort --config "$CONFIG_PATH" "$@"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Neither axon-reconstructor nor ${PYTHON_BIN} was found in PATH." >&2
  exit 127
fi

run_with_python "$PYTHON_BIN" "$@"
