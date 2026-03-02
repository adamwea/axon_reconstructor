#!/usr/bin/env bash
set -euo pipefail

# Editable local runner for invoking any combination of pipeline stages.
#
# Defaults are intentionally project-like:
# - Uses tools/debug/debug.env for shared defaults.
# - Runs canonical package CLI commands.
# - Lets you edit stage list + per-stage extra args directly in this file.
#
# Typical usage:
#   bash tools/debug/run_stage_combo.sh
#   STAGES_CSV="preprocess,spikesort" bash tools/debug/run_stage_combo.sh
#   DRY_RUN=1 bash tools/debug/run_stage_combo.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"

ENV_FILE="${ENV_FILE:-$ROOT_DIR/debug.env}"

# Keep this in tools/debug as a project-template artifact.
CROSS_WELL_CONFIG="${CROSS_WELL_CONFIG:-$ROOT_DIR/cross_well_config.yml}"

# Edit this list directly or provide STAGES_CSV at runtime.
DEFAULT_STAGES_CSV="preprocess,spikesort,waveforms,templates,reconstruct,analysis"
STAGES_CSV="${STAGES_CSV:-$DEFAULT_STAGES_CSV}"

# Optional run mode toggles.
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"

# Optional common overrides (when unset, values resolve from ENV_FILE).
H5_PATH_OVERRIDE="${H5_PATH_OVERRIDE:-}"
STREAM_ID_OVERRIDE="${STREAM_ID_OVERRIDE:-}"
MEA_OUTPUT_ROOT_OVERRIDE="${MEA_OUTPUT_ROOT_OVERRIDE:-}"

# Optional common stage kwargs source.
STAGE_KWARGS_FILE="${STAGE_KWARGS_FILE:-}"

# Optional additional args applied to every stage invocation.
COMMON_EXTRA_ARGS=()

# Optional per-stage extra args (edit these arrays as needed).
PREPROCESS_ARGS=()
SPIKESORT_ARGS=()
WAVEFORMS_ARGS=()
TEMPLATES_ARGS=()
RECONSTRUCT_ARGS=()
ANALYSIS_ARGS=()

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE" >&2
  exit 2
fi

# Parse comma-separated stage list.
IFS=',' read -r -a _raw_stages <<< "$STAGES_CSV"
STAGES=()
for s in "${_raw_stages[@]}"; do
  trimmed="$(echo "$s" | xargs)"
  if [[ -n "$trimmed" ]]; then
    STAGES+=("$trimmed")
  fi
done

if [[ ${#STAGES[@]} -eq 0 ]]; then
  echo "No stages requested. Set STAGES_CSV or edit DEFAULT_STAGES_CSV." >&2
  exit 2
fi

build_stage_args() {
  local stage="$1"
  local -a args=(
    python -m axon_reconstructor.cli stage "$stage"
    --env-file "$ENV_FILE"
  )

  if [[ -n "$H5_PATH_OVERRIDE" ]]; then
    args+=(--h5-path "$H5_PATH_OVERRIDE")
  fi
  if [[ -n "$STREAM_ID_OVERRIDE" ]]; then
    args+=(--stream-id "$STREAM_ID_OVERRIDE")
  fi
  if [[ -n "$MEA_OUTPUT_ROOT_OVERRIDE" ]]; then
    args+=(--mea-output-root "$MEA_OUTPUT_ROOT_OVERRIDE")
  fi
  if [[ -n "$STAGE_KWARGS_FILE" ]]; then
    args+=(--stage-kwargs-file "$STAGE_KWARGS_FILE")
  fi

  args+=("${COMMON_EXTRA_ARGS[@]}")

  case "$stage" in
    preprocess)
      args+=("${PREPROCESS_ARGS[@]}")
      ;;
    spikesort)
      args+=("${SPIKESORT_ARGS[@]}")
      ;;
    waveforms)
      args+=("${WAVEFORMS_ARGS[@]}")
      ;;
    templates)
      args+=("${TEMPLATES_ARGS[@]}")
      ;;
    reconstruct)
      args+=("${RECONSTRUCT_ARGS[@]}")
      ;;
    analysis)
      args+=("${ANALYSIS_ARGS[@]}")
      ;;
    *)
      echo "Unsupported stage in STAGES_CSV: $stage" >&2
      return 2
      ;;
  esac

  printf '%q ' "${args[@]}"
  printf '\n'
}

cd "$REPO_ROOT"
export PYTHONPATH=src

echo "run_stage_combo: env_file=$ENV_FILE"
echo "run_stage_combo: cross_well_config=$CROSS_WELL_CONFIG"
echo "run_stage_combo: stages=${STAGES[*]}"

for stage in "${STAGES[@]}"; do
  cmd="$(build_stage_args "$stage")"
  echo "[$stage] $cmd"

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  if ! eval "$cmd"; then
    echo "Stage failed: $stage" >&2
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
      exit 1
    fi
  fi
done

echo "run_stage_combo: done"
