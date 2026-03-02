#!/usr/bin/env bash
set -euo pipefail

# Editable local runner for stage-barrier scope runs.
#
# Keeps project-template assets in tools/debug:
# - debug.env
# - cross_well_config.yml
#
# Uses canonical package CLI commands only:
# - scope-config-build
# - scope-run
#
# Typical usage:
#   bash tools/debug/run_scope_combo.sh
#   STAGE_ORDER_CSV="preprocess,spikesort" bash tools/debug/run_scope_combo.sh
#   DRY_RUN=1 bash tools/debug/run_scope_combo.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"

ENV_FILE="${ENV_FILE:-$ROOT_DIR/debug.env}"
CROSS_WELL_CONFIG="${CROSS_WELL_CONFIG:-$ROOT_DIR/cross_well_config.yml}"

DEFAULT_STAGE_ORDER_CSV="preprocess,spikesort,unit_match,merge_update,waveforms,templates,reconstruct,analysis"
STAGE_ORDER_CSV="${STAGE_ORDER_CSV:-$DEFAULT_STAGE_ORDER_CSV}"

PER_WELL_PARALLELISM="${PER_WELL_PARALLELISM:-1}"
FAIL_FAST="${FAIL_FAST:-1}"
FORCE_RESTART="${FORCE_RESTART:-}"
DRY_RUN="${DRY_RUN:-1}"
DEBUG_LOGS="${DEBUG_LOGS:-0}"

# Optional output override for generated scope config and run summary.
SCOPE_CONFIG_OUT="${SCOPE_CONFIG_OUT:-$ROOT_DIR/logs/scope_combo_generated.json}"
SUMMARY_OUT="${SUMMARY_OUT:-$ROOT_DIR/logs/scope_combo_summary.json}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE" >&2
  exit 2
fi
if [[ ! -f "$CROSS_WELL_CONFIG" ]]; then
  echo "Cross-well config not found: $CROSS_WELL_CONFIG" >&2
  exit 2
fi

mkdir -p "$(dirname "$SCOPE_CONFIG_OUT")" "$(dirname "$SUMMARY_OUT")"

cd "$REPO_ROOT"
export PYTHONPATH=src

echo "run_scope_combo: env_file=$ENV_FILE"
echo "run_scope_combo: cross_well_config=$CROSS_WELL_CONFIG"
echo "run_scope_combo: stage_order=$STAGE_ORDER_CSV"

build_cmd=(
  python -m axon_reconstructor.cli scope-config-build
  --cross-well-config "$CROSS_WELL_CONFIG"
  --env-file "$ENV_FILE"
  --out "$SCOPE_CONFIG_OUT"
  --stage-order "$STAGE_ORDER_CSV"
  --per-well-parallelism "$PER_WELL_PARALLELISM"
  --fail-fast "$FAIL_FAST"
)

if [[ -n "$FORCE_RESTART" ]]; then
  build_cmd+=(--force-restart "$FORCE_RESTART")
fi

echo "run_scope_combo: building scope config"
printf '  %q ' "${build_cmd[@]}"; printf '\n'
"${build_cmd[@]}" >/dev/null

echo "run_scope_combo: generated scope config at $SCOPE_CONFIG_OUT"

run_cmd=(
  python -m axon_reconstructor.cli scope-run
  --config "$SCOPE_CONFIG_OUT"
  --env-file "$ENV_FILE"
  --summary-out "$SUMMARY_OUT"
)

if [[ "$DRY_RUN" == "1" ]]; then
  run_cmd+=(--dry-run)
fi
if [[ "$DEBUG_LOGS" == "1" ]]; then
  run_cmd+=(--debug)
fi

echo "run_scope_combo: executing scope run"
printf '  %q ' "${run_cmd[@]}"; printf '\n'
"${run_cmd[@]}"

echo "run_scope_combo: done"
