#!/usr/bin/env bash
set -euo pipefail

# Single-unit minimal stage runner for footprinting + reconstruction debugging.
#
# Default target unit (editable):
#   /home/adamm/dev/symlinks/local_RBS_data/outputs/Media_Density_T3_07012025_AR/250728/M07137/AxonTracking/000225/well001/reconstruction_outputs/by_unit/unit_112
#
# Typical usage:
#   bash tools/debug/run_single_unit_reconstruct.sh
#   FORCE_RESTART=1 bash tools/debug/run_single_unit_reconstruct.sh
#   ENABLE_GIFS=1 bash tools/debug/run_single_unit_reconstruct.sh
#   DRY_RUN=1 bash tools/debug/run_single_unit_reconstruct.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"

ENV_FILE="${ENV_FILE:-$ROOT_DIR/debug.env}"
EXAMPLE_UNIT_DIR="/home/adamm/dev/symlinks/local_RBS_data/outputs/Media_Density_T3_07012025_AR/250728/M07137/AxonTracking/000225/well001/reconstruction_outputs/by_unit/unit_112"
UNIT_DIR="${UNIT_DIR:-$EXAMPLE_UNIT_DIR}"
UNIT_BASE_DIR="${UNIT_BASE_DIR:-${EXAMPLE_UNIT_DIR%/unit_*}}"
UNIT_ID="${UNIT_ID:-}"

# Runtime toggles
FORCE_RESTART="${FORCE_RESTART:-0}"   # 0|1
FORCE_RESTART_WAVEFORMS="${FORCE_RESTART_WAVEFORMS:-}"   # optional 0|1; falls back to FORCE_RESTART
FORCE_RESTART_SPIKESORT="${FORCE_RESTART_SPIKESORT:-}"   # optional 0|1; falls back to FORCE_RESTART
FORCE_RESTART_TEMPLATES="${FORCE_RESTART_TEMPLATES:-}"   # optional 0|1; falls back to FORCE_RESTART
FORCE_RESTART_RECONSTRUCT="${FORCE_RESTART_RECONSTRUCT:-}"   # optional 0|1; falls back to FORCE_RESTART
FORCE_RESTART_ANALYSIS="${FORCE_RESTART_ANALYSIS:-}"   # optional 0|1; falls back to FORCE_RESTART
RUN_SPIKESORT="${RUN_SPIKESORT:-0}"   # 0|1 (default off for downstream-only workflow)
RUN_WAVEFORMS="${RUN_WAVEFORMS:-1}"   # 0|1
RUN_TEMPLATES="${RUN_TEMPLATES:-1}"   # 0|1
RUN_RECONSTRUCT="${RUN_RECONSTRUCT:-1}"   # 0|1
ENABLE_GIFS="${ENABLE_GIFS:-0}"       # 0|1 (default off)
RUN_ANALYSIS="${RUN_ANALYSIS:-0}"     # 0|1
DRY_RUN="${DRY_RUN:-0}"               # 0|1
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
WAVEFORMS_DEBUG_MAX_UNITS="${WAVEFORMS_DEBUG_MAX_UNITS:-}"  # optional int; waveforms supports first N units, not explicit unit_ids
TEMPLATES_INCLUDE_SEGMENTS="${TEMPLATES_INCLUDE_SEGMENTS:-1}"   # 0|1
TEMPLATES_RUN_UNIT_MERGING="${TEMPLATES_RUN_UNIT_MERGING:-1}"   # 0|1

# Optional overrides (otherwise resolved from ENV_FILE)
H5_PATH_OVERRIDE="${H5_PATH_OVERRIDE:-}"
STREAM_ID_OVERRIDE="${STREAM_ID_OVERRIDE:-}"
MEA_OUTPUT_ROOT_OVERRIDE="${MEA_OUTPUT_ROOT_OVERRIDE:-}"
MEA_ANALYSIS_REPO_ROOT_OVERRIDE="${MEA_ANALYSIS_REPO_ROOT_OVERRIDE:-}"

export ENV_FILE
export H5_PATH_OVERRIDE
export STREAM_ID_OVERRIDE
export MEA_OUTPUT_ROOT_OVERRIDE

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE" >&2
  exit 2
fi

unit_id_from_path="$(echo "$UNIT_DIR" | sed -n 's#.*unit_\([0-9][0-9]*\).*#\1#p')"
if [[ -n "$UNIT_ID" ]]; then
  if [[ "$UNIT_DIR" =~ /by_unit/unit_[0-9]+$ ]]; then
    UNIT_DIR="${UNIT_DIR%/unit_*}/unit_${UNIT_ID}"
  elif [[ -n "$UNIT_BASE_DIR" ]]; then
    UNIT_DIR="${UNIT_BASE_DIR%/}/unit_${UNIT_ID}"
  fi
  unit_id_from_path="$(echo "$UNIT_DIR" | sed -n 's#.*unit_\([0-9][0-9]*\).*#\1#p')"
fi

if [[ -z "$unit_id_from_path" ]]; then
  echo "Could not determine unit id. Set UNIT_ID or provide UNIT_DIR ending in /by_unit/unit_<id>." >&2
  exit 2
fi
UNIT_ID="${UNIT_ID:-$unit_id_from_path}"
export UNIT_ID

stream_id_from_path="$(echo "$UNIT_DIR" | sed -n 's#.*\(/well[0-9][0-9][0-9]\)/.*#\1#p' | sed 's#^/##')"
if [[ -n "$STREAM_ID_OVERRIDE" ]]; then
  STREAM_ID="$STREAM_ID_OVERRIDE"
elif [[ -n "$stream_id_from_path" ]]; then
  STREAM_ID="$stream_id_from_path"
else
  STREAM_ID="${STREAM_ID:-well001}"
fi
export STREAM_ID

if [[ "$ENABLE_GIFS" == "1" ]]; then
  export AXON_RECON_RECON_WRITE_TEMPLATE_MOVIE_GIF=1
else
  export AXON_RECON_RECON_WRITE_TEMPLATE_MOVIE_GIF=0
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/axon-recon-single-unit.XXXXXX")"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

TEMPLATES_KWARGS_JSON="$TMP_DIR/templates_kwargs.json"
RECON_KWARGS_JSON="$TMP_DIR/reconstruct_kwargs.json"
WAVEFORMS_KWARGS_JSON="$TMP_DIR/waveforms_kwargs.json"

cat > "$TEMPLATES_KWARGS_JSON" <<EOF
{
  "unit_ids": [${UNIT_ID}],
  "unit_limit": 1,
  "include_concat": true,
  "include_segments": $( [[ "$TEMPLATES_INCLUDE_SEGMENTS" == "1" ]] && echo true || echo false ),
  "require_curated_units": false,
  "run_unit_merging": $( [[ "$TEMPLATES_RUN_UNIT_MERGING" == "1" ]] && echo true || echo false ),
  "plot_templates_grid_pdf": true,
  "plot_multi_source_templates_pdf": false
}
EOF

cat > "$RECON_KWARGS_JSON" <<EOF
{
  "unit_ids": [${UNIT_ID}],
  "unit_limit": 1,
  "write_all_units_overview_pdf": false,
  "unit_workers": 1
}
EOF

if [[ -n "$WAVEFORMS_DEBUG_MAX_UNITS" ]]; then
  cat > "$WAVEFORMS_KWARGS_JSON" <<EOF
{
  "debug_max_units": ${WAVEFORMS_DEBUG_MAX_UNITS}
}
EOF
fi

stage_force_restart() {
  local stage="$1"
  local stage_force="$FORCE_RESTART"

  case "$stage" in
    spikesort)
      if [[ -n "$FORCE_RESTART_SPIKESORT" ]]; then
        stage_force="$FORCE_RESTART_SPIKESORT"
      fi
      ;;
    waveforms)
      if [[ -n "$FORCE_RESTART_WAVEFORMS" ]]; then
        stage_force="$FORCE_RESTART_WAVEFORMS"
      fi
      ;;
    templates)
      if [[ -n "$FORCE_RESTART_TEMPLATES" ]]; then
        stage_force="$FORCE_RESTART_TEMPLATES"
      fi
      ;;
    reconstruct)
      if [[ -n "$FORCE_RESTART_RECONSTRUCT" ]]; then
        stage_force="$FORCE_RESTART_RECONSTRUCT"
      fi
      ;;
    analysis)
      if [[ -n "$FORCE_RESTART_ANALYSIS" ]]; then
        stage_force="$FORCE_RESTART_ANALYSIS"
      fi
      ;;
  esac

  if [[ "$stage_force" == "1" ]]; then
    echo "--force-restart"
  else
    echo "--no-force-restart"
  fi
}

is_stage_force_restart_enabled() {
  local stage="$1"
  local stage_force="$FORCE_RESTART"

  case "$stage" in
    spikesort)
      if [[ -n "$FORCE_RESTART_SPIKESORT" ]]; then
        stage_force="$FORCE_RESTART_SPIKESORT"
      fi
      ;;
    waveforms)
      if [[ -n "$FORCE_RESTART_WAVEFORMS" ]]; then
        stage_force="$FORCE_RESTART_WAVEFORMS"
      fi
      ;;
    templates)
      if [[ -n "$FORCE_RESTART_TEMPLATES" ]]; then
        stage_force="$FORCE_RESTART_TEMPLATES"
      fi
      ;;
    reconstruct)
      if [[ -n "$FORCE_RESTART_RECONSTRUCT" ]]; then
        stage_force="$FORCE_RESTART_RECONSTRUCT"
      fi
      ;;
    analysis)
      if [[ -n "$FORCE_RESTART_ANALYSIS" ]]; then
        stage_force="$FORCE_RESTART_ANALYSIS"
      fi
      ;;
  esac

  [[ "$stage_force" == "1" ]]
}

compute_well_out_dir() {
  python - <<'PY'
import os
from pathlib import Path

from axon_reconstructor import env_utils
from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

env_file = Path(os.environ.get("ENV_FILE", "")).expanduser()
if env_file.exists():
    env_utils.load_env_files_into_os(env_files=[env_file], override_existing=False)

h5_path = os.environ.get("H5_PATH_OVERRIDE") or os.environ.get("AXON_RECON_H5_PATH")
output_root = os.environ.get("MEA_OUTPUT_ROOT_OVERRIDE") or os.environ.get("AXON_RECON_MEA_OUTPUT_ROOT")
stream_id = os.environ.get("STREAM_ID")

if not (h5_path and output_root and stream_id):
    raise SystemExit("Could not resolve well output path inputs (h5/output_root/stream_id)")

well_out_dir = compute_mea_analysis_output_dir(
    output_root=Path(output_root).expanduser(),
    data_file=Path(h5_path).expanduser(),
    well=str(stream_id),
)
print(well_out_dir)
PY
}

cleanup_stage_outputs_if_forced() {
  local stage="$1"
  local well_out_dir="$2"

  if ! is_stage_force_restart_enabled "$stage"; then
    return 0
  fi

  local target=""
  case "$stage" in
    spikesort)
      target="${well_out_dir%/}/stg2_spikesorting_outputs"
      ;;
    waveforms)
      target="${well_out_dir%/}/stg3_waveforms_outputs"
      ;;
    *)
      return 0
      ;;
  esac

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$stage] DRY_RUN would remove previous artifacts at: $target"
    return 0
  fi

  if [[ -e "$target" ]]; then
    echo "[$stage] force-restart cleanup: removing previous artifacts at: $target"
    rm -rf -- "$target"
  else
    echo "[$stage] force-restart cleanup: no previous artifacts found at: $target"
  fi
}

build_common_args() {
  local stage="$1"
  local -a args=(--env-file "$ENV_FILE")

  if [[ -n "$H5_PATH_OVERRIDE" ]]; then
    args+=(--h5-path "$H5_PATH_OVERRIDE")
  fi
  args+=(--stream-id "$STREAM_ID")

  if [[ -n "$MEA_OUTPUT_ROOT_OVERRIDE" ]]; then
    args+=(--mea-output-root "$MEA_OUTPUT_ROOT_OVERRIDE")
  fi

  args+=("$(stage_force_restart "$stage")")

  printf '%q ' "${args[@]}"
  printf '\n'
}

run_stage() {
  local stage="$1"
  local extra="$2"
  local common
  common="$(build_common_args "$stage")"

  local cmd="python -m axon_reconstructor.cli stage ${stage} ${common} ${extra}"
  echo "[$stage] $cmd"

  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  if ! eval "$cmd"; then
    echo "Stage failed: $stage" >&2
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
      exit 1
    fi
  fi
}

cd "$REPO_ROOT"
export PYTHONPATH=src

echo "run_single_unit_reconstruct: env_file=$ENV_FILE"
echo "run_single_unit_reconstruct: unit_dir=$UNIT_DIR"
echo "run_single_unit_reconstruct: unit_id=$UNIT_ID stream_id=$STREAM_ID"
echo "run_single_unit_reconstruct: force_restart_default=$FORCE_RESTART spikesort=${FORCE_RESTART_SPIKESORT:-<default>} waveforms=${FORCE_RESTART_WAVEFORMS:-<default>} templates=${FORCE_RESTART_TEMPLATES:-<default>} reconstruct=${FORCE_RESTART_RECONSTRUCT:-<default>} analysis=${FORCE_RESTART_ANALYSIS:-<default>}"
echo "run_single_unit_reconstruct: run_spikesort=$RUN_SPIKESORT run_waveforms=$RUN_WAVEFORMS run_templates=$RUN_TEMPLATES run_reconstruct=$RUN_RECONSTRUCT"
echo "run_single_unit_reconstruct: templates_include_segments=$TEMPLATES_INCLUDE_SEGMENTS templates_run_unit_merging=$TEMPLATES_RUN_UNIT_MERGING"
echo "run_single_unit_reconstruct: gifs_enabled=$ENABLE_GIFS run_analysis=$RUN_ANALYSIS"
echo "run_single_unit_reconstruct: preprocess is always skipped in this runner"

WELL_OUT_DIR=""
if [[ "$RUN_SPIKESORT" == "1" || "$RUN_WAVEFORMS" == "1" ]]; then
  WELL_OUT_DIR="$(compute_well_out_dir)"
  echo "run_single_unit_reconstruct: resolved well_out_dir=$WELL_OUT_DIR"
fi

if [[ "$RUN_SPIKESORT" == "1" ]]; then
  cleanup_stage_outputs_if_forced "spikesort" "$WELL_OUT_DIR"
  run_stage "spikesort" ""
else
  echo "run_single_unit_reconstruct: skipping spikesort"
fi

waveforms_extra=""
if [[ -n "$WAVEFORMS_DEBUG_MAX_UNITS" ]]; then
  waveforms_extra="--stage-kwargs-file $(printf '%q' "$WAVEFORMS_KWARGS_JSON")"
  echo "run_single_unit_reconstruct: waveforms uses debug_max_units=$WAVEFORMS_DEBUG_MAX_UNITS (first N units, not explicit unit_ids)"
fi
if [[ "$RUN_WAVEFORMS" == "1" ]]; then
  cleanup_stage_outputs_if_forced "waveforms" "$WELL_OUT_DIR"
  run_stage "waveforms" "$waveforms_extra"
else
  echo "run_single_unit_reconstruct: skipping waveforms"
fi

if [[ "$RUN_TEMPLATES" == "1" ]]; then
  run_stage "templates" "--stage-kwargs-file $(printf '%q' "$TEMPLATES_KWARGS_JSON")"
else
  echo "run_single_unit_reconstruct: skipping templates"
fi

if [[ "$RUN_RECONSTRUCT" == "1" ]]; then
  run_stage "reconstruct" "--stage-kwargs-file $(printf '%q' "$RECON_KWARGS_JSON")"
else
  echo "run_single_unit_reconstruct: skipping reconstruct"
fi

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  run_stage "analysis" "--unit-ids ${UNIT_ID} --unit-limit 1"
fi

echo "run_single_unit_reconstruct: done"

# Best-effort expected path under current naming contract.
python - <<'PY'
import os
from pathlib import Path

from axon_reconstructor import env_utils
from axon_reconstructor.pipeline.output_paths import compute_mea_analysis_output_dir

env_file = Path(os.environ.get("ENV_FILE", "")).expanduser()
if env_file.exists():
    env_utils.load_env_files_into_os(env_files=[env_file], override_existing=False)

h5_path = os.environ.get("H5_PATH_OVERRIDE") or os.environ.get("AXON_RECON_H5_PATH")
output_root = os.environ.get("MEA_OUTPUT_ROOT_OVERRIDE") or os.environ.get("AXON_RECON_MEA_OUTPUT_ROOT")
stream_id = os.environ.get("STREAM_ID")
unit_id = os.environ.get("UNIT_ID")

if h5_path and output_root and stream_id and unit_id:
    well_out_dir = compute_mea_analysis_output_dir(
        output_root=Path(output_root).expanduser(),
        data_file=Path(h5_path).expanduser(),
        well=str(stream_id),
    )
    unit_dir = well_out_dir / "stg5_reconstruction_outputs" / "by_unit" / f"unit_{unit_id}"
    print(f"expected_unit_dir={unit_dir}")
PY
