#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_PATH_DEFAULT="$ROOT_DIR/cross_well_config.yml"
ENV_FILE_DEFAULT="$ROOT_DIR/debug.env"
LOG_ROOT_DEFAULT="$ROOT_DIR/logs/multidataset_waveforms_templates_reconstruction_runs"
MAX_PARALLEL_DEFAULT=3
MAX_CORES_DEFAULT=32

usage() {
  cat <<'USAGE' >&2
Usage:
  run_multidataset_waveforms_templates_reconstruction.sh [--max-parallel N] [--max-cores N] [--recon-unit-workers N] [--recon-json-only] [cfg_path] [env_file] [log_root] [-- extra_args...]

Examples:
  ./run_multidataset_waveforms_templates_reconstruction.sh
  ./run_multidataset_waveforms_templates_reconstruction.sh --max-parallel 3 --max-cores 32
  ./run_multidataset_waveforms_templates_reconstruction.sh --max-parallel 2 --recon-unit-workers 2
  ./run_multidataset_waveforms_templates_reconstruction.sh --max-parallel 2 --recon-unit-workers 2 --recon-json-only --force-restart
  ./run_multidataset_waveforms_templates_reconstruction.sh --max-parallel 3 --max-cores 32 --force-restart
  ./run_multidataset_waveforms_templates_reconstruction.sh --max-parallel 9 --max-cores 32 -- --force-restart
  ./run_multidataset_waveforms_templates_reconstruction.sh ./cross_well_config.yml ./debug.env ./logs/overnight_postsort
USAGE
}

MAX_PARALLEL="$MAX_PARALLEL_DEFAULT"
MAX_CORES="$MAX_CORES_DEFAULT"
RECON_UNIT_WORKERS=""
RECON_JSON_ONLY=0
pos_args=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    --max-parallel)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --max-parallel" >&2
        usage
        exit 2
      fi
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --max-cores)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --max-cores" >&2
        usage
        exit 2
      fi
      MAX_CORES="$2"
      shift 2
      ;;
    --recon-unit-workers)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --recon-unit-workers" >&2
        usage
        exit 2
      fi
      RECON_UNIT_WORKERS="$2"
      shift 2
      ;;
    --unit-workers)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --unit-workers" >&2
        usage
        exit 2
      fi
      RECON_UNIT_WORKERS="$2"
      shift 2
      ;;
    --recon-json-only)
      RECON_JSON_ONLY=1
      shift
      ;;
    --force-restart)
      EXTRA_ARGS+=("--force-restart")
      shift
      ;;
    *)
      pos_args+=("$1")
      shift
      ;;
  esac
done

CFG_PATH="$CFG_PATH_DEFAULT"
ENV_FILE="$ENV_FILE_DEFAULT"
LOG_ROOT="$LOG_ROOT_DEFAULT"

if [[ ${#pos_args[@]} -ge 1 ]] && [[ -n "${pos_args[0]}" ]]; then
  CFG_PATH="${pos_args[0]}"
fi
if [[ ${#pos_args[@]} -ge 2 ]] && [[ -n "${pos_args[1]}" ]]; then
  ENV_FILE="${pos_args[1]}"
fi
if [[ ${#pos_args[@]} -ge 3 ]] && [[ -n "${pos_args[2]}" ]]; then
  LOG_ROOT="${pos_args[2]}"
fi
if [[ ${#pos_args[@]} -gt 3 ]]; then
  echo "Too many positional args." >&2
  usage
  exit 2
fi

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "--max-parallel must be a positive integer (got: $MAX_PARALLEL)" >&2
  usage
  exit 2
fi

if ! [[ "$MAX_CORES" =~ ^[0-9]+$ ]] || [[ "$MAX_CORES" -lt 1 ]]; then
  echo "--max-cores must be a positive integer (got: $MAX_CORES)" >&2
  usage
  exit 2
fi

if [[ -n "$RECON_UNIT_WORKERS" ]] && { ! [[ "$RECON_UNIT_WORKERS" =~ ^[0-9]+$ ]] || [[ "$RECON_UNIT_WORKERS" -lt 1 ]]; }; then
  echo "--recon-unit-workers must be a positive integer (got: $RECON_UNIT_WORKERS)" >&2
  usage
  exit 2
fi

if [[ ! -f "$CFG_PATH" ]]; then
  echo "Config not found: $CFG_PATH" >&2
  usage
  exit 2
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE" >&2
  usage
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_ROOT/$STAMP"
mkdir -p "$RUN_DIR"

echo "Starting overnight multidataset post-sort run"
echo "- cfg: $CFG_PATH"
echo "- env: $ENV_FILE"
echo "- max parallel wells: $MAX_PARALLEL"
echo "- max cores: $MAX_CORES"
if [[ -n "$RECON_UNIT_WORKERS" ]]; then
  echo "- reconstruction unit workers: $RECON_UNIT_WORKERS"
fi
if [[ "$RECON_JSON_ONLY" -eq 1 ]]; then
  echo "- reconstruction json-only mode: enabled"
fi

EST_RECON_UNIT_WORKERS="$RECON_UNIT_WORKERS"
if [[ -z "$EST_RECON_UNIT_WORKERS" ]]; then
  EST_RECON_UNIT_WORKERS=$(awk -F= '/^AXON_RECON_RECON_UNIT_WORKERS=/{print $2; exit}' "$ENV_FILE" | tr -d ' ')
fi
if [[ -z "$EST_RECON_UNIT_WORKERS" ]]; then
  EST_RECON_UNIT_WORKERS="1"
fi
if ! [[ "$EST_RECON_UNIT_WORKERS" =~ ^[0-9]+$ ]] || [[ "$EST_RECON_UNIT_WORKERS" -lt 1 ]]; then
  EST_RECON_UNIT_WORKERS="1"
fi
EST_RECON_TOTAL=$((MAX_PARALLEL * EST_RECON_UNIT_WORKERS))
echo "- estimated reconstruction workers: $EST_RECON_TOTAL (${MAX_PARALLEL} wells x ${EST_RECON_UNIT_WORKERS} unit workers)"
if [[ "$EST_RECON_TOTAL" -gt 8 ]]; then
  echo "[WARN] High reconstruction concurrency for WSL; this may OOM. Consider --max-parallel 2 --recon-unit-workers 2" >&2
fi
echo "- run dir: $RUN_DIR"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "- stage extra args: ${EXTRA_ARGS[*]}"
fi

t_start=$(date +%s)

# echo "[1/3] Waveforms"
# bash "$ROOT_DIR/run_multidataset_waveforms.sh" \
#   --max-parallel "$MAX_PARALLEL" \
#   --max-cores "$MAX_CORES" \
#   "$CFG_PATH" "$ENV_FILE" "$RUN_DIR/waveforms" \
#   -- "${EXTRA_ARGS[@]}"

# echo "[2/3] Templates"
# bash "$ROOT_DIR/run_multidataset_templates.sh" \
#   --max-parallel "$MAX_PARALLEL" \
#   --max-cores "$MAX_CORES" \
#   "$CFG_PATH" "$ENV_FILE" "$RUN_DIR/templates" \
#   -- "${EXTRA_ARGS[@]}"

echo "[3/3] Reconstruction"
echo "- reconstruction override: AXON_RECON_RECON_WRITE_TEMPLATE_MOVIE_GIF=0"
recon_opts=()
if [[ -n "$RECON_UNIT_WORKERS" ]]; then
  recon_opts+=(--unit-workers "$RECON_UNIT_WORKERS")
fi
if [[ "$RECON_JSON_ONLY" -eq 1 ]]; then
  recon_opts+=(--json-only)
fi
AXON_RECON_RECON_WRITE_TEMPLATE_MOVIE_GIF=0 bash "$ROOT_DIR/run_multidataset_reconstruction.sh" \
  --max-parallel "$MAX_PARALLEL" \
  --max-cores "$MAX_CORES" \
  "${recon_opts[@]}" \
  "$CFG_PATH" "$ENV_FILE" "$RUN_DIR/reconstruction" \
  -- "${EXTRA_ARGS[@]}"

t_end=$(date +%s)
dt=$((t_end - t_start))
echo "All stages finished in ${dt}s"
echo "Logs: $RUN_DIR"
