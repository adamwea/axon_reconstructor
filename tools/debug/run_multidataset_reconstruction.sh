#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_PATH_DEFAULT="$ROOT_DIR/cross_well_config.yml"
ENV_FILE_DEFAULT="$ROOT_DIR/debug.env"
LOG_DIR_DEFAULT="$ROOT_DIR/logs/multidataset_reconstruction_runs"
MAX_PARALLEL_DEFAULT=3
MAX_CORES_DEFAULT=32

usage() {
  cat <<'USAGE' >&2
Usage:
  run_multidataset_reconstruction.sh [--max-parallel N] [--max-cores N] [--unit-workers N] [--json-only] [cfg_path] [env_file] [log_dir] [-- extra_args...]

Examples:
  ./run_multidataset_reconstruction.sh
  ./run_multidataset_reconstruction.sh --max-parallel 3
  ./run_multidataset_reconstruction.sh --max-parallel 3 --max-cores 32
  ./run_multidataset_reconstruction.sh --max-parallel 2 --unit-workers 2
  ./run_multidataset_reconstruction.sh --max-parallel 2 --unit-workers 2 --json-only -- --force-restart
  ./run_multidataset_reconstruction.sh -- --force-restart
USAGE
}

# Parse options + optional positional args + optional "--" passthrough.
MAX_PARALLEL="$MAX_PARALLEL_DEFAULT"
MAX_CORES="$MAX_CORES_DEFAULT"
UNIT_WORKERS=""
JSON_ONLY=0
pos_args=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      EXTRA_ARGS=("$@")
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
    --unit-workers)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --unit-workers" >&2
        usage
        exit 2
      fi
      UNIT_WORKERS="$2"
      shift 2
      ;;
    --json-only)
      JSON_ONLY=1
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
LOG_DIR="$LOG_DIR_DEFAULT"

if [[ ${#pos_args[@]} -ge 1 ]] && [[ -n "${pos_args[0]}" ]]; then
  CFG_PATH="${pos_args[0]}"
fi
if [[ ${#pos_args[@]} -ge 2 ]] && [[ -n "${pos_args[1]}" ]]; then
  ENV_FILE="${pos_args[1]}"
fi
if [[ ${#pos_args[@]} -ge 3 ]] && [[ -n "${pos_args[2]}" ]]; then
  LOG_DIR="${pos_args[2]}"
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

if [[ -n "$UNIT_WORKERS" ]] && { ! [[ "$UNIT_WORKERS" =~ ^[0-9]+$ ]] || [[ "$UNIT_WORKERS" -lt 1 ]]; }; then
  echo "--unit-workers must be a positive integer (got: $UNIT_WORKERS)" >&2
  usage
  exit 2
fi

mkdir -p "$LOG_DIR"
cd "$ROOT_DIR"

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

TARGETS_TSV="$LOG_DIR/targets.tsv"
FAILURES_TSV="$LOG_DIR/failures.tsv"
: > "$FAILURES_TSV"

python - <<PY > "$TARGETS_TSV"
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("$CFG_PATH").read_text())
for d in cfg.get("datasets", []):
    h5 = d["raw_data_h5_path"]
    for w in d.get("wells", []):
        print(f"{h5}\t{w['well_id']}")
PY

total=$(wc -l < "$TARGETS_TSV" | tr -d ' ')

# Estimate per-well workers used by reconstruction.
# Priority: --unit-workers > --extra-args --unit-workers > AXON_RECON_RECON_UNIT_WORKERS in env file > 1 default
PER_WELL_JOBS=""
if [[ -n "$UNIT_WORKERS" ]]; then
  PER_WELL_JOBS="$UNIT_WORKERS"
fi
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
  if [[ "${EXTRA_ARGS[$i]}" == "--unit-workers" ]] && [[ $((i+1)) -lt ${#EXTRA_ARGS[@]} ]]; then
    PER_WELL_JOBS="${EXTRA_ARGS[$((i+1))]}"
    break
  fi
done
if [[ -z "$PER_WELL_JOBS" ]]; then
  PER_WELL_JOBS=$(awk -F= '/^AXON_RECON_RECON_UNIT_WORKERS=/{print $2; exit}' "$ENV_FILE" | tr -d ' ')
fi
if [[ -z "$PER_WELL_JOBS" ]]; then
  PER_WELL_JOBS="1"
fi
if ! [[ "$PER_WELL_JOBS" =~ ^[0-9]+$ ]] || [[ "$PER_WELL_JOBS" -lt 1 ]]; then
  PER_WELL_JOBS="1"
fi

EST_TOTAL_JOBS=$((MAX_PARALLEL * PER_WELL_JOBS))

echo "Starting reconstruction batch for $total targets"
echo "- cfg: $CFG_PATH"
echo "- env: $ENV_FILE"
echo "- logs: $LOG_DIR"
echo "- max parallel wells: $MAX_PARALLEL"
echo "- per-well unit workers (estimated): $PER_WELL_JOBS"
echo "- estimated total jobs: $EST_TOTAL_JOBS (max cores: $MAX_CORES)"
if [[ "$EST_TOTAL_JOBS" -gt "$MAX_CORES" ]]; then
  echo "[WARN] Potential oversubscription: estimated jobs $EST_TOTAL_JOBS > max cores $MAX_CORES" >&2
fi
if [[ -n "$UNIT_WORKERS" ]]; then
  echo "- forced unit workers: $UNIT_WORKERS"
fi
if [[ "$JSON_ONLY" -eq 1 ]]; then
  echo "- json-only mode: enabled (reconstruction plots disabled)"
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "- extra args: ${EXTRA_ARGS[*]}"
fi

t_start=$(date +%s)
idx=0

run_one() {
  local idx="$1"
  local h5_path="$2"
  local well_id="$3"
  local tag
  local log_file
  local status_file
  local exit_code

  tag="$(basename "$(dirname "$h5_path")")__${well_id}"
  log_file="$LOG_DIR/${idx}_of_${total}__${tag}.log"
  status_file="$LOG_DIR/${idx}_of_${total}__${tag}.status.tsv"

  echo "[$idx/$total] Reconstruction $well_id :: $h5_path"

  set +e
  cmd=(
    conda run --no-capture-output -n axon_recon
    python "$ROOT_DIR/debug_steps.py"
    --env-file "$ENV_FILE"
    --h5-path "$h5_path"
    --stream-id "$well_id"
    --steps debug_reconstruction_step.py
  )
  stage_args=()
  if [[ -n "$UNIT_WORKERS" ]]; then
    stage_args+=(--unit-workers "$UNIT_WORKERS")
  fi
  if [[ "$JSON_ONLY" -eq 1 ]]; then
    stage_args+=(--no-write-unit-pdfs --no-write-all-units-overview-pdf)
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    stage_args+=("${EXTRA_ARGS[@]}")
  fi
  if [[ ${#stage_args[@]} -gt 0 ]]; then
    cmd+=(--extra-args "${stage_args[@]}")
  fi
  "${cmd[@]}" |& tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  set -e

  printf "%s\t%s\t%s\t%s\n" "$exit_code" "$h5_path" "$well_id" "$log_file" > "$status_file"
  if [[ "$exit_code" -ne 0 ]]; then
    echo "  -> FAILED (exit $exit_code). Logged: $log_file" >&2
  fi
  return 0
}

while IFS=$'\t' read -r h5_path well_id; do
  idx=$((idx+1))
  run_one "$idx" "$h5_path" "$well_id" &

  while [[ "$(jobs -pr | wc -l | tr -d ' ')" -ge "$MAX_PARALLEL" ]]; do
    wait -n || true
  done
done < "$TARGETS_TSV"

wait || true

ok=0
fail=0
while IFS=$'\t' read -r exit_code h5_path well_id log_file; do
  if [[ "$exit_code" -eq 0 ]]; then
    ok=$((ok+1))
  else
    fail=$((fail+1))
    printf "%s\t%s\t%s\t%s\n" "$exit_code" "$h5_path" "$well_id" "$log_file" >> "$FAILURES_TSV"
  fi
done < <(cat "$LOG_DIR"/*.status.tsv)

t_end=$(date +%s)
dt=$((t_end - t_start))

echo "Done reconstruction batch in ${dt}s. ok=$ok fail=$fail"
if [[ $fail -gt 0 ]]; then
  echo "Failures written to: $FAILURES_TSV"
  echo "(columns: exit_code, h5_path, well_id, log_file)"
  exit 1
fi
