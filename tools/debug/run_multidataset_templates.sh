#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_PATH_DEFAULT="$ROOT_DIR/cross_well_config.yml"
ENV_FILE_DEFAULT="$ROOT_DIR/debug.env"
LOG_DIR_DEFAULT="$ROOT_DIR/logs/multidataset_templates_runs"
MAX_PARALLEL_DEFAULT=3
MAX_CORES_DEFAULT=32

usage() {
  cat <<'USAGE' >&2
Usage:
  run_multidataset_templates.sh [--max-parallel N] [--max-cores N] [cfg_path] [env_file] [log_dir] [-- extra_args...]

Examples:
  ./run_multidataset_templates.sh
  ./run_multidataset_templates.sh --max-parallel 3
  ./run_multidataset_templates.sh --max-parallel 3 --max-cores 32
  ./run_multidataset_templates.sh -- --force-restart
  ./run_multidataset_templates.sh ./cross_well_config.yml ./debug.env ./logs/templates -- --n-jobs 8
USAGE
}

# Parse options + optional positional args + optional "--" passthrough.
MAX_PARALLEL="$MAX_PARALLEL_DEFAULT"
MAX_CORES="$MAX_CORES_DEFAULT"
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

# Estimate per-well n_jobs used by debug_templates_step.py
# Priority: --extra-args --n-jobs > AXON_RECON_N_JOBS in env file > 8 default
PER_WELL_N_JOBS=""
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
  if [[ "${EXTRA_ARGS[$i]}" == "--n-jobs" ]] && [[ $((i+1)) -lt ${#EXTRA_ARGS[@]} ]]; then
    PER_WELL_N_JOBS="${EXTRA_ARGS[$((i+1))]}"
    break
  fi
done
if [[ -z "$PER_WELL_N_JOBS" ]]; then
  PER_WELL_N_JOBS=$(awk -F= '/^AXON_RECON_N_JOBS=/{print $2; exit}' "$ENV_FILE" | tr -d ' ')
fi
if [[ -z "$PER_WELL_N_JOBS" ]]; then
  PER_WELL_N_JOBS="8"
fi
if ! [[ "$PER_WELL_N_JOBS" =~ ^[0-9]+$ ]] || [[ "$PER_WELL_N_JOBS" -lt 1 ]]; then
  PER_WELL_N_JOBS="8"
fi

EST_TOTAL_JOBS=$((MAX_PARALLEL * PER_WELL_N_JOBS))

echo "Starting templates batch for $total targets"
echo "- cfg: $CFG_PATH"
echo "- env: $ENV_FILE"
echo "- logs: $LOG_DIR"
echo "- max parallel wells: $MAX_PARALLEL"
echo "- per-well n_jobs (estimated): $PER_WELL_N_JOBS"
echo "- estimated total jobs: $EST_TOTAL_JOBS (max cores: $MAX_CORES)"
if [[ "$EST_TOTAL_JOBS" -gt "$MAX_CORES" ]]; then
  echo "[WARN] Potential oversubscription: estimated jobs $EST_TOTAL_JOBS > max cores $MAX_CORES" >&2
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

  echo "[$idx/$total] Templates $well_id :: $h5_path"

  set +e
  cmd=(
    conda run --no-capture-output -n axon_recon
    python "$ROOT_DIR/debug_steps.py"
    --env-file "$ENV_FILE"
    --h5-path "$h5_path"
    --stream-id "$well_id"
    --steps debug_templates_step.py
  )
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=(--extra-args "${EXTRA_ARGS[@]}")
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

echo "Done templates batch in ${dt}s. ok=$ok fail=$fail"
if [[ $fail -gt 0 ]]; then
  echo "Failures written to: $FAILURES_TSV"
  echo "(columns: exit_code, h5_path, well_id, log_file)"
  exit 1
fi
