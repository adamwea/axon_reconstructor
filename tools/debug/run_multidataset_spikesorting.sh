#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_PATH_DEFAULT="$ROOT_DIR/cross_well_config.yml"
ENV_FILE_DEFAULT="$ROOT_DIR/debug.env"
LOG_DIR_DEFAULT="$ROOT_DIR/logs/multidataset_spikesorting_runs"

usage() {
  cat <<'USAGE' >&2
Usage:
  run_multidataset_spikesorting.sh [cfg_path] [env_file] [log_dir] [-- extra_args...]

Examples:
  ./run_multidataset_spikesorting.sh
  ./run_multidataset_spikesorting.sh -- --force-restart
  ./run_multidataset_spikesorting.sh ./cross_well_config.yml ./debug.env ./logs/sort -- --n-jobs 8
USAGE
}

# Parse optional positional args + optional "--" passthrough.
args=("$@")
sep=-1
for i in "${!args[@]}"; do
  if [[ "${args[$i]}" == "--" ]]; then
    sep=$i
    break
  fi
done

pos_args=()
EXTRA_ARGS=()
if [[ $sep -ge 0 ]]; then
  pos_args=("${args[@]:0:$sep}")
  EXTRA_ARGS=("${args[@]:$((sep+1))}")
else
  pos_args=("${args[@]}")
fi

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
echo "Starting spikesorting batch for $total targets"
echo "- cfg: $CFG_PATH"
echo "- env: $ENV_FILE"
echo "- logs: $LOG_DIR"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "- extra args: ${EXTRA_ARGS[*]}"
fi

t_start=$(date +%s)
idx=0
ok=0
fail=0

while IFS=$'\t' read -r h5_path well_id; do
  idx=$((idx+1))
  tag="$(basename "$(dirname "$h5_path")")__${well_id}"
  log_file="$LOG_DIR/${idx}_of_${total}__${tag}.log"

  echo "[$idx/$total] Spikesorting $well_id :: $h5_path"

  set +e
  cmd=(
    conda run --no-capture-output -n axon_recon
    python "$ROOT_DIR/debug_steps.py"
    --env-file "$ENV_FILE"
    --h5-path "$h5_path"
    --stream-id "$well_id"
    --steps debug_spikesorting_step.py
  )
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=(--extra-args "${EXTRA_ARGS[@]}")
  fi
  "${cmd[@]}" |& tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ $exit_code -eq 0 ]]; then
    ok=$((ok+1))
  else
    fail=$((fail+1))
    printf "%s\t%s\t%s\t%s\n" "$exit_code" "$h5_path" "$well_id" "$log_file" >> "$FAILURES_TSV"
    echo "  -> FAILED (exit $exit_code). Logged: $log_file" >&2
  fi

done < "$TARGETS_TSV"

t_end=$(date +%s)
dt=$((t_end - t_start))

echo "Done spikesorting batch in ${dt}s. ok=$ok fail=$fail"
if [[ $fail -gt 0 ]]; then
  echo "Failures written to: $FAILURES_TSV"
  echo "(columns: exit_code, h5_path, well_id, log_file)"
  exit 1
fi
