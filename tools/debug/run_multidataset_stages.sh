#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/adamm/dev/projects/260121_debugging_individual_axon_recon_steps"
CFG_PATH="$ROOT_DIR/cross_well_config.yml"
ENV_FILE="$ROOT_DIR/debug.env"
LOG_DIR="$ROOT_DIR/logs/multidataset_stage_runs"
mkdir -p "$LOG_DIR"

cd "$ROOT_DIR"

python - <<'PY' > "$LOG_DIR/targets.tsv"
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("/home/adamm/dev/projects/260121_debugging_individual_axon_recon_steps/cross_well_config.yml").read_text())
for d in cfg.get("datasets", []):
    h5 = d["raw_data_h5_path"]
    for w in d.get("wells", []):
        print(f"{h5}\t{w['well_id']}")
PY

total=$(wc -l < "$LOG_DIR/targets.tsv" | tr -d ' ')
echo "Starting stage batch for $total targets"

idx=0
while IFS=$'\t' read -r h5_path well_id; do
  idx=$((idx+1))
  tag="$(basename "$(dirname "$h5_path")")__${well_id}"
  log_file="$LOG_DIR/${idx}_of_${total}__${tag}.log"

  echo "[$idx/$total] Running stages for $well_id :: $h5_path"

  conda run --no-capture-output -n axon_recon \
    python "$ROOT_DIR/debug_steps.py" \
      --env-file "$ENV_FILE" \
      --h5-path "$h5_path" \
      --stream-id "$well_id" \
      --steps \
        debug_preprocessing_step.py \
        debug_spikesorting_step.py \
        debug_waveforms_step.py \
        debug_templates_step.py \
        debug_reconstruction_step.py \
    |& tee "$log_file"

done < "$LOG_DIR/targets.tsv"

echo "All targets completed. Logs: $LOG_DIR"
