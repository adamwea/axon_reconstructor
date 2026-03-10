#!/usr/bin/env bash
set -euo pipefail

# Runs axon_reconstructor stage=spikesort with UnitMatch dry-run enabled.
# Intended for resume behavior on previously processed data so sorting/analyzer
# stages should return quickly and emit UnitMatch candidate artifacts.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/debug.env}"

# Override stream to the dataset we were debugging previously.
STREAM_ID_OVERRIDE="${STREAM_ID_OVERRIDE:-well001}"

# Use the project env by default; override with PYTHON_CMD if desired.
PYTHON_CMD="${PYTHON_CMD:-conda run -n axon_recon python}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file not found: $ENV_FILE" >&2
  exit 2
fi

STAGE_KWARGS_FILE="$(mktemp /tmp/axon_recon_spikesort_unitmatch_dryrun.XXXXXX.json)"
trap 'rm -f "$STAGE_KWARGS_FILE"' EXIT

cat > "$STAGE_KWARGS_FILE" <<'JSON'
{
  "force_restart": false,
  "force_merge_on_resume": true,
  "unitmatch_merge_units": true,
  "unitmatch_dry_run": true,
  "auto_merge_units": false
}
JSON

cd "$REPO_ROOT"
export PYTHONPATH=src

echo "run_spikesort_unitmatch_dryrun: env_file=$ENV_FILE"
echo "run_spikesort_unitmatch_dryrun: stream_id=$STREAM_ID_OVERRIDE"
echo "run_spikesort_unitmatch_dryrun: stage_kwargs_file=$STAGE_KWARGS_FILE"

CMD="$PYTHON_CMD -m axon_reconstructor.cli stage spikesort --env-file '$ENV_FILE' --stream-id '$STREAM_ID_OVERRIDE' --stage-kwargs-file '$STAGE_KWARGS_FILE'"

echo "$CMD"
eval "$CMD"

echo "done: spikesort stage completed"
echo "check outputs under .../stg2_spikesorting_outputs/unitmatch/"
