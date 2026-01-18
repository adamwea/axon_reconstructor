#!/usr/bin/env bash
set -euo pipefail

# Run all GPU-node smoke tests from a login node by requesting ONE interactive GPU allocation,
# then running each smoke script inside that allocation.
#
# Uses allocation defaults from ../_shared/00_config.sh (override via env vars).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$SMOKE_ROOT/_shared/00_config.sh"

if [[ -z "${GPU_SMOKE_SALLOC_ACCOUNT:-}" ]]; then
  echo "ERROR: GPU_SMOKE_SALLOC_ACCOUNT is empty. Set it in config or export it before running." >&2
  exit 2
fi

repo_root="$AXON_REPO"

# Optional extra args (space-delimited).
extra_args=()
if [[ -n "${GPU_SMOKE_SALLOC_EXTRA_ARGS:-}" ]]; then
  # Intentionally allow word-splitting here.
  # shellcheck disable=SC2206
  extra_args=( ${GPU_SMOKE_SALLOC_EXTRA_ARGS} )
fi

salloc_cmd=(
  salloc
  -A "$GPU_SMOKE_SALLOC_ACCOUNT"
  -q "$GPU_SMOKE_SALLOC_QOS"
  -C "$GPU_SMOKE_SALLOC_CONSTRAINT"
  -t "$GPU_SMOKE_SALLOC_TIME"
  -N "$GPU_SMOKE_SALLOC_NODES"
  --gpus="$GPU_SMOKE_SALLOC_GPUS"
  --cpus-per-task="$GPU_SMOKE_SALLOC_CPUS_PER_TASK"
  "${extra_args[@]}"
)

suite_cmd=(
  bash -lc
  "set -euo pipefail; \
   cd \"$repo_root\"; \
   echo '[gpu smoke suite] SLURM_JOB_ID='\"\${SLURM_JOB_ID:-unset}\"' host='\"\$(hostname)\"; \
   bash scripts/nersc_perlmutter/smoke_tests/interactive_gpu_node/08_gpu_require_gpu_dry_check.sh; \
   bash scripts/nersc_perlmutter/smoke_tests/interactive_gpu_node/11_shifter_image_inventory.sh; \
   bash scripts/nersc_perlmutter/smoke_tests/interactive_gpu_node/09_gpu_readiness_check.sh; \
   bash scripts/nersc_perlmutter/smoke_tests/interactive_gpu_node/07_gpu_node_smoketest_no_sort_container_plugin_default.sh; \
   echo '[gpu smoke suite] done'"
)

echo "Requesting interactive GPU allocation:" >&2
printf '  %q' "${salloc_cmd[@]}" >&2
echo >&2

"${salloc_cmd[@]}" "${suite_cmd[@]}"