#!/bin/bash
# Submit a dependent spikesort -> reconstruct sbatch chain on Perlmutter.
#
# Flow:
#   1. Scan dev/debug_NERSC/debug.data.yml for datasets that have at least
#      one included well missing the spikesort completion marker
#      (<well>/spikesort_outputs/merge_SLAy/run-output.json).
#   2. If any incomplete datasets found: submit perlmutter_spikesort.sbatch
#      targeting just those indices, on 4 GPU nodes (regular QoS, 4h).
#   3. Submit perlmutter_reconstruct.sbatch on 4 CPU nodes (regular QoS, 4h)
#      targeting the reconstruct dataset list (default: 0-8, "all except the
#      last 4 datasets").
#   4. Reconstruct depends on spikesort completing with exit 0 (--dependency=
#      afterok). If no incomplete spikesort, reconstruct goes in with no
#      dependency and runs as soon as the CPU queue picks it up.
#
# Override the reconstruct target list via env:
#   RECONSTRUCT_TARGETS="0,1,2,3,4,5,6,7,8" examples/perlmutter_pipeline_chain.sh
#
# Override the runtime config via env:
#   RUNTIME_CFG=dev/debug_NERSC/debug.runtime.yml examples/perlmutter_pipeline_chain.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_CFG="${RUNTIME_CFG:-dev/debug_NERSC/debug.runtime.yml}"
RECONSTRUCT_TARGETS="${RECONSTRUCT_TARGETS:-0,1,2,3,4,5,6,7,8}"

cd "$REPO_ROOT"

# Step 1: detect datasets where at least one well is missing the spikesort
# completion marker. Empty output = all datasets fully spikesorted.
echo "scanning for incomplete spikesort (this reads ${RUNTIME_CFG} + its data yml)..."
INCOMPLETE="$(python3 examples/detect_incomplete_spikesort.py "$RUNTIME_CFG")"
if [[ -n "${INCOMPLETE}" ]]; then
    echo "incomplete spikesort datasets: ${INCOMPLETE}"
else
    echo "all included datasets are fully spikesorted; skipping spikesort sbatch"
fi
echo

# Step 2: submit spikesort sbatch if any incomplete
SPIKESORT_JOB=""
JOB_DEPS=()
if [[ -n "${INCOMPLETE}" ]]; then
    SPIKESORT_JOB="$(sbatch --parsable examples/perlmutter_spikesort.sbatch "${INCOMPLETE}")"
    echo "submitted spikesort job ${SPIKESORT_JOB} (datasets ${INCOMPLETE})"
    JOB_DEPS=(--dependency="afterok:${SPIKESORT_JOB}")
fi

# Step 3: submit reconstruct sbatch (with afterok dependency if spikesort was submitted)
RECONSTRUCT_JOB="$(sbatch --parsable "${JOB_DEPS[@]}" examples/perlmutter_reconstruct.sbatch "${RECONSTRUCT_TARGETS}")"
echo "submitted reconstruct job ${RECONSTRUCT_JOB} (datasets ${RECONSTRUCT_TARGETS})"

# Step 4: summary
echo
echo "=== Pipeline chain submitted ==="
if [[ -n "${SPIKESORT_JOB}" ]]; then
    echo "  spikesort:    ${SPIKESORT_JOB}   (datasets ${INCOMPLETE})"
    echo "  reconstruct:  ${RECONSTRUCT_JOB}   (datasets ${RECONSTRUCT_TARGETS}, depends on ${SPIKESORT_JOB})"
else
    echo "  reconstruct:  ${RECONSTRUCT_JOB}   (datasets ${RECONSTRUCT_TARGETS}, no dependencies)"
fi
echo
echo "Monitor with: squeue -u \$USER -o '%.10i %.9P %.2t %.10M %.10L %.20R'"
