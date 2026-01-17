#!/usr/bin/env bash
set -euo pipefail

# Run axon_reconstructor using the MEA_Analysis outputs staged into OUT_ROOT.
# This does NOT run spikesorting itself; it only loads sorter_output via the MEA path contract.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/00_config.sh"

H5_PARENT_DIR="$(dirname "$RAW_H5")"

cd "$AXON_REPO"

CMD=(
  PYTHONPATH=src
  python3 -m axon_reconstructor.cli run "$H5_PARENT_DIR"
  --mea-environment nersc
  --mea-output-root "$OUT_ROOT"
  --mea-analysis-repo-root "$MEA_REPO"
)

echo "Running axon_reconstructor against MEA outputs:"
echo "  ${CMD[*]}"

"${CMD[@]}"
