#!/usr/bin/env bash
set -euo pipefail

# Run all login-node smoke tests (no allocations required).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck disable=SC1091
source "$SMOKE_ROOT/_shared/00_config.sh"

if [[ -z "${RAW_H5:-}" ]]; then
	echo "ERROR: RAW_H5 is not set. Set RAW_H5 or configure tools/smoke_tests/perlmutter/smoke_tests.local.toml." >&2
	exit 2
fi

echo "[login smoke] start host=$(hostname) time=$(date -Is)"

bash "$SCRIPT_DIR/05_login_smoketest_no_sort.sh"
bash "$SCRIPT_DIR/06_login_smoketest_force_restart.sh"

echo "[login smoke] done"