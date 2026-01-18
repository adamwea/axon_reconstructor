#!/usr/bin/env bash
set -euo pipefail

# Run all login-node smoke tests (no allocations required).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[login smoke] start host=$(hostname) time=$(date -Is)"

bash "$SCRIPT_DIR/05_login_smoketest_no_sort.sh"
bash "$SCRIPT_DIR/06_login_smoketest_force_restart.sh"

echo "[login smoke] done"