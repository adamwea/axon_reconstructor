#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH=src

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MISSING_ENV_FILE="$TMP_DIR/does_not_exist.env"

# 1) Canonical help surfaces must be present.
python -m axon_reconstructor.cli stage --help > "$TMP_DIR/stage_help.txt"
python -m axon_reconstructor.cli analysis-deck --help > "$TMP_DIR/analysis_deck_help.txt"
python -m axon_reconstructor.cli scope-run --help > "$TMP_DIR/scope_run_help.txt"

grep -q "usage: axon-reconstructor stage" "$TMP_DIR/stage_help.txt"
grep -q "usage: axon-reconstructor analysis-deck" "$TMP_DIR/analysis_deck_help.txt"
grep -q "usage: axon-reconstructor scope-run" "$TMP_DIR/scope_run_help.txt"

# 2) --env-file path should be accepted on canonical stage and fail deterministically when missing.
python -m axon_reconstructor.cli stage preprocess --env-file "$MISSING_ENV_FILE" > "$TMP_DIR/pre_stage_stdout.txt" 2> "$TMP_DIR/pre_stage_stderr.txt" || true
cat "$TMP_DIR/pre_stage_stdout.txt" "$TMP_DIR/pre_stage_stderr.txt" | grep -q "Env file not found"

# 3) Retired debug alias should fail fast as an invalid command.
python -m axon_reconstructor.cli debug-preprocess > "$TMP_DIR/debug_pre_stdout.txt" 2> "$TMP_DIR/debug_pre_stderr.txt" || true
cat "$TMP_DIR/debug_pre_stdout.txt" "$TMP_DIR/debug_pre_stderr.txt" | grep -qi "invalid choice"

echo "CLI parity checks passed"
