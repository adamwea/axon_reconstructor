#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Bootstrap sibling editable dependencies for axon_reconstructor.

This script clones missing sibling repositories into the parent directory of the
axon_reconstructor checkout and installs them into the selected Python
environment in editable mode with --no-deps.

Usage:
  bash tools/bootstrap_editable_deps.sh [--deps-root DIR] [--python PYTHON] [--dry-run]

Options:
  --deps-root DIR  Parent directory that will contain sibling checkouts.
                   Defaults to the parent of the axon_reconstructor repo.
  --python PYTHON  Python executable to use for pip installs.
                   Defaults to $CONDA_PREFIX/bin/python when available, else python.
  --dry-run        Print planned git/pip commands without executing them.
  -h, --help       Show this help message.
EOF
}

log() {
  printf '[bootstrap] %s\n' "$*"
}

die() {
  printf '[bootstrap] ERROR: %s\n' "$*" >&2
  exit 1
}

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DEPS_ROOT=$(cd "$REPO_ROOT/.." && pwd)
DRY_RUN=0

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="python"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deps-root)
      [[ $# -ge 2 ]] || die "--deps-root requires a value"
      DEPS_ROOT="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

if [[ "$PYTHON_BIN" != */* ]]; then
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"
  PYTHON_BIN=$(command -v "$PYTHON_BIN")
fi

DEPS_ROOT=$(cd "$DEPS_ROOT" && pwd)

ensure_checkout() {
  local name="$1"
  local url="$2"
  local branch="$3"
  local repo_dir="$4"

  if [[ -e "$repo_dir" && ! -d "$repo_dir/.git" ]]; then
    die "$name path exists but is not a git checkout: $repo_dir"
  fi

  if [[ ! -d "$repo_dir/.git" ]]; then
    log "Cloning $name ($branch) into $repo_dir"
    run git clone --branch "$branch" --single-branch "$url" "$repo_dir"
    return
  fi

  local current_branch=""
  local current_url=""
  current_branch=$(git -C "$repo_dir" branch --show-current 2>/dev/null || true)
  current_url=$(git -C "$repo_dir" remote get-url origin 2>/dev/null || true)

  if [[ -n "$current_url" && "$current_url" != "$url" ]]; then
    log "Warning: $name origin is $current_url (expected $url); using the local checkout as-is"
  fi

  if [[ "$current_branch" == "$branch" ]]; then
    log "Using existing $name checkout on branch $branch"
    return
  fi

  if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
    die "$name checkout has local changes and is on branch ${current_branch:-<detached>}; switch it to $branch manually first"
  fi

  log "Switching $name checkout to branch $branch"
  run git -C "$repo_dir" fetch origin "$branch"
  if git -C "$repo_dir" rev-parse --verify "$branch" >/dev/null 2>&1; then
    run git -C "$repo_dir" checkout "$branch"
  else
    run git -C "$repo_dir" checkout -b "$branch" "origin/$branch"
  fi
}

install_editable() {
  local name="$1"
  local repo_dir="$2"

  [[ -d "$repo_dir" ]] || die "$name checkout is missing: $repo_dir"

  log "Installing $name in editable mode from $repo_dir"
  run "$PYTHON_BIN" -m pip install --no-deps -e "$repo_dir"
}

log "Repo root: $REPO_ROOT"
log "Deps root: $DEPS_ROOT"
log "Python: $PYTHON_BIN"

ensure_checkout "axon_velocity" "https://github.com/adamwea/axon_velocity.git" "main" "$DEPS_ROOT/axon_velocity"
ensure_checkout "UnitMatch" "https://github.com/adamwea/UnitMatch.git" "enable_hdmea" "$DEPS_ROOT/UnitMatch"
ensure_checkout "SLAy" "https://github.com/adamwea/SLAy.git" "main" "$DEPS_ROOT/SLAy"
ensure_checkout "MEA_Analysis" "https://github.com/roybens/MEA_Analysis.git" "aw_dev" "$DEPS_ROOT/MEA_Analysis"

install_editable "axon_velocity" "$DEPS_ROOT/axon_velocity"
install_editable "UnitMatch" "$DEPS_ROOT/UnitMatch"
install_editable "SLAy" "$DEPS_ROOT/SLAy"
install_editable "MEA_Analysis" "$DEPS_ROOT/MEA_Analysis"

log "Bootstrap complete"