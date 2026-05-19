#!/usr/bin/env bash
# install_dev_siblings.sh — idempotent editable-install helper.
#
# Installs sibling packages that axon_recon's [full] extra would pull
# from public git URLs, but in editable mode for active local
# development. Without this script (or its parent `setup_env.sh`), the
# `[full]` extra installs read-only copies from PyPI / GitHub; you
# couldn't modify, e.g., UnitMatchPy and have axon_recon pick up
# changes without re-installing.
#
# Targets (env_install_unification slice 4 scope — gated on USER
# INJECTION #4 for kssynth/unitlink):
#   - UnitMatchPy: subdirectory of upstream EnnyvanBeest/UnitMatch
#   - SLAy:        upstream saikoukunt/SLAy
#
# kssynth + unitlink will be added once their GitHub remotes exist
# (currently held local-only — see USER INJECTION #4).
#
# Behavior:
#   --from-local PATH    : Look for sibling clones under PATH first; if
#                          found, `pip install -e <PATH>/<sibling>/`.
#                          When PATH doesn't have a sibling, falls back
#                          to cloning into the repo-local `deps/` dir.
#                          When --from-local isn't supplied, all
#                          siblings come from `deps/`.
#   --dry-run            : Print what would be done; install nothing.
#   --siblings A,B,...   : Comma-separated subset of siblings to handle
#                          (default: all). Names: UnitMatchPy, SLAy.
#   -h / --help          : Show usage.
#
# Idempotency: subsequent runs no-op when the sibling is already
# pip-importable AND the editable install points at the expected path.
# When pip's existing install differs from the requested path, the
# script re-installs.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly DEPS_DIR="$REPO_ROOT/deps"

# Python interpreter used for the pip calls. Defaults to `python3` so
# the script works whether or not a conda env is activated; export
# PYTHON to override when running outside an active env. Examples:
#   PYTHON=$HOME/.conda/envs/axon_recon/bin/python ./tools/install_dev_siblings.sh
readonly PYTHON="${PYTHON:-python3}"

# Sibling registry. Each row:
#   pkg_name|local_clone_name|upstream-git-url|relative-subdir
#
# - pkg_name: Python package name (used for `pip show <pkg>` probes).
# - local_clone_name: directory name under --from-local PATH or the
#   `deps/` fallback. Often equals pkg_name, but distinguishes cases
#   like UnitMatchPy living inside a parent `UnitMatch/` repo.
# - upstream-git-url: cloned into deps/<local_clone_name>/ when no
#   --from-local override exists.
# - relative-subdir: package directory within the clone (empty means
#   the clone root itself is the package).
readonly SIBLINGS=(
  "UnitMatchPy|UnitMatch|https://github.com/EnnyvanBeest/UnitMatch.git|UnitMatchPy"
  "SLAy|SLAy|https://github.com/saikoukunt/SLAy.git|"
)

FROM_LOCAL=""
DRY_RUN=0
SIBLING_FILTER=""

usage() {
  sed -n '1,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

log() {
  printf '[install_dev_siblings] %s\n' "$*" >&2
}

run() {
  # Restore IFS for the printf so dry-run output reads cleanly even
  # when the caller's IFS was temporarily modified for record parsing.
  local IFS=$' \t\n'
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '[install_dev_siblings] [dry-run] %s\n' "$*" >&2
  else
    log "$*"
    "$@"
  fi
}

# Parse CLI args.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-local)
      FROM_LOCAL="${2:?--from-local requires a PATH}"
      shift 2
      ;;
    --from-local=*)
      FROM_LOCAL="${1#--from-local=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --siblings)
      SIBLING_FILTER="${2:?--siblings requires a comma-separated list}"
      shift 2
      ;;
    --siblings=*)
      SIBLING_FILTER="${1#--siblings=}"
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "[install_dev_siblings] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

# Resolve which siblings to operate on.
sibling_should_run() {
  local name="$1"
  if [[ -z "$SIBLING_FILTER" ]]; then
    return 0
  fi
  # Comma-separated allowlist match.
  if [[ ",$SIBLING_FILTER," == *",${name},"* ]]; then
    return 0
  fi
  return 1
}

# Resolve a sibling's local clone path. Preference order:
#   1. <from-local>/<local_clone_name>/ if --from-local supplied
#   2. <deps>/<local_clone_name>/ (cloned on demand)
sibling_resolve_path() {
  local local_clone_name="$1"
  local git_url="$2"

  if [[ -n "$FROM_LOCAL" ]]; then
    local candidate="$FROM_LOCAL/$local_clone_name"
    if [[ -d "$candidate/.git" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  local fallback="$DEPS_DIR/$local_clone_name"
  if [[ ! -d "$fallback/.git" ]]; then
    run mkdir -p "$DEPS_DIR"
    run git clone "$git_url" "$fallback"
  fi
  printf '%s\n' "$fallback"
}

# Compute the package directory inside the clone (UnitMatchPy lives in
# a subdir of the parent UnitMatch repo; SLAy is the repo root itself).
sibling_pkg_dir() {
  local clone_path="$1"
  local subdir="$2"
  if [[ -n "$subdir" ]]; then
    printf '%s\n' "$clone_path/$subdir"
  else
    printf '%s\n' "$clone_path"
  fi
}

# Idempotency probe: True when pip already shows an editable install
# whose location matches `target_path`.
already_installed_at() {
  local pkg="$1"
  local target_path="$2"
  local show_out
  if ! show_out=$("$PYTHON" -m pip show "$pkg" 2>/dev/null); then
    return 1
  fi
  local current_location
  current_location=$(printf '%s\n' "$show_out" | awk -F': ' '/^Editable project location:/{print $2}')
  if [[ -z "$current_location" ]]; then
    current_location=$(printf '%s\n' "$show_out" | awk -F': ' '/^Location:/{print $2}')
  fi
  # Compare resolved paths to be robust against trailing slashes.
  if [[ -z "$current_location" ]]; then
    return 1
  fi
  if [[ "$(cd "$current_location" 2>/dev/null && pwd)" == "$(cd "$target_path" 2>/dev/null && pwd)" ]]; then
    return 0
  fi
  return 1
}

install_sibling() {
  local row="$1"
  local IFS='|'
  read -ra parts <<<"$row"
  local pkg_name="${parts[0]}"
  local local_clone_name="${parts[1]}"
  local git_url="${parts[2]}"
  local subdir="${parts[3]:-}"

  if ! sibling_should_run "$pkg_name"; then
    log "skip $pkg_name (not in --siblings filter)"
    return 0
  fi

  log "resolving $pkg_name (clone_name=$local_clone_name subdir=${subdir:-<root>})"
  local clone_path
  clone_path=$(sibling_resolve_path "$local_clone_name" "$git_url")
  if [[ "$DRY_RUN" -eq 1 && ! -d "$clone_path/.git" ]]; then
    log "dry-run: skipping further work for $pkg_name (clone would happen here)"
    return 0
  fi
  local pkg_dir
  pkg_dir=$(sibling_pkg_dir "$clone_path" "$subdir")
  if [[ ! -d "$pkg_dir" ]]; then
    log "ERROR: package dir not found: $pkg_dir"
    return 1
  fi

  if already_installed_at "$pkg_name" "$pkg_dir"; then
    log "$pkg_name already editable-installed at $pkg_dir; skipping"
    return 0
  fi

  run "$PYTHON" -m pip install --no-deps -e "$pkg_dir"
}

main() {
  log "deps_dir=$DEPS_DIR from_local=${FROM_LOCAL:-<none>} dry_run=$DRY_RUN siblings_filter=${SIBLING_FILTER:-<all>}"
  for row in "${SIBLINGS[@]}"; do
    install_sibling "$row"
  done
  log "done"
}

main "$@"
