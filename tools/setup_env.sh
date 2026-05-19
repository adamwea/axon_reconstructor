#!/usr/bin/env bash
# setup_env.sh — one-command developer install.
#
# Wraps:
#   1. `conda env create -f environment.yml` (skip when env exists)
#   2. `conda activate axon_recon`
#   3. `pip install -e .[dev,full]` (idempotent re-install)
#   4. Optional: `tools/install_dev_siblings.sh` for editable sibling
#      development (only when --editable-siblings is passed).
#
# Behavior:
#   --editable-siblings           Install siblings (UnitMatchPy, SLAy)
#                                 from local clones via
#                                 tools/install_dev_siblings.sh. After
#                                 the base `pip install -e .[dev,full]`
#                                 brings them in from git URLs, the
#                                 sibling helper re-installs them
#                                 editable, overriding the git URL
#                                 versions.
#   --from-local PATH             Forwarded to install_dev_siblings.sh.
#                                 Defaults to $HOME/dev/pkgs/ when
#                                 --editable-siblings is set.
#   --skip-base-install           Don't run `pip install -e .[dev,full]`
#                                 (assumes you've already installed
#                                 it). Useful when re-running just to
#                                 refresh sibling installs.
#   --env-name NAME               Override the default env name
#                                 (axon_recon).
#   -h / --help                   Show usage.
#
# Idempotent: re-running with the same args is safe. The env-create
# step short-circuits when the env already exists; pip install -e is a
# no-op when nothing has changed.
#
# Assumes `conda` is on PATH (e.g. miniconda3 installed). If not,
# the script bails with instructions.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly ENV_FILE="$REPO_ROOT/environment.yml"

ENV_NAME="axon_recon"
EDITABLE_SIBLINGS=0
FROM_LOCAL=""
SKIP_BASE_INSTALL=0

log() {
  printf '[setup_env] %s\n' "$*" >&2
}

usage() {
  sed -n '1,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --editable-siblings)
      EDITABLE_SIBLINGS=1
      shift
      ;;
    --from-local)
      FROM_LOCAL="${2:?--from-local requires a PATH}"
      shift 2
      ;;
    --from-local=*)
      FROM_LOCAL="${1#--from-local=}"
      shift
      ;;
    --skip-base-install)
      SKIP_BASE_INSTALL=1
      shift
      ;;
    --env-name)
      ENV_NAME="${2:?--env-name requires a NAME}"
      shift 2
      ;;
    --env-name=*)
      ENV_NAME="${1#--env-name=}"
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "[setup_env] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  cat >&2 <<EOF
[setup_env] ERROR: \`conda\` is not on PATH.

This script assumes miniconda or anaconda is installed and conda is
shell-accessible. Install miniconda first:
  https://docs.conda.io/en/latest/miniconda.html

Then re-run this script.
EOF
  exit 1
fi

# Initialize conda for this shell so `conda activate` works inside the
# script. `conda info --base` returns the conda install dir; its
# `etc/profile.d/conda.sh` is what `conda init` adds to ~/.bashrc.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "conda env '$ENV_NAME' already exists; skipping create"
else
  log "creating conda env '$ENV_NAME' from $ENV_FILE"
  conda env create -f "$ENV_FILE" -n "$ENV_NAME"
fi

log "activating $ENV_NAME"
conda activate "$ENV_NAME"

if [[ $SKIP_BASE_INSTALL -eq 0 ]]; then
  log "installing axon_recon + [dev,full] extras (editable)"
  cd "$REPO_ROOT"
  python -m pip install -e ".[dev,full]"
else
  log "--skip-base-install: leaving \`pip install -e .[dev,full]\` to the caller"
fi

if [[ $EDITABLE_SIBLINGS -eq 1 ]]; then
  if [[ -z "$FROM_LOCAL" ]]; then
    FROM_LOCAL="$HOME/dev/pkgs"
    log "--editable-siblings without --from-local: defaulting to $FROM_LOCAL"
  fi
  log "re-installing siblings editable via tools/install_dev_siblings.sh"
  PYTHON="$(command -v python)" "$SCRIPT_DIR/install_dev_siblings.sh" \
    --from-local "$FROM_LOCAL"
fi

cat <<EOF
[setup_env] done.

  conda activate $ENV_NAME
  pytest                      # run the test suite
  axon-recon --help           # CLI entry point

EOF
