#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_local_image.sh [options]

Options:
  --image IMAGE          Image tag to build (default: axon-recon:local)
  --unitmatch PATH       UnitMatchPy package path (default: ../UnitMatch/UnitMatchPy if present)
  --slay PATH            SLAy package path (default: ../SLAy if present)
  --no-unitmatch         Do not include UnitMatchPy in the temporary context
  --no-slay              Do not include SLAy in the temporary context
  --docker CLI           Container CLI (default: AXON_RECON_CONTAINER_CLI or docker)
  --extra ARG            Extra docker build argument, repeatable
  --dry-run              Print the docker build command without running it
  --help                 Show this help

The script copies this repo plus optional sibling package directories into a temporary
context, then installs copied siblings from /opt/axon_recon/external/... inside
the image. This keeps runtime imports package-based instead of workspace-path based.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
container_cli="${AXON_RECON_CONTAINER_CLI:-docker}"
image="axon-recon:local"
unitmatch_path="$(cd "$repo_root/../UnitMatch/UnitMatchPy" 2>/dev/null && pwd || true)"
slay_path="$(cd "$repo_root/../SLAy" 2>/dev/null && pwd || true)"
include_unitmatch=1
include_slay=1
dry_run=0
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      image="${2:?--image requires a value}"
      shift 2
      ;;
    --unitmatch)
      unitmatch_path="$(cd "${2:?--unitmatch requires a value}" && pwd)"
      shift 2
      ;;
    --slay)
      slay_path="$(cd "${2:?--slay requires a value}" && pwd)"
      shift 2
      ;;
    --no-unitmatch)
      include_unitmatch=0
      shift
      ;;
    --no-slay)
      include_slay=0
      shift
      ;;
    --docker)
      container_cli="${2:?--docker requires a value}"
      shift 2
      ;;
    --extra)
      extra_args+=("${2:?--extra requires a value}")
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

copy_tree() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  (
    cd "$src"
    tar \
      --exclude='.git' \
      --exclude='__pycache__' \
      --exclude='.pytest_cache' \
      --exclude='.ruff_cache' \
      --exclude='.mypy_cache' \
      --exclude='.ipynb_checkpoints' \
      --exclude='*.egg-info' \
      --exclude='build' \
      --exclude='dist' \
      --exclude='outputs' \
      --exclude='scratch' \
      -cf - .
  ) | (
    cd "$dst"
    tar -xf -
  )
}

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
context_dir="$tmp_dir/context"

copy_tree "$repo_root" "$context_dir"

build_args=()
if [[ "$include_unitmatch" -eq 1 && -n "$unitmatch_path" && -d "$unitmatch_path" ]]; then
  copy_tree "$unitmatch_path" "$context_dir/external/UnitMatchPy"
  build_args+=(--build-arg UNITMATCH_SPEC=/opt/axon_recon/external/UnitMatchPy)
fi
if [[ "$include_slay" -eq 1 && -n "$slay_path" && -d "$slay_path" ]]; then
  copy_tree "$slay_path" "$context_dir/external/SLAy"
  build_args+=(--build-arg SLAY_SPEC=/opt/axon_recon/external/SLAy)
fi

cmd=(
  "$container_cli" build
  -f "$context_dir/containers/axon-recon/Dockerfile"
  -t "$image"
  "${build_args[@]}"
  "${extra_args[@]}"
  "$context_dir"
)

if [[ "$dry_run" -eq 1 ]]; then
  printf 'Resolved build command:\n'
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"