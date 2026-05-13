#!/usr/bin/env bash
# Rebuild + push + Shifter-refresh the axon-recon container in one shot.
#
# Intended workflow (NERSC):
#   1. Edit code in this repo.
#   2. Run:   containers/axon-recon/rebuild_shifter.sh
#   3. The next sbatch / srun shifter ... picks up the new image.
#
# What it does:
#   - Picks the best available container CLI (podman-hpc on NERSC, podman or
#     docker elsewhere).
#   - Calls `build_local_image.sh` to bake a fresh image from the current repo
#     state, including sibling UnitMatch / SLAy checkouts when present.
#   - Pushes the image to Docker Hub.
#   - Asks shifterimg to refresh its cached copy so compute nodes see the new
#     digest.
#
# Usage:
#   rebuild_shifter.sh                              # uses default tag pipeline-v2
#   rebuild_shifter.sh adammwea/axon-recon:<tag>    # build/push/pull a specific tag
#
# Environment overrides:
#   AXON_RECON_CONTAINER_CLI=<podman-hpc|podman|docker>
#       Force a specific CLI instead of auto-detect.
#   AXON_RECON_REBUILD_SKIP_PUSH=1
#       Build only (no docker.io push, no shifterimg pull). Useful for iteration
#       when the image isn't ready to ship yet.
#   AXON_RECON_REBUILD_SKIP_SHIFTERIMG=1
#       Skip the shifterimg pull step (used on hosts without shifter).
#
# Prerequisites:
#   - `podman login docker.io` (or the equivalent for your chosen CLI) has been
#     run at least once. The push step will fail loudly otherwise.

set -euo pipefail

DEFAULT_IMAGE="adammwea/axon-recon:pipeline-v2"
IMAGE="${1:-$DEFAULT_IMAGE}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pick_container_cli() {
	if [[ -n "${AXON_RECON_CONTAINER_CLI:-}" ]]; then
		printf '%s\n' "$AXON_RECON_CONTAINER_CLI"
		return 0
	fi
	for candidate in podman-hpc podman docker; do
		if command -v "$candidate" >/dev/null 2>&1; then
			printf '%s\n' "$candidate"
			return 0
		fi
	done
	return 1
}

if ! CLI="$(pick_container_cli)"; then
	echo "ERROR: no container CLI found. Install podman-hpc / podman / docker, or set AXON_RECON_CONTAINER_CLI." >&2
	exit 2
fi

echo "image:           ${IMAGE}"
echo "container CLI:   ${CLI}"
echo

echo "=== Step 1/3: build with ${CLI} ==="
AXON_RECON_CONTAINER_CLI="$CLI" "$script_dir/build_local_image.sh" --image "$IMAGE" --docker "$CLI"
echo

if [[ "${AXON_RECON_REBUILD_SKIP_PUSH:-0}" == "1" ]]; then
	echo "AXON_RECON_REBUILD_SKIP_PUSH=1 — skipping push + shifterimg refresh."
	echo "Image is built locally. Test with:"
	echo "  ${CLI} run --rm ${IMAGE} axon-recon --help"
	exit 0
fi

echo "=== Step 2/3: push ${IMAGE} to docker.io ==="
"$CLI" push "$IMAGE"
echo

if [[ "${AXON_RECON_REBUILD_SKIP_SHIFTERIMG:-0}" == "1" ]] || ! command -v shifterimg >/dev/null 2>&1; then
	echo "Step 3/3: skipping shifterimg pull (not available or skipped via env)"
	echo
	echo "Done. Trigger Shifter refresh manually on a NERSC head node with:"
	echo "  shifterimg pull ${IMAGE}"
	exit 0
fi

echo "=== Step 3/3: refresh Shifter cache ==="
shifterimg pull "$IMAGE"
echo

echo "Done."
echo "Verify the new image is live with:"
echo "  shifter --image=${IMAGE} axon-recon status --config dev/debug_NERSC/debug.runtime.yml"
