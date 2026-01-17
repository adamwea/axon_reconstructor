#!/usr/bin/env bash
set -euo pipefail

# Build + push the Shifter image from your local machine.
# Usage:
#   TAG=v8 ./scripts/docker/build_and_push_axonkilo.sh
# Optional:
#   IMAGE=adammwea/axonkilo_docker TAG=v8 PLATFORM=linux/amd64 ./scripts/docker/build_and_push_axonkilo.sh

IMAGE="${IMAGE:-adammwea/axonkilo_docker}"
TAG="${TAG:?Set TAG, e.g. TAG=v8}"
PLATFORM="${PLATFORM:-linux/amd64}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

echo "Building $IMAGE:$TAG (platform=$PLATFORM)"
docker build --platform="$PLATFORM" \
  -t "$IMAGE:$TAG" \
  -f environments/docker/Dockerfile \
  environments

echo "Pushing $IMAGE:$TAG"
docker push "$IMAGE:$TAG"

echo "Done. Next on NERSC:"
echo "  shifterimg pull docker:$IMAGE:$TAG"
