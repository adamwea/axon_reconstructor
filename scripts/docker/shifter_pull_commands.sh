#!/usr/bin/env bash
set -euo pipefail

# Print the Shifter commands to run on NERSC after pushing a new Docker tag.
# Usage:
#   TAG=v8 ./scripts/docker/shifter_pull_commands.sh

IMAGE="${IMAGE:-adammwea/axonkilo_docker}"
TAG="${TAG:?Set TAG, e.g. TAG=v8}"

cat <<EOF
shifterimg pull docker:$IMAGE:$TAG
shifterimg images | grep axonkilo || true

# Interactive GPU allocation using that image:
salloc -A <YOUR_ALLOCATION> -q interactive -C gpu -t 04:00:00 --nodes=1 --gpus=1 --image=$IMAGE:$TAG
EOF
