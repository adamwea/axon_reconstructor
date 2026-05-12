#!/bin/bash
# Local (single-host, no container, no MPI) preprocess smoke launch.
# Override RUNTIME_CFG to point at debug_NERSC/debug.runtime.yml, default.runtime.yml, etc.
RUNTIME_CFG="${RUNTIME_CFG:-debug_local/debug.runtime.yml}"

axon-recon stages preprocess --config "$RUNTIME_CFG" \
  --target-dataset 11 --limit-wells 1 --alloc --task-backend local_affinity
