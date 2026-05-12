#!/bin/bash
# Container-wrapped preprocess launch.
# Override RUNTIME_CFG to point at dev/debug_NERSC/debug.runtime.yml, src/axon_recon/default.runtime.yml, etc.
RUNTIME_CFG="${RUNTIME_CFG:-dev/debug_local/debug.runtime.yml}"

axon-recon-container stages preprocess --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --alloc --task-backend local_affinity

axon-recon-container stages preprocess --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --alloc --task-backend mpi
