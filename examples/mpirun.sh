#!/bin/bash

# Working multi-rank shapes for axon_recon on the lab server.
#
# Override RUNTIME_CFG to point at dev/debug_NERSC/debug.runtime.yml, src/axon_recon/default.runtime.yml, etc.
#
# Two supported paths today:
#
# A. Host binary, no container — host mpirun launches axon-recon directly.
#    Validated end-to-end for preprocess after commit 82ed42c.
#
#    /usr/bin/mpirun -np 2 \
#      --map-by ppr:2:node:pe=10 \
#      --bind-to core \
#      --report-bindings \
#      axon-recon stages preprocess --config "$RUNTIME_CFG" \
#      --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart
#
# B. axon-recon-container --mpi-ranks N — ONE docker run, with `mpirun -np N`
#    inside the container. Default behavior (no flag) is byte-for-byte identical
#    to today's single-rank invocation. The wrapper owns the rank count; host
#    `mpirun -np N axon-recon-container …` is NOT supported (the strategy note
#    `dev/notes/guardrails/container_mpi_strategy_note.md` Option A, explicitly
#    out of scope).
#
#    Inside the container, per-rank CUDA_VISIBLE_DEVICES partitioning runs in
#    mpi_adapter before any torch/cupy/kilosort import; spikesort.sort
#    fails fast when ranks exceed visible GPUs. See
#    dev/notes/plans/completed/container_shifter_shape_plan.md §6.
#
#    axon-recon-container --mpi-ranks 2 stages preprocess \
#      --config "$RUNTIME_CFG" \
#      --target-dataset 11,12 --limit-wells 1 --limit-segments 2 \
#      --task-backend mpi --force-restart

RUNTIME_CFG="${RUNTIME_CFG:-dev/debug_local/debug.runtime.yml}"

# Default: keep the validated host-binary preprocess smoke runnable as before.
/usr/bin/mpirun -np 2 \
  --map-by ppr:2:node:pe=10 \
  --bind-to core \
  --report-bindings \
  axon-recon stages preprocess --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart
