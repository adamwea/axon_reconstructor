#!/bin/bash

# this seems to work
# but it runs double.
# mpirun -np 2 \
#   --map-by ppr:2:node:pe=10 \
#   --bind-to core \
#   --report-bindings \
#   -x OMP_NUM_THREADS=10 \
#   -x MKL_NUM_THREADS=10 \
#   -x OPENBLAS_NUM_THREADS=10 \
#   -x NUMEXPR_NUM_THREADS=10 \
#   axon-recon-container --task-backend mpi stages preprocess --config debug/debug.runtime.yml \
#   --target-dataset 11, 12 --limit-wells 1 --alloc

# this does not.
# # okay this works. Why not just mpirun?
# /usr/bin/mpirun -np 2 \
#   --map-by ppr:2:node:pe=10 \
#   --bind-to core \
#   --report-bindings \
#   -x OMP_NUM_THREADS=10 \
#   -x MKL_NUM_THREADS=10 \
#   -x OPENBLAS_NUM_THREADS=10 \
#   -x NUMEXPR_NUM_THREADS=10 \
#   axon-recon stages preprocess --config debug/debug.runtime.yml \
#   --target-dataset 11,12 --limit-wells 1 --alloc --task-backend mpi

# testing alloc
# /usr/bin/mpirun -np 2 \
#   --map-by ppr:2:node:pe=10 \
#   --bind-to core \
#   --report-bindings \
#   axon-recon stages preprocess --config debug/debug.runtime.yml \
#   --target-dataset 11,12 --limit-wells 1 --alloc --task-backend mpi

# testing preprocess
/usr/bin/mpirun -np 2 \
  --map-by ppr:2:node:pe=10 \
  --bind-to core \
  --report-bindings \
  axon-recon stages preprocess --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart