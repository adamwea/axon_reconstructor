#!/bin/bash
# End-to-end spikesort + reconstruct smoketest on the lab server.
# Override RUNTIME_CFG to point at debug_NERSC/debug.runtime.yml, default.runtime.yml, etc.
RUNTIME_CFG="${RUNTIME_CFG:-debug_local/debug.runtime.yml}"

## Spikesort stage
# concat
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages spikesort.bootstrap_concat_binary --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --force-restart --task-backend mpi

# sort. mpi backend doesnt work exactly right with container and container is needed for spikesort for now.
axon-recon-container --gpus all stages spikesort.sort --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --force-restart

# summarize. not a super critical phase, but lets see if it works with mpi backend.
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages spikesort.summarize_sort --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --force-restart

## Reconstruct Stage
#analyzers
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.analyzers --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# extract_partial_templates
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.extract_partial_templates --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# build_templates
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.build_templates --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# plot_templates_v2
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.plot_templates_v2 --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# report_templates
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.report_templates --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# generate_gtrs
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.generate_gtrs --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# plot_recons
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.plot_recons --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# plot_branch_propagations
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.plot_branch_propagations --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# plot_branch_velocities
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.plot_branch_velocities --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# plot_unit_summary
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.plot_unit_summary --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# report_recons
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.report_recons --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# report_recon_grid
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.report_recon_grid --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# report_full_chip_layout
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.report_full_chip_layout --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# report_summaries
/usr/bin/mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.report_summaries --config "$RUNTIME_CFG" \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart
