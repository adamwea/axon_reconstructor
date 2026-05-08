## Spikesort stage
# concat
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon --gpus all stages spikesort.bootstrap_concat_binary --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --force-restart --task-backend mpi

# sort. mpi backend doesnt work exactly right with container and container is needed for spikesort for now.
axon-recon-container --gpus all stages spikesort.sort --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --force-restart

# summarize. not a super critical phase, but lets see if it works with mpi backend.
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon --gpus all stages spikesort.summarize_sort --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --force-restart

## Reconstruct Stage
#analyzers
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.analyzers --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# extract_partial_templates
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.extract_partial_templates --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# build_templates
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.build_templates --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# plot_templates_v2
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.plot_templates_v2 --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# report_templates
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.report_templates --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# generate_gtrs
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.generate_gtrs --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# plot_recons
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.plot_recons --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# plot_branch_propagations
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.plot_branch_propagations --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# plot_branch_velocities
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.plot_branch_velocities --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# plot_unit_summary
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.plot_unit_summary --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# report_recons
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.report_recons --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# report_recon_grid
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.report_recon_grid --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# report_full_chip_layout
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.report_full_chip_layout --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart

# report_summaries
mpirun -np 2 --map-by ppr:2:node:pe=10 --bind-to core \
  axon-recon stages reconstruct.report_summaries --config debug/debug.runtime.yml \
  --target-dataset 11,12 --limit-wells 1 --task-backend mpi --force-restart