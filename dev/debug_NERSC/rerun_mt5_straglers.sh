  #!/bin/bash
  set -euo pipefail
  cd /global/u2/a/adammwea/dev/pkgs/axon_recon

  # rebuild shifter
  # containers/axon-recon/rebuild_shifter.sh this cant run in an interactive session? or job I guess? needs to be login?

  # enable GPU support for MPI
  export MPICH_GPU_SUPPORT_ENABLED=1

  echo "=== Command 1: dataset 1 wells 003 + 005 (Probably won't work, weak signal) ==="
  srun --cpu-bind=cores --threads-per-core=1 \
    shifter --image=adammwea/axon-recon:pipeline-v2 \
    axon-recon stages spikesort \
      --config dev/debug_NERSC/debug.runtime.yml \
      --task-profile perlmutter_gpu \
      --target-dataset 1 \
      --target-wells well003 well005 \
      --task-backend mpi \
      --force-restart

  echo "=== Command 2: dataset 8 well004 → full spikesort chain ==="
  srun --cpu-bind=cores --threads-per-core=1 \
    shifter --image=adammwea/axon-recon:pipeline-v2 \
    axon-recon stages spikesort \
      --config dev/debug_NERSC/debug.runtime.yml \
      --task-profile perlmutter_gpu \
      --target-dataset 8 \
      --target-wells well004 \
      --task-backend mpi \
      --force-restart

  echo "=== Done. Verifying with status ==="
  shifter --image=adammwea/axon-recon:pipeline-v2 \
    axon-recon status --config dev/debug_NERSC/debug.runtime.yml \
    --stage spikesort --target-dataset 1 8