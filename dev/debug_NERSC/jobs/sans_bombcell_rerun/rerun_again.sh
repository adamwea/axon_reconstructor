#!/bin/bash

cd /global/u2/a/adammwea/dev/pkgs/axon_recon
export MPICH_GPU_SUPPORT_ENABLED=1
#mkdir -p dev/debug_NERSC/jobs/sans_bombcell_rerun/logs

for WELL in well000 well001 well002 well003 well004 well005; do
  LOG=dev/debug_NERSC/jobs/sans_bombcell_rerun/logs/spikesort_ds4_${WELL}_${SLURM_JOB_ID}.out
  srun --exclusive -N 1 -n 1 -c 16 --cpu-bind=cores --threads-per-core=1 --gpus=1 --mem=56G \
    shifter axon-recon stages spikesort \
      --config dev/debug_NERSC/debug.runtime.yml \
      --profile perlmutter_gpu \
      --task-backend mpi \
      --cpus-per-task 16 \
      --target-dataset 4 \
      --target-wells "$WELL" \
      --force-restart \
      > "$LOG" 2>&1 &
done
#wait

# --exclusive makes each srun pin to its own GPU + cpu shard; 4 will run concurrently and the last 2 queue
# automatically.

#Inside the allocation:
#cd /global/u2/a/adammwea/dev/pkgs/axon_recon
#export MPICH_GPU_SUPPORT_ENABLED=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # mitigate fragmentation seen in ds12_well002
#mkdir -p dev/debug_NERSC/jobs/sans_bombcell_rerun/logs

# (IDX, WELL) pairs that OOM'd previously:
# (IDX, WELL) pairs that OOM'd previously:
#   ds1  = 260224/M06804/000032/well002
#   ds1  = 260224/M06804/000032/well002
#   ds6  = 260305/M06804/000099/well000
#   ds6  = 260305/M06804/000099/well000
#   ds6  = 260305/M06804/000099/well002
#   ds12 = 260319/M06804/000174/well002
#   ds6  = 260305/M06804/000099/well002
#   ds12 = 260319/M06804/000174/well002
#   ds12 = 260319/M06804/000174/well002
for SPEC in "1 well002" "6 well000" "6 well002" "12 well002"; do
  set -- $SPEC; IDX=$1; WELL=$2
  set -- $SPEC; IDX=$1; WELL=$2
  LOG=dev/debug_NERSC/jobs/sans_bombcell_rerun/logs/merge_SLAy_ds${IDX}_${WELL}_${SLURM_JOB_ID}.out
  srun --exclusive -N 1 -n 1 -c 16 --cpu-bind=cores --threads-per-core=1 --gpus=1 --mem=56G \
    shifter axon-recon stages spikesort.merge_SLAy \
      --config dev/debug_NERSC/debug.runtime.yml \
      --profile perlmutter_gpu \
      --task-backend mpi \
      --cpus-per-task 16 \
      --target-dataset "$IDX" \
      --target-wells "$WELL" \
      --force-restart \
      > "$LOG" 2>&1 &
done
wait