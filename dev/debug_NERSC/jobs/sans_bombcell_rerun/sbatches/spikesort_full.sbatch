#!/bin/bash
#SBATCH --qos=shared
#SBATCH --constraint=gpu
#SBATCH --account=m2043_g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=56G
#SBATCH --time=04:00:00

# NERSC shared GPU policy: -q shared -C gpu -G N -A <_g acct>, cpus_per_task >= 32 per GPU.
# Expects IDX and WELL via sbatch --export=ALL,IDX=...,WELL=...
set -euo pipefail
cd "$REPO_ROOT"

export MPICH_GPU_SUPPORT_ENABLED=1

# sbatch reserves 32 LOGICAL cpus (= 16 physical w/ SMT2) -- NERSC's per-GPU shared minimum.
# srun then pins to 16 PHYSICAL cores via --threads-per-core=1; -c 32 here would request
# 32 physical and fail with "More processors requested than permitted".
srun -n 1 -c 16 --cpu-bind=cores --threads-per-core=1 --gpus=1 \
	shifter --image=adammwea/axon-recon:pipeline-v2 \
		axon-recon stages spikesort \
			--config dev/debug_NERSC/debug.runtime.yml \
			--task-profile perlmutter_gpu \
			--task-backend mpi \
			--cpus-per-task 16 \
			--target-dataset "$IDX" \
			--target-wells "$WELL"
