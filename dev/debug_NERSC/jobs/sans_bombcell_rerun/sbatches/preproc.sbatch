#!/bin/bash
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --account=m2043
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# Expects IDX and WELL via sbatch --export=ALL,IDX=...,WELL=...
set -euo pipefail
cd "$REPO_ROOT"

srun -n 1 -c 16 --cpu-bind=cores --threads-per-core=1 \
	shifter --image=adammwea/axon-recon:pipeline-v2 \
		axon-recon stages preprocess \
			--config dev/debug_NERSC/debug.runtime.yml \
			--task-profile perlmutter_cpu \
			--task-backend mpi \
			--cpus-per-task 16 \
			--target-dataset "$IDX" \
			--target-wells "$WELL"
