#!/bin/bash
# Perlmutter sbatch: full reconstruct stage on a subset of datasets.
#
# Usage:
#   sbatch perlmutter_reconstruct.sbatch <target_datasets>
#
#   target_datasets: comma-separated 0-based dataset indices into the active
#     data.yml's `datasets:` list (e.g. "0,1,2,3,4,5,6,7,8" for everything
#     except the last 4 datasets).
#
# Defaults below are sized for adamm/m2043 on Perlmutter. Override via env
# (RUNTIME_CFG, REPO_ROOT) if running from a different checkout or against a
# different config. See examples/README.md for the Slurm × shifter × profile
# truth table that explains why --threads-per-core=1 is required.

#SBATCH --job-name=axon-reconstruct
#SBATCH --account=m2043
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --threads-per-core=1
#SBATCH --image=adammwea/axon-recon:pipeline-v2
#SBATCH --output=/pscratch/sd/a/adammwea/run_logs/reconstruct/%x.%j.out
#SBATCH --error=/pscratch/sd/a/adammwea/run_logs/reconstruct/%x.%j.err

set -euo pipefail

TARGET_DATASETS="${1:?perlmutter_reconstruct.sbatch requires a comma-separated dataset index list as the first positional arg}"

REPO_ROOT="${REPO_ROOT:-/global/u2/a/adammwea/dev/pkgs/axon_recon}"
RUNTIME_CFG="${RUNTIME_CFG:-dev/debug_NERSC/debug.runtime.yml}"

mkdir -p /pscratch/sd/a/adammwea/run_logs/reconstruct
cd "$REPO_ROOT"

echo "=== axon-reconstruct sbatch ==="
echo "job:             ${SLURM_JOB_ID}"
echo "nodes:           ${SLURM_NNODES} (${SLURM_JOB_NODELIST})"
echo "tasks per node:  ${SLURM_NTASKS_PER_NODE}"
echo "cpus per task:   ${SLURM_CPUS_PER_TASK} physical (--threads-per-core=1)"
echo "target datasets: ${TARGET_DATASETS}"
echo "runtime cfg:     ${REPO_ROOT}/${RUNTIME_CFG}"
echo

srun --cpu-bind=cores --threads-per-core=1 \
  shifter --image=adammwea/axon-recon:pipeline-v2 \
  axon-recon stages reconstruct \
    --config "${RUNTIME_CFG}" \
    --task-profile perlmutter_cpu \
    --target-dataset "${TARGET_DATASETS}" \
    --task-backend mpi
