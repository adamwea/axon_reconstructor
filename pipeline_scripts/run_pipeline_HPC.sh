#!/bin/bash
#SBATCH -A m2043_g               # Account
#SBATCH -C gpu                   # Request GPU nodes
#SBATCH -q regular                 # Queue
#SBATCH -t 02:00:00              # Time limit
#SBATCH -N 2                     # Number of nodes
#SBATCH --gpus-per-task=2           # One GPU per task
#SBATCH --mail-type=ALL             # Send email on all job events

# Bind CPUs for optimal performance
export SLURM_CPU_BIND="cores"

# run one mpi thread per python process, give each process 64 cores
## TODO: need to fix mpi allocation for this, currently it's running serially


#----------------------------------------------

# for i in {0..5}; do
#     echo "Running stream $i..."
#     srun -N 1 --ntasks-per-node=1 -c 64 --gpus-per-task=1 shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select $i
#     # echo "Stream $i srun added to queue"
# done

# Launch your application
# for i in {0..5}; do
#     echo "Running stream $i..."
#     srun -N 1 --ntasks-per-node=1 -c 64 --gpus-per-task=1 shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select $i
#     # echo "Stream $i srun added to queue"
# done

# #count down from 5 to 0 instead
# for i in {5..0}; do
#     echo "Running stream $i..."
#     srun -N 1 --ntasks-per-node=1 -c 64 --gpus-per-task=1 shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select $i
#     # echo "Stream $i srun added to queue"
# done

#----------------------------------------------

#target well 0 for testing
#srun shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select 0

# # #target well 1 for testing
#srun shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select 1

# # #target well 2 for testing
#srun shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select 2 

# # #target well 3 for testing
#srun shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select 3

# # #target well 4 for testing
#srun shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select 4

# # #target well 5 for testing
#srun shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select 5

#----------------------------------------------

# # #target well 1 for testing
#srun -N 1 --ntasks-per-node=1 -c 64 --gpus-per-task=1 shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select 1

# # #target well 2 for testing
#srun -N 1 --ntasks-per-node=1 -c 64 --gpus-per-task=1 shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select 2

# # #target well 3 for testing
#srun -N 1 --ntasks-per-node=1 -c 64 --gpus-per-task=1 shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select 3

# # #target well 4 for testing
# srun -N 1 --ntasks-per-node=1 -c 64 --gpus-per-task=1 shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select 4

# # #target well 5 for testing
#srun -N 1 --ntasks-per-node=1 -c 64 --gpus-per-task=1 shifter --image=adammwea/axonkilo_docker:v7 python /pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC_latest.py --stream_select 5