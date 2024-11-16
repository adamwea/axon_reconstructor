#!/bin/bash
#SBATCH --job-name=pipeline_analysis_debug
#SBATCH -A m2043
#SBATCH -t 24:00:00
#SBATCH -N 1  # Single node for debugging
#SBATCH --gpus=1  # Allocate one GPU for debugging
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug  # Use debug queue for shorter sessions
#SBATCH -C gpu  # Ensure GPU is available
#SBATCH --output=./NERSC/output/debug_%j.out
#SBATCH --error=./NERSC/output/debug_%j.err

echo "Loading NVIDIA programming environment..."
module load PrgEnv-nvidia
module load conda
#source activate axon_env

echo "Generating plate files..."
PLATE_FILES=()
for h5_dir in "/pscratch/sd/a/adammwea/RBS_synology_rsync/B6J_DensityTest_10012024_AR"; do
    while IFS= read -r -d '' file; do
        if [[ "$file" == *"/AxonTracking/"* ]]; then
            PLATE_FILES+=("$file")
        fi
    done < <(find "$h5_dir" -type f -name '*.h5' -print0)
done
export PLATE_FILE=${PLATE_FILES[2]}  # Example: use the 3rd file for debugging
echo "Plate file selected for debugging: $PLATE_FILE"

echo "Launching pipeline inside Shifter..."
# Start the process inside the Shifter container
shifter --image=rohanmalige/benshalom:v3 /bin/bash -c "
    export HDF5_PLUGIN_PATH=/usr/local/hdf5/lib/plugin  # Ensure the HDF5 plugin path is set
    conda activate axon_env  # Activate the conda environment
    python3 /pscratch/sd/a/adammwea/RBS_axonal_reconstructions/development_scripts/developing_nbs_for_AR_density_projects/run_pipeline_HPC.py \
        --plate_file $PLATE_FILE \
        --stream_select 0 \
        --output_dir /pscratch/sd/a/adammwea/zRBS_axon_reconstruction_output &
    echo \$! > /pscratch/sd/a/adammwea/debug_pid.txt
"

echo "Reading PID of running process..."
# Wait briefly to ensure the process starts
sleep 5
DEBUG_PID=$(cat /pscratch/sd/a/adammwea/debug_pid.txt)

if [ -z "$DEBUG_PID" ]; then
    echo "Failed to retrieve process PID."
    exit 1
fi
echo "Process running with PID: $DEBUG_PID"

echo "Launching cuda-gdb for external debugging..."
cuda-gdb -p "$DEBUG_PID" <<EOF
set breakpoint pending on
break my_cuda_function  # Replace with your CUDA function
continue
EOF
