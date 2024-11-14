#!/bin/bash
#SBATCH --job-name=pipeline_analysis
#SBATCH -A m2043
#SBATCH -t 24:00:00
#SBATCH -N 10  # Number of nodes requested, adjust as needed
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --exclusive
#SBATCH --output=./NERSC/output/analysis_%j.out
#SBATCH --error=./NERSC/output/analysis_%j.err

# Load necessary modules
echo "Loading modules and activating environment..."
echo "Loading conda..."
module load conda
echo "Activating conda environment..."
conda activate axon_env
echo "Loading parallel module..."
module load parallel
echo "Initializing shifter environment..."
shifter --image=rohanmalige/benshalom:v3 &
echo "Shifter environment initialized."
echo "Running pipeline analysis..."

# Define the HDF5 directories containing wells (you may want to specify these as arguments instead)
H5_PARENT_DIRS=(
    "/pscratch/sd/a/adammwea/RBS_synology_rsync/B6J_DensityTest_10012024_AR"
)

# Define output and log directory (ensure these exist before running)
OUTPUT_DIR="/pscratch/sd/a/adammwea/zRBS_axon_reconstruction_output"
LOG_DIR="/pscratch/sd/a/adammwea/zRBS_axon_reconstruction_output/logs"
mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

# Define the maximum number of parallel jobs (adjust based on available resources)
MAX_JOBS=4  # Change as needed, matching the number of nodes or available resources

# Generate the list of wells (HDF5 files) to process
WELL_FILES=()
for h5_dir in "${H5_PARENT_DIRS[@]}"; do
    while IFS= read -r -d '' file; do
        WELL_FILES+=("$file")
    done < <(find "$h5_dir" -type f -name '*.h5' -print0)
done

# output info about the number of wells
echo "Number of wells to process: ${#WELL_FILES[@]}"
#echo "WELL_FILES: ${WELL_FILES[@]}"
echo OUTPUT_DIR: $OUTPUT_DIR
echo LOG_DIR: $LOG_DIR

# Define the full path to the Python script
PYTHON_SCRIPT_PATH="/pscratch/sd/a/adammwea/RBS_axonal_reconstructions/development_scripts/developing_nbs_for_AR_density_projects/run_pipeline_HPC.py"

# Run pipeline on one well to test:
srun --exclusive -N1 -n1 python3 "$PYTHON_SCRIPT_PATH" --well_file "${WELL_FILES[0]}" --output_dir "${OUTPUT_DIR}"

# # Run the pipeline on each well in parallel
# export OUTPUT_DIR  # Export environment variables for parallel jobs
# export LOG_DIR

# echo "Starting parallel processing for ${#WELL_FILES[@]} wells..."
# parallel --jobs $MAX_JOBS --delay 1 --joblog ${LOG_DIR}/parallel_joblog.txt --resume \
#     srun --exclusive -N1 -n1 python3 run_pipeline.py --well_file {} --output_dir ${OUTPUT_DIR} \
#     ::: "${WELL_FILES[@]}"
