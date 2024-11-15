#!/bin/bash
#SBATCH --job-name=pipeline_test
#SBATCH -A m2043_g
#SBATCH -t 00:30:00  # Time limit in the form hh:mm:ss
#SBATCH -N 2  # Number of nodes requested, adjust as needed
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH --gpus=2
#SBATCH --output=parallel_pipeline_test_%j.out
#SBATCH --error=parallel_pipeline_test_%j.err

# Functions ====================================================================
# Function to generate the list of wells (HDF5 files) to process
generate_plate_files() {
    PLATE_FILES=()
    for h5_dir in "${H5_PARENT_DIRS[@]}"; do
        while IFS= read -r -d '' file; do
            if [[ "$file" == *"/AxonTracking/"* ]]; then
                PLATE_FILES+=("$file")
            fi
        done < <(find "$h5_dir" -type f -name '*.h5' -print0)
    done
}

# Function to extract recording details and generate unique log file names
generate_log_file_names() {
    local plate_file=$1
    local stream_select=$2

    local parent_dir=$(dirname "$plate_file")
    local runID=$(basename "$parent_dir")

    local grandparent_dir=$(dirname "$parent_dir")
    local scan_type=$(basename "$grandparent_dir")

    local great_grandparent_dir=$(dirname "$grandparent_dir")
    local chipID=$(basename "$great_grandparent_dir")

    local ggg_dir=$(dirname "$great_grandparent_dir")
    local date=$(basename "$ggg_dir")
    
    local gggg_dir=$(dirname "$ggg_dir")
    local project_name=$(basename "$gggg_dir")

    local log_file="${LOG_DIR}/${project_name}_${date}_${chipID}_${runID}_stream${stream_select}.log"
    local error_log_file="${LOG_DIR}/${project_name}_${date}_${chipID}_${runID}_stream${stream_select}.err"

    echo "$log_file" "$error_log_file"
}

# Args ========================================================================
# Define the HDF5 directories containing wells
H5_PARENT_DIRS=(
    "/pscratch/sd/a/adammwea/RBS_synology_rsync/B6J_DensityTest_10012024_AR"
)

# Define output and log directories
export OUTPUT_DIR="/pscratch/sd/a/adammwea/zRBS_axon_reconstruction_output"
LOG_DIR="/pscratch/sd/a/adammwea/zRBS_axon_reconstruction_output/logs"
mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

# Define the maximum number of parallel jobs
MAX_JOBS=20  # Adjust based on your resources

# Define the full path to the Python script
PYTHON_SCRIPT_PATH="/pscratch/sd/a/adammwea/RBS_axonal_reconstructions/development_scripts/developing_nbs_for_AR_density_projects/run_pipeline_HPC.py"

# Define the shifter image
SHIFTER_IMAGE="adammwea/axonkilo_docker:v7"

# Main Script ==================================================================

# Load necessary modules
echo "Loading modules..."
module load parallel
module load conda
conda activate axon_env

# Generate the list of wells (HDF5 files) to process
generate_plate_files
echo "Number of plates to process: ${#PLATE_FILES[@]}"
echo "Number of streams per plate: 6"
echo "Total number of tasks: $(( ${#PLATE_FILES[@]} * 6 ))"

# Prepare input file for GNU Parallel
input_file="parallel_input.txt"
> $input_file
for plate_file in "${PLATE_FILES[@]}"; do
    for stream in $(seq 0 5); do
        echo "$plate_file $stream" >> $input_file
    done
done

# Function to process each task
process_task() {
    local plate_file="$1"
    local stream="$2"
    read log_file error_log_file < <(generate_log_file_names "$plate_file" "$stream")
    mkdir -p "$(dirname "$log_file")"  # Ensure the log directory exists
    echo "Processing plate file: $plate_file, stream: $stream"
    srun -n 1 --gres=gpu:1 shifter --image=$SHIFTER_IMAGE python3 $PYTHON_SCRIPT_PATH \
        --plate_file "$plate_file" --stream_select "$stream" --output_dir "$OUTPUT_DIR" \
        > "$log_file" 2> "$error_log_file"
}

export -f process_task
export SHIFTER_IMAGE PYTHON_SCRIPT_PATH OUTPUT_DIR LOG_DIR
export -f generate_log_file_names

# Run tasks in parallel using GNU Parallel
parallel --jobs "$MAX_JOBS" --colsep ' ' process_task :::: $input_file

echo "All tasks completed."