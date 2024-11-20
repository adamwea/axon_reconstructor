#!/bin/bash
#SBATCH --job-name=Human_E_Neurons_T1_06072024
#SBATCH -A m2043_g
#SBATCH -t 06:00:00  # Time limit in the form hh:mm:ss
#SBATCH -N 6  # Number of nodes requested, adjust as needed
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH --gpus=6
#SBATCH --output=Human_E_Neurons_T1_06072024_%j.out
#SBATCH --error=Human_E_Neurons_T1_06072024_%j.err

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
    "/pscratch/sd/a/adammwea/xRBS_input_data/Human_E_Neurons_T1_06072024"
)

# Define output and log directories
export OUTPUT_DIR="/pscratch/sd/a/adammwea/zRBS_axon_reconstruction_output"
export LOG_DIR="/pscratch/sd/a/adammwea/zRBS_axon_reconstruction_output/logs"
export ANALYZED_FILE="${OUTPUT_DIR}/analyzed_plate_files.txt"
mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

# Define the maximum number of parallel jobs
MAX_JOBS=6  # Adjust based on your resources

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
echo "Maximum number of parallel jobs: $MAX_JOBS"

# Load analyzed plate files if the file exists
declare -A ANALYZED_PLATE_FILES
if [[ -f "$ANALYZED_FILE" ]]; then
    while IFS= read -r line; do
        ANALYZED_PLATE_FILES["$line"]=1
    done < "$ANALYZED_FILE"
fi

# Prepare input file for GNU Parallel
input_file="parallel_input.txt"
> $input_file
for plate_file in "${PLATE_FILES[@]}"; do
    for stream in $(seq 0 5); do
        pair="${plate_file} ${stream}"
        if [[ -z "${ANALYZED_PLATE_FILES[$pair]}" ]]; then
            echo "$pair" >> $input_file
        fi
    done
done
# Reverse the order of tasks in the input file
tac $input_file > temp_file && mv temp_file $input_file

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
    echo "$plate_file $stream" >> "$ANALYZED_FILE"
}

export -f process_task
export SHIFTER_IMAGE PYTHON_SCRIPT_PATH OUTPUT_DIR LOG_DIR
export -f generate_log_file_names

# Run tasks in parallel using GNU Parallel
parallel --jobs "$MAX_JOBS" --colsep ' ' process_task :::: $input_file

echo "All tasks completed."