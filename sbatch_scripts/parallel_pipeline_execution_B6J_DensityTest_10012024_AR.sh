#!/bin/bash
#SBATCH --job-name=pipeline_test
#SBATCH -A m2043_g
#SBATCH -t 00:30:00  # Time limit in the form hh:mm:ss
#SBATCH -N 4  # Number of nodes requested, adjust as needed
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH --gpus=4
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
    #echo "Generating log file names..."
    local plate_file="$1"
    local stream_select="$2"

    local parent_dir
    parent_dir=$(dirname "$plate_file")
    local runID
    runID=$(basename "$parent_dir")

    local grandparent_dir
    grandparent_dir=$(dirname "$parent_dir")
    local scan_type
    scan_type=$(basename "$grandparent_dir")

    local great_grandparent_dir
    great_grandparent_dir=$(dirname "$grandparent_dir")
    local chipID
    chipID=$(basename "$great_grandparent_dir")

    local ggg_dir
    ggg_dir=$(dirname "$great_grandparent_dir")
    local date
    date=$(basename "$ggg_dir")
    
    local gggg_dir
    gggg_dir=$(dirname "$ggg_dir")
    local project_name
    project_name=$(basename "$gggg_dir")

    # well_dir=project/date/chipID/runID/scan_type/plate_file/stream_select
    local well_id="well00${stream_select}"
    local well_dir="${OUTPUT_DIR}/${project_name}/${date}/${chipID}/${scan_type}/${runID}/${well_id}"

    local log_file="${well_dir}/${well_id}.log"
    local error_log_file="${well_dir}/${well_id}.err"

    echo "$log_file" "$error_log_file"
}

# Args ========================================================================
# Define the HDF5 directories containing wells
H5_PARENT_DIRS=(
    "/pscratch/sd/a/adammwea/xRBS_input_data/B6J_DensityTest_10012024_AR"
)

# Define output and log directories
export OUTPUT_DIR="/pscratch/sd/a/adammwea/zRBS_axon_reconstruction_output"
export LOG_DIR="/pscratch/sd/a/adammwea/zRBS_axon_reconstruction_output/logs"
export ANALYZED_FILE="${OUTPUT_DIR}/analyzed_plate_files.txt"
mkdir -p ${OUTPUT_DIR} ${LOG_DIR}

# Define the maximum number of parallel jobs
MAX_JOBS=$(( ${SLURM_NNODES:-1} * 2 )) # Number of nodes, default to 1 if not set

# Define the full path to the Python script
PYTHON_SCRIPT_PATH="/pscratch/sd/a/adammwea/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC.py"

# Define the shifter image
SHIFTER_IMAGE="adammwea/axonkilo_docker:v7"

# Main Script ==================================================================

# Load necessary modules
echo "Loading modules..."
module load parallel
module load conda
#conda activate axon_env

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

# Function to process each task
process_task() {
    local plate_file="$1"
    local stream="$2"
    #echo "Processing plate file: $plate_file"
    #echo "stream: $stream"

    # Uncomment and modify the following lines as needed
    #echo Generating log file names...
    read log_file error_log_file < <(generate_log_file_names $plate_file $stream)
    #echo "Log file: $log_file"
    #echo "Error log file: $error_log_file"

    # Ensure the log directory exists
    mkdir -p "$(dirname "$log_file")"  # Ensure the log directory exists

    # Run the Python script using Shifter
    # Example: srun -n 1 --gres=gpu:1 shifter --image=adammwea/axonkilo_docker:v7 python3 run_pipeline_HPC.py --plate_file "$plate_file" --stream_select "$stream" --output_dir "$OUTPUT_DIR"   
    #echo "Running Python script..."
    # srun -n 1 --gres=gpu:1
    srun -n 1 shifter --image=$SHIFTER_IMAGE python3 $PYTHON_SCRIPT_PATH \
        --plate_file "$plate_file" --stream_select "$stream" --output_dir "$OUTPUT_DIR" \
        > "$log_file" 2> "$error_log_file"
    #echo "$plate_file $stream" >> "$ANALYZED_FILE"
    echo "Task completed: $plate_file $stream"
}

# Prepare input for GNU Parallel
echo "Preparing input for GNU Parallel..."
input_data=""
for plate_file in "${PLATE_FILES[@]}"; do
    for stream in $(seq 0 5); do
        pair="${plate_file} ${stream}"
        # Check if the pair is already analyzed
        # if not, add it to the input data
        # if [[ -z "${ANALYZED_PLATE_FILES[$pair]}" ]]; then
        #     echo "$pair" "added to input data"
        #     input_data+="$pair"$'\n'
        # fi
        #echo "$pair" "added to input data"
        input_data+="$pair"$'\n'
    done
done

# Save input data to a file
#test_txt="/pscratch/sd/a/adammwea/parallel_input.txt"
parallel_input="parallel_input.txt"
echo "$input_data" > "$parallel_input"

#make sure there are no trailing or leading returns in the file
sed -i '/^$/d' "$parallel_input"

echo "Exporting functions and variables for parallel tasks..."
export -f process_task
export SHIFTER_IMAGE PYTHON_SCRIPT_PATH OUTPUT_DIR LOG_DIR
export -f generate_log_file_names

# Run tasks in parallel using GNU Parallel
echo "Running tasks in parallel..."
echo "$MAX_JOBS tasks will be run in parallel."
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "LOG_DIR: $LOG_DIR"
echo "PYTHON_SCRIPT_PATH: $PYTHON_SCRIPT_PATH"
echo "SHIFTER_IMAGE: $SHIFTER_IMAGE"
parallel --jobs "$MAX_JOBS" --colsep ' ' process_task :::: "$parallel_input"

echo "All tasks completed."