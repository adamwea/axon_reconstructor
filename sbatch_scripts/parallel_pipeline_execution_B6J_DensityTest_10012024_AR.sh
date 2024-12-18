#!/bin/bash
#SBATCH --job-name=pipeline_test
#SBATCH -A m2043_g
#SBATCH -t 00:30:00  # Time limit in the format hh:mm:ss
#SBATCH -N 8  # Number of nodes requested
#SBATCH --mail-user=amwe@ucdavis.edu
#SBATCH --mail-type=ALL
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH --gpus=8
#SBATCH --output=parallel_pipeline_test_%j.out
#SBATCH --error=parallel_pipeline_test_%j.err

# ============================== Functions =====================================

# Function to generate the list of wells (HDF5 files) to process
generate_plate_files() {
    echo "Generating list of HDF5 plate files to process..."
    PLATE_FILES=()
    for h5_dir in "${H5_PARENT_DIRS[@]}"; do
        while IFS= read -r -d '' file; do
            if [[ "$file" == *"/AxonTracking/"* ]]; then
                PLATE_FILES+=("$file")
            fi
        done < <(find "$h5_dir" -type f -name '*.h5' -print0)
    done
    echo "Found ${#PLATE_FILES[@]} plate files to process."
}

# Function to extract recording details and generate unique log file names
generate_log_file_names() {
    local plate_file="$1"
    local stream_select="$2"

    local parent_dir runID grandparent_dir scan_type
    local great_grandparent_dir chipID ggg_dir date gggg_dir project_name

    parent_dir=$(dirname "$plate_file")
    runID=$(basename "$parent_dir")
    grandparent_dir=$(dirname "$parent_dir")
    scan_type=$(basename "$grandparent_dir")
    great_grandparent_dir=$(dirname "$grandparent_dir")
    chipID=$(basename "$great_grandparent_dir")
    ggg_dir=$(dirname "$great_grandparent_dir")
    date=$(basename "$ggg_dir")
    gggg_dir=$(dirname "$ggg_dir")
    project_name=$(basename "$gggg_dir")

    local well_id="well00${stream_select}"
    local well_dir="${OUTPUT_DIR}/${project_name}/${date}/${chipID}/${scan_type}/${runID}/${well_id}"

    local log_file="${well_dir}/${well_id}.log"
    local error_log_file="${well_dir}/${well_id}.err"

    echo "$log_file" "$error_log_file"
}

# Function to validate the existence of expected output files
validate_output_files() {
    local well_output_dir="$1"
    local all_files_exist=true

    local sortings_dir="${well_output_dir}/sortings"
    local waveforms_dir="${well_output_dir}/waveforms"
    local reconstructions_dir="${well_output_dir}/reconstructions"

    [[ -d "$sortings_dir" ]] || { echo "Missing: $sortings_dir"; all_files_exist=false; }
    [[ -d "$waveforms_dir" ]] || { echo "Missing: $waveforms_dir"; all_files_exist=false; }
    [[ -d "$reconstructions_dir" ]] || { echo "Missing: $reconstructions_dir"; all_files_exist=false; }

    $all_files_exist
}

# Function to process each task
process_task() {
    local plate_file="$1"
    local stream="$2"
    
    read log_file error_log_file < <(generate_log_file_names "$plate_file" "$stream")
    local well_output_dir
    well_output_dir=$(dirname "$log_file")
    mkdir -p "$well_output_dir"

    echo "Processing: $plate_file (Stream: $stream)"
    echo "Output Directory: $well_output_dir"

    if [[ "$fresh_recon" == "True" ]]; then
        echo "Fresh reconstruction requested. Proceeding with task execution."
        run_srun=true
    else
        echo "Validating existing output files..."
        if validate_output_files "$well_output_dir"; then
            echo "Output files exist. Skipping task."
            run_srun=false
        else
            run_srun=true
        fi
    fi

    if [[ "$run_srun" == "true" ]]; then
        echo "Running task with srun..."
        echo "Log file: $log_file"
        echo "Error log file: $error_log_file"
        echo "Shifter image: $SHIFTER_IMAGE"
        echo "Python script: $PYTHON_SCRIPT_PATH"
        echo "Plate file: $plate_file"
        #echo "Stream: $stream"
        #echo "Output directory: $OUTPUT_DIR"
        #break the script here
        #pause
        srun -n 1 shifter --image="$SHIFTER_IMAGE" python3 "$PYTHON_SCRIPT_PATH" \
            --plate_file "$plate_file" --stream_select "$stream" --output_dir "$OUTPUT_DIR" \
            > "$log_file" 2> "$error_log_file"
        # shifter --image="$SHIFTER_IMAGE" python3 "$PYTHON_SCRIPT_PATH" \
        #     --plate_file "$plate_file" --stream_select "$stream" --output_dir "$OUTPUT_DIR" \
        #     > "$log_file" 2> "$error_log_file"

        echo "Task completed: $plate_file $stream"
        validate_output_files "$well_output_dir"
    fi
}

# ============================== Arguments =====================================

# Define the HDF5 directories containing wells
H5_PARENT_DIRS=(
    #"/pscratch/sd/a/adammwea/xRBS_input_data/B6J_DensityTest_10012024_AR",
    "/pscratch/sd/a/adammwea/workspace/xInputs/xRBS_input_data/B6J_DensityTest_10012024_AR"

)

# Define output and log directories
#export OUTPUT_DIR="/pscratch/sd/a/adammwea/zRBS_axon_reconstruction_output"
export OUTPUT_DIR="//pscratch/sd/a/adammwea/workspace/yThroughput/zRBS_axon_reconstruction_output"
export LOG_DIR="${OUTPUT_DIR}/logs"
export ANALYZED_FILE="${OUTPUT_DIR}/analyzed_plate_files.txt"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Define the maximum number of parallel jobs
MAX_JOBS=$(( ${SLURM_NNODES:-1} * 2 ))

# Define the Python script and Shifter image
#PYTHON_SCRIPT_PATH="/pscratch/sd/a/adammwea/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC.py"
PYTHON_SCRIPT_PATH="/pscratch/sd/a/adammwea/workspace/RBS_axonal_reconstructions/pipeline_scripts/run_pipeline_HPC.py"
SHIFTER_IMAGE="adammwea/axonkilo_docker:v7"

# ============================== Main Script ===================================

module load parallel
module load conda

# Generate list of plate files to process
generate_plate_files

# Load previously analyzed plate files
declare -A ANALYZED_PLATE_FILES
if [[ -f "$ANALYZED_FILE" ]]; then
    while IFS= read -r line; do
        ANALYZED_PLATE_FILES["$line"]=1
    done < "$ANALYZED_FILE"
fi

# Prepare input for GNU Parallel
echo "Preparing input for parallel execution..."
input_data=""
for plate_file in "${PLATE_FILES[@]}"; do
    for stream in $(seq 0 5); do
        pair="${plate_file} ${stream}"
        input_data+="$pair"$'\n'
    done
done

parallel_input="${LOG_DIR}/parallel_input.txt"
echo "$input_data" > "$parallel_input"
sed -i '/^$/d' "$parallel_input"

# Export functions and variables for parallel tasks
export -f process_task
export SHIFTER_IMAGE PYTHON_SCRIPT_PATH OUTPUT_DIR LOG_DIR
export -f generate_log_file_names validate_output_files

#user options
export fresh_recon="True"

# Run tasks in parallel using GNU Parallel
echo "Running tasks in parallel..."
echo "Max jobs: $MAX_JOBS"
parallel --jobs "$MAX_JOBS" --colsep ' ' process_task :::: "$parallel_input"

echo "All tasks completed."
