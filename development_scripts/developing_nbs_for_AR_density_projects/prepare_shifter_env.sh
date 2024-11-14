#bin/bash

module load conda
shifter --image=rohanmalige/benshalom:v3 /bin/bash
conda activate axon_env