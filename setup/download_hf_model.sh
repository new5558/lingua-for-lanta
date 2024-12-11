#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Exit the script if no input is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <arg1> <arg2> <arg3>"
    exit 1
fi

# Assign the arguments to variables
CONDA_PATH=$1
REPO_ID=$2
DOWNLOAD_PATH=$3

# Set up module
ml purge
ml Mamba/23.11.0-0
conda deactivate
conda activate $CONDA_PATH

python setup/download_hf_model.py $REPO_ID $DOWNLOAD_PATH