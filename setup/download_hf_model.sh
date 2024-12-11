#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Exit the script if no input is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <arg1> <arg2>"
    exit 1
fi

# Assign the arguments to variables
REPO_ID=$1
DOWNLOAD_PATH=$2

env_path=/project/lt200304-dipmt/new_norapat/lingua_conda

# Set up module
ml purge
ml Mamba/23.11.0-0
conda deactivate
conda activate $env_path

python setup/download_hf_model.py $REPO_ID $DOWNLOAD_PATH