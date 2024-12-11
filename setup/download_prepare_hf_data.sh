#!/bin/bash


# Exit immediately if a command exits with a non-zero status
set -e

# Exit the script if no input is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <arg1> <arg2>"
    exit 1
fi

# Assign the arguments to variables
CONDA_PATH=$1
DOWNLOAD_PATH=$2

# Set up module
ml purge
ml Mamba/23.11.0-0
conda deactivate
conda activate $CONDA_PATH
# Run the last command in tmux
python setup/download_prepare_hf_data.py $DOWNLOAD_PATH 12 --data_dir ./data --seed 42
