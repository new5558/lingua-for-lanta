#!/bin/bash


# Exit immediately if a command exits with a non-zero status
set -e

# Exit the script if no input is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path>"
  exit 1
fi

CONDA_PATH="$1"

# Set up module
ml purge
ml Mamba/23.11.0-0
conda deactivate
conda activate $CONDA_PATH
# Run the last command in tmux
python setup/download_prepare_hf_data.py fineweb_edu_10bt 12 --data_dir ./data --seed 42
