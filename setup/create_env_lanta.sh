#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Exit the script if no input is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path>"
  exit 1
fi

INPUT_PATH="$1"

# Get the absolute path
FULL_PATH=$(realpath "$INPUT_PATH")

# Start timer
start_time=$(date +%s)

# Get the current date
current_date=$(date +%y%m%d)

# Create environment name with the current date
env_prefix=lingua
env_path=${FULL_PATH}/${env_prefix}_conda

# set up module
ml purge
ml Mamba/23.11.0-0
conda deactivate

# Create the conda environment

# source $CONDA_ROOT/etc/profile.d/conda.sh
conda create --prefix $env_path python=3.11 -y -c anaconda
conda activate $env_path

# For downloading datasets in background
conda install -y conda-forge::tmux

echo "Currently in env $(which python)"

# Install packages
pip install torch==2.5.0 xformers --index-url https://download.pytorch.org/whl/cu121
pip install ninja
pip install --requirement requirements.txt

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"