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
python setup/download_eval_datasets_lanta.py hellaswag --data_dir ./data 
python setup/download_eval_datasets_lanta.py piqa --data_dir ./data 
python setup/download_eval_datasets_lanta.py nq_open --data_dir ./data 