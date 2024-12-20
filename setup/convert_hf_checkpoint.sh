#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Exit the script if no input is provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <arg1> <arg2> <arg3>"
  exit 1
fi

CONDA_PATH=$1
HF_CHECKPOINT_PATH=$2
MODEL_FAMILY=$3

# Set up module
ml purge
ml Mamba/23.11.0-0
conda deactivate
conda activate $CONDA_PATH

python setup/convert_hf_checkpoint.py $HF_CHECKPOINT_PATH $HF_CHECKPOINT_PATH-converted $MODEL_FAMILY