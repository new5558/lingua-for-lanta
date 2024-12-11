#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Exit the script if no input is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <path>"
  exit 1
fi

HF_CHECKPOINT_PATH="$1"

env_path=/project/lt200304-dipmt/new_norapat/lingua_conda

# Set up module
ml purge
ml Mamba/23.11.0-0
conda deactivate
conda activate $env_path

python setup/convert_hf_checkpoint.py $HF_CHECKPOINT_PATH/original/consolidated.00.pth $HF_CHECKPOINT_PATH-converted