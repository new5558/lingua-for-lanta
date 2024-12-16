#!/bin/bash


# Exit immediately if a command exits with a non-zero status
set -e


# Check if exactly two arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <arg1> <arg2> <arg3>"
    exit 1
fi

# Assign the arguments to variables
CONDA_PATH=$1
HUGGINGFACE_API=$2
TOKENIZER_NAME=$3

tokenizer_path=./tokenizer_file

# Set up module
ml purge
ml Mamba/23.11.0-0
conda deactivate
conda activate $CONDA_PATH

python setup/download_tokenizer.py $TOKENIZER_NAME "${tokenizer_path}_${TOKENIZER_NAME}" --api_key  $HUGGINGFACE_API