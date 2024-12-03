#!/bin/bash


# Exit immediately if a command exits with a non-zero status
set -e


# Check if exactly two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <arg1> <arg2>"
    exit 1
fi

# Assign the arguments to variables
CONDA_PATH=$1
HUGGINGFACE_API=$2

tokenizer_path=./tokenizer_file

# Set up module
ml purge
ml Mamba/23.11.0-0
conda deactivate
conda activate $CONDA_PATH

python setup/download_tokenizer.py llama3 "$tokenizer_path" --api_key  $HUGGINGFACE_API