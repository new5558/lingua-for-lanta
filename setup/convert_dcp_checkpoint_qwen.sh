#!/bin/bash
#SBATCH --job-name=convert_weight_llama
#SBATCH -t 02:00:00
#SBATCH --partition=compute
#SBATCH -A <PROJECT_NAME>                 # Specify project name
#SBATCH -N 1 -c 128
#SBATCH --output=./logs/convert_qwen_%j.stdout

# Exit immediately if a command exits with a non-zero status
set -e

# Exit the script if no input is provided
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <arg1> <arg2> <arg3> <arg4>"
  exit 1
fi

CONDA_PATH=$1
LINGUA_CHECKPOINT_PATH=$2
TOKENIZER_PATH=$3
ORIGINAL_QWEN_HF_PATH=$4

# Set up module
ml purge
ml Mamba/23.11.0-0
ml git-lfs/3.2.0
conda deactivate
conda activate $CONDA_PATH

rm -rf $LINGUA_CHECKPOINT_PATH/hf_preparation
mkdir $LINGUA_CHECKPOINT_PATH/hf_preparation

cp  \
    $TOKENIZER_PATH/* \
    $LINGUA_CHECKPOINT_PATH/hf_preparation \

python setup/convert_dcp_checkpoint.py \
    $LINGUA_CHECKPOINT_PATH  \
    $LINGUA_CHECKPOINT_PATH/hf_preparation \

python setup/convert_qwen_to_hf.py \
    $LINGUA_CHECKPOINT_PATH/hf_preparation \
    $LINGUA_CHECKPOINT_PATH/hf \
    $ORIGINAL_QWEN_HF_PATH \
