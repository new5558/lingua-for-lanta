#!/bin/bash
#SBATCH --job-name=convert_weight_llama
#SBATCH -t 02:00:00
#SBATCH --partition=compute
#SBATCH -A <PROJECT_NAME>                 # Specify project name
#SBATCH -N 1 -c 128
#SBATCH --output=./logs/convert_llama_%j.stdout

# Exit immediately if a command exits with a non-zero status
set -e

# Exit the script if no input is provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <arg1> <arg2> <arg3>"
  exit 1
fi

CONDA_PATH=$1
LINGUA_CHECKPOINT_PATH=$2
TOKENIZER_PATH=$3

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

cp  \
    $LINGUA_CHECKPOINT_PATH/params.json \
    $LINGUA_CHECKPOINT_PATH/hf_preparation/params.json \

python setup/convert_dcp_checkpoint.py \
    $LINGUA_CHECKPOINT_PATH  \
    $LINGUA_CHECKPOINT_PATH/hf_preparation \

python -m transformers.models.llama.convert_llama_weights_to_hf \
    --input_dir $LINGUA_CHECKPOINT_PATH/hf_preparation \
    --output_dir $LINGUA_CHECKPOINT_PATH/hf \
    --llama_version 3.2 \
    --num_shards 1 \