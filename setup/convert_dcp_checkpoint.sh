#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Exit the script if no input is provided
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <arg1> <arg2> <arg3> <arg4> <arg5>"
  exit 1
fi

CONDA_PATH=$1
LINGUA_CHECKPOINT_PATH=$2
TOKENIZER_PATH=$3
HF_TOKEN=$4
PUSH_HF_REPO=$5

# Set up module
ml purge
ml Mamba/23.11.0-0
ml git-lfs/3.2.0
conda deactivate
conda activate $CONDA_PATH

cp  \
    $TOKENIZER_PATH \
    $LINGUA_CHECKPOINT_PATH/consolidated \


python setup/convert_dcp_checkpoint.py \
    $LINGUA_CHECKPOINT_PATH/consolidated/consolidated.pth  \
    $LINGUA_CHECKPOINT_PATH/consolidated/consolidated.00.pth \


python -m transformers.models.llama.convert_llama_weights_to_hf \
    --input_dir $LINGUA_CHECKPOINT_PATH/consolidated \
    --output_dir $LINGUA_CHECKPOINT_PATH/hf \
    --llama_version 3 \
    --num_shards 1 \

huggingface-cli upload --token $HF_TOKEN $PUSH_HF_REPO $LINGUA_CHECKPOINT_PATH/hf .