# Lingua Modified for Lanta

This is lingua refactored for Lanta HPC Cluster.
The original readme of this repository is [HERE](README_original.md)

## Changelog

1. Removed `unlimit` command from code because Lanta don't have permission to change shell resource limit
2. created script to create environement specifically made for Lanta: `create_env_lanta.sh`
3. created script to download training data and tokenizer `setup/download_prepare_hf_data.sh`, `setup/prepare_tokenizer_lanta.sh`
4. modified code to not download eval dataset on the file because Lanta gpu node do not have internet connection.
   `setup/download_eval_datasets_lanta.py` and `setup/prepare_eval_data_lanta.sh` specifically made for Lanta.
5. Replace `lm-eval` with [Custom version of lm-eval](https://github.com/new5558/lm-evaluation-harness-lanta) that support loading dataset from disk
6. Added Fine-tuning tutorial and with additional support scripts: `setup/convert_hf_checkpoint.sh` and `setup/download_hf_model.sh`
7. Added Fine-tuning 1B Llama3 and Pre-traning configuration example.
8. Fixed Memory leak in evaluation steps
9. Fixed Multinode training on Lanta
10. Added custom dataset [dummy_zhth](https://huggingface.co/datasets/peerachet/dummy_zhth) support for the original data processing script.
11. Added support for Qwen2, Qwen2.5 Models Family

## Set up

#### Create Conda Environment

```sh
sh setup/create_env_lanta.sh <path_to_store_conda_environment>
```

- `<path_to_store_conda_environment>`: Recommend putting the environement path outside of lingua folder to avoid stool.py indexing when running train script.
- `<conda_path>` will be created at `<path_to_store_conda_environment>/lingua_conda`

#### Download training data from Huggingface

```sh
sh setup/download_prepare_hf_data.sh <conda_path> <data_repo>
```

- `<data_repo>` Can be one of `"fineweb_edu", "fineweb_edu_10bt", "dclm_baseline_1.0", "dclm_baseline_1.0_10prct", "dummy_zhth"` Please choose `fineweb_edu_10bt` or `dummy_zhth` for this tutorial because it use less disk space.
- Training dataset will be downloaded to `<current_directory>/data/<data_repo>_shuffled`

#### Download tokenizer from Huggingface

```sh
sh setup/prepare_tokenizer_lanta.sh <conda_path> <huggingface_privatekey> <tokenizer_name>
```

- Get Hugginface private key from this [link](https://huggingface.co/settings/tokens)
- `<tokenizer_name>` Can be on what listed in `setup/download_tokenizer.py` `llama2, llama3, gemma, qwen2`
- Tokenizer will be saved to `<tokenizer_path>` = `<current_directory>/tokenizer_file_<tokenizer_name>`

#### Download eval data from Huggingface

```sh
sh setup/prepare_eval_data_lanta.sh <conda_path>
```

- Eval datasets will be downloaded to `<current_directory>/data/<dataset_name>` each datasets will have one folder

#### Activate Conda Environment

```sh
ml purge
ml Mamba/23.11.0-0
conda deactivate
conda activate <conda_path>
```

### Pre-training

Edit `lanta_pretrain.yaml` and run slurm job

```sh
python -m lingua.stool script=apps.main.train config=apps/main/configs/lanta_pretrain.yaml nodes=<num_nodes> partition=gpu project_name=<project_name> time=02:00:00
```

### Fine-tuning

#### Download checkpoint from Huggingface

```sh
sh setup/download_hf_model.sh <conda_path> <REPO_ID> <DOWNLOAD_PATH>
```

- `<REPO_ID>` should be a varaint of Llama3 models family. In this demo we will use [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- `<DOWNLOAD_PATH>` is where the model will be downloaded into

#### Convert Checkpoint to DCP [DCP](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) format

```sh
sh setup/convert_hf_checkpoint.sh <conda_path> <DOWNLOAD_PATH> <MODEL_FAMMILY>
```

- `<DOWNLOAD_PATH>` is the same path we downloaded checkpoint from Huggingface.
- `<MODEL_FAMMILY>` Must be one of `"llama3" or "qwen2"`
- Output will be at `<DOWNLOAD_PATH>-converted`

#### Edit `lanta_finetune_1B.yaml` (Llama3.2 1B) or `lanta_finetune_1B_qwen.yaml` (Qwen2.5 1.5B) and run slurm job

```sh
python -m lingua.stool script=apps.main.train config=apps/main/configs/lanta_finetune_1B.yaml nodes=<num_nodes> partition=gpu project_name=<project_name> time=02:00:00
```

or

```sh
python -m lingua.stool script=apps.main.train config=apps/main/configs/lanta_finetune_1B_qwen.yaml nodes=<num_nodes> partition=gpu project_name=<project_name> time=02:00:00
```

### Convert Checkpoint to Hugggingface Format

#### Llama3

We need to run checkpoint conversion in SLURM to avoid OOM.

1. Edit `<PROJECT_NAME>` in `setup/convert_dcp_checkpoint_llama.sh`
2. Run Command

```sh
sbatch setup/convert_dcp_checkpoint_llama.sh \
   <CONDA_PATH> \
   <LINGUA_CHECKPOINT_PATH> \
   <TOKENIZER_DIR> \
```

- `<LINGUA_CHECKPOINT_PATH>` example: <full_path>/checkpoints/0000000300
- `<TOKENIZER_DIR>` example: <full_path>/tokenizer_file_llama3/original/

#### Qwen2

We need to run checkpoint conversion in SLURM to avoid OOM.

1. Edit `<PROJECT_NAME>` in `setup/convert_dcp_checkpoint_qwen.sh`
2. Run Command

```sh
sbatch setup/convert_dcp_checkpoint_qwen.sh \
   <CONDA_PATH> \
   <LINGUA_CHECKPOINT_PATH> \
   <TOKENIZER_DIR> \
   <ORIGINAL_QWEN_HF_PATH>
```

- `<LINGUA_CHECKPOINT_PATH>` example: <full_path>/checkpoints/0000000300
- `<TOKENIZER_DIR>` example: <full_path>/tokenizer_file_qwen2
- `<ORIGINAL_QWEN_HF_PATH>` full path to original qwen huggingface checkpoint example: <full_path>/Qwen2.5-1.5B

### Upload to Huggingface

Prerequisite: Convert Checkpoint to Huggingface Format

```sh
huggingface-cli upload --token <HF_TOKEN> <PUSH_HF_REPO> <LINGUA_CHECKPOINT_PATH>/hf .
```

- `<LINGUA_CHECKPOINT_PATH>` example: <full_path>/checkpoints/0000000300
- `<PUSH_HF_REPO>` hf repository example: lst-nectec/llama-1b-finetuned
