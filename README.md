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
6. Add Fine-tuning tutorial and with additional support scripts: `setup/convert_hf_checkpoint.sh` and `setup/download_hf_model.sh`
7. Add Fine-tuning 1B Llama3 and Pre-traning configuration example.

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

- `<data_repo>` Can be one of `"fineweb_edu", "fineweb_edu_10bt", "dclm_baseline_1.0", "dclm_baseline_1.0_10prct"` Please choose `fineweb_edu_10bt` for this tutorial because it use less disk space.
- Training dataset will be downloaded to `<current_directory>/data/<data_repo>_shuffled`

#### Download tokenizer from Huggingface

```sh
sh setup/prepare_tokenizer_lanta.sh <conda_path> <huggingface_privatekey>
```

- Get Hugginface private key from this [link](https://huggingface.co/settings/tokens)
- Tokenizer will be saved to `<tokenizer_path>` = `<current_directory>/tokenizer_file`

#### Download eval data from Huggingface

```sh
sh setup/prepare_eval_data_lanta.sh <conda_path>
```

- Eval datasets will be downloaded to `<current_directory>/data/<dataset_name>` each datasets will have one folder

### Pre-training

Edit `lanta_pretrain.yaml` and run slurm job

```sh
python -m lingua.stool script=apps.main.train config=apps/main/configs/lanta_pretrain.yaml nodes=<num_nodes> partition=gpu project_name=<project_name> time=02:00:00
```

### Fine-tuning

Download checkpoint from Huggingface

```sh
sh setup/download_llama_hf.sh <REPO_ID> <DOWNLOAD_PATH>
```

`REPO_ID` should be a varaint of Llama3 models family. In this demo we will use [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)

Convert Checkpoint to [DCP](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) format

```sh
sh setup/convert_hf_checkpoint.sh <DOWNLOAD_PATH>
```

`DOWNLOAD_PATH` is the same path we downloaded checkpoint from Huggingface.
Output will be at `<DOWNLOAD_PATH>-converted`

Edit `lanta_finetune_1B.yaml` and run slurm job

```sh
python -m lingua.stool script=apps.main.train config=apps/main/configs/lanta_finetune_1B.yaml nodes=<num_nodes> partition=gpu project_name=<project_name> time=02:00:00
```

#### Activate Conda Environment

```sh
ml purge
ml Mamba/23.11.0-0
conda deactivate
conda activate <path_to_store_conda_environment>
```
