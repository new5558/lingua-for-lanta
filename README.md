## Lingua Refectored for Lanta

This is lingua refactored for Lanta HPC Cluster.
The original readme of this repository [HERE](README_original.md)

### Changes log

1. Removed `unlimit` command from code because Lanta don't have permission to change shell resource limit
2. modified script to create environement specifically made for Lanta: `create_env_lanta.sh`
3. created script to download training data and tokenizer `setup/prepare_data_lanta.sh`, `setup/prepare_tokenizer_lanta.sh`
4. modify code to not download eval dataset on the file because Lanta gpu node do not have internet connection.
   `setup/download_eval_datasets_lanta.py` and `setup/prepare_eval_data_lanta.sh` specifically made for Lanta.
5. Used [Custom version of lm-eval](https://github.com/new5558/lm-evaluation-harness-lanta) that support loading dataset from disk

### Set up

```
sh setup/create_env_lanta.sh <path_to_store_conda_environment>
```

- `<path_to_store_conda_environment>`: Recommend putting the environement path outside of lingua folder to avoid stool.py indexing when running train script.

Activate Conda Environment:

```
conda activate <path_to_store_conda_environment>
```

<!-- Install additional depedencies

```
pip install huggingface-hub
``` -->

Download training data from Huggingface:

```
sh setup/prepare_data_lanta.sh <path_to_store_conda_environment>
```

Download tokenizer from Huggingface:

```
sh setup/prepare_tokenizer_lanta.sh <path_to_store_conda_environment> <huggingface_privatekey>
```

Download eval data from Huggingface:

```
sh setup/prepare_eval_data_lanta.sh <path_to_store_conda_environment>
```

Edit `lanta_pretrain.yaml` and run slurm job

```
python -m lingua.stool script=apps.main.train config=apps/main/configs/lanta_pretrain.yaml nodes=1 partition=gpu project_name=<project_name> time=02:00:00
```

python -m lingua.stool script=apps.main.train config=apps/main/configs/lanta_pretrain.yaml nodes=1 partition=gpu project_name=lt200304 time=02:00:00

python setup/download_prepare_hf_data.py fineweb_edu <MEMORY> --data_dir ./data --seed 42 --nchunks <NCHUNKS>

python setup/download_tokenizer.py llama3 <SAVE_PATH> --api_key <HUGGINGFACE_TOKEN>

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

conda activate /project/lt200304-dipmt/new_norapat/.conda_lingua_lanta/lingua_conda
