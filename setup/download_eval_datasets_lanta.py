# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import time
import requests
from datasets import load_dataset


def download_dataset(repo_id, local_dir):
    print(f"Downloading dataset from {repo_id}...")
    
    dataset = load_dataset(repo_id)
    dataset.save_to_disk(local_dir)
    print(f"Dataset downloaded to {local_dir}")


def main(dataset, data_dir):
    # Configuration
    src_dir = f"{data_dir}/{dataset}"

    # Download dataset
    download_dataset(dataset, src_dir)

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--data_dir", type=str, default="data")

    args = parser.parse_args()

    main(args.dataset, args.data_dir)
