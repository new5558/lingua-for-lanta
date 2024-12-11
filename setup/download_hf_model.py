from huggingface_hub import snapshot_download
import argparse

def main(repo_id: str, local_dir: str) -> None:
    snapshot_download(repo_id=repo_id, local_dir=local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", type=str)
    parser.add_argument("local_dir", type=str)

    args = parser.parse_args()

    main(args.repo_id, args.local_dir)