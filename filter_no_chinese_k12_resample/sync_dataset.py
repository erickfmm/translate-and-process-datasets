import os
import argparse
from datasets import load_from_disk
from huggingface_hub import HfApi, create_repo

def push_dataset(local_path: str, repo_id: str, private: bool) -> None:
    if not os.path.exists(local_path):
        print(f"Error: Local filtered dataset not found at '{local_path}'.")
        return
    print(f"Loading filtered dataset from: {local_path}")
    try:
        filtered_dataset = load_from_disk(local_path)
        print(f"Loaded dataset with {len(filtered_dataset)} rows.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    api = HfApi()
    print(f"Ensuring remote repository: {repo_id}")
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"Error creating/accessing repository: {e}")
        return

    print("Pushing dataset to hub ... this may take a while")
    try:
        filtered_dataset.push_to_hub(repo_id)
        print("Success! View at: https://huggingface.co/datasets/" + repo_id)
    except Exception as e:
        print(f"Error during push: {e}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Push a locally saved filtered HF dataset to the Hugging Face Hub.")
    p.add_argument('--local-path', default='./filtered_k12_resample_no_chinese', help='Path to local saved dataset directory')
    p.add_argument('--repo-id', required=True, help='Target dataset repo id: username/dataset-name')
    p.add_argument('--private', action='store_true', help='Create as private (default public)')
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    push_dataset(local_path=args.local_path, repo_id=args.repo_id, private=args.private)


if __name__ == "__main__":
    main()