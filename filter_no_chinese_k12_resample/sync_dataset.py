import os
from datasets import load_from_disk
from huggingface_hub import HfApi, create_repo

# --- Configuration ---
# The local path where your filtered dataset is saved.
# This should match the OUTPUT_DATASET_PATH from your filtering script.
LOCAL_FILTERED_DATASET_PATH = "./filtered_k12_resample_no_chinese"

# The name of the new repository on Hugging Face Hub where you want to upload the dataset.
# It will be created under your Hugging Face username (e.g., "your-username/my-filtered-dataset").
# Choose a descriptive name!
HF_REPO_ID = "your-username/k12-resample-no-chinese-filtered" # IMPORTANT: Change 'your-username' to your actual Hugging Face username!

# Set to True if you want the repository to be private, False for public.
IS_PRIVATE_REPO = False

# --- Main Program Logic ---
def main():
    print(f"--- Starting Dataset Sync Program ---")

    # 1. Check if the local filtered dataset exists
    if not os.path.exists(LOCAL_FILTERED_DATASET_PATH):
        print(f"Error: Local filtered dataset not found at '{LOCAL_FILTERED_DATASET_PATH}'.")
        print("Please ensure the filtering script has been run successfully and the path is correct.")
        return

    # 2. Load the filtered dataset from disk
    print(f"Loading filtered dataset from: '{LOCAL_FILTERED_DATASET_PATH}'")
    try:
        filtered_dataset = load_from_disk(LOCAL_FILTERED_DATASET_PATH)
        print(f"Filtered dataset loaded successfully with {len(filtered_dataset)} examples.")
    except Exception as e:
        print(f"Error loading filtered dataset from disk: {e}")
        return

    # 3. Initialize Hugging Face API
    api = HfApi()

    # 4. Create a new repository on Hugging Face Hub (if it doesn't exist)
    print(f"Attempting to create/get repository: '{HF_REPO_ID}'")
    try:
        # Check if repo exists, if not, create it
        # Note: create_repo handles existing repos gracefully if exist_ok=True
        create_repo(repo_id=HF_REPO_ID, repo_type="dataset", private=IS_PRIVATE_REPO, exist_ok=True)
        print(f"Repository '{HF_REPO_ID}' ensured on Hugging Face Hub.")
    except Exception as e:
        print(f"Error creating/accessing Hugging Face repository: {e}")
        print("Please ensure you are logged in via 'huggingface-cli login' and have correct permissions.")
        return

    # 5. Push the filtered dataset to the Hugging Face Hub
    print(f"\nPushing filtered dataset to '{HF_REPO_ID}'...")
    try:
        # The push_to_hub method handles uploading the dataset files
        filtered_dataset.push_to_hub(HF_REPO_ID)
        print("\nDataset successfully pushed to Hugging Face Hub!")
        print(f"You can view it here: https://huggingface.co/datasets/{HF_REPO_ID}")
    except Exception as e:
        print(f"Error pushing dataset to Hugging Face Hub: {e}")
        print("Common issues: not logged in, incorrect repo ID, network problems.")

    print(f"\n--- Program Finished ---")

if __name__ == "__main__":
    main()