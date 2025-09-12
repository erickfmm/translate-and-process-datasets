import os
import argparse
from typing import Optional
from datasets import load_dataset, Dataset
from ultralytics import YOLO
from PIL import Image

# Default configuration (can be overridden via CLI)
DEFAULT_DATASET_NAME = "Jakumetsu/k12-resample"
DEFAULT_YOLO_MODEL_PATH = "yolo_chinese_m.pt"
DEFAULT_OUTPUT_DATASET_PATH = "./filtered_k12_resample_no_chinese"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_PROGRESS_INTERVAL = 100

# --- Function to Detect Chinese Characters ---
def detect_chinese_characters(image: Image.Image, model: YOLO) -> bool:
    """
    Detects Chinese characters in a given PIL Image using the provided YOLO model.

    Args:
        image (PIL.Image.Image): The input image to scan for characters.
        model (ultralytics.YOLO): The loaded YOLO model.

    Returns:
        bool: True if Chinese characters are detected, False otherwise.
              This function assumes that if the model detects any object,
              it's a Chinese character, given the model's specific training.
              If your model has multiple classes and you only want to filter
              based on a specific 'Chinese character' class, you would need
              to inspect `r.boxes.cls` and `model.names` to match the class ID.
    """
    # Run inference on the image
    results = model(image)

    # Iterate through the detection results
    for r in results:
        # If any bounding box is detected, it means a Chinese character
        # (or whatever the model was trained to detect) is present.
        if len(r.boxes) > 0:
            return True  # Chinese characters detected

    return False  # No Chinese characters detected


# --- Main Program Logic ---
def process_dataset(dataset_name: str,
                    split: str,
                    yolo_model_path: str,
                    output_path: str,
                    max_rows: Optional[int] = None,
                    progress_interval: int = DEFAULT_PROGRESS_INTERVAL) -> None:
    print("--- Starting Dataset Filtering Program ---")
    print(f"Dataset: {dataset_name} | Split: {split}")

    # 1. Download / load dataset
    try:
        dataset = load_dataset(dataset_name, split=split)
        print(f"Loaded dataset with {len(dataset)} examples.")
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        return

    # 2. Load YOLO model
    if not os.path.exists(yolo_model_path):
        print(f"Error: YOLO model file not found: {yolo_model_path}")
        return
    try:
        model = YOLO(yolo_model_path)
        print("YOLO model loaded.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    filtered_rows = []
    total_rows = len(dataset) if max_rows is None else min(len(dataset), max_rows)
    print(f"Processing up to {total_rows} rows...")

    for i, row in enumerate(dataset):
        if max_rows is not None and i >= max_rows:
            break
        if 'images' not in row:
            if (i + 1) % progress_interval == 0:
                print(f"Row {i+1}: missing 'images' column, skipped.")
            continue

        images = row['images']
        is_chinese = False
        for image in images:
            if not isinstance(image, Image.Image):
                try:
                    image = Image.open(image)
                except Exception as e:
                    print(f"Row {i+1}: could not open image: {e}")
                    continue
            if detect_chinese_characters(image, model):
                is_chinese = True
                break

        if not is_chinese:
            filtered_rows.append(row)

        if (i + 1) % progress_interval == 0 or (i + 1) == total_rows:
            print(f"Progress {i+1}/{total_rows} | Kept {len(filtered_rows)}")

    print("Finished scanning.")
    print(f"Total kept rows: {len(filtered_rows)}")

    if not filtered_rows:
        print("No rows without detections; nothing to save.")
        return

    print("Creating filtered dataset object...")
    filtered_dataset = Dataset.from_list(filtered_rows)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(f"Saving to: {output_path}")
    try:
        filtered_dataset.save_to_disk(output_path)
        print("Saved filtered dataset.")
        print(f"Load later with: from datasets import load_from_disk; ds = load_from_disk('{output_path}')")
    except Exception as e:
        print(f"Error saving dataset: {e}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filter a dataset by removing rows whose images contain Chinese characters detected by a YOLO model.")
    p.add_argument('--dataset-name', default=DEFAULT_DATASET_NAME, help='HF dataset name (repo[/config])')
    p.add_argument('--split', default=DEFAULT_DATASET_SPLIT, help='Dataset split to load (default: train)')
    p.add_argument('--yolo-model-path', default=DEFAULT_YOLO_MODEL_PATH, help='Path to YOLO model weights file')
    p.add_argument('--output-path', default=DEFAULT_OUTPUT_DATASET_PATH, help='Directory path to save filtered dataset')
    p.add_argument('--max-rows', type=int, default=None, help='Optional: limit number of rows to process (debugging)')
    p.add_argument('--progress-interval', type=int, default=DEFAULT_PROGRESS_INTERVAL, help='How often to print progress (rows)')
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    process_dataset(dataset_name=args.dataset_name,
                    split=args.split,
                    yolo_model_path=args.yolo_model_path,
                    output_path=args.output_path,
                    max_rows=args.max_rows,
                    progress_interval=args.progress_interval)

if __name__ == "__main__":
    main()