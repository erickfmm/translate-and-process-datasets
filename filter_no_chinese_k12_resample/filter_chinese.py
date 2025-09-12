import os
from datasets import load_dataset, Dataset
from ultralytics import YOLO
from PIL import Image

# --- Configuration ---
# The name of the dataset to download from Hugging Face
DATASET_NAME = "Jakumetsu/k12-resample"
# The path to your YOLO model file.
# Make sure 'yolo_chinese_m.pt' is in the same directory as this script,
# or provide the full path to the file.
YOLO_MODEL_PATH = "yolo_chinese_m.pt"
# The directory where the filtered dataset will be saved locally.
OUTPUT_DATASET_PATH = "./filtered_k12_resample_no_chinese"
# The split of the dataset to process (e.g., "train", "validation", "test").
# Adjust this if your dataset uses a different split name.
DATASET_SPLIT = "train"

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
def main():
    print(f"--- Starting Dataset Filtering Program ---")

    # 1. Download the dataset
    print(f"Attempting to download dataset: '{DATASET_NAME}' split: '{DATASET_SPLIT}'")
    try:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        print(f"Dataset '{DATASET_NAME}' loaded successfully with {len(dataset)} examples.")
    except Exception as e:
        print(f"Error downloading or loading dataset: {e}")
        print("Please check the dataset name and your internet connection.")
        return

    # 2. Load the YOLO model
    print(f"Attempting to load YOLO model from: '{YOLO_MODEL_PATH}'")
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: YOLO model file '{YOLO_MODEL_PATH}' not found.")
        print("Please ensure the model file is in the correct directory or provide its full path.")
        return
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Ensure 'ultralytics' is installed and the model file is valid.")
        return

    filtered_rows = []
    total_rows = len(dataset)
    print(f"\nProcessing {total_rows} rows from the dataset...")

    # 3. Go through each row and detect Chinese characters
    for i, row in enumerate(dataset):
        # Check if the row contains an 'image' column
        if 'images' not in row:
            print(f"Warning: Row {i+1} does not contain an 'images' column. Skipping this row.")
            continue

        images = row['images']

        is_chinese = False
        for image in images:
            # Ensure the image is a PIL Image object
            if not isinstance(image, Image.Image):
                print(f"Warning: Image in row {i+1} is not a PIL Image. Attempting to open it (assuming it's a path).")
                try:
                    # If the 'image' column contains a path, try to open it
                    image = Image.open(image)
                except Exception as e:
                    print(f"Could not load image from row {i+1} (path: {row['image'] if isinstance(row['image'], str) else 'N/A'}): {e}. Skipping this row.")
                    continue

            # Detect Chinese characters in the image
            if detect_chinese_characters(image, model):
                is_chinese = True
                break
        if not is_chinese:
            filtered_rows.append(row) # Add row to filtered list if no Chinese characters are found

        # Print progress periodically
        if (i + 1) % 100 == 0 or (i + 1) == total_rows:
            print(f"Processed {i + 1}/{total_rows} rows. Found {len(filtered_rows)} rows without Chinese characters so far.")

    print(f"\nFinished processing all rows.")
    print(f"Total rows processed: {total_rows}")
    print(f"Total rows without Chinese characters: {len(filtered_rows)}")

    # 4. Save to another dataset the filtered rows
    if filtered_rows:
        print(f"\nCreating new dataset from filtered rows...")
        # Create a new Hugging Face Dataset object from the list of dictionaries
        filtered_dataset = Dataset.from_list(filtered_rows)

        print(f"Saving filtered dataset to local disk at: '{OUTPUT_DATASET_PATH}'")
        try:
            filtered_dataset.save_to_disk(OUTPUT_DATASET_PATH)
            print("Filtered dataset saved successfully!")
            print(f"You can load it later using: `from datasets import load_from_disk; filtered_data = load_from_disk('{OUTPUT_DATASET_PATH}')`")
        except Exception as e:
            print(f"Error saving filtered dataset: {e}")
    else:
        print("No rows were found without Chinese characters. No new dataset was saved.")

    print(f"\n--- Program Finished ---")

if __name__ == "__main__":
    main()