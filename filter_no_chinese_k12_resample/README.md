# Filter K12 Resample (Remove Chinese Characters)

Tools to:
1. Filter out rows whose images contain Chinese characters from `Jakumetsu/k12-resample` using a YOLO detection model trained for Chinese characters.
2. Export the filtered dataset (saved with `datasets.save_to_disk`) to a CSV + PNG images.
3. Sync (push) the resulting filtered dataset to the Hugging Face Hub.

## Contents
| Script | Purpose |
|--------|---------|
| `filter_chinese.py` | Loads the source dataset, runs YOLO per image list, retains rows with no detected Chinese characters, saves result to disk. |
| `export_to_csv.py` | Loads the locally saved filtered dataset and exports images as PNG plus a `data.csv` manifest. |
| `sync_dataset.py` | Loads the filtered dataset from disk and pushes it to a new / existing HF dataset repo. |

## Prerequisites
- Python 3.9+
- Packages: `datasets`, `ultralytics`, `Pillow`, `huggingface_hub`
- YOLO model weights file: `yolo_chinese_m.pt` (place alongside the scripts or adjust the path)
- Sufficient disk space for the dataset and filtered copy.

Install (example):
```bash
pip install datasets ultralytics pillow huggingface_hub
```

## 1. Filtering
Adjust constants at the top of `filter_chinese.py` if needed:
- `DATASET_NAME`
- `YOLO_MODEL_PATH`
- `OUTPUT_DATASET_PATH`
- `DATASET_SPLIT`

Run:
```bash
python filter_chinese.py
```
Result: Saved dataset directory at `OUTPUT_DATASET_PATH` (e.g. `./filtered_k12_resample_no_chinese`).

## 2. Export to CSV + PNG
Update the call in `export_to_csv.py` if your output path differs.
```bash
python export_to_csv.py
```
Creates:
- `output/images/` (PNG files named `<row_id>_<idx>.png`)
- `output/data.csv`

## 3. Push to Hugging Face Hub
Edit in `sync_dataset.py`:
- `LOCAL_FILTERED_DATASET_PATH`
- `HF_REPO_ID` (set your username, e.g. `myname/k12-resample-no-chinese-filtered`)
- `IS_PRIVATE_REPO`

Login first if not already:
```bash
huggingface-cli login
```
Then:
```bash
python sync_dataset.py
```

## Notes & Tips
- The YOLO script treats any detection as a Chinese character. For multi‑class models, inspect `r.boxes.cls` and filter by class ID.
- Large datasets: consider batching or limiting rows if memory constrained.
- Repro: store the exact YOLO weights checksum.

## Troubleshooting
| Issue | Cause / Fix |
|-------|-------------|
| "model file not found" | Place `yolo_chinese_m.pt` in folder or change path. |
| OOM / slow inference | Use a smaller YOLO model, run on GPU, or resize images. |
| Empty filtered dataset | Model may be over‑detecting; verify with sample images. |
| Push auth error | Run `huggingface-cli login` or check `HF_REPO_ID`. |

## License
See repository root `LICENSE`.
