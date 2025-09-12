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

### CLI Options (`filter_chinese.py`)
All parameters are configurable via flags (run `--help`).

| Flag | Description | Default |
|------|-------------|---------|
| `--dataset-name` | Source HF dataset repository | `Jakumetsu/k12-resample` |
| `--split` | Dataset split | `train` |
| `--yolo-model-path` | Path to YOLO weights | `yolo_chinese_m.pt` |
| `--output-path` | Destination directory for filtered dataset | `./filtered_k12_resample_no_chinese` |
| `--max-rows` | Limit number of rows processed (debug) | None |
| `--progress-interval` | Rows between progress prints | 100 |

Example:
```bash
python filter_chinese.py --dataset-name Jakumetsu/k12-resample --yolo-model-path models/yolo_chinese_m.pt --max-rows 500
```

## 2. Export to CSV + PNG
Update the call in `export_to_csv.py` if your output path differs.
```bash
python export_to_csv.py
```
Creates:
- `output/images/` (PNG files named `<row_id>_<idx>.png`)
- `output/data.csv`

### CLI Options (`export_to_csv.py`)
| Flag | Description | Default |
|------|-------------|---------|
| `--dataset-path` | Path to saved dataset (load_from_disk) | `./filtered_k12_resample_no_chinese` |
| `--output-dir` | Output directory | `output` |
| `--csv-name` | CSV filename | `data.csv` |
| `--image-format` | `png` / `jpg` / `jpeg` | `png` |
| `--limit` | Limit number of rows exported | None |
| `--no-overwrite` | Prevent overwriting existing CSV | (off) |

Example:
```bash
python export_to_csv.py --dataset-path ./filtered_k12_resample_no_chinese --output-dir export --image-format jpg
```

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

### CLI Options (`sync_dataset.py`)
| Flag | Description | Default |
|------|-------------|---------|
| `--local-path` | Local filtered dataset directory | `./filtered_k12_resample_no_chinese` |
| `--repo-id` | Target dataset repo (required) | (none) |
| `--private` | Create as private (public if omitted) | (off) |

Example:
```bash
python sync_dataset.py --local-path ./filtered_k12_resample_no_chinese --repo-id yourname/k12-resample-no-chinese-filtered --private
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
