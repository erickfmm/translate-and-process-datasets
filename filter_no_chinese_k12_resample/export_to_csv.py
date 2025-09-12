import os
import csv
import argparse
from typing import Optional
from datasets import load_from_disk
from PIL import Image

def export_images(dataset_path: str,
                  output_dir: str = "output",
                  csv_name: str = "data.csv",
                  image_format: str = "png",
                  overwrite: bool = True,
                  limit: Optional[int] = None) -> None:
    ds = load_from_disk(dataset_path)
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_name)

    if not overwrite and os.path.exists(csv_path):
        raise FileExistsError(f"File exists and overwrite disabled: {csv_path}")

    fieldnames = ["id", "image_filenames", "problem", "answer", "pass"]
    written = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in ds:
            if limit is not None and written >= limit:
                break
            row_id = row.get("id")
            imgs = row.get("images", [])
            if not imgs:
                continue
            filenames = []
            for idx, img in enumerate(imgs):
                img_filename = f"{row_id}_{idx}.{image_format.lower()}"
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA")
                img.save(os.path.join(images_dir, img_filename), format=image_format.upper())
                filenames.append(img_filename)
            writer.writerow({
                "id": row_id,
                "image_filenames": "||".join(filenames),
                "problem": row.get("problem", "").replace("\n", " "),
                "answer": row.get("answer", ""),
                "pass": row.get("pass", "")
            })
            written += 1
            if written % 100 == 0:
                print(f"Exported {written} rows...")

    print(f"Export completed: {written} rows | images in '{images_dir}' | CSV at '{csv_path}'")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export images + metadata from a saved Hugging Face dataset directory.")
    p.add_argument('--dataset-path', default='./filtered_k12_resample_no_chinese', help='Path to saved dataset (load_from_disk)')
    p.add_argument('--output-dir', default='output', help='Directory to write images and CSV')
    p.add_argument('--csv-name', default='data.csv', help='CSV file name')
    p.add_argument('--image-format', default='png', choices=['png', 'jpg', 'jpeg'], help='Image output format')
    p.add_argument('--no-overwrite', action='store_true', help='Do not overwrite existing CSV')
    p.add_argument('--limit', type=int, default=None, help='Optional limit of rows to export (debug)')
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    export_images(dataset_path=args.dataset_path,
                  output_dir=args.output_dir,
                  csv_name=args.csv_name,
                  image_format=args.image_format,
                  overwrite=not args.no_overwrite,
                  limit=args.limit)


if __name__ == "__main__":
    main()