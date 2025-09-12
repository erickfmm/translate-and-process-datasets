import os
import csv
from datasets import load_from_disk
from PIL import Image

def export_images_with_png(dataset_path, output_dir="output"):
    ds = load_from_disk(dataset_path)
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "data.csv")

    fieldnames = ["id", "image_filenames", "problem", "answer", "pass"]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in ds:
            row_id = row.get("id")
            imgs = row.get("images", [])
            if not imgs:
                continue

            filenames = []
            for idx, img in enumerate(imgs):
                img_filename = f"{row_id}_{idx}.png"

                # Convert to RGBA if necessary (PNG supports transparency)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA")

                img.save(os.path.join(images_dir, img_filename))
                filenames.append(img_filename)

            writer.writerow({
                "id": row_id,
                "image_filenames": "||".join(filenames),
                "problem": row.get("problem", "").replace("\n", " "),
                "answer": row.get("answer", ""),
                "pass": row.get("pass", "")
            })

    print(f"Export completed: images in '{images_dir}', CSV at '{csv_path}'")

if __name__ == "__main__":
    export_images_with_png("./filtered_k12_resample_no_chinese")