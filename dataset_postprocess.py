import numpy as np
import os
import json
import shutil
import random
from pathlib import Path
import argparse

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640


def convert_npy_to_yolo_txt(directory_path):
    for filename in os.listdir(directory_path):
        if not filename.endswith(".npy"):
            continue

        np_filename = os.path.splitext(filename)[0]
        numeric_part = np_filename[-4:]
        json_filename = np_filename[:-4] + "labels_" + numeric_part + ".json"

        json_path = os.path.join(directory_path, json_filename)
        if not os.path.exists(json_path):
            print(f"⚠️ Missing JSON for {filename}")
            continue

        bb_info = np.load(os.path.join(directory_path, filename), allow_pickle=True)

        yolo_path = os.path.join(directory_path, f"rgb_{numeric_part}.txt")
        open(yolo_path, "w").close()

        for item in bb_info:
            semantic_id = int(item["semanticId"])
            x_min, y_min = item["x_min"], item["y_min"]
            x_max, y_max = item["x_max"], item["y_max"]

            x_center = (x_min + x_max) / (2 * IMAGE_WIDTH)
            y_center = (y_min + y_max) / (2 * IMAGE_HEIGHT)
            width = (x_max - x_min) / IMAGE_WIDTH
            height = (y_max - y_min) / IMAGE_HEIGHT

            with open(yolo_path, "a") as f:
                f.write(
                    f"{semantic_id} {x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f}\n"
                )

    print("✅ YOLO label generation complete")


def split_yolo_dataset(dataset_dir, output_dir, train_ratio=0.8, move_files=True):
    images_out = os.path.join(output_dir, "images")
    labels_out = os.path.join(output_dir, "labels")

    for split in ["train", "val"]:
        Path(os.path.join(images_out, split)).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(labels_out, split)).mkdir(parents=True, exist_ok=True)

    images = [f for f in os.listdir(dataset_dir) if f.endswith(".png")]
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train, val = images[:split_idx], images[split_idx:]

    def handle(files, split):
        for img in files:
            lbl = os.path.splitext(img)[0] + ".txt"

            src_img = os.path.join(dataset_dir, img)
            src_lbl = os.path.join(dataset_dir, lbl)

            dst_img = os.path.join(images_out, split, img)
            dst_lbl = os.path.join(labels_out, split, lbl)

            shutil.move(src_img, dst_img)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)

    handle(train, "train")
    handle(val, "val")

    print("✅ Dataset split complete")
    
def main():
    parser = argparse.ArgumentParser(description="Postprocess the dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the processed dataset")
    
    args = parser.parse_args()
    
    # Use args.dataset_dir and args.output_dir here for processing
    convert_npy_to_yolo_txt(args.dataset_dir)
    split_yolo_dataset(args.dataset_dir, args.output_dir)

if __name__ == "__main__":
    main()

