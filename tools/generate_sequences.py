# ============================================================
# Generate Sequences from UR Dataset (FRAME VERSION)
# ============================================================

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append('/content/Fall-Detection-Webapp')

from app.pose_estimator import PoseEstimator

# 📁 Paths
DATASET_PATH = "/content/drive/MyDrive/fall_project/dataset/UR_fall_detection_dataset_cam0_rgb"
OUTPUT_PATH  = "/content/drive/MyDrive/fall_project/dataset_sequences"

SEQUENCE_LENGTH = 30


def process_folder(folder_path):

    pose = PoseEstimator()
    sequence = []
    count = 0

    # Detect label from folder name
    name = folder_path.name.lower()

    if "fall" in name:
        label = "fall"
    else:
        label = "nonfall"

    print(f"\n📂 Processing {folder_path.name} → {label}")

    images = sorted(list(folder_path.glob("*.png")) + 
                    list(folder_path.glob("*.jpg")))

    for img_path in images:

        frame = cv2.imread(str(img_path))

        if frame is None:
            continue

        frame, keypoints, _ = pose.process(frame)

        if keypoints is not None:
            sequence.append(keypoints)

        if len(sequence) == SEQUENCE_LENGTH:

            save_dir = Path(OUTPUT_PATH, label)
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / f"{folder_path.name}_{count}.npy"

            np.save(save_path, np.array(sequence))

            sequence.pop(0)
            count += 1

    pose.release()

    print(f"✅ Saved {count} sequences")


def run():

    folders = list(Path(DATASET_PATH).glob("*"))

    print(f"\n📊 Total folders: {len(folders)}")

    for folder in tqdm(folders):
        if folder.is_dir():
            process_folder(folder)


if __name__ == "__main__":
    print("🚀 Generating sequences from image folders...\n")
    run()
    print("\n🎉 Done!")
