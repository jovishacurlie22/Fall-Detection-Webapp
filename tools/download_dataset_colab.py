# ============================================================
# Download UR Fall Dataset directly into Google Drive
# Works inside Google Colab
# ============================================================

import os
import zipfile
import subprocess

# CHANGE if needed
KAGGLE_DATASET = "shahliza27/ur-fall-detection-dataset"

# Save dataset directly to Drive storage
BASE_DIR = "/content/drive/MyDrive/fall_project"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")


def download_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)

    print("\n📥 Downloading dataset from Kaggle...\n")

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        DATASET_DIR
    ]

    subprocess.run(cmd, check=True)
    print("✅ Download finished")


def extract_dataset():
    print("\n📦 Extracting dataset...\n")

    for file in os.listdir(DATASET_DIR):
        if file.endswith(".zip"):
            path = os.path.join(DATASET_DIR, file)

            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(DATASET_DIR)

            print(f"✅ Extracted {file}")


if __name__ == "__main__":
    download_dataset()
    extract_dataset()

    print("\n🎉 Dataset stored in Google Drive!")
