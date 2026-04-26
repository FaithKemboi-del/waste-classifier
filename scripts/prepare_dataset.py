import os
import shutil
import random
from pathlib import Path

# Your exact dataset path
SRC = Path(r"C:\Users\Faith\OneDrive\Documents\waste-classifier\Garbage classification\Garbage classification")

# Where to put the split dataset
DST = Path(r"C:\Users\Faith\OneDrive\Documents\waste-classifier\dataset")

# We only want these 4 classes (ignore glass, trash etc.)
CLASS_MAP = {
    "plastic":   "plastic",
    "paper":     "paper",
    "metal":     "metal",
    "cardboard": "organic",
}

SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED = 42

def prepare():
    random.seed(SEED)

    # Create destination folders
    for split in SPLITS:
        for label in set(CLASS_MAP.values()):
            (DST / split / label).mkdir(parents=True, exist_ok=True)

    for src_folder, label in CLASS_MAP.items():
        src_path = SRC / src_folder
        if not src_path.exists():
            print(f"[SKIP] '{src_folder}' not found — check folder name")
            continue

        images = sorted([
            f for f in src_path.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLITS["train"])
        n_val   = int(n * SPLITS["val"])

        buckets = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:],
        }

        for split, files in buckets.items():
            for f in files:
                # Add class prefix to avoid name collisions
                dst_name = f"{src_folder}_{f.name}"
                dst = DST / split / label / dst_name
                shutil.copy2(f, dst)

        print(f"[OK] {src_folder} → {label}: {n} images split into train/val/test")

    print("\nDone! Dataset ready at:", DST)

if __name__ == "__main__":
    prepare()