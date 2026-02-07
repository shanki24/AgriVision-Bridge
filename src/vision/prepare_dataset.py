import os
import shutil
import random
from tqdm import tqdm


# PATHS (Windows-safe)

RAW_DIR = r"C:\Users\Sujal\OneDrive\Desktop\AgriVision-Bridge\data\raw\plantvillage\images"
OUTPUT_DIR = r"C:\Users\Sujal\OneDrive\Desktop\AgriVision-Bridge\data\processed\plantvillage"
TRAIN_RATIO = 0.8


# CREATE OUTPUT DIRS

train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)


# SPLIT DATASET

for cls in tqdm(os.listdir(RAW_DIR), desc="Processing classes"):
    cls_path = os.path.join(RAW_DIR, cls)

    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(train_dir, cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(val_dir, cls, img)
        )

print("Dataset split completed successfully")
