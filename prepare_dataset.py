"""
Dataset hazırlama scripti.
images/ klasöründeki hayvan resimlerini %80 train / %20 val olarak böler
ve animal-dataset/ klasörüne kopyalar (YOLOv8 classification formatı).
"""
import shutil
import random
from pathlib import Path

random.seed(42)

SOURCE_DIR = Path("images")
DEST_DIR   = Path("animal-dataset")
CLASSES    = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'horse', 'sheep']
TRAIN_RATIO = 0.80


def prepare_dataset():
    print("=" * 55)
    print("  Dataset Hazırlanıyor (YOLOv8 Classification Format)")
    print("=" * 55)

    for split in ['train', 'val']:
        for cls in CLASSES:
            (DEST_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    total_train = total_val = 0

    for cls in CLASSES:
        src_folder = SOURCE_DIR / cls
        if not src_folder.exists():
            print(f"  UYARI: {src_folder} bulunamadı, atlanıyor...")
            continue

        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']:
            images.extend(src_folder.glob(ext))

        # Deduplicate (Windows glob is case-insensitive, avoid double-counting)
        seen = set()
        unique_images = []
        for p in images:
            key = p.name.lower()
            if key not in seen:
                seen.add(key)
                unique_images.append(p)
        images = unique_images

        if not images:
            print(f"  UYARI: {cls} klasöründe resim bulunamadı")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:split_idx]
        val_imgs   = images[split_idx:]

        copied_train = copied_val = 0
        for img in train_imgs:
            dst = DEST_DIR / "train" / cls / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
            copied_train += 1

        for img in val_imgs:
            dst = DEST_DIR / "val" / cls / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
            copied_val += 1

        print(f"  {cls:12s}: {copied_train:4d} train  |  {copied_val:4d} val")
        total_train += copied_train
        total_val   += copied_val

    print("-" * 55)
    print(f"  TOPLAM   : {total_train:4d} train  |  {total_val:4d} val")
    print(f"  Konum    : {DEST_DIR.resolve()}")
    print("=" * 55)
    print("Dataset hazır! Şimdi: python train_classifier.py")


if __name__ == "__main__":
    prepare_dataset()
