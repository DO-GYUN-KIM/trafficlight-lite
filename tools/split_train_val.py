import os
import random
import shutil
from pathlib import Path

DATASET_DIR = Path(r"C:\summer\trafficlight-lite\datasets\etri_raw")
TRAIN = DATASET_DIR / "train"
VAL = DATASET_DIR / "val"

VAL_RATIO = 0.2       # train의 20%를 val로 이동(원하면 조절)
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def main():
    random.seed(SEED)
    (VAL / "images").mkdir(parents=True, exist_ok=True)
    (VAL / "labels").mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in (TRAIN / "images").iterdir() if p.suffix.lower() in IMG_EXTS])
    pairs = []
    for img in imgs:
        lbl = TRAIN / "labels" / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            print(f"[WARN] 라벨 없음: {img.name}")

    total = len(pairs)
    k = int(total * VAL_RATIO)
    val_pairs = set(random.sample(pairs, k))
    print(f"[INFO] train 총 {total}개, val로 이동 {k}개")

    moved = 0
    for img, lbl in val_pairs:
        shutil.move(str(img), str(VAL / "images" / img.name))
        shutil.move(str(lbl), str(VAL / "labels" / lbl.name))
        moved += 1
        if moved % 1000 == 0:
            print(f"[MOVE] {moved}/{k}")

    print("[DONE] val 분리 완료.")

if __name__ == "__main__":
    main()
