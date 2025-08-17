import os, cv2
from pathlib import Path

# ----------------------------
# 경로 설정
# ----------------------------
ROOT = Path(r"C:\summer\trafficlight-lite\datasets\etri_raw")  # 폴더명 summer 로 변경
PREVIEW_DIR = ROOT / "previews"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)  # previews 폴더 자동 생성

# 이미지 확장자
IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".BMP"}

# 이미지/라벨 매핑
imgs = {p.stem: p for p in ROOT.rglob("*") if p.suffix in IMG_EXTS}
lbls = sorted([p for p in ROOT.rglob("*.txt")])

samples = {}
for lb in lbls:
    with open(lb, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            ps = line.strip().split()
            if len(ps) < 5:
                continue
            c = ps[0]
            if c not in samples:
                samples[c] = []
            if len(samples[c]) < 6:  # 클래스별 최대 6개
                samples[c].append(lb)

# ----------------------------
# 클래스별 프리뷰 생성
# ----------------------------
for cid in sorted(samples, key=lambda x: int(x)):
    print(f"Class {cid} -> showing up to 6")
    idxs = samples[cid]
    canvas, k = None, 0

    for lb in idxs:
        stem = lb.stem
        img = imgs.get(stem)
        if not img:
            print(f"  [WARN] 이미지 파일 없음: {stem}")
            continue

        im = cv2.imread(str(img))
        if im is None:
            print(f"  [WARN] 이미지 로드 실패: {img}")
            continue

        H, W = im.shape[:2]

        with open(lb, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ps = line.strip().split()
                if len(ps) < 5:
                    continue
                if ps[0] != cid:
                    continue

                cx, cy, w, h = map(float, ps[1:5])
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)

                crop = im[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
                if crop.size == 0:
                    continue

                crop = cv2.resize(crop, (160, 160))
                canvas = crop if canvas is None else cv2.hconcat([canvas, crop])

                k += 1
                if k == 6:
                    break
        if k == 6:
            break

    if canvas is not None:
        out = PREVIEW_DIR / f"class_{cid}_preview.jpg"
        cv2.imwrite(str(out), canvas)
        print("  saved:", out)
    else:
        print(f"  [INFO] Class {cid} 이미지 없음")
