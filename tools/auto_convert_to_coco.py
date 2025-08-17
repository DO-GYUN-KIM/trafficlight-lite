# C:\trafficlight-lite\tools\auto_convert_to_coco.py
import os, json, shutil, argparse, random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".BMP"}
LBL_EXTS = {".txt"}

FINAL_CLASSES = ["red","yellow","green","off"]   # 최종 4클래스
COLOR_TO_ID = {c:i for i,c in enumerate(FINAL_CLASSES)}

# ★ 먼저는 '리포트 모드'로 돌려서 class id 분포를 확인하고,
#   아래 매핑 딕셔너리를 채운 뒤 '변환 모드'로 돌리는 걸 권장.
CLASS_ID_TO_COLOR = {
    # 예) 0:"red", 1:"yellow", 2:"green", 3:"off",
    # 모르면 일단 비워두고 --report로 돌려보고 채워넣기
}

def walk_pairs(root: Path):
    """재귀적으로 이미지/라벨 페어 찾기: 같은 stem을 가진 (img, txt) 반환"""
    files = list(root.rglob("*"))
    imgs = {}
    lbls = {}
    for p in files:
        if p.suffix in IMG_EXTS:
            imgs[p.stem] = p
        elif p.suffix in LBL_EXTS:
            lbls[p.stem] = p
    # 교집합만
    stems = sorted(set(imgs.keys()) & set(lbls.keys()))
    return [(imgs[s], lbls[s]) for s in stems]

def is_yolo_normalized(coords):
    """coords: [cx,cy,w,h] floats -> True if looks like 0~1 normalized"""
    return all(0.0 <= c <= 1.0 for c in coords)

def decide_split(stem: str):
    h = abs(hash(stem)) % 10
    if h < 8: return "train"
    elif h < 9: return "val"
    else: return "test"

def save_jpg(src_img: Path, dst_img: Path):
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    if src_img.suffix.lower() == ".bmp":
        im = Image.open(src_img).convert("RGB")
        im.save(dst_img, quality=95)
    else:
        shutil.copy2(src_img, dst_img)

def report_classes(pairs):
    """class id 빈도 리포트 (변환 전 탐색용)"""
    from collections import Counter
    cnt = Counter()
    sample_lines = {}
    for img, lbl in tqdm(pairs, desc="scan"):
        with open(lbl, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls = parts[0]
                cnt[cls]+=1
                if cls not in sample_lines:
                    sample_lines[cls] = line.strip()
    print("\n[CLASS REPORT]")
    for k,v in sorted(cnt.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0]):
        print(f"class {k}: {v} boxes | e.g. '{sample_lines[k]}'")
    print("\n위 리포트를 보고 CLASS_ID_TO_COLOR 딕셔너리를 채워서 다시 실행하세요.\n")

def convert(root: Path, out_dir: Path):
    pairs = walk_pairs(root)
    out_img = out_dir/"images"
    out_ann = out_dir/"annotations"
    (out_img/"train").mkdir(parents=True, exist_ok=True)
    (out_img/"val").mkdir(parents=True, exist_ok=True)
    (out_img/"test").mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    # COCO 틀
    coco = {
        "train": {"images": [], "annotations": [], "categories": []},
        "val":   {"images": [], "annotations": [], "categories": []},
        "test":  {"images": [], "annotations": [], "categories": []},
    }
    for i,c in enumerate(FINAL_CLASSES):
        cat = {"id": i, "name": c}
        for s in ["train","val","test"]:
            coco[s]["categories"].append(cat)

    img_id = {"train":1,"val":1,"test":1}
    ann_id = {"train":1,"val":1,"test":1}

    kept = 0
    for img, lbl in tqdm(pairs, desc="convert"):
        split = decide_split(img.stem)
        dst_img = out_img/split/f"{img.stem}.jpg"
        save_jpg(img, dst_img)
        W, H = Image.open(dst_img).size

        coco[split]["images"].append({
            "id": img_id[split],
            "file_name": f"{split}/{img.stem}.jpg",
            "width": W, "height": H
        })
        cur_img_id = img_id[split]
        img_id[split]+=1

        with open(lbl, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: 
                    continue
                cls_raw = parts[0]
                # class id를 int로 시도
                try:
                    cls_id = int(float(cls_raw))
                except:
                    # 문자열이면 여기서 매핑 규칙 추가 가능
                    continue
                if cls_id not in CLASS_ID_TO_COLOR:
                    # 관심 없는 클래스는 스킵
                    continue
                color = CLASS_ID_TO_COLOR[cls_id]
                cid = COLOR_TO_ID[color]

                cx, cy, bw, bh = map(float, parts[1:5])
                # 정규화 여부 판정
                if is_yolo_normalized([cx,cy,bw,bh]):
                    x = (cx - bw/2) * W
                    y = (cy - bh/2) * H
                    w = bw * W
                    h = bh * H
                else:
                    # 픽셀 좌표로 가정: parts = [cls, x, y, w, h, ...]
                    x, y, w, h = cx, cy, bw, bh

                if w <= 1 or h <= 1:
                    continue

                coco[split]["annotations"].append({
                    "id": ann_id[split],
                    "image_id": cur_img_id,
                    "category_id": cid,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w*h),
                    "iscrowd": 0
                })
                ann_id[split]+=1
        kept += 1

    for s in ["train","val","test"]:
        with open(out_ann/f"{s}.json", "w", encoding="utf-8") as f:
            json.dump(coco[s], f)
    print(f"\nDone. images processed: {kept}\nCOCO saved to: {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="압축 푼 최상위 폴더(etri_raw)")
    ap.add_argument("--dst", type=str, required=True, help="COCO 출력 폴더(etri_coco)")
    ap.add_argument("--report", action="store_true", help="변환 대신 클래스 리포트만 출력")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if args.report:
        pairs = walk_pairs(src)
        report_classes(pairs)
    else:
        if not CLASS_ID_TO_COLOR:
            raise SystemExit("CLASS_ID_TO_COLOR 비어있음. 먼저 --report 로 클래스 분포 확인 후 매핑을 채워주세요.")
        convert(src, dst)
