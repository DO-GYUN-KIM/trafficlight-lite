# C:\summer\trafficlight-lite\tools\video_trafficlight_system.py
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# Class / Group definitions (data.yaml 기준)
# =========================
ID_TO_NAME = {
    0: "green_straight",
    1: "green_left",
    2: "red",
    3: "red_left",
    4: "green_arrow",
    5: "yellow",
    6: "yellow_left",
    7: "countdown_display",
    8: "off_signal",
    9: "red_pedestrian",
    10: "green_pedestrian",
    11: "blue_straight",
    12: "red_alt",
    13: "pedestrian_warning",
}

GROUPS = {
    "RED":        {2, 3, 12, 9},
    "YELLOW":     {5, 6},
    "GREEN":      {0, 1, 4, 11, 10},
    "WARN":       {13},
    "COUNTDOWN":  {7},
    "OFF":        {8},
}
GROUP_PRIORITY = ["RED", "YELLOW", "GREEN", "WARN", "COUNTDOWN", "OFF"]


# =========================
# Caption (EN)
# =========================
def caption_text(group_name: str, top_cls: int) -> str | None:
    if group_name == "RED":
        if top_cls == 3:
            return "Red light / Left turn stop"
        return "Red light / Stop"
    if group_name == "GREEN":
        if top_cls == 1:
            return "Green light / Turn left"
        if top_cls == 4:
            return "Green light / Directional arrow"
        return "Green light / Go straight"
    if group_name == "YELLOW":
        if top_cls == 6:
            return "Yellow light / Caution (left)"
        return "Yellow light / Caution"
    if group_name == "WARN":
        return "Pedestrian warning"
    if group_name == "COUNTDOWN":
        return "Pedestrian countdown"
    return None


# =========================
# Utils
# =========================
def parse_args():
    p = argparse.ArgumentParser("Traffic-light system with trapezoid ROI and captions")
    # Basics
    p.add_argument("--weights", type=str, default="C:/summer/trafficlight-lite/best.pt")
    p.add_argument("--source", type=str, default="C:/summer/trafficlight-lite/test.mp4")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.55)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--name", type=str, default="exp")
    # Filters
    p.add_argument("--debounce_ms", type=int, default=800)
    p.add_argument("--min_area", type=int, default=120)
    p.add_argument("--min_conf_by_cls", type=str, default="")
    # ROI: normalized. 8 numbers = polygon (x1,y1,...,x4,y4), 4 numbers = rect (nx,ny,nw,nh)
    p.add_argument("--roi", type=str, default="")
    p.add_argument("--draw_roi", action="store_true")
    # Captions thresholds
    p.add_argument("--captions", action="store_true")
    p.add_argument("--caption_min_tracks", type=int, default=1)
    p.add_argument("--caption_min_conf", type=float, default=0.50)
    p.add_argument("--caption_min_frames", type=int, default=6, help="Same caption must repeat N consecutive frames")
    # Display
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def parse_min_conf_map(s: str) -> dict[int, float]:
    m: dict[int, float] = {}
    if not s:
        return m
    for tok in s.split(","):
        if ":" in tok:
            k, v = tok.split(":")
            try:
                m[int(k)] = float(v)
            except Exception:
                pass
    return m


def build_roi_polygon(norm_vals: list[float], W: int, H: int) -> np.ndarray | None:
    """정규화 roi 문자열을 픽셀 단위 polygon으로 변환"""
    if not norm_vals:
        return None
    if len(norm_vals) == 8:
        pts = [
            (norm_vals[0] * W, norm_vals[1] * H),
            (norm_vals[2] * W, norm_vals[3] * H),
            (norm_vals[4] * W, norm_vals[5] * H),
            (norm_vals[6] * W, norm_vals[7] * H),
        ]
        return np.array(pts, dtype=np.int32)
    if len(norm_vals) == 4:
        nx, ny, nw, nh = norm_vals
        pts = [
            (nx * W, ny * H),
            ((nx + nw) * W, ny * H),
            ((nx + nw) * W, (ny + nh) * H),
            (nx * W, (ny + nh) * H),
        ]
        return np.array(pts, dtype=np.int32)
    return None


def inside_polygon(cx: float, cy: float, polygon: np.ndarray | None) -> bool:
    if polygon is None:
        return True
    return cv2.pointPolygonTest(polygon, (int(cx), int(cy)), False) >= 0


def group_of(cls_id: int) -> str | None:
    for g, s in GROUPS.items():
        if cls_id in s:
            return g
    return None


# =========================
# Main
# =========================
def main():
    a = parse_args()

    # 모델/소스 체크
    if not Path(a.weights).exists():
        raise FileNotFoundError(f"Weights not found: {a.weights}")
    if not Path(a.source).exists():
        raise FileNotFoundError(f"Source not found: {a.source}")

    model = YOLO(a.weights)

    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {a.source}")

    # 첫 프레임에서 폭/높이 얻고 ROI polygon 구성
    ok, frame0 = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read first frame.")
    H0, W0 = frame0.shape[:2]

    roi_vals = []
    if a.roi:
        try:
            roi_vals = [float(x) for x in a.roi.split(",") if x.strip() != ""]
        except Exception:
            roi_vals = []
    roi_poly = build_roi_polygon(roi_vals, W0, H0)

    # 안정화 상태
    stable_caption: str | None = None
    last_change_ms = 0.0
    # 연속 프레임 후보 상태
    cand_caption: str | None = None
    cand_count: int = 0

    # 클래스별 임계치
    min_conf_map = parse_min_conf_map(a.min_conf_by_cls)

    # 첫 프레임도 처리하려면 되감기
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]

        # 추론
        r = model.predict(
            frame, imgsz=a.imgsz, conf=a.conf, iou=a.iou,
            device=a.device, verbose=False
        )
        boxes = r[0].boxes if r and len(r) > 0 else None

        detections = []
        if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                # 면적 필터
                area = (x2 - x1) * (y2 - y1)
                if area < a.min_area:
                    continue

                # ROI 필터 (중심점 기준)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if not inside_polygon(cx, cy, roi_poly):
                    continue

                # 클래스별 최소 conf
                thr = max(a.conf, min_conf_map.get(int(k), a.conf))
                if float(c) < thr:
                    continue

                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": float(c),
                    "cls": int(k),
                    "name": ID_TO_NAME.get(int(k), f"class_{int(k)}")
                })

        # 그룹/클래스 통계
        group_counts: dict[str, int] = {}
        group_best_conf: dict[str, float] = {g: 0.0 for g in GROUPS.keys()}
        class_counts: dict[int, int] = {}

        for d in detections:
            gid = group_of(d["cls"])
            if gid:
                group_counts[gid] = group_counts.get(gid, 0) + 1
                group_best_conf[gid] = max(group_best_conf[gid], d["conf"])
            class_counts[d["cls"]] = class_counts.get(d["cls"], 0) + 1

        present_groups = [g for g in GROUP_PRIORITY if g in group_counts]
        frame_state = present_groups[0] if present_groups else "NONE"

        # 프레임 후보 캡션(즉시)
        perframe_caption: str | None = None
        if a.captions and frame_state in GROUPS:
            # 해당 그룹 대표 클래스(가장 많이 보이는 클래스) 선택
            top_cls = None
            top_cnt = 0
            for k, cnt in class_counts.items():
                if k in GROUPS[frame_state] and cnt > top_cnt:
                    top_cnt = cnt
                    top_cls = k

            if (group_counts.get(frame_state, 0) >= a.caption_min_tracks and
                    group_best_conf.get(frame_state, 0.0) >= a.caption_min_conf and
                    top_cls is not None):
                perframe_caption = caption_text(frame_state, top_cls)

        # N 프레임 연속 조건
        if perframe_caption != cand_caption:
            cand_caption = perframe_caption
            cand_count = 1 if perframe_caption else 0
        else:
            if perframe_caption:
                cand_count += 1

        new_caption: str | None = None
        if cand_caption and cand_count >= a.caption_min_frames:
            new_caption = cand_caption

        # 디바운스(자막 변경 최소 간격)
        now_ms = time.time() * 1000.0
        if new_caption != stable_caption:
            if (now_ms - last_change_ms) >= a.debounce_ms:
                stable_caption = new_caption
                last_change_ms = now_ms

        # 그리기
        vis = frame

        if a.draw_roi and roi_poly is not None:
            cv2.polylines(vis, [roi_poly], isClosed=True, color=(255, 180, 0), thickness=2)

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = f'{d["name"]}:{d["conf"]:.2f}'
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(vis, label, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 3, cv2.LINE_AA)
            cv2.putText(vis, label, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 255, 240), 1, cv2.LINE_AA)

        cv2.putText(vis, f"FRAME_STATE: {frame_state}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 255), 2, cv2.LINE_AA)

        if stable_caption:
            cv2.rectangle(vis, (0, H - 60), (W, H), (0, 0, 0), -1)
            cv2.putText(vis, stable_caption, (16, H - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if a.show:
            cv2.imshow("traffic-light", vis)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
