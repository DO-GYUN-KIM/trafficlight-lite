import argparse, time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ===== Caption text (English) =====
def caption_text(group_name: str, top_cls: int) -> str | None:
    if group_name == "RED":
        if top_cls == 3:   # red_left
            return "Red light / Stop (Left turn stop)"
        return "Red light / Stop"
    if group_name == "GREEN":
        if top_cls == 1:   # green_left
            return "Green light / Turn left"
        if top_cls == 0:   # green_straight
            return "Green light / Go straight"
        return "Green light"
    if group_name == "YELLOW":
        return "Yellow light / Caution"
    if group_name == "WARN":
        return "Pedestrian warning signal"
    if group_name == "COUNTDOWN":
        return "Pedestrian countdown display"
    return None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.55)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--name", type=str, default="exp")
    p.add_argument("--debounce_ms", type=int, default=800)
    p.add_argument("--min_area", type=int, default=120)
    p.add_argument("--min_conf_by_cls", type=str, default="")
    p.add_argument("--roi", type=str, default="0,0,1,1")  # polygon points
    p.add_argument("--draw_roi", action="store_true")
    p.add_argument("--captions", action="store_true")
    p.add_argument("--show", action="store_true")

    # 새로 추가된 인자
    p.add_argument("--caption_min_tracks", type=int, default=2,
                   help="Minimum number of stable tracks to trigger caption")
    p.add_argument("--caption_min_conf", type=float, default=0.6,
                   help="Minimum confidence to trigger caption")

    return p.parse_args()

def inside_polygon(pt, polygon):
    x, y = pt
    return cv2.pointPolygonTest(np.array(polygon, np.int32), (int(x), int(y)), False) >= 0

def main():
    a = parse_args()
    cap = cv2.VideoCapture(a.source)
    if not cap.isOpened():
        print(f"Error: cannot open source {a.source}")
        return

    model = YOLO(a.weights)

    # ROI polygon parsing
    roi_vals = [float(x) for x in a.roi.split(",")]
    if len(roi_vals) == 8:
        roi_polygon = [
            (roi_vals[0]*cap.get(cv2.CAP_PROP_FRAME_WIDTH), roi_vals[1]*cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            (roi_vals[2]*cap.get(cv2.CAP_PROP_FRAME_WIDTH), roi_vals[3]*cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            (roi_vals[4]*cap.get(cv2.CAP_PROP_FRAME_WIDTH), roi_vals[5]*cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            (roi_vals[6]*cap.get(cv2.CAP_PROP_FRAME_WIDTH), roi_vals[7]*cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ]
    else:
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        roi_polygon = [(0,0),(w,0),(w,h),(0,h)]

    stable_state = None
    last_change = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=a.imgsz, conf=a.conf, iou=a.iou, device=a.device, verbose=False)
        detections = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes else []

        caption = None
        if len(detections) > 0:
            # ROI filter
            filtered = []
            for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
                x1,y1,x2,y2 = box.tolist()
                cx, cy = (x1+x2)/2, (y1+y2)/2
                if inside_polygon((cx,cy), roi_polygon):
                    filtered.append((cls.item(), conf.item()))

            if len(filtered) >= a.caption_min_tracks:
                # pick top by confidence
                top_cls, top_conf = max(filtered, key=lambda x:x[1])
                if top_conf >= a.caption_min_conf:
                    group = "GREEN" if int(top_cls) in [0,1,4] else "RED" if int(top_cls) in [2,3,12] else "YELLOW" if int(top_cls) in [5,6] else "WARN"
                    caption = caption_text(group, int(top_cls))

        # debounce logic
        now = int(time.time()*1000)
        if caption != stable_state and (now - last_change) >= a.debounce_ms:
            stable_state = caption
            last_change = now

        vis = frame.copy()
        if a.draw_roi:
            cv2.polylines(vis, [np.array(roi_polygon, np.int32)], isClosed=True, color=(255,0,0), thickness=2)

        if stable_state and a.captions:
            cv2.putText(vis, stable_state, (30, vis.shape[0]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(vis, stable_state, (30, vis.shape[0]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

        if a.show:
            cv2.imshow("traffic-light", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
