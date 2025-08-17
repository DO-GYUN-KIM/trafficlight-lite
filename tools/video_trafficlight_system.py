# C:\summer\trafficlight-lite\tools\video_trafficlight_system.py
import argparse, json, time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

OUT_ROOT = r"C:\summer\trafficlight-lite\runs\detect"

# ===== 클래스 이름(데이터셋에 맞게 필요시 수정) =====
ID_TO_NAME = {
    0: "green_straight",
    1: "green_left",
    2: "red",
    3: "red_left",
    4: "green_arrow",
    5: "yellow",
    6: "yellow_left",
    7: "countdown",
    8: "off_signal",
    9: "ped_red",
    10:"ped_green",
    11:"blue_straight",
    12:"red_alt",
    13:"ped_warning"
}

# ===== 그룹(프레임 요약)과 우선순위 =====
GROUPS = {
    "RED":        {2, 3, 12, 9},
    "YELLOW":     {5, 6},
    "GREEN":      {0, 1, 4, 11, 10},
    "OFF":        {8},
    "COUNTDOWN":  {7},
    "WARN":       {13}
}
GROUP_PRIORITY = ["RED", "YELLOW", "GREEN", "WARN", "COUNTDOWN", "OFF"]

def parse_args():
    p = argparse.ArgumentParser("Traffic-light video system (ROI/draw/captions)")
    p.add_argument("--weights", required=True)
    p.add_argument("--source",  required=True)
    p.add_argument("--device",  default="0")
    p.add_argument("--imgsz",   type=int, default=512)
    p.add_argument("--conf",    type=float, default=0.55)
    p.add_argument("--iou",     type=float, default=0.7)
    p.add_argument("--name",    default="video_system")
    p.add_argument("--debounce_ms", type=int, default=800)
    p.add_argument("--min_conf_by_cls", type=str, default="8:0.80,7:0.70")
    p.add_argument("--min_area", type=int, default=120)
    p.add_argument("--roi", type=str, default="0,0,1,0.55")   # nx,ny,nw,nh
    p.add_argument("--draw_roi", action="store_true")         # ROI 사각형 오버레이
    p.add_argument("--captions", action="store_true")         # 한글 자막 표시
    p.add_argument("--caption_min_tracks", type=int, default=2)
    p.add_argument("--caption_min_conf",   type=float, default=0.60)
    p.add_argument("--show", action="store_true")
    return p.parse_args()

def parse_min_conf_map(s: str):
    m = {}
    if not s: return m
    for tok in s.split(","):
        if ":" in tok:
            k,v = tok.split(":")
            try: m[int(k)] = float(v)
            except: pass
    return m

def in_roi(box, W, H, roi_norm):
    x1,y1,x2,y2 = box
    rx, ry, rw, rh = roi_norm
    RX1, RY1, RX2, RY2 = int(rx*W), int(ry*H), int((rx+rw)*W), int((ry+rh)*H)
    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
    return (RX1 <= cx <= RX2) and (RY1 <= cy <= RY2)

def draw_rect(vis, x1, y1, x2, y2, color, t=2):
    cv2.rectangle(vis, (x1,y1), (x2,y2), color, t)

def draw_label(vis, box, label, color=(0,255,0)):
    x1,y1,x2,y2 = map(int, box)
    draw_rect(vis, x1,y1,x2,y2, color, 2)
    cv2.putText(vis, label, (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# ===== 자막 문구 생성 (그룹/클래스 기반) =====
def caption_text(group_name: str, top_cls: int) -> str | None:
    if group_name == "RED":
        if top_cls in {3}:
            return "전방 좌회전 정지 신호(빨간불)가 켜져 있습니다."
        return "전방에 정지 신호(빨간불)가 켜져 있습니다."
    if group_name == "GREEN":
        if top_cls in {1}:
            return "전방 좌회전 진행 신호(초록불)가 켜져 있습니다."
        if top_cls in {4}:
            return "전방 방향 진행 신호(초록 화살표)가 켜져 있습니다."
        return "전방에 직진 진행 신호(초록불)가 켜져 있습니다."
    if group_name == "YELLOW":
        return "전방에 주의 신호(노란불)가 켜져 있습니다."
    if group_name == "WARN":
        return "보행자 주의 신호가 켜져 있습니다."
    if group_name == "COUNTDOWN":
        return "보행자 잔여 시간 표시가 보입니다."
    return None

def main():
    a = parse_args()
    src = Path(a.source)
    assert src.exists(), f"소스 없음: {src}"

    # FPS 읽기
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    save_dir = Path(OUT_ROOT) / a.name / src.stem
    save_dir.mkdir(parents=True, exist_ok=True)
    out_video = save_dir / f"{src.stem}_pred.mp4"
    log_path  = save_dir / "events.json"

    min_conf_map = parse_min_conf_map(a.min_conf_by_cls)
    roi_norm = tuple(map(float, a.roi.split(","))) if a.roi else (0,0,1,1)

    model = YOLO(a.weights)
    gen = model.track(
        source=str(src),
        device=a.device,
        imgsz=a.imgsz,
        conf=a.conf,
        iou=a.iou,
        tracker="botsort.yaml",
        persist=True,
        stream=True,
        verbose=False,
        save=False
    )

    tracks = {}    # tid -> {"last":cls,"stable":cls,"since":ms}
    events = []
    t0 = time.time()
    vw = None

    for r in gen:
        frame = r.orig_img
        H, W = frame.shape[:2]
        if vw is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(str(out_video), fourcc, fps, (W, H))

        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            tids = (r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1]*len(cls))
            for bb, c, k, tid in zip(xyxy, conf, cls, tids):
                x1,y1,x2,y2 = bb
                area = (x2-x1)*(y2-y1)
                if area < a.min_area:                         # 면적 필터
                    continue
                if not in_roi((x1,y1,x2,y2), W, H, roi_norm): # ROI 필터
                    continue
                th = max(a.conf, min_conf_map.get(int(k), a.conf))  # 클래스별 최소 conf
                if c < th:
                    continue
                dets.append({"bbox":bb, "conf":float(c), "cls":int(k), "tid":int(tid)})

        # 추적 안정화
        now_ms = (time.time()-t0)*1000.0
        for d in dets:
            tid = d["tid"]
            if tid < 0:
                continue
            rec = tracks.get(tid, {"last":None,"stable":None,"since":now_ms})
            cand = d["cls"]
            if rec["last"] is None:
                rec["last"], rec["since"] = cand, now_ms
            else:
                if cand != rec["last"]:
                    rec["since"] = now_ms
                rec["last"] = cand
                if (now_ms - rec["since"]) >= a.debounce_ms and cand != rec["stable"]:
                    rec["stable"] = cand
                    events.append({
                        "t_ms": round(now_ms,1),
                        "track_id": tid,
                        "state_id": int(cand),
                        "state_name": ID_TO_NAME.get(int(cand), f"class_{int(cand)}"),
                        "group": next((g for g,ids in GROUPS.items() if int(cand) in ids), "UNK")
                    })
            tracks[tid] = rec

        # 프레임 상태 요약 (안정화된 트랙 집계)
        present_groups = set()
        stable_cls_counts = {}
        group_best_conf = {g: 0.0 for g in GROUPS.keys()}

        # 각 안정화된 트랙에 대해 집계
        for tid, rec in tracks.items():
            k = rec.get("stable")
            if k is None:
                continue
            stable_cls_counts[k] = stable_cls_counts.get(k, 0) + 1
            gid = next((g for g, ids in GROUPS.items() if k in ids), None)
            if gid:
                present_groups.add(gid)

        # 현재 프레임의 dets(즉시 신뢰도)에서 그룹별 최고 conf 갱신
        for d in dets:
            gid = next((g for g, ids in GROUPS.items() if d["cls"] in ids), None)
            if gid:
                group_best_conf[gid] = max(group_best_conf[gid], d["conf"])

        frame_state = next((g for g in GROUP_PRIORITY if g in present_groups), "NONE")

        # ===== 자막 판단 =====
        caption = None
        if a.captions and frame_state in GROUPS:
            # 이 그룹에서 가장 많은 클래스(top_cls)와 안정화된 트랙 수
            total_stable_in_group = 0
            top_cls, top_cnt = None, 0
            for c_id, cnt in stable_cls_counts.items():
                if c_id in GROUPS[frame_state]:
                    total_stable_in_group += cnt
                    if cnt > top_cnt:
                        top_cnt, top_cls = cnt, c_id

            if total_stable_in_group >= a.caption_min_tracks and group_best_conf[frame_state] >= a.caption_min_conf and top_cls is not None:
                caption = caption_text(frame_state, top_cls)

        # 시각화
        vis = frame.copy()
        if a.draw_roi:
            rx, ry, rw, rh = roi_norm
            x1, y1, x2, y2 = int(rx*W), int(ry*H), int((rx+rw)*W), int((ry+rh)*H)
            draw_rect(vis, x1, y1, x2, y2, (255, 180, 0), 2)

        for d in dets:
            tid = d["tid"]
            nm  = ID_TO_NAME.get(d["cls"], f"class_{d['cls']}")
            label = f"id{tid}:{nm} {d['conf']:.2f}"
            color = (0,255,0)
            if tid in tracks and tracks[tid]["stable"] is not None:
                sname = ID_TO_NAME.get(tracks[tid]["stable"], f"class_{tracks[tid]['stable']}")
                label = f"id{tid}:{sname} {d['conf']:.2f}"
                color = (0,200,255)
            draw_label(vis, d["bbox"], label, color)

        cv2.putText(vis, f"FRAME_STATE: {frame_state}", (12,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,220,255), 2, cv2.LINE_AA)

        if caption:
            cv2.rectangle(vis, (0, H-60), (W, H), (0,0,0), -1)
            cv2.putText(vis, caption, (16, H-18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        vw.write(vis)
        if a.show:
            cv2.imshow("traffic-light", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    if vw is not None: vw.release()
    cv2.destroyAllWindows()

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved video : {out_video}")
    print(f"[OK] Saved events: {log_path}")

if __name__ == "__main__":
    main()
