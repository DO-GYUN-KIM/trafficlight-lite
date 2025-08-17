import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser("YOLOv8 inference")
    p.add_argument("--weights", default=r"C:\summer\trafficlight-lite\runs\detect\train_gpu\weights\best.pt")
    p.add_argument("--source",  default=r"C:\summer\trafficlight-lite\datasets\etri_raw\test\images")
    p.add_argument("--imgsz",   type=int, default=512)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--device",  default="0", help='"0" GPU / "cpu" CPU')
    p.add_argument("--name",    default="predict_gpu")
    return p.parse_args()

def main():
    a = parse_args()
    model = YOLO(a.weights)
    out = model.predict(
        source=a.source,
        device=a.device,
        imgsz=a.imgsz,
        conf=a.conf,
        save=True,
        project=r"C:\summer\trafficlight-lite\runs\detect",
        name=a.name,
        show=False,
        vid_stride=1
    )
    save_dir = Path(r"C:\summer\trafficlight-lite\runs\detect") / a.name
    print(f"[INFO] Predictions saved to: {save_dir}")

if __name__ == "__main__":
    main()
