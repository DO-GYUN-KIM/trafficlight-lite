import argparse
from pathlib import Path
import sys
import torch
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser("YOLOv8 - ETRI (GPU 안정화 세팅)")
    p.add_argument("--data",   type=str, default=r"C:\summer\trafficlight-lite\data.yaml",
                   help="data.yaml 경로")
    p.add_argument("--model",  type=str, default="yolov8n.pt",
                   help="기본 가중치 (yolov8n.pt / yolov8s.pt ...)")
    p.add_argument("--epochs", type=int, default=20, help="학습 에폭 수")
    p.add_argument("--imgsz",  type=int, default=512, help="이미지 학습 크기")
    p.add_argument("--device", type=str, default="0", help='"0"=첫 번째 GPU(dGPU), "cpu"=CPU')
    p.add_argument("--name",   type=str, default="train_gpu", help="runs/detect/ 아래 런 이름")
    p.add_argument("--batch",  type=int, default=32, help="배치 사이즈(GPU). CPU면 내부에서 8로 고정")
    p.add_argument("--workers",type=int, default=0, help="DataLoader workers (Windows는 0~2 권장)")
    p.add_argument("--cache",  type=str, default="disk", choices=["disk","ram","none"],
                   help="이미지 캐시: disk(권장), ram(메모리 여유 시), none")
    p.add_argument("--seed",   type=int, default=42, help="랜덤 시드")
    return p.parse_args()

def main():
    args = parse_args()

    # 경로/파일 체크
    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml이 없음: {data_yaml}")
        sys.exit(1)

    # 디바이스 점검
    if args.device != "cpu" and not torch.cuda.is_available():
        print("[WARN] CUDA 사용 불가 → CPU로 전환")
        args.device = "cpu"

    # 시드 고정
    torch.manual_seed(args.seed)

    # 모델 로드
    model = YOLO(args.model)

    # 캐시 옵션 매핑
    cache_opt = {"disk": "disk", "ram": "ram", "none": False}[args.cache]

    # CPU/GUI별 안정 세팅
    if args.device == "cpu":
        batch = 8               # CPU에선 작게
        workers = 0             # 윈도우 멀티프로세싱 이슈 회피
        amp = False
        cache = False           # CPU에선 캐시 비권장
    else:
        batch = max(4, args.batch)   # 메모리 여유에 맞춰 조절 가능
        workers = max(0, args.workers)
        amp = True                    # 혼합 정밀로 가속
        cache = cache_opt             # 기본 'disk' 권장

    # 학습
    results = model.train(
        data=str(data_yaml),
        device=args.device,                 # "0" → dGPU
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        workers=workers,
        cache=cache,                        # 'disk' 기본(메모리 에러 방지)
        amp=amp,
        project=str(Path(r"C:\summer\trafficlight-lite\runs\detect")),
        name=args.name,
        optimizer="SGD",
        lr0=0.01,
        patience=max(10, args.epochs // 2), # 조기 종료 여유
        plots=True,
    )

    # 결과 안내
    print("\n[INFO] Training finished.")
    print("Save dir   :", results.save_dir)
    print("Best weight:", Path(results.save_dir) / "weights" / "best.pt")

if __name__ == "__main__":
    main()
