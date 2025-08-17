# 🚦 Traffic Light Recognition System (with GUI)

YOLO 기반 교통 신호등 인식 시스템입니다.  
PyQt5 GUI를 통해 ROI(사다리꼴), 민감도(conf), 임계값 등을 조절하면서 실시간 영상에서 신호등을 탐지하고,  
안정적인 결과일 경우 영어 자막으로 안내합니다.

---

## 📂 프로젝트 구조

```
trafficlight-lite/
│
├─ tools/
│ ├─ video_trafficlight_system.py   # 메인 실행 (CLI)
│ ├─ gui_trafficlight_system.py     # GUI 실행
│ ├─ trafficlight_detector.py       # 탐지 로직
│ ├─ draw.py                        # ROI, 자막 등 영상 처리
│ ├─ train_yolo.py                  # 학습 스크립트
│ ├─ predict_yolo.py                # 추론 스크립트
│ ├─ preview_classes.py             # 클래스 미리보기
│ ├─ split_train_val.py             # 학습/검증 분할
│ ├─ auto_convert_to_coco.py        # COCO 포맷 변환
│ └─ run_video_gui.bat              # Windows 실행 배치 파일
│
├─ best.pt          # 학습된 가중치
├─ data.yaml        # 클래스 정의
├─ requirements.txt # 공통 패키지
├─ requirements-gpu.txt # GPU 전용 (권장)
├─ requirements-cpu.txt # CPU 전용 (옵션)
├─ .gitignore
└─ README.md
```

---

## ⚙️ 설치 방법 (conda 불필요)

**Python 권장 버전:** 3.9 ~ 3.11

```bash
# 1. 저장소 클론
git clone https://github.com/DO-GYUN-KIM/trafficlight-lite.git
cd trafficlight-lite

# 2. 가상환경 생성 및 활성화
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 3. 환경에 맞게 패키지 설치
# GPU (CUDA 12.1 예시)
pip install -r requirements-gpu.txt

# CPU only
# pip install -r requirements-cpu.txt

# 4. 설치 확인
python -c "import torch; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

---

## 🚀 실행 방법

### A) GUI 실행
```bash
cd tools
python gui_trafficlight_system.py
```

Windows에서는:
```bash
tools\run_video_gui.bat
```

### B) CLI 실행
```bash
cd tools
python video_trafficlight_system.py ^
  --weights C:\summer\trafficlight-lite\best.pt ^
  --source  C:\summer\trafficlight-lite\test.mp4 ^
  --device  0 ^
  --conf 0.55 --iou 0.7 ^
  --debounce_ms 800 --min_area 120 ^
  --captions --draw_roi --show ^
  --caption_min_frames 6 --caption_min_tracks 1 --caption_min_conf 0.5
```

**CPU로 강제 실행하려면:**
```bash
--device cpu
```

---

## 🖥️ GUI 주요 입력값

### 📂 입력 경로
- **Weights (.pt):** `C:\summer\trafficlight-lite\best.pt` (기본값)  
- **Source (video):** `C:\summer\trafficlight-lite\test.mp4` (원하는 영상으로 변경)

### 💻 Device
- `0` → 첫 번째 GPU 사용 (CUDA)  
- `1` → 두 번째 GPU  
- `cpu` → CPU 강제 실행  

### ⚙️ 기본 설정
- **conf:** 0.4 ~ 0.6 권장 (낮출수록 민감, 오탐 ↑)  
- **imgsz:** 640 기본 (정확도/속도 균형). GPU 여유 있으면 960 시도  

### 🔲 ROI (Trapezoid, 사다리꼴)
- **Top width:** 위쪽 평행선 길이 (0~1)  
- **Bottom width:** 아래쪽 평행선 길이 (0~1)  
- **Height:** ROI 높이 (0~1)  
- **Bottom X center:** 아래 평행선 중심 x좌표 (0~1)  

> ROI 설정은 `gui_config.json`에 자동 저장되어 다음 실행에도 유지됩니다.

### 🏳️ Flags
- **draw_roi:** ROI 폴리곤 그리기  
- **captions:** 자막 출력 ON/OFF  
- **show window:** 영상창 표시  

### 🔧 Advanced
- **min_area:** 너무 작은 박스 제거 (픽셀 면적). 기본 120  
- **debounce_ms:** 문구 변경 최소 간격(ms). 기본 800  
- **iou:** NMS IoU. 0.6~0.7 권장  
- **min_conf_by_cls:** 클래스별 최소 conf  
  - 예: `5:0.40,6:0.40` (노란불/좌회전 노란불 완화)  
- **caption_min_frames:** 같은 신호가 몇 프레임 연속일 때 자막 표기 (기본 6; 30fps ≈ 0.2s)  
- **caption_min_tracks:** 같은 그룹 동시 검출 최소 개수 (기본 1)  
- **caption_min_conf:** 자막 표기 최소 확신도 (기본 0.50)  

### 💡 팁 (노란불 놓침 시)
- `min_conf_by_cls`에 `5:0.40,6:0.40` 추가  
- conf 값을 살짝 낮추기 (예: 0.45)  
- ROI 위쪽 확장 또는 높이 늘리기  

### ▶️ 실행 버튼
- **Run:** 현재 설정으로 `video_trafficlight_system.py` 실행  
- **Stop:** 실행 중인 프로세스 강제 종료 (Windows에서도 확실히 종료됨)  

---

## 🗣️ 자막 규칙 (영어)

- **Red:** `Red light / Stop` (`red_left` → `Left turn stop`)  
- **Green:** `Green light / Go straight` / `Turn left` / `Directional arrow`  
- **Yellow:** `Yellow light / Caution` / `(left)`  
- **Pedestrian:** `Pedestrian warning`, `Countdown`  

**안정화 로직**
- 같은 캡션이 **N 프레임 연속(caption_min_frames)** 이어야 출력  
- 캡션 변경 최소 유지 시간: **debounce_ms**  
- 그룹 내 동시 검출 개수: **caption_min_tracks**  
- 최고 확신도가 **caption_min_conf** 이상이어야 출력  

---

## 🧪 자주 겪는 문제

- **경로 에러:** Weights/Source 절대경로 확인  
- **GPU인데 CPU로 실행됨:** `--device 0` 확인, `CUDA: True`인지 확인  
- **자막이 깜빡거림:** `caption_min_frames` 또는 `debounce_ms` 조정  
- **노란불 인식 약함:** `min_conf_by_cls` 조정, ROI 확장, conf 값 낮추기  

---

## 📌 개발 메모

- 설정은 `tools/gui_config.json`에 자동 저장/로드  
- `.gitignore`에 `gui_config.json`, `config.json` 제외됨  
- Python 3.9~3.11 권장  
- 기본값: `weights=C:\summer\trafficlight-lite\best.pt`, `source=C:\summer\trafficlight-lite\test.mp4`  

---

## 📝 라이선스
MIT
