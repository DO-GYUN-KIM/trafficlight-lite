# 🚦 Traffic Light Recognition System (with GUI)

YOLO 기반 교통 신호등 인식 시스템입니다.  
PyQt5 GUI를 통해 ROI, 민감도, 임계값 등을 조절하면서 실시간 영상에서 신호등을 탐지하고  
확실한 신호 상황일 경우 한국어 자막으로 안내합니다.  

---

## 📂 프로젝트 구조
trafficlight-lite/
│
├─ tools/
│ ├─ video_trafficlight_system.py # 메인 실행 파일
│ ├─ gui_trafficlight_system.py # GUI 실행 파일
│ ├─ trafficlight_detector.py # 탐지 로직
│ ├─ draw.py # ROI, 자막 등 영상 처리
│ ├─ train_yolo.py # 학습 스크립트
│ ├─ predict_yolo.py # 추론 스크립트
│ ├─ preview_classes.py # 클래스 미리보기
│ ├─ split_train_val.py # 학습/검증 분할
│ ├─ auto_convert_to_coco.py # COCO 포맷 변환
│ └─ run_video_gui.bat # Windows 실행 배치 파일
├─ test.mp4
├─ best.pt # 학습된 가중치
├─ data.yaml # 클래스 정의
├─ requirements.txt # 필요 패키지
├─ .gitignore
└─ README.md

---

## ⚙️ 설치 방법
```bash
# 1. 저장소 클론
git clone https://github.com/DO-GYUN-KIM/trafficlight-lite.git
cd trafficlight-lite

# 2. 가상환경 생성 (선택)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# 3. 필수 패키지 설치
pip install -r requirements.txt


🚀 실행 방법

GUI 실행 또는 또는 배치 파일 실행 