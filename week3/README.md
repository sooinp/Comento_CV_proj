# Comento CV Week 3 - AI Object Detection and Visualization

## 주제

YOLOv8 기반 객체 탐지 모델 학습 및 OpenCV 결과 시각화

## 개요

본 브랜치는 코멘토 Computer Vision 직무부트캠 3차 업무를 수행한 브랜치이다.

Roboflow에서 내려받은 실제 이미지 데이터셋으로 YOLOv8n 객체 탐지 모델을 전이 학습하고, 테스트 이미지에서 객체의 위치·클래스·신뢰도를 탐지한다. 이후 OpenCV와 Matplotlib을 이용해 탐지 결과와 모델 성능을 시각화한다.

## 수행 목표

- 객체 탐지용 이미지 데이터셋 구성
- YOLOv8 사전 학습 모델 기반 전이 학습
- 테스트 이미지 객체 탐지
- Bounding Box, 클래스명, Confidence 시각화
- Precision, Recall, mAP 성능 평가
- 데이터 증강 및 학습 설정 변경을 통한 성능 비교
- Git을 활용한 코드 및 결과 관리

## 사용 기술

- Python
- PyTorch / Torchvision
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Matplotlib
- Roboflow (데이터셋 다운로드)
- python-dotenv (API 키 관리)
- Git / GitHub

## 프로젝트 구조

```text
week3/
├── README.md
├── data.yaml
├── train.py
├── detect.py
├── evaluate.py
├── visualize_metrics.py
├── day3.ipynb
├── bus.jpg
├── yolov8n.pt                 # gitignore (사전 학습 가중치)
├── .env                       # gitignore (ROBOFLOW_API_KEY)
├── datasets/                  # gitignore
│   ├── train/
│   │   ├── images/            # 154장
│   │   └── labels/
│   ├── valid/
│   │   ├── images/            # 25장
│   │   └── labels/
│   └── test/
│       ├── images/            # 13장
│       └── labels/
├── runs/                      # gitignore
│   └── detect/
│       ├── runs/week3_yolov8n/
│       │   └── weights/
│       │       ├── best.pt
│       │       └── last.pt
│       └── val/
└── outputs/
    ├── detection_results/      # 탐지 결과 이미지
    └── evaluation_results/     # metrics.json, metrics_bar_chart.png
```

## 데이터셋 구성

Roboflow Universe의 [`my-workspace-1/natural-object-detection`](https://universe.roboflow.com/my-workspace-1/natural-object-detection) (version 3, CC BY 4.0) 데이터셋을 사용한다.

- 규모: train 154장 / valid 25장 / test 13장 (총 192장)
- 클래스 22개: 실제 객체(`Person`, `Squirrel`, `cars`, `cats`, `cheetah`, `dogs`, `fox`, `frog`, `lion`, `owl`, `stop sign`, `tiger`, `traffic light`, `wolf`) + 촬영 조건 태그(`Blur`, `Camouflage`, `Foggy`, `Overlap`, `Patches`, `Poor lighting`, `Rain`, `Snow`)

라벨 파일은 YOLO 형식(`class_id x_center y_center width height`)을 사용하며, 좌표와 크기는 이미지 전체 크기 기준 `0~1`로 정규화되어 있다.

Roboflow API 키는 `week3/.env`에 `ROBOFLOW_API_KEY=...` 형태로 저장하고 `.gitignore`에 등록해 Git에는 올리지 않는다.

### `data.yaml`

```yaml
# Roboflow Universe: my-workspace-1/natural-object-detection (version 3, CC BY 4.0)
train: ./datasets/train/images
val: ./datasets/valid/images
test: ./datasets/test/images

nc: 22
names:
  - Blur
  - Camouflage
  - Foggy
  - Overlap
  - Patches
  - Person
  - Poor lighting
  - Rain
  - Snow
  - Squirrel
  - cars
  - cats
  - cheetah
  - dogs
  - fox
  - frog
  - lion
  - owl
  - stop sign
  - tiger
  - traffic light
  - wolf
```

## 환경 설정

### 1. 가상환경 생성 및 활성화

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 2. 패키지 설치

```bash
pip install torch torchvision ultralytics opencv-python numpy matplotlib roboflow python-dotenv
```

## 주요 구현 내용

### 1. 사전 학습 모델 동작 확인

전이 학습 전에 Ultralytics 공식 예제 이미지(`bus.jpg`)로 `yolov8n.pt`의 추론 파이프라인이 정상 동작하는지 먼저 확인했다. person 4개, bus 1개, stop sign 1개, 총 6개 객체를 탐지했다(결과: `outputs/detection_results/pretrained_demo.jpg`).

### 2. YOLOv8 전이 학습 (`train.py`)

```python
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        project="runs",
        name="week3_yolov8n",
    )


if __name__ == "__main__":
    main()
```

CPU 환경에서 학습에 약 1.15시간이 걸렸다.

### 3. 객체 탐지 (`detect.py`)

학습된 가중치(`runs/detect/runs/week3_yolov8n/weights/best.pt`)로 테스트 이미지 13장을 추론하고, 결과 이미지를 `outputs/detection_results/`에 저장한다.

```python
WEIGHTS_PATH = "runs/detect/runs/week3_yolov8n/weights/best.pt"

model = YOLO(WEIGHTS_PATH)
results = model(image)
annotated_image = results[0].plot()
```

### 4. 모델 성능 평가 (`evaluate.py`)

검증 데이터셋 기준 Precision, Recall, mAP50, mAP50-95를 계산해 `outputs/evaluation_results/metrics.json`에 저장한다.

| 평가 지표 | 의미 |
|---|---|
| Precision | 탐지했다고 판단한 객체 중 실제 정답 비율 |
| Recall | 실제 객체 중 모델이 올바르게 탐지한 비율 |
| mAP50 | IoU 0.5 기준 평균 정밀도 |
| mAP50-95 | IoU 0.5~0.95 구간의 종합 평균 정밀도 |

### 5. 성능 시각화 (`visualize_metrics.py`)

`metrics.json`을 막대 그래프로 그려 `outputs/evaluation_results/metrics_bar_chart.png`에 저장한다.

## 실행 방법

```bash
python train.py
python detect.py
python evaluate.py
python visualize_metrics.py
```

## 결과

`train.py`(CPU, 1.15시간), `detect.py`, `evaluate.py`, `visualize_metrics.py`를 실제로 실행한 결과는 다음과 같다.

| 항목 | 결과 |
|---|---|
| 사용 모델 | YOLOv8n |
| Epoch | 20 |
| 이미지 크기 | 640 |
| 클래스 수 | 22 |
| Precision | 0.200 |
| Recall | 0.296 |
| mAP50 | 0.371 |
| mAP50-95 | 0.281 |
| 추론 시간 | 약 70~170ms/이미지 (CPU) |

탐지 결과 이미지: `outputs/detection_results/`
성능 평가 그래프: `outputs/evaluation_results/metrics_bar_chart.png`

## 결과 분석

이미지 192장(train 154 / valid 25 / test 13)이라는 적은 데이터와 20 epoch만으로는 Precision/Recall이 낮은 편이다. 특히 `Person`, `Squirrel`, `traffic light`처럼 인스턴스가 1~2개뿐인 클래스는 거의 탐지되지 않았다. 반면 `Patches`(mAP50 0.963), `Rain`(0.995)처럼 인스턴스 수가 상대적으로 많거나 시각적으로 뚜렷한 클래스는 성능이 높았다.

## 성능 개선 아이디어

기본 모델 결과를 기준으로 다음 조건을 바꿔가며 성능을 비교해볼 수 있다.

### 데이터 증강

- 좌우 반전
- 이미지 회전
- 밝기 및 대비 조절
- Scale 및 Crop
- 노이즈 추가

### 하이퍼파라미터 조정

- Epoch 증가
- Batch Size 변경
- 입력 이미지 크기 변경
- Learning Rate 조정
- Confidence Threshold 조정

### 모델 크기 비교

| 모델 | 특징 |
|---|---|
| YOLOv8n | 가장 가볍고 빠름 |
| YOLOv8s | 속도와 정확도의 균형이 좋음 |
| YOLOv8m | 정확도가 높지만 연산량이 큼 |

정확도만 높이는 것이 아니라 추론 속도와 자원 사용량도 함께 비교해야, 실제 서비스에 어떤 모델이 적합한지 판단할 수 있다.

## 한계 및 개선 방향

- 데이터가 192장으로 적고 클래스별 인스턴스 수가 불균형해 일부 클래스(`Person`, `Squirrel`, `traffic light`)는 거의 탐지되지 않았다.
- 20 epoch, CPU 환경이라는 제약으로 학습이 충분하지 않았을 가능성이 있다.
- 실제 서비스 적용을 위해서는 더 많은 데이터와 클래스별 균형 잡힌 라벨링이 필요하다.
- 향후 FastAPI를 연동해 객체 탐지 API로 확장할 수 있다.

## 학습 내용

이번 과제에서는 사전 학습 모델을 실제 데이터셋에 적용하고 학습·평가·추론 결과를 분석했다.

객체 탐지 모델은 출력 이미지만 확인하는 것이 아니라 Precision, Recall, mAP를 함께 분석해야 하며, 데이터 수와 클래스 불균형이 실제 성능에 어떻게 반영되는지 확인할 수 있었다.
