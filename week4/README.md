# GalaxyLens — Galaxy10 DECaLS 은하 형태 분류

코멘토 CV 직무부트캠프 4차 업무 (희망 제품/SW 개발 프로젝트)

## 1. 기획

- **주제**: 은하 이미지 1장을 입력하면 형태(Merging, Barred Spiral 등 10종)를 분류하는 도구
- **배경**: Galaxy Zoo 시민과학 프로젝트의 분류 작업을 AI로 자동화하는 시나리오.
  1~3주차 기술을 종합 적용 — HF streaming 데이터 수집(1주차), 예외 처리·Unit Test(2주차),
  전이 학습·성능 평가·막대그래프 시각화(3주차)
- **핵심 실험**: YOLOv8n-cls vs YOLO11n-cls를 동일 조건으로 비교 학습하여
  top-1 정확도 기준 최적 버전을 선정한 뒤, 우승 버전으로 본 학습을 진행
- **데이터**: Hugging Face `matthieulel/galaxy10_decals` (Galaxy10 DECaLS, 256×256, 10클래스)
  — streaming으로 클래스당 200장 샘플링 (총 2,000장, train 1,600 / val 400)
- **기술 스택**: Python · Ultralytics YOLO(분류 모드) · Hugging Face datasets ·
  OpenCV · Matplotlib · pytest · Git/GitHub

## 2. 실행 순서

```bash
pip install -r requirements.txt

# 1) 데이터 구축 (HF streaming → YOLO 분류 폴더 구조)
python app.py build-data --per-class 200

# 2) 버전 비교 (동일 조건: epochs 10, imgsz 224, seed 42)
python app.py compare --epochs 10

# 3) 우승 버전으로 본 학습 (예: yolo11n-cls가 이겼다면)
python app.py train --model yolo11n-cls.pt --epochs 30

# 4) 예측 + 시각화
python app.py predict --image data/galaxy10/val/Merging/Merging_0190.jpg

# 단위 테스트 (14개, 모델 로드 없이 동작)
python -m pytest -q tests
```

## 3. 비교 실험 설계 (공정성 원칙)

| 항목 | 값 |
|---|---|
| 후보 | yolov8n-cls.pt · yolo11n-cls.pt |
| 데이터 | 동일 서브셋 (클래스당 200장, 고정 분할) |
| 조건 | epochs 10 · imgsz 224 · seed 42 동일 |
| 지표 | top-1(주지표) · top-5 · 학습 시간 |
| 선정 | top-1 최고, 동률이면 학습 시간 짧은 쪽 |

산출물: `outputs/model_comparison.csv`, `outputs/model_comparison_chart.png`

## 4. 프로젝트 구조

```
week4/
├ app.py                    # CLI (build-data / compare / train / predict)
├ src/
│ ├ dataset_builder.py      # HF streaming → YOLO 분류 폴더
│ ├ model_compare.py        # 버전 비교 학습 + 우승 선정 + CSV/차트
│ └ classifier.py           # 예측 + OpenCV 시각화
├ tests/                    # pytest 14개 (순수 로직, 모델 불필요)
└ outputs/                  # 비교표·차트·예측 이미지
```
