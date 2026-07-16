# Comento CV Week 1 - Image Preprocessing Hard

## 주제

Git을 활용한 코드 관리 및 픽셀 단위 이미지 처리 실습

## 개요

week1 브랜치는 기본 과제에 더해 Hugging Face Food101 데이터셋을 활용한 이미지 전처리 및 이상치 필터링 심화 과제를 수행하였다.

기본 과제에서는 `sample.jpg`를 대상으로 OpenCV 기반 색상 필터링을 수행했고, 심화 과제에서는 `ethz/food101` 데이터셋에서 이미지를 가져와 AI 학습용 전처리 파이프라인을 구성했다.

## 사용 기술

- Python
- OpenCV
- NumPy
- Hugging Face Datasets
- Git / GitHub

## 파일 구성

```text
.
├── day1.ipynb
├── image_processing.py
├── image_preprocessing_hard.py
├── sample.jpg
├── outputs/
├── preprocessed_samples_hard/
└── README.md

```

## 과제 설명

### 1. 기본 이미지 처리

`image_processing.py`에서는 제공된 `sample.jpg` 이미지를 대상으로 OpenCV 기반의 기본 이미지 처리 실습을 수행했다.

#### 구현 내용

- 이미지 로드
- 이미지 크기 조정: 224 × 224
- BGR 이미지를 HSV 색상 공간으로 변환
- 빨간색 픽셀 영역 감지
- `cv2.threshold()`를 활용한 마스크 이진화
- 빨간색 영역 필터링
- Grayscale 변환
- Gaussian Blur를 활용한 노이즈 완화
- 처리 결과 이미지 저장

#### 실행 방법

```bash
pip install opencv-python numpy
python image_processing.py
```

#### 결과물

실행 후 `outputs/` 폴더에 다음 결과 이미지가 저장된다.

```text
outputs/
├── 01_resized.jpg
├── 02_red_mask.jpg
├── 03_red_filtered.jpg
├── 04_grayscale.jpg
└── 05_blurred.jpg
```

### 2. Hugging Face 데이터셋 기반 심화 전처리

`image_preprocessing_hard.py`에서는 Hugging Face의 `ethz/food101` 데이터셋을 사용하여 AI 학습용 이미지 전처리 파이프라인을 구성했다.

전체 데이터셋을 모두 다운로드하지 않고, `streaming=True` 옵션을 사용하여 필요한 샘플만 순차적으로 불러오도록 구현했다.

#### 데이터셋

- Dataset: `ethz/food101`
- Source: Hugging Face Datasets
- 처리 방식: `streaming=True`를 사용하여 이미지 샘플을 순차적으로 로드

### 3. 전처리 과정

심화 과제에서는 AI 학습에 활용 가능한 형태로 이미지를 변환하기 위해 다음 전처리 과정을 적용했다.

#### 3-1. 크기 조정

모든 이미지를 모델 입력 크기로 자주 사용되는 `224 × 224` 크기로 변환했다.

```python
resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
```

#### 3-2. 색상 변환

OpenCV에서 불러온 BGR 이미지를 Grayscale 이미지로 변환했다.  
이후 `cv2.normalize()`를 사용하여 픽셀 값을 0~255 범위로 정규화했다.

```python
gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
```

#### 3-3. 노이즈 제거

이미지의 작은 노이즈를 완화하기 위해 Gaussian Blur 필터를 적용했다.

```python
blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
```

#### 3-4. 데이터 증강

데이터 다양성을 확보하기 위해 다음 증강 기법을 적용했다.

- 좌우 반전
- 15도 회전
- HSV 기반 색상 변화
  - Hue 조정
  - Saturation 조정
  - Value 조정

```python
flipped = cv2.flip(color_augmented, 1)
rotated = rotate_image(flipped, angle=15)
```

### 4. 이상치 필터링

심화 문제 요구사항에 따라 전처리 전에 이상치 이미지를 필터링했다.

#### 4-1. 너무 어두운 이미지 제거

이미지를 Grayscale로 변환한 뒤 평균 밝기를 계산했다.  
평균 밝기가 기준값보다 낮은 이미지는 학습에 적합하지 않은 이미지로 판단하여 제외했다.

```python
DARK_MEAN_THRESHOLD = 45
```

```python
def is_too_dark(image, threshold=DARK_MEAN_THRESHOLD):
    mean_brightness = get_mean_brightness(image)
    return mean_brightness < threshold
```

#### 4-2. 객체 크기가 너무 작은 이미지 제거

객체가 너무 작게 포함된 이미지는 학습에 필요한 시각적 정보가 부족할 수 있으므로 제거했다.

객체 크기 추정은 다음 방식으로 진행했다.

1. HSV 색공간에서 채도와 명도 기반 마스크 생성
2. Grayscale 이미지에서 Canny Edge 추출
3. 색상 마스크와 Edge 마스크 결합
4. 가장 큰 contour의 bounding box 면적 계산
5. 전체 이미지 대비 객체 면적 비율이 기준값보다 작으면 제외

```python
MIN_OBJECT_AREA_RATIO = 0.08
```

```python
def is_object_too_small(image, threshold=MIN_OBJECT_AREA_RATIO):
    object_area_ratio = estimate_object_area_ratio(image)
    return object_area_ratio < threshold
```

### 5. 실행 방법

심화 전처리 코드는 다음 명령어로 실행한다.

```bash
pip install opencv-python numpy pillow datasets
python image_preprocessing_hard.py
```

### 6. 결과물

실행 후 `preprocessed_samples_hard/` 폴더에 처리된 이미지 5장과 전처리 로그가 저장된다.

```text
preprocessed_samples_hard/
├── sample_01.jpg
├── sample_02.jpg
├── sample_03.jpg
├── sample_04.jpg
├── sample_05.jpg
└── preprocessing_log.csv
```

`preprocessing_log.csv`에는 저장된 이미지의 파일명, 라벨, 평균 밝기, 객체 면적 비율이 기록된다.

### 7. 브랜치 작업 흐름

본 과제는 Git 브랜치를 활용하여 작업했다.

```text
main  : 기본 이미지 처리 과제 기준 브랜치
week1 : 심화 전처리 및 이상치 필터링 구현 브랜치
```

`week1` 브랜치에서 심화 과제 코드를 작성한 뒤, Pull Request를 통해 `main` 브랜치와의 변경 사항을 비교할 수 있도록 구성했다.

## 회고

이번 과제에서는 OpenCV를 활용한 기본 이미지 처리뿐 아니라, Hugging Face 데이터셋을 사용해 실제 AI 학습용 전처리 파이프라인을 구성했다.

기본 과제에서는 단일 이미지에 대해 HSV 색상 공간 변환, threshold 기반 마스크 생성, 특정 색상 필터링을 실습했다.  
심화 과제에서는 Food101 데이터셋 이미지를 가져와 크기 조정, Grayscale 변환, Normalize, Blur, 데이터 증강을 적용했다.

또한 평균 밝기와 객체 크기 기준의 이상치 필터링을 적용하여, 모델 학습 전 데이터 품질을 관리하는 과정이 중요하다는 점을 확인했다.

추후에는 단순 threshold 및 contour 기반 필터링 대신, 객체 탐지 모델이나 이미지 임베딩 기반 이상치 탐지 방법을 적용하여 더 정교한 데이터 품질 검사를 수행할 수 있을 것이다.
