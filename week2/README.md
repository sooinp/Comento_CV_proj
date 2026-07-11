# Comento CV Week 2 - Unit Test and 2D to 3D Conversion

## 1. 과제 주제

Unit Test 구성 및 2D → 3D 변환 실습이다.

## 2. 진행 목표

- Python `pytest`를 활용한 Unit Test 작성 및 코드 검증
- OpenCV와 NumPy를 활용한 2D 이미지의 Grayscale / Depth Map 변환
- 이미지 좌표와 밝기 값을 이용한 간단한 3D Point Cloud 배열 생성
- 테스트 결과와 변환 결과 문서화

## 3. 사용 기술

- Python
- OpenCV
- NumPy
- pytest
- Matplotlib
- Git / GitHub

## 4. 프로젝트 구조

```text
.
├── sample.jpg
├── src/
│   ├── __init__.py
│   └── day2_depth_processing.py
├── tests/
│   └── test_day2_depth_processing.py
├── outputs/
│   ├── grayscale.png
│   ├── depth_map.png
│   ├── point_cloud_preview.png
│   └── pytest_result.txt
├── README.md
├── ppt_outline.md
├── requirements.txt
└── .gitignore
```

## 5. 구현 내용

### 5.1 이미지 로드

`cv2.imread()`를 사용하여 입력 이미지를 BGR 형식으로 로드했다. 파일이 없거나 읽을 수 없는 경우 예외 처리를 추가했다.

### 5.2 Grayscale 변환

BGR 이미지를 `cv2.cvtColor()`를 사용하여 Grayscale 이미지로 변환했다.

### 5.3 Depth Map 생성

Grayscale 밝기 값을 0~255 범위로 정규화한 뒤, `cv2.applyColorMap()`을 적용하여 가상의 Depth Map 이미지를 생성했다.

### 5.4 Point Cloud 생성

이미지의 x, y 좌표와 Grayscale 밝기 값을 z값으로 사용하여 `(x, y, z)` 형태의 3D 포인트 배열을 생성했다.

## 6. Unit Test

`pytest`를 사용하여 다음 항목을 검증했다.

- Grayscale 변환 결과가 2D 배열인지 확인
- Depth Map 결과가 원본 이미지와 같은 크기를 가지는지 확인
- Point Cloud 결과가 `(N, 3)` 형태인지 확인
- `None` 입력에 대한 예외 처리 확인
- 잘못된 `stride` 입력에 대한 예외 처리 확인

테스트 실행 명령어는 아래와 같다.

```bash
python -m pytest -q tests
```

## 7. 실행 결과

- Unit Test 실행 결과: `outputs/pytest_result.txt`
- Grayscale 이미지: `outputs/grayscale.png`
- Depth Map 이미지: `outputs/depth_map.png`
- Point Cloud 미리보기: `outputs/point_cloud_preview.png`

## 8. GitHub 업로드 기준

GitHub에는 코드, 테스트, 문서, 재현에 필요한 최소 결과물을 올린다.
큰 원본 데이터, 캐시 파일, 가상환경, 개인 정보가 들어간 파일은 올리지 않는다.

## 9. 한계점 및 개선 방향

이번 실습의 Depth Map은 단일 이미지의 밝기 값을 이용해 생성한 가상의 깊이 표현이다.
따라서 실제 거리 정보를 의미하지는 않는다. 실제 3D 복원을 위해서는 Stereo Vision, Depth Sensor, 또는 MiDaS와 같은 사전학습 Depth Estimation 모델을 적용할 수 있다.
