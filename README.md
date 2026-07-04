## 주제
Git을 활용한 코드 관리 및 픽셀 단위 이미지 처리 실습

## 구현 내용
- OpenCV를 활용한 이미지 로드
- 이미지 크기 조정
- BGR 이미지를 HSV 색상 공간으로 변환
- 특정 색상인 빨간색 픽셀 감지
- cv2.threshold()를 활용한 마스크 이진화
- 빨간색 영역 필터링
- Grayscale 변환
- Gaussian Blur를 활용한 노이즈 완화
- 결과 이미지 저장

## 실행 방법

```bash
pip install opencv-python numpy
python image_processing.py