import cv2
import numpy as np
from pathlib import Path

INPUT_IMAGE = "sample.jpg"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    image = cv2.imread(INPUT_IMAGE)

    if image is None:
        raise FileNotFoundError(f"{INPUT_IMAGE} 파일을 읽을 수 없습니다.")

    # 이미지 크기 조정
    resized = cv2.resize(image, (224, 224))

    # HSV 색상 공간 변환
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # 빨간색 범위 지정
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # 빨간색 마스크 생성
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # threshold 적용
    _, threshold_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 빨간색 영역 필터링
    red_filtered = cv2.bitwise_and(resized, resized, mask=threshold_mask)

    # Grayscale / Blur 전처리
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 결과 저장
    cv2.imwrite(str(OUTPUT_DIR / "01_resized.jpg"), resized)
    cv2.imwrite(str(OUTPUT_DIR / "02_red_mask.jpg"), threshold_mask)
    cv2.imwrite(str(OUTPUT_DIR / "03_red_filtered.jpg"), red_filtered)
    cv2.imwrite(str(OUTPUT_DIR / "04_grayscale.jpg"), gray)
    cv2.imwrite(str(OUTPUT_DIR / "05_blurred.jpg"), blurred)

    print("이미지 처리 완료")
    print(f"결과 저장 위치: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()