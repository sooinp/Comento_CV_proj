import cv2
import numpy as np
from pathlib import Path
from datasets import load_dataset

# =========================
# 1주차 심화 전처리 과제
# - Hugging Face ethz/food101 데이터셋 사용
# - Resize 224x224
# - Grayscale + Normalize
# - Blur 필터
# - 데이터 증강: 좌우 반전, 회전, 색상 변화
# - 이상치 제거: 너무 어두운 이미지, 객체 크기가 너무 작은 이미지
# - 처리된 이미지 5장 저장
# =========================

OUTPUT_DIR = Path("preprocessed_samples_hard")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_SIZE = (224, 224)
MAX_SAVE_COUNT = 5

# 이상치 필터 기준값
DARK_MEAN_THRESHOLD = 45          # 평균 밝기 45 미만이면 너무 어두운 이미지로 판단
MIN_OBJECT_AREA_RATIO = 0.08      # 가장 큰 객체 bbox가 전체 이미지의 8% 미만이면 작은 객체로 판단


def pil_to_bgr(pil_image):
    """Hugging Face에서 가져온 PIL 이미지를 OpenCV BGR 형식으로 변환"""
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def resize_image(image, size=TARGET_SIZE):
    """이미지 크기 조정"""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def get_mean_brightness(image):
    """이미지 평균 밝기 계산"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def is_too_dark(image, threshold=DARK_MEAN_THRESHOLD):
    """평균 밝기 기준으로 너무 어두운 이미지 제거"""
    mean_brightness = get_mean_brightness(image)
    return mean_brightness < threshold


def estimate_object_area_ratio(image):
    """
    객체 크기 추정 알고리즘
    1. HSV 색공간에서 채도/명도 기반 마스크 생성
    2. Canny Edge 기반 윤곽선 보강
    3. 가장 큰 contour의 bounding box 면적 비율 계산
    """
    resized = resize_image(image)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # 음식/객체 영역은 배경보다 채도 또는 명도 변화가 있는 경우가 많으므로 마스크 생성
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    sat_mask = cv2.inRange(saturation, 30, 255)
    val_mask = cv2.inRange(value, 35, 255)
    color_mask = cv2.bitwise_and(sat_mask, val_mask)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    mask = cv2.bitwise_or(color_mask, edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    image_area = resized.shape[0] * resized.shape[1]
    bbox_area = w * h

    return bbox_area / image_area


def is_object_too_small(image, threshold=MIN_OBJECT_AREA_RATIO):
    """객체 크기가 너무 작은 이미지 제거"""
    object_area_ratio = estimate_object_area_ratio(image)
    return object_area_ratio < threshold


def color_jitter(image):
    """색상 변화 데이터 증강: HSV 기반 Hue, Saturation, Value 조정"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)

    # Hue 변화
    hsv[:, :, 0] = (hsv[:, :, 0] + 8) % 180

    # Saturation, Value 변화
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.10 + 5, 0, 255)

    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def rotate_image(image, angle=15):
    """회전 데이터 증강"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return rotated


def preprocess_image(image):
    """
    과제 요구 전처리 4종 수행
    1. 크기 조정 224x224
    2. 색상 변화 증강
    3. 좌우 반전, 회전 증강
    4. Grayscale 변환 + Normalize
    5. Blur 필터 적용
    """
    resized = resize_image(image)

    # 데이터 증강: 색상 변화
    color_augmented = color_jitter(resized)

    # 데이터 증강: 좌우 반전
    flipped = cv2.flip(color_augmented, 1)

    # 데이터 증강: 회전
    rotated = rotate_image(flipped, angle=15)

    # Grayscale 변환
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    # Normalize 적용: 픽셀값을 0~255 범위로 재조정
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Blur 필터 적용: 노이즈 완화
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)

    return blurred


def main():
    print("Hugging Face ethz/food101 데이터셋을 불러오는 중입니다.")
    print("전체 다운로드를 피하기 위해 streaming=True로 필요한 샘플만 순차적으로 가져옵니다.")

    dataset = load_dataset("ethz/food101", split="train", streaming=True)

    saved_count = 0
    scanned_count = 0

    log_lines = []
    log_lines.append("file_name,label,mean_brightness,object_area_ratio\n")

    for item in dataset:
        scanned_count += 1

        image = pil_to_bgr(item["image"])
        label = item.get("label", "unknown")

        mean_brightness = get_mean_brightness(image)
        object_area_ratio = estimate_object_area_ratio(image)

        # 심화 문제 1: 너무 어두운 이미지 제거
        if is_too_dark(image):
            continue

        # 심화 문제 2: 객체 크기가 너무 작은 이미지 제거
        if is_object_too_small(image):
            continue

        processed = preprocess_image(image)

        output_path = OUTPUT_DIR / f"sample_{saved_count + 1:02d}.jpg"
        cv2.imwrite(str(output_path), processed)

        log_lines.append(
            f"{output_path.name},{label},{mean_brightness:.2f},{object_area_ratio:.4f}\n"
        )

        print(
            f"[SAVE] {output_path.name} | "
            f"label={label} | "
            f"brightness={mean_brightness:.2f} | "
            f"object_ratio={object_area_ratio:.4f}"
        )

        saved_count += 1

        if saved_count >= MAX_SAVE_COUNT:
            break

        # 너무 많은 이미지를 훑지 않도록 제한
        if scanned_count >= 300:
            break

    log_path = OUTPUT_DIR / "preprocessing_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    print("\n작업 완료")
    print(f"스캔한 이미지 수: {scanned_count}")
    print(f"저장한 이미지 수: {saved_count}")
    print(f"저장 폴더: {OUTPUT_DIR.resolve()}")
    print(f"로그 파일: {log_path.resolve()}")


if __name__ == "__main__":
    main()
