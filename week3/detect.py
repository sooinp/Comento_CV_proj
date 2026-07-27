from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS_PATH = "runs/detect/runs/week3_yolov8n/weights/best.pt"
TEST_IMAGE_DIR = Path("datasets/test/images")
OUTPUT_DIR = Path("outputs/detection_results")


def detect_and_save(model, image_path, output_dir):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없다: {image_path}")

    results = model(image)
    annotated_image = results[0].plot()

    output_path = output_dir / f"detected_{image_path.name}"
    cv2.imwrite(str(output_path), annotated_image)

    return output_path, len(results[0].boxes)


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(WEIGHTS_PATH)

    image_paths = sorted(TEST_IMAGE_DIR.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"테스트 이미지가 없다: {TEST_IMAGE_DIR}")

    for image_path in image_paths:
        output_path, num_objects = detect_and_save(model, image_path, output_dir)
        print(f"{image_path.name}: 탐지된 객체 수 {num_objects} -> {output_path}")


if __name__ == "__main__":
    main()
