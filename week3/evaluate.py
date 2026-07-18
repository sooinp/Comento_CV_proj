import json
from pathlib import Path

from ultralytics import YOLO

WEIGHTS_PATH = "runs/detect/runs/week3_yolov8n/weights/best.pt"
OUTPUT_PATH = Path("outputs/evaluation_results/metrics.json")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(WEIGHTS_PATH)
    metrics = model.val(data="data.yaml")

    result = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Precision:", result["precision"])
    print("Recall:", result["recall"])
    print("mAP50:", result["mAP50"])
    print("mAP50-95:", result["mAP50-95"])
    print("평가 결과 저장 위치:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
