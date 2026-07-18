import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 스크립트로 실행할 때 GUI 창 없이 파일로만 저장한다.
import matplotlib.pyplot as plt

METRICS_PATH = Path("outputs/evaluation_results/metrics.json")
OUTPUT_PATH = Path("outputs/evaluation_results/metrics_bar_chart.png")


def main():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("YOLOv8 Week3 Model Performance")
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")

    print("성능 그래프 저장 위치:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
