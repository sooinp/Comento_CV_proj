"""YOLO 분류 모델 버전 비교 모듈.

같은 데이터·같은 하이퍼파라미터로 후보 모델들을 학습시키고
top-1 / top-5 정확도와 학습 시간을 비교해 최적 버전을 고른다.
"""
import csv
import time
from pathlib import Path

# 비교 후보: YOLOv8/YOLO11 각각 nano·small·medium 크기로 총 6개 비교
DEFAULT_CANDIDATES = [
    "yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt",
    "yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt",
]


def pick_best(results):
    """비교 결과에서 최고 모델을 고른다.

    기준: top-1 정확도 우선, 동률이면 학습 시간이 짧은 쪽.
    """
    if not results:
        raise ValueError("비교 결과가 비어 있습니다.")
    return sorted(results, key=lambda r: (-r["top1"], r["train_sec"]))[0]


def save_comparison_csv(results, output_dir="outputs"):
    """비교 결과를 CSV로 저장한다."""
    if results is None:
        raise ValueError("비교 결과가 None입니다.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model_comparison.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "top1", "top5", "train_sec"])
        writer.writeheader()
        writer.writerows(results)
    return str(path)


def save_comparison_chart(results, output_dir="outputs"):
    """top-1 정확도 비교 막대그래프를 저장한다 (3주차 visualize_metrics 방식 재활용)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = [r["model"].replace("-cls.pt", "") for r in results]
    top1 = [r["top1"] for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, top1, color="#E5484D")
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("YOLO Classification — Version Comparison")
    for b, v in zip(bars, top1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
    path = out / "model_comparison_chart.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def compare_models(data_dir, candidates=None, epochs=10, imgsz=224, seed=42):
    """후보 모델들을 동일 조건으로 학습·검증해 비교 결과 리스트를 반환한다."""
    from ultralytics import YOLO  # 지연 임포트

    candidates = candidates or DEFAULT_CANDIDATES
    results = []
    for name in candidates:
        print(f"\n===== {name} 학습 시작 (epochs={epochs}, imgsz={imgsz}) =====")
        model = YOLO(name)
        t0 = time.time()
        model.train(data=data_dir, epochs=epochs, imgsz=imgsz, seed=seed,
                    project="runs_compare", name=name.replace(".pt", ""), verbose=False)
        train_sec = round(time.time() - t0, 1)
        metrics = model.val()
        results.append({
            "model": name,
            "top1": round(float(metrics.top1), 4),
            "top5": round(float(metrics.top5), 4),
            "train_sec": train_sec,
        })
        print(f"===== {name}: top1={results[-1]['top1']} · {train_sec}s =====")
    return results
