"""GalaxyLens CLI — Galaxy10 DECaLS 기반 은하 형태 분류.

단계별 사용법:
  1) 데이터 구축:  python app.py build-data --per-class 200
  2) 버전 비교:    python app.py compare --epochs 10
  3) 본 학습:      python app.py train --model yolo11n-cls.pt --epochs 30
  4) 예측:         python app.py predict --image data/galaxy10/val/Merging/xxx.jpg
"""
import argparse
from pathlib import Path

import cv2


def cmd_build(args):
    from src.dataset_builder import build_dataset
    build_dataset(output_dir=args.data_dir, per_class=args.per_class, val_ratio=args.val_ratio)


def cmd_compare(args):
    from src.model_compare import compare_models, pick_best, save_comparison_csv, save_comparison_chart
    results = compare_models(args.data_dir, epochs=args.epochs, imgsz=args.imgsz)
    print("\n[비교 결과]")
    for r in results:
        print(f"  {r['model']}: top1={r['top1']} top5={r['top5']} {r['train_sec']}s")
    best = pick_best(results)
    print(f"\n[우승 모델] {best['model']} (top1={best['top1']})")
    print("저장:", save_comparison_csv(results, args.output_dir))
    print("저장:", save_comparison_chart(results, args.output_dir))


def cmd_train(args):
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=args.data_dir, epochs=args.epochs, imgsz=args.imgsz, seed=42)
    metrics = model.val()
    print(f"[본 학습 완료] {args.model} — top1={metrics.top1:.4f} top5={metrics.top5:.4f}")


def cmd_predict(args):
    from src.classifier import GalaxyClassifier, annotate
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {args.image}")
    clf = GalaxyClassifier(model_path=args.weights)
    pred = clf.predict(image)
    print(f"[예측] {pred['label']} ({pred['confidence']:.2%})")
    for name, conf in pred["topk"]:
        print(f"  - {name}: {conf:.2%}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"pred_{Path(args.image).stem}.jpg"
    cv2.imwrite(str(out_path), annotate(image, pred))
    print(f"[저장] {out_path}")


def main():
    p = argparse.ArgumentParser(description="GalaxyLens: 은하 형태 분류")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-data")
    b.add_argument("--data-dir", default="data/galaxy10")
    b.add_argument("--per-class", type=int, default=200)
    b.add_argument("--val-ratio", type=float, default=0.2)
    b.set_defaults(fn=cmd_build)

    c = sub.add_parser("compare")
    c.add_argument("--data-dir", default="data/galaxy10")
    c.add_argument("--output-dir", default="outputs")
    c.add_argument("--epochs", type=int, default=10)
    c.add_argument("--imgsz", type=int, default=224)
    c.set_defaults(fn=cmd_compare)

    t = sub.add_parser("train")
    t.add_argument("--model", default="yolo11n-cls.pt")
    t.add_argument("--data-dir", default="data/galaxy10")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--imgsz", type=int, default=224)
    t.set_defaults(fn=cmd_train)

    d = sub.add_parser("predict")
    d.add_argument("--image", required=True)
    d.add_argument("--weights", default="runs/classify/train/weights/best.pt")
    d.add_argument("--output-dir", default="outputs")
    d.set_defaults(fn=cmd_predict)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
