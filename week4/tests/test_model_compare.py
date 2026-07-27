import csv

import pytest

from src.model_compare import pick_best, save_comparison_csv


RESULTS = [
    {"model": "yolov8n-cls.pt", "top1": 0.81, "top5": 0.98, "train_sec": 300.0},
    {"model": "yolo11n-cls.pt", "top1": 0.84, "top5": 0.99, "train_sec": 280.0},
]


def test_pick_best_by_top1():
    assert pick_best(RESULTS)["model"] == "yolo11n-cls.pt"


def test_pick_best_tie_breaks_by_time():
    tied = [
        {"model": "a", "top1": 0.8, "top5": 0.9, "train_sec": 300.0},
        {"model": "b", "top1": 0.8, "top5": 0.9, "train_sec": 200.0},
    ]
    assert pick_best(tied)["model"] == "b"


def test_pick_best_empty_raises():
    with pytest.raises(ValueError):
        pick_best([])


def test_save_comparison_csv(tmp_path):
    path = save_comparison_csv(RESULTS, tmp_path)
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[1]["model"] == "yolo11n-cls.pt"


def test_save_comparison_none_raises():
    with pytest.raises(ValueError):
        save_comparison_csv(None)
