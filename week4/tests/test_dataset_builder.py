import pytest

from src.dataset_builder import (CLASS_NAMES, assign_split, collection_done,
                                 sanitize_class_name)


def test_class_names_count():
    assert len(CLASS_NAMES) == 10


def test_sanitize_class_name():
    assert sanitize_class_name("Round Smooth") == "Round_Smooth"
    assert sanitize_class_name(" Edge-on/Bulge ") == "Edge-on-Bulge"
    with pytest.raises(ValueError):
        sanitize_class_name("")


def test_assign_split_deterministic():
    # per_class=10, val_ratio=0.2 → 0~7 train, 8~9 val
    splits = [assign_split(i, per_class=10, val_ratio=0.2) for i in range(10)]
    assert splits.count("train") == 8
    assert splits.count("val") == 2
    assert splits[:8] == ["train"] * 8


def test_assign_split_invalid_args():
    with pytest.raises(ValueError):
        assign_split(0, per_class=0)
    with pytest.raises(ValueError):
        assign_split(0, per_class=10, val_ratio=1.5)


def test_collection_done():
    assert collection_done({0: 5, 1: 5}, per_class=5) is True
    assert collection_done({0: 5, 1: 4}, per_class=5) is False
    with pytest.raises(ValueError):
        collection_done(None, per_class=5)
