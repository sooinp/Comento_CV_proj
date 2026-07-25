import numpy as np
import pytest

from src.classifier import GalaxyClassifier, annotate


def _dummy_image():
    return np.zeros((256, 256, 3), dtype=np.uint8)


def test_predict_none_raises():
    clf = GalaxyClassifier(model=object())  # 가짜 모델 주입 → 실제 로드 없음
    with pytest.raises(ValueError):
        clf.predict(None)


def test_predict_invalid_topk_raises():
    clf = GalaxyClassifier(model=object())
    with pytest.raises(ValueError):
        clf.predict(_dummy_image(), topk=0)


def test_annotate_shape_and_nondestructive():
    image = _dummy_image()
    pred = {"label": "Merging", "confidence": 0.93, "topk": [("Merging", 0.93)]}
    out = annotate(image, pred)
    assert out.shape == image.shape and out.dtype == image.dtype
    assert image.sum() == 0  # 원본 불변


def test_annotate_none_raises():
    with pytest.raises(ValueError):
        annotate(None, {"label": "x", "confidence": 0.5})
    with pytest.raises(ValueError):
        annotate(_dummy_image(), None)
