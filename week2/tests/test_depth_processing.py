import numpy as np
import pytest

from src.depth_processing import (
    to_grayscale,
    generate_depth_map,
    create_point_cloud,
)


def make_sample_image(height=80, width=100):
    """테스트용 BGR 샘플 이미지를 생성한다."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = 50
    image[:, :, 1] = 120
    image[:, :, 2] = 200
    return image


def test_to_grayscale_returns_2d_array():
    image = make_sample_image()

    gray = to_grayscale(image)

    assert isinstance(gray, np.ndarray)
    assert gray.ndim == 2
    assert gray.shape == image.shape[:2]


def test_generate_depth_map_keeps_image_size_and_has_three_channels():
    image = make_sample_image()

    depth_map = generate_depth_map(image)

    assert isinstance(depth_map, np.ndarray)
    assert depth_map.shape[:2] == image.shape[:2]
    assert depth_map.shape[2] == 3


def test_create_point_cloud_returns_xyz_points():
    image = make_sample_image(height=20, width=30)

    points = create_point_cloud(image, stride=5)

    assert isinstance(points, np.ndarray)
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert len(points) > 0


def test_none_image_raises_value_error():
    with pytest.raises(ValueError):
        to_grayscale(None)


def test_invalid_stride_raises_value_error():
    image = make_sample_image()

    with pytest.raises(ValueError):
        create_point_cloud(image, stride=0)
