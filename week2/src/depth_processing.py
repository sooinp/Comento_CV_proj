from pathlib import Path

import cv2
import numpy as np


def load_image(image_path):
    """이미지 파일을 OpenCV BGR 형식으로 로드한다."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없다: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"이미지를 읽을 수 없다. 파일 형식을 확인해야 한다: {path}")

    return image


def _validate_image_array(image):
    """입력 이미지 배열이 처리 가능한 형태인지 확인한다."""
    if image is None:
        raise ValueError("입력 이미지가 None이다.")

    if not isinstance(image, np.ndarray):
        raise TypeError("입력 이미지는 numpy.ndarray 타입이어야 한다.")

    if image.ndim not in (2, 3):
        raise ValueError("입력 이미지는 2D Grayscale 또는 3D BGR 이미지여야 한다.")

    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError("3D 이미지는 BGR 3채널이어야 한다.")

    return image


def to_grayscale(image):
    """BGR 이미지 또는 Grayscale 이미지를 Grayscale 2D 배열로 변환한다."""
    image = _validate_image_array(image)

    if image.ndim == 2:
        return image.copy()

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def generate_depth_map(image, colormap=cv2.COLORMAP_JET):
    """Grayscale 밝기 값을 기반으로 가상의 Depth Map 이미지를 생성한다."""
    gray = to_grayscale(image)

    # 입력 이미지의 밝기 범위를 0~255로 정규화한다.
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)

    # 밝기 값에 색상 맵을 적용하여 Depth Map처럼 시각화한다.
    depth_map = cv2.applyColorMap(normalized, colormap)

    return depth_map


def create_point_cloud(image, stride=4):
    """이미지의 x, y 좌표와 밝기 값을 z로 사용하여 간단한 3D 포인트 배열을 생성한다.

    Args:
        image: BGR 또는 Grayscale 이미지 배열
        stride: 모든 픽셀을 쓰면 포인트가 너무 많으므로 샘플링 간격을 지정한다.

    Returns:
        shape이 (N, 3)인 numpy.ndarray이다. 각 행은 [x, y, z]를 의미한다.
    """
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride는 1 이상의 정수여야 한다.")

    gray = to_grayscale(image).astype(np.float32)
    sampled_gray = gray[::stride, ::stride]

    height, width = sampled_gray.shape
    y_grid, x_grid = np.mgrid[0:height, 0:width]

    x = (x_grid * stride).astype(np.float32)
    y = (y_grid * stride).astype(np.float32)
    z = sampled_gray

    points_3d = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    return points_3d


def save_outputs(image_path, output_dir, stride=4):
    """입력 이미지로부터 Grayscale, Depth Map, Point Cloud 결과물을 저장한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    image = load_image(image_path)
    gray = to_grayscale(image)
    depth_map = generate_depth_map(image)
    points_3d = create_point_cloud(image, stride=stride)

    grayscale_path = output_dir / "grayscale.png"
    depth_map_path = output_dir / "depth_map.png"
    point_cloud_path = output_dir / "point_cloud.npy"

    cv2.imwrite(str(grayscale_path), gray)
    cv2.imwrite(str(depth_map_path), depth_map)
    np.save(str(point_cloud_path), points_3d)

    return {
        "grayscale_path": str(grayscale_path),
        "depth_map_path": str(depth_map_path),
        "point_cloud_path": str(point_cloud_path),
        "point_cloud_shape": points_3d.shape,
    }
