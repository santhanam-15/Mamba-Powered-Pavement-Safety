import math
from typing import Dict, Tuple

import cv2
import numpy as np

from config import DEFAULT_CAMERA_ANGLE_DEG, DEFAULT_CAMERA_HEIGHT_MM


def scale_coefficient(height_mm: float) -> float:
    return 4.5019e-3 - 1.3980e-5 * height_mm + 1.0382e-6 * (height_mm ** 2)


def field_of_view_deg(height_mm: float) -> float:
    return 7.6341 * (height_mm ** 0.0253)


def _half_fov_tangent(height_mm: float) -> float:
    return math.tan(math.radians(field_of_view_deg(height_mm) / 2.0))


def _solve_monotonic_root(function, low: float, high: float, max_iterations: int = 200, tolerance: float = 1e-8) -> float:
    f_low = function(low)
    f_high = function(high)
    if f_low == 0:
        return low
    if f_high == 0:
        return high
    if f_low * f_high > 0:
        raise ValueError("Root is not bracketed in the provided interval.")

    for _ in range(max_iterations):
        midpoint = (low + high) / 2.0
        f_mid = function(midpoint)
        if abs(f_mid) < tolerance or abs(high - low) < tolerance:
            return midpoint
        if f_low * f_mid < 0:
            high = midpoint
            f_high = f_mid
        else:
            low = midpoint
            f_low = f_mid
    return (low + high) / 2.0


def solve_pc1(camera_height_mm: float, camera_angle_deg: float) -> float:
    theta = math.radians(camera_angle_deg)

    def equation(value: float) -> float:
        return value + value * math.tan(theta) * _half_fov_tangent(value) - camera_height_mm

    return _solve_monotonic_root(equation, 1e-6, max(camera_height_mm, 1.0))


def solve_pc2(camera_height_mm: float, camera_angle_deg: float) -> float:
    theta = math.radians(camera_angle_deg)

    def equation(value: float) -> float:
        return value - value * math.tan(theta) * _half_fov_tangent(value) - camera_height_mm

    low = max(camera_height_mm, 1.0)
    high = low * 2.0
    while equation(high) < 0:
        high *= 2.0
        if high > 1e7:
            raise ValueError("Failed to bracket a root for PC2.")
    return _solve_monotonic_root(equation, low, high)


def build_stage1_homography(
    image_width: int,
    image_height: int,
    camera_height_mm: float = DEFAULT_CAMERA_HEIGHT_MM,
    camera_angle_deg: float = DEFAULT_CAMERA_ANGLE_DEG,
) -> Tuple[np.ndarray, Tuple[int, int], Dict[str, float]]:
    theta = math.radians(camera_angle_deg)
    pc0 = camera_height_mm
    pc1 = solve_pc1(pc0, camera_angle_deg)
    pc2 = solve_pc2(pc0, camera_angle_deg)

    r0 = scale_coefficient(pc0)
    r1 = scale_coefficient(pc1)
    r2 = scale_coefficient(pc2)

    x_right = (pc2 - pc1) / (math.sin(theta) * r0)
    y_bottom_left = image_height * (r2 / r0)
    y_top_right = (image_height / 2.0) * ((r2 - r1) / r0)
    y_bottom_right = (image_height / 2.0) * ((r2 + r1) / r0)

    src_points = np.float32(
        [
            [0, 0],
            [0, image_height - 1],
            [image_width - 1, 0],
            [image_width - 1, image_height - 1],
        ]
    )
    dst_points = np.float32(
        [
            [0, 0],
            [0, y_bottom_left],
            [x_right, y_top_right],
            [x_right, y_bottom_right],
        ]
    )

    min_x = float(np.min(dst_points[:, 0]))
    min_y = float(np.min(dst_points[:, 1]))
    dst_points[:, 0] -= min_x
    dst_points[:, 1] -= min_y

    output_width = int(math.ceil(float(np.max(dst_points[:, 0])))) + 1
    output_height = int(math.ceil(float(np.max(dst_points[:, 1])))) + 1

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    metadata = {
        "camera_height_mm": pc0,
        "camera_angle_deg": camera_angle_deg,
        "pc1_mm": pc1,
        "pc2_mm": pc2,
        "r0_mm_per_px": r0,
        "r1_mm_per_px": r1,
        "r2_mm_per_px": r2,
    }
    return matrix, (output_width, output_height), metadata


def crop_valid_region(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if image.ndim == 3:
        valid_mask = np.any(image > 0, axis=2)
    else:
        valid_mask = image > 0

    if not np.any(valid_mask):
        height, width = image.shape[:2]
        return image, (0, 0, width, height)

    ys, xs = np.where(valid_mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def correct_stage1_image(
    image: np.ndarray,
    camera_height_mm: float = DEFAULT_CAMERA_HEIGHT_MM,
    camera_angle_deg: float = DEFAULT_CAMERA_ANGLE_DEG,
) -> tuple[np.ndarray, Dict[str, float]]:
    matrix, output_size, metadata = build_stage1_homography(
        image_width=image.shape[1],
        image_height=image.shape[0],
        camera_height_mm=camera_height_mm,
        camera_angle_deg=camera_angle_deg,
    )
    corrected = cv2.warpPerspective(
        image,
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    corrected, _ = crop_valid_region(corrected)
    return corrected, metadata
