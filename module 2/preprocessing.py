import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from config import (
    DEFAULT_CAMERA_ANGLE_DEG,
    DEFAULT_CAMERA_HEIGHT_MM,
    IMG_SIZE,
    STAGE1_OUTPUT_PATH,
)


def get_perspective_matrix(
    width: int,
    height: int,
    jitter: float = 0.08,
) -> np.ndarray:
    """Create a mild random perspective matrix for a given image size."""
    max_dx = width * jitter
    max_dy = height * jitter

    source = np.float32(
        [
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1],
        ]
    )

    destination = np.float32(
        [
            [random.uniform(0, max_dx), random.uniform(0, max_dy)],
            [width - 1 - random.uniform(0, max_dx), random.uniform(0, max_dy)],
            [random.uniform(0, max_dx), height - 1 - random.uniform(0, max_dy)],
            [width - 1 - random.uniform(0, max_dx), height - 1 - random.uniform(0, max_dy)],
        ]
    )

    return cv2.getPerspectiveTransform(source, destination)


def resize(image: np.ndarray, mask: np.ndarray, size: int = IMG_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """Resize image and mask to the same spatial size."""
    resized_image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask


def normalize(image: np.ndarray) -> np.ndarray:
    """Scale image pixels to the [0, 1] range."""
    return image.astype(np.float32) / 255.0


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert grayscale mask to a binary mask."""
    threshold = 0 if mask.max() <= 1 else 127
    return (mask > threshold).astype(np.uint8)


def combine_masks(mask_list: Iterable[np.ndarray]) -> np.ndarray:
    """Combine multiple binary masks with a logical OR."""
    masks = list(mask_list)
    if not masks:
        raise ValueError("mask_list must contain at least one mask.")

    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask)
    return combined_mask.astype(np.uint8)


def perspective_transform(
    image: np.ndarray,
    mask: np.ndarray,
    jitter: float = 0.08,
    matrix: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the same random perspective warp to image and mask.

    The transform is intentionally mild to preserve road-structure realism.
    """
    height, width = image.shape[:2]
    if matrix is None:
        matrix = get_perspective_matrix(width, height, jitter=jitter)

    warped_image = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped_mask = cv2.warpPerspective(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_image, warped_mask


def scale_coefficient(height_mm: float) -> float:
    """Equation (1) from the paper: physical size / pixel size."""
    return 4.5019e-3 - 1.3980e-5 * height_mm + 1.0382e-6 * (height_mm ** 2)


def field_of_view_deg(height_mm: float) -> float:
    """Equation (2) from the paper."""
    return 7.6341 * (height_mm ** 0.0253)


def _half_fov_tangent(height_mm: float) -> float:
    half_fov_deg = field_of_view_deg(height_mm) / 2.0
    return math.tan(math.radians(half_fov_deg))


def _solve_monotonic_root(
    function,
    low: float,
    high: float,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> float:
    """Solve a root in [low, high] using bisection."""
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
    """Solve equation (8) from the paper."""
    theta = math.radians(camera_angle_deg)

    def equation(value: float) -> float:
        return value + value * math.tan(theta) * _half_fov_tangent(value) - camera_height_mm

    lower_bound = 1e-6
    upper_bound = max(camera_height_mm, 1.0)
    return _solve_monotonic_root(equation, lower_bound, upper_bound)


def solve_pc2(camera_height_mm: float, camera_angle_deg: float) -> float:
    """Solve equation (9) from the paper."""
    theta = math.radians(camera_angle_deg)

    def equation(value: float) -> float:
        return value - value * math.tan(theta) * _half_fov_tangent(value) - camera_height_mm

    lower_bound = max(camera_height_mm, 1.0)
    upper_bound = lower_bound * 2.0
    while equation(upper_bound) < 0:
        upper_bound *= 2.0
        if upper_bound > 1e7:
            raise ValueError("Failed to bracket a root for PC2.")

    return _solve_monotonic_root(equation, lower_bound, upper_bound)


def build_stage1_homography(
    image_width: int,
    image_height: int,
    camera_height_mm: float = DEFAULT_CAMERA_HEIGHT_MM,
    camera_angle_deg: float = DEFAULT_CAMERA_ANGLE_DEG,
) -> Tuple[np.ndarray, Tuple[int, int], Dict[str, float]]:
    """
    Build the stage-1 perspective correction transform from the paper.

    The original publication uses 512x512 images. This implementation
    generalizes the destination geometry to arbitrary image height while
    preserving the same perspective-correction derivation.
    """
    if camera_height_mm <= 0:
        raise ValueError("camera_height_mm must be positive.")
    if not (0 < camera_angle_deg < 89):
        raise ValueError("camera_angle_deg must be between 0 and 89 degrees.")

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

    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    metadata = {
        "camera_height_mm": pc0,
        "camera_angle_deg": camera_angle_deg,
        "fov_deg": field_of_view_deg(pc0),
        "pc1_mm": pc1,
        "pc2_mm": pc2,
        "r0_mm_per_px": r0,
        "r1_mm_per_px": r1,
        "r2_mm_per_px": r2,
        "output_width": output_width,
        "output_height": output_height,
    }
    return transform_matrix, (output_width, output_height), metadata


def correct_stage1_image(
    image: np.ndarray,
    camera_height_mm: float = DEFAULT_CAMERA_HEIGHT_MM,
    camera_angle_deg: float = DEFAULT_CAMERA_ANGLE_DEG,
    is_mask: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply the paper's stage-1 perspective correction to an image or mask."""
    height, width = image.shape[:2]
    matrix, output_size, metadata = build_stage1_homography(
        image_width=width,
        image_height=height,
        camera_height_mm=camera_height_mm,
        camera_angle_deg=camera_angle_deg,
    )
    corrected = cv2.warpPerspective(
        image,
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return corrected, metadata


def warp_with_stage1_homography(
    image: np.ndarray,
    transform_matrix: np.ndarray,
    output_size: Tuple[int, int],
    is_mask: bool = False,
) -> np.ndarray:
    """Warp an image or mask with a precomputed stage-1 homography."""
    return cv2.warpPerspective(
        image,
        transform_matrix,
        output_size,
        flags=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def crop_valid_region(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Remove zero-valued padding introduced by perspective warping.

    Returns the cropped image and the bounding box in (x0, y0, x1, y1) form.
    """
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


def _iter_stage1_groups(input_root: Path) -> List[Dict[str, Path]]:
    grouped: Dict[str, Dict[str, Path]] = {}
    for file_path in input_root.rglob("*"):
        if not file_path.is_file():
            continue
        stem = file_path.stem
        if "_" not in stem:
            continue
        prefix, suffix = stem.rsplit("_", 1)
        suffix = suffix.upper()
        if suffix not in {"RAW", "CRACK", "POTHOLE", "LANE"}:
            continue
        grouped.setdefault(prefix, {})[suffix] = file_path

    samples: List[Dict[str, Path]] = []
    for prefix, files in grouped.items():
        if "RAW" not in files:
            continue
        sample: Dict[str, Path] = {"id": Path(prefix)}
        sample.update({key.lower(): value for key, value in files.items()})
        samples.append(sample)

    samples.sort(key=lambda item: str(item["id"]))
    return samples


def run_stage1_correction(
    input_root: Path,
    output_root: Path = Path(STAGE1_OUTPUT_PATH),
    camera_height_mm: float = DEFAULT_CAMERA_HEIGHT_MM,
    camera_angle_deg: float = DEFAULT_CAMERA_ANGLE_DEG,
) -> Path:
    """
    Batch-correct the dataset with the stage-1 perspective transform.

    The original folder hierarchy is preserved under the output root, and
    any available masks are warped with the same homography as the raw image
    so alignment remains intact for downstream segmentation training.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = _iter_stage1_groups(input_root)
    if not samples:
        raise FileNotFoundError(f"No RAW samples found under '{input_root}'.")

    metadata_rows = []
    for index, sample in enumerate(samples, start=1):
        raw_path = sample["raw"]
        raw_image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw_image is None:
            raise ValueError(f"Failed to read image: {raw_path}")

        transform_matrix, output_size, metadata = build_stage1_homography(
            image_width=raw_image.shape[1],
            image_height=raw_image.shape[0],
            camera_height_mm=camera_height_mm,
            camera_angle_deg=camera_angle_deg,
        )
        corrected_raw = warp_with_stage1_homography(
            raw_image,
            transform_matrix=transform_matrix,
            output_size=output_size,
            is_mask=False,
        )
        corrected_raw, crop_box = crop_valid_region(corrected_raw)

        relative_dir = raw_path.parent.relative_to(input_root)
        sample_output_dir = output_root / relative_dir
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        raw_output_path = sample_output_dir / raw_path.name
        cv2.imwrite(str(raw_output_path), corrected_raw)

        for mask_key in ("crack", "pothole", "lane"):
            mask_path = sample.get(mask_key)
            if mask_path is None:
                continue
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask_image is None:
                raise ValueError(f"Failed to read mask: {mask_path}")
            corrected_mask = warp_with_stage1_homography(
                mask_image,
                transform_matrix=transform_matrix,
                output_size=output_size,
                is_mask=True,
            )
            x0, y0, x1, y1 = crop_box
            corrected_mask = corrected_mask[y0:y1, x0:x1]
            cv2.imwrite(str(sample_output_dir / mask_path.name), corrected_mask)

        metadata_rows.append(
            {
                "sample_id": raw_path.stem.rsplit("_", 1)[0],
                "source_raw": str(raw_path),
                "corrected_raw": str(raw_output_path),
                "camera_height_mm": metadata["camera_height_mm"],
                "camera_angle_deg": metadata["camera_angle_deg"],
                "fov_deg": metadata["fov_deg"],
                "pc1_mm": metadata["pc1_mm"],
                "pc2_mm": metadata["pc2_mm"],
                "r0_mm_per_px": metadata["r0_mm_per_px"],
                "r1_mm_per_px": metadata["r1_mm_per_px"],
                "r2_mm_per_px": metadata["r2_mm_per_px"],
                "output_width": corrected_raw.shape[1],
                "output_height": corrected_raw.shape[0],
                "crop_x0": crop_box[0],
                "crop_y0": crop_box[1],
                "crop_x1": crop_box[2],
                "crop_y1": crop_box[3],
            }
        )

    metadata_path = output_root / "stage1_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    return output_root
