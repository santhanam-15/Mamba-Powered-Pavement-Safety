from pathlib import Path

import cv2
import numpy as np
import torch

from config import (
    CHECKPOINT_PATH,
    DEFAULT_CAMERA_ANGLE_DEG,
    DEFAULT_CAMERA_HEIGHT_MM,
    IMG_SIZE,
    PREDICTION_DIR,
)
from model import CMSegNet
from stage1_utils import correct_stage1_image


def load_model(checkpoint_path: Path, device: torch.device) -> CMSegNet:
    model = CMSegNet(img_size=IMG_SIZE).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and not checkpoint.get("trained", False):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' is not a trained model. Run train.py first."
        )
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    threshold = checkpoint.get("best_threshold", 0.5) if isinstance(checkpoint, dict) else 0.5
    return model, float(threshold)


def preprocess_image(image_path: Path) -> tuple[np.ndarray, torch.Tensor, dict[str, float]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read input image: {image_path}")

    corrected_bgr, metadata = correct_stage1_image(
        image,
        camera_height_mm=DEFAULT_CAMERA_HEIGHT_MM,
        camera_angle_deg=DEFAULT_CAMERA_ANGLE_DEG,
    )
    corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(corrected_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy((resized.astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).float()
    return corrected_rgb, tensor, metadata


def save_area_summary(output_dir: Path, white_pixels: int, r_value: float) -> tuple[Path, Path]:
    area_mm2 = white_pixels * r_value
    area_cm2 = area_mm2 / 100.0

    summary_txt_path = output_dir / "area_summary.txt"
    with summary_txt_path.open("w", encoding="utf-8") as handle:
        handle.write(f"white_pixels={white_pixels}\n")
        handle.write(f"r={r_value:.8f}\n")
        handle.write(f"area_mm2={area_mm2:.4f}\n")
        handle.write(f"area_cm2={area_cm2:.4f}\n")

    summary_img = np.full((220, 1100, 3), 255, dtype=np.uint8)
    cv2.putText(summary_img, f"White Pixels: {white_pixels}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(summary_img, f"r: {r_value:.8f}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(summary_img, f"Area: {area_cm2:.4f} cm2", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 60, 170), 2, cv2.LINE_AA)

    summary_img_path = output_dir / "area_summary.png"
    cv2.imwrite(str(summary_img_path), summary_img)
    return summary_txt_path, summary_img_path


def save_prediction_outputs(original_bgr: np.ndarray, corrected_rgb: np.ndarray, mask: np.ndarray, image_path: Path) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(PREDICTION_DIR) / image_path.stem / "everything"
    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / "original.png"
    corrected_path = output_dir / "corrected.png"
    bitmask_path = output_dir / "predicted_bitmask.png"
    overlay_path = output_dir / "overlay.png"
    corrected_bgr = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
    overlay = corrected_bgr.copy()
    red_layer = np.zeros_like(corrected_bgr, dtype=np.uint8)
    red_layer[:, :, 2] = 255
    region = mask > 0
    if np.any(region):
        overlay[region] = cv2.addWeighted(overlay[region], 0.65, red_layer[region], 0.35, 0)
    cv2.imwrite(str(original_path), original_bgr)
    cv2.imwrite(str(corrected_path), corrected_bgr)
    cv2.imwrite(str(bitmask_path), mask)
    cv2.imwrite(str(overlay_path), overlay)
    return original_path, corrected_path, bitmask_path, overlay_path


def predict_image(image_path: str) -> tuple[Path, Path, Path, Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found. Train the model first with train.py."
        )

    print(f"Using device: {device}")
    model, threshold = load_model(checkpoint_path, device)
    image_path_obj = Path(image_path)
    original_bgr = cv2.imread(str(image_path_obj), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise FileNotFoundError(f"Unable to read input image: {image_path_obj}")
    corrected_rgb, tensor, metadata = preprocess_image(image_path_obj)

    with torch.no_grad():
        logits, _ = model(tensor.to(device))
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    bitmask = (probs > threshold).astype(np.uint8) * 255
    bitmask = cv2.resize(bitmask, (corrected_rgb.shape[1], corrected_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    original_path, corrected_path, bitmask_path, overlay_path = save_prediction_outputs(original_bgr, corrected_rgb, bitmask, image_path_obj)
    white_pixels = int(np.count_nonzero(bitmask))
    r_value = float(metadata.get("r0_mm_per_px", 0.0))
    area_txt_path, area_img_path = save_area_summary(Path(PREDICTION_DIR) / image_path_obj.stem / "everything", white_pixels, r_value)
    print(f"Saved original image to: {original_path}")
    print(f"Saved corrected image to: {corrected_path}")
    print(f"Saved predicted bitmask to: {bitmask_path}")
    print(f"Saved overlay image to: {overlay_path}")
    print(f"Saved area summary text to: {area_txt_path}")
    print(f"Saved area summary image to: {area_img_path}")
    return original_path, corrected_path, bitmask_path, overlay_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: py predict.py <path_to_road_image>")

    predict_image(sys.argv[1])
