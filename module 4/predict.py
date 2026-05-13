from pathlib import Path
from typing import Optional

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
from model import CMSegNet, ShadowNet
from stage1_utils import correct_stage1_image

cmssegnet: Optional[CMSegNet] = None
shadow_net = ShadowNet()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[CMSegNet, float]:
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
    global cmssegnet
    cmssegnet = model
    return model, float(threshold)


def forward_module4(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if cmssegnet is None:
        raise RuntimeError("cmssegnet is not initialized. Call load_model() first.")

    target_device = image.device
    if next(cmssegnet.parameters()).device != target_device:
        cmssegnet.to(target_device)
    if next(shadow_net.parameters()).device != target_device:
        shadow_net.to(target_device)
    shadow_net.eval()

    pothole_logits, _ = cmssegnet(image)
    pothole_mask = torch.sigmoid(pothole_logits)
    shadow_mask = shadow_net(image)

    # Suppress only high-confidence shadow areas to avoid deleting weak true positives.
    shadow_conf = torch.clamp((shadow_mask - 0.55) / 0.45, min=0.0, max=1.0)
    final_mask = pothole_mask * (1 - 0.85 * shadow_conf)
    overlay = image * final_mask
    return final_mask, pothole_mask, shadow_mask, overlay


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


def create_transparent_red_overlay(base_bgr: np.ndarray, bitmask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    overlay_bgr = base_bgr.copy()
    red_layer = np.zeros_like(base_bgr, dtype=np.uint8)
    red_layer[:, :, 2] = 255
    # Blend only on predicted pixels so the original image remains visible.
    region = bitmask > 0
    if np.any(region):
        blended = cv2.addWeighted(overlay_bgr[region], 1.0 - alpha, red_layer[region], alpha, 0)
        overlay_bgr[region] = blended
    return overlay_bgr


def postprocess_bitmask(bitmask: np.ndarray, min_component_area: int = 40, border_min_area: int = 300) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(bitmask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    filtered = np.zeros_like(opened)
    height, width = opened.shape[:2]

    for component_idx in range(1, num_labels):
        x = int(stats[component_idx, cv2.CC_STAT_LEFT])
        y = int(stats[component_idx, cv2.CC_STAT_TOP])
        comp_width = int(stats[component_idx, cv2.CC_STAT_WIDTH])
        comp_height = int(stats[component_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_idx, cv2.CC_STAT_AREA])

        elongated_component = max(comp_width, comp_height) / max(1, min(comp_width, comp_height)) >= 3.5
        if area < min_component_area and not (elongated_component and area >= 20):
            continue

        touches_border = x == 0 or y == 0 or (x + comp_width) >= width or (y + comp_height) >= height
        if touches_border and area < border_min_area and not elongated_component:
            continue

        filtered[labels == component_idx] = 255

    return filtered


def resolve_output_dir(image_path: Path, output_dir: Optional[Path] = None) -> Path:
    if output_dir is not None:
        return output_dir
    return Path(PREDICTION_DIR) / image_path.stem


def save_prediction_outputs(
    original_bgr: np.ndarray,
    corrected_bgr: np.ndarray,
    bitmask: np.ndarray,
    overlay_bgr: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / "original.png"
    corrected_path = output_dir / "corrected.png"
    bitmask_path = output_dir / "predicted_bitmask.png"
    overlay_path = output_dir / "overlay.png"
    cv2.imwrite(str(original_path), original_bgr)
    cv2.imwrite(str(corrected_path), corrected_bgr)
    cv2.imwrite(str(bitmask_path), bitmask)
    cv2.imwrite(str(overlay_path), overlay_bgr)
    return original_path, corrected_path, bitmask_path, overlay_path


def predict_image(
    image_path: str,
    checkpoint_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> tuple[Path, Path, Path, Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_checkpoint_path = checkpoint_path if checkpoint_path is not None else Path(CHECKPOINT_PATH)
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint '{resolved_checkpoint_path}' not found. Train the model first with train.py."
        )

    print(f"Using device: {device}")
    _, threshold = load_model(resolved_checkpoint_path, device)
    image_path_obj = Path(image_path)
    original_bgr = cv2.imread(str(image_path_obj), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise FileNotFoundError(f"Unable to read input image: {image_path_obj}")

    corrected_rgb, tensor, metadata = preprocess_image(image_path_obj)
    corrected_bgr = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        final_mask, pothole_mask, shadow_mask, _ = forward_module4(tensor.to(device))
        shadow_mean = float(shadow_mask.mean().item())
        shadow_std = float(shadow_mask.std(unbiased=False).item())
        # Untrained ShadowNet often outputs near-constant ~0.5, which over-suppresses pothole predictions.
        if abs(shadow_mean - 0.5) < 0.08 and shadow_std < 0.06:
            final_mask = pothole_mask
            print("ShadowNet appears untrained (near-constant output). Using CMSegNet mask for stable results.")
        probs = final_mask.squeeze().cpu().numpy()

    bitmask = (probs > threshold).astype(np.uint8) * 255
    bitmask = cv2.resize(bitmask, (corrected_bgr.shape[1], corrected_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    bitmask = postprocess_bitmask(bitmask)
    overlay_bgr = create_transparent_red_overlay(corrected_bgr, bitmask)
    resolved_output_dir = resolve_output_dir(image_path_obj, output_dir)
    original_path, corrected_path, bitmask_path, overlay_path = save_prediction_outputs(
        original_bgr,
        corrected_bgr,
        bitmask,
        overlay_bgr,
        output_dir=resolved_output_dir,
    )
    white_pixels = int(np.count_nonzero(bitmask))
    r_value = float(metadata.get("r0_mm_per_px", 0.0))
    area_txt_path, area_img_path = save_area_summary(resolved_output_dir, white_pixels, r_value)
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
