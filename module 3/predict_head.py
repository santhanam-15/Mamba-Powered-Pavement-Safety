from pathlib import Path

import cv2
import numpy as np
import torch

from config import (
    CHECKPOINT_PATH,
    DEFAULT_CAMERA_ANGLE_DEG,
    DEFAULT_CAMERA_HEIGHT_MM,
    IMG_SIZE,
    NUM_CLASSES,
    PREDICTION_DIR,
)
from model import CMSegNet
from stage1_utils import correct_stage1_image


def load_model(checkpoint_path: Path, device: torch.device) -> CMSegNet:
    model = CMSegNet(img_size=IMG_SIZE, out_channels=NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and not checkpoint.get("trained", False):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' is not a trained model. Run train.py first."
        )
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path: Path) -> tuple[np.ndarray, torch.Tensor]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read input image: {image_path}")

    corrected_bgr, _ = correct_stage1_image(
        image,
        camera_height_mm=DEFAULT_CAMERA_HEIGHT_MM,
        camera_angle_deg=DEFAULT_CAMERA_ANGLE_DEG,
    )
    corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(corrected_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy((resized.astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).float()
    return corrected_rgb, tensor


def save_prediction_outputs(corrected_rgb: np.ndarray, mask: np.ndarray, image_path: Path) -> tuple[Path, Path]:
    output_dir = Path(PREDICTION_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_path = output_dir / f"{image_path.stem}_corrected.png"
    bitmask_path = output_dir / f"{image_path.stem}_bitmask.png"
    cv2.imwrite(str(corrected_path), cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(bitmask_path), mask)
    return corrected_path, bitmask_path


def predict_image(image_path: str) -> tuple[Path, Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found. Train the model first with train.py."
        )

    print(f"Using device: {device}")
    model = load_model(checkpoint_path, device)
    corrected_rgb, tensor = preprocess_image(Path(image_path))

    with torch.no_grad():
        logits = model(tensor.to(device))
        predicted_classes = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Optional post-processing: keep only pothole pixels (class 1).
    bitmask = (predicted_classes == 1).astype(np.uint8) * 255
    bitmask = cv2.resize(bitmask, (corrected_rgb.shape[1], corrected_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    corrected_path, bitmask_path = save_prediction_outputs(corrected_rgb, bitmask, Path(image_path))
    print(f"Saved corrected image to: {corrected_path}")
    print(f"Saved predicted bitmask to: {bitmask_path}")
    return corrected_path, bitmask_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: py predict.py <path_to_road_image>")

    predict_image(sys.argv[1])
