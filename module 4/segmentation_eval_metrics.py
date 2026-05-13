"""
Binary segmentation evaluation metrics for pothole detection.

This script computes:
1) Pixel Accuracy
2) IoU (Intersection over Union)
3) F1 Score (Dice Score)

It also prints the confusion matrix counts:
- TP (True Positive)
- TN (True Negative)
- FP (False Positive)
- FN (False Negative)

Supports input as NumPy arrays or PyTorch tensors.
"""

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # Optional dependency
    torch = None

try:
    from PIL import Image, ImageDraw
except ImportError:  # Optional dependency
    Image = None
    ImageDraw = None


def to_numpy_binary_mask(mask: Any) -> np.ndarray:
    """Convert input mask (NumPy array or Torch tensor) to a binary NumPy array."""
    if torch is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    else:
        mask = np.asarray(mask)

    unique_values = np.unique(mask)
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValueError(
            f"Mask must contain only binary values 0 or 1. Found values: {unique_values.tolist()}"
        )

    return mask.astype(np.uint8)


def compute_confusion_values(predicted_mask: Any, ground_truth_mask: Any) -> Dict[str, int]:
    """Compute TP, TN, FP, and FN for binary segmentation masks."""
    pred = to_numpy_binary_mask(predicted_mask)
    gt = to_numpy_binary_mask(ground_truth_mask)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: predicted_mask shape {pred.shape} vs ground_truth_mask shape {gt.shape}")

    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)

    tp = int(np.sum((pred_flat == 1) & (gt_flat == 1)))
    tn = int(np.sum((pred_flat == 0) & (gt_flat == 0)))
    fp = int(np.sum((pred_flat == 1) & (gt_flat == 0)))
    fn = int(np.sum((pred_flat == 0) & (gt_flat == 1)))

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def safe_divide(numerator: float, denominator: float, eps: float = 1e-8) -> float:
    """Safely divide to avoid division-by-zero errors."""
    return float(numerator / (denominator + eps))


def calculate_segmentation_metrics(
    predicted_mask: Any,
    ground_truth_mask: Any,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Calculate Pixel Accuracy, IoU, and F1 (Dice) score."""
    conf = compute_confusion_values(predicted_mask, ground_truth_mask)
    tp, tn, fp, fn = conf["TP"], conf["TN"], conf["FP"], conf["FN"]

    total_pixels = tp + tn + fp + fn
    intersection = tp
    union = tp + fp + fn

    pixel_accuracy = safe_divide(tp + tn, total_pixels, eps)
    iou = safe_divide(intersection, union, eps)
    f1_dice = safe_divide(2 * tp, (2 * tp) + fp + fn, eps)

    return {
        "TP": float(tp),
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
        "Pixel Accuracy": pixel_accuracy,
        "IoU": iou,
        "F1 Score (Dice)": f1_dice,
    }


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print all confusion values and segmentation metrics in a readable format."""
    print("\n=== Binary Segmentation Evaluation Metrics ===")
    print(f"TP (True Positive):  {int(metrics['TP'])}")
    print(f"TN (True Negative):  {int(metrics['TN'])}")
    print(f"FP (False Positive): {int(metrics['FP'])}")
    print(f"FN (False Negative): {int(metrics['FN'])}")

    print("\n--- Scores ---")
    print(f"Pixel Accuracy     : {metrics['Pixel Accuracy']:.6f}")
    print(f"IoU                : {metrics['IoU']:.6f}")
    print(f"F1 Score (Dice)    : {metrics['F1 Score (Dice)']:.6f}")


def save_metrics_image(metrics: Dict[str, float], output_path: str = "area_summary.jpg") -> None:
    """Save the metrics summary as a simple image file."""
    if Image is None or ImageDraw is None:
        print("\nPillow is not installed, so image export is skipped.")
        return

    lines = [
        "Binary Segmentation Evaluation Metrics",
        "",
        f"TP (True Positive):  {int(metrics['TP'])}",
        f"TN (True Negative):  {int(metrics['TN'])}",
        f"FP (False Positive): {int(metrics['FP'])}",
        f"FN (False Negative): {int(metrics['FN'])}",
        "",
        f"Pixel Accuracy  : {metrics['Pixel Accuracy']:.6f}",
        f"IoU             : {metrics['IoU']:.6f}",
        f"F1 Score (Dice) : {metrics['F1 Score (Dice)']:.6f}",
    ]

    img_width, img_height = 760, 360
    image = Image.new("RGB", (img_width, img_height), color=(255, 255, 255))
    drawer = ImageDraw.Draw(image)

    y = 20
    for line in lines:
        drawer.text((20, y), line, fill=(0, 0, 0))
        y += 30

    image.save(output_path)
    print(f"\nSaved metrics summary image: {output_path}")


if __name__ == "__main__":
    # Example usage with dummy masks (from your requirement)
    pred = np.array([[0, 1, 1], [0, 1, 0]], dtype=np.uint8)
    gt = np.array([[0, 1, 0], [0, 1, 1]], dtype=np.uint8)

    results = calculate_segmentation_metrics(pred, gt)
    print_metrics(results)
    summary_image_path = str(Path(__file__).with_name("area_summary.jpg"))
    save_metrics_image(results, output_path=summary_image_path)

    # Optional: Torch tensor input also works (if torch is installed)
    if torch is not None:
        pred_t = torch.tensor([[0, 1, 1], [0, 1, 0]])
        gt_t = torch.tensor([[0, 1, 0], [0, 1, 1]])
        torch_results = calculate_segmentation_metrics(pred_t, gt_t)
        print("\nTorch input check passed. IoU:", f"{torch_results['IoU']:.6f}")
