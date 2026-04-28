"""
Threshold tuning script: Find optimal detection threshold for precision/recall trade-off.

This script:
1. Loads the trained model
2. Tests on all v1 samples at different thresholds (0.1 to 0.9)
3. Computes metrics (Precision, Recall, F1, IoU) for each threshold
4. Identifies optimal thresholds for different priorities:
   - Max F1 (balanced)
   - Max Recall (catch all defects, accept false positives)
   - Max Precision (minimize false positives, accept missed defects)
5. Saves results to CSV and visualization
"""

from pathlib import Path
from typing import Dict, List
import argparse

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

from config import CHECKPOINT_PATH, IMG_SIZE, DEFAULT_CAMERA_HEIGHT_MM, DEFAULT_CAMERA_ANGLE_DEG
from model import CMSegNet, ShadowNet
from stage1_utils import correct_stage1_image


def load_model(checkpoint_path: Path, device: torch.device) -> CMSegNet:
    """Load trained CMSegNet model."""
    model = CMSegNet(img_size=IMG_SIZE).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert mask to binary (0 or 1)."""
    threshold = 0 if mask.max() <= 1 else 127
    return (mask > threshold).astype(np.uint8)


def preprocess_image(image_path: Path) -> tuple[np.ndarray, torch.Tensor]:
    """Read and preprocess raw image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    
    corrected_bgr, _ = correct_stage1_image(
        image,
        camera_height_mm=DEFAULT_CAMERA_HEIGHT_MM,
        camera_angle_deg=DEFAULT_CAMERA_ANGLE_DEG,
    )
    corrected_rgb = cv2.cvtColor(corrected_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(corrected_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(
        (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
    ).unsqueeze(0).float()
    return corrected_rgb, tensor


def run_inference(
    model: CMSegNet,
    tensor: torch.Tensor,
    device: torch.device,
    corrected_shape: tuple,
) -> np.ndarray:
    """Run model inference and return probability map (0-1)."""
    shadow_net = ShadowNet().to(device)
    shadow_net.eval()
    
    with torch.no_grad():
        pothole_logits, _ = model(tensor.to(device))
        pothole_mask = torch.sigmoid(pothole_logits)
        shadow_mask = shadow_net(tensor.to(device))
        
        shadow_mean = float(shadow_mask.mean().item())
        shadow_std = float(shadow_mask.std(unbiased=False).item())
        
        if abs(shadow_mean - 0.5) < 0.08 and shadow_std < 0.06:
            final_mask = pothole_mask
        else:
            shadow_conf = torch.clamp((shadow_mask - 0.55) / 0.45, min=0.0, max=1.0)
            final_mask = pothole_mask * (1 - 0.85 * shadow_conf)
        
        probs = final_mask.squeeze().cpu().numpy()
    
    # Resize to original shape
    probs = cv2.resize(
        probs,
        (corrected_shape[1], corrected_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return probs


def compute_metrics(predicted_probs: np.ndarray, ground_truth: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute metrics at a specific threshold."""
    pred = (predicted_probs > threshold).astype(np.uint8)
    gt = (ground_truth > 127).astype(np.uint8)
    
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    
    tp = int(np.sum((pred_flat == 1) & (gt_flat == 1)))
    tn = int(np.sum((pred_flat == 0) & (gt_flat == 0)))
    fp = int(np.sum((pred_flat == 1) & (gt_flat == 0)))
    fn = int(np.sum((pred_flat == 0) & (gt_flat == 1)))
    
    eps = 1e-8
    
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max((tp + fp), eps)
    recall = tp / max((tp + fn), eps)
    f1 = (2 * tp) / max((2 * tp) + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    
    return {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou,
    }


def scan_v1_samples(v1_path: Path) -> List[Dict[str, Path]]:
    """Scan v1 folder and collect all sample triplets."""
    samples: Dict[str, Dict[str, Path]] = {}
    
    for folder in v1_path.iterdir():
        if not folder.is_dir():
            continue
        
        for file in folder.iterdir():
            if not file.is_file():
                continue
            
            stem = file.stem
            if "_" not in stem:
                continue
            
            prefix, suffix = stem.rsplit("_", 1)
            suffix = suffix.upper()
            
            if suffix not in {"RAW", "CRACK", "POTHOLE"}:
                continue
            
            if prefix not in samples:
                samples[prefix] = {}
            samples[prefix][suffix] = file
    
    result = []
    for prefix, files in sorted(samples.items()):
        if {"RAW", "CRACK", "POTHOLE"}.issubset(files):
            result.append({
                "id": prefix,
                "raw": files["RAW"],
                "crack": files["CRACK"],
                "pothole": files["POTHOLE"],
            })
    
    return result


def tune_thresholds(
    v1_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> None:
    """Test model at different thresholds and find optimal values."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    
    print(f"Scanning v1 folder at {v1_path}...")
    samples = scan_v1_samples(v1_path)
    print(f"Found {len(samples)} test samples.\n")
    
    if not samples:
        raise FileNotFoundError(f"No test samples found in {v1_path}")
    
    # Thresholds to test
    thresholds = np.arange(0.1, 0.95, 0.05)
    threshold_results = {t: [] for t in thresholds}
    
    print("Running inference on all samples...")
    for sample in tqdm(samples):
        sample_id = sample["id"]
        raw_path = sample["raw"]
        crack_path = sample["crack"]
        pothole_path = sample["pothole"]
        
        try:
            corrected_rgb, tensor = preprocess_image(raw_path)
            probs = run_inference(model, tensor, device, corrected_rgb.shape)
            
            crack = cv2.imread(str(crack_path), cv2.IMREAD_GRAYSCALE)
            pothole = cv2.imread(str(pothole_path), cv2.IMREAD_GRAYSCALE)
            
            if crack is None or pothole is None:
                continue
            
            crack = binarize_mask(crack)
            pothole = binarize_mask(pothole)
            ground_truth = np.logical_or(crack, pothole).astype(np.uint8) * 255
            ground_truth = cv2.resize(
                ground_truth,
                (corrected_rgb.shape[1], corrected_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            
            # Compute metrics for each threshold
            for threshold in thresholds:
                metrics = compute_metrics(probs, ground_truth, threshold)
                threshold_results[threshold].append(metrics)
        
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            continue
    
    # Aggregate results across all samples
    aggregated_results = []
    for threshold in sorted(thresholds):
        results_at_threshold = threshold_results[threshold]
        
        if not results_at_threshold:
            continue
        
        # Sum up counts
        total_tp = sum(r["TP"] for r in results_at_threshold)
        total_fp = sum(r["FP"] for r in results_at_threshold)
        total_fn = sum(r["FN"] for r in results_at_threshold)
        total_tn = sum(r["TN"] for r in results_at_threshold)
        
        eps = 1e-8
        
        # Aggregate metrics
        accuracy = (total_tp + total_tn) / max(total_tp + total_tn + total_fp + total_fn, 1)
        precision = total_tp / max((total_tp + total_fp), eps)
        recall = total_tp / max((total_tp + total_fn), eps)
        f1 = (2 * total_tp) / max((2 * total_tp) + total_fp + total_fn, eps)
        iou = total_tp / max(total_tp + total_fp + total_fn, eps)
        
        aggregated_results.append({
            "Threshold": f"{threshold:.2f}",
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "IoU": iou,
            "Accuracy": accuracy,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "TN": total_tn,
        })
    
    # Save results to CSV
    csv_path = output_dir / "threshold_tuning_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Threshold", "Precision", "Recall", "F1", "IoU", "Accuracy", "TP", "FP", "FN", "TN"],
        )
        writer.writeheader()
        for result in aggregated_results:
            writer.writerow(result)
    
    print(f"\nThreshold tuning results saved to: {csv_path}")
    
    # Find optimal thresholds
    thresholds_list = [float(r["Threshold"]) for r in aggregated_results]
    f1_scores = [r["F1"] for r in aggregated_results]
    recalls = [r["Recall"] for r in aggregated_results]
    precisions = [r["Precision"] for r in aggregated_results]
    
    best_f1_idx = np.argmax(f1_scores)
    best_recall_idx = np.argmax(recalls)
    best_precision_idx = np.argmax(precisions)
    
    best_f1_threshold = thresholds_list[best_f1_idx]
    best_recall_threshold = thresholds_list[best_recall_idx]
    best_precision_threshold = thresholds_list[best_precision_idx]
    
    print(f"\n{'='*80}")
    print(f"OPTIMAL THRESHOLDS")
    print(f"{'='*80}")
    print(f"Best F1 Score:       Threshold={best_f1_threshold:.2f}, F1={f1_scores[best_f1_idx]:.4f}")
    print(f"  Precision: {precisions[best_f1_idx]:.4f}, Recall: {recalls[best_f1_idx]:.4f}")
    print(f"")
    print(f"Best Recall:         Threshold={best_recall_threshold:.2f}, Recall={recalls[best_recall_idx]:.4f}")
    print(f"  Precision: {precisions[best_recall_idx]:.4f}, F1: {f1_scores[best_recall_idx]:.4f}")
    print(f"")
    print(f"Best Precision:      Threshold={best_precision_threshold:.2f}, Precision={precisions[best_precision_idx]:.4f}")
    print(f"  Recall: {recalls[best_precision_idx]:.4f}, F1: {f1_scores[best_precision_idx]:.4f}")
    print(f"{'='*80}\n")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Threshold Tuning Analysis", fontsize=16, fontweight="bold")
    
    # Plot 1: Precision vs Recall (ROC-like)
    ax = axes[0, 0]
    ax.plot(recalls, precisions, "b-", linewidth=2, marker="o")
    ax.axvline(recalls[best_f1_idx], color="r", linestyle="--", label=f"Best F1: Th={best_f1_threshold:.2f}")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision vs Recall Trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: F1 Score vs Threshold
    ax = axes[0, 1]
    ax.plot(thresholds_list, f1_scores, "g-", linewidth=2, marker="o")
    ax.axvline(best_f1_threshold, color="r", linestyle="--", label=f"Best: {best_f1_threshold:.2f}")
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("F1 Score vs Threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Recall and Precision vs Threshold
    ax = axes[1, 0]
    ax.plot(thresholds_list, recalls, "b-", linewidth=2, marker="o", label="Recall")
    ax.plot(thresholds_list, precisions, "r-", linewidth=2, marker="s", label="Precision")
    ax.axvline(best_f1_threshold, color="g", linestyle="--", alpha=0.7, label=f"F1 Peak: {best_f1_threshold:.2f}")
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Recall & Precision vs Threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: IoU vs Threshold
    ax = axes[1, 1]
    ious = [r["IoU"] for r in aggregated_results]
    ax.plot(thresholds_list, ious, "purple", linewidth=2, marker="D")
    best_iou_idx = np.argmax(ious)
    best_iou_threshold = thresholds_list[best_iou_idx]
    ax.axvline(best_iou_threshold, color="r", linestyle="--", label=f"Best: {best_iou_threshold:.2f}")
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("IoU", fontsize=11)
    ax.set_title("IoU vs Threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plot_path = output_dir / "threshold_tuning.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {plot_path}")
    
    # Save recommendations to text file
    recommendations_path = output_dir / "threshold_recommendations.txt"
    with open(recommendations_path, "w", encoding="utf-8") as f:
        f.write("THRESHOLD TUNING RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SCENARIO 1: Balanced Performance (Recommended for most use cases)\n")
        f.write(f"  Threshold: {best_f1_threshold:.2f}\n")
        f.write(f"  F1 Score: {f1_scores[best_f1_idx]:.4f}\n")
        f.write(f"  Precision: {precisions[best_f1_idx]:.4f} (Avoid false positives)\n")
        f.write(f"  Recall: {recalls[best_f1_idx]:.4f} (Catch actual defects)\n\n")
        
        f.write("SCENARIO 2: High Recall (Maximize defect detection, accept more false positives)\n")
        f.write(f"  Threshold: {best_recall_threshold:.2f}\n")
        f.write(f"  Recall: {recalls[best_recall_idx]:.4f}\n")
        f.write(f"  Precision: {precisions[best_recall_idx]:.4f}\n")
        f.write(f"  F1 Score: {f1_scores[best_recall_idx]:.4f}\n")
        f.write(f"  Use case: Road inspection - don't miss any defects\n\n")
        
        f.write("SCENARIO 3: High Precision (Minimize false positives, accept missed defects)\n")
        f.write(f"  Threshold: {best_precision_threshold:.2f}\n")
        f.write(f"  Precision: {precisions[best_precision_idx]:.4f}\n")
        f.write(f"  Recall: {recalls[best_precision_idx]:.4f}\n")
        f.write(f"  F1 Score: {f1_scores[best_precision_idx]:.4f}\n")
        f.write(f"  Use case: Automated maintenance scheduling - only fix real defects\n\n")
        
        f.write("SCENARIO 4: Maximum IoU (Best segmentation accuracy)\n")
        f.write(f"  Threshold: {best_iou_threshold:.2f}\n")
        f.write(f"  IoU: {ious[best_iou_idx]:.4f}\n")
        f.write(f"  Precision: {precisions[best_iou_idx]:.4f}\n")
        f.write(f"  Recall: {recalls[best_iou_idx]:.4f}\n\n")
    
    print(f"Recommendations saved to: {recommendations_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold tuning for optimal detection performance.")
    parser.add_argument(
        "--v1-path",
        type=Path,
        default=Path("v1"),
        help="Path to v1 folder",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(CHECKPOINT_PATH),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("threshold_tuning"),
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.v1_path.exists():
        raise FileNotFoundError(f"v1 path not found: {args.v1_path}")
    
    tune_thresholds(args.v1_path, args.checkpoint, args.output)
