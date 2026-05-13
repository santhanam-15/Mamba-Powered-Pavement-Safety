"""
Enhanced comprehensive model testing script for v1 dataset.

This script:
1. Tests each image individually
2. Saves per-image metrics to CSV
3. Creates comparison visualizations (prediction vs ground truth)
4. Generates detailed per-image analysis report
5. Saves all results in structured format for further analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import CHECKPOINT_PATH, IMG_SIZE
from model import CMSegNet, ShadowNet
from stage1_utils import correct_stage1_image, DEFAULT_CAMERA_HEIGHT_MM, DEFAULT_CAMERA_ANGLE_DEG


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[CMSegNet, float]:
    """Load trained CMSegNet model."""
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
    threshold: float,
    corrected_shape: tuple,
) -> np.ndarray:
    """Run model inference and return binary prediction mask."""
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
    
    bitmask = (probs > threshold).astype(np.uint8)
    bitmask = cv2.resize(
        bitmask,
        (corrected_shape[1], corrected_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return bitmask * 255


def compute_metrics(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive metrics: TP, TN, FP, FN, Accuracy, Precision, Recall, Specificity, F1, IoU, Dice."""
    pred = (predicted_mask > 127).astype(np.uint8)
    gt = (ground_truth_mask > 127).astype(np.uint8)
    
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    
    tp = int(np.sum((pred_flat == 1) & (gt_flat == 1)))
    tn = int(np.sum((pred_flat == 0) & (gt_flat == 0)))
    fp = int(np.sum((pred_flat == 1) & (gt_flat == 0)))
    fn = int(np.sum((pred_flat == 0) & (gt_flat == 1)))
    
    total_pixels = tp + tn + fp + fn
    intersection = tp
    union = tp + fp + fn
    
    eps = 1e-8
    
    # Basic metrics
    accuracy = (tp + tn) / max(total_pixels, 1)
    
    # Precision: of predicted positives, how many are correct
    precision = tp / max((tp + fp), eps)
    
    # Recall (Sensitivity): of actual positives, how many did we find
    recall = tp / max((tp + fn), eps)
    
    # Specificity: of actual negatives, how many did we correctly identify
    specificity = tn / max((tn + fp), eps)
    
    # False Positive Rate
    fpr = fp / max((fp + tn), eps)
    
    # False Negative Rate
    fnr = fn / max((fn + tp), eps)
    
    # F1 Score: harmonic mean of precision and recall
    f1 = (2 * tp) / max((2 * tp) + fp + fn, eps)
    
    # IoU (Jaccard Index): intersection over union
    iou = intersection / max(union, eps)
    
    # Dice Coefficient: similar to F1 but calculated differently
    dice = (2 * tp) / max((2 * tp) + fp + fn, eps)
    
    # Matthews Correlation Coefficient (MCC)
    mcc_denominator = np.sqrt(float(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1)))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denominator
    
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "FPR": fpr,
        "FNR": fnr,
        "F1": f1,
        "IoU": iou,
        "Dice": dice,
        "MCC": mcc,
    }


def create_overlay_image(corrected_bgr: np.ndarray, bitmask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Create red overlay on corrected image where bitmask is positive."""
    overlay_bgr = corrected_bgr.copy()
    red_layer = np.zeros_like(corrected_bgr, dtype=np.uint8)
    red_layer[:, :, 2] = 255  # Red channel
    region = bitmask > 127
    if np.any(region):
        blended = cv2.addWeighted(overlay_bgr[region], 1.0 - alpha, red_layer[region], alpha, 0)
        overlay_bgr[region] = blended
    return overlay_bgr


def save_area_summary(output_dir: Path, bitmask: np.ndarray, r_value: float = 0.00015625) -> None:
    """Save area summary text and visualization."""
    white_pixels = int(np.sum(bitmask > 127))
    area_mm2 = white_pixels * r_value
    area_cm2 = area_mm2 / 100.0
    
    # Save text file
    summary_txt_path = output_dir / "area_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(f"white_pixels={white_pixels}\n")
        f.write(f"r={r_value:.8f}\n")
        f.write(f"area_mm2={area_mm2:.4f}\n")
        f.write(f"area_cm2={area_cm2:.4f}\n")
    
    # Save summary image
    summary_img = np.full((220, 1100, 3), 255, dtype=np.uint8)
    cv2.putText(summary_img, f"White Pixels: {white_pixels}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(summary_img, f"r: {r_value:.8f}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(summary_img, f"Area: {area_cm2:.4f} cm2", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 60, 170), 2, cv2.LINE_AA)
    
    summary_img_path = output_dir / "area_summary.png"
    cv2.imwrite(str(summary_img_path), summary_img)


def create_comparison_image(
    corrected_rgb: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """Create side-by-side comparison image: prediction (RED) vs ground truth (GREEN)."""
    h, w = corrected_rgb.shape[:2]
    
    pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
    pred_colored[prediction > 127] = [255, 0, 0]  # RED for prediction
    
    gt_colored = np.zeros((h, w, 3), dtype=np.uint8)
    gt_colored[ground_truth > 127] = [0, 255, 0]  # GREEN for ground truth
    
    pred_overlay = cv2.addWeighted(corrected_rgb, 0.7, pred_colored, 0.3, 0)
    gt_overlay = cv2.addWeighted(corrected_rgb, 0.7, gt_colored, 0.3, 0)
    
    comparison = np.hstack([pred_overlay, gt_overlay])
    
    h_text = 30
    comparison_with_text = np.vstack([
        np.ones((h_text, comparison.shape[1], 3), dtype=np.uint8) * 255,
        comparison
    ])
    
    cv2.putText(
        comparison_with_text,
        "PREDICTION (Red)",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        comparison_with_text,
        "GROUND TRUTH (Green)",
        (w + 10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    
    return comparison_with_text


def scan_v1_samples(v1_path: Path) -> List[Dict[str, Path]]:
    """Scan v1 folder and collect all sample triplets (RAW, CRACK, POTHOLE)."""
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


def test_model(
    v1_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    save_comparisons: bool = True,
) -> None:
    """Test model on all v1 samples and save detailed per-image results to predictions folder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {checkpoint_path}...")
    model, threshold = load_model(checkpoint_path, device)
    
    print(f"Scanning v1 folder at {v1_path}...")
    samples = scan_v1_samples(v1_path)
    print(f"Found {len(samples)} test samples.\n")
    
    if not samples:
        raise FileNotFoundError(f"No test samples found in {v1_path}")
    
    overall_metrics = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    }
    per_image_results: List[Dict] = []
    
    csv_path = output_dir / "per_image_metrics.csv"
    
    print("Running inference on all samples...")
    with tqdm(total=len(samples), desc="Processing") as pbar:
        for sample in samples:
            sample_id = sample["id"]
            raw_path = sample["raw"]
            crack_path = sample["crack"]
            pothole_path = sample["pothole"]
            
            try:
                corrected_rgb, tensor = preprocess_image(raw_path)
                prediction = run_inference(model, tensor, device, threshold, corrected_rgb.shape)
                
                crack = cv2.imread(str(crack_path), cv2.IMREAD_GRAYSCALE)
                pothole = cv2.imread(str(pothole_path), cv2.IMREAD_GRAYSCALE)
                
                if crack is None or pothole is None:
                    pbar.update(1)
                    continue
                
                crack = binarize_mask(crack)
                pothole = binarize_mask(pothole)
                ground_truth = np.logical_or(crack, pothole).astype(np.uint8) * 255
                ground_truth = cv2.resize(
                    ground_truth,
                    (corrected_rgb.shape[1], corrected_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                
                metrics = compute_metrics(prediction, ground_truth)
                
                overall_metrics["TP"] += metrics["TP"]
                overall_metrics["TN"] += metrics["TN"]
                overall_metrics["FP"] += metrics["FP"]
                overall_metrics["FN"] += metrics["FN"]
                
                per_image_results.append({
                    "sample_id": sample_id,
                    **metrics,
                })
                
                # Save per-image results to predictions/<sample_id>/ folder
                image_output_dir = output_dir / sample_id
                image_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save original raw image
                original_img = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
                original_path = image_output_dir / "original.png"
                cv2.imwrite(str(original_path), original_img)
                
                # Save corrected image
                corrected_bgr = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)
                corrected_path = image_output_dir / "corrected.png"
                cv2.imwrite(str(corrected_path), corrected_bgr)
                
                # Save predicted bitmask
                bitmask_path = image_output_dir / "predicted_bitmask.png"
                cv2.imwrite(str(bitmask_path), prediction)
                
                # Save overlay (red overlay on corrected image)
                overlay = create_overlay_image(corrected_bgr, prediction, alpha=0.35)
                overlay_path = image_output_dir / "overlay.png"
                cv2.imwrite(str(overlay_path), overlay)
                
                # Save area summary (PNG + TXT)
                save_area_summary(image_output_dir, prediction)
                
                # Save comparison image (prediction vs ground truth)
                comparison = create_comparison_image(corrected_rgb, prediction, ground_truth)
                comp_path = image_output_dir / "comparison.png"
                cv2.imwrite(str(comp_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                
                # Save ground truth mask
                gt_path = image_output_dir / "ground_truth_mask.png"
                cv2.imwrite(str(gt_path), ground_truth)
                
                # Save metrics as JSON for this image
                metrics_json = {
                    "sample_id": sample_id,
                    **metrics,
                }
                metrics_path = image_output_dir / "metrics.json"
                with open(metrics_path, "w") as f:
                    json.dump(metrics_json, f, indent=2)
            
            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
            
            pbar.update(1)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", 
                       "Specificity", "FPR", "FNR", "F1", "IoU", "Dice", "MCC"],
        )
        writer.writeheader()
        for result in per_image_results:
            writer.writerow(result)
    
    print(f"\nPer-image metrics saved to: {csv_path}")
    print(f"Per-image results (with all files) saved to: predictions/<sample_id>/")
    print(f"  Each folder contains: original.png, corrected.png, predicted_bitmask.png, overlay.png,")
    print(f"                        area_summary.png, area_summary.txt, comparison.png, metrics.json")
    
    total_pixels = overall_metrics["TP"] + overall_metrics["TN"] + overall_metrics["FP"] + overall_metrics["FN"]
    intersection = overall_metrics["TP"]
    union = overall_metrics["TP"] + overall_metrics["FP"] + overall_metrics["FN"]
    
    eps = 1e-8
    overall_accuracy = (overall_metrics["TP"] + overall_metrics["TN"]) / max(total_pixels, 1)
    overall_precision = overall_metrics["TP"] / max((overall_metrics["TP"] + overall_metrics["FP"]), eps)
    overall_recall = overall_metrics["TP"] / max((overall_metrics["TP"] + overall_metrics["FN"]), eps)
    overall_specificity = overall_metrics["TN"] / max((overall_metrics["TN"] + overall_metrics["FP"]), eps)
    overall_fpr = overall_metrics["FP"] / max((overall_metrics["FP"] + overall_metrics["TN"]), eps)
    overall_fnr = overall_metrics["FN"] / max((overall_metrics["FN"] + overall_metrics["TP"]), eps)
    overall_iou = intersection / max(union, eps)
    overall_f1 = (2 * overall_metrics["TP"]) / max((2 * overall_metrics["TP"]) + overall_metrics["FP"] + overall_metrics["FN"], eps)
    overall_dice = overall_f1  # Dice = F1 for binary segmentation
    mcc_denominator = np.sqrt(float(max((overall_metrics["TP"] + overall_metrics["FP"]) * (overall_metrics["TP"] + overall_metrics["FN"]) * (overall_metrics["TN"] + overall_metrics["FP"]) * (overall_metrics["TN"] + overall_metrics["FN"]), 1)))
    overall_mcc = ((overall_metrics["TP"] * overall_metrics["TN"]) - (overall_metrics["FP"] * overall_metrics["FN"])) / mcc_denominator
    
    avg_accuracy = np.mean([r["Accuracy"] for r in per_image_results])
    avg_precision = np.mean([r["Precision"] for r in per_image_results])
    avg_recall = np.mean([r["Recall"] for r in per_image_results])
    avg_specificity = np.mean([r["Specificity"] for r in per_image_results])
    avg_iou = np.mean([r["IoU"] for r in per_image_results])
    avg_f1 = np.mean([r["F1"] for r in per_image_results])
    avg_dice = np.mean([r["Dice"] for r in per_image_results])
    avg_mcc = np.mean([r["MCC"] for r in per_image_results])
    
    report_lines = [
        "=" * 100,
        "COMPREHENSIVE MODEL TEST REPORT (Per-Image Results with Extended Metrics)",
        "=" * 100,
        f"\nDataset: v1 ({len(samples)} images)",
        f"Checkpoint: {checkpoint_path}",
        f"Device: {device}",
        f"Threshold: {threshold:.4f}",
        "",
        "-" * 100,
        "OVERALL METRICS (Aggregated across all pixels)",
        "-" * 100,
        f"TP (True Positive):   {overall_metrics['TP']:,}",
        f"TN (True Negative):   {overall_metrics['TN']:,}",
        f"FP (False Positive):  {overall_metrics['FP']:,}",
        f"FN (False Negative):  {overall_metrics['FN']:,}",
        f"Total Pixels:         {total_pixels:,}",
        "",
        "Classification Metrics:",
        f"  Accuracy:     {overall_accuracy:.6f}  (Overall correctness)",
        f"  Precision:    {overall_precision:.6f}  (Of predicted positives, how many correct)",
        f"  Recall:       {overall_recall:.6f}  (Of actual positives, how many found)",
        f"  Specificity:  {overall_specificity:.6f}  (Of actual negatives, how many correct)",
        f"  FPR:          {overall_fpr:.6f}  (False Positive Rate)",
        f"  FNR:          {overall_fnr:.6f}  (False Negative Rate)",
        "",
        "Segmentation Metrics:",
        f"  IoU (Jaccard):       {overall_iou:.6f}  (Intersection over Union)",
        f"  Dice Coefficient:    {overall_dice:.6f}  (F1-based overlap metric)",
        f"  F1 Score:            {overall_f1:.6f}  (Harmonic mean of Precision & Recall)",
        f"  MCC:                 {overall_mcc:.6f}  (Matthews Correlation Coefficient)",
        "",
        "-" * 100,
        "PER-IMAGE METRICS (Average across all images)",
        "-" * 100,
        f"Mean Accuracy:    {avg_accuracy:.6f}",
        f"Mean Precision:   {avg_precision:.6f}",
        f"Mean Recall:      {avg_recall:.6f}",
        f"Mean Specificity: {avg_specificity:.6f}",
        f"Mean F1:          {avg_f1:.6f}",
        f"Mean IoU:         {avg_iou:.6f}",
        f"Mean Dice:        {avg_dice:.6f}",
        f"Mean MCC:         {avg_mcc:.6f}",
        "",
        "-" * 100,
        "TOP 10 BEST PERFORMING IMAGES (by IoU)",
        "-" * 100,
    ]
    
    top_10_best = sorted(per_image_results, key=lambda x: x["IoU"], reverse=True)[:10]
    for i, result in enumerate(top_10_best, 1):
        report_lines.append(
            f"{i:2d}. {result['sample_id']:50s} | IoU: {result['IoU']:.6f} | "
            f"Prec: {result['Precision']:.6f} | Rec: {result['Recall']:.6f} | F1: {result['F1']:.6f}"
        )
    
    report_lines.extend([
        "",
        "-" * 100,
        "TOP 10 WORST PERFORMING IMAGES (by IoU)",
        "-" * 100,
    ])
    
    bottom_10_worst = sorted(per_image_results, key=lambda x: x["IoU"])[:10]
    for i, result in enumerate(bottom_10_worst, 1):
        report_lines.append(
            f"{i:2d}. {result['sample_id']:50s} | IoU: {result['IoU']:.6f} | "
            f"Prec: {result['Precision']:.6f} | Rec: {result['Recall']:.6f} | F1: {result['F1']:.6f}"
        )
    
    report_lines.extend([
        "",
        "-" * 100,
        "ALL PER-IMAGE RESULTS",
        "-" * 100,
        f"{'Sample ID':<50} | {'Acc':>8} | {'Prec':>8} | {'Rec':>8} | {'F1':>8} | {'IoU':>8}",
        "-" * 100,
    ])
    
    for result in sorted(per_image_results, key=lambda x: x["IoU"], reverse=True):
        report_lines.append(
            f"{result['sample_id']:<50} | {result['Accuracy']:>8.6f} | "
            f"{result['Precision']:>8.6f} | {result['Recall']:>8.6f} | "
            f"{result['F1']:>8.6f} | {result['IoU']:>8.6f}"
        )
    
    report_lines.extend([
        "",
        "=" * 100,
        "METRIC DEFINITIONS:",
        "=" * 100,
        "  Accuracy:   (TP + TN) / Total - Overall correctness of predictions",
        "  Precision:  TP / (TP + FP) - Of predicted positives, how many are actually positive",
        "  Recall:     TP / (TP + FN) - Of actual positives, how many did we predict",
        "  Specificity: TN / (TN + FP) - Of actual negatives, how many did we correctly identify",
        "  F1 Score:   2*TP / (2*TP + FP + FN) - Harmonic mean of Precision and Recall",
        "  IoU (Jaccard): TP / (TP + FP + FN) - Intersection over Union",
        "  Dice:       Same as F1 for binary segmentation - Overlap metric",
        "  MCC:        Matthews Correlation Coefficient - Correlation between predicted and actual",
        "  FPR:        FP / (FP + TN) - False Positive Rate (Type I Error)",
        "  FNR:        FN / (FN + TP) - False Negative Rate (Type II Error)",
        "",
        "=" * 100,
        f"CSV file with all metrics: per_image_metrics.csv",
        f"Per-image results saved to: predictions/<sample_id>/",
        f"  - original.png (raw input image)",
        f"  - corrected.png (stage1 perspective corrected)",
        f"  - predicted_bitmask.png (model binary prediction)",
        f"  - overlay.png (predicted areas highlighted in red)",
        f"  - area_summary.png (area summary visualization)",
        f"  - area_summary.txt (area metrics: pixels, mm2, cm2)",
        f"  - comparison.png (prediction vs ground truth side-by-side)",
        f"  - ground_truth_mask.png (merged crack+pothole masks)",
        f"  - metrics.json (per-image metrics: all evaluation metrics)",
        "=" * 100,
    ])
    
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    report_path = output_dir / "test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced model testing on v1 dataset with per-image results.")
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
        default=Path("predictions"),
        help="Output directory for results (each image gets a subfolder)",
    )
    parser.add_argument(
        "--save-comparisons",
        action="store_true",
        help="Save comparison images (prediction vs ground truth)",
    )
    
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.v1_path.exists():
        raise FileNotFoundError(f"v1 path not found: {args.v1_path}")
    
    test_model(args.v1_path, args.checkpoint, args.output, args.save_comparisons)
