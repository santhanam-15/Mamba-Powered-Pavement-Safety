"""
Comprehensive model testing script for v1 dataset.

This script:
1. Finds all RAW images in the v1 folder (with their corresponding CRACK, POTHOLE masks)
2. Runs inference on each image using the trained model
3. Compares predictions against ground truth (merged CRACK + POTHOLE masks)
4. Calculates per-image and overall model metrics (TP, TN, FP, FN, Accuracy, IoU, F1)
5. Generates a detailed report
"""

from __future__ import annotations

import argparse
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
    """Compute TP, TN, FP, FN, Accuracy, IoU, F1."""
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
    accuracy = (tp + tn) / max(total_pixels, 1)
    iou = intersection / max(union, eps)
    f1 = (2 * tp) / max((2 * tp) + fp + fn, eps)
    
    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "IoU": iou,
        "F1": f1,
    }


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


def test_model(v1_path: Path, checkpoint_path: Path, output_report: Path) -> None:
    """Test model on all v1 samples and generate comprehensive report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
            
            pbar.update(1)
    
    total_pixels = overall_metrics["TP"] + overall_metrics["TN"] + overall_metrics["FP"] + overall_metrics["FN"]
    intersection = overall_metrics["TP"]
    union = overall_metrics["TP"] + overall_metrics["FP"] + overall_metrics["FN"]
    
    eps = 1e-8
    overall_accuracy = (overall_metrics["TP"] + overall_metrics["TN"]) / max(total_pixels, 1)
    overall_iou = intersection / max(union, eps)
    overall_f1 = (2 * overall_metrics["TP"]) / max((2 * overall_metrics["TP"]) + overall_metrics["FP"] + overall_metrics["FN"], eps)
    
    avg_accuracy = np.mean([r["Accuracy"] for r in per_image_results])
    avg_iou = np.mean([r["IoU"] for r in per_image_results])
    avg_f1 = np.mean([r["F1"] for r in per_image_results])
    
    report_lines = [
        "=" * 80,
        "COMPREHENSIVE MODEL TEST REPORT",
        "=" * 80,
        f"\nDataset: v1 ({len(samples)} images)",
        f"Checkpoint: {checkpoint_path}",
        f"Device: {device}",
        f"Threshold: {threshold:.4f}",
        "",
        "-" * 80,
        "OVERALL METRICS (Aggregated across all pixels)",
        "-" * 80,
        f"TP (True Positive):  {overall_metrics['TP']:,}",
        f"TN (True Negative):  {overall_metrics['TN']:,}",
        f"FP (False Positive): {overall_metrics['FP']:,}",
        f"FN (False Negative): {overall_metrics['FN']:,}",
        f"Total Pixels:        {total_pixels:,}",
        "",
        f"Pixel Accuracy (Overall): {overall_accuracy:.6f}",
        f"IoU (Overall):            {overall_iou:.6f}",
        f"F1 Score (Overall):       {overall_f1:.6f}",
        "",
        "-" * 80,
        "PER-IMAGE METRICS (Average)",
        "-" * 80,
        f"Mean Pixel Accuracy: {avg_accuracy:.6f}",
        f"Mean IoU:            {avg_iou:.6f}",
        f"Mean F1 Score:       {avg_f1:.6f}",
        "",
        "-" * 80,
        "TOP 10 BEST PERFORMING IMAGES (by IoU)",
        "-" * 80,
    ]
    
    top_10_best = sorted(per_image_results, key=lambda x: x["IoU"], reverse=True)[:10]
    for i, result in enumerate(top_10_best, 1):
        report_lines.append(
            f"{i:2d}. {result['sample_id']:50s} | IoU: {result['IoU']:.6f} | "
            f"Acc: {result['Accuracy']:.6f} | F1: {result['F1']:.6f}"
        )
    
    report_lines.extend([
        "",
        "-" * 80,
        "TOP 10 WORST PERFORMING IMAGES (by IoU)",
        "-" * 80,
    ])
    
    bottom_10_worst = sorted(per_image_results, key=lambda x: x["IoU"])[:10]
    for i, result in enumerate(bottom_10_worst, 1):
        report_lines.append(
            f"{i:2d}. {result['sample_id']:50s} | IoU: {result['IoU']:.6f} | "
            f"Acc: {result['Accuracy']:.6f} | F1: {result['F1']:.6f}"
        )
    
    report_lines.extend([
        "",
        "=" * 80,
    ])
    
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"\nReport saved to: {output_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive model testing on v1 dataset.")
    parser.add_argument(
        "--v1-path",
        type=Path,
        default=Path("../Dataset/Cracks-and-Potholes-in-Road-Images/v1"),
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
        default=Path("test_report.txt"),
        help="Output report file path",
    )
    
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.v1_path.exists():
        raise FileNotFoundError(f"v1 path not found: {args.v1_path}")
    
    test_model(args.v1_path, args.checkpoint, args.output)
