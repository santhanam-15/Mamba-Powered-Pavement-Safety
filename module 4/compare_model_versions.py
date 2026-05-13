"""
Comprehensive model version comparison script.

Compares performance of model versions (v1, v2, v3) across multiple metrics.
Generates detailed comparison visualizations and reports.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from config import CHECKPOINT_PATH, IMG_SIZE
from model import CMSegNet, ShadowNet
from stage1_utils import correct_stage1_image, DEFAULT_CAMERA_HEIGHT_MM, DEFAULT_CAMERA_ANGLE_DEG


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[CMSegNet, float]:
    """Load trained CMSegNet model."""
    model = CMSegNet(img_size=IMG_SIZE).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and not checkpoint.get("trained", False):
        raise ValueError(f"Checkpoint '{checkpoint_path}' is not a trained model.")
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
    """Compute comprehensive metrics."""
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
    precision = tp / max((tp + fp), eps)
    recall = tp / max((tp + fn), eps)
    specificity = tn / max((tn + fp), eps)
    fpr = fp / max((fp + tn), eps)
    fnr = fn / max((fn + tp), eps)
    f1 = (2 * tp) / max((2 * tp) + fp + fn, eps)
    iou = intersection / max(union, eps)
    dice = (2 * tp) / max((2 * tp) + fp + fn, eps)
    mcc_denominator = np.sqrt(float(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1)))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denominator
    
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": accuracy, "Precision": precision, "Recall": recall,
        "Specificity": specificity, "FPR": fpr, "FNR": fnr,
        "F1": f1, "IoU": iou, "Dice": dice, "MCC": mcc,
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


def test_model_version(
    model_name: str,
    checkpoint_path: Path,
    v1_path: Path,
) -> Dict[str, float]:
    """Test a single model version and return overall metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*80}")
    print(f"Testing Model: {model_name}")
    print(f"{'='*80}")
    
    print(f"Loading model from {checkpoint_path}...")
    model, threshold = load_model(checkpoint_path, device)
    
    print(f"Scanning v1 folder at {v1_path}...")
    samples = scan_v1_samples(v1_path)
    print(f"Found {len(samples)} test samples.\n")
    
    if not samples:
        raise FileNotFoundError(f"No test samples found in {v1_path}")
    
    overall_metrics = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    per_image_results = []
    
    print(f"Running inference on all samples...")
    with tqdm(total=len(samples), desc="Processing", leave=True) as pbar:
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
                
                per_image_results.append({"sample_id": sample_id, **metrics})
            
            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
            
            pbar.update(1)
    
    # Calculate aggregated metrics
    total_pixels = overall_metrics["TP"] + overall_metrics["TN"] + overall_metrics["FP"] + overall_metrics["FN"]
    intersection = overall_metrics["TP"]
    union = overall_metrics["TP"] + overall_metrics["FP"] + overall_metrics["FN"]
    
    eps = 1e-8
    overall_accuracy = (overall_metrics["TP"] + overall_metrics["TN"]) / max(total_pixels, 1)
    overall_precision = overall_metrics["TP"] / max((overall_metrics["TP"] + overall_metrics["FP"]), eps)
    overall_recall = overall_metrics["TP"] / max((overall_metrics["TP"] + overall_metrics["FN"]), eps)
    overall_specificity = overall_metrics["TN"] / max((overall_metrics["TN"] + overall_metrics["FP"]), eps)
    overall_iou = intersection / max(union, eps)
    overall_f1 = (2 * overall_metrics["TP"]) / max((2 * overall_metrics["TP"]) + overall_metrics["FP"] + overall_metrics["FN"], eps)
    overall_dice = overall_f1
    mcc_denominator = np.sqrt(float(max((overall_metrics["TP"] + overall_metrics["FP"]) * (overall_metrics["TP"] + overall_metrics["FN"]) * (overall_metrics["TN"] + overall_metrics["FP"]) * (overall_metrics["TN"] + overall_metrics["FN"]), 1)))
    overall_mcc = ((overall_metrics["TP"] * overall_metrics["TN"]) - (overall_metrics["FP"] * overall_metrics["FN"])) / mcc_denominator
    
    # Calculate per-image averages
    avg_accuracy = np.mean([r["Accuracy"] for r in per_image_results])
    avg_precision = np.mean([r["Precision"] for r in per_image_results])
    avg_recall = np.mean([r["Recall"] for r in per_image_results])
    avg_specificity = np.mean([r["Specificity"] for r in per_image_results])
    avg_iou = np.mean([r["IoU"] for r in per_image_results])
    avg_f1 = np.mean([r["F1"] for r in per_image_results])
    avg_dice = np.mean([r["Dice"] for r in per_image_results])
    avg_mcc = np.mean([r["MCC"] for r in per_image_results])
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {overall_accuracy:.6f}")
    print(f"  Precision: {overall_precision:.6f}")
    print(f"  Recall:    {overall_recall:.6f}")
    print(f"  F1 Score:  {overall_f1:.6f}")
    print(f"  IoU:       {overall_iou:.6f}")
    
    return {
        "Model": model_name,
        "Accuracy": overall_accuracy,
        "Precision": overall_precision,
        "Recall": overall_recall,
        "Specificity": overall_specificity,
        "F1": overall_f1,
        "IoU": overall_iou,
        "Dice": overall_dice,
        "MCC": overall_mcc,
        "Avg_Accuracy": avg_accuracy,
        "Avg_Precision": avg_precision,
        "Avg_Recall": avg_recall,
        "Avg_Specificity": avg_specificity,
        "Avg_F1": avg_f1,
        "Avg_IoU": avg_iou,
        "Avg_Dice": avg_dice,
        "Avg_MCC": avg_mcc,
    }


def create_comparison_visualizations(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Create comprehensive comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics to compare
    overall_metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "IoU", "Dice", "MCC"]
    avg_metrics = ["Avg_Accuracy", "Avg_Precision", "Avg_Recall", "Avg_Specificity", "Avg_F1", "Avg_IoU", "Avg_Dice", "Avg_MCC"]
    
    # 1. Overall Metrics Comparison (Bar Chart)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Model Version Comparison - Overall Metrics (Aggregated Pixels)", fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(overall_metrics):
        ax = axes[idx // 4, idx % 4]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_overall_metrics_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: 01_overall_metrics_comparison.png")
    plt.close()
    
    # 2. Per-Image Average Metrics Comparison (Bar Chart)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Model Version Comparison - Per-Image Average Metrics", fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(avg_metrics):
        ax = axes[idx // 4, idx % 4]
        clean_metric = metric.replace('Avg_', '')
        bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
        ax.set_ylabel(f"Avg {clean_metric}", fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_per_image_avg_metrics_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: 02_per_image_avg_metrics_comparison.png")
    plt.close()
    
    # 3. Radar Chart Comparison (Multiple metrics per model)
    from math import pi
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 'IoU', 'Dice', 'MCC']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    colors_radar = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, model in enumerate(results_df['Model']):
        values = results_df.loc[results_df['Model'] == model, overall_metrics].values.flatten().tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, fontweight='bold')
    
    plt.title("Model Performance Radar Chart (Overall Metrics)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "03_radar_chart_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: 03_radar_chart_comparison.png")
    plt.close()
    
    # 4. Heatmap of all metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Overall metrics heatmap
    overall_data = results_df[overall_metrics].T
    sns.heatmap(overall_data, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax1, 
                cbar_kws={'label': 'Metric Value'}, vmin=0, vmax=1, linewidths=0.5)
    ax1.set_title("Overall Metrics (Aggregated Pixels)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Model", fontweight='bold')
    ax1.set_ylabel("Metric", fontweight='bold')
    
    # Per-image average metrics heatmap
    avg_data = results_df[avg_metrics].T
    avg_data.index = [col.replace('Avg_', '') for col in avg_data.index]
    sns.heatmap(avg_data, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax2,
                cbar_kws={'label': 'Metric Value'}, vmin=0, vmax=1, linewidths=0.5)
    ax2.set_title("Per-Image Average Metrics", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Model", fontweight='bold')
    ax2.set_ylabel("Metric", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "04_heatmap_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: 04_heatmap_comparison.png")
    plt.close()
    
    # 5. Line plot showing metrics progression
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(overall_metrics))
    width = 0.25
    
    for idx, model in enumerate(results_df['Model']):
        values = results_df.loc[results_df['Model'] == model, overall_metrics].values.flatten()
        ax.plot(x_pos, values, marker='o', linewidth=2.5, markersize=8, 
               label=model, color=colors_radar[idx])
    
    ax.set_xlabel("Metrics", fontweight='bold', fontsize=12)
    ax.set_ylabel("Score", fontweight='bold', fontsize=12)
    ax.set_title("Model Performance Comparison - Overall Metrics", fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(overall_metrics, rotation=45, ha='right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_dir / "05_line_plot_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: 05_line_plot_comparison.png")
    plt.close()
    
    # 6. Summary table image
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    table_data.append(['Metric'] + results_df['Model'].tolist())
    
    for metric in overall_metrics:
        row = [metric] + [f"{val:.6f}" for val in results_df[metric].values]
        table_data.append(row)
    
    table_data.append([''] * (len(results_df) + 1))  # Empty row
    
    for metric in avg_metrics:
        clean_name = metric.replace('Avg_', 'Avg ')
        row = [clean_name] + [f"{val:.6f}" for val in results_df[metric].values]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2] + [0.25] * len(results_df))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(results_df) + 1):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style metric names
    for i in range(1, len(table_data)):
        table[(i, 0)].set_facecolor('#e8e8e8')
        table[(i, 0)].set_text_props(weight='bold')
    
    plt.title("Model Version Comparison - Detailed Metrics Summary", 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_dir / "06_metrics_summary_table.png", dpi=300, bbox_inches='tight')
    print(f"Saved: 06_metrics_summary_table.png")
    plt.close()


def generate_comparison_report(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate a text report comparing model versions."""
    report_path = output_dir / "model_comparison_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("MODEL VERSION COMPARISON REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 100 + "\n")
        
        # Find best model for each metric
        overall_metrics = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "IoU", "Dice", "MCC"]
        
        for metric in overall_metrics:
            best_idx = results_df[metric].idxmax()
            best_model = results_df.loc[best_idx, 'Model']
            best_value = results_df.loc[best_idx, metric]
            f.write(f"\nBest {metric:15s}: {best_model:10s} ({best_value:.6f})\n")
        
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("OVERALL METRICS (Aggregated Pixels)\n")
        f.write("=" * 100 + "\n\n")
        
        for _, row in results_df.iterrows():
            f.write(f"\n{row['Model']}\n")
            f.write("-" * 50 + "\n")
            for metric in overall_metrics:
                f.write(f"  {metric:15s}: {row[metric]:.6f}\n")
        
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("PER-IMAGE AVERAGE METRICS\n")
        f.write("=" * 100 + "\n\n")
        
        avg_metrics = ["Avg_Accuracy", "Avg_Precision", "Avg_Recall", "Avg_Specificity", "Avg_F1", "Avg_IoU", "Avg_Dice", "Avg_MCC"]
        
        for _, row in results_df.iterrows():
            f.write(f"\n{row['Model']}\n")
            f.write("-" * 50 + "\n")
            for metric in avg_metrics:
                clean_name = metric.replace('Avg_', '')
                f.write(f"  {clean_name:15s}: {row[metric]:.6f}\n")
        
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 100 + "\n\n")
        
        # Calculate improvements
        if len(results_df) >= 2:
            first_model = results_df.iloc[0]['Model']
            last_model = results_df.iloc[-1]['Model']
            
            f.write(f"Comparison: {first_model} vs {last_model}\n")
            f.write("-" * 50 + "\n")
            
            for metric in overall_metrics:
                first_val = results_df.iloc[0][metric]
                last_val = results_df.iloc[-1][metric]
                improvement = ((last_val - first_val) / first_val * 100) if first_val > 0 else 0
                
                arrow = "↑" if improvement > 0 else "↓" if improvement < 0 else "→"
                f.write(f"  {metric:15s}: {arrow} {abs(improvement):6.2f}% "
                       f"({first_val:.6f} → {last_val:.6f})\n")
        
        f.write("\n" + "=" * 100 + "\n")
    
    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare model versions (v1, v2, v3).")
    parser.add_argument(
        "--v1-path",
        type=Path,
        default=Path("v1"),
        help="Path to v1 test dataset",
    )
    parser.add_argument(
        "--checkpoint-v1",
        type=Path,
        default=Path("cmsegnet_stage2.pt"),
        help="Path to v1 model checkpoint",
    )
    parser.add_argument(
        "--checkpoint-v2",
        type=Path,
        default=Path("../module 3/cmsegnet_stage2.pt"),
        help="Path to v2 model checkpoint (module 3)",
    )
    parser.add_argument(
        "--checkpoint-v3",
        type=Path,
        default=Path("../module 3/cmsegnet_stage2_head.pt"),
        help="Path to v3 model checkpoint (module 3 head)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_comparison_results"),
        help="Output directory for comparison results",
    )
    
    args = parser.parse_args()
    
    # Test each model version
    results = []
    
    try:
        results.append(test_model_version("v1 (Current)", args.checkpoint_v1, args.v1_path))
    except Exception as e:
        print(f"Error testing v1: {e}")
    
    try:
        results.append(test_model_version("v2 (Module 3)", args.checkpoint_v2, args.v1_path))
    except Exception as e:
        print(f"Error testing v2: {e}")
    
    try:
        results.append(test_model_version("v3 (Module 3 Head)", args.checkpoint_v3, args.v1_path))
    except Exception as e:
        print(f"Error testing v3: {e}")
    
    if not results:
        print("No models were successfully tested!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_df.to_csv(args.output / "comparison_results.csv", index=False)
    print(f"\nResults saved to: {args.output / 'comparison_results.csv'}")
    
    # Generate visualizations
    print("\nGenerating comparison visualizations...")
    create_comparison_visualizations(results_df, args.output)
    
    # Generate report
    print("\nGenerating comparison report...")
    generate_comparison_report(results_df, args.output)
    
    print("\n" + "=" * 80)
    print(f"Comparison complete! All results saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
