"""
Unified Model Comparison Script - Single File Solution

Evaluates multiple models and generates comprehensive visualizations automatically.

Usage:
    python compare.py
    python compare.py --output my_results
    python compare.py --test-dir ../Dataset/TestSet --output results
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv

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


# ===== EDIT HERE: Add your models =====
MODEL_CONFIG = {
    "Baseline (BCE, low weight)": {
        "path": "checkpoints/cmsegnet_stage2.pt",
        "description": "Original with pos_weight=2.5"
    },
    "Improved (Focal Loss)": {
        "path": "cmsegnet_stage2.pt",
        "description": "With Focal Loss, pos_weight=30"
    },
    # Add more models here:
    # "Model Name": {"path": "path/to/checkpoint.pt", "description": "Description"},
}


# Set matplotlib style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


# ===== UTILITY FUNCTIONS =====

def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[CMSegNet, float]:
    """Load trained model."""
    model = CMSegNet(img_size=IMG_SIZE).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise FileNotFoundError(f"Cannot load checkpoint {checkpoint_path}: {e}")
    
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    threshold = checkpoint.get("best_threshold", 0.5) if isinstance(checkpoint, dict) else 0.5
    return model, float(threshold)


def preprocess_image(image_path: Path) -> Tuple[np.ndarray, torch.Tensor]:
    """Preprocess image."""
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
    corrected_shape: Tuple,
) -> np.ndarray:
    """Run inference and return binary mask."""
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
        interpolation=cv2.INTER_NEAREST
    )
    return bitmask


def compute_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    pred_flat = prediction.flatten().astype(bool)
    gt_flat = ground_truth.flatten().astype(bool)
    
    tp = np.sum(pred_flat & gt_flat)
    tn = np.sum(~pred_flat & ~gt_flat)
    fp = np.sum(pred_flat & ~gt_flat)
    fn = np.sum(~pred_flat & gt_flat)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "IoU": iou,
    }


def evaluate_model(
    model_name: str,
    checkpoint_path: Path,
    test_images_dir: Path,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate single model on test set."""
    print(f"  📊 {model_name}")
    
    if not checkpoint_path.exists():
        print(f"    ❌ Checkpoint not found: {checkpoint_path}")
        return {}
    
    try:
        model, threshold = load_model(checkpoint_path, device)
    except Exception as e:
        print(f"    ❌ Failed to load: {e}")
        return {}
    
    metrics_list = []
    image_files = sorted(test_images_dir.glob("*_raw.png"))
    
    if not image_files:
        print(f"    ⚠️  No test images found")
        return {}
    
    for image_file in tqdm(image_files, desc=f"    Evaluating", leave=False):
        try:
            corrected_rgb, tensor = preprocess_image(image_file)
            prediction = run_inference(model, tensor, device, threshold, corrected_rgb.shape)
            
            gt_file = image_file.parent / image_file.name.replace("_raw", "_pothole")
            if gt_file.exists():
                ground_truth = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
                if ground_truth is not None:
                    ground_truth = (ground_truth > 127).astype(np.uint8)
                    metrics = compute_metrics(prediction, ground_truth)
                    metrics_list.append(metrics)
        except Exception:
            continue
    
    if not metrics_list:
        print(f"    ⚠️  No valid results")
        return {}
    
    aggregated = {}
    for key in metrics_list[0].keys():
        aggregated[key] = np.mean([m[key] for m in metrics_list])
    
    print(f"    ✅ {len(metrics_list)} images evaluated")
    return aggregated


def create_comparison_dataframe(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert results to DataFrame."""
    df_data = []
    for model_name, metrics in results.items():
        if metrics:
            row = {"Model": model_name}
            row.update(metrics)
            df_data.append(row)
    return pd.DataFrame(df_data)


def generate_report(df: pd.DataFrame, output_dir: Path) -> str:
    """Generate comprehensive report."""
    metrics = [col for col in df.columns if col != "Model"]
    
    report_lines = [
        "=" * 140,
        "MODEL COMPARISON REPORT",
        "=" * 140,
        f"Models Compared: {len(df)}",
        f"Timestamp: {pd.Timestamp.now()}",
        "",
    ]
    
    # Results table
    report_lines.append(df.to_string(index=False))
    report_lines.append("")
    
    # Best performers
    report_lines.extend([
        "=" * 140,
        "🏆 BEST PERFORMERS (per metric):",
        "=" * 140,
    ])
    
    for metric in metrics:
        if metric in df.columns:
            best_idx = df[metric].idxmax()
            best_model = df.loc[best_idx, "Model"]
            best_score = df.loc[best_idx, metric]
            report_lines.append(f"  {metric:<20}: {str(best_model):<60} ({best_score:.4f})")
    
    # Comparative analysis
    if len(df) > 1:
        report_lines.extend([
            "",
            "=" * 140,
            "📊 COMPARATIVE ANALYSIS:",
            "=" * 140,
        ])
        baseline_model = df.loc[0, "Model"]
        
        for idx in range(1, len(df)):
            model_name = df.loc[idx, "Model"]
            report_lines.append(f"\n{idx}. {model_name} vs {baseline_model}:")
            report_lines.append("-" * 100)
            
            for metric in metrics:
                baseline_val = df.loc[0, metric]
                current_val = df.loc[idx, metric]
                diff = current_val - baseline_val
                pct = (diff / baseline_val * 100) if baseline_val != 0 else 0
                
                status = "✓" if diff > 0 else ("✗" if diff < 0 else "=")
                report_lines.append(
                    f"  {metric:<20}: {baseline_val:.4f} → {current_val:.4f} "
                    f"({diff:+.4f}, {pct:+.2f}%) {status}"
                )
    
    # Summary
    report_lines.extend([
        "",
        "=" * 140,
        "📝 SUMMARY:",
        "=" * 140,
    ])
    
    overall_best = df.loc[df[metrics].mean(axis=1).idxmax(), "Model"]
    report_lines.append(f"Best Overall Model: {overall_best}")
    
    report_text = "\n".join(report_lines)
    
    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    return report_text


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Create all visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = [col for col in df.columns if col != "Model"]
    num_models = len(df)
    
    # Dynamic colors
    if num_models <= 3:
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"][:num_models]
    elif num_models <= 6:
        colors = list(plt.cm.Set2(np.linspace(0, 1, num_models)))
    else:
        colors = list(plt.cm.tab20(np.linspace(0, 1, num_models)))
    
    print(f"\n📊 Creating {len(metrics)} visualizations...")
    
    # Chart 1: Bar Chart
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(metrics))
    width = 0.8 / num_models
    
    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in metrics]
        offset = (idx - num_models/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=row["Model"], color=colors[idx], alpha=0.85)
    
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(f"Model Comparison - {num_models} Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_bar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Chart 2: Line Chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in metrics]
        ax.plot(metrics, values, marker='o', linewidth=2.5, markersize=9,
               label=row["Model"], color=colors[idx], alpha=0.85)
    
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Trends", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / "02_line_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Chart 3: Heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(metrics)), min(12, len(df) * 0.8)))
    
    heatmap_data = df.set_index("Model")[metrics]
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="RdYlGn",
               vmin=0, vmax=1, cbar_kws={"label": "Score"}, ax=ax,
               linewidths=0.5, linecolor="gray")
    ax.set_title("Comparison Heatmap", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "03_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Chart 4: Radar (if ≤6 models)
    if num_models <= 6:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for idx, (_, row) in enumerate(df.iterrows()):
            values = [row[m] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row["Model"],
                   color=colors[idx], alpha=0.75)
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10)
        ax.set_ylim(0, 1)
        ax.set_title("Radar Chart", fontsize=14, fontweight="bold", pad=20)
        ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        plt.savefig(output_dir / "04_radar.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    # Chart 5: Best Models
    fig, ax = plt.subplots(figsize=(12, 6))
    
    best_scores = []
    best_models = []
    for metric in metrics:
        best_idx = df[metric].idxmax()
        best_score = df.loc[best_idx, metric]
        best_model = str(df.loc[best_idx, "Model"])[:30]
        best_scores.append(best_score)
        best_models.append(best_model)
    
    bars = ax.barh(metrics, best_scores, color=colors[:len(metrics)], alpha=0.85)
    ax.set_xlabel("Best Score", fontsize=12, fontweight="bold")
    ax.set_title("Best Model per Metric", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    
    for bar, model in zip(bars, best_models):
        width = bar.get_width()
        ax.text(width - 0.03, bar.get_y() + bar.get_height()/2.,
               f'{model} ({width:.4f})',
               ha='right', va='center', fontsize=9, fontweight="bold", color="white")
    
    plt.tight_layout()
    plt.savefig(output_dir / "05_best_models.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Chart 6: Average Score
    fig, ax = plt.subplots(figsize=(12, 6))
    
    avg_scores = df[metrics].mean(axis=1).values
    model_names = df["Model"].values
    
    bars = ax.bar(range(len(model_names)), avg_scores, color=colors, alpha=0.85)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([str(m)[:40] for m in model_names], rotation=45, ha='right')
    ax.set_ylabel("Average Score", fontsize=12, fontweight="bold")
    ax.set_title("Average Performance Ranking", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    
    for bar, score in zip(bars, avg_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{score:.4f}',
               ha='center', va='bottom', fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_dir / "06_average_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Saved 6 visualizations to: {output_dir}")


def save_csv(df: pd.DataFrame, output_dir: Path) -> None:
    """Save results to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Model Comparison - Evaluation + Visualization in One",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", type=str, default="results/model_comparisons",
                       help="Output directory for results")
    parser.add_argument("--test-dir", type=str, default="../../Dataset/Stage1_Corrected",
                       help="Test images directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    test_dir = Path(args.test_dir)
    
    print("\n" + "=" * 100)
    print("🚀 UNIFIED MODEL COMPARISON")
    print("=" * 100)
    print(f"Models to compare: {len(MODEL_CONFIG)}")
    print(f"Test directory: {test_dir}")
    print(f"Output directory: {output_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Evaluate all models
    print("📊 Evaluating Models:")
    results = {}
    for model_name, model_info in MODEL_CONFIG.items():
        checkpoint_path = Path(model_info["path"])
        metrics = evaluate_model(model_name, checkpoint_path, test_dir, device)
        results[model_name] = metrics
    
    # Create DataFrame
    df = create_comparison_dataframe(results)
    
    if df.empty:
        print("❌ No valid results. Check model paths and test directory.")
        return
    
    # Display results
    print("\n" + "=" * 100)
    print("📈 RESULTS")
    print("=" * 100)
    print(df.to_string(index=False))
    
    # Save CSV
    save_csv(df, output_dir)
    
    # Generate report
    print("\n📄 Generating report...")
    report = generate_report(df, output_dir)
    print(report)
    
    # Create visualizations
    create_visualizations(df, output_dir)
    
    print("\n" + "=" * 100)
    print("✅ COMPARISON COMPLETE")
    print("=" * 100)
    print(f"📁 Output: {output_dir}")
    print(f"   - comparison_results.csv")
    print(f"   - comparison_report.txt")
    print(f"   - 6 visualization PNG files")


if __name__ == "__main__":
    main()
