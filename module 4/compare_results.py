"""
Comparison visualization script: Generate before/after improvement graphs.

This script:
1. Compares old model (low pos_weight) with new model (high pos_weight/focal loss)
2. Visualizes metrics improvements
3. Creates summary reports
4. Generates comparison charts for presentation
"""

from pathlib import Path
from typing import Dict, List
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def load_csv_metrics(csv_path: Path) -> List[Dict]:
    """Load metrics from CSV file (e.g., per_image_metrics.csv)."""
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return []
    
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            numeric_fields = ["TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", 
                            "Specificity", "FPR", "FNR", "F1", "IoU", "Dice", "MCC"]
            for field in numeric_fields:
                if field in row:
                    row[field] = float(row[field])
            results.append(row)
    
    return results


def aggregate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Aggregate per-image metrics to overall metrics."""
    if not results:
        return {}
    
    metrics = {
        "Accuracy": np.mean([r.get("Accuracy", 0) for r in results]),
        "Precision": np.mean([r.get("Precision", 0) for r in results]),
        "Recall": np.mean([r.get("Recall", 0) for r in results]),
        "Specificity": np.mean([r.get("Specificity", 0) for r in results]),
        "F1": np.mean([r.get("F1", 0) for r in results]),
        "IoU": np.mean([r.get("IoU", 0) for r in results]),
        "Dice": np.mean([r.get("Dice", 0) for r in results]),
        "MCC": np.mean([r.get("MCC", 0) for r in results]),
    }
    return metrics


def create_comparison_report(
    old_metrics: Dict[str, float],
    new_metrics: Dict[str, float],
    old_name: str = "Baseline (pos_weight=2.5)",
    new_name: str = "Improved (pos_weight=30)",
    output_path: Path = None,
) -> str:
    """Create comparison report with improvements."""
    
    report_lines = [
        "=" * 100,
        "MODEL IMPROVEMENT COMPARISON REPORT",
        "=" * 100,
        "",
        f"Baseline Model:  {old_name}",
        f"Improved Model:  {new_name}",
        "",
        "-" * 100,
        f"{'Metric':<20} | {'Baseline':>15} | {'Improved':>15} | {'Change':>15} | {'% Improvement':>15}",
        "-" * 100,
    ]
    
    metrics_to_compare = ["Precision", "Recall", "F1", "IoU", "Accuracy", "Specificity", "Dice", "MCC"]
    
    for metric in metrics_to_compare:
        old_val = old_metrics.get(metric, 0)
        new_val = new_metrics.get(metric, 0)
        change = new_val - old_val
        pct_improvement = (change / abs(old_val) * 100) if old_val != 0 else 0
        
        report_lines.append(
            f"{metric:<20} | {old_val:>15.4f} | {new_val:>15.4f} | {change:>15.4f} | {pct_improvement:>14.1f}%"
        )
    
    report_lines.extend([
        "-" * 100,
        "",
        "KEY IMPROVEMENTS:",
        "",
        f"✓ Recall improved by {((new_metrics.get('Recall', 0) - old_metrics.get('Recall', 0)) / max(old_metrics.get('Recall', 1e-6), 1e-6) * 100):.1f}%",
        f"  (Model now catches more actual defects)",
        "",
        f"✓ Precision improved by {((new_metrics.get('Precision', 0) - old_metrics.get('Precision', 0)) / max(old_metrics.get('Precision', 1e-6), 1e-6) * 100):.1f}%",
        f"  (Fewer false alarms)",
        "",
        f"✓ F1 Score improved by {((new_metrics.get('F1', 0) - old_metrics.get('F1', 0)) / max(old_metrics.get('F1', 1e-6), 1e-6) * 100):.1f}%",
        f"  (Overall balanced performance)",
        "",
        f"✓ IoU improved by {((new_metrics.get('IoU', 0) - old_metrics.get('IoU', 0)) / max(old_metrics.get('IoU', 1e-6), 1e-6) * 100):.1f}%",
        f"  (Better segmentation accuracy)",
        "",
        "=" * 100,
    ])
    
    report_text = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
    
    return report_text


def create_comparison_visualizations(
    old_metrics: Dict[str, float],
    new_metrics: Dict[str, float],
    old_name: str = "Baseline",
    new_name: str = "Improved",
    output_dir: Path = None,
) -> None:
    """Create comparison visualization charts."""
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_list = ["Precision", "Recall", "F1", "IoU"]
    old_vals = [old_metrics.get(m, 0) for m in metrics_list]
    new_vals = [new_metrics.get(m, 0) for m in metrics_list]
    
    # Chart 1: Metrics Comparison (Bar)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_list))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, old_vals, width, label=old_name, color="#FF6B6B", alpha=0.8)
    bars2 = ax.bar(x + width/2, new_vals, width, label=new_name, color="#4ECDC4", alpha=0.8)
    
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Model Comparison: Key Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "comparison_metrics.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # Chart 2: Improvement Percentage
    fig, ax = plt.subplots(figsize=(12, 6))
    
    improvements = []
    for m, old_v, new_v in zip(metrics_list, old_vals, new_vals):
        if old_v > 0:
            improvement_pct = ((new_v - old_v) / old_v) * 100
        else:
            improvement_pct = 0 if new_v == 0 else 100
        improvements.append(improvement_pct)
    
    colors = ["#2ECC71" if x > 0 else "#E74C3C" for x in improvements]
    bars = ax.bar(metrics_list, improvements, color=colors, alpha=0.8)
    
    ax.set_ylabel("Improvement %", fontsize=12, fontweight="bold")
    ax.set_title("Performance Improvement (%)", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:+.1f}%',
               ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "improvement_percentage.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # Chart 3: Extended Metrics Comparison
    extended_metrics = ["Accuracy", "Specificity", "Dice", "MCC"]
    extended_old = [old_metrics.get(m, 0) for m in extended_metrics]
    extended_new = [new_metrics.get(m, 0) for m in extended_metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(extended_metrics))
    bars1 = ax.bar(x - width/2, extended_old, width, label=old_name, color="#FF6B6B", alpha=0.8)
    bars2 = ax.bar(x + width/2, extended_new, width, label=new_name, color="#4ECDC4", alpha=0.8)
    
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Extended Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(extended_metrics)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "extended_metrics.png", dpi=150, bbox_inches="tight")
    plt.show()


def generate_comparison(
    old_csv_path: Path,
    new_csv_path: Path,
    output_dir: Path,
    old_name: str = "Baseline",
    new_name: str = "Improved",
) -> None:
    """Generate full comparison between two model results."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading old model results from {old_csv_path}...")
    old_results = load_csv_metrics(old_csv_path)
    old_metrics = aggregate_metrics(old_results)
    
    print(f"Loading new model results from {new_csv_path}...")
    new_results = load_csv_metrics(new_csv_path)
    new_metrics = aggregate_metrics(new_results)
    
    # Print metrics
    print("\nOld Model Metrics:")
    for k, v in old_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nNew Model Metrics:")
    for k, v in new_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Create report
    report_path = output_dir / "comparison_report.txt"
    report = create_comparison_report(old_metrics, new_metrics, old_name, new_name, report_path)
    print("\n" + report)
    
    # Create visualizations
    create_comparison_visualizations(old_metrics, new_metrics, old_name, new_name, output_dir)
    
    print(f"\nComparison results saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate before/after comparison visualizations.")
    parser.add_argument(
        "--old-csv",
        type=Path,
        help="Path to baseline model metrics CSV (e.g., predictions/per_image_metrics.csv)",
    )
    parser.add_argument(
        "--new-csv",
        type=Path,
        help="Path to improved model metrics CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("comparison_results"),
        help="Output directory for comparison results",
    )
    parser.add_argument(
        "--old-name",
        type=str,
        default="Baseline (pos_weight=2.5)",
        help="Name for baseline model",
    )
    parser.add_argument(
        "--new-name",
        type=str,
        default="Improved (pos_weight=30)",
        help="Name for improved model",
    )
    
    args = parser.parse_args()
    
    if not args.old_csv or not args.new_csv:
        # Demo mode: create sample comparison
        print("Demo mode: Creating sample comparison with synthetic data...")
        
        # Baseline metrics
        old_metrics = {
            "Accuracy": 0.975,
            "Precision": 0.151,
            "Recall": 0.186,
            "Specificity": 0.998,
            "F1": 0.167,
            "IoU": 0.091,
            "Dice": 0.167,
            "MCC": 0.245,
        }
        
        # Improved metrics (with class weighting)
        new_metrics = {
            "Accuracy": 0.892,
            "Precision": 0.582,
            "Recall": 0.641,
            "Specificity": 0.903,
            "F1": 0.610,
            "IoU": 0.438,
            "Dice": 0.610,
            "MCC": 0.581,
        }
        
        output_dir = Path("comparison_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / "comparison_report.txt"
        report = create_comparison_report(old_metrics, new_metrics, args.old_name, args.new_name, report_path)
        print("\n" + report)
        
        create_comparison_visualizations(old_metrics, new_metrics, args.old_name, args.new_name, output_dir)
        
        print(f"\nComparison results saved to {output_dir}/")
    else:
        generate_comparison(args.old_csv, args.new_csv, args.output, args.old_name, args.new_name)
