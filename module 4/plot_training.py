"""
Training Metrics Visualization Script
Plots training progress from metrics.csv for analysis and reporting.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

def load_metrics(csv_file: str = "metrics.csv") -> pd.DataFrame:
    """Load metrics from CSV file."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Metrics file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded metrics for {len(df)} epochs")
    return df


def plot_training_metrics(df: pd.DataFrame, output_file: str = "training_analysis.png") -> None:
    """Create comprehensive training visualization."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Validation IoU Progress
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(df['Epoch'], df['Val IoU'], 'b-', linewidth=2, label='Val IoU', marker='o', markersize=4)
    best_iou = df['Val IoU'].max()
    best_epoch = df.loc[df['Val IoU'].idxmax(), 'Epoch']
    ax1.axhline(y=best_iou, color='g', linestyle='--', linewidth=2, label=f'Best: {best_iou:.4f}')
    ax1.axhline(y=0.3397, color='orange', linestyle='--', linewidth=2, label='Previous Best: 0.3397')
    ax1.scatter([best_epoch], [best_iou], color='g', s=100, zorder=5, marker='*')
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Validation IoU', fontsize=11, fontweight='bold')
    ax1.set_title('Validation IoU Progress', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(best_epoch, best_iou + 0.01, f'Epoch {int(best_epoch)}', ha='center', fontsize=9, fontweight='bold')
    
    # 2. Learning Rate Decay
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(df['Epoch'], df['Learning Rate'], 'r-', linewidth=2, label='Learning Rate', marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    ax2.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='best', fontsize=10)
    
    # 3. Loss Comparison
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(df['Epoch'], df['Train Loss'], 'g-', linewidth=2, label='Train Loss', marker='^', markersize=4)
    ax3.plot(df['Epoch'], df['Val Loss'], 'r-', linewidth=2, label='Val Loss', marker='v', markersize=4)
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax3.set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Checkpoint Saves Timeline
    ax4 = plt.subplot(2, 3, 4)
    checkpoints = df[df['Checkpoint Saved'] == 'Yes']
    no_checkpoints = df[df['Checkpoint Saved'] == 'No']
    ax4.scatter(checkpoints['Epoch'], checkpoints['Val IoU'], color='g', s=200, label='Checkpoint Saved', marker='*', zorder=5)
    ax4.scatter(no_checkpoints['Epoch'], no_checkpoints['Val IoU'], color='gray', s=50, label='No Checkpoint', alpha=0.5)
    ax4.plot(df['Epoch'], df['Val IoU'], 'b--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Validation IoU', fontsize=11, fontweight='bold')
    ax4.set_title('Checkpoint Saves', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Best Threshold Trend
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(df['Epoch'], df['Best Threshold'], 'purple', linewidth=2, label='Best Threshold', marker='d', markersize=4)
    ax5.axhline(y=df['Best Threshold'].mean(), color='orange', linestyle='--', linewidth=1.5, label=f'Mean: {df["Best Threshold"].mean():.2f}')
    ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Threshold', fontsize=11, fontweight='bold')
    ax5.set_title('Optimal Threshold per Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylim([0.3, 0.7])
    ax5.legend(loc='best', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Improvement Rate
    ax6 = plt.subplot(2, 3, 6)
    improvement = df['Val IoU'].diff()
    colors = ['g' if x >= 0 else 'r' for x in improvement]
    ax6.bar(df['Epoch'], improvement, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax6.axhline(y=0, color='black', linewidth=1)
    ax6.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax6.set_ylabel('IoU Change from Previous', fontsize=11, fontweight='bold')
    ax6.set_title('Epoch-to-Epoch Improvement', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_file}")
    plt.show()


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "="*70)
    print("TRAINING SUMMARY STATISTICS")
    print("="*70)
    
    best_idx = df['Val IoU'].idxmax()
    worst_idx = df['Val IoU'].idxmin()
    
    print(f"\nTotal Epochs Trained: {len(df)}")
    print(f"Checkpoints Saved: {(df['Checkpoint Saved'] == 'Yes').sum()}")
    
    print(f"\n{'Metric':<30} {'Value':<15} {'At Epoch':<10}")
    print("-" * 55)
    print(f"{'Best Val IoU':<30} {df.loc[best_idx, 'Val IoU']:<15.6f} {int(df.loc[best_idx, 'Epoch']):<10}")
    print(f"{'Worst Val IoU':<30} {df.loc[worst_idx, 'Val IoU']:<15.6f} {int(df.loc[worst_idx, 'Epoch']):<10}")
    print(f"{'Best Threshold (at best IoU)':<30} {df.loc[best_idx, 'Best Threshold']:<15.2f}")
    print(f"{'Final Train Loss':<30} {df.iloc[-1]['Train Loss']:<15.6f}")
    print(f"{'Final Val Loss':<30} {df.iloc[-1]['Val Loss']:<15.6f}")
    print(f"{'Final Val IoU':<30} {df.iloc[-1]['Val IoU']:<15.6f}")
    print(f"{'Initial Learning Rate':<30} {df.iloc[0]['Learning Rate']:<15.2e}")
    print(f"{'Final Learning Rate':<30} {df.iloc[-1]['Learning Rate']:<15.2e}")
    
    improvement = df.iloc[-1]['Val IoU'] - df.iloc[0]['Val IoU']
    improvement_pct = (improvement / df.iloc[0]['Val IoU'] * 100) if df.iloc[0]['Val IoU'] > 0 else 0
    print(f"\nTotal IoU Improvement: {improvement:+.6f} ({improvement_pct:+.2f}%)")
    
    epochs_improved = (df['Val IoU'].diff() > 0).sum()
    epochs_degraded = (df['Val IoU'].diff() < 0).sum()
    print(f"Epochs with Improvement: {epochs_improved}")
    print(f"Epochs with Degradation: {epochs_degraded}")
    
    print("\n" + "="*70)


def main():
    """Main execution."""
    try:
        # Load metrics
        df = load_metrics("metrics.csv")
        
        # Print statistics
        print_summary_stats(df)
        
        # Create plots
        plot_training_metrics(df, "training_analysis.png")
        
        print("\n✓ Analysis complete! Check 'training_analysis.png' for visualizations.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
