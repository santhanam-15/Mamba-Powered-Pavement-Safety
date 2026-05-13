# 🚀 Unified Model Comparison - Single File

## Quick Start

### Step 1: Add Models
Edit `compare.py` - find `MODEL_CONFIG`:

```python
MODEL_CONFIG = {
    "Baseline (BCE, low weight)": {
        "path": "checkpoints/cmsegnet_stage2.pt",
        "description": "Original with pos_weight=2.5"
    },
    "Improved (Focal Loss)": {
        "path": "cmsegnet_stage2.pt",
        "description": "With Focal Loss, pos_weight=30"
    },
    # Add more:
    # "Model 3": {"path": "path/to/model.pt", "description": "..."},
}
```

### Step 2: Run (One Command!)

From `module 4 -v2` directory:

```bash
cd src
python ../comparison_tools/compare.py
```

Or with custom output:

```bash
python ../comparison_tools/compare.py --output results/my_comparison --test-dir ../../Dataset/Stage1_Corrected
```

### Step 3: View Results

```
results/model_comparisons/
├── comparison_results.csv        # Raw metrics
├── comparison_report.txt         # Text report
├── 01_bar_comparison.png         # Bar chart
├── 02_line_comparison.png        # Line chart
├── 03_heatmap.png                # Heatmap
├── 04_radar.png                  # Radar (if ≤6 models)
├── 05_best_models.png            # Best per metric
└── 06_average_scores.png         # Overall ranking
```

---

## Features

✅ **Single File** - No more two-step process  
✅ **Add Any Number of Models** - Edit `MODEL_CONFIG`  
✅ **Automatic Evaluation** - Runs inference on all test images  
✅ **6 Visualizations** - Bar, line, heatmap, radar, best models, averages  
✅ **Detailed Report** - Text report with improvements vs baseline  
✅ **CSV Output** - For Excel analysis  

---

## Example: 3-Model Comparison

Edit `MODEL_CONFIG`:
```python
MODEL_CONFIG = {
    "Baseline": {
        "path": "checkpoints/cmsegnet_stage2.pt",
        "description": "Original"
    },
    "Improved": {
        "path": "cmsegnet_stage2.pt",
        "description": "Focal Loss"
    },
    "v3 Best": {
        "path": "v3/cmsegnet_stage2.pt",
        "description": "Extended training"
    },
}
```

Run:
```bash
python ../comparison_tools/compare.py
```

Done! ✨

---

## Command Options

```bash
# Default (saves to results/model_comparisons)
python ../comparison_tools/compare.py

# Custom output
python ../comparison_tools/compare.py --output my_results

# Custom test directory
python ../comparison_tools/compare.py --test-dir ../../Dataset/Custom

# Both
python ../comparison_tools/compare.py --output my_results --test-dir ../../Dataset/Custom
```

---

## What It Does

1. **Loads each model checkpoint**
2. **Runs inference on all test images** (computes masks)
3. **Compares predictions with ground truth** (computes metrics)
4. **Generates 6 different visualizations**
5. **Creates detailed text report** with improvements
6. **Saves CSV** for further analysis

All in **one command**! 🎉
