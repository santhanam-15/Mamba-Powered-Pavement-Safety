# Module 4 - v2: Fresh Start with Improvements

## Changes Made

### 1. **Configuration Upgrades in train.py**

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| POS_WEIGHT_STRATEGY | `"high"` (30.0) | `"extreme"` (57.0) | OLD: Recall only 16.5% → Need STRONGER class weighting |
| LOSS_TYPE | `"focal"` | `"focal"` | Focal Loss is more aggressive than BCE for imbalance |
| FOCAL_ALPHA | 0.25 | 0.30 | More weight on hard positive examples |
| FOCAL_GAMMA | 2.0 | 2.5 | Focus even harder on difficult samples |
| THRESHOLD_CANDIDATES | 0.2-0.71 | **0.15-0.55** | OLD: Recall poor at 0.7 → Search in lower range |

### 2. **Fresh Training (Checkpoint Deleted)**
- ❌ **Deleted**: `cmsegnet_stage2.pt` (old weights)
- ✅ **Result**: Will start from random initialization → fresh learning

---

## 📋 Step-by-Step Workflow

### **Phase 1: VERIFY Configuration (5 min)**

```bash
cd "d:\I FILES\Studies\sem6\MINI\module 4 -v2"
python switch_training_config.py --show
```

**Expected Output:**
```
Current Configuration:
  Loss Type: focal
  Pos Weight Strategy: extreme
  Pos Weight Value: 57.0
  Focal Alpha: 0.30
  Focal Gamma: 2.5
  Threshold Range: (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55)
```

✅ If this matches → Continue to Phase 2

---

### **Phase 2: TRAIN Model (4-8 hours)**

```bash
python train.py
```

**What to Expect:**
- Training log will show: `Using pos_weight=57.0 with Focal Loss`
- You'll see validation metrics improve over 50 epochs
- File `cmsegnet_stage2.pt` will be created (new model)
- Best threshold automatically saved in checkpoint

**Monitor Progress:**
```bash
# In another terminal, check metrics as they're saved:
tail -f "training.log"

# Or check CSV:
python -c "import pandas as pd; df = pd.read_csv('metrics.csv'); print(df[['epoch', 'val_loss', 'val_iou']].tail(10))"
```

**Target During Training:**
- Val IoU should increase from 0.09 → 0.30+ by epoch 50
- Val Recall should increase from 16% → 60%+ by epoch 50
- Val Precision should increase from 15% → 60%+ by epoch 50

---

### **Phase 3: TUNE Threshold (5 min)**

Once training completes:

```bash
python tune_threshold.py
```

**What to Expect:**
- Creates: `threshold_tuning_results.csv`
- Creates: `threshold_tuning.png` (4 plots)
- Creates: `threshold_recommendations.txt` (optimal thresholds)

**Example Output (Expected Values):**
```
Recommended Thresholds:
  - Maximize Recall (>70%): 0.15 → Recall: 75%, Precision: 55%, F1: 0.64
  - Balanced (>60% both): 0.20 → Recall: 68%, Precision: 62%, F1: 0.65
  - Conservative (>70%): 0.35 → Recall: 52%, Precision: 72%, F1: 0.60
```

---

### **Phase 4: TEST on V1 Samples (10 min)**

```bash
python test_model_enhanced.py
```

**What to Expect:**
- Processes all 224 test images
- Creates: `predictions/` folder with:
  - `per_image_metrics.csv`
  - `test_report.txt` (comprehensive summary)
  - Per-image folders with visualizations

**Target Results (Success Criteria):**
- ✅ **Recall**: >70% (was 16.5%)
- ✅ **Precision**: >60% (was 15.1%)
- ✅ **F1 Score**: >0.5 (was 0.157)
- ✅ **IoU**: >0.3 (was 0.086)
- ✅ **Fewer 0-IoU images**: <5 (was 10)

---

## 🐛 Troubleshooting

### **If Metrics Don't Improve:**

**Problem:** "Training ran but recall is still ~16%"

**Solution 1 - Check if Focal Loss was used:**
```bash
cat training.log | grep -i "focal\|pos_weight"
```
Expected: `"Using pos_weight=57.0 with Focal Loss"`

If you see `"Using BCE"`, then edit train.py again:
```python
LOSS_TYPE = "focal"  # ← Make sure this is set
```

**Solution 2 - Try extreme Focal Loss settings:**
```python
# In train.py, try:
FOCAL_ALPHA = 0.50  # Even higher
FOCAL_GAMMA = 3.0   # Even more focus
```

**Solution 3 - Try BCE with extreme pos_weight:**
```python
LOSS_TYPE = "bce"
POS_WEIGHT_STRATEGY = "extreme"  # Keep this
```

---

### **If Recall is Low (>30% but <70%):**

The THRESHOLD tuning will help! The problem might just be the detection threshold:
- Run: `python tune_threshold.py`
- Look at recommendations for "Maximize Recall" scenario
- Test model uses that threshold automatically

---

### **If Training Takes Too Long:**

Reduce epochs in train.py:
```python
NUM_EPOCHS = 30  # From 50, but not recommended
```

Or run for 50 epochs but check after 30:
- If Val IoU not improving for 10+ epochs → can stop early

---

## 📊 Comparison: v2 vs v1

| Metric | Module 4 (v1) | Module 4 - v2 (Expected) |
|--------|---------------|--------------------------|
| Pos Weight | 30.0 | **57.0** ✓ |
| Loss Function | Focal | **Focal (tuned)** ✓ |
| Focal Alpha | 0.25 | **0.30** ✓ |
| Focal Gamma | 2.0 | **2.5** ✓ |
| Threshold Range | 0.2-0.71 | **0.15-0.55** ✓ |
| Checkpoint | Old (loaded) | **Fresh** ✓ |
| **Expected Recall** | **16.5%** | **>70%** ⬆️ |
| **Expected Precision** | **15.1%** | **>60%** ⬆️ |
| **Expected F1** | **0.157** | **>0.5** ⬆️ |
| **Expected IoU** | **0.086** | **>0.3** ⬆️ |

---

## 🚀 Quick Start Command

Run all three phases in sequence:

```bash
cd "d:\I FILES\Studies\sem6\MINI\module 4 -v2"

# Verify config
python switch_training_config.py --show

# Train (will take 4-8 hours)
echo "Starting training..."
python train.py

# After training completes:
echo "Training done! Now tuning threshold..."
python tune_threshold.py

# Then test on all 224 images
echo "Testing model..."
python test_model_enhanced.py

# View results
cat predictions/test_report.txt
```

---

## ✅ When Done

Check `predictions/test_report.txt` for final metrics. If they show:
- Recall > 70%
- Precision > 60%
- F1 > 0.5
- IoU > 0.3

**🎉 SUCCESS! The class imbalance problem is FIXED!**

---

## 📝 Files Modified

- `train.py`: pos_weight_strategy changed to "extreme", Focal Loss parameters tuned, threshold range lowered
- `cmsegnet_stage2.pt`: DELETED (fresh training)

## 📝 Files Unchanged (but ready to use)

- `tune_threshold.py`: Already implemented
- `test_model_enhanced.py`: Already implemented
- `switch_training_config.py`: Configuration utility
- All data folders: Dataset/, etc.
