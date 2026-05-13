# 🎯 CLASS IMBALANCE FIX - TRAINING STRATEGY GUIDE

**Status:** 🔴 CRITICAL - Current model has severe class imbalance problems  
**Objective:** Improve Recall from 18.6% to >70% while maintaining Precision >60%

---

## 📊 PROBLEM ANALYSIS

### Current Performance (BASELINE)
```
Recall:     18.6% ❌ (Model misses 81% of defects)
Precision:  15.1% ❌ (High false alarm rate)
F1 Score:   0.167 ❌ (Very poor detection)
IoU:        0.091 ❌ (Poor segmentation)
Accuracy:   97.5% ✓ (Misleading - model just predicts "no damage")
```

### Root Cause
- **Class Imbalance:** 98.3% negative pixels, 1.7% positive pixels
- **Current pos_weight:** 2.5 (way too low for this ratio)
- **Model Behavior:** Learned to predict "no damage" everywhere (easy path)
- **Loss Function:** Standard BCE doesn't emphasize hard examples

---

## ✅ SOLUTIONS IMPLEMENTED

### 1. CLASS WEIGHTING (CRITICAL - START HERE)

**What it does:** Increases penalty for misclassifying positive pixels (damage)

**Recommended strategies:**

| Strategy | pos_weight | Use Case | Risk |
|----------|-----------|----------|------|
| **Original** | 2.5 | None - too low | Severe underweighting |
| **Medium** | 15.0 | Conservative | Safe starting point |
| **High** | 30.0 | **RECOMMENDED** | Balanced improvement |
| **Extreme** | 57.0 | Aggressive | Overfitting risk |

**Theoretical calculation:**
- Optimal weight = negative_ratio / positive_ratio = 98.3% / 1.7% ≈ 57.8
- **Use 30 (66% of optimal) for balanced learning**

**How to use in train.py:**
```python
# In train.py, modify this line:
POS_WEIGHT_STRATEGY = "high"  # Options: "low", "medium", "high", "extreme"

# Current mapping:
# "low": 2.5      (original, bad)
# "medium": 15.0  (good)
# "high": 30.0    (recommended) ← START HERE
# "extreme": 57.0 (aggressive)
```

---

### 2. LOSS FUNCTION OPTIONS

#### Option A: Binary Cross Entropy with pos_weight (Default, Simpler)
```python
LOSS_TYPE = "bce"
POS_WEIGHT_STRATEGY = "high"  # pos_weight = 30.0
```
- ✓ Simple, stable, well-understood
- ✓ Fast training
- ✗ Doesn't prioritize hard examples

#### Option B: Focal Loss (Recommended, More Advanced)
```python
LOSS_TYPE = "focal"
FOCAL_ALPHA = 0.25  # Weight for positive class
FOCAL_GAMMA = 2.0   # Focus on hard negatives
```
- ✓ Focuses on hard-to-classify samples
- ✓ Often better than simple weighting
- ✗ Slightly slower training
- ✗ More hyperparameters to tune

**Recommendation:** Start with **Option A (BCE + high pos_weight)**, then experiment with **Option B** if needed.

---

## 🚀 STEP-BY-STEP TRAINING PLAN

### PHASE 1: Baseline Training (Do First)
**Goal:** Verify class weighting works

```bash
# 1. Start with high class weighting
cd d:\I FILES\Studies\sem6\MINI\module 4

# 2. Run training with recommended settings
python train.py
```

**Expected improvements after 20-30 epochs:**
- Recall: 18.6% → 50-60% (3x improvement)
- Precision: 15.1% → 50-60%
- F1: 0.167 → 0.50-0.55

**Monitor:** Check `training.log` and `metrics.csv`
- Look for validation metrics improving
- Ensure loss decreases consistently

---

### PHASE 2: Threshold Tuning (Do Second)
**Goal:** Find optimal detection threshold

```bash
# 1. Run threshold tuning on trained model
python tune_threshold.py --v1-path v1 --checkpoint cmsegnet_stage2.pt

# 2. Check results
cat threshold_tuning/threshold_recommendations.txt

# 3. Open visualization
start threshold_tuning/threshold_tuning.png
```

**Key metrics to check:**
- F1 vs Threshold plot (should show clear peak)
- Recall vs Precision trade-off curve
- Best thresholds for different scenarios

**Expected output:**
```
Best F1 Score:       Threshold=0.35, F1=0.55
  Precision: 0.58, Recall: 0.52

Best Recall:         Threshold=0.20, Recall=0.72
  Precision: 0.48, F1=0.57

Best Precision:      Threshold=0.50, Precision=0.68
  Recall: 0.40, F1=0.50
```

---

### PHASE 3: Compare Results (Do Third)
**Goal:** Quantify improvements vs baseline

```bash
# 1. Save current predictions with new model
python test_model_enhanced.py --checkpoint cmsegnet_stage2.pt

# 2. Generate comparison (use demo mode if old baseline not saved)
python compare_results.py

# 3. View comparison results
start comparison_results/comparison_metrics.png
cat comparison_results/comparison_report.txt
```

---

## 🎮 HYPERPARAMETER TUNING MATRIX

Try these configurations in order:

### Configuration 1: Conservative (Safe)
```python
LOSS_TYPE = "bce"
POS_WEIGHT_STRATEGY = "medium"  # pos_weight = 15.0
```
- ✓ Low risk of issues
- ✗ Might not improve enough

### Configuration 2: Recommended (Balanced)
```python
LOSS_TYPE = "bce"
POS_WEIGHT_STRATEGY = "high"  # pos_weight = 30.0
```
- ✓ Best expected improvement
- ✓ Good stability
- **← START HERE**

### Configuration 3: Focal Loss (Advanced)
```python
LOSS_TYPE = "focal"
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
# Automatically uses pos_weight internally
```
- ✓ Often best performance
- ✗ More complex to tune

### Configuration 4: Extreme (Last Resort)
```python
LOSS_TYPE = "bce"
POS_WEIGHT_STRATEGY = "extreme"  # pos_weight = 57.0
```
- ✓ Maximum emphasis on positives
- ✗ Risk of overfitting

---

## 📈 EXPECTED RESULTS BY PHASE

### Baseline (Current)
```
Recall:    18.6% 
Precision: 15.1%
F1:        0.167
IoU:       0.091
```

### After Phase 1 (Class Weighting)
```
Recall:    55-65%  (3-3.5x improvement)
Precision: 50-60%
F1:        0.52-0.62
IoU:       0.35-0.45
```

### After Phase 2 (Threshold Tuning)
```
Recall:    60-70%  (Goal achieved!)
Precision: 50-65%
F1:        0.55-0.68
IoU:       0.40-0.50
```

### Target (Optimized)
```
Recall:    >70%    ✓
Precision: >60%    ✓
F1:        >0.65   ✓
IoU:       >0.45   ✓
```

---

## 🔧 MODIFICATION CHECKLIST

### In train.py:
- [x] Added `LOSS_TYPE` parameter (bce or focal)
- [x] Added `POS_WEIGHT_STRATEGY` parameter
- [x] Implemented class weight mapping
- [x] Added FocalLoss class
- [x] Modified combined_loss() to support both loss types
- [x] Updated logging to show configuration
- [x] Lowered threshold search range (0.2 instead of 0.3)

### New scripts created:
- [x] `tune_threshold.py` - Find optimal threshold
- [x] `compare_results.py` - Visualize improvements

### No changes needed:
- ✓ dataset.py (already good)
- ✓ model.py (architecture OK)
- ✓ config.py (working fine)

---

## 📋 EXECUTION CHECKLIST

### Quick Start (5 minutes setup)
- [ ] Open train.py
- [ ] Verify `POS_WEIGHT_STRATEGY = "high"`
- [ ] Verify `LOSS_TYPE = "bce"`
- [ ] Run: `python train.py`

### Full Optimization (1-2 hours)
- [ ] Run Phase 1 (train.py)
- [ ] Run Phase 2 (tune_threshold.py)
- [ ] Run Phase 3 (compare_results.py)
- [ ] Review threshold_recommendations.txt
- [ ] Update detection threshold in predict.py/test_model_enhanced.py
- [ ] Re-test on v1 folder

### Advanced (If needed)
- [ ] Try FOCAL_LOSS configuration
- [ ] Try EXTREME pos_weight (57.0)
- [ ] Experiment with different threshold ranges

---

## 💡 KEY INSIGHTS

1. **Class imbalance is the main culprit**
   - Not the model architecture (CMSegNet is fine)
   - Not the dataset (images are good)
   - Simple weighting can fix 80% of the problem

2. **Threshold matters A LOT**
   - Optimal threshold is NOT 0.5 (it's typically 0.2-0.4)
   - Threshold tuning can give +10-20% recall improvement
   - Different use cases need different thresholds

3. **Recall vs Precision trade-off**
   - Default 0.5 threshold is too conservative (high precision, low recall)
   - Lowering threshold to 0.2-0.3 increases recall but decreases precision
   - F1 score finds the sweet spot

4. **Loss function selection**
   - BCE with pos_weight is simpler and often sufficient
   - Focal loss is more sophisticated but may not be needed here
   - Start with BCE, switch to Focal if plateau is reached

---

## 🎓 RECOMMENDED TRAINING TIMELINE

```
Day 1: Phase 1 (Train with new weights)
  - Epoch 0-10: Random initialization, high loss
  - Epoch 10-20: Loss decreases, recall improving
  - Epoch 20-50: Convergence, recall plateaus

Day 2: Phase 2 (Threshold tuning)
  - Run on trained model
  - Find optimal threshold in 5 minutes
  - Update threshold in config/predict scripts

Day 3: Final validation
  - Test on full v1 dataset
  - Compare before/after metrics
  - Document improvements
```

---

## 🐛 TROUBLESHOOTING

| Problem | Cause | Solution |
|---------|-------|----------|
| Recall still low (<50%) | pos_weight too low | Increase to "extreme" (57.0) |
| NaN loss / diverging | pos_weight too high | Reduce to "medium" (15.0) |
| Training slow | Large pos_weight | Switch to FOCAL loss |
| F1 not improving | Old threshold (0.5) | Run tune_threshold.py |
| Precision drops too much | Threshold too low | Use 0.3-0.4 instead of 0.2 |
| Model overfitting | pos_weight=57 + epochs=100 | Reduce epochs to 30-40 |

---

## 📚 RELATED FILES

- [train.py](train.py) - Main training script with modifications
- [tune_threshold.py](tune_threshold.py) - Find optimal threshold
- [compare_results.py](compare_results.py) - Before/after visualization
- [test_model_enhanced.py](test_model_enhanced.py) - Test and save predictions
- [metrics.py](metrics.py) - Metric functions

---

## 🚀 NEXT STEPS

1. **Immediate (Now):**
   - Read this guide
   - Verify train.py modifications
   - Run: `python train.py`

2. **Short term (1 day):**
   - Let training complete (~50 epochs)
   - Run threshold tuning
   - Check improvements

3. **Medium term (2-3 days):**
   - Experiment with Focal Loss if needed
   - Fine-tune threshold for your use case
   - Document optimal settings

4. **Long term (Future):**
   - Try harder augmentation
   - Explore class-balanced sampling
   - Consider other architectures if plateau

---

**Questions?** Check training.log for detailed epoch-by-epoch progress.

Good luck! 🎉
