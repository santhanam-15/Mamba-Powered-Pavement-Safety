#!/bin/bash
# QUICK START SCRIPT - Copy this and run to start training with improvements

echo "🚀 Road Damage Detection Model - Training with Class Imbalance Fixes"
echo "=================================================================="

# Step 1: Show current configuration
echo ""
echo "Step 1: Current Configuration"
python switch_training_config.py --show

# Step 2: Set recommended configuration
echo ""
echo "Step 2: Applying RECOMMENDED configuration (high pos_weight, BCE loss)"
python switch_training_config.py --strategy high --loss bce

# Step 3: Start training
echo ""
echo "Step 3: Starting training..."
echo "  - Monitor progress in training.log"
echo "  - Training will run for 50 epochs"
echo "  - GPU will be used if available"
echo ""

python train.py

echo ""
echo "✓ Training completed!"
echo ""
echo "Next steps:"
echo "  1. Run threshold tuning:"
echo "     python tune_threshold.py --v1-path v1 --checkpoint cmsegnet_stage2.pt"
echo ""
echo "  2. Test on all samples:"
echo "     python test_model_enhanced.py --checkpoint cmsegnet_stage2.pt"
echo ""
echo "  3. Generate comparison (optional):"
echo "     python compare_results.py"
echo ""
