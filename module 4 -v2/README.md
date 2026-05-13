# Module 4 - Road Damage Segmentation

This project implements an end-to-end road damage segmentation pipeline using a CMSegNet-style model, stage-1 perspective correction, and optional shadow suppression for more stable pothole/crack predictions.

## Model Variants

### Model 1: CMSegNet

CMSegNet is the main segmentation model used in this project. It learns to detect road damage regions such as cracks and potholes from corrected road images. The model combines convolution-based local feature extraction with a lightweight global-context block, making it suitable for segmentation on complex road scenes.

### Model 2: CMSegNet Edge

CMSegNet Edge is the enhanced version of CMSegNet with an auxiliary edge-detection head. In addition to the main segmentation output, it learns boundary information so the predicted damage masks become sharper and more precise around object edges. This helps improve segmentation quality on small or irregular damage regions.

### Model 3: CMSegNet + ShadowNet

CMSegNet + ShadowNet is the full inference pipeline that combines CMSegNet with a shadow-suppression model. CMSegNet predicts the damage mask, while ShadowNet estimates shadow regions and reduces false positives caused by dark road shadows. This version is used for more stable real-world predictions during inference.

## Objectives

1. Build an accurate road-damage segmentation model for cracks and potholes from road images.
2. Improve performance on imbalanced data using focal/class-weighted loss and threshold tuning.
3. Provide a complete workflow for training, inference, and model-version comparison.

## Outcomes

1. The model produces cleaner segmentation masks with improved IoU/Dice quality.
2. Minority damage regions are detected more consistently with fewer missed positives.
3. The project generates reproducible checkpoints, prediction artifacts, and comparison reports.

## Project Structure (Key Files)

- `train.py`: Trains CMSegNet with class-imbalance strategy, LR scheduling, and checkpoint saving.
- `predict.py`: Runs single-image inference and saves corrected image, bitmask, overlay, and area summary.
- `run_module4.py`: CLI wrapper for single-image or folder batch inference.
- `compare_model_versions.py`: Compares multiple checkpoints (v1/v2/v3 style) on common test data.
- `compare_results.py`: Visual comparison between two result CSV files.
- `dataset.py`: Dataset loader expecting `_RAW`, `_CRACK`, `_POTHOLE` triplets.
- `model.py`: CMSegNet, auxiliary edge head, and ShadowNet definitions.
- `stage1_utils.py`: Perspective correction and geometric calibration utilities.
- `config.py`: Core settings like paths, image size, and default hyperparameters.

## Requirements

- Python 3.9+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Seaborn
- tqdm

Install dependencies:

```bash
pip install torch torchvision torchaudio opencv-python numpy matplotlib pandas seaborn tqdm
```

## Dataset Format

The loader scans recursively under `DATASET_PATH` (set in `config.py`) and expects 3 files per sample with the same prefix:

- `<id>_RAW.*`
- `<id>_CRACK.*`
- `<id>_POTHOLE.*`

Example:

```text
Dataset/Stage1_Corrected/
  1092842_DF_251_251BDF0052_00665_RAW.png
  1092842_DF_251_251BDF0052_00665_CRACK.png
  1092842_DF_251_251BDF0052_00665_POTHOLE.png
```

## Configuration

Main defaults are in `config.py`:

- `DATASET_PATH = "../Dataset/Stage1_Corrected"`
- `CHECKPOINT_PATH = "cmsegnet_stage2.pt"`
- `PREDICTION_DIR = "predictions"`
- `IMG_SIZE = 256`

Update these before training/inference if your paths differ.

## Training

Run:

```bash
python train.py
```

What it does:

- Splits data into train/validation subsets.
- Trains CMSegNet with segmentation + edge loss.
- Applies class-imbalance strategy (focal or weighted BCE).
- Searches best threshold on validation set.
- Saves best checkpoint to `CHECKPOINT_PATH`.
- Writes `training.log` and `metrics.csv`.

## Inference

### Single image

```bash
python run_module4.py test.jpg
```

### Batch folder

```bash
python run_module4.py --folder path/to/images
```

### Optional checkpoint/output override

```bash
python run_module4.py test.jpg --checkpoint cmsegnet_stage2.pt --output-dir predictions/custom_output
```

For each image, outputs are saved in the image-specific prediction folder:

- `original.png`
- `corrected.png`
- `predicted_bitmask.png`
- `overlay.png`
- `area_summary.txt`
- `area_summary.png`

## Model Comparison

### Compare multiple model versions

```bash
python compare_model_versions.py --v1-path v1 --checkpoint-v1 path/to/v1.pt --checkpoint-v2 v2.pt --checkpoint-v3 cmsegnet_stage2.pt --output model_comparison_results
```

Outputs include CSV metrics, plots, and a text report in the output directory.

### Compare two CSV result files

```bash
python compare_results.py --old-csv path/to/old/per_image_metrics.csv --new-csv path/to/new/per_image_metrics.csv --output comparison_results
```

## Typical Workflow

1. Set paths in `config.py`.
2. Train model with `python train.py`.
3. Run inference with `python run_module4.py ...`.
4. Evaluate or compare checkpoints with `compare_model_versions.py`.
5. Prepare presentation graphs with `compare_results.py`.

## Notes

- If checkpoint is missing or untrained, inference scripts raise a clear error.
- If ShadowNet output appears near-constant, inference falls back to CMSegNet mask for stability.
- Default training settings in `train.py` are tuned for imbalance handling and may differ from `config.py` epoch/split values.
