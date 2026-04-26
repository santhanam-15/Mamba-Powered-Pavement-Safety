# Module 3 Guide

## What This Module Does

Module 3 is **Stage 2: model architecture, training, and prediction**.

It uses the corrected output from Module 2, trains the segmentation model, and later predicts a binary damage mask for a new road image.

## Files And Purpose

- `config.py`
  Stores stage-2 settings:
  - image size
  - batch size
  - learning rate
  - checkpoint path
  - prediction output folder

- `dataset.py`
  Loads corrected images from Module 2 and combines:
  - crack mask
  - pothole mask
  into one final binary target mask.

- `model.py`
  Implements the stage-2 CMSegNet-style structure:
  - `CMEncoder`
  - `CNNBranch`
  - `OptimizedMambaBlock`
  - `MSAAModule`
  - `LightweightMLPDecoder`

- `metrics.py`
  Loss/metric helpers:
  - Dice loss
  - IoU score

- `train.py`
  Trains the model and saves the best checkpoint.

- `stage1_utils.py`
  Reuses the stage-1 correction logic during prediction so a new raw road image is corrected before inference.

- `predict.py`
  End-to-end prediction script:
  1. take raw road image
  2. run Module 2 style correction
  3. run stage-2 model
  4. save corrected image
  5. save predicted bitmask

## Full Flow

### Training Flow

1. Module 2 produces corrected images in `Dataset/Stage1_Corrected`.
2. `dataset.py` loads corrected raw image + corrected crack mask + corrected pothole mask.
3. Crack and pothole masks are merged into one final binary target.
4. `train.py` splits the corrected dataset into train and validation parts.
5. `model.py` trains the stage-2 segmentation network.
6. Best checkpoint is saved as `cmsegnet_stage2.pt`.

### Prediction Flow

1. You give one new raw road image.
2. `predict.py` runs stage-1 correction first.
3. The corrected image is resized and passed into the trained model.
4. The model outputs a probability map.
5. Probability map is thresholded into a binary bitmask.
6. Final output folder contains:
   - corrected image
   - predicted bitmask

## Meaning Of `Train Batches` And `Val Batches`

If training shows:

- `Train batches: 894`
- `Val batches: 224`

this means:

- total corrected samples = `2235`
- train split = `80%` = `1788` images
- validation split = `20%` = `447` images
- batch size = `2`

So:

- `1788 / 2 = 894` training batches
- `447 / 2 = 223.5`, so PyTorch makes `224` validation batches

One **batch** means one mini-group of images processed together in one step.

## Why Training Takes Around 5 Minutes On RTX 3050 4 GB

That is normal for your setup because:

- dataset size is fairly large: `2235` samples
- batch size is small (`2`) because your GPU has only `4 GB`
- stage-2 model is heavier than a tiny CNN
- each epoch must run through `894` training batches plus `224` validation batches
- Windows + laptop GPU + dataloader overhead also add some delay

So around 5 minutes is not surprising, especially if you are training for a full epoch with validation.

## How To Run

### Train

```powershell
cd "D:\I FILES\Studies\sem6\MINI\module 3"
py -3.10 train.py
```

### Predict

Run this only **after training has saved a real checkpoint**:

```powershell
cd "D:\I FILES\Studies\sem6\MINI\module 3"
py -3.10 predict.py "..\Dataset\Cracks-and-Potholes-in-Road-Images\v1\1007599_RS_386_386RS289112_28920\1007599_RS_386_386RS289112_28920_RAW.jpg"
```

## Final Output Of Module 3

Prediction results are saved in:

- `D:\I FILES\Studies\sem6\MINI\module 3\predictions`

For each predicted image you now get:

- `<name>_corrected.png`
  Module-2 style corrected image

- `<name>_bitmask.png`
  Predicted binary damage mask
