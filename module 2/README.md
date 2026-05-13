# Module 2 Guide

## What This Module Does

Module 2 is **Stage 1: image preprocessing / geometric correction**.

It takes the original oblique road image and converts it into a corrected top-down style image using the perspective-transformation equations taken from the paper.

Input:
- Original dataset in `Dataset/Cracks-and-Potholes-in-Road-Images/v1`

Output:
- Corrected dataset in `Dataset/Stage1_Corrected`

## Files And Purpose

- `config.py`
  Stores module-2 settings like dataset path, output path, camera height, and camera angle.

- `preprocessing.py`
  Contains the full stage-1 logic:
  - scale coefficient equation
  - field-of-view equation
  - PC1 / PC2 solving
  - homography generation
  - perspective correction
  - batch correction for the whole dataset

- `stage1_preprocess.py`
  Runs the full stage-1 correction pipeline on the dataset.

- `dataset.py`
  Dataset loader used for training-ready loading and mask combination.

- `train_loader.py`
  Creates a PyTorch `DataLoader`.

- `main.py`
  Quick check script for loading samples and visualizing masks.

- `requirements.txt`
  Python dependencies.

## Flow

1. Read each sample folder.
2. Load:
   - `*_RAW.jpg`
   - `*_CRACK.png`
   - `*_POTHOLE.png`
   - ignore `*_LANE.png`
3. Apply perspective correction using the paper equations.
4. Save corrected image and masks into `Dataset/Stage1_Corrected`.
5. Preserve alignment between corrected image and corrected masks.

## How To Run

From project root:

```powershell
cd "D:\I FILES\Studies\sem6\MINI\module 2"
py -3.10 stage1_preprocess.py
```

## Final Output Of Module 2

Module 2 final output is the corrected image dataset:

- `D:\I FILES\Studies\sem6\MINI\Dataset\Stage1_Corrected`

That corrected dataset is the input to Module 3.
