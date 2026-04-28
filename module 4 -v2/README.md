# Module 4 Guide (Flow-First Explanation)

This README explains the full flow of Module 4 clearly so you can directly convert it into a flow diagram.

## 1. What Module 4 Does

Module 4 is an end-to-end pothole segmentation pipeline that:

1. Takes one raw road image.
2. Applies stage-1 geometric correction (perspective correction).
3. Runs a segmentation model (CMSegNet) to detect pothole regions.
4. Uses a shadow suppression model (ShadowNet) to reduce false positives from shadows.
5. Post-processes the mask and saves all outputs.
6. Calculates area information and saves it as text + image summary.

Main entry point for inference: `run_module4.py`

## 2. Key Files and Their Role

- `run_module4.py`
  - CLI wrapper.
  - Validates input image + checkpoint.
  - Calls `predict_image(...)` from `predict.py`.

- `predict.py`
  - Core inference pipeline.
  - Loads model checkpoint.
  - Preprocesses image.
  - Runs CMSegNet + ShadowNet.
  - Builds final bitmask.
  - Creates overlay.
  - Computes and saves area summary.

- `stage1_utils.py`
  - Camera geometry and homography logic.
  - Corrects perspective distortion.
  - Returns corrected image + metadata (including `r0_mm_per_px`).

- `model.py`
  - Defines:
    - `CMSegNet` (main segmentation network)
    - `ShadowNet` (shadow probability map)
  - CMSegNet internals:
    - `CMEncoder`
    - `CNNBranch` (local detail)
    - `OptimizedMambaBlock` (global context)
    - `MSAAModule` (multi-scale attention fusion)
    - `LightweightMLPDecoder` (segmentation output)
    - `EdgeDetectionHead` (auxiliary boundary prediction)

- `dataset.py`
  - Training dataset loader using corrected samples.
  - Merges crack and pothole masks into one binary target.

- `train.py`
  - Training loop.
  - Combined segmentation + edge loss.
  - Validation IoU and threshold search.
  - Saves best checkpoint.

- `metrics.py`
  - Dice loss and IoU helper functions.

- `segmentation_eval_metrics.py`
  - Standalone TP/TN/FP/FN + Pixel Accuracy + IoU + F1 evaluator.
  - Can save a metrics image summary.

## 3. Inference Flow (Exact Runtime Order)

This is the flow when you run `run_module4.py`.

### Step 0: Command entry

User runs:

```powershell
py run_module4.py <image_path> --checkpoint cmsegnet_stage2.pt
```

### Step 1: Input validation (`run_module4.py`)

1. Parse CLI arguments.
2. Check if `image_path` exists.
3. Check if checkpoint exists.
4. Resolve output directory:
   - default = `predictions/<image_stem>`

### Step 2: Model load (`predict.py` -> `load_model`)

1. Select device (`cuda` if available, else `cpu`).
2. Create `CMSegNet`.
3. Load checkpoint.
4. Read `best_threshold` from checkpoint (default 0.5).
5. Set model to eval mode.

### Step 3: Stage-1 correction + preprocessing (`predict.py` -> `preprocess_image`)

1. Read original BGR image.
2. Call `correct_stage1_image(...)` from `stage1_utils.py`.
3. Convert corrected BGR -> RGB.
4. Resize to `IMG_SIZE x IMG_SIZE` (256x256).
5. Normalize to [0,1], convert to tensor `[1,3,H,W]`.

### Step 4: Dual-model forward pass (`predict.py` -> `forward_module4`)

1. CMSegNet outputs pothole logits.
2. Apply sigmoid -> pothole probability mask.
3. ShadowNet outputs shadow probability mask.
4. Compute shadow confidence and suppress pothole scores in high-shadow regions.
5. Build `final_mask`.

### Step 5: Safety fallback for untrained ShadowNet

If ShadowNet output is near-constant around 0.5 (typical untrained behavior), pipeline falls back to plain CMSegNet mask to avoid over-suppression.

### Step 6: Bitmask generation + post-processing

1. Threshold probability mask using `best_threshold`.
2. Convert to binary 0/255 mask.
3. Resize back to corrected image size.
4. Run connected-component filtering (`postprocess_bitmask`) to remove tiny/noisy regions while keeping meaningful elongated damage regions.

### Step 7: Visualization and save

1. Create transparent red overlay on corrected image (`create_transparent_red_overlay`).
2. Save:
   - `original.png`
   - `corrected.png`
   - `predicted_bitmask.png`
   - `overlay.png`

### Step 8: Area estimation (`save_area_summary`)

1. Count white pixels in final bitmask.
2. Use `r0_mm_per_px` from stage-1 metadata.
3. Compute:
   - `area_mm2 = white_pixels * r`
   - `area_cm2 = area_mm2 / 100`
4. Save:
   - `area_summary.txt`
   - `area_summary.png`

## 4. Diagram-Ready Flow Blocks

Use these nodes directly in your flow diagram:

1. Start
2. Read CLI Args
3. Validate Input Image
4. Validate Checkpoint
5. Load CMSegNet + Threshold
6. Read Raw Image
7. Stage-1 Perspective Correction
8. Resize + Normalize + Tensor Convert
9. CMSegNet Forward (Pothole Probability)
10. ShadowNet Forward (Shadow Probability)
11. Shadow Suppression Fusion
12. ShadowNet Reliability Check (Fallback if unstable)
13. Threshold to Binary Mask
14. Morphology + Connected Component Filtering
15. Create Overlay
16. Save Images
17. Compute White Pixel Area
18. Save Area Summary (TXT + PNG)
19. End

## 5. Training Flow (How Checkpoint is Produced)

1. `dataset.py` scans corrected dataset (`RAW`, `CRACK`, `POTHOLE` triplets).
2. Crack + pothole masks are merged into one binary target.
3. `train.py` builds train/validation splits.
4. Model outputs segmentation logits + edge map.
5. Loss = segmentation BCE + Dice + weighted edge BCE.
6. Validation checks IoU across threshold candidates.
7. Best model saved to `cmsegnet_stage2.pt` with:
   - `model_state_dict`
   - `best_val_iou`
   - `best_threshold`
   - `trained=True`

## 6. Outputs You Should Expect Per Image

Inside `predictions/<image_stem>/`:

1. `original.png`
2. `corrected.png`
3. `predicted_bitmask.png`
4. `overlay.png`
5. `area_summary.txt`
6. `area_summary.png`

## 7. Quick Run Commands

### Inference

```powershell
cd "D:\I FILES\Studies\sem6\MINI\module 4"
py run_module4.py "..\Dataset\Cracks-and-Potholes-in-Road-Images\v1\<sample_folder>\<sample_RAW.jpg>"
```

### Training

```powershell
cd "D:\I FILES\Studies\sem6\MINI\module 4"
py train.py
```

## 8. Notes for Presentation/Flow Diagram

1. Keep two highlighted branches in the diagram:
   - Pothole branch (CMSegNet)
   - Shadow branch (ShadowNet)
2. Show where both branches merge (shadow suppression fusion).
3. Show post-processing as a separate block after thresholding.
4. Show area computation as final analytics block after final bitmask generation.
