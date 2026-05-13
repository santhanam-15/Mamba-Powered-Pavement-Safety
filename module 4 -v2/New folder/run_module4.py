from __future__ import annotations

import argparse
from pathlib import Path

from config import CHECKPOINT_PATH, PREDICTION_DIR
from predict import predict_image


def run(image_path: Path, checkpoint_path: Path, output_dir: Path) -> None:
    original_path, corrected_path, bitmask_path, overlay_path = predict_image(
        str(image_path),
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )
    print("Saved outputs:")
    print(f"- {original_path}")
    print(f"- {corrected_path}")
    print(f"- {bitmask_path}")
    print(f"- {overlay_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run module4 dual-model inference (CMSegNet + ShadowNet).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("image_path", type=Path, nargs="?", help="Path to input road image")
    group.add_argument("-f", "--folder", type=Path, help="Folder containing images to process in batch mode")
    parser.add_argument("--checkpoint", type=Path, default=Path(CHECKPOINT_PATH), help="CMSegNet checkpoint path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override output directory (default: predictions/<photo_name> or predictions/<image_name> for batch)",
    )
    args = parser.parse_args()

    if args.folder:
        if not args.folder.exists() or not args.folder.is_dir():
            raise FileNotFoundError(f"Input folder not found: {args.folder}")
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        image_files = sorted([p for p in args.folder.iterdir() if p.suffix.lower() in exts and p.is_file()])
        if not image_files:
            raise FileNotFoundError(f"No image files found in folder: {args.folder}")
        print(f"Found {len(image_files)} images in {args.folder}. Starting batch inference...")
        for img_path in image_files:
            print(f"\nProcessing: {img_path.name}")
            resolved_output_dir = args.output_dir if args.output_dir is not None else Path(PREDICTION_DIR) / img_path.stem
            try:
                run(img_path, args.checkpoint, resolved_output_dir)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        print("\nBatch inference complete.")
    else:
        if not args.image_path or not args.image_path.exists():
            raise FileNotFoundError(f"Input image not found: {args.image_path}")
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        resolved_output_dir = args.output_dir if args.output_dir is not None else Path(PREDICTION_DIR) / args.image_path.stem
        run(args.image_path, args.checkpoint, resolved_output_dir)
