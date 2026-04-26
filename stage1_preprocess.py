from pathlib import Path

from config import (
    DATASET_PATH,
    DEFAULT_CAMERA_ANGLE_DEG,
    DEFAULT_CAMERA_HEIGHT_MM,
    STAGE1_OUTPUT_PATH,
)
from preprocessing import run_stage1_correction


def main() -> None:
    input_root = Path(DATASET_PATH)
    output_root = Path(STAGE1_OUTPUT_PATH)

    corrected_root = run_stage1_correction(
        input_root=input_root,
        output_root=output_root,
        camera_height_mm=DEFAULT_CAMERA_HEIGHT_MM,
        camera_angle_deg=DEFAULT_CAMERA_ANGLE_DEG,
    )

    print(f"Stage-1 correction complete: {corrected_root}")
    print(f"Camera height (mm): {DEFAULT_CAMERA_HEIGHT_MM}")
    print(f"Camera angle (deg): {DEFAULT_CAMERA_ANGLE_DEG}")


if __name__ == "__main__":
    main()
