from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import DATASET_PATH, IMG_SIZE


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    threshold = 0 if mask.max() <= 1 else 127
    return (mask > threshold).astype(np.uint8)


def encode_multiclass_mask(pothole_mask: np.ndarray, shadow_mask: np.ndarray) -> np.ndarray:
    """
    Encode a segmentation mask with:
    0 = background
    1 = pothole
    2 = shadow

    Pothole takes precedence when both masks overlap.
    """
    encoded_mask = np.zeros_like(pothole_mask, dtype=np.uint8)
    encoded_mask[shadow_mask > 0] = 2
    encoded_mask[pothole_mask > 0] = 1
    return encoded_mask


class CorrectedRoadDamageDataset(Dataset):
    """Dataset for module 3 training on stage-1 corrected images."""

    def __init__(self, dataset_path: str = DATASET_PATH, img_size: int = IMG_SIZE) -> None:
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.samples = self._scan_samples()

        if not self.samples:
            raise FileNotFoundError(f"No corrected samples found in '{self.dataset_path}'.")

    def _scan_samples(self) -> List[Dict[str, Path]]:
        grouped: Dict[str, Dict[str, Path]] = {}
        for file_path in self.dataset_path.rglob("*"):
            if not file_path.is_file():
                continue
            stem = file_path.stem
            if "_" not in stem:
                continue
            prefix, suffix = stem.rsplit("_", 1)
            suffix = suffix.upper()
            if suffix not in {"RAW", "CRACK", "POTHOLE", "SHADOW"}:
                continue
            grouped.setdefault(prefix, {})[suffix] = file_path

        samples: List[Dict[str, Path]] = []
        for prefix, files in grouped.items():
            shadow_path = files.get("SHADOW") or files.get("CRACK")
            if files.get("RAW") and files.get("POTHOLE") and shadow_path:
                samples.append(
                    {
                        "id": prefix,
                        "raw": files["RAW"],
                        "shadow": shadow_path,
                        "pothole": files["POTHOLE"],
                    }
                )

        samples.sort(key=lambda item: item["id"])
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]

        image = cv2.imread(str(sample["raw"]), cv2.IMREAD_COLOR)
        shadow = cv2.imread(str(sample["shadow"]), cv2.IMREAD_GRAYSCALE)
        pothole = cv2.imread(str(sample["pothole"]), cv2.IMREAD_GRAYSCALE)

        if image is None or shadow is None or pothole is None:
            raise ValueError(f"Failed to read one or more files for sample '{sample['id']}'.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        shadow = cv2.resize(binarize_mask(shadow), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        pothole = cv2.resize(binarize_mask(pothole), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = encode_multiclass_mask(pothole_mask=pothole, shadow_mask=shadow)

        image = image.astype(np.float32) / 255.0

        return {
            "id": sample["id"],
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(mask).long(),
        }
