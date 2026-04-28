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


class CorrectedRoadDamageDataset(Dataset):
    """Dataset for module 3 training on stage-1 corrected images."""

    def __init__(self, dataset_path: str = DATASET_PATH, img_size: int = IMG_SIZE, augment: bool = False) -> None:
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.augment = augment
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
            if suffix not in {"RAW", "CRACK", "POTHOLE"}:
                continue
            grouped.setdefault(prefix, {})[suffix] = file_path

        samples: List[Dict[str, Path]] = []
        for prefix, files in grouped.items():
            if {"RAW", "CRACK", "POTHOLE"}.issubset(files):
                samples.append(
                    {
                        "id": prefix,
                        "raw": files["RAW"],
                        "crack": files["CRACK"],
                        "pothole": files["POTHOLE"],
                    }
                )

        samples.sort(key=lambda item: item["id"])
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(np.flip(image, axis=1))
            mask = np.ascontiguousarray(np.flip(mask, axis=1))

        if np.random.rand() < 0.4:
            alpha = np.random.uniform(0.9, 1.1)
            beta = np.random.uniform(-0.05, 0.05)
            image = np.clip(image * alpha + beta, 0.0, 1.0)

        return image, mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]

        image = cv2.imread(str(sample["raw"]), cv2.IMREAD_COLOR)
        crack = cv2.imread(str(sample["crack"]), cv2.IMREAD_GRAYSCALE)
        pothole = cv2.imread(str(sample["pothole"]), cv2.IMREAD_GRAYSCALE)

        if image is None or crack is None or pothole is None:
            raise ValueError(f"Failed to read one or more files for sample '{sample['id']}'.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        crack = cv2.resize(binarize_mask(crack), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        pothole = cv2.resize(binarize_mask(pothole), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = np.logical_or(crack, pothole).astype(np.float32)

        image = image.astype(np.float32) / 255.0
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        return {
            "id": sample["id"],
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
        }
