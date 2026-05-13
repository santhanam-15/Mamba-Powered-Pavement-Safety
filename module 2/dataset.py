from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import DATASET_PATH, IMG_SIZE
from preprocessing import (
    binarize_mask,
    combine_masks,
    get_perspective_matrix,
    normalize,
    perspective_transform,
    resize,
)


class RoadDamageDataset(Dataset):
    """
    Dataset for pothole + crack segmentation.

    Each sample is built from:
    - *_RAW.jpg
    - *_CRACK.png
    - *_POTHOLE.png

    Any *_LANE.png file is ignored.
    """

    def __init__(
        self,
        dataset_path: str = DATASET_PATH,
        img_size: int = IMG_SIZE,
        apply_perspective: bool = False,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.img_size = img_size
        self.apply_perspective = apply_perspective
        self.samples = self._build_samples()

        if not self.samples:
            raise FileNotFoundError(
                f"No valid samples found in '{self.dataset_path}'. "
                "Expected *_RAW.jpg, *_CRACK.png, and *_POTHOLE.png files."
            )

    def _build_samples(self) -> List[Dict[str, Path]]:
        grouped_files: Dict[str, Dict[str, Path]] = {}

        for file_path in self.dataset_path.rglob("*"):
            if not file_path.is_file():
                continue

            stem = file_path.stem
            if "_" not in stem:
                continue

            prefix, suffix = stem.rsplit("_", 1)
            suffix = suffix.upper()

            if suffix == "LANE":
                continue

            if suffix not in {"RAW", "CRACK", "POTHOLE"}:
                continue

            grouped_files.setdefault(prefix, {})[suffix] = file_path

        samples: List[Dict[str, Path]] = []
        for prefix, files in grouped_files.items():
            if {"RAW", "CRACK", "POTHOLE"}.issubset(files):
                samples.append(
                    {
                        "id": prefix,
                        "raw": files["RAW"],
                        "crack": files["CRACK"],
                        "pothole": files["POTHOLE"],
                    }
                )

        samples.sort(key=lambda sample: sample["id"])
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: Path) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_mask(self, mask_path: Path) -> np.ndarray:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        return binarize_mask(mask)

    def _prepare_sample(self, sample: Dict[str, Path]) -> Dict[str, np.ndarray]:
        image = self._load_image(sample["raw"])
        crack_mask = self._load_mask(sample["crack"])
        pothole_mask = self._load_mask(sample["pothole"])

        final_mask = combine_masks([crack_mask, pothole_mask])

        if self.apply_perspective:
            original_image = image.copy()
            matrix = get_perspective_matrix(image.shape[1], image.shape[0])
            image, crack_mask = perspective_transform(original_image, crack_mask, matrix=matrix)
            _, pothole_mask = perspective_transform(original_image, pothole_mask, matrix=matrix)
            _, final_mask = perspective_transform(original_image, final_mask, matrix=matrix)

        image, crack_mask = resize(image, crack_mask, self.img_size)
        _, pothole_mask = resize(image, pothole_mask, self.img_size)
        _, final_mask = resize(image, final_mask, self.img_size)

        crack_mask = binarize_mask(crack_mask)
        pothole_mask = binarize_mask(pothole_mask)
        final_mask = binarize_mask(final_mask)
        image = normalize(image)

        return {
            "image": image,
            "crack_mask": crack_mask,
            "pothole_mask": pothole_mask,
            "final_mask": final_mask,
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        prepared = self._prepare_sample(sample)

        image_tensor = torch.from_numpy(prepared["image"].transpose(2, 0, 1)).float()
        crack_tensor = torch.from_numpy(prepared["crack_mask"]).unsqueeze(0).float()
        pothole_tensor = torch.from_numpy(prepared["pothole_mask"]).unsqueeze(0).float()
        final_tensor = torch.from_numpy(prepared["final_mask"]).unsqueeze(0).float()

        return {
            "id": sample["id"],
            "image": image_tensor,
            "crack_mask": crack_tensor,
            "pothole_mask": pothole_tensor,
            "mask": final_tensor,
        }

    def get_visualization_sample(self, index: int = 0) -> Dict[str, np.ndarray]:
        """Return numpy arrays for plotting without tensor conversion."""
        sample = self.samples[index]
        prepared = self._prepare_sample(sample)
        return {
            "id": sample["id"],
            "image": prepared["image"],
            "crack_mask": prepared["crack_mask"],
            "pothole_mask": prepared["pothole_mask"],
            "mask": prepared["final_mask"],
        }
