from torch.utils.data import DataLoader

from config import BATCH_SIZE, DATASET_PATH
from dataset import RoadDamageDataset


def create_train_loader(
    dataset_path: str = DATASET_PATH,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    apply_perspective: bool = False,
) -> DataLoader:
    """Create the training DataLoader."""
    dataset = RoadDamageDataset(
        dataset_path=dataset_path,
        apply_perspective=apply_perspective,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
