import matplotlib.pyplot as plt

from config import DATASET_PATH
from dataset import RoadDamageDataset
from train_loader import create_train_loader


def main() -> None:
    dataset = RoadDamageDataset(dataset_path=DATASET_PATH)
    train_loader = create_train_loader(dataset_path=DATASET_PATH)

    print(f"Total samples: {len(dataset)}")

    first_item = dataset[0]
    print(f"Image tensor shape: {tuple(first_item['image'].shape)}")
    print(f"Mask tensor shape: {tuple(first_item['mask'].shape)}")

    batch = next(iter(train_loader))
    print(f"Batch image shape: {tuple(batch['image'].shape)}")
    print(f"Batch mask shape: {tuple(batch['mask'].shape)}")

    sample = dataset.get_visualization_sample(0)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(sample["image"])
    axes[0].set_title("Original Image")
    axes[1].imshow(sample["crack_mask"], cmap="gray")
    axes[1].set_title("Crack Mask")
    axes[2].imshow(sample["pothole_mask"], cmap="gray")
    axes[2].set_title("Pothole Mask")
    axes[3].imshow(sample["mask"], cmap="gray")
    axes[3].set_title("Combined Mask")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
