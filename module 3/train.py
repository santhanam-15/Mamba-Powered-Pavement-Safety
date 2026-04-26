import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    LEARNING_RATE,
    LOG_INTERVAL,
    NUM_CLASSES,
    NUM_WORKERS,
    NUM_EPOCHS,
    SEED,
    TRAIN_SPLIT,
    WEIGHT_DECAY,
)
from dataset import CorrectedRoadDamageDataset
from metrics import iou_score
from model import CMSegNet


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders() -> tuple[DataLoader, DataLoader]:
    dataset = CorrectedRoadDamageDataset()
    train_size = int(len(dataset) * TRAIN_SPLIT)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def evaluate(
    model: CMSegNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            total_loss += criterion(logits, masks).item()
            total_iou += iou_score(logits, masks)
    model.train()
    return total_loss / max(len(loader), 1), total_iou / max(len(loader), 1)


def main() -> None:
    set_seed(SEED)
    train_loader, val_loader = build_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMSegNet(out_channels=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    best_val_iou = -1.0
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % LOG_INTERVAL == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS} "
                    f"- batch {step}/{len(train_loader)} "
                    f"- loss: {loss.item():.4f}"
                )

        train_loss = epoch_loss / max(len(train_loader), 1)
        val_loss, val_iou = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_iou: {val_iou:.4f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_iou": best_val_iou,
                    "trained": True,
                    "img_size": 256,
                    "num_classes": NUM_CLASSES,
                },
                CHECKPOINT_PATH,
            )
            print(f"Saved best checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
