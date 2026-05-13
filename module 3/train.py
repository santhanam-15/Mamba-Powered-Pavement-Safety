import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    LEARNING_RATE,
    LOG_INTERVAL,
    NUM_WORKERS,
    NUM_EPOCHS,
    SEED,
    TRAIN_SPLIT,
    WEIGHT_DECAY,
)
from dataset import CorrectedRoadDamageDataset
from metrics import dice_loss, iou_score
from model import CMSegNet

SEG_POS_WEIGHT = 1.0
THRESHOLD_CANDIDATES = tuple(round(x, 2) for x in torch.arange(0.3, 0.71, 0.05).tolist())


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders() -> tuple[DataLoader, DataLoader]:
    full_dataset = CorrectedRoadDamageDataset()
    train_dataset = CorrectedRoadDamageDataset(augment=False)
    val_dataset = CorrectedRoadDamageDataset(augment=False)
    train_size = int(len(full_dataset) * TRAIN_SPLIT)
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)
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


def generate_edge_targets(masks: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=masks.device,
        dtype=masks.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=masks.device,
        dtype=masks.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(masks, sobel_x, padding=1)
    grad_y = F.conv2d(masks, sobel_y, padding=1)
    edge_strength = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
    return (edge_strength > threshold).float()


def combined_loss(
    seg_logits: torch.Tensor,
    edge_pred: torch.Tensor,
    masks: torch.Tensor,
    edge_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=seg_logits.device.type, enabled=False):
        seg_logits_fp32 = seg_logits.float()
        edge_pred_fp32 = edge_pred.float()
        masks_fp32 = masks.float()
        edge_targets_fp32 = edge_targets.float()
        pos_weight = torch.tensor([SEG_POS_WEIGHT], device=seg_logits.device, dtype=seg_logits_fp32.dtype)

        seg_bce = F.binary_cross_entropy_with_logits(seg_logits_fp32, masks_fp32, pos_weight=pos_weight)
        seg_dice = dice_loss(seg_logits_fp32, masks_fp32)
        seg_loss = seg_bce + seg_dice
        edge_loss = F.binary_cross_entropy(edge_pred_fp32, edge_targets_fp32)
    total_loss = seg_loss + 0.4 * edge_loss
    return total_loss, seg_loss, edge_loss, edge_targets


def evaluate(model: CMSegNet, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    logits_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            seg_logits, edge_pred = model(images)
            edge_targets = generate_edge_targets(masks)
            loss, _, _, _ = combined_loss(seg_logits, edge_pred, masks, edge_targets)
            total_loss += loss.item()
            logits_batches.append(seg_logits.detach().cpu())
            mask_batches.append(masks.detach().cpu())
    model.train()
    if not logits_batches:
        return 0.0, 0.0, 0.5

    all_logits = torch.cat(logits_batches, dim=0)
    all_masks = torch.cat(mask_batches, dim=0)
    best_threshold = 0.5
    best_iou = -1.0
    for threshold in THRESHOLD_CANDIDATES:
        score = iou_score(all_logits, all_masks, threshold=threshold)
        if score > best_iou:
            best_iou = score
            best_threshold = threshold
    return total_loss / max(len(loader), 1), best_iou, best_threshold


def main() -> None:
    set_seed(SEED)
    train_loader, val_loader = build_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMSegNet().to(device)
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
    best_threshold = 0.5
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            edge_targets = generate_edge_targets(masks)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                seg_logits, edge_pred = model(images)
                loss, seg_loss, edge_loss, _ = combined_loss(seg_logits, edge_pred, masks, edge_targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % LOG_INTERVAL == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS} "
                    f"- batch {step}/{len(train_loader)} "
                    f"- loss: {loss.item():.4f} "
                    f"- seg_loss: {seg_loss.item():.4f} "
                    f"- edge_loss: {edge_loss.item():.4f}"
                )

        train_loss = epoch_loss / max(len(train_loader), 1)
        val_loss, val_iou, val_threshold = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_iou: {val_iou:.4f} "
            f"- best_thr: {val_threshold:.2f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_threshold = val_threshold
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_iou": best_val_iou,
                    "best_threshold": best_threshold,
                    "trained": True,
                    "img_size": 256,
                },
                CHECKPOINT_PATH,
            )
            print(f"Saved best checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
