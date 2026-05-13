import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import logging
import csv
import os
from datetime import datetime

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

SEG_POS_WEIGHT = 2.5
THRESHOLD_CANDIDATES = tuple(round(x, 2) for x in torch.arange(0.3, 0.71, 0.05).tolist())
TRAIN_SPLIT_OVERRIDE = 0.90
NUM_EPOCHS_OVERRIDE = 50
MAX_GRAD_NORM = 1.0
WARMUP_EPOCHS = 5
INIT_LR = 1e-5


def setup_logging(log_file: str = "training.log") -> logging.Logger:
    """Configure logging to write to both console and file."""
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def setup_metrics_csv(csv_file: str = "metrics.csv") -> str:
    """Initialize CSV file for metrics tracking."""
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val IoU', 'Best Threshold', 'Learning Rate', 'Checkpoint Saved'])
    
    return csv_file


def log_metrics_to_csv(csv_file: str, epoch: int, train_loss: float, val_loss: float, 
                       val_iou: float, val_threshold: float, lr: float, checkpoint_saved: bool) -> None:
    """Log epoch metrics to CSV file."""
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_iou:.6f}", 
                        f"{val_threshold:.2f}", f"{lr:.2e}", "Yes" if checkpoint_saved else "No"])


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders() -> tuple[DataLoader, DataLoader]:
    full_dataset = CorrectedRoadDamageDataset()
    train_dataset = CorrectedRoadDamageDataset(augment=True)
    val_dataset = CorrectedRoadDamageDataset(augment=False)
    dataset_size = len(full_dataset)
    if dataset_size <= 1:
        train_size = dataset_size
    else:
        train_size = int(dataset_size * TRAIN_SPLIT_OVERRIDE)
        train_size = max(1, min(train_size, dataset_size - 1))
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
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
    logger = setup_logging("training.log")
    metrics_csv = setup_metrics_csv("metrics.csv")
    set_seed(SEED)
    train_loader, val_loader = build_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMSegNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    # Warmup scheduler: linearly increase LR from INIT_LR to LEARNING_RATE
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=INIT_LR/LEARNING_RATE, total_iters=WARMUP_EPOCHS
    )
    
    # Learning rate scheduler: reduce LR when val_iou plateaus (after warmup)
    reduce_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    logger.info(f"Train split override: {TRAIN_SPLIT_OVERRIDE:.2f} ({train_size} samples)")
    logger.info(f"Val split: {1.0 - TRAIN_SPLIT_OVERRIDE:.2f} ({val_size} samples)")
    logger.info(f"Epoch override: {NUM_EPOCHS_OVERRIDE}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Max gradient norm (clipping): {MAX_GRAD_NORM}")
    logger.info(f"Pos weight (class imbalance): {SEG_POS_WEIGHT}")
    logger.info(f"Learning rate scheduler: LinearLR warmup ({WARMUP_EPOCHS} epochs) + ReduceLROnPlateau")
    logger.info(f"Initial LR (warmup): {INIT_LR:.2e}, Target LR: {LEARNING_RATE:.2e}")
    logger.info(f"Metrics CSV file: metrics.csv")

    best_val_iou = -1.0
    best_threshold = 0.5
    patience = 7
    epochs_no_improve = 0
    for epoch in range(NUM_EPOCHS_OVERRIDE):
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
            
            # Gradient clipping for stability (GPU memory optimization)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % LOG_INTERVAL == 0 or step == len(train_loader):
                log_msg = (
                    f"Epoch {epoch + 1}/{NUM_EPOCHS_OVERRIDE} "
                    f"- batch {step}/{len(train_loader)} "
                    f"- loss: {loss.item():.4f} "
                    f"- seg_loss: {seg_loss.item():.4f} "
                    f"- edge_loss: {edge_loss.item():.4f}"
                )
                logger.info(log_msg)

        train_loss = epoch_loss / max(len(train_loader), 1)
        val_loss, val_iou, val_threshold = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']

        log_msg = (
            f"Epoch {epoch + 1}/{NUM_EPOCHS_OVERRIDE} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_iou: {val_iou:.4f} "
            f"- best_thr: {val_threshold:.2f} "
            f"- lr: {current_lr:.2e}"
        )
        logger.info(log_msg)

        checkpoint_saved = False
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_threshold = val_threshold
            epochs_no_improve = 0
            checkpoint_saved = True

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
            logger.info(f"Saved best checkpoint to {CHECKPOINT_PATH}")

        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epochs")

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                logger.info(f"Training completed. Best val_iou: {best_val_iou:.4f} at threshold: {best_threshold:.2f}")
                log_metrics_to_csv(metrics_csv, epoch + 1, train_loss, val_loss, val_iou, val_threshold, current_lr, checkpoint_saved)
                break
        
        # Update learning rate scheduler based on validation IoU
        # Use warmup scheduler for first WARMUP_EPOCHS, then switch to ReduceLROnPlateau
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
            logger.info(f"Warmup phase: epoch {epoch + 1}/{WARMUP_EPOCHS}")
        else:
            reduce_lr_scheduler.step(val_iou)
        
        # Log metrics to CSV
        log_metrics_to_csv(metrics_csv, epoch + 1, train_loss, val_loss, val_iou, val_threshold, current_lr, checkpoint_saved)


if __name__ == "__main__":
    logger = setup_logging("training.log")
    logger.info("="*80)
    logger.info("Starting training script with optimizations for RTX 3050 (4GB)")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    main()
    logger.info("="*80)
    logger.info("Training complete")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("CSV Metrics saved to: metrics.csv")
    logger.info("="*80)
