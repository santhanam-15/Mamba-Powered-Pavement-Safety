import torch


def iou_score(logits: torch.Tensor, targets: torch.Tensor, target_class: int = 1, eps: float = 1e-6) -> float:
    preds = torch.argmax(logits, dim=1)
    preds = (preds == target_class).float()
    targets = (targets == target_class).float()
    intersection = (preds * targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - intersection
    score = (intersection + eps) / (union + eps)
    return float(score.mean().item())
