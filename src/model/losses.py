"""Masked binary objectives for water segmentation."""

from __future__ import annotations

from typing import Any, Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F


class FloodLoss(nn.Module):
    """Weighted BCE plus Dice, with optional focal loss and exact ignore masking.

    The runner supplies Step-10 TRAIN class_balance in stats. Normalization-only
    stats do not contain class counts; without either weight or balance, weight=1.
    Dice/focal reductions include valid pixels only, and all-ignore batches yield
    a differentiable zero. Components are unweighted; total applies their weights.
    """

    def __init__(
        self, pos_weight: float | None = None, *, stats: Mapping[str, Any] | None = None,
        bce_weight: float = 1.0, dice_weight: float = 1.0, focal: bool = False,
        focal_weight: float = 1.0, focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if pos_weight is None:
            stats = stats or {}
            pos_weight = stats.get("pos_weight", (stats.get("class_balance") or {}).get("pos_weight", 1.0))
        if pos_weight is None or not math.isfinite(pos_weight) or pos_weight <= 0:
            raise ValueError("pos_weight must be finite and positive; TRAIN needs both classes")
        if any(not math.isfinite(x) or x < 0 for x in (bce_weight, dice_weight, focal_weight, focal_gamma)):
            raise ValueError("Loss weights and focal_gamma must be finite and nonnegative")
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32))
        self.bce_weight, self.dice_weight = bce_weight, dice_weight
        self.focal, self.focal_weight, self.focal_gamma = focal, focal_weight, focal_gamma

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits.ndim == label.ndim + 1 and logits.shape[1] == 1:
            logits = logits[:, 0]
        if logits.shape != label.shape:
            raise ValueError("Logits and labels must have matching spatial/batch shapes")
        valid = label != 255
        x = logits[valid].float()
        y = label[valid].float()
        if not x.numel():
            zero = x.sum()
            return {"total": zero, "bce": zero, "dice": zero, "focal": zero}
        if not torch.all((y == 0) | (y == 1)):
            raise ValueError("Labels must be 0, 1, or 255")
        bce = F.binary_cross_entropy_with_logits(x, y, pos_weight=self.pos_weight, reduction="mean")
        probability = torch.sigmoid(x)
        dice = 1 - (2 * (probability * y).sum() + 1e-6) / (probability.sum() + y.sum() + 1e-6)
        focal = x.sum() * 0
        if self.focal:
            pt = torch.where(y == 1, probability, 1 - probability)
            terms = F.binary_cross_entropy_with_logits(x, y, pos_weight=self.pos_weight, reduction="none")
            focal = ((1 - pt) ** self.focal_gamma * terms).mean()
        total = self.bce_weight * bce + self.dice_weight * dice + self.focal_weight * focal
        return {"total": total, "bce": bce, "dice": dice, "focal": focal}
