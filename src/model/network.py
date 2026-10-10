"""Water-logit networks with explicit, portable architecture metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
from typing import Any
import warnings

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    """Persist the actual backend and fallback dimensions, not just a model name."""

    arch: str = "unet"
    encoder: str = "resnet34"
    in_channels: int = 2
    channels: list[str] = field(default_factory=lambda: ["post_vv", "post_vh"])
    encoder_weights: str | None = None
    classes: int = 1
    backend: str = "auto"
    depth: int = 3
    width: int = 16

    def __post_init__(self) -> None:
        if self.in_channels != len(self.channels) or len(set(self.channels)) != len(self.channels):
            raise ValueError("in_channels must match the unique ordered channel list")
        if self.classes != 1:
            raise ValueError("Binary water segmentation requires classes=1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelConfig:
        return cls(**value)


class CompactUNet(nn.Module):
    """Pure-torch U-Net; group normalization also works with tiny batches."""

    def __init__(self, in_channels: int, classes: int = 1, depth: int = 3, width: int = 16) -> None:
        super().__init__()
        if in_channels < 1 or classes != 1 or not 1 <= depth <= 5 or width < 2:
            raise ValueError("Require positive channels, classes=1, depth 1..5, width>=2")

        def block(inputs: int, outputs: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                nn.GroupNorm(1, outputs), nn.ReLU(inplace=True),
                nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                nn.GroupNorm(1, outputs), nn.ReLU(inplace=True),
            )

        widths = [width * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([block(in_channels, widths[0])]
                                 + [block(a, b) for a, b in zip(widths, widths[1:])])
        self.up = nn.ModuleList([block(a + b, b) for a, b in zip(widths[:0:-1], widths[-2::-1])])
        self.head = nn.Conv2d(width, classes, 1)
        self.depth = depth

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if min(image.shape[-2:]) < 2 ** self.depth:
            raise ValueError(f"Input sides must be at least {2 ** self.depth} pixels")
        skips = []
        for index, block in enumerate(self.down):
            if index:
                image = F.max_pool2d(image, 2)
            image = block(image)
            skips.append(image)
        for block, skip in zip(self.up, reversed(skips[:-1])):
            image = F.interpolate(image, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            image = block(torch.cat((image, skip), dim=1))
        return self.head(image)


def _import_smp() -> Any:
    return importlib.import_module("segmentation_models_pytorch")


def build_model(
    arch: str = "unet", encoder: str = "resnet34", in_channels: int = 2,
    classes: int = 1, encoder_weights: str | None = None, *,
    backend: str = "auto", depth: int = 3, width: int = 16,
) -> nn.Module:
    """Build logits only; pretrained downloads are opt-in, never used on checkpoint load."""
    if classes != 1 or in_channels < 1 or encoder_weights not in (None, "imagenet"):
        raise ValueError("Require classes=1, positive in_channels, weights None or imagenet")
    if backend not in ("auto", "smp", "fallback"):
        raise ValueError("backend must be auto, smp, or fallback")
    reason = None
    if backend != "fallback":
        try:
            smp = _import_smp()
        except Exception as error:
            if backend == "smp":
                raise RuntimeError("Checkpoint requires segmentation-models-pytorch; "
                                   "install it with compatible torch/torchvision wheels") from error
            reason = f"SMP import failed ({type(error).__name__}: {error}); using compact U-Net"
        else:
            constructors = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus,
                            "fpn": smp.FPN, "deeplabv3plus": smp.DeepLabV3Plus}
            if arch.lower() not in constructors:
                raise ValueError(f"Unsupported architecture: {arch}")
            model = constructors[arch.lower()](encoder_name=encoder, in_channels=in_channels,
                                               classes=classes, encoder_weights=encoder_weights,
                                               activation=None)
            model.backend = "smp"
            model.backend_warning = None
            return model
    if arch.lower() != "unet":
        raise ValueError("The pure-torch fallback supports arch=unet only")
    if encoder_weights is not None:
        reason = (reason or "Compact U-Net requested") + "; ImageNet weights unavailable for fallback"
    model = CompactUNet(in_channels, classes, depth, width)
    model.backend = "fallback"
    model.backend_warning = reason
    if reason:
        warnings.warn(reason, RuntimeWarning, stacklevel=2)
    return model
