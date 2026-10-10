"""Report a checkpoint on one official split using the shared validation path."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from src.model.data import FloodChipDataset, Sen1Floods11Index
from src.model.losses import FloodLoss
from src.model.metrics import DATA_CAVEAT
from src.model.predict import load_checkpoint, select_device
from src.model.train import validate


def evaluate_checkpoint(
    checkpoint: str | Path, data_root: str | Path, split: str = "test", *,
    device: str | None = None, batch_size: int = 2, num_workers: int = 0,
    output: str | Path | None = None,
) -> dict[str, Any]:
    if split not in ("val", "test", "bolivia") or batch_size < 1 or num_workers < 0:
        raise ValueError("Choose val/test/bolivia, positive batch size, nonnegative workers")
    device = select_device(device)
    model, meta = load_checkpoint(checkpoint, device)
    source = Sen1Floods11Index(data_root)
    native_channels = [name for name in meta["channels"] if not name.endswith("_ratio")]
    dataset = FloodChipDataset(source=source, split=split, channels=native_channels,
                               stats=meta["normalization_stats"], augment=False,
                               add_ratio=any(name.endswith("_ratio") for name in meta["channels"]))
    if dataset.channels != meta["channels"]:
        raise ValueError("Checkpoint channel order cannot be reproduced by the dataset adapter")
    if not len(dataset):
        raise ValueError(f"Official {split} split is empty")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    metrics = validate(model, loader, device, threshold=meta["best_threshold"],
                       loss_fn=FloodLoss(pos_weight=meta["pos_weight"]).to(device))
    report = {"split": split, "checkpoint_path": meta["checkpoint_path"],
              "threshold_selection_split": "val", "data_caveat": DATA_CAVEAT,
              "split_size": len(dataset), "metrics": metrics,
              "warnings": list(dict.fromkeys([*meta.get("warnings", []), *source.warnings]))}
    path = Path(output) if output else Path(checkpoint).parent / f"metrics_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/external/sen1floods11")
    parser.add_argument("--split", choices=("val", "test", "bolivia"), default="test")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output")
    report = evaluate_checkpoint(**vars(parser.parse_args()))
    print(json.dumps(report, indent=2, allow_nan=False))
    print("Region                 IoU         F1      Valid pixels")
    for region, scores in report["metrics"]["per_region"].items():
        iou = f"{scores['iou']:.4f}" if scores["iou"] is not None else "n/a"
        f1 = f"{scores['f1']:.4f}" if scores["f1"] is not None else "n/a"
        print(f"{region:20} {iou:>8} {f1:>10} {scores['valid_pixels']:>15}")


if __name__ == "__main__":
    main()
