"""Reproducible offline-data training; execution is always an explicit CLI action."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from dataclasses import replace
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.data import FloodChipDataset, class_balance, make_dataloaders
from src.model.losses import FloodLoss
from src.model.metrics import DATA_CAVEAT, MetricAccumulator
from src.model.network import ModelConfig, build_model
from src.model.predict import load_checkpoint, read_checkpoint, select_device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate(
    model: torch.nn.Module, loader: Any, device: str | torch.device = "cpu", *,
    threshold: float = .5, loss_fn: FloodLoss | None = None,
    max_batches: int | None = None, select_threshold: bool = False,
) -> dict[str, Any]:
    """Shared validation path; test/holdout use the stored VAL threshold only."""
    device = torch.device(device)
    model.eval()
    accumulator = MetricAccumulator(threshold, thresholds=None if select_threshold else [threshold])
    losses: dict[str, float] = {}
    count = 0
    batches = 0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            image = batch["image"].to(device, dtype=torch.float32)
            label = batch["label"].to(device)
            logits = model(image)
            accumulator.update(torch.sigmoid(logits.float()), label, batch.get("meta"))
            valid = int((label != 255).sum().item())
            if loss_fn is not None and valid:
                for name, value in loss_fn(logits, label).items():
                    losses[name] = losses.get(name, 0) + value.item() * valid
                count += valid
            batches += 1
    result = accumulator.compute()
    result["batches"] = batches
    result["loss"] = {name: value / count for name, value in losses.items()} if count else None
    return result


def train_epoch(
    model: torch.nn.Module, loader: Any, optimizer: torch.optim.Optimizer,
    loss_fn: FloodLoss, device: str | torch.device = "cpu", *,
    scheduler: Any = None, scaler: Any = None, grad_clip: float = 1.0,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Skip all-ignore batches completely, including AdamW weight decay and LR steps."""
    device = torch.device(device)
    model.train()
    totals: dict[str, float] = {}
    count, steps, skipped = 0, 0, 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        label = batch["label"].to(device)
        valid = int((label != 255).sum().item())
        if not valid:
            skipped += 1
            continue
        image = batch["image"].to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            components = loss_fn(model(image), label)
        if not torch.isfinite(components["total"]):
            raise ValueError("Training loss is nonfinite; no successful checkpoint can be declared")
        if scaler is not None and scaler.is_enabled():
            scaler.scale(components["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            stepped = scaler.get_scale() >= previous_scale
        else:
            components["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            stepped = True
        if scheduler is not None and stepped:
            scheduler.step()
        for name, value in components.items():
            totals[name] = totals.get(name, 0) + value.detach().item() * valid
        count += valid
        steps += int(stepped)
    if not count or not steps:
        raise ValueError("No usable training steps: batches were ignored or optimizer steps failed")
    return {"loss": {name: value / count for name, value in totals.items()},
            "optimizer_steps": steps, "ignored_batches": skipped, "valid_pixels": count}


def _rng_state(loaders: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    state = np.random.get_state()
    result = {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "datasets": {split: loader.dataset.rng.bit_generator.state
                     for split, loader in loaders.items() if hasattr(loader.dataset, "rng")},
        "loaders": {split: loader.generator.get_state()
                    for split, loader in loaders.items() if loader.generator is not None},
    }
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        result["mps"] = torch.mps.get_rng_state()
    return result


def _restore_rng(state: dict[str, Any], loaders: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values[0], np.array(values[1], dtype=np.uint32), *values[2:]))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    for split, value in state["datasets"].items():
        if split in loaders and hasattr(loaders[split].dataset, "rng"):
            loaders[split].dataset.rng.bit_generator.state = value
    for split, value in state["loaders"].items():
        if split in loaders and loaders[split].generator is not None:
            loaders[split].generator.set_state(value)


def _versions() -> dict[str, str | None]:
    result = {}
    for name in ("torch", "torchvision", "segmentation-models-pytorch", "numpy", "rasterio"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                              capture_output=True, text=True, check=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_training(
    loaders: Mapping[str, Any], out_dir: str | Path, config: ModelConfig,
    stats: dict[str, Any], training_args: dict[str, Any] | None = None,
    *, model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Explicit runner, also exercised only with synthetic inputs in unit tests.

    Selection uses VAL IoU at 0.5; the chosen checkpoint's threshold maximizes
    VAL F1. That one threshold is then fixed for val/test/Bolivia reporting.
    Resume restores optimizer, scheduler, scaler, and RNGs (exact with workers=0).
    """
    args = {"epochs": 20, "lr": 1e-3, "seed": 0, "device": None, "patience": 7,
            "warmup_epochs": 1, "grad_clip": 1.0, "max_batches": None,
            "resume": None, "smoke": False, **(training_args or {})}
    if (args["epochs"] < 1 or args["lr"] <= 0 or not math.isfinite(args["lr"])
            or args["patience"] < 1 or args["warmup_epochs"] < 0 or args["grad_clip"] <= 0
            or (args["max_batches"] is not None and args["max_batches"] < 1)):
        raise ValueError("Invalid epoch, learning-rate, patience, warmup, clipping, or batch limit")
    if any(split not in loaders or len(loaders[split]) == 0 for split in ("train", "val", "test")):
        raise ValueError("Nonempty official train, val, and test loaders are required")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args["resume"] and any((out_dir / name).exists() for name in ("best.pt", "last.pt")):
        raise FileExistsError("Output already contains checkpoints; use --resume or a new out-dir")
    device = select_device(args["device"])
    print(f"device={device}; AMP={'cuda' if device.type == 'cuda' else 'off'}")
    seed_everything(args["seed"])
    resumed = read_checkpoint(args["resume"]) if args["resume"] else None
    if resumed:
        if args["epochs"] <= resumed["epoch"]:
            raise ValueError("--epochs is a total target and must exceed the resumed epoch")
        stored = ModelConfig.from_dict(resumed["model_config"])
        for name in ("arch", "encoder", "channels", "in_channels", "classes", "depth", "width"):
            if getattr(config, name) != getattr(stored, name):
                raise ValueError(f"Resume model configuration differs: {name}")
        if config.backend not in ("auto", stored.backend):
            raise ValueError("Resume backend differs from checkpoint")
        if (stats["channels"] != resumed["normalization_stats"]["channels"]
                or stats.get("training_ids") != resumed["normalization_stats"].get("training_ids")
                or args["seed"] != resumed["seed"]):
            raise ValueError("Resume requires the same training constants, identities, and seed")
        for name in ("lr", "warmup_epochs", "grad_clip", "focal", "speckle_filter_prob",
                     "db_gain_jitter", "augment", "crop_size", "batch_size", "num_workers", "max_batches"):
            if name in resumed["training_args"] and args.get(name) != resumed["training_args"][name]:
                raise ValueError(f"Resume training setting differs: {name}")
        config = stored
        stats = resumed["normalization_stats"]
    if model is None:
        model = build_model(config.arch, config.encoder, config.in_channels, config.classes,
                            None if resumed else config.encoder_weights, backend=config.backend,
                            depth=config.depth, width=config.width)
    config = replace(config, backend=model.backend,
                     encoder_weights=config.encoder_weights if model.backend == "smp" else None)
    model.to(device)
    balance = stats.get("class_balance")
    if balance is None and hasattr(loaders["train"].dataset, "source"):
        balance = class_balance(loaders["train"].dataset)
    weight = stats.get("pos_weight", (balance or {}).get("pos_weight"))
    if weight is None:
        raise ValueError("TRAIN class balance/pos_weight is missing or water is absent")
    stats = {**stats, "pos_weight": weight, "class_balance": balance}
    loss_fn = FloodLoss(stats=stats, focal=args.get("focal", False)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args.get("weight_decay", 1e-4))
    steps = min(len(loaders["train"]), args["max_batches"] or len(loaders["train"]))
    total_steps = args["epochs"] * steps
    warmup = min(args["warmup_epochs"] * steps, max(0, total_steps - 1))

    def schedule(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1, max(0, (step - warmup) / max(1, total_steps - warmup)))
        return .5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_iou, stale = 1, -1.0, 0
    if resumed:
        model.load_state_dict(resumed["state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scheduler.load_state_dict(resumed["scheduler_state_dict"])
        scaler.load_state_dict(resumed["scaler_state_dict"])
        start_epoch, best_iou = resumed["epoch"] + 1, resumed["best_iou"]
        stale = resumed["epochs_without_improvement"]
        _restore_rng(resumed["rng_state"], loaders, device)
        old_best = Path(args["resume"]).parent / "best.pt"
        if not old_best.exists():
            if not resumed["is_best"]:
                raise FileNotFoundError("Resume needs the original best.pt to preserve model selection")
            old_best = Path(args["resume"])
        if old_best.resolve() != (out_dir / "best.pt").resolve():
            shutil.copy2(old_best, out_dir / "best.pt")
    serial_args = json.loads(json.dumps(args, default=str))
    (out_dir / "config.json").write_text(json.dumps({"model_config": config.to_dict(),
                                                   "training_args": serial_args}, indent=2))
    receipt = {
        "format_version": 1, "model_config": config.to_dict(), "channels": config.channels,
        "normalization_stats": stats, "pos_weight": float(weight), "training_args": serial_args,
        "git_commit": _git_commit(), "seed": args["seed"], "library_versions": _versions(),
        "split_sizes": {split: len(loader.dataset) for split, loader in loaders.items()},
        "split_regions": {split: sorted({r.region for r in getattr(loader.dataset, "records", [])})
                          for split, loader in loaders.items()},
        "training_source": args.get("training_source", "sen1floods11"), "data_caveat": DATA_CAVEAT,
        "warnings": list(dict.fromkeys([*(resumed or {}).get("warnings", []),
                                        *(getattr(loaders["train"].dataset.source, "warnings", [])
                                           if hasattr(loaders["train"].dataset, "source") else []),
                                        *([model.backend_warning] if model.backend_warning else []),
                                        *(["Smoke run uses limited batches; metrics are not full-split results"]
                                          if args["smoke"] else [])])),
    }
    if args.get("num_workers", 0) > 0:
        receipt["warnings"].append("Multiworker resume cannot restore worker-local augmentation RNG states")
    for epoch in range(start_epoch, args["epochs"] + 1):
        started = time.perf_counter()
        trained = train_epoch(model, loaders["train"], optimizer, loss_fn, device,
                              scheduler=scheduler, scaler=scaler, grad_clip=args["grad_clip"],
                              max_batches=args["max_batches"])
        val = validate(model, loaders["val"], device, loss_fn=loss_fn,
                       max_batches=args["max_batches"], select_threshold=True)
        if val["iou"] is None:
            raise ValueError("Validation has no valid pixels; cannot select a best model")
        improved = val["iou"] > best_iou
        if improved:
            best_iou, stale = val["iou"], 0
        else:
            stale += 1
        log = {"epoch": epoch, "train_loss": trained["loss"], "train": trained,
               "val_metrics": val, "per_region_val_iou": {k: v["iou"] for k, v in val["per_region"].items()},
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - started}
        with (out_dir / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(log, allow_nan=False) + "\n")
        payload = {**receipt, "state_dict": model.state_dict(), "epoch": epoch,
                   "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                   "scaler_state_dict": scaler.state_dict(), "rng_state": _rng_state(loaders, device),
                   "best_iou": best_iou, "best_threshold": val["best_threshold"],
                   "epochs_without_improvement": stale, "is_best": improved}
        if improved:
            _save_checkpoint(payload, out_dir / "best.pt")
        _save_checkpoint(payload, out_dir / "last.pt")
        print(f"epoch={epoch} train_loss={trained['loss']['total']:.4f} val_iou={val['iou']:.4f} "
              f"seconds={log['seconds']:.1f}")
        if stale >= args["patience"]:
            break
    best_model, meta = load_checkpoint(out_dir / "best.pt", device)
    results = {}
    for split in ("val", "test", "bolivia"):
        if split not in loaders or not len(loaders[split]):
            results[split] = {"status": "unavailable", "reason": "No official split samples provided"}
        else:
            results[split] = validate(best_model, loaders[split], device,
                                      threshold=meta["best_threshold"], loss_fn=loss_fn,
                                      max_batches=args["max_batches"])
    report = {"data_caveat": DATA_CAVEAT, "threshold_selection_split": "val",
              "threshold": meta["best_threshold"], "selection_iou_threshold": .5,
              "best_epoch": meta["epoch"], "split_sizes": receipt["split_sizes"],
              "smoke": args["smoke"], "max_batches": args["max_batches"],
              "warnings": receipt["warnings"], "splits": results}
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/external/sen1floods11")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--arch", default="unet")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--backend", choices=("auto", "smp", "fallback"), default="auto")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--channels", nargs="+", default=["post_vv", "post_vh"])
    parser.add_argument("--add-ratio", action="store_true")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speckle-filter-prob", type=float, default=0.0)
    parser.add_argument("--db-gain-jitter", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--focal", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.epochs, args.batch_size, args.crop_size, args.pretrained = 2, 1, 64, False
    seed_everything(args.seed)
    print(f"selected device: {select_device(args.device)}")
    loaders = make_dataloaders(
        args.data_root, args.channels, args.batch_size, args.num_workers, args.seed,
        crop_size=args.crop_size, speckle_filter_prob=args.speckle_filter_prob,
        db_gain_jitter=args.db_gain_jitter, add_ratio=args.add_ratio,
    )
    loaders["train"].dataset.augment = args.augment
    source = loaders["train"].dataset.source
    stats = loaders["train"].dataset.stats
    if "bolivia" in source.splits:
        dataset = FloodChipDataset(split="bolivia", source=source, channels=args.channels,
                                   stats=stats, augment=False, add_ratio=args.add_ratio)
        loaders["bolivia"] = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    actual_channels = loaders["train"].dataset.channels
    config = ModelConfig(arch=args.arch, encoder=args.encoder, in_channels=len(actual_channels),
                         channels=actual_channels, encoder_weights="imagenet" if args.pretrained else None,
                         backend=args.backend, depth=args.depth, width=args.width)
    run_training(loaders, args.out_dir, config, stats,
                 {**vars(args), "max_batches": 2 if args.smoke else None})


if __name__ == "__main__":
    main()
