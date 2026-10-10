"""Checkpoint-owned normalization and bounded-memory overlapping inference."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
import struct
from typing import Any
import zlib

import numpy as np
import torch
from torch import nn

from src.model.data import DB_CLIP
from src.model.network import ModelConfig, build_model


def select_device(device: str | torch.device | None = None) -> torch.device:
    """Prefer CUDA, then MPS, then CPU; an explicit CPU request is respected."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load tensor/primitive payloads only; never unpickle arbitrary model objects."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_config", "state_dict", "normalization_stats", "channels", "best_threshold"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Checkpoint lacks model/channel/normalization/threshold metadata")
    config = ModelConfig.from_dict(payload["model_config"])
    if config.backend == "auto" or config.channels != payload["channels"]:
        raise ValueError("Checkpoint must name its actual backend and consistent ordered channels")
    if not 0 <= payload["best_threshold"] <= 1:
        raise ValueError("Checkpoint threshold must be in [0,1]")
    return payload


def load_checkpoint(path: str | Path, device: str | torch.device | None = None) -> tuple[nn.Module, dict[str, Any]]:
    payload = read_checkpoint(path)
    config = ModelConfig.from_dict(payload["model_config"])
    model = build_model(arch=config.arch, encoder=config.encoder, in_channels=config.in_channels,
                        classes=config.classes, encoder_weights=None, backend=config.backend,
                        depth=config.depth, width=config.width)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(select_device(device)).eval()
    excluded = {"state_dict", "optimizer_state_dict", "scheduler_state_dict",
                "scaler_state_dict", "rng_state"}
    meta = {key: value for key, value in payload.items() if key not in excluded}
    meta["checkpoint_path"] = str(Path(path).resolve())
    return model, meta


def stack_to_model_input(stack: Any, meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Select named bands, apply checkpoint constants, and honor the stack mask.

    Ratio channels subtract individually clipped VV/VH exactly as in Step 10.
    A terrain-trained checkpoint needs a source providing named terrain bands;
    an ordinary S1Stack cannot silently substitute missing terrain channels.
    """
    names = meta["channels"]
    config = ModelConfig.from_dict(meta["model_config"])
    if names != config.channels:
        raise ValueError("Checkpoint channel list differs from ModelConfig")
    if len(set(stack.band_names)) != len(stack.band_names) or len(stack.band_names) != stack.array.shape[0]:
        raise ValueError("Stack must have unique band names matching its array")
    bands = dict(zip(stack.band_names, stack.array))
    valid = np.array(stack.valid_mask, dtype=bool, copy=True)
    if valid.shape != stack.array.shape[1:]:
        raise ValueError("Stack validity mask and grid shapes differ")
    raw = []
    for name in names:
        required = [name]
        if name.endswith("_ratio"):
            prefix = name.removesuffix("_ratio")
            required = [f"{prefix}_vv", f"{prefix}_vh"]
        for band in required:
            if band not in bands:
                raise ValueError(f"Model channel {band!r} is unavailable in this stack")
            valid &= np.isfinite(bands[band])
        if name.endswith("_ratio"):
            values = np.clip(bands[required[0]], *DB_CLIP) - np.clip(bands[required[1]], *DB_CLIP)
        else:
            values = np.asarray(bands[name], dtype=np.float32)
            if name.endswith(("_vv", "_vh")):
                values = np.clip(values, *DB_CLIP)
        raw.append(values)
    image = np.stack(raw).astype(np.float32)
    stats = meta["normalization_stats"]
    if stats.get("clip_db", list(DB_CLIP)) != list(DB_CLIP):
        raise ValueError("Checkpoint dB clipping differs from the supported training contract")
    for i, name in enumerate(names):
        constants = stats.get("channels", {}).get(name)
        if (constants is None or not np.isfinite(constants["mean"])
                or not np.isfinite(constants["std"]) or constants["std"] <= 0):
            raise ValueError(f"Missing or invalid checkpoint constants for {name}")
        image[i] = (image[i] - constants["mean"]) / constants["std"]
    image[:, ~valid] = 0
    return image, valid


def _starts(length: int, tile: int, stride: int) -> list[int]:
    starts = list(range(0, max(length - tile, 0) + 1, stride))
    if starts[-1] + tile < length:
        starts.append(starts[-1] + stride)
    return starts


def _patch(array: np.ndarray, y: int, x: int, tile: int, halo: int) -> np.ndarray:
    height, width = array.shape[1:]
    top, left, bottom, right = y - halo, x - halo, y + tile + halo, x + tile + halo
    patch = array[:, max(0, top):min(height, bottom), max(0, left):min(width, right)]
    padding = ((0, 0), (max(0, -top), max(0, bottom - height)),
               (max(0, -left), max(0, right - width)))
    return np.pad(patch, padding, mode="reflect" if min(patch.shape[1:]) > 1 else "edge")


def predict_probabilities(
    model: nn.Module, array: np.ndarray, valid_mask: np.ndarray, tile: int = 512,
    overlap: int = 128, device: str | torch.device | None = None, tta: bool = False,
    *, batch_size: int = 2,
) -> np.ndarray:
    """Reflect-pad bounded tile batches; discard context halos and Hann-blend cores.

    A halo of overlap/2 (at least one pixel) suppresses convolution edge seams.
    Spatial normalization or receptive fields larger than the halo can still make
    a network tile-dependent; overlap is not a mathematical full-image guarantee.
    Standard settings align the stride and halo to the encoder's 32-pixel stride.
    Only two full-grid float32 accumulators live alongside the caller's input.
    """
    array = np.asarray(array, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    if (array.ndim != 3 or mask.shape != array.shape[1:] or min(mask.shape) < 1
            or type(tile) is not int or type(overlap) is not int
            or tile < 2 or not 0 <= overlap < tile or batch_size < 1):
        raise ValueError("Require (C,H,W), matching mask, tile>=2, 0<=overlap<tile, batch_size>=1")
    if np.any(mask & ~np.isfinite(array).all(axis=0)):
        raise ValueError("Valid pixels contain nonfinite model inputs")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0, posinf=0, neginf=0)
    parameter = next(model.parameters(), None)
    device = select_device(device) if device is not None else (
        parameter.device if parameter is not None else torch.device("cpu")
    )
    model.to(device)
    height, width = mask.shape
    total = np.zeros(mask.shape, dtype=np.float32)
    weights = np.zeros_like(total)
    axis = np.hanning(tile + 2)[1:-1].astype(np.float32)
    window = axis[:, None] * axis[None, :]
    stride, halo = tile - overlap, max(1, overlap // 2)
    positions = [(y, x) for y in _starts(height, tile, stride) for x in _starts(width, tile, stride)]
    was_training = model.training
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    was_benchmark, was_cudnn_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
    model.eval()
    try:
        with torch.inference_mode():
            for offset in range(0, len(positions), batch_size):
                batch_positions = positions[offset:offset + batch_size]
                patches = np.stack([_patch(array, y, x, tile, halo) for y, x in batch_positions])
                tensor = torch.from_numpy(patches).to(device)
                flips = ((), (-1,), (-2,), (-2, -1)) if tta else ((),)
                probability = None
                for flip in flips:
                    logits = model(torch.flip(tensor, flip) if flip else tensor)
                    if logits.shape != (len(patches), 1, tile + 2 * halo, tile + 2 * halo):
                        raise ValueError("Model must return one logit channel on the input grid")
                    value = torch.sigmoid(logits.float())
                    if flip:
                        value = torch.flip(value, flip)
                    probability = value if probability is None else probability + value
                values = (probability / len(flips))[:, 0, halo:halo + tile, halo:halo + tile].cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("Model produced nonfinite probabilities")
                for (y, x), value in zip(batch_positions, values):
                    h, w = min(tile, height - y), min(tile, width - x)
                    total[y:y + h, x:x + w] += value[:h, :w] * window[:h, :w]
                    weights[y:y + h, x:x + w] += window[:h, :w]
    finally:
        model.train(was_training)
        torch.use_deterministic_algorithms(was_deterministic, warn_only=was_warn_only)
        torch.backends.cudnn.benchmark = was_benchmark
        torch.backends.cudnn.deterministic = was_cudnn_deterministic
    total /= weights
    np.clip(total, 0, 1, out=total)
    total[~mask] = np.nan
    return total


def write_prediction_quicklook(stack: Any, probability: np.ndarray, path: str | Path) -> None:
    """Overlay water probability in blue over post-VV dB, leaving invalid pixels gray."""
    vv = stack.array[list(stack.band_names).index("post_vv")]
    gray = np.nan_to_num((np.clip(vv, *DB_CLIP) - DB_CLIP[0]) / 40)
    pixels = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2).astype(np.float32)
    alpha = np.nan_to_num(probability, nan=0)[..., None] * .7
    pixels = (pixels * (1 - alpha) + np.array([0, 110, 255]) * alpha).astype(np.uint8)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    height, width = probability.shape
    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def predict_incident(
    incident_or_stack: Any, checkpoint: str | Path, *, device: str | None = None,
    tile: int = 512, overlap: int = 128, tta: bool = False, quicklook: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if hasattr(incident_or_stack, "array"):
        stack = incident_or_stack
    else:
        from src.data.incident import load_incident, load_layers
        incident = load_incident(incident_or_stack) if isinstance(incident_or_stack, (str, Path)) else incident_or_stack
        stack = load_layers(incident).s1_stack
    model, meta = load_checkpoint(checkpoint, device)
    image, valid = stack_to_model_input(stack, meta)
    warnings = [*stack.warnings, *meta.get("warnings", [])]
    if list(stack.band_names) != meta["channels"]:
        warnings.append(f"Model trained on {meta['channels']}; stack has {list(stack.band_names)}; "
                        "input is selected by name, with other bands unused")
    if float(valid.mean()) < .9:
        warnings.append(f"Model-input valid_fraction={valid.mean():.4f} is below 0.9")
    if stack.source == "rtc" and meta.get("training_source") == "sen1floods11":
        warnings.append("Sen1Floods11-to-RTC processing/calibration domain gap has not been validated")
    probability = predict_probabilities(model, image, valid, tile, overlap, device, tta)
    if quicklook is not None:
        write_prediction_quicklook(stack, probability, quicklook)
    return probability, valid, {
        "model_config": meta["model_config"], "checkpoint_path": meta["checkpoint_path"],
        "threshold": meta["best_threshold"], "valid_fraction": float(valid.mean()),
        "warnings": list(dict.fromkeys(warnings)),
    }
