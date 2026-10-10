"""Offline flood chips, official splits, and training-only normalization."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import rasterio

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None


DB_CLIP = (-35.0, 5.0)
IGNORE_INDEX = 255
CHANNELS = frozenset({
    "pre_vv", "pre_vh", "post_vv", "post_vh", "elevation", "slope", "hand",
})
DEFAULT_CHANNELS = ("post_vv", "post_vh")
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "val": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


@dataclass(frozen=True)
class ChipRecord:
    """Stable chip identity and its official split membership."""

    id: str
    region: str
    split: str
    image_path: Path | None = None
    label_path: Path | None = None


@dataclass(frozen=True)
class ChipSample:
    """Named raw channels, binary/255 labels, and source validity on one grid."""

    channels: Mapping[str, np.ndarray]
    label: np.ndarray
    valid_mask: np.ndarray


class SampleSource(Protocol):
    """Adapters supply official records and aligned, named raw sample arrays."""

    available_channels: Sequence[str]
    splits: Mapping[str, Sequence[ChipRecord]]

    def read(self, record: ChipRecord) -> ChipSample:
        """Return channels in dB for SAR, meters/degrees for terrain."""
        ...


def map_labels(values: np.ndarray, invalid: np.ndarray | None = None) -> np.ndarray:
    """Map only 0/1 to classes; missing, -1, and other values become 255."""
    result = np.full(values.shape, IGNORE_INDEX, dtype=np.int64)
    good = np.isfinite(values) & ((values == 0) | (values == 1))
    if invalid is not None:
        good &= ~invalid
    result[good] = values[good].astype(np.int64)
    return result


class Sen1Floods11Index:
    """Pair local VV/VH chips and read the unmodified headerless official CSVs.

    Bolivia is a separate holdout, not the official test CSV. Train/val/test
    share regions in these lists; only chip identities must be disjoint.
    Permanent-water CSVs describe different, locally unavailable imagery.
    """

    available_channels = DEFAULT_CHANNELS

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.warnings: list[str] = []
        images = self._scan("S1Hand", "_S1Hand.tif")
        labels = self._scan("LabelHand", "_LabelHand.tif")
        if not images:
            raise ValueError(f"No S1Hand chips found in {self.root}")
        if images.keys() != labels.keys():
            missing_labels = sorted(images.keys() - labels.keys())
            missing_images = sorted(labels.keys() - images.keys())
            raise ValueError(
                f"Unpaired chips: missing labels {missing_labels}; "
                f"missing images {missing_images}"
            )
        for chip_id in images:
            with rasterio.open(images[chip_id]) as image, rasterio.open(labels[chip_id]) as label:
                if image.count != 2 or label.count != 1:
                    raise ValueError(f"{chip_id}: expected two SAR bands and one label band")
                descriptions = tuple((s or "").upper() for s in image.descriptions)
                if any(descriptions) and descriptions != ("VV", "VH"):
                    raise ValueError(f"{chip_id}: expected band order VV, VH; got {descriptions}")
                corners = np.array(((0, 0, 1), (image.width, 0, 1),
                                    (0, image.height, 1), (image.width, image.height, 1))).T
                pixel_transform = (np.asarray(~image.transform).reshape(3, 3)
                                   @ np.asarray(label.transform).reshape(3, 3))
                aligned = np.allclose(pixel_transform @ corners, corners, rtol=0, atol=1e-6)
                if image.shape != label.shape or not aligned or image.crs != label.crs:
                    raise ValueError(f"{chip_id}: image and label grids do not align")
                if label.nodata in (0, 1):
                    self.warnings.append(
                        f"{chip_id}: label nodata={label.nodata:g} conflicts with a class; "
                        "the declared nodata pixels are ignored"
                    )
        self.splits: dict[str, list[ChipRecord]] = {}
        seen: dict[str, str] = {}
        for split, filename in SPLIT_FILES.items():
            path = self.root / "splits" / "flood_handlabeled" / filename
            if split == "bolivia" and not path.exists():
                continue
            if not path.exists():
                raise ValueError(f"Missing official split file: {path}")
            records = []
            with path.open(newline="") as handle:
                for row in csv.reader(handle):
                    if not row:
                        continue
                    if len(row) != 2:
                        raise ValueError(f"{path}: expected image,label filenames per row")
                    image_name, label_name = (Path(s.strip()).name for s in row)
                    chip_id = image_name.removesuffix("_S1Hand.tif")
                    if (chip_id not in images or image_name != images[chip_id].name
                            or label_name != labels[chip_id].name):
                        raise ValueError(f"{path}: missing or mismatched pair {row}")
                    if chip_id in seen:
                        raise ValueError(
                            f"Tile id overlap or duplicate: {chip_id} in {seen[chip_id]} and {split}"
                        )
                    seen[chip_id] = split
                    records.append(ChipRecord(
                        chip_id, chip_id.rsplit("_", 1)[0], split,
                        images[chip_id], labels[chip_id],
                    ))
            self.splits[split] = records
        unassigned = sorted(images.keys() - seen.keys())
        if unassigned:
            raise ValueError(f"Chips absent from official split files: {unassigned}")
        self.regions = {
            split: sorted({record.region for record in records})
            for split, records in self.splits.items()
        }

    def _scan(self, folder: str, suffix: str) -> dict[str, Path]:
        paths = sorted((self.root / folder).glob("*.tif"))
        for path in paths:
            if not path.name.endswith(suffix):
                raise ValueError(f"Unexpected chip filename: {path}")
        return {path.name.removesuffix(suffix): path for path in paths}

    def read(self, record: ChipRecord) -> ChipSample:
        """Read a chip, honoring each raster's nodata declaration."""
        with rasterio.open(record.image_path) as image:
            bands = image.read(masked=True).astype(np.float32).filled(np.nan)
        with rasterio.open(record.label_path) as label:
            raw = label.read(1, masked=True)
            mapped = map_labels(raw.data, np.ma.getmaskarray(raw))
        return ChipSample(
            dict(zip(self.available_channels, bands)), mapped, mapped != IGNORE_INDEX,
        )


def _channel_names(source: SampleSource, channels: Sequence[str], add_ratio: bool) -> list[str]:
    names = list(channels)
    if not names or len(set(names)) != len(names):
        raise ValueError("channels must be nonempty and contain no duplicates")
    for name in names:
        if name not in CHANNELS:
            raise ValueError(f"Unsupported channel: {name}")
        if name not in source.available_channels:
            raise ValueError(f"Requested channel {name!r} is unavailable from this sample source")
    if add_ratio:
        ratios = [f"{time}_ratio" for time in ("pre", "post")
                  if f"{time}_vv" in names and f"{time}_vh" in names]
        if not ratios:
            raise ValueError("add_ratio requires both VV and VH for an acquisition")
        names.extend(ratios)
    return names


def _arrays(sample: ChipSample, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip SAR dB, derive differences, and combine source/label/channel validity."""
    label = map_labels(sample.label)
    valid = np.asarray(sample.valid_mask, dtype=bool) & (label != IGNORE_INDEX)
    if valid.shape != label.shape or label.ndim != 2:
        raise ValueError("Label and validity must be aligned two-dimensional arrays")
    arrays = []
    for name in names:
        if name.endswith("_ratio"):
            prefix = name.removesuffix("_ratio")
            array = (np.clip(sample.channels[f"{prefix}_vv"], *DB_CLIP)
                     - np.clip(sample.channels[f"{prefix}_vh"], *DB_CLIP))
        else:
            if name not in sample.channels:
                raise ValueError(f"Requested channel {name!r} is unavailable in this sample")
            array = np.asarray(sample.channels[name], dtype=np.float32)
            if name.endswith(("_vv", "_vh")):
                array = np.where(np.isfinite(array), np.clip(array, *DB_CLIP), np.nan)
        array = np.asarray(array, dtype=np.float32)
        if array.shape != label.shape:
            raise ValueError(f"Channel {name} does not align with the label")
        arrays.append(array)
        valid &= np.isfinite(array)
    label[~valid] = IGNORE_INDEX
    return np.stack(arrays), label, valid


def _training_signature(source: SampleSource) -> str:
    entries = []
    for record in source.splits["train"]:
        files = []
        for path in (record.image_path, record.label_path):
            if path is not None:
                stat = path.stat()
                files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        entries.append((record.id, files))
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def compute_train_stats(
    source: SampleSource,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    add_ratio: bool = False,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute fixed population mean/std from full TRAIN chips, never other splits.

    SAR is clipped to [-35, 5] dB; ratios subtract clipped VV/VH (no further
    clipping). Terrain keeps meters/degrees. Only jointly valid, labeled pixels
    contribute. Constant channels use std=1. Persisted constants remain fixed
    during cropping, augmentation, and validation/test loading.
    """
    names = _channel_names(source, channels, add_ratio)
    if not source.splits.get("train"):
        raise ValueError("Training split is empty; cannot compute normalization")
    signature = _training_signature(source)
    destination = Path(path) if path is not None else (
        source.root / "stats.json" if isinstance(source, Sen1Floods11Index) else None
    )
    if destination is not None and destination.exists():
        saved = json.loads(destination.read_text())
        if (saved.get("version") == 1 and saved.get("training_signature") == signature
                and saved.get("channel_names") == names and saved.get("clip_db") == list(DB_CLIP)):
            return saved
    count = 0
    means = np.zeros(len(names), dtype=np.float64)
    m2 = np.zeros_like(means)
    for record in source.splits["train"]:
        image, _, valid = _arrays(source.read(record), names)
        values = image[:, valid].astype(np.float64)
        n = values.shape[1]
        if not n:
            continue
        batch_mean = values.mean(axis=1)
        batch_m2 = np.square(values - batch_mean[:, None]).sum(axis=1)
        delta = batch_mean - means
        m2 += batch_m2 + np.square(delta) * count * n / (count + n)
        means += delta * n / (count + n)
        count += n
    if not count:
        raise ValueError("Training split has no valid labeled pixels")
    stds = np.sqrt(m2 / count)
    stds[stds < 1e-6] = 1.0
    result = {
        "version": 1, "clip_db": list(DB_CLIP), "channel_names": names,
        "training_signature": signature,
        "training_ids": [record.id for record in source.splits["train"]],
        "channels": {name: {"mean": float(mean), "std": float(std), "count": count}
                     for name, mean, std in zip(names, means, stds)},
    }
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)
    return result


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required: .venv/bin/python -m pip install torch")


class FloodChipDataset(Dataset):
    """Named normalized channels; train crops/geometry augmentation, full heldout chips.

    A supplied source can provide pre-event and terrain channels. Val/test never
    crop or augment; crop_size=None also keeps full training chips. Each dataset
    owns a seeded generator, so identical access sequences reproduce transforms.

    Optional TRAIN-only radiometry addresses a possible domain gap: local chips
    have unset units/processing metadata, so unfiltered sigma0 is an assumption,
    not verified provenance. Inference uses Lee-filtered gamma0 RTC. Random Lee
    filtering converts SAR dB to power and reuses the preprocessing functions;
    one shared gain offset affects all SAR channels, never terrain units. These
    options simulate variation; they do not convert sigma0 into gamma0.
    """

    def __init__(
        self,
        root: str | Path | Sen1Floods11Index | None = None,
        split: str = "train",
        channels: Sequence[str] = DEFAULT_CHANNELS,
        crop_size: int | None = 256,
        augment: bool = True,
        add_ratio: bool = False,
        *,
        source: SampleSource | None = None,
        stats: Mapping[str, Any] | None = None,
        seed: int = 0,
        speckle_filter_prob: float = 0.0,
        db_gain_jitter: float = 0.0,
    ) -> None:
        _require_torch()
        if source is None:
            if root is None:
                raise ValueError("Provide a dataset root or a SampleSource")
            source = root if isinstance(root, Sen1Floods11Index) else Sen1Floods11Index(root)
        if split not in source.splits:
            raise ValueError(f"Unknown official split: {split}")
        if crop_size is not None and crop_size <= 0:
            raise ValueError("crop_size must be positive or None")
        if (not np.isfinite(speckle_filter_prob) or not 0 <= speckle_filter_prob <= 1
                or not np.isfinite(db_gain_jitter) or db_gain_jitter < 0):
            raise ValueError("speckle_filter_prob must be in [0,1]; db_gain_jitter nonnegative")
        self.source = source
        self.split = split
        self.records = list(source.splits[split])
        self.channels = _channel_names(source, channels, add_ratio)
        self.stats = dict(stats) if stats is not None else compute_train_stats(source, channels, add_ratio)
        for name in self.channels:
            constants = self.stats.get("channels", {}).get(name, {})
            if (not np.isfinite(constants.get("mean", np.nan))
                    or not np.isfinite(constants.get("std", np.nan))
                    or constants.get("std", 0) <= 0):
                raise ValueError(f"Missing or invalid normalization constants for {name}")
        self.crop_size = crop_size
        self.augment = augment and split == "train"
        self.rng = np.random.default_rng(seed)
        self.speckle_filter_prob = speckle_filter_prob
        self.db_gain_jitter = db_gain_jitter

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = self.source.read(record)
        if self.split == "train" and (self.speckle_filter_prob or self.db_gain_jitter):
            filtered = self.speckle_filter_prob > 0 and self.rng.random() < self.speckle_filter_prob
            gain = self.rng.uniform(-self.db_gain_jitter, self.db_gain_jitter) if self.db_gain_jitter else 0
            raw = dict(sample.channels)
            for name, values in raw.items():
                if not name.endswith(("_vv", "_vh")):
                    continue
                if filtered:
                    from src.data.s1_preprocess import lee_filter, power_to_db
                    power = np.power(10.0, np.asarray(values, dtype=np.float64) / 10)
                    values = power_to_db(lee_filter(power, window_size=5))
                raw[name] = np.asarray(values, dtype=np.float32) + gain
            sample = ChipSample(raw, sample.label, sample.valid_mask)
        image, label, valid = _arrays(sample, self.channels)
        for band, name in enumerate(self.channels):
            constants = self.stats["channels"][name]
            image[band] = (image[band] - constants["mean"]) / constants["std"]
        image[:, ~valid] = 0.0
        if self.split == "train" and self.crop_size is not None:
            height, width = label.shape
            size = self.crop_size
            if size > min(height, width):
                raise ValueError(f"crop_size={size} exceeds chip shape {label.shape}")
            y = int(self.rng.integers(height - size + 1))
            x = int(self.rng.integers(width - size + 1))
            image = image[:, y:y + size, x:x + size]
            label = label[y:y + size, x:x + size]
            valid = valid[y:y + size, x:x + size]
        if self.augment:
            for axis in (0, 1):
                if self.rng.integers(2):
                    image = np.flip(image, axis=axis + 1)
                    label = np.flip(label, axis=axis)
                    valid = np.flip(valid, axis=axis)
            turns = int(self.rng.integers(4))
            image = np.rot90(image, turns, axes=(1, 2))
            label = np.rot90(label, turns)
            valid = np.rot90(valid, turns)
        return {
            "image": torch.from_numpy(image.copy()).to(torch.float32),
            "label": torch.from_numpy(label.copy()).to(torch.int64),
            "valid_mask": torch.from_numpy(valid.copy()).to(torch.bool),
            "meta": {"id": record.id, "region": record.region, "split": self.split},
        }


def class_balance(dataset: FloodChipDataset) -> dict[str, Any]:
    """Count full chips, without random crops; fractions include ignored pixels.

    pos_weight is non-water/water. Balanced class weights use valid pixels only;
    unavailable classes produce None instead of an invented finite weight.
    """
    counts = np.zeros(3, dtype=np.int64)
    regions: dict[str, np.ndarray] = {}
    for record in dataset.records:
        _, label, _ = _arrays(dataset.source.read(record), dataset.channels)
        current = np.array([(label == 1).sum(), (label == 0).sum(),
                            (label == IGNORE_INDEX).sum()], dtype=np.int64)
        counts += current
        regions.setdefault(record.region, np.zeros(3, dtype=np.int64))[:] += current

    def summary(values: np.ndarray) -> dict[str, Any]:
        water, dry, ignored = map(int, values)
        total = water + dry + ignored
        return {
            "counts": {"water": water, "non_water": dry, "ignore": ignored},
            "fractions": {"water": water / total if total else 0.0,
                          "non_water": dry / total if total else 0.0,
                          "ignore": ignored / total if total else 0.0},
            "valid_water_fraction": water / (water + dry) if water + dry else None,
        }

    water, dry, _ = map(int, counts)
    return {
        "split": dataset.split, **summary(counts),
        "per_region": {region: summary(values) for region, values in sorted(regions.items())},
        "pos_weight": dry / water if water else None,
        "class_weights": [(water + dry) / (2 * dry) if dry else None,
                          (water + dry) / (2 * water) if water else None],
    }


def _seed_worker(worker_id: int) -> None:
    """Give each worker a deterministic, distinct augmentation generator."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))


def make_dataloaders(
    root: str | Path,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 0,
    *,
    crop_size: int | None = 256,
    add_ratio: bool = False,
    speckle_filter_prob: float = 0.0,
    db_gain_jitter: float = 0.0,
) -> dict[str, Any]:
    """Build deterministic official train/val/test loaders; workers default to zero."""
    _require_torch()
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    source = Sen1Floods11Index(root)
    stats = compute_train_stats(source, channels, add_ratio)
    loaders = {}
    for offset, split in enumerate(("train", "val", "test")):
        dataset = FloodChipDataset(
            split=split, channels=channels, crop_size=crop_size, add_ratio=add_ratio,
            source=source, stats=stats, seed=seed + offset,
            speckle_filter_prob=speckle_filter_prob, db_gain_jitter=db_gain_jitter,
        )
        generator = torch.Generator().manual_seed(seed + offset)
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=split == "train", num_workers=num_workers,
            worker_init_fn=_seed_worker, generator=generator,
        )
    return loaders
