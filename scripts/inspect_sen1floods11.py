"""Inspect local Sen1Floods11 files, train constants, balance, and six previews."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import sys
import zlib

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.data import (  # noqa: E402
    DB_CLIP, FloodChipDataset, Sen1Floods11Index, class_balance, compute_train_stats,
)


def inspect_dataset(index: Sen1Floods11Index) -> dict:
    """Read every local raster to report exact ranges and label/nodata values."""
    facts = {}
    for folder in ("S1Hand", "LabelHand"):
        files = sorted((index.root / folder).glob("*.tif"))
        shapes, dtypes, descriptions, units = set(), set(), set(), set()
        nodata = Counter()
        low, high = float("inf"), float("-inf")
        band_ranges = {}
        label_counts = Counter()
        nonfinite = 0
        empty = []
        for path in files:
            with rasterio.open(path) as raster:
                array = raster.read()
                shapes.add((raster.count, raster.height, raster.width))
                dtypes.add(raster.dtypes)
                descriptions.add(raster.descriptions)
                units.add(raster.units)
                nodata[str(raster.nodata)] += 1
            nonfinite += int((~np.isfinite(array)).sum())
            finite = array[np.isfinite(array)]
            if finite.size:
                low = min(low, float(finite.min()))
                high = max(high, float(finite.max()))
            else:
                empty.append(path.name)
            for band, values in enumerate(array, start=1):
                values = values[np.isfinite(values)]
                if values.size:
                    previous = band_ranges.get(str(band), [float("inf"), float("-inf")])
                    band_ranges[str(band)] = [min(previous[0], float(values.min())),
                                             max(previous[1], float(values.max()))]
            if folder == "LabelHand":
                values, counts = np.unique(array, return_counts=True)
                for value, count in zip(values, counts):
                    label_counts[str(int(value))] += int(count)
        facts[folder] = {
            "count": len(files), "filename_examples": [path.name for path in files[:3]],
            "shapes_C_H_W": sorted(shapes), "dtypes": sorted(dtypes),
            "band_descriptions": sorted(descriptions, key=str), "units": sorted(units, key=str),
            "nodata_file_counts": dict(nodata),
            "finite_range": [low, high] if low != float("inf") else None,
            "band_ranges": band_ranges, "label_pixel_counts": dict(label_counts),
            "nonfinite_pixels": nonfinite, "entirely_nonfinite_chips": empty,
        }
    facts["splits"] = {
        split: {"count": len(records), "regions": index.regions[split]}
        for split, records in index.splits.items()
    }
    facts["split_files"] = {
        str(path.relative_to(index.root)): {
            "rows": len(path.read_text().splitlines()),
            "first_row": path.read_text().splitlines()[:1],
        }
        for path in sorted((index.root / "splits").glob("*/*.csv"))
    }
    facts["split_format"] = "Headerless CSV: image filename,label filename"
    facts["tile_id_overlap"] = False  # The index rejects overlapping identities.
    facts["shared_train_val_test_regions"] = sorted(
        set(index.regions["train"]) & set(index.regions["val"]) & set(index.regions["test"])
    )
    facts["interpretation"] = (
        "Bands VV,VH are treated as dB: negative backscatter values are consistent "
        "with dB, but raster unit fields are unset. Labels: -1 invalid, 0 non-water, "
        "1 water; declared nodata is also ignored. Bolivia has its own CSV, not "
        "the official test CSV. Permanent-water CSVs refer to different chips."
    )
    facts["warnings"] = list(index.warnings)
    if facts["S1Hand"]["entirely_nonfinite_chips"]:
        facts["warnings"].append("Entirely nonfinite SAR chips yield only ignored pixels")
    return facts


def write_preview(vv: np.ndarray, label: np.ndarray, path: Path) -> None:
    """Write side-by-side VV grayscale and water/ignore overlays without extra packages."""
    gray = np.nan_to_num((np.clip(vv, *DB_CLIP) - DB_CLIP[0]) / (DB_CLIP[1] - DB_CLIP[0]))
    rgb = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2)
    overlay = rgb.copy()
    for value, color in ((1, [0, 130, 255]), (255, [190, 40, 190])):
        mask = label == value
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.45
                         + np.array(color, dtype=np.float32) * 0.55).astype(np.uint8)
    pixels = np.concatenate((rgb, overlay), axis=1)
    height, width = pixels.shape[:2]

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload)))

    rows = b"".join(b"\x00" + row.tobytes() for row in pixels)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/external/sen1floods11"))
    args = parser.parse_args()
    index = Sen1Floods11Index(args.root)
    print("Data facts:")
    print(json.dumps(inspect_dataset(index), indent=2, allow_nan=False))
    stats = compute_train_stats(index)
    print("Training-only constants:")
    print(json.dumps(stats["channels"], indent=2, allow_nan=False))
    balances = {}
    previews = []
    for split in index.splits:
        dataset = FloodChipDataset(index, split=split, crop_size=None, augment=False, stats=stats)
        balances[split] = class_balance(dataset)
        if split not in ("train", "val", "test"):
            continue
        for record in index.splits[split][:2]:
            sample = index.read(record)
            label = sample.label.copy()
            label[~(sample.valid_mask & np.isfinite(sample.channels["post_vv"])
                    & np.isfinite(sample.channels["post_vh"]))] = 255
            path = args.root / "previews" / f"{split}_{record.id}.png"
            write_preview(sample.channels["post_vv"], label, path)
            previews.append(str(path))
    print("Class balance (full chips; ignore includes SAR invalidity and declared nodata):")
    print(json.dumps(balances, indent=2, allow_nan=False))
    print(json.dumps({"stats_path": str(args.root / "stats.json"), "previews": previews}, indent=2))


if __name__ == "__main__":
    main()
