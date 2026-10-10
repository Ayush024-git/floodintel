"""Offline training-input contracts using small synthetic GeoTIFFs."""

from dataclasses import dataclass
import json
import subprocess
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin
import torch

from src.model.data import (
    ChipRecord, ChipSample, FloodChipDataset, Sen1Floods11Index, class_balance,
    compute_train_stats, make_dataloaders,
)


def write_pair(root, chip_id, image, label, label_nodata=None):
    for folder in ("S1Hand", "LabelHand"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", height=label.shape[0], width=label.shape[1],
                   crs="EPSG:32645", transform=from_origin(100, 200, 10, 10))
    with rasterio.open(root / "S1Hand" / f"{chip_id}_S1Hand.tif", "w", **profile,
                       count=2, dtype="float32", nodata=np.nan) as dst:
        dst.write(image.astype(np.float32))
        dst.set_band_description(1, "VV")
        dst.set_band_description(2, "VH")
    with rasterio.open(root / "LabelHand" / f"{chip_id}_LabelHand.tif", "w", **profile,
                       count=1, dtype="int16", nodata=label_nodata) as dst:
        dst.write(label.astype(np.int16), 1)


def make_root(tmp_path, samples=None, label_nodata=None):
    root = tmp_path / "chips"
    if samples is None:
        pattern = np.arange(64).reshape(8, 8) % 2
        image = np.stack((-20 + pattern * 10, -25 + pattern * 10))
        samples = {split: [(f"{region}_{i}", image, pattern) for i in ids]
                   for split, region, ids in (
                       ("train", "Ghana", (1, 2)), ("val", "India", (3, 4)),
                       ("test", "Spain", (5, 6)), ("bolivia", "Bolivia", (7,)),
                   )}
    directory = root / "splits/flood_handlabeled"
    directory.mkdir(parents=True)
    filenames = {"train": "flood_train_data.csv", "val": "flood_valid_data.csv",
                 "test": "flood_test_data.csv", "bolivia": "flood_bolivia_data.csv"}
    for split in ("train", "val", "test", "bolivia"):
        if split not in samples and split == "bolivia":
            continue
        rows = []
        for chip_id, image, label in samples.get(split, []):
            write_pair(root, chip_id, image, label, label_nodata)
            rows.append(f"{chip_id}_S1Hand.tif,{chip_id}_LabelHand.tif")
        (directory / filenames[split]).write_text("\n".join(rows) + ("\n" if rows else ""))
    return root


def identity_stats(names):
    return {"channels": {name: {"mean": 0.0, "std": 1.0} for name in names}}


def test_pairs_official_splits_and_regions(tmp_path):
    index = Sen1Floods11Index(make_root(tmp_path))
    assert [r.id for r in index.splits["train"]] == ["Ghana_1", "Ghana_2"]
    assert index.regions == {"train": ["Ghana"], "val": ["India"],
                             "test": ["Spain"], "bolivia": ["Bolivia"]}
    assert index.splits["test"][0].image_path.name == "Spain_5_S1Hand.tif"
    assert index.read(index.splits["train"][0]).channels.keys() == {"post_vv", "post_vh"}


def test_unpaired_file_raises(tmp_path):
    root = make_root(tmp_path)
    (root / "LabelHand/Ghana_1_LabelHand.tif").unlink()
    with pytest.raises(ValueError, match="Unpaired.*Ghana_1"):
        Sen1Floods11Index(root)


def test_missing_split_pair_raises(tmp_path):
    root = make_root(tmp_path)
    (root / "splits/flood_handlabeled/flood_train_data.csv").write_text(
        "Missing_42_S1Hand.tif,Missing_42_LabelHand.tif\n"
    )
    with pytest.raises(ValueError, match="missing or mismatched"):
        Sen1Floods11Index(root)


def test_grid_roundoff_is_accepted_but_pixel_shift_is_rejected(tmp_path):
    root = make_root(tmp_path)
    path = root / "LabelHand/Ghana_1_LabelHand.tif"
    with rasterio.open(path, "r+") as raster:
        t = raster.transform
        raster.transform = Affine(t.a, t.b, t.c + 1e-7, t.d, t.e, t.f)
    Sen1Floods11Index(root)
    with rasterio.open(path, "r+") as raster:
        t = raster.transform
        raster.transform = Affine(t.a, t.b, t.c + 1, t.d, t.e, t.f)
    with pytest.raises(ValueError, match="grids do not align"):
        Sen1Floods11Index(root)


def test_overlap_is_rejected_even_across_regions(tmp_path):
    root = make_root(tmp_path)
    with (root / "splits/flood_handlabeled/flood_valid_data.csv").open("a") as handle:
        handle.write("Ghana_1_S1Hand.tif,Ghana_1_LabelHand.tif\n")
    with pytest.raises(ValueError, match="Tile id overlap"):
        Sen1Floods11Index(root)


def test_shared_regions_are_preserved_not_resplit(tmp_path):
    image = np.full((2, 2, 2), -15, dtype=np.float32)
    label = np.zeros((2, 2), dtype=np.int16)
    root = make_root(tmp_path, {split: [(f"Ghana_{i}", image, label)]
                               for i, split in enumerate(("train", "val", "test"))})
    index = Sen1Floods11Index(root)
    assert index.regions == {split: ["Ghana"] for split in ("train", "val", "test")}


@dataclass
class MemorySource:
    available_channels: tuple
    splits: dict
    sample: ChipSample

    def read(self, record):
        return self.sample


def memory_source(channels):
    return MemorySource(tuple(channels), {
        split: [ChipRecord(f"Valley_{i}", "Valley", split)]
        for i, split in enumerate(("train", "val", "test"))
    }, ChipSample(channels, np.ones((4, 4), dtype=np.int64), np.ones((4, 4), bool)))


def test_two_four_and_terrain_channel_sources(tmp_path):
    dataset = FloodChipDataset(make_root(tmp_path), split="val")
    assert dataset[0]["image"].shape == (2, 8, 8)
    names = ("pre_vv", "pre_vh", "post_vv", "post_vh", "elevation", "slope", "hand")
    source = memory_source({name: np.full((4, 4), i - 20, dtype=np.float32)
                            for i, name in enumerate(names)})
    four = FloodChipDataset(source=source, channels=names[:4], split="val")
    assert four[0]["image"].shape == (4, 4, 4)
    terrain = FloodChipDataset(source=source, channels=names, split="val")
    assert terrain[0]["image"].shape == (7, 4, 4)


@pytest.mark.parametrize("name", ["pre_vv", "pre_vh", "elevation", "slope", "hand"])
def test_unavailable_channels_raise(tmp_path, name):
    with pytest.raises(ValueError, match=f"{name}.*unavailable"):
        FloodChipDataset(make_root(tmp_path), channels=["post_vv", name])


def test_ratio_is_vv_minus_vh_in_db(tmp_path):
    index = Sen1Floods11Index(make_root(tmp_path))
    names = ("post_vv", "post_vh", "post_ratio")
    dataset = FloodChipDataset(index, split="val", add_ratio=True, stats=identity_stats(names))
    assert dataset.channels == list(names)
    assert torch.all(dataset[0]["image"][2] == 5)


def test_train_only_stats_clipping_and_reuse(tmp_path, monkeypatch):
    label = np.zeros((2, 2), dtype=np.int16)
    train = np.stack((np.array([[-50, -10], [-20, 50]]), np.full((2, 2), -25)))
    other = np.full((2, 2, 2), 100)
    root = make_root(tmp_path, {"train": [("Train_1", train, label)],
                               "val": [("Other_2", other, label)],
                               "test": [("Other_3", other, label)]})
    index = Sen1Floods11Index(root)
    stats = compute_train_stats(index)
    assert stats["channels"]["post_vv"]["mean"] == -15
    assert stats["channels"]["post_vv"]["std"] == pytest.approx(np.std([-35, -10, -20, 5]))
    assert stats["channels"]["post_vh"]["std"] == 1
    assert stats["training_ids"] == ["Train_1"]
    assert json.loads((root / "stats.json").read_text()) == stats
    write_pair(root, "Other_2", np.full((2, 2, 2), -100), label)
    monkeypatch.setattr(index, "read", lambda record: pytest.fail("cached stats must not read chips"))
    assert compute_train_stats(index) == stats


def test_label_nodata_invalid_sar_and_valid_mask(tmp_path):
    label = np.array([[-1, 0, 1], [-9999, 1, 0]], dtype=np.int16)
    image = np.full((2, 2, 3), -20, dtype=np.float32)
    image[0, 1, 1] = np.nan
    root = make_root(tmp_path, {"train": [("Ghana_1", image, label)]}, label_nodata=-9999)
    dataset = FloodChipDataset(root, crop_size=None, augment=False)
    result = dataset[0]
    assert result["label"].tolist() == [[255, 0, 1], [255, 255, 0]]
    assert result["valid_mask"].tolist() == [[False, True, True], [False, False, True]]
    assert torch.all(result["image"][:, ~result["valid_mask"]] == 0)
    assert torch.isfinite(result["image"]).all()


def test_conflicting_nodata_zero_is_honored_and_warned(tmp_path):
    label = np.array([[0, 1], [1, 0]])
    image = np.full((2, 2, 2), -20)
    index = Sen1Floods11Index(make_root(
        tmp_path, {"train": [("Ghana_1", image, label)]}, label_nodata=0,
    ))
    assert "conflicts with a class" in index.warnings[0]
    assert index.read(index.splits["train"][0]).label.tolist() == [[255, 1], [1, 255]]


def test_augmentation_keeps_asymmetric_image_label_and_mask_aligned():
    pattern = np.array([[0, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 0, 1]])
    valid = np.ones((4, 4), bool)
    valid[0, 0] = False
    source = memory_source({"post_vv": pattern.astype(np.float32),
                            "post_vh": pattern.astype(np.float32)})
    source.sample = ChipSample(source.sample.channels, pattern, valid)
    dataset = FloodChipDataset(source=source, crop_size=None, seed=4,
                              stats=identity_stats(source.available_channels))
    transformed = set()
    for _ in range(12):
        result = dataset[0]
        mask = result["valid_mask"]
        assert torch.equal(result["image"][0][mask].to(torch.int64), result["label"][mask])
        assert torch.equal(mask, result["label"] != 255)
        transformed.add(result["label"].numpy().tobytes())
    assert len(transformed) > 1


def test_train_crop_and_full_val_test(tmp_path):
    index = Sen1Floods11Index(make_root(tmp_path))
    assert FloodChipDataset(index, crop_size=4)[0]["image"].shape == (2, 4, 4)
    for split in ("val", "test", "bolivia"):
        dataset = FloodChipDataset(index, split=split, crop_size=4, augment=True)
        assert not dataset.augment
        assert dataset[0]["image"].shape == (2, 8, 8)
    with pytest.raises(ValueError, match="exceeds chip"):
        FloodChipDataset(index, crop_size=9)[0]


def test_class_balance_counts_full_chips_not_crops(tmp_path):
    label = np.array([[1, 1, 0], [0, -1, 0]], dtype=np.int16)
    image = np.full((2, 2, 3), -15)
    dataset = FloodChipDataset(make_root(tmp_path, {"train": [("Ghana_1", image, label)]}),
                               crop_size=1)
    balance = class_balance(dataset)
    assert balance["counts"] == {"water": 2, "non_water": 3, "ignore": 1}
    assert balance["fractions"] == pytest.approx({"water": 2 / 6, "non_water": 3 / 6, "ignore": 1 / 6})
    assert balance["per_region"]["Ghana"]["valid_water_fraction"] == 0.4
    assert balance["pos_weight"] == 1.5
    assert balance["class_weights"] == pytest.approx([5 / 6, 5 / 4])


def test_seeded_dataset_reproduces_crop_and_augmentation(tmp_path):
    index = Sen1Floods11Index(make_root(tmp_path))
    first = FloodChipDataset(index, crop_size=4, seed=42)
    second = FloodChipDataset(index, crop_size=4, seed=42)
    for _ in range(5):
        a, b = first[0], second[0]
        for key in ("image", "label", "valid_mask"):
            assert torch.equal(a[key], b[key])


def test_dataloaders_shapes_types_and_seed(tmp_path):
    root = make_root(tmp_path)
    first = make_dataloaders(root, batch_size=2, crop_size=4, seed=11)
    second = make_dataloaders(root, batch_size=2, crop_size=4, seed=11)
    a = next(iter(first["train"]))
    b = next(iter(second["train"]))
    assert a["image"].shape == (2, 2, 4, 4)
    assert a["label"].shape == a["valid_mask"].shape == (2, 4, 4)
    assert a["image"].dtype == torch.float32
    assert a["label"].dtype == torch.int64
    assert a["valid_mask"].dtype == torch.bool
    assert a["meta"]["id"] == b["meta"]["id"]
    assert torch.equal(a["image"], b["image"])
    assert next(iter(first["val"]))["image"].shape == (2, 2, 8, 8)
    assert first["train"].num_workers == 0


def test_inspection_cli_on_synthetic_data_writes_six_pngs(tmp_path):
    root = make_root(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/inspect_sen1floods11.py", "--root", str(root)],
        check=True, capture_output=True, text=True,
    )
    assert '"tile_id_overlap": false' in result.stdout
    assert '"bolivia"' in result.stdout
    assert (root / "stats.json").exists()
    previews = list((root / "previews").glob("*.png"))
    assert len(previews) == 6
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in previews)
