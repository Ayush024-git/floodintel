"""Synthetic optimizer steps, resumable artifacts, and optional radiometry only."""

from dataclasses import dataclass
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch
from torch.utils.data import DataLoader, Dataset

from src.model.data import ChipRecord, ChipSample, FloodChipDataset, make_dataloaders
from src.model.evaluate import evaluate_checkpoint
from src.model.losses import FloodLoss
from src.model.network import ModelConfig, build_model
from src.model.predict import load_checkpoint, read_checkpoint, select_device
from src.model.train import run_training, train_epoch, validate


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyChips(Dataset):
    def __init__(self, split="train", ignored=False):
        y, x = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
        self.label = (x >= 8).long()
        if ignored:
            self.label[:] = 255
        self.image = torch.stack(((x >= 8).float() * 2 - 1, y.float() / 16))
        self.split = split

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"image": self.image.clone(), "label": self.label.clone(),
                "valid_mask": self.label != 255,
                "meta": {"id": str(index), "region": "Bolivia" if self.split == "bolivia" else "Ghana",
                         "split": self.split}}


def tiny_loaders():
    return {split: DataLoader(TinyChips(split), batch_size=2, generator=torch.Generator().manual_seed(2))
            for split in ("train", "val", "test", "bolivia")}


def tiny_stats():
    return {"channels": {"post_vv": {"mean": -20., "std": 5.},
                         "post_vh": {"mean": -25., "std": 5.}}, "pos_weight": 1.0}


def tiny_config():
    return ModelConfig(backend="fallback", width=4, depth=1)


def test_synthetic_runner_two_steps_artifacts_and_resume(tmp_path):
    loaders = tiny_loaders()
    args = {"epochs": 2, "device": "cpu", "lr": .02, "warmup_epochs": 0,
            "seed": 2, "smoke": True, "max_batches": 1, "training_source": "synthetic"}
    report = run_training(loaders, tmp_path, tiny_config(), tiny_stats(), args)
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    assert set(report["splits"]) == {"val", "test", "bolivia"}
    assert "15 chips" in report["data_caveat"]
    assert report["threshold_selection_split"] == "val"
    assert report["splits"]["test"]["threshold"] == report["splits"]["bolivia"]["threshold"]
    payload = read_checkpoint(tmp_path / "last.pt")
    assert payload["epoch"] == 2
    assert payload["model_config"]["backend"] == "fallback"
    for key in ("pos_weight", "git_commit", "seed", "library_versions", "split_sizes", "rng_state"):
        assert key in payload
    run_training(tiny_loaders(), tmp_path, tiny_config(), tiny_stats(),
                 {**args, "epochs": 3, "resume": str(tmp_path / "last.pt")})
    assert read_checkpoint(tmp_path / "last.pt")["epoch"] == 3
    history = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert all(row["train"]["optimizer_steps"] == 1 for row in history)


def test_tiny_model_overfits_one_batch():
    torch.manual_seed(4)
    model = build_model(backend="fallback", width=4, depth=1)
    loader = DataLoader(TinyChips(), batch_size=2)
    loss_fn = FloodLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02)
    before = validate(model, loader, loss_fn=loss_fn)["loss"]["total"]
    for _ in range(12):
        train_epoch(model, loader, optimizer, loss_fn, max_batches=1)
    after = validate(model, loader, loss_fn=loss_fn)["loss"]["total"]
    assert after < before * .6


def test_ignore_batch_does_not_update_weights_or_decay():
    model = build_model(backend="fallback", width=4, depth=1)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=1)
    with pytest.raises(ValueError, match="No usable"):
        train_epoch(model, DataLoader(TinyChips(ignored=True), batch_size=2), optimizer, FloodLoss())
    assert all(torch.equal(value, original[key]) for key, value in model.state_dict().items())


@pytest.mark.parametrize("backend", ["fallback", "smp"])
def test_checkpoint_roundtrip_identical_and_cpu_forced(tmp_path, backend):
    if backend == "smp":
        from src.model.network import _import_smp
        try:
            _import_smp()
        except Exception as error:
            pytest.skip(f"Optional SMP backend unavailable: {error}")
    model = build_model(backend=backend, width=4, depth=1).eval()
    config = ModelConfig(backend=backend, width=4, depth=1)
    path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "model_config": config.to_dict(),
                "channels": config.channels, "normalization_stats": tiny_stats(),
                "best_threshold": .4}, path)
    loaded, meta = load_checkpoint(path, device="cpu")
    x = torch.randn(1, 2, 32, 32)
    assert torch.equal(model(x), loaded(x))
    assert meta["best_threshold"] == .4 and select_device("cpu").type == "cpu"


@dataclass
class Source:
    sample: ChipSample
    available_channels = ("post_vv", "post_vh")

    @property
    def splits(self):
        return {split: [ChipRecord(f"Ghana_{i}", "Ghana", split)]
                for i, split in enumerate(("train", "val", "test"))}

    def read(self, record):
        return self.sample


def radiometry_source():
    rng = np.random.default_rng(3)
    vv = (-20 + rng.normal(0, 3, (8, 8))).astype(np.float32)
    vv[0, 0] = np.nan
    return Source(ChipSample({"post_vv": vv, "post_vh": vv - 5},
                             np.zeros((8, 8), np.int64), np.ones((8, 8), bool)))


def raw_stats():
    return {"channels": {name: {"mean": 0, "std": 1} for name in Source.available_channels}}


def test_radiometry_defaults_bit_identical_and_gain_seeded_train_only():
    source = radiometry_source()
    options = {"source": source, "crop_size": 4, "stats": raw_stats(), "seed": 4}
    default = FloodChipDataset(**options)
    explicit = FloodChipDataset(**options, speckle_filter_prob=0, db_gain_jitter=0)
    for _ in range(3):
        a, b = default[0], explicit[0]
        assert all(torch.equal(a[key], b[key]) for key in ("image", "label", "valid_mask"))
    options.update(crop_size=None, augment=False)
    gain1 = FloodChipDataset(**options, db_gain_jitter=2)[0]
    gain2 = FloodChipDataset(**options, db_gain_jitter=2)[0]
    raw = FloodChipDataset(**options)[0]
    mask = raw["valid_mask"]
    delta = gain1["image"][:, mask] - raw["image"][:, mask]
    assert torch.allclose(delta, torch.full_like(delta, delta[0, 0]), atol=2e-6)
    assert 0 < abs(delta[0, 0]) <= 2
    assert torch.equal(gain1["image"], gain2["image"])
    for split in ("val", "test"):
        off = FloodChipDataset(**options, split=split)[0]
        on = FloodChipDataset(**options, split=split, speckle_filter_prob=1, db_gain_jitter=2)[0]
        assert torch.equal(off["image"], on["image"])


def test_speckle_reuses_preprocess_lee_and_is_seeded(monkeypatch):
    import src.data.s1_preprocess as preprocessing
    original = preprocessing.lee_filter
    calls = []

    def spy(power, window_size=5):
        calls.append(power.copy())
        assert window_size == 5
        return original(power, window_size)

    monkeypatch.setattr(preprocessing, "lee_filter", spy)
    source = radiometry_source()
    options = {"source": source, "crop_size": None, "augment": False,
               "stats": raw_stats(), "speckle_filter_prob": 1, "seed": 1}
    result = FloodChipDataset(**options)[0]
    again = FloodChipDataset(**options)[0]
    assert len(calls) == 4
    assert np.isnan(calls[0][0, 0])
    assert np.allclose(calls[0][1:], 10 ** (source.sample.channels["post_vv"][1:] / 10))
    assert torch.equal(result["image"], again["image"])
    assert result["image"][0][result["valid_mask"]].var() < torch.tensor(
        source.sample.channels["post_vv"][np.isfinite(source.sample.channels["post_vv"])]
    ).var()


def write_local_chips(root):
    directory = root / "splits/flood_handlabeled"
    directory.mkdir(parents=True)
    filenames = {"train": "train", "val": "valid", "test": "test", "bolivia": "bolivia"}
    label = np.zeros((16, 16), np.int16)
    label[:, 8:] = 1
    for i, (split, suffix) in enumerate(filenames.items()):
        chip = f"{'Bolivia' if split == 'bolivia' else 'Ghana'}_{i}"
        for folder, name, count, dtype, data in (
            ("S1Hand", "S1Hand", 2, "float32", np.stack((-20 + label * 5, -25 + label * 5))),
            ("LabelHand", "LabelHand", 1, "int16", label[None]),
        ):
            (root / folder).mkdir(exist_ok=True)
            with rasterio.open(root / folder / f"{chip}_{name}.tif", "w", driver="GTiff",
                               height=16, width=16, count=count, dtype=dtype, crs="EPSG:32645",
                               transform=from_origin(0, 160, 10, 10)) as dst:
                dst.write(data.astype(dtype))
        (directory / f"flood_{suffix}_data.csv").write_text(f"{chip}_S1Hand.tif,{chip}_LabelHand.tif\n")


def test_loader_options_and_shared_evaluation_on_synthetic_rasters(tmp_path):
    root = tmp_path / "data"
    write_local_chips(root)
    loaders = make_dataloaders(root, crop_size=8, batch_size=1,
                               speckle_filter_prob=.5, db_gain_jitter=2)
    assert loaders["train"].dataset.speckle_filter_prob == .5
    assert loaders["train"].dataset.db_gain_jitter == 2
    stats = loaders["train"].dataset.stats
    source = loaders["train"].dataset.source
    loaders["bolivia"] = DataLoader(FloodChipDataset(source=source, split="bolivia", stats=stats), batch_size=1)
    run_training(loaders, tmp_path / "run", tiny_config(), stats,
                 {"epochs": 1, "device": "cpu", "seed": 0, "training_source": "synthetic"})
    report = evaluate_checkpoint(tmp_path / "run/best.pt", root, "bolivia", device="cpu")
    assert report["split_size"] == 1
    assert list(report["metrics"]["per_region"]) == ["Bolivia"]
    saved = json.loads((tmp_path / "run/metrics.json").read_text())
    assert report["metrics"]["confusion"] == saved["splits"]["bolivia"]["confusion"]


def test_ratio_checkpoint_can_be_scored_without_new_stats(tmp_path, monkeypatch):
    root = tmp_path / "data"
    write_local_chips(root)
    loaders = make_dataloaders(root, crop_size=8, batch_size=1, add_ratio=True)
    stats = loaders["train"].dataset.stats
    config = ModelConfig(in_channels=3, channels=loaders["train"].dataset.channels,
                         backend="fallback", width=4, depth=1)
    model = build_model(in_channels=3, backend="fallback", width=4, depth=1)
    checkpoint = tmp_path / "ratio.pt"
    torch.save({"state_dict": model.state_dict(), "model_config": config.to_dict(),
                "channels": config.channels, "normalization_stats": stats,
                "best_threshold": .5, "pos_weight": 1}, checkpoint)
    import src.model.data as data
    monkeypatch.setattr(data, "compute_train_stats", lambda *a, **k: pytest.fail("Use checkpoint stats"))
    report = evaluate_checkpoint(checkpoint, root, split="test", device="cpu")
    assert report["metrics"]["valid_pixels"] == 256
