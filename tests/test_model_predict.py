"""Named-band mapping and memory-bounded, seam-free local convolution inference."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.model.network import ModelConfig, build_model
from src.model.predict import predict_incident, predict_probabilities, stack_to_model_input


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def stack():
    names = ("post_vh", "pre_vv", "post_vv", "pre_vh")
    array = np.stack([np.full((16, 16), value, np.float32) for value in (-25, -10, -20, -15)])
    array[2, 0, 0] = np.nan
    valid = np.ones((16, 16), bool)
    valid[1, 1] = False
    return SimpleNamespace(array=array, band_names=names, valid_mask=valid,
                           warnings=[], source="rtc", valid_fraction=float(valid.mean()))


def metadata(names):
    config = ModelConfig(in_channels=len(names), channels=names, backend="fallback", width=4, depth=1)
    return {"model_config": config.to_dict(), "channels": names,
            "normalization_stats": {"clip_db": [-35., 5.], "channels": {
                name: {"mean": -20, "std": 5} for name in names}},
            "best_threshold": .4, "training_source": "sen1floods11"}


@pytest.mark.parametrize("names,expected", [
    (["post_vv", "post_vh"], [0, -1]),
    (["pre_vv", "pre_vh", "post_vv", "post_vh"], [2, 1, 0, -1]),
])
def test_band_selection_normalization_and_validity(names, expected):
    data = stack()
    image, valid = stack_to_model_input(data, metadata(names))
    assert image.dtype == np.float32 and valid.dtype == bool
    assert np.allclose(image[:, 2, 2], expected)
    assert not valid[0, 0] and not valid[1, 1]
    assert np.all(image[:, ~valid] == 0)
    assert image.shape == (len(names), 16, 16)


def test_clipping_ratios_and_missing_channels():
    data = stack()
    data.array[2, 2, 2] = 30
    names = ["post_vv", "post_vh", "post_ratio"]
    meta = metadata(names)
    meta["normalization_stats"]["channels"]["post_ratio"] = {"mean": 0, "std": 1}
    image, valid = stack_to_model_input(data, meta)
    assert image[0, 2, 2] == 5
    assert image[2, 2, 2] == 30
    assert valid[2, 2]
    with pytest.raises(ValueError, match="hand.*unavailable"):
        stack_to_model_input(data, metadata(["post_vv", "hand"]))


class LocalConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 3, padding=1, padding_mode="reflect", bias=False)
        with torch.no_grad():
            self.conv.weight[:] = torch.arange(18).reshape(1, 2, 3, 3) / 100

    def forward(self, value):
        return self.conv(value)


def test_tiling_matches_full_local_conv_and_tile_borders():
    rng = np.random.default_rng(2)
    image = rng.normal(size=(2, 113, 127)).astype(np.float32)
    mask = np.ones((113, 127), bool)
    mask[4:9, 8:12] = False
    model = LocalConv().eval()
    with torch.no_grad():
        reference = torch.sigmoid(model(torch.from_numpy(image[None])))[0, 0].numpy()
    tiled = predict_probabilities(model, image, mask, tile=32, overlap=16, batch_size=3)
    assert tiled.shape == mask.shape and tiled.dtype == np.float32
    np.testing.assert_allclose(tiled[mask], reference[mask], atol=2e-7, rtol=1e-6)
    for border in range(16, 113, 16):
        np.testing.assert_allclose(tiled[border], reference[border], atol=2e-7, rtol=1e-6)
    assert np.isnan(tiled[~mask]).all()
    repeated = predict_probabilities(model, image, mask, tile=32, overlap=16, batch_size=3)
    np.testing.assert_array_equal(tiled, repeated)


def test_small_reflected_edges_and_tta_determinism():
    image = np.arange(30, dtype=np.float32).reshape(2, 3, 5) / 10
    model = nn.Conv2d(2, 1, 1).eval()
    mask = np.ones((3, 5), bool)
    plain = predict_probabilities(model, image, mask, tile=16, overlap=8)
    tta = predict_probabilities(model, image, mask, tile=16, overlap=8, tta=True)
    np.testing.assert_allclose(plain, tta, atol=1e-7)
    assert np.isfinite(tta).all()


def test_large_grid_uses_bounded_batches():
    class Pointwise(nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def forward(self, value):
            self.shapes.append(tuple(value.shape))
            return value[:, :1]

    model = Pointwise()
    image = np.full((2, 2048, 2048), .25, np.float32)
    mask = np.ones((2048, 2048), bool)
    probability = predict_probabilities(model, image, mask, tile=256, overlap=64, batch_size=2)
    assert probability.shape == mask.shape
    np.testing.assert_allclose(probability, 1 / (1 + np.exp(-.25)), atol=2e-7)
    assert max(shape[0] for shape in model.shapes) <= 2
    assert max(shape[-1] for shape in model.shapes) == 320


def test_predict_incident_metadata_warnings_and_png(tmp_path):
    data = stack()
    data.valid_mask[:8] = False
    meta = metadata(["post_vv", "post_vh"])
    meta["warnings"] = ["diagnostic checkpoint only"]
    model = build_model(backend="fallback", width=4, depth=1).eval()
    path = tmp_path / "checkpoint.pt"
    torch.save({**meta, "state_dict": model.state_dict()}, path)
    output, valid, receipt = predict_incident(data, path, device="cpu", tile=16, overlap=8,
                                             quicklook=tmp_path / "preview.png")
    assert output.shape == valid.shape == (16, 16)
    assert np.isnan(output[~valid]).all()
    assert receipt["threshold"] == .4
    assert any("valid_fraction" in warning for warning in receipt["warnings"])
    assert any("Model trained on" in warning for warning in receipt["warnings"])
    assert any("domain gap" in warning for warning in receipt["warnings"])
    assert "diagnostic checkpoint only" in receipt["warnings"]
    assert (tmp_path / "preview.png").read_bytes().startswith(b"\x89PNG")
