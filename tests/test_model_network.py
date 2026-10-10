"""Offline architecture, masked-loss, and probability-metric contracts."""

import importlib.metadata
import json

import numpy as np
import pytest
import torch

import src.model.network as network
from src.model.losses import FloodLoss
from src.model.metrics import MetricAccumulator, calibration, confusion_metrics, threshold_sweep


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("channels", [2, 4])
def test_build_model_logits_shape_no_weights(channels):
    model = network.build_model(in_channels=channels, encoder_weights=None).eval()
    with torch.no_grad():
        assert model(torch.randn(1, channels, 32, 32)).shape == (1, 1, 32, 32)


def test_machine_backend_report():
    model = network.build_model(encoder_weights=None)
    count = sum(parameter.numel() for parameter in model.parameters())
    print(f"STEP11_ENV torch={torch.__version__} backend={model.backend} parameters={count} "
          f"cuda={torch.cuda.is_available()} mps={torch.backends.mps.is_available()} "
          f"reason={model.backend_warning!r}")
    try:
        print("SMP distribution:", importlib.metadata.version("segmentation-models-pytorch"))
    except importlib.metadata.PackageNotFoundError:
        print("SMP distribution: not installed")
    assert count > 0


def test_config_json_roundtrip():
    config = network.ModelConfig(in_channels=4, channels=["pre_vv", "pre_vh", "post_vv", "post_vh"])
    assert network.ModelConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(ValueError, match="in_channels"):
        network.ModelConfig(in_channels=4)


def test_import_failure_uses_explicit_fallback(monkeypatch):
    def unavailable():
        raise ImportError("synthetic missing SMP")
    monkeypatch.setattr(network, "_import_smp", unavailable)
    with pytest.warns(RuntimeWarning, match="using compact"):
        model = network.build_model(in_channels=4, depth=2, width=4)
    assert model.backend == "fallback"
    assert model(torch.zeros(2, 4, 19, 23)).shape == (2, 1, 19, 23)
    with pytest.raises(RuntimeError, match="Checkpoint requires"):
        network.build_model(backend="smp")


@pytest.mark.parametrize("focal", [False, True])
def test_ignored_logits_have_zero_loss_and_gradient(focal):
    loss = FloodLoss(pos_weight=2, focal=focal)
    labels = torch.tensor([[[1, 0], [255, 255]]])
    logits = torch.tensor([[[[.2, -.3], [100, -100]]]], requires_grad=True)
    changed = logits.detach().clone()
    changed[0, 0, 1] = torch.tensor([float("nan"), float("inf")])
    original = loss(logits, labels)
    altered = loss(changed, labels)
    for key in original:
        assert torch.equal(original[key], altered[key])
    original["total"].backward()
    assert torch.all(logits.grad[0, 0, 1] == 0)


def test_all_ignore_is_differentiable_zero():
    logits = torch.full((1, 1, 2, 2), float("nan"), requires_grad=True)
    components = FloodLoss()(logits, torch.full((1, 2, 2), 255))
    assert components["total"].item() == 0
    components["total"].backward()
    assert torch.all(logits.grad == 0)


def test_loss_improves_and_applies_pos_weight():
    labels = torch.tensor([[[1, 0]]])
    bad = torch.tensor([[[[-2., 2.]]]])
    good = -bad
    loss = FloodLoss(pos_weight=3)
    assert loss(good, labels)["total"] < loss(bad, labels)["total"]
    one = FloodLoss(pos_weight=1, dice_weight=0)(torch.zeros_like(bad), labels)["bce"]
    three = FloodLoss(stats={"class_balance": {"pos_weight": 3}}, dice_weight=0)(
        torch.zeros_like(bad), labels,
    )["bce"]
    assert three.item() == pytest.approx(2 * one.item())


def test_confusion_ignore_and_streaming_regions():
    labels = np.array([[[1, 1, 0, 255]], [[1, 0, 0, 255]]])
    probabilities = np.array([[[.9, .1, .8, np.nan]], [[.9, .1, .1, np.inf]]])
    metrics = MetricAccumulator()
    metrics.update(probabilities[:1], labels[:1], {"region": ["Ghana"]})
    metrics.update(probabilities[1:], labels[1:], {"region": ["Bolivia"]})
    result = metrics.compute()
    assert result["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    assert result["iou"] == .5
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["precision"] == result["recall"] == result["accuracy"] == pytest.approx(2 / 3)
    assert result["valid_pixels"] == 6 and result["ignored_pixels"] == 2
    assert result["per_region"]["Bolivia"]["iou"] == 1
    assert result["per_region"]["Ghana"]["iou"] == pytest.approx(1 / 3)
    assert confusion_metrics([0, 0, 0, 0])["iou"] is None


def test_calibration_and_threshold_selection_known_cases():
    reliability = calibration(np.array([[.1, .9, .8]]), np.array([[0, 1, 255]]))
    assert reliability["ece"] == pytest.approx(.1)
    assert sum(bin["count"] for bin in reliability["reliability"]) == 2
    sweep = threshold_sweep(np.array([[.2, .4, .7]]), np.array([[0, 1, 1]]), [.3, .5, .8])
    assert sweep["best_threshold"] == .3
    assert sweep["sweep"][0]["f1"] == 1
    perfect = calibration(np.array([[0., 1.]]), np.array([[0, 1]]))
    assert perfect["ece"] == 0
