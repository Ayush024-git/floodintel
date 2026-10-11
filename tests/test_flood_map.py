"""Offline water-change, exclusion, persistence, and cached-incident contracts."""

from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import shape
import torch
from torch import nn

from src.analysis import flood_map as fm_module
from src.data.aoi import AOI
from src.data.incident import Incident, save_incident
from src.data.s1_preprocess import S1Stack
from src.data.terrain import TerrainStack
from src.model.network import ModelConfig
from src.model.predict import stack_to_model_input


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class WaterModel(nn.Module):
    """Pointwise water logit; no learned weights, downloads, or training."""

    def forward(self, image):
        return -image[:, :1] - 20


def db(probability):
    probability = np.asarray(probability, dtype=np.float32)
    return -20 - np.log(probability / (1 - probability))


def metadata():
    channels = ["post_vv", "post_vh"]
    config = ModelConfig(channels=channels, in_channels=2, backend="fallback", width=4, depth=1)
    return {"channels": channels, "model_config": config.to_dict(), "best_threshold": .5,
            "normalization_stats": {"clip_db": [-35., 5.], "channels": {
                channel: {"mean": 0., "std": 1.} for channel in channels}},
            "checkpoint_path": "synthetic.pt"}


def stacks(size=24):
    transform = from_origin(300000, 3100000, 10, 10)
    crs = CRS.from_epsg(32645)
    array = np.full((4, size, size), db(.05), np.float32)
    stack = S1Stack(array=array, valid_mask=np.ones((size, size), bool), transform=transform,
                    crs=crs, resolution=10, bbox=(85., 27.8, 85.01, 27.81),
                    band_names=("pre_vv", "pre_vh", "post_vv", "post_vh"), source="rtc",
                    pair_metadata={}, valid_fraction=1., warnings=[])
    zeros = np.zeros((size, size), np.float32)
    terrain = TerrainStack(elevation=zeros.copy(), slope_deg=zeros.copy(), hand_m=zeros.copy(),
                           stream_mask=np.zeros((size, size), bool), transform=transform, crs=crs,
                           bbox=stack.bbox, source="cop-dem-glo-30", hydro_resolution=30,
                           stream_threshold_km2=1., buffer_km=5, warnings=[])
    return stack, terrain


def water(stack, region, pre=.05, post=.95):
    for band, probability in ((0, pre), (1, pre), (2, post), (3, post)):
        stack.array[band][region] = db(probability)


def build(stack, terrain, **params):
    options = dict(tile=16, overlap=8, device="cpu", min_pixels=1, fill_hole_pixels=0)
    options.update(params)
    return fm_module.build_flood_map(stack, terrain, WaterModel(), metadata(), **options)


def codes(fm):
    return {flag["code"]: flag["severity"] for flag in fm.flags}


def test_pre_as_post_aliases_by_name_without_mutation():
    stack, _ = stacks()
    stack.array[0] = -28
    stack.array[1] = -23
    original = stack.array.copy()
    view = fm_module.pre_as_post(stack)
    image, valid = stack_to_model_input(view, metadata())
    assert np.all(image[0] == -28) and np.all(image[1] == -23)
    assert valid.all() and view.band_names == ("post_vv", "post_vh")
    view.array[:] = 0
    view.valid_mask[:] = False
    np.testing.assert_array_equal(stack.array, original)
    assert stack.valid_mask.all()


def test_preexisting_river_is_subtracted_and_new_water_retained():
    stack, terrain = stacks()
    river = np.s_[2:8, 2:5]
    flood = np.s_[2:8, 10:15]
    water(stack, river, pre=.95)
    water(stack, flood)
    result = build(stack, terrain)
    assert result.preexisting_water[river].all() and not result.flood_mask[river].any()
    assert result.flood_mask[flood].all()
    assert result.stats["preexisting_water_km2"] == pytest.approx(18 * .0001)
    assert result.params["threshold"] == result.params["pre_threshold"] == .5
    assert result.params["threshold_source"] == "checkpoint"
    assert not result.debris_candidate[result.flood_mask].any()


def test_threshold_overrides_and_reordered_stack_bands():
    stack, terrain = stacks()
    water(stack, (3, 4), pre=.6, post=.7)
    from dataclasses import replace
    stack = replace(stack, array=stack.array[[3, 0, 2, 1]],
                    band_names=("post_vh", "pre_vv", "post_vv", "pre_vh"))
    result = build(stack, terrain, threshold=.65, pre_threshold=.8)
    assert result.flood_mask[3, 4]
    assert result.params["threshold_source"] == result.params["pre_threshold_source"] == "override"


def test_terrain_exclusions_missing_hand_and_counts():
    stack, terrain = stacks()
    water(stack, np.s_[2:6, 2:6])
    terrain.hand_m[2, 2] = 16
    terrain.slope_deg[2, 3] = 21
    terrain.hand_m[2, 4] = np.nan
    terrain.slope_deg[2, 5] = np.nan
    result = build(stack, terrain)
    assert result.terrain_excluded.sum() == 4
    assert result.stats["terrain_excluded_km2"] == pytest.approx(.0004)
    assert result.stats["hand_missing_px"] == result.stats["slope_missing_px"] == 1
    assert not result.flood_mask[2, 2:6].any()
    assert result.hand_missing[2, 4]


def test_cleanup_removes_speck_and_reports_filled_hole():
    stack, terrain = stacks()
    water(stack, np.s_[4:11, 4:11])
    water(stack, (7, 7), post=.05)
    water(stack, (18, 18))
    result = build(stack, terrain, min_pixels=2, fill_hole_pixels=2)
    assert not result.flood_mask[18, 18] and result.cleanup_removed[18, 18]
    assert result.flood_mask[7, 7] and result.holes_filled[7, 7]
    assert result.confidence[7, 7] == 1
    assert result.stats["cleanup_removed_pixels"] == result.stats["holes_filled_pixels"] == 1
    assert result.stats["component_count"] == 1


@pytest.mark.parametrize("excluded", ["pre_water", "invalid", "high_hand", "missing_hand", "steep"])
def test_hole_filling_cannot_restore_protected_pixels(excluded):
    stack, terrain = stacks()
    water(stack, np.s_[4:11, 4:11])
    hole = (7, 7)
    if excluded == "pre_water":
        water(stack, hole, pre=.95)
    elif excluded == "invalid":
        stack.valid_mask[hole] = False
    elif excluded == "high_hand":
        terrain.hand_m[hole] = 100
    elif excluded == "missing_hand":
        terrain.hand_m[hole] = np.nan
    else:
        terrain.slope_deg[hole] = 45
    result = build(stack, terrain, fill_hole_pixels=2)
    assert not result.flood_mask[hole] and not result.holes_filled.any()
    if excluded != "invalid":
        assert result.hole_fill_blocked[hole]


def test_eight_connectivity_and_strict_hole_size():
    stack, terrain = stacks()
    water(stack, (3, 3))
    water(stack, (4, 4))
    result = build(stack, terrain, min_pixels=2)
    assert result.flood_mask.sum() == 2 and result.stats["component_count"] == 1
    water(stack, np.s_[10:15, 10:15])
    water(stack, (12, 12), post=.05)
    result = build(stack, terrain, min_pixels=2, fill_hole_pixels=1)
    assert not result.flood_mask[12, 12]


def test_confidence_tiers_and_pixel_area_math():
    stack, terrain = stacks()
    water(stack, (5, 5), pre=.1, post=.95)
    water(stack, (5, 6), pre=.1, post=.7)
    water(stack, (5, 7), pre=.4, post=.7)
    result = build(stack, terrain)
    np.testing.assert_array_equal(result.confidence[5, 5:8], [3, 2, 1])
    assert result.stats["flood_area_km2"] == pytest.approx(.0003)
    tiers = result.stats["area_per_confidence_tier_km2"]
    assert all(value == pytest.approx(.0001) for value in tiers.values())
    assert sum(tiers.values()) == pytest.approx(result.stats["flood_area_km2"])
    assert result.stats["valid_area_km2"] == pytest.approx(24 * 24 * .0001)
    assert result.stats["largest_component_km2"] == pytest.approx(.0003)
    assert not result.change_supports[5, 7]  # Change support is diagnostic, not a hard gate.


def test_vh_can_supply_darkening_evidence():
    stack, terrain = stacks()
    water(stack, (3, 3), pre=.4, post=.7)
    stack.array[3, 3, 3] = stack.array[1, 3, 3] - 4
    result = build(stack, terrain)
    assert result.change_db_vv[3, 3] > -3 and result.change_db_vh[3, 3] == -4
    assert result.change_supports[3, 3] and result.confidence[3, 3] == 2


def test_invalid_pixels_and_nan_probabilities_are_not_classified(monkeypatch):
    stack, terrain = stacks()
    water(stack, np.s_[2:6, 2:6])
    stack.valid_mask[3, 3] = False
    predictor = fm_module.predict_probabilities

    def with_nan(*args, **kwargs):
        result = predictor(*args, **kwargs)
        result[3, 4] = np.nan
        return result

    monkeypatch.setattr(fm_module, "predict_probabilities", with_nan)
    result = build(stack, terrain)
    assert result.stats["invalid_pixels"] == 2
    assert result.stats["valid_area_km2"] == pytest.approx((576 - 2) * .0001)
    for name in ("flood_mask", "preexisting_water", "terrain_excluded", "debris_candidate",
                 "water_post", "water_pre", "change_supports", "cleanup_removed", "holes_filled"):
        assert not getattr(result, name)[~result.valid_mask].any()
    assert np.isnan(result.p_post[~result.valid_mask]).all()


def test_debris_candidates_are_near_streams_separate_and_flagged():
    stack, terrain = stacks()
    terrain.stream_mask[:, 8] = True
    stack.array[3, 5:10, 9:11] += 5
    stack.array[3, 5:10, 19:21] += 5
    water(stack, np.s_[14:18, 9:11])
    result = build(stack, terrain, debris_buffer_m=20)
    assert result.debris_candidate[5:10, 9:11].all()
    assert not result.debris_candidate[:, 19:21].any()
    assert not (result.debris_candidate & result.flood_mask).any()
    assert result.stats["debris_candidate_km2"] == pytest.approx(.001)
    assert codes(result)["debris_is_candidate_only"] == "info"
    assert any("cannot confirm debris" in warning for warning in result.warnings)
    assert result.params["debris_slope_max_deg"] == 15
    terrain.slope_deg[5:10, 9:11] = 16
    assert not build(stack, terrain, debris_buffer_m=20).debris_candidate.any()


def test_required_flags_and_calibration_receipt():
    stack, terrain = stacks()
    water(stack, np.s_[1:5, 1:5], pre=.95)
    water(stack, np.s_[10:12, 10:12])
    terrain.hand_m[10:12, 10:12] = 25
    result = build(stack, terrain)
    expected = {"domain_gap": "caution", "probabilities_uncalibrated": "info", "terrain_heuristic": "info",
                "mountain_valley_hand_caution": "caution", "pre_post_water_overlap_high": "caution",
                "no_flood_detected": "info", "debris_is_candidate_only": "info"}
    assert codes(result) == expected
    meta = metadata()
    meta["metrics"] = {"val": {"calibration": {"ece": .1, "reliability": [{"count": 8}]}}}
    calibrated = fm_module.build_flood_map(stack, terrain, WaterModel(), meta,
                                          tile=16, overlap=8, min_pixels=1, fill_hole_pixels=0)
    assert "probabilities_uncalibrated" not in codes(calibrated)
    assert any("do not establish calibration" in warning for warning in calibrated.warnings)


@pytest.mark.parametrize("fraction,severity", [(.8, "caution"), (.5, "limiting")])
def test_low_valid_fraction_inherited(fraction, severity):
    from dataclasses import replace
    stack, terrain = stacks()
    stack = replace(stack, valid_fraction=fraction)
    assert codes(build(stack, terrain))["low_valid_fraction"] == severity


def test_all_zero_and_all_nan_inputs_are_honest(monkeypatch):
    stack, terrain = stacks()
    monkeypatch.setattr(fm_module, "predict_probabilities", lambda model, image, mask, **kwargs:
                        np.zeros(mask.shape, np.float32))
    empty = build(stack, terrain)
    assert not empty.flood_mask.any() and "no_flood_detected" in codes(empty)
    stack.array[:] = np.nan
    monkeypatch.setattr(fm_module, "predict_probabilities", Mock(side_effect=AssertionError("must skip empty input")))
    missing = build(stack, terrain)
    assert not missing.valid_mask.any() and not missing.flood_mask.any()
    assert missing.stats["valid_area_km2"] == 0 and missing.stats["flood_fraction_of_valid"] is None
    assert codes(missing)["no_valid_pixels"] == "limiting"
    json.dumps(missing.to_dict(), allow_nan=False)


def test_numpy_morphology_fallback_matches_scipy(monkeypatch):
    stack, terrain = stacks(16)
    water(stack, np.s_[3:9, 3:9])
    water(stack, (5, 5), post=.05)
    water(stack, (13, 13))
    terrain.stream_mask[:, 10] = True
    stack.array[3, 11:14, 10:12] += 5
    reference = build(stack, terrain, min_pixels=2, fill_hole_pixels=2, debris_buffer_m=20)
    monkeypatch.setattr(fm_module, "ndimage", None)
    fallback = build(stack, terrain, min_pixels=2, fill_hole_pixels=2, debris_buffer_m=20)
    for name in ("flood_mask", "holes_filled", "cleanup_removed", "debris_candidate", "confidence"):
        np.testing.assert_array_equal(getattr(reference, name), getattr(fallback, name))
    assert "numpy_morphology_fallback" in codes(fallback)


@pytest.mark.parametrize("params", [{"threshold": 1.1}, {"change_db_min": 0}, {"min_pixels": -1},
                                    {"debris_slope_min_deg": 30}, {"tile": 2, "overlap": 3}])
def test_invalid_parameters_raise(params):
    stack, terrain = stacks()
    with pytest.raises(ValueError):
        build(stack, terrain, **params)


def test_grid_and_four_band_model_are_rejected():
    from dataclasses import replace
    stack, terrain = stacks()
    shifted = replace(terrain, transform=from_origin(300005, 3100000, 10, 10))
    with pytest.raises(ValueError, match="exact S1"):
        build(stack, shifted)
    with pytest.raises(ValueError, match="two-band"):
        fm_module.build_flood_map(stack, terrain, WaterModel(),
                                  {**metadata(), "channels": list(stack.band_names)})


def test_save_load_roundtrip_and_valid_geojson(tmp_path):
    stack, terrain = stacks()
    water(stack, np.s_[3:9, 3:9])
    water(stack, (12, 12))
    water(stack, (13, 13))  # Diagonal contacts must still produce valid vector geometry.
    terrain.stream_mask[:, 16] = True
    stack.array[3, 15:18, 16:18] += 5
    result = build(stack, terrain)
    path = fm_module.save_flood_map(result, tmp_path)
    restored = fm_module.load_flood_map(path)
    for name in fm_module.FLOAT_BANDS + fm_module.MASK_BANDS:
        np.testing.assert_array_equal(getattr(result, name), getattr(restored, name))
    assert restored.stats == result.stats and restored.params == result.params
    assert restored.flags == result.flags and restored.warnings == result.warnings
    assert restored.transform == stack.transform and restored.crs == stack.crs
    assert restored.confidence.dtype == np.uint8 and restored.flood_mask.dtype == bool
    with rasterio.open(tmp_path / "flood_masks.tif") as dataset:
        assert set(dataset.dtypes) == {"uint8"} and dataset.descriptions == fm_module.MASK_BANDS
    for filename in ("flood_extent.geojson", "debris_candidates.geojson"):
        collection = json.loads((tmp_path / filename).read_text())
        assert collection["type"] == "FeatureCollection" and collection["features"]
        for feature in collection["features"]:
            assert shape(feature["geometry"]).is_valid
            assert feature["properties"]["area_km2"] > 0
            assert 84 < shape(feature["geometry"]).centroid.x < 90
            assert "max_confidence" in feature["properties"] and "median_confidence" in feature["properties"]
    assert (tmp_path / "quicklook.png").read_bytes().startswith(b"\x89PNG")


def test_vector_feature_cap_is_auditable(tmp_path):
    stack, terrain = stacks()
    for pixel in ((3, 3), (9, 9), (15, 15)):
        water(stack, pixel)
    result = build(stack, terrain, max_vector_features=2)
    fm_module.save_flood_map(result, tmp_path)
    assert result.flood_mask.sum() == 3 and result.flood_vector_omitted.sum() == 1
    assert result.stats["flood_vector_omitted_components"] == 1
    assert "flood_vector_feature_cap" in codes(result)
    assert len(json.loads((tmp_path / "flood_extent.geojson").read_text())["features"]) == 2
    assert fm_module.load_flood_map(tmp_path).flood_vector_omitted.sum() == 1


def incident(path, status="ready"):
    aoi = AOI("Trishuli, Nepal", (85., 27.8, 85.01, 27.81), date(2026, 8, 10),
              (date(2026, 7, 10), date(2026, 8, 9)), (date(2026, 8, 10), date(2026, 8, 22)), 1., [])
    return Incident("inc_123456789abc", "2026-10-11T00:00:00Z", status, aoi.to_dict(),
                    {"reason": "no same-orbit pair"} if status == "blocked" else {},
                    {"s1_stack": "s1.tif", "terrain": "terrain.tif"}, [], {"trust_summary": "strong", "flags": []},
                    {"request": {"place": "Trishuli, Nepal"}}, 0., str(path))


def test_blocked_or_missing_s1_incident_fails_before_checkpoint(monkeypatch, tmp_path):
    import src.data.incident as incident_module
    blocked = incident(tmp_path / "incident.json", "blocked")
    monkeypatch.setattr(incident_module, "load_incident", lambda *args, **kwargs: blocked)
    with pytest.raises(ValueError, match="blocked.*no same-orbit pair"):
        fm_module.analyze_incident_flood(blocked.incident_id, tmp_path / "absent.pt")
    blocked.status = "partial"
    blocked.layers = {}
    with pytest.raises(ValueError, match="no S1 stack"):
        fm_module.analyze_incident_flood(blocked.incident_id, tmp_path / "absent.pt")


def test_incident_cache_reuses_results_and_keys_include_threshold(monkeypatch, tmp_path):
    import src.data.incident as incident_module
    stack, terrain = stacks()
    water(stack, np.s_[3:8, 3:8])
    receipt = incident(tmp_path / "incident.json")
    monkeypatch.setattr(incident_module, "load_incident", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(incident_module, "load_layers", lambda _: SimpleNamespace(s1_stack=stack, terrain=terrain))
    loader = Mock(return_value=(WaterModel(), metadata()))
    monkeypatch.setattr(fm_module, "load_checkpoint", loader)
    checkpoint = tmp_path / "synthetic.pt"
    checkpoint.write_bytes(b"synthetic checkpoint: not used by real loader")
    params = dict(tile=16, overlap=8, min_pixels=1, fill_hole_pixels=0, device="cpu")
    first = fm_module.analyze_incident_flood(receipt.incident_id, checkpoint, **params)
    assert Path(first.cache_path).is_relative_to(tmp_path / "flood")
    loader.reset_mock()
    builder = fm_module.build_flood_map
    forbidden = Mock(side_effect=AssertionError("cache must skip predictions"))
    monkeypatch.setattr(fm_module, "build_flood_map", forbidden)
    second = fm_module.analyze_incident_flood(receipt.incident_id, checkpoint, **params)
    assert first.cache_path == second.cache_path
    np.testing.assert_array_equal(first.flood_mask, second.flood_mask)
    loader.assert_not_called()
    forbidden.assert_not_called()
    monkeypatch.setattr(fm_module, "build_flood_map", builder)
    third = fm_module.analyze_incident_flood(receipt.incident_id, checkpoint, threshold=.6, **params)
    assert third.cache_path != first.cache_path
    assert fm_module.flood_cache_key("one", {"threshold": .5}) != fm_module.flood_cache_key("one", {"threshold": .6})
    checkpoint.write_bytes(b"changed checkpoint")
    fourth = fm_module.analyze_incident_flood(receipt.incident_id, checkpoint, **params)
    assert fourth.cache_path != first.cache_path


def test_cached_place_resolution_never_geocodes_or_prepares(monkeypatch, tmp_path):
    import src.data.aoi as aoi_module
    import src.data.incident as incident_module
    receipt = incident(tmp_path / "incidents" / "inc_123456789abc" / "incident.json")
    Path(receipt.cache_path).parent.mkdir(parents=True)
    save_incident(receipt)
    forbidden = Mock(side_effect=AssertionError("no fetching allowed"))
    monkeypatch.setattr(aoi_module, "geocode_place", forbidden)
    monkeypatch.setattr(incident_module, "prepare_incident", forbidden)
    resolved = fm_module.resolve_cached_incident("trishuli, nepal", "2026-08-10", tmp_path)
    assert resolved.incident_id == receipt.incident_id
    with pytest.raises(FileNotFoundError, match="run prepare_incident first"):
        fm_module.resolve_cached_incident("Different place", "2026-08-10", tmp_path)
    forbidden.assert_not_called()


def test_cli_missing_cached_place_has_clear_message(monkeypatch, capsys, tmp_path):
    import sys
    monkeypatch.setattr(sys, "argv", ["flood_map", "--place", "Trishuli, Nepal", "--date", "2026-08-10",
                                     "--checkpoint", "runs/smoke/best.pt", "--cache-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        fm_module.main()
    assert error.value.code == 2
    assert "run prepare_incident first" in capsys.readouterr().err
