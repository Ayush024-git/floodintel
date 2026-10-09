"""Offline DEM mosaicking, D8 drainage, grid alignment and cache checks."""

from datetime import date
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import array_bounds, from_origin

from src.data.aoi import build_aoi
from src.data.s1_preprocess import S1Stack, make_grid
from src.data import terrain as t


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Terrain tests must not open a live catalog")

    monkeypatch.setattr(t.Client, "open", unexpected)


@pytest.fixture
def aoi():
    return build_aoi(bbox=(85, 27.9, 85.003, 27.903), flood_date=date(2020, 8, 10))


@pytest.fixture
def stack(aoi):
    grid = make_grid(aoi)
    return S1Stack(
        array=np.zeros((4, grid.height, grid.width), dtype="float32"),
        valid_mask=np.ones((grid.height, grid.width), dtype=bool),
        transform=grid.transform, crs=grid.crs, resolution=10, bbox=aoi.bbox,
        band_names=("pre_vv", "pre_vh", "post_vv", "post_vh"), source="rtc",
        pair_metadata={}, valid_fraction=1, warnings=[],
    )


def valley(height=31, width=21):
    rows, cols = np.indices((height, width))
    return (np.abs(cols - width // 2) * 5 + (height - 1 - rows) * 0.1).astype("float32")


def metric_dem(stack, padding=3, shifted=True):
    west, south, east, north = array_bounds(*stack.array.shape[-2:], stack.transform)
    left = math.floor(west / 30) * 30 - padding * 30
    top = math.ceil(north / 30) * 30 + padding * 30
    width = math.ceil((east - left) / 30) + padding
    height = math.ceil((top - south) / 30) + padding
    affine = from_origin(left + (7 if shifted else 0), top + (11 if shifted else 0), 30, 30)
    return t.DEMData(valley(height, width), affine, stack.crs, [], ["synthetic"])


@pytest.fixture
def fake_dem(stack, monkeypatch):
    dem = metric_dem(stack)
    monkeypatch.setattr(t, "fetch_dem", lambda *args, **kwargs: dem)
    return dem


def test_v_valley_hand_and_channel():
    dem = valley()
    result = t.compute_hand(dem, 30, stream_threshold_km2=0.012)
    assert result.stream_mask[:, 10].all()
    assert not result.stream_mask[:, :10].any()
    assert not result.stream_mask[:, 11:].any()
    np.testing.assert_allclose(result.hand_m[:, 10], 0)
    np.testing.assert_allclose(result.hand_m[:, 8], 10, atol=1e-5)
    np.testing.assert_allclose(result.hand_m[:, 6], 20, atol=1e-5)
    assert result.hand_m.dtype == np.float32
    assert result.stream_mask.dtype == bool


def test_slope_of_known_tilted_plane():
    rows, cols = np.indices((9, 11))
    dem = 100 + 0.1 * cols * 30 + 0.2 * rows * 30
    expected = math.degrees(math.atan(math.hypot(0.1, 0.2)))
    np.testing.assert_allclose(t.compute_slope(dem, 30), expected, atol=1e-5)
    dem = dem.astype("float32")
    dem[4, 5] = np.nan
    assert np.isnan(t.compute_slope(dem, 30)[4, 5])


def test_priority_flood_fills_pit_without_trapping_flow():
    dem = valley()
    dem[15, 6] -= 100
    filled, _ = t._priority_flood(dem)
    assert filled[15, 6] > dem[15, 6] + 90
    result = t.compute_hand(dem, 30, stream_threshold_km2=0.02)
    assert not result.stream_mask[15, 6]
    assert np.isfinite(result.hand_m[15, 6])
    assert any("depression pixels" in warning for warning in result.warnings)
    assert any("negative HAND" in warning for warning in result.warnings)


def test_flat_plateau_routes_to_an_outlet():
    dem = np.full((15, 15), 10, dtype="float32")
    dem[-1, 7] = 9
    result = t.compute_hand(dem, 30, stream_threshold_km2=0.01)
    assert result.stream_mask[-1, 7]
    assert np.isfinite(result.hand_m[7, 7])
    assert 0 <= result.hand_m[7, 7] <= 1


class FakeClient:
    def __init__(self, items):
        self.items = items
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(items=lambda: iter(self.items))


def write_tile(path, array, affine, nodata=-9999):
    with rasterio.open(
        path, "w", driver="GTiff", width=array.shape[1], height=array.shape[0],
        count=1, dtype="float32", transform=affine, crs="EPSG:4326", nodata=nodata,
    ) as dst:
        dst.write(array.astype("float32"), 1)
    return str(path)


def test_tile_mosaic_is_continuous_and_windowed(tmp_path):
    rows, cols = np.indices((7, 8))
    values = (100 + rows + cols * 10).astype("float32")
    left = write_tile(tmp_path / "left.tif", values[:, :4], from_origin(85, 28.005, 0.001, 0.001))
    right = write_tile(tmp_path / "right.tif", values[:, 4:], from_origin(85.004, 28.005, 0.001, 0.001))
    client = FakeClient([
        {"id": "left", "assets": {"data": {"href": left}}},
        {"id": "right", "assets": {"data": {"href": right}}},
    ])
    bbox = (85.00125, 27.99925, 85.00675, 28.00375)
    result = t.fetch_dem(bbox, 0, client)
    assert result.elevation.shape == (5, 6)
    np.testing.assert_allclose(result.elevation, values[1:6, 1:7])
    assert result.crs == CRS.from_epsg(4326)
    assert result.tile_ids == ["left", "right"]
    assert client.calls == [{"collections": ["cop-dem-glo-30"], "bbox": list(bbox)}]


def test_zero_elevation_is_valid_without_declared_nodata(tmp_path):
    href = write_tile(tmp_path / "sea.tif", np.zeros((5, 5)), from_origin(85, 28.005, 0.001, 0.001), nodata=None)
    client = FakeClient([{"id": "sea", "assets": {"data": {"href": href}}}])
    result = t.fetch_dem((85.0001, 28.0001, 85.0049, 28.0049), 0, client)
    assert np.isfinite(result.elevation).all()
    np.testing.assert_array_equal(result.elevation, 0)


def test_small_holes_interpolate_but_large_and_edge_holes_remain():
    rows, cols = np.indices((15, 15))
    dem = (rows * 2 + cols).astype("float32")
    expected = dem[4, 4]
    dem[4, 4] = np.nan
    dem[8:13, 8:13] = np.nan
    dem[0, 0] = np.nan
    filled, warnings = t.fill_small_holes(dem)
    assert filled[4, 4] == pytest.approx(expected)
    assert np.isnan(filled[8:13, 8:13]).all()
    assert np.isnan(filled[0, 0])
    assert any("Interpolated 1 DEM" in warning for warning in warnings)
    assert any("26 DEM pixels remain missing" in warning for warning in warnings)


def test_expansion_and_buffered_search(tmp_path):
    bbox = (85.001, 28.001, 85.002, 28.002)
    expanded = t.expanded_bbox(bbox, 0.1)
    assert expanded[0] < bbox[0] < bbox[2] < expanded[2]
    assert expanded[1] < bbox[1] < bbox[3] < expanded[3]
    href = write_tile(tmp_path / "buffered.tif", np.ones((20, 20)), from_origin(84.995, 28.01, 0.001, 0.001))
    client = FakeClient([{"id": "wide", "assets": {"data": {"href": href}}}])
    t.fetch_dem(bbox, 0.1, client)
    assert client.calls[0]["bbox"] == list(expanded)


def test_output_matches_exact_s1_grid(tmp_path, aoi, stack, fake_dem, monkeypatch):
    resolutions = []
    original = t.compute_hand

    def capture(dem, resolution, threshold):
        resolutions.append(resolution)
        return original(dem, resolution, threshold)

    monkeypatch.setattr(t, "compute_hand", capture)
    result = t.build_terrain(aoi, stack, tmp_path, stream_threshold_km2=0.005)
    assert result.transform == stack.transform
    assert result.crs == stack.crs
    assert result.bbox == stack.bbox
    for array in (result.elevation, result.slope_deg, result.hand_m, result.stream_mask):
        assert array.shape == stack.array.shape[-2:]
    assert result.elevation.dtype == result.slope_deg.dtype == result.hand_m.dtype == np.float32
    assert result.stream_mask.dtype == bool
    assert resolutions == [30]
    assert any("native ~30 m" in warning and "no terrain detail" in warning for warning in result.warnings)
    assert t.SURFACE_WARNING in result.warnings
    assert t.BACKEND_WARNING in result.warnings


def test_buffer_preserves_incoming_channel_at_crop_edge(tmp_path, aoi, stack, monkeypatch):
    west, south, east, north = array_bounds(*stack.array.shape[-2:], stack.transform)
    channel_x = math.floor(((west + east) / 2) / 30) * 30 + 15

    def fetch(bbox, buffer_km, **kwargs):
        padding = 10 if buffer_km else 0
        left = math.floor(west / 30) * 30 - padding * 30
        top = math.ceil(north / 30) * 30 + padding * 30
        width = math.ceil((east - left) / 30) + padding
        height = math.ceil((top - south) / 30) + padding
        rows, cols = np.indices((height, width))
        xs, ys = left + (cols + 0.5) * 30, top - (rows + 0.5) * 30
        dem = 100 + np.abs(xs - channel_x) * 0.3 + (ys - south) * 0.05
        return t.DEMData(dem.astype("float32"), from_origin(left, top, 30, 30), stack.crs, [])

    monkeypatch.setattr(t, "fetch_dem", fetch)
    buffered = t.build_terrain(aoi, stack, tmp_path, buffer_km=0.3, stream_threshold_km2=0.08)
    unbuffered = t.build_terrain(aoi, stack, tmp_path, buffer_km=0, stream_threshold_km2=0.08)
    col = round((channel_x - west) / 10 - 0.5)
    assert buffered.hand_m[0, col] == pytest.approx(0)
    assert buffered.stream_mask[0, col]
    assert unbuffered.hand_m[0, col] > 1
    assert not unbuffered.stream_mask[0, col]


def test_unreached_hand_is_nan_and_counted():
    dem = valley(9, 9)
    dem[2, 2] = np.nan
    result = t.compute_hand(dem, 30, stream_threshold_km2=100)
    assert np.isnan(result.hand_m).all()
    assert not result.stream_mask.any()
    assert any("80 valid hydrology pixels never reach" in warning for warning in result.warnings)
    assert any("1 missing/outside-coverage" in warning for warning in result.warnings)


def test_output_nan_hand_stats_and_warning(tmp_path, aoi, stack, fake_dem):
    result = t.build_terrain(aoi, stack, tmp_path, stream_threshold_km2=100)
    assert result.to_dict()["hand_nan_fraction"] == 1
    assert result.to_dict()["stats"]["hand_m"] == {"min": None, "max": None, "mean": None}
    assert any(f"HAND is NaN for {result.hand_m.size}/{result.hand_m.size}" in warning for warning in result.warnings)
    json.dumps(result.to_dict(), allow_nan=False)


def test_cache_skips_fetch_and_hydrology(tmp_path, aoi, stack, fake_dem, monkeypatch):
    first = t.build_terrain(aoi, stack, tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("Cache hit must skip all fetching and hydrology")

    monkeypatch.setattr(t, "fetch_dem", unexpected)
    monkeypatch.setattr(t, "compute_hand", unexpected)
    second = t.build_terrain(aoi, stack, tmp_path)
    assert second.cache_path == first.cache_path
    np.testing.assert_array_equal(second.hand_m, first.hand_m)
    assert second.to_dict() == first.to_dict()


def test_cache_key_changes_with_threshold_buffer_and_grid(tmp_path, aoi, stack, fake_dem):
    from dataclasses import replace

    first = t.build_terrain(aoi, stack, tmp_path)
    threshold = t.build_terrain(aoi, stack, tmp_path, stream_threshold_km2=0.005)
    buffer = t.build_terrain(aoi, stack, tmp_path, buffer_km=6)
    shifted = replace(stack, transform=from_origin(stack.transform.c + 1, stack.transform.f, 10, 10))
    grid = t.build_terrain(aoi, shifted, tmp_path)
    assert len({first.cache_path, threshold.cache_path, buffer.cache_path, grid.cache_path}) == 4
    assert threshold.stream_mask.sum() > first.stream_mask.sum()


def test_save_load_roundtrip_preserves_arrays_and_metadata(tmp_path, aoi, stack, fake_dem):
    original = t.build_terrain(aoi, stack, tmp_path / "cache", stream_threshold_km2=0.005)
    path = tmp_path / "roundtrip.tif"
    t.save_terrain(original, path)
    loaded = t.load_terrain(path)
    for name in ("elevation", "slope_deg", "hand_m", "stream_mask"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(original, name))
    assert loaded.transform == original.transform
    assert loaded.crs == original.crs
    metadata = original.to_dict()
    metadata["cache_path"] = str(path)
    assert loaded.to_dict() == metadata
    with rasterio.open(path) as src:
        assert src.descriptions == t.BAND_NAMES
        assert src.dtypes == ("float32",) * 4
        assert src.units == ("m", "degree", "m", "1")
        assert set(np.unique(src.read(4))) <= {0, 1}
    png = Path(original.cache_path).with_name("quicklook.png")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("kwargs, message", [
    ({"buffer_km": -1}, "buffer_km"), ({"buffer_km": float("inf")}, "buffer_km"),
    ({"stream_threshold_km2": 0}, "stream_threshold_km2"),
    ({"stream_threshold_km2": -1}, "stream_threshold_km2"),
    ({"hydro_resolution": 0}, "hydro_resolution"),
    ({"hydro_resolution": 10}, "at least 30 m"),
])
def test_invalid_parameters_fail_before_fetch(tmp_path, aoi, stack, kwargs, message):
    with pytest.raises(ValueError, match=message):
        t.build_terrain(aoi, stack, tmp_path, **kwargs)


def test_no_dem_tiles_is_a_clear_error(aoi):
    with pytest.raises(ValueError, match="No DEM tiles"):
        t.fetch_dem(aoi.bbox, 5, FakeClient([]))


def test_cli_prints_metadata(tmp_path, aoi, stack, fake_dem, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [
        "terrain", "--place", "Trishuli, Nepal", "--date", "2020-08-10",
        "--cache-dir", str(tmp_path), "--stream-threshold-km2", "0.005",
    ])
    monkeypatch.setattr(t, "build_aoi", lambda **kwargs: aoi)
    monkeypatch.setattr(t, "find_best_pair", lambda area: "pair")
    monkeypatch.setattr(t, "preprocess_pair", lambda *args, **kwargs: stack)
    t.main()
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["shape"] == list(stack.array.shape[-2:])
    assert metadata["source"] == "cop-dem-glo-30"
    assert metadata["stats"]["hand_m"]["min"] == 0
    assert metadata["runtime_seconds"] > 0
    assert Path(metadata["cache_path"]).exists()
