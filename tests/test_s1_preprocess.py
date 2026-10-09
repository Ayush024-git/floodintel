"""Offline preprocessing checks using small, local synthetic GeoTIFFs."""

from dataclasses import replace
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import xy
from rasterio.windows import Window, transform as window_transform

from src.data.aoi import build_aoi
from src.data.s1_search import S1Pass, select_pairs
from src.data import s1_preprocess as s1


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail_open(*args, **kwargs):
        pytest.fail("Offline tests must inject a fake STAC client")

    monkeypatch.setattr(s1.Client, "open", fail_open)


@pytest.fixture
def aoi():
    return build_aoi(bbox=(85, 27.9, 85.001, 27.901), flood_date="2020-08-10")


@pytest.fixture
def grid(aoi):
    return s1.make_grid(aoi)


def pass_(name, day):
    return S1Pass(
        "descending", 19, date.fromisoformat(day), [name], ["sentinel-1a"],
        1.0, f"{day}T06:00:00Z", f"{day}T06:00:00Z",
    )


@pytest.fixture
def pair_result(aoi):
    return select_pairs(aoi, [pass_("pre", "2020-08-09"), pass_("post", "2020-08-11")])


def write_raster(path, values, grid, *, affine=None, gcps=False):
    affine = grid.transform if affine is None else affine
    options = {"transform": affine}
    if gcps:
        height, width = values.shape
        options = {"gcps": [
            GroundControlPoint(
                row=row, col=col,
                x=xy(affine, row, col, offset="ul")[0],
                y=xy(affine, row, col, offset="ul")[1],
            )
            for row, col in ((0, 0), (0, width), (height, 0), (height, width))
        ]}
    with rasterio.open(
        path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
        count=1, dtype="float32", nodata=-32768, crs=grid.crs, **options,
    ) as dst:
        dst.write(values.astype("float32"), 1)
    return str(path)


def item(name, day, href, *, direction="descending", orbit=19):
    return {
        "id": name,
        "properties": {
            "datetime": f"{day}T06:00:00Z",
            "sat:orbit_state": direction,
            "sat:relative_orbit": orbit,
        },
        "assets": {key: {"href": href} for key in ("vv", "vh")},
    }


class FakeClient:
    def __init__(self, rtc_items=(), grd_items=()):
        self.rtc_items = list(rtc_items)
        self.grd_items = list(grd_items)
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["collections"] == ["sentinel-1-rtc"]:
            items = self.rtc_items
        else:
            items = [value for value in self.grd_items if value["id"] in kwargs["ids"]]
        return SimpleNamespace(items=lambda: iter(items))


@pytest.fixture
def rtc_client(tmp_path, grid):
    values = np.full((grid.height, grid.width), 0.1, dtype="float32")
    href = write_raster(tmp_path / "power.tif", values, grid)
    return FakeClient([
        item("pre_rtc", "2020-08-09", href),
        item("post_rtc", "2020-08-11", href),
    ])


def test_rtc_search_matches_exact_pass_and_sorts(aoi):
    values = [
        item("z", "2020-08-09", "unused"), item("a", "2020-08-09", "unused", orbit="19"),
        item("wrong-day", "2020-08-10", "unused"),
        item("wrong-orbit", "2020-08-09", "unused", orbit=20),
        item("wrong-direction", "2020-08-09", "unused", direction="ascending"),
        item("boolean-orbit", "2020-08-09", "unused", orbit=True),
        {"id": "missing", "properties": {}, "assets": {}},
    ]
    client = FakeClient(values)
    matches = s1.find_rtc_items(aoi, pass_("pre", "2020-08-09"), client)
    assert [value["id"] for value in matches] == ["a", "z"]
    assert client.calls == [{
        "collections": ["sentinel-1-rtc"], "bbox": list(aoi.bbox),
        "datetime": "2020-08-09T00:00:00Z/2020-08-09T23:59:59.999999Z",
    }]


def test_rtc_search_supports_stac_stubs(aoi):
    value = item("stub", "2020-08-09", "unused")
    client = FakeClient([SimpleNamespace(id=value["id"], properties=value["properties"], assets=value["assets"])])
    assert s1.find_rtc_items(aoi, pass_("pre", "2020-08-09"), client)[0].id == "stub"


def test_load_band_reads_bbox_window_only(tmp_path, aoi, grid):
    values = np.full((grid.height + 10, grid.width + 10), 0.25, dtype="float32")
    values[6, 6] = -32768
    affine = window_transform(Window(-5, -5, 1, 1), grid.transform)
    href = write_raster(tmp_path / "large.tif", values, grid, affine=affine)
    band = s1.load_band(item("window", "2020-08-09", href), "vv", aoi)
    assert band.array.shape == (grid.height, grid.width)
    assert band.array.size < values.size
    assert band.transform == grid.transform
    assert band.crs == grid.crs
    assert np.isnan(band.array[1, 1])
    assert band.array[0, 0] == 0.25


def test_adjacent_frames_mosaic_with_valid_overlap(tmp_path, aoi, grid):
    split = grid.width // 2
    left = np.full((grid.height, split + 1), 0.1, dtype="float32")
    left[:, -1] = -32768
    right = np.full((grid.height, grid.width - split), 0.2, dtype="float32")
    left_href = write_raster(tmp_path / "left.tif", left, grid)
    right_href = write_raster(
        tmp_path / "right.tif", right, grid,
        affine=window_transform(Window(split, 0, 1, 1), grid.transform),
    )
    output = s1.mosaic_pass([
        item("left", "2020-08-09", left_href), item("right", "2020-08-09", right_href),
    ], "vv", aoi, grid)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output[:, :split], 0.1)
    np.testing.assert_allclose(output[:, split:], 0.2)


def test_lee_reduces_noise_and_preserves_edge():
    rng = np.random.default_rng(42)
    power = np.ones((128, 128), dtype="float32")
    power[:, :64] = 0.1
    power *= rng.gamma(8, 1 / 8, size=power.shape)
    power[0, 0] = np.nan
    filtered = s1.lee_filter(power)
    assert np.nanvar(filtered[10:-10, 10:55]) < 0.7 * np.var(power[10:-10, 10:55])
    assert np.var(filtered[10:-10, 75:-10]) < 0.7 * np.var(power[10:-10, 75:-10])
    assert filtered[10:-10, 60:63].mean() < 0.16
    assert filtered[10:-10, 65:68].mean() > 0.85
    assert np.isnan(filtered[0, 0])
    assert filtered.dtype == np.float32


def test_db_conversion_clipping_and_invalid_mask():
    output = s1.power_to_db(np.array([[0.1, 1, 1e-6, 100, 0, -1, np.nan, np.inf]], dtype="float32"))
    np.testing.assert_allclose(output[0, :4], [-10, 0, -35, 5])
    assert np.isnan(output[0, 4:]).all()


def test_shifted_post_is_aligned_to_pre_grid(tmp_path, aoi, grid, pair_result):
    y, x = np.indices((grid.height, grid.width))
    pre = 0.05 + x * 0.002 + y * 0.003
    post = 0.05 + (x + 0.5) * 0.002 + (y + 0.5) * 0.003
    pre_href = write_raster(tmp_path / "pre.tif", pre, grid)
    post_href = write_raster(
        tmp_path / "post.tif", post, grid,
        affine=window_transform(Window(0.5, 0.5, 1, 1), grid.transform),
    )
    client = FakeClient([
        item("pre_rtc", "2020-08-09", pre_href), item("post_rtc", "2020-08-11", post_href),
    ])
    stack = s1.preprocess_pair(aoi, pair_result, tmp_path / "cache", client, filter_window=1)
    assert stack.transform == grid.transform
    assert stack.crs == grid.crs
    assert stack.array.shape == (4, grid.height, grid.width)
    np.testing.assert_allclose(stack.array[0, 2:-2, 2:-2], stack.array[2, 2:-2, 2:-2], atol=2e-5)


def test_common_valid_mask_fraction_and_warning(tmp_path, aoi, grid, pair_result):
    power = np.full((grid.height, grid.width), 0.1, dtype="float32")
    power[:grid.height // 2] = -32768
    power[-1, -1] = 0
    href = write_raster(tmp_path / "holes.tif", power, grid)
    client = FakeClient([item("pre_rtc", "2020-08-09", href), item("post_rtc", "2020-08-11", href)])
    stack = s1.preprocess_pair(aoi, pair_result, tmp_path / "cache", client)
    expected = (power > 0)
    np.testing.assert_array_equal(stack.valid_mask, expected)
    assert stack.valid_mask.dtype == bool
    assert stack.valid_fraction == pytest.approx(expected.mean())
    assert any("valid_fraction=" in warning and "below 0.9" in warning for warning in stack.warnings)
    assert np.isnan(stack.array[:, ~expected]).all()


def test_oversized_grid_auto_coarsens():
    large = build_aoi(bbox=(85, 27.9, 85.45, 28.3), flood_date="2020-08-10", max_side_km=60)
    grid = s1.make_grid(large)
    assert grid.resolution == 20
    assert max(grid.width, grid.height) <= 4000
    assert any("auto-coarsened" in warning for warning in grid.warnings)


def test_explicit_twenty_meter_grid(aoi):
    ten, twenty = s1.make_grid(aoi, 10), s1.make_grid(aoi, 20)
    assert twenty.resolution == 20
    assert twenty.width < ten.width
    assert twenty.height < ten.height


def test_grd_gcp_window_and_amplitude_conversion(tmp_path, aoi, grid):
    values = np.full((grid.height + 10, grid.width + 10), 2, dtype="float32")
    affine = window_transform(Window(-5, -5, 1, 1), grid.transform)
    href = write_raster(tmp_path / "amplitude.tif", values, grid, affine=affine, gcps=True)
    value = item("raw", "2020-08-09", href)
    band = s1.load_band(value, "vv", aoi)
    assert band.array.shape == (grid.height, grid.width)
    assert band.transform is None
    assert band.gcps[0].row == pytest.approx(-5)
    assert band.gcps[0].col == pytest.approx(-5)
    power = s1.load_grd_fallback(value, "vv", aoi, grid)
    np.testing.assert_allclose(power, 4)


@pytest.mark.parametrize("partial_rtc", [False, True])
def test_fallback_always_warns_and_never_mixes_scales(tmp_path, aoi, grid, pair_result, partial_rtc):
    href = write_raster(tmp_path / "raw.tif", np.ones((grid.height, grid.width)), grid, gcps=True)
    raw = [item("pre", "2020-08-09", href), item("post", "2020-08-11", href)]
    rtc = [item("pre_rtc", "2020-08-09", "must-not-be-read")] if partial_rtc else []
    client = FakeClient(rtc, raw)
    stack = s1.preprocess_pair(aoi, pair_result, tmp_path / "cache", client)
    assert stack.source == "grd_fallback"
    assert s1.FALLBACK_WARNING in stack.warnings
    assert any("both passes use GRD" in warning for warning in stack.warnings)
    assert any("uncalibrated DN power" in warning for warning in stack.warnings)
    np.testing.assert_allclose(stack.array, 0, atol=1e-5)
    assert stack.pair_metadata["input_item_ids"] == {"pre": ["pre"], "post": ["post"]}


def test_missing_grd_frame_is_an_error(tmp_path, aoi, pair_result):
    with pytest.raises(ValueError, match="GRD frames are missing"):
        s1.preprocess_pair(aoi, pair_result, tmp_path, FakeClient())


def test_cache_reuses_pixels(tmp_path, aoi, pair_result, rtc_client, monkeypatch):
    first = s1.preprocess_pair(aoi, pair_result, tmp_path / "cache", rtc_client)

    def fail_loader(*args, **kwargs):
        pytest.fail("Cached stack must not read or process source rasters")

    monkeypatch.setattr(s1, "load_band", fail_loader)
    monkeypatch.setattr(s1, "lee_filter", fail_loader)
    second = s1.preprocess_pair(aoi, pair_result, tmp_path / "cache", rtc_client)
    assert second.cache_path == first.cache_path
    np.testing.assert_array_equal(second.array, first.array)
    assert second.to_dict() == first.to_dict()


def test_cache_key_includes_filter_and_resolution(tmp_path, aoi, pair_result, rtc_client):
    first = s1.preprocess_pair(aoi, pair_result, tmp_path, rtc_client)
    other_filter = s1.preprocess_pair(aoi, pair_result, tmp_path, rtc_client, filter_window=3)
    coarser = s1.preprocess_pair(aoi, pair_result, tmp_path, rtc_client, resolution=20)
    assert len({first.cache_path, other_filter.cache_path, coarser.cache_path}) == 3


@pytest.mark.parametrize("status", ["no_scenes", "no_same_orbit_pair"])
def test_non_ok_pair_result_raises(tmp_path, aoi, pair_result, status):
    result = replace(pair_result, status=status, pair=None, reason="No matching scenes")
    with pytest.raises(ValueError, match=f"status={status}; No matching scenes"):
        s1.preprocess_pair(aoi, result, tmp_path)


def test_save_load_roundtrip(tmp_path, aoi, pair_result, rtc_client):
    original = s1.preprocess_pair(aoi, pair_result, tmp_path / "cache", rtc_client)
    path = tmp_path / "roundtrip.tif"
    s1.save_stack(original, path)
    loaded = s1.load_stack(path)
    np.testing.assert_array_equal(loaded.array, original.array)
    np.testing.assert_array_equal(loaded.valid_mask, original.valid_mask)
    assert loaded.transform == original.transform
    assert loaded.crs == original.crs
    assert loaded.band_names == s1.BAND_NAMES
    assert loaded.pair_metadata == original.pair_metadata
    assert loaded.warnings == original.warnings
    assert loaded.valid_fraction == original.valid_fraction
    assert loaded.array.dtype == np.float32
    with rasterio.open(path) as src:
        assert src.descriptions == s1.BAND_NAMES
        assert src.units == ("dB",) * 4
        assert np.isnan(src.nodata)
    assert json.loads(path.with_suffix(".json").read_text())["source"] == "rtc"
    quicklook = Path(original.cache_path).with_name("quicklook.png")
    assert quicklook.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("window", [0, 2, -1, 1.5, True])
def test_invalid_filter_window(window):
    with pytest.raises(ValueError, match="positive odd integer"):
        s1.lee_filter(np.ones((3, 3)), window)


@pytest.mark.parametrize("resolution", [0, -10, float("nan"), float("inf"), True])
def test_invalid_resolution(aoi, resolution):
    with pytest.raises(ValueError, match="finite and positive"):
        s1.make_grid(aoi, resolution)


def test_cli_metadata_output(tmp_path, aoi, pair_result, rtc_client, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", [
        "s1_preprocess", "--place", "Trishuli, Nepal", "--date", "2020-08-10",
        "--cache-dir", str(tmp_path / "cache"),
    ])
    monkeypatch.setattr(s1, "build_aoi", lambda **kwargs: aoi)
    monkeypatch.setattr(s1, "find_best_pair", lambda value: pair_result)
    monkeypatch.setattr(s1.Client, "open", lambda *args, **kwargs: rtc_client)
    s1.main()
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["shape"][0] == 4
    assert metadata["source"] == "rtc"
    assert metadata["valid_fraction"] == 1
    assert Path(metadata["cache_path"]).is_file()
