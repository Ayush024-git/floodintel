"""Offline scene search, pass coverage, and same-orbit pair selection checks."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
import json
from types import SimpleNamespace

import pytest
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry

from src.data.aoi import build_aoi
from src.data import s1_search as s1


BBOX = (0.0, 0.0, 1.0, 1.0)
FLOOD = date(2020, 8, 10)


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch):
    def unexpected_open(*args, **kwargs):
        pytest.fail("Tests must use an injected client, never a live catalog")

    monkeypatch.setattr(s1.Client, "open", unexpected_open)


@pytest.fixture
def aoi():
    return build_aoi(bbox=BBOX, flood_date=FLOOD, max_side_km=200)


def scene(item_id, day, *, orbit=42, direction="ascending", footprint=None, platform="sentinel-1a"):
    return {
        "id": item_id,
        "datetime": f"{day}T06:00:00Z",
        "orbit_state": direction,
        "relative_orbit": orbit,
        "platform": platform,
        "geometry": box(*BBOX) if footprint is None else footprint,
    }


def item(item_id, day, **kwargs):
    value = scene(item_id, day, **kwargs)
    return {
        "id": value["id"],
        "geometry": mapping(value["geometry"]),
        "properties": {
            "datetime": value["datetime"],
            "sat:orbit_state": value["orbit_state"],
            "sat:relative_orbit": value["relative_orbit"],
            "platform": value["platform"],
            "sar:instrument_mode": "IW",
            "sar:polarizations": ["VV", "VH"],
        },
    }


class FakeClient:
    def __init__(self, *batches):
        self.batches = list(batches)
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        batch = self.batches.pop(0)
        return SimpleNamespace(items=lambda: iter(batch))


def select(aoi, *scenes):
    return s1.select_pairs(aoi, s1.group_into_passes(list(scenes), aoi.bbox))


def test_same_orbit_pair(aoi):
    result = select(aoi, scene("pre", "2020-08-09"), scene("post", "2020-08-11"))
    assert result.status == "ok"
    assert result.pair.pre.item_ids == ["pre"]
    assert result.pair.post.item_ids == ["post"]
    assert result.pair.relative_orbit == 42
    assert result.pair.orbit_state == "ascending"
    assert result.pair.days_before_flood == result.pair.days_after_flood == 1
    assert result.n_pre_scenes == result.n_post_scenes == 1
    assert result.reason


@pytest.mark.parametrize("post_kwargs", [{"direction": "descending"}, {"orbit": 43}])
def test_different_viewing_geometry_is_not_paired(aoi, post_kwargs):
    result = select(aoi, scene("pre", "2020-08-09"), scene("post", "2020-08-11", **post_kwargs))
    assert result.status == "no_same_orbit_pair"
    assert result.pair is None
    assert result.alternatives == []
    assert result.reason


def test_adjacent_frames_union_to_full_coverage(aoi):
    scenes = [
        scene("left", "2020-08-09", footprint=box(-1, -1, 0.5, 2)),
        scene("right", "2020-08-09", footprint=box(0.5, -1, 2, 2), platform="sentinel-1b"),
        scene("post", "2020-08-11"),
    ]
    scenes[1]["datetime"] = "2020-08-09T06:01:00Z"
    result = select(aoi, *scenes)
    assert result.pair.pre.coverage == pytest.approx(1)
    assert result.pair.pre.item_ids == ["left", "right"]
    assert result.pair.pre.platforms == ["sentinel-1a", "sentinel-1b"]
    assert result.pair.pre.earliest_datetime == "2020-08-09T06:00:00+00:00"
    assert result.pair.pre.latest_datetime == "2020-08-09T06:01:00+00:00"
    assert result.n_pre_scenes == 2


def test_overlapping_footprints_are_not_double_counted():
    passes = s1.group_into_passes([
        scene("one", "2020-08-09", footprint=box(0, 0, 0.5, 1)),
        scene("two", "2020-08-09", footprint=box(0.25, 0, 0.75, 1)),
    ], BBOX)
    assert passes[0].coverage == pytest.approx(0.75)


def test_passes_group_by_utc_date_orbit_and_direction():
    offset = scene("offset", "2020-08-09")
    offset["datetime"] = "2020-08-10T01:00:00+05:00"
    passes = s1.group_into_passes([
        offset, scene("utc", "2020-08-09"),
        scene("next-day", "2020-08-10"),
        scene("other-orbit", "2020-08-09", orbit=43),
        scene("descending", "2020-08-09", direction="descending"),
    ], BBOX)
    assert len(passes) == 4
    combined = next(p for p in passes if "offset" in p.item_ids)
    assert combined.item_ids == ["offset", "utc"]
    assert combined.acquisition_date == date(2020, 8, 9)


def test_low_coverage_warning(aoi):
    result = select(aoi, scene("pre", "2020-08-09", footprint=box(0, 0, 0.6, 1)), scene("post", "2020-08-11"))
    assert result.status == "ok"
    assert result.pair.min_coverage == pytest.approx(0.6)
    assert any("0.600000" in warning for warning in result.warnings)
    assert result.pair.warnings == result.warnings


@pytest.mark.parametrize("coverage, expected", [(0.49, "no_same_orbit_pair"), (0.5, "ok")])
def test_minimum_usable_coverage(aoi, coverage, expected):
    result = select(aoi, scene("pre", "2020-08-09", footprint=box(0, 0, coverage, 1)), scene("post", "2020-08-11"))
    assert result.status == expected
    assert result.n_pre_scenes == 1


def test_ranking_coverage_then_post_then_pre(aoi):
    result = select(aoi,
        scene("pre-older", "2020-08-01"),
        scene("pre-nearer", "2020-08-09"),
        scene("partial-close", "2020-08-10", footprint=box(0, 0, 0.94, 1)),
        scene("full-far", "2020-08-14"),
        scene("full-close", "2020-08-11"),
    )
    assert result.pair.post.item_ids == ["full-close"]
    assert result.pair.pre.item_ids == ["pre-nearer"]
    assert len(result.alternatives) == 3
    assert result.alternatives[0].post.item_ids == ["full-close"]
    assert result.alternatives[1].post.item_ids == ["full-far"]
    assert result.pair not in result.alternatives


def test_coverage_is_bucketed(aoi):
    result = select(aoi,
        scene("pre", "2020-08-09"),
        scene("near", "2020-08-11", footprint=box(0, 0, 0.95, 1)),
        scene("far", "2020-08-12"),
    )
    assert result.pair.post.item_ids == ["near"]
    assert result.pair.warnings == []


@pytest.mark.parametrize("day, text", [
    ("2020-08-10", "acquisition time may precede the event"),
    ("2020-08-17", "flood may have receded; revisit gap"),
])
def test_acquisition_timing_warnings(aoi, day, text):
    result = select(aoi, scene("pre", "2020-08-09"), scene("post", day))
    assert any(text in warning for warning in result.pair.warnings)
    assert any(text in warning for warning in result.warnings)


def test_six_day_gap_has_no_revisit_warning(aoi):
    result = select(aoi, scene("pre", "2020-08-09"), scene("post", "2020-08-16"))
    assert result.pair.warnings == []


def test_no_scenes(aoi):
    result = s1.select_pairs(aoi, [])
    assert result.status == "no_scenes"
    assert result.pair is None
    assert result.n_pre_scenes == result.n_post_scenes == 0
    assert result.reason


def test_window_boundaries_and_scene_counts(aoi):
    result = select(aoi,
        scene("pre-start", aoi.pre_window[0].isoformat()),
        scene("post-end", aoi.post_window[1].isoformat()),
        scene("outside-pre", (aoi.pre_window[0] - timedelta(days=1)).isoformat()),
        scene("outside-post", (aoi.post_window[1] + timedelta(days=1)).isoformat()),
    )
    assert result.status == "ok"
    assert result.n_pre_scenes == result.n_post_scenes == 1


def test_search_normalizes_dict_and_stub_items(aoi):
    first = item("first", "2020-08-09")
    second = item("second", "2020-08-11", orbit="42")
    second["properties"].pop("platform")
    client = FakeClient([first, SimpleNamespace(id=second["id"], properties=second["properties"], geometry=second["geometry"]), first])
    scenes = s1.search_s1(aoi, collection="custom-collection", client=client)
    assert len(scenes) == 2
    assert scenes[1]["relative_orbit"] == 42
    assert scenes[1]["platform"] is None
    assert all(isinstance(value["geometry"], BaseGeometry) for value in scenes)
    assert set(scenes[0]) == {"id", "datetime", "orbit_state", "relative_orbit", "platform", "geometry"}
    assert client.calls == [{
        "collections": ["custom-collection"], "bbox": list(BBOX),
        "datetime": "2020-06-11T00:00:00Z/2020-08-22T23:59:59.999999Z",
    }]


@pytest.mark.parametrize("key, value", [
    ("sar:instrument_mode", "EW"), ("sar:instrument_mode", None),
    ("sar:polarizations", ["VV"]), ("sar:polarizations", "VV,VH"),
    ("sat:orbit_state", None), ("sat:orbit_state", "unknown"),
    ("sat:relative_orbit", None), ("sat:relative_orbit", True),
    ("sat:relative_orbit", 42.5), ("sat:relative_orbit", "bad"),
    ("datetime", None), ("datetime", "bad"),
    ("datetime", "2020-08-23T00:00:00Z"),
])
def test_search_skips_ineligible_metadata(aoi, key, value):
    bad = item("bad", "2020-08-09")
    bad["properties"][key] = value
    assert s1.search_s1(aoi, client=FakeClient([bad])) == []


@pytest.mark.parametrize("geometry", [None, {}, {"type": "Point", "coordinates": [0, 0]}, mapping(box(0, 0, 0, 0))])
def test_search_skips_unusable_footprints(aoi, geometry):
    bad = item("bad", "2020-08-09")
    bad["geometry"] = geometry
    assert s1.search_s1(aoi, client=FakeClient([bad])) == []


def test_catalog_open_uses_signing_modifier(aoi, monkeypatch):
    calls = []
    client = FakeClient([])

    def fake_open(url, **kwargs):
        calls.append((url, kwargs))
        return client

    monkeypatch.setattr(s1.Client, "open", fake_open)
    assert s1.search_s1(aoi) == []
    assert calls == [(s1.STAC_URL, {"modifier": s1.planetary_computer.sign_inplace})]


def test_widening_retry_succeeds(aoi):
    client = FakeClient([], [item("pre", "2020-08-09"), item("late-post", "2020-08-25")])
    result = s1.find_best_pair(aoi, collection="custom", client=client)
    assert result.status == "ok"
    assert result.pair.post.item_ids == ["late-post"]
    assert len(client.calls) == 2
    assert client.calls[0]["datetime"].endswith("2020-08-22T23:59:59.999999Z")
    assert client.calls[1]["datetime"].endswith("2020-09-03T23:59:59.999999Z")
    assert all(call["datetime"].startswith("2020-06-11T00:00:00Z/") for call in client.calls)
    assert all(call["collections"] == ["custom"] for call in client.calls)
    assert any("window was widened" in warning for warning in result.warnings)
    assert aoi.post_window[1] == date(2020, 8, 22)


def test_widening_after_orbit_mismatch(aoi):
    pre = item("pre", "2020-08-09")
    post = item("wrong-post", "2020-08-11", orbit=43)
    client = FakeClient([pre, post], [pre, post, item("matching-post", "2020-08-25")])
    assert s1.find_best_pair(aoi, client=client).pair.post.item_ids == ["matching-post"]


def test_success_does_not_retry(aoi):
    client = FakeClient([item("pre", "2020-08-09"), item("post", "2020-08-11")])
    assert s1.find_best_pair(aoi, client=client).status == "ok"
    assert len(client.calls) == 1


def test_unsuccessful_retry_remains_honest(aoi):
    result = s1.find_best_pair(aoi, client=FakeClient([], []))
    assert result.status == "no_scenes"
    assert result.pair is None
    assert result.reason
    assert any("widened" in warning for warning in result.warnings)


def test_widening_caps_at_today(aoi, monkeypatch):
    class FixedDate(date):
        @classmethod
        def today(cls):
            return date(2020, 8, 24)

    monkeypatch.setattr(s1, "date", FixedDate)
    client = FakeClient([], [])
    s1.find_best_pair(aoi, client=client)
    assert client.calls[1]["datetime"].endswith("2020-08-24T23:59:59.999999Z")


def test_retry_at_today_reports_no_available_extension(aoi, monkeypatch):
    class FixedDate(date):
        @classmethod
        def today(cls):
            return aoi.post_window[1]

    monkeypatch.setattr(s1, "date", FixedDate)
    client = FakeClient([], [])
    result = s1.find_best_pair(aoi, client=client)
    assert len(client.calls) == 2
    assert client.calls[0] == client.calls[1]
    assert any("no extension" in warning for warning in result.warnings)


def test_to_dict_is_json_serializable_and_dataclasses_are_frozen(aoi):
    aoi = replace(aoi, warnings=["AOI warning"])
    result = select(aoi, scene("pre", "2020-08-09"), scene("post", "2020-08-11"), scene("other", "2020-08-12"))
    for value in (result, result.pair, result.pair.pre, s1.select_pairs(aoi, [])):
        assert json.loads(json.dumps(value.to_dict(), allow_nan=False)) == value.to_dict()
        with pytest.raises(FrozenInstanceError):
            value.extra = "forbidden"
    assert result.to_dict()["pair"]["pre"]["acquisition_date"] == "2020-08-09"
    assert "AOI warning" in result.warnings


@pytest.mark.parametrize("area_args", [["--place", "Trishuli, Nepal"], ["--bbox", "0", "0", "1", "1"]])
def test_cli_offline(aoi, monkeypatch, capsys, area_args):
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return aoi

    monkeypatch.setattr("sys.argv", ["s1_search", *area_args, "--date", "2020-08-10"])
    monkeypatch.setattr(s1, "build_aoi", fake_build)
    monkeypatch.setattr(s1, "find_best_pair", lambda value: s1.select_pairs(value, []))
    s1.main()
    assert json.loads(capsys.readouterr().out)["status"] == "no_scenes"
    assert calls[0]["flood_date"] == "2020-08-10"
    assert calls[0]["place"] == ("Trishuli, Nepal" if "--place" in area_args else None)
