"""Offline checks for geographic bounds, observation windows and serialization."""

from datetime import date, datetime, timedelta
import json
import math

import pytest

from src.data import aoi


SMALL_BBOX = (85.0, 27.0, 85.1, 27.1)
FLOOD_DATE = date(2020, 8, 10)


def test_date_windows():
    result = aoi.build_aoi(bbox=SMALL_BBOX, flood_date="2020-08-10")
    assert result.pre_window == (date(2020, 6, 11), date(2020, 8, 9))
    assert result.post_window == (FLOOD_DATE, date(2020, 8, 22))
    custom = aoi.build_aoi(bbox=SMALL_BBOX, flood_date=FLOOD_DATE, pre_days=10, post_days=3)
    assert custom.pre_window == (date(2020, 7, 31), date(2020, 8, 9))
    assert custom.post_window == (FLOOD_DATE, date(2020, 8, 13))


def test_place_path(monkeypatch):
    calls = []

    def fake_geocode(place):
        calls.append(place)
        return SMALL_BBOX

    monkeypatch.setattr(aoi, "geocode_place", fake_geocode)
    result = aoi.build_aoi(place="Trishuli, Nepal", flood_date=FLOOD_DATE)
    assert calls == ["Trishuli, Nepal"]
    assert result.name == "Trishuli, Nepal"
    assert result.bbox == SMALL_BBOX


def test_bbox_path(monkeypatch):
    def unexpected_geocode(place):
        pytest.fail("Explicit bounds must not call the geocoder")

    monkeypatch.setattr(aoi, "geocode_place", unexpected_geocode)
    result = aoi.build_aoi(bbox=SMALL_BBOX, flood_date=FLOOD_DATE)
    assert result.bbox == SMALL_BBOX
    assert result.name == "Custom bbox"
    expected_area = 0.1 * 111.32 * math.cos(math.radians(27.05)) * 0.1 * 111.32
    assert result.area_km2 == pytest.approx(expected_area)
    assert result.warnings == []


@pytest.mark.parametrize("kwargs", [{}, {"place": "Nepal", "bbox": SMALL_BBOX}])
def test_exactly_one_area(kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        aoi.build_aoi(flood_date=FLOOD_DATE, **kwargs)


@pytest.mark.parametrize("bbox", [
    (1, 2, 1, 3), (2, 2, 1, 3), (1, 3, 2, 3), (1, 4, 2, 3),
    (-181, 0, 0, 1), (0, 0, 181, 1), (0, -91, 1, 0), (0, 0, 1, 91),
    (0, 0, float("nan"), 1), (0, 0, float("inf"), 1), (0, 1, 2),
    (0, 1, 2, 3, 4), ("bad", 0, 1, 2),
])
def test_invalid_bbox(bbox):
    with pytest.raises(ValueError):
        aoi.build_aoi(bbox=bbox, flood_date=FLOOD_DATE)


def test_future_flood_date():
    with pytest.raises(ValueError, match="future"):
        aoi.build_aoi(bbox=SMALL_BBOX, flood_date=date.today() + timedelta(days=1))


@pytest.mark.parametrize("invalid", ["2020-02-30", "20200810", "bad", None, datetime(2020, 8, 10)])
def test_invalid_flood_date(invalid):
    with pytest.raises(ValueError):
        aoi.build_aoi(bbox=SMALL_BBOX, flood_date=invalid)


@pytest.mark.parametrize("bbox", [(80, 25, 90, 35), (80, 60, 90, 60.1), (80, 25, 80.1, 35)])
def test_size_clamp(bbox):
    result = aoi.build_aoi(bbox=bbox, flood_date=FLOOD_DATE, max_side_km=50)
    west, south, east, north = result.bbox
    center_lat = (south + north) / 2
    width = (east - west) * 111.32 * math.cos(math.radians(center_lat))
    height = (north - south) * 111.32
    assert width <= 50 + 1e-9
    assert height <= 50 + 1e-9
    assert (west + east) / 2 == pytest.approx((bbox[0] + bbox[2]) / 2)
    assert center_lat == pytest.approx((bbox[1] + bbox[3]) / 2)
    assert result.area_km2 == pytest.approx(width * height)
    assert any("clamped from" in warning and "to" in warning for warning in result.warnings)


def test_roundtrip():
    result = aoi.build_aoi(bbox=(80, 25, 90, 35), flood_date=FLOOD_DATE)
    data = result.to_dict()
    assert data["flood_date"] == "2020-08-10"
    assert data["pre_window"] == ["2020-06-11", "2020-08-09"]
    assert json.loads(result.to_json()) == data
    assert aoi.AOI.from_dict(data) == result
    assert aoi.AOI.from_dict(json.loads(result.to_json())) == result


def test_post_window_truncated():
    today = date.today()
    flood_date = today - timedelta(days=2)
    result = aoi.build_aoi(bbox=SMALL_BBOX, flood_date=flood_date)
    assert result.post_window == (flood_date, today)
    assert any("past today" in warning for warning in result.warnings)


@pytest.mark.parametrize("kwargs", [
    {"pre_days": 0}, {"pre_days": 1.5}, {"post_days": -1},
    {"max_side_km": 0}, {"max_side_km": float("nan")},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        aoi.build_aoi(bbox=SMALL_BBOX, flood_date=FLOOD_DATE, **kwargs)
