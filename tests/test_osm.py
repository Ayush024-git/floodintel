"""Offline historical Overpass acquisition, cleaning and provenance checks."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
import pytest
from pyproj import CRS
import requests
from shapely.geometry import box

from src.data.aoi import build_aoi
from src.data import osm


ENDPOINTS = ("https://first.invalid/api", "https://second.invalid/api")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Tests must never contact an HTTP endpoint")

    monkeypatch.setattr(requests.Session, "post", forbidden)
    monkeypatch.setattr(osm.time, "sleep", lambda seconds: None)


@pytest.fixture
def aoi():
    return build_aoi(bbox=(85, 27.9, 85.01, 27.91), flood_date="2026-08-10")


def element(kind, number, tags, *, coordinates=None, center=None, location=None, timestamp="2020-01-01T00:00:00Z"):
    value = {"type": kind, "id": number, "version": 1, "tags": tags, "timestamp": timestamp}
    if coordinates is not None:
        value["geometry"] = [{"lon": x, "lat": y} for x, y in coordinates]
    if center is not None:
        value["center"] = {"lon": center[0], "lat": center[1]}
    if location is not None:
        value.update(lon=location[0], lat=location[1])
    return value


@pytest.fixture
def payload():
    return {"osm3s": {"timestamp_osm_base": "2026-10-09T00:00:00Z"}, "elements": [
        element("way", 1, {"highway": "primary", "name": "Main", "surface": "asphalt", "oneway": "yes"},
                coordinates=[(84.999, 27.902), (85.005, 27.902), (85.011, 27.902)]),
        element("way", 2, {"highway": "track"}, coordinates=[(85.001, 27.901), (85.002, 27.904)]),
        element("way", 3, {"highway": "secondary", "bridge": "yes"}, coordinates=[(85.004, 27.903), (85.006, 27.903)]),
        element("way", 4, {"highway": "residential", "tunnel": "yes", "bridge": "no"},
                coordinates=[(85.004, 27.904), (85.006, 27.904)]),
        element("way", 5, {"highway": "path", "ford": "yes"}, coordinates=[(85.001, 27.906), (85.002, 27.906)]),
        element("way", 6, {"building": "yes", "name": "House"},
                coordinates=[(85.002, 27.903), (85.003, 27.903), (85.003, 27.904), (85.002, 27.904), (85.002, 27.903)]),
        element("node", 7, {"amenity": "hospital", "name": "Hospital"}, location=(85.003, 27.905)),
        element("way", 8, {"healthcare": "clinic", "name": "Clinic"}, center=(85.006, 27.906)),
        element("node", 9, {"place": "village", "name": "Village", "population": "1,234"}, location=(85.004, 27.907)),
        element("way", 10, {"place": "village", "population": "unknown"}, center=(85.008, 27.908)),
        element("relation", 11, {"building": "yes", "type": "multipolygon"}, timestamp="2026-08-01T12:00:00Z"),
    ]}


class Response:
    def __init__(self, payload=None, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return deepcopy(self.payload)


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def build(aoi, tmp_path, payload, **kwargs):
    session = Session([Response(payload)] * 100)
    return osm.build_osm(aoi, cache_dir=tmp_path, session=session, context_buffer_km=0,
                         endpoints=ENDPOINTS, **kwargs)


def test_tag_parsing_bridge_view_and_ids(aoi, tmp_path, payload):
    result = build(aoi, tmp_path, payload)
    roads = result.roads.set_index("osm_id")
    assert roads.loc["way/3", "is_bridge"]
    assert not roads.loc["way/4", "is_bridge"]
    assert roads.loc["way/4", "is_tunnel"]
    assert roads.loc["way/5", "is_ford"]
    assert roads.loc["way/1", "surface"] == "asphalt"
    assert roads.loc["way/1", "oneway"] == "yes"
    assert_geodataframe_equal(result.bridges, result.roads.loc[result.roads.is_bridge])
    assert result.bridges.osm_id.tolist() == ["way/3"]
    assert result.hospitals.set_index("osm_id").tier.to_dict() == {"node/7": "hospital", "way/8": "clinic"}
    settlements = result.settlements.set_index("osm_id")
    assert settlements.loc["node/9", "population"] == 1234
    assert settlements.population.isna().loc["way/10"]
    assert result.quality["dropped_building_relations"] == 1
    assert any("Dropped 1 building relations" in warning for warning in result.warnings)


def test_verified_snapshot_and_query_header(aoi, tmp_path, payload):
    session = Session([Response(payload)])
    result = osm.build_osm(aoi, cache_dir=tmp_path, session=session, context_buffer_km=0, endpoints=ENDPOINTS)
    assert result.snapshot_verified is True
    assert result.snapshot_date == "2026-08-10T00:00:00Z"
    assert result.newest_edit == "2026-08-01T12:00:00Z"
    query = session.calls[0][1]["data"]["data"]
    assert '[out:json][timeout:180][date:"2026-08-10T00:00:00Z"];' in query
    assert "out meta geom;" in query and "out meta center;" in query
    assert ">>; out meta;" in query
    assert session.calls[0][1]["headers"]["User-Agent"] == osm.USER_AGENT


def test_newer_dependency_node_rejects_endpoint(aoi, tmp_path, payload):
    bad = deepcopy(payload)
    bad["elements"].append(element("node", 999, {}, location=(85.001, 27.901), timestamp="2026-08-11T00:00:00Z"))
    session = Session([Response(bad), Response(payload)])
    result = osm.build_osm(aoi, cache_dir=tmp_path, session=session, context_buffer_km=0, endpoints=ENDPOINTS)
    assert result.snapshot_verified is True
    assert result.endpoint_used == ENDPOINTS[1]
    assert [call[0] for call in session.calls] == list(ENDPOINTS)


def test_lagging_database_cannot_verify_a_later_snapshot(aoi, tmp_path, payload):
    lagging = deepcopy(payload)
    lagging["osm3s"]["timestamp_osm_base"] = "2026-05-31T22:37:44Z"
    session = Session([Response(lagging), Response(payload)])
    result = osm.build_osm(aoi, cache_dir=tmp_path, session=session,
                           context_buffer_km=0, endpoints=ENDPOINTS)
    assert result.snapshot_verified
    assert result.endpoint_used == ENDPOINTS[1]
    assert len(session.calls) == 2
    assert any("database date=2026-05-31" in warning for warning in result.warnings)
    non_strict = build(aoi, tmp_path / "non-strict", lagging, strict=False)
    assert not non_strict.snapshot_verified
    assert non_strict.quality["timestamp_audit"]["database_behind_snapshot"]
    assert any("UNVERIFIED SNAPSHOT" in warning for warning in non_strict.warnings)


@pytest.mark.parametrize("bad_time", ["2026-08-11T00:00:00Z", None, "2020-01-01T00:00:00"])
def test_none_honor_snapshot_raises(aoi, tmp_path, payload, bad_time):
    payload["elements"][0]["timestamp"] = bad_time
    session = Session([Response(payload), Response(payload)])
    with pytest.raises(osm.OSMSnapshotError, match="timestamp-verified pre-event snapshot"):
        osm.build_osm(aoi, cache_dir=tmp_path, session=session, context_buffer_km=0, endpoints=ENDPOINTS)


def test_non_strict_is_loud_and_not_reused_in_strict_mode(aoi, tmp_path, payload):
    bad = deepcopy(payload)
    bad["elements"][0]["timestamp"] = "2026-08-11T00:00:00Z"
    result = build(aoi, tmp_path, bad, strict=False)
    assert result.snapshot_verified is False
    assert any("UNVERIFIED SNAPSHOT" in warning for warning in result.warnings)
    assert osm.load_osm(result.cache_path).snapshot_verified is False
    session = Session([Response(payload)])
    verified = osm.build_osm(aoi, cache_dir=tmp_path, session=session, context_buffer_km=0, endpoints=ENDPOINTS)
    assert verified.snapshot_verified is True
    assert len(session.calls) == 1


def test_429_backoff_then_success(aoi, tmp_path, payload, monkeypatch):
    sleeps = []
    monkeypatch.setattr(osm.time, "sleep", sleeps.append)
    session = Session([Response(status=429), Response(payload)])
    result = osm.build_osm(aoi, cache_dir=tmp_path, session=session, context_buffer_km=0, endpoints=ENDPOINTS)
    assert result.snapshot_verified
    assert sleeps == [1, 0.25]


def test_timeout_backoff_and_endpoint_fallback(aoi, tmp_path, payload, monkeypatch):
    sleeps = []
    monkeypatch.setattr(osm.time, "sleep", sleeps.append)
    session = Session([Response(status=504)] * 4 + [requests.Timeout(), Response(payload)])
    result = osm.build_osm(aoi, cache_dir=tmp_path, session=session, context_buffer_km=0, endpoints=ENDPOINTS)
    assert result.endpoint_used == ENDPOINTS[1]
    assert sleeps == [1, 2, 4, 1, 0.25]
    assert len(session.calls) == 6


def test_all_http_endpoints_fail(aoi, tmp_path):
    session = Session([Response(status=503)] * 8)
    with pytest.raises(osm.OSMError, match="All Overpass endpoints failed"):
        osm.build_osm(aoi, cache_dir=tmp_path, session=session, context_buffer_km=0, endpoints=ENDPOINTS)
    assert len(session.calls) == 8


def test_tile_fallback_preserves_verified_tiles_and_endpoint_provenance(aoi, tmp_path, payload):
    wide = replace(aoi, bbox=(85, 27.9, 85.14, 27.91))
    assert len(osm.tile_bbox(wide.bbox)) == 2
    session = Session([Response(payload)] + [Response(status=504)] * 4 + [Response(payload)])
    result = osm.build_osm(wide, cache_dir=tmp_path, session=session,
                           context_buffer_km=0, endpoints=ENDPOINTS)
    assert result.snapshot_verified
    assert result.endpoint_used == "; ".join(ENDPOINTS)
    assert result.quality["timestamp_audit"]["endpoint_tile_counts"] == dict.fromkeys(ENDPOINTS, 1)
    assert len(session.calls) == 6
    Path(result.cache_path).with_suffix(".json").unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("Verified tiles from both endpoints should avoid all HTTP calls")

    cached = osm.build_osm(wide, cache_dir=tmp_path, context_buffer_km=0,
                           post_fn=forbidden, endpoints=ENDPOINTS)
    assert cached.snapshot_verified
    assert cached.endpoint_used == result.endpoint_used
    assert cached.quality == result.quality


def test_partial_overpass_response_is_rejected(aoi, tmp_path, payload):
    bad = {**payload, "remark": "runtime error: Query timed out"}
    session = Session([Response(bad)] * 4 + [Response(payload)])
    result = osm.build_osm(aoi, cache_dir=tmp_path, session=session, context_buffer_km=0, endpoints=ENDPOINTS)
    assert result.endpoint_used == ENDPOINTS[1]


def test_tiling_merge_and_deduplication(aoi, tmp_path, payload):
    large = replace(aoi, bbox=(85, 27.9, 85.25, 28.125))
    tiles = osm.tile_bbox(large.bbox)
    assert len(tiles) > 1
    session = Session([Response(payload)] * len(tiles))
    result = osm.build_osm(large, cache_dir=tmp_path, session=session, context_buffer_km=0, endpoints=ENDPOINTS)
    assert len(session.calls) == len(tiles)
    assert result.roads.osm_id.is_unique
    assert result.hospitals.osm_id.is_unique
    assert result.quality["dropped_building_relations"] == 1
    assert result.quality["timestamp_audit"]["returned_records"] == len(tiles) * len(payload["elements"])


def test_requested_or_stack_crs(aoi, tmp_path, payload):
    explicit = build(aoi, tmp_path, payload, crs="EPSG:32644")
    assert explicit.crs == CRS.from_epsg(32644)
    assert explicit.roads.crs == CRS.from_epsg(32644)
    stack = SimpleNamespace(crs="EPSG:32646")
    inherited = build(aoi, tmp_path, payload, stack=stack)
    assert inherited.crs == CRS.from_epsg(32646)
    assert inherited.buildings.crs == CRS.from_epsg(32646)


def test_clipping_split_segments_and_context(aoi, tmp_path, payload):
    payload["elements"].extend([
        element("node", 12, {"place": "hamlet"}, location=(85.011, 27.905)),
        element("node", 13, {"amenity": "doctors"}, location=(85.011, 27.906)),
        element("way", 14, {"highway": "track"}, coordinates=[(85.011, 27.905), (85.012, 27.906)]),
        element("way", 15, {"building": "yes"}, coordinates=[
            (85.011, 27.905), (85.012, 27.905), (85.012, 27.906), (85.011, 27.906), (85.011, 27.905)]),
    ])
    session = Session([Response(payload)] * 20)
    result = osm.build_osm(aoi, cache_dir=tmp_path, session=session, context_buffer_km=1, endpoints=ENDPOINTS)
    roads = result.roads.to_crs(4326)
    assert all(box(*aoi.bbox).buffer(1e-8).covers(geometry) for geometry in roads.loc[roads.in_aoi].geometry)
    primary = roads.loc[roads.osm_id == "way/1"]
    assert len(primary) == 3
    assert primary.segment_id.is_unique
    assert primary.in_aoi.sum() == 1
    assert "way/14" not in roads.osm_id.values
    assert "way/15" not in result.buildings.osm_id.values
    assert not result.settlements.set_index("osm_id").loc["node/12", "in_aoi"]
    assert not result.hospitals.set_index("osm_id").loc["node/13", "in_aoi"]
    assert result.hospitals.set_index("osm_id").loc["node/13", "tier"] == "clinic"


def test_road_reentering_aoi_has_distinct_stable_segments(aoi, tmp_path, payload):
    payload["elements"] = [element("way", 88, {"highway": "track"}, coordinates=[
        (85.001, 27.901), (85.011, 27.901), (85.011, 27.905), (85.001, 27.905),
    ])]
    result = build(aoi, tmp_path, payload)
    assert len(result.roads) == 2
    assert result.roads.osm_id.tolist() == ["way/88", "way/88"]
    assert set(result.roads.segment_id) == {"way/88#0", "way/88#1"}


def test_quality_density_staleness_and_sparse_flags(aoi, tmp_path, payload):
    result = build(aoi, tmp_path, payload)
    quality = result.quality
    assert quality["total_road_km"] == pytest.approx(result.roads.length.sum() / 1000)
    assert quality["road_density_km_per_km2"] == pytest.approx(quality["total_road_km"] / quality["aoi_area_km2"])
    assert quality["counts"] == {"roads": 5, "buildings": 1, "hospitals": 2, "settlements": 2, "bridges": 1}
    assert quality["roads_edited_over_5_years_share"] == 1
    assert quality["median_element_age_years"] > 5
    sparse = {"elements": [element("way", 99, {"highway": "path"}, coordinates=[(85.001, 27.901), (85.00101, 27.901)])]}
    result = build(aoi, tmp_path / "sparse", sparse)
    assert any("Sparse road network" in warning for warning in result.warnings)
    assert any("No hospitals/clinics" in warning for warning in result.warnings)
    assert any("No settlements" in warning for warning in result.warnings)
    assert any("Zero buildings" in warning for warning in result.warnings)
    assert osm.COVERAGE_WARNING in result.warnings


def test_cache_hit_makes_no_http_calls(aoi, tmp_path, payload):
    first = build(aoi, tmp_path, payload)

    def forbidden(*args, **kwargs):
        pytest.fail("Bundle cache hit must skip HTTP entirely")

    second = osm.build_osm(aoi, cache_dir=tmp_path, context_buffer_km=0, post_fn=forbidden, endpoints=ENDPOINTS)
    assert second.to_dict() == first.to_dict()


def test_tile_cache_can_rebuild_bundle_without_http(aoi, tmp_path, payload):
    first = build(aoi, tmp_path, payload)
    Path(first.cache_path).with_suffix(".json").unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("Verified tile cache should avoid re-fetching")

    rebuilt = osm.build_osm(aoi, cache_dir=tmp_path, context_buffer_km=0, post_fn=forbidden, endpoints=ENDPOINTS)
    assert rebuilt.snapshot_verified
    assert rebuilt.quality == first.quality


def test_snapshot_offset_changes_cache_key(aoi, tmp_path, payload):
    first = build(aoi, tmp_path, payload)
    earlier = build(aoi, tmp_path, payload, snapshot_offset_days=1)
    assert earlier.cache_path != first.cache_path
    assert earlier.snapshot_date == "2026-08-09T00:00:00Z"


def test_roundtrip_preserves_ids_and_metadata(aoi, tmp_path, payload):
    original = build(aoi, tmp_path, payload)
    path = tmp_path / "roundtrip.gpkg"
    osm.save_osm(original, path)
    restored = osm.load_osm(path)
    for name in osm.LAYER_COLUMNS:
        assert_geodataframe_equal(getattr(restored, name), getattr(original, name))
    metadata = original.to_dict()
    metadata["cache_path"] = str(path)
    assert restored.to_dict() == metadata
    assert set(gpd.list_layers(path).name) == set(osm.LAYER_COLUMNS)


def test_geojson_fallback_preserves_empty_layers(aoi, tmp_path, payload, monkeypatch):
    payload["elements"] = payload["elements"][:1]

    def fail_gpkg(*args, **kwargs):
        raise OSError("Synthetic driver failure")

    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", fail_gpkg)
    result = build(aoi, tmp_path, payload)
    restored = osm.load_osm(result.cache_path)
    assert any("GeoPackage writing failed" in warning for warning in restored.warnings)
    assert restored.snapshot_verified
    assert len(restored.roads) == 1
    assert restored.hospitals.empty
    assert restored.roads.osm_id.tolist() == result.roads.osm_id.tolist()


def test_invalid_polygon_is_repaired(aoi, tmp_path, payload):
    payload["elements"][5]["geometry"] = [{"lon": x, "lat": y} for x, y in [
        (85.002, 27.903), (85.004, 27.905), (85.002, 27.905), (85.004, 27.903), (85.002, 27.903),
    ]]
    result = build(aoi, tmp_path, payload)
    assert result.buildings.geometry.is_valid.all()
    assert result.quality["repaired_polygons"] == 1


def test_empty_tiles_do_not_invalidate_nonempty_snapshot(aoi, tmp_path, payload):
    large = replace(aoi, bbox=(85, 27.9, 85.25, 28.125))
    tiles = osm.tile_bbox(large.bbox)
    session = Session([Response(payload)] + [Response({"elements": []})] * (len(tiles) - 1))
    result = osm.build_osm(large, cache_dir=tmp_path, session=session, context_buffer_km=0, endpoints=ENDPOINTS)
    assert result.snapshot_verified


@pytest.mark.parametrize("kwargs", [{"snapshot_offset_days": -1}, {"context_buffer_km": -1}, {"strict": None}])
def test_invalid_parameters(aoi, tmp_path, kwargs):
    with pytest.raises(ValueError):
        osm.build_osm(aoi, cache_dir=tmp_path, **kwargs)
