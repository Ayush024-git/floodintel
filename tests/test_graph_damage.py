"""Inundation intersection, raster evidence, and bulk building performance."""
from time import perf_counter
from dataclasses import replace

import geopandas as gpd
import numpy as np
import pytest
from shapely import box as boxes
from shapely.geometry import LineString

from src.graph.damage import assess_damage, FloodExposure
from test_road_graph import A, B, bundle, flood, frame


def test_road_overlap_threshold_and_fraction_and_tier():
    exposure = FloodExposure(flood(), approach_buffer_m=0)
    values = exposure.measure_line(LineString([A, B]))
    assert values["hit"] and values["length_in_flood_m"] == pytest.approx(100)
    assert values["fraction_in_flood"] == pytest.approx(.5)
    assert values["max_tier"] == values["median_tier"] == values["damage_tier"] == 3
    narrow = LineString([(300040, 3100000), (300060, 3100000)])
    below_threshold = FloodExposure(flood(), min_overlap_m=30, min_fraction=.6).measure_line(narrow)
    assert below_threshold["hit"] is False and below_threshold["length_in_flood_m"] > 0
    assert below_threshold["evidence_label"] == "no qualifying inundation hit"
    assert FloodExposure(flood(), min_overlap_m=30, min_fraction=.1).measure_line(narrow)["hit"]


def test_bridge_approach_buffer_and_tunnel_endpoint_rule_and_ford():
    exposure = FloodExposure(flood(), approach_buffer_m=30)
    bridge = LineString([(300050, 3100020), (300150, 3100020)])
    assert not exposure.measure_line(bridge)["hit"]
    assert exposure.measure_line(bridge, is_bridge=True)["bridge_approach_hit"]
    assert not exposure.measure_line(LineString([A, B]), is_tunnel=True)["hit"]
    approach = LineString([(300060, 3100000), B])
    assert exposure.measure_line(approach, is_tunnel=True)["hit"]
    assert exposure.measure_line(LineString([A, B]), is_ford=True)["damage_tier"] == 1


def test_building_point_intersection_and_stable_ids():
    osm = bundle()
    osm.buildings.geometry = [boxes(300055, 3099995, 300065, 3100005)]
    result = assess_damage(osm, flood(), approach_buffer_m=0)
    assert result.buildings.iloc[0].hit and result.buildings.iloc[0].damage_tier == 3
    assert result.buildings.iloc[0].osm_id == "way/building"
    assert list(result.bridges.osm_id) == ["way/1"]
    assert result.summary["counts"]["roads"] == 1
    assert result.summary["road_km_by_highway"]["primary"] == pytest.approx(.2)


def test_building_bulk_intersection_100k_under_60_seconds():
    osm = bundle()
    indices = np.arange(100000)
    x = 300000 + indices % 1000
    y = 3099990 + indices // 1000
    footprints = boxes(x, y, x + .5, y + .5)
    buildings = gpd.GeoDataFrame({"osm_id": [f"way/{value}" for value in indices], "geometry": footprints}, crs=osm.crs)
    started = perf_counter()
    tiers = FloodExposure(flood()).polygon_tiers(buildings)
    assert perf_counter() - started < 60
    assert np.count_nonzero(tiers) > 0 and np.count_nonzero(tiers) < 100000
    assert set(np.unique(tiers)) == {0, 3}
