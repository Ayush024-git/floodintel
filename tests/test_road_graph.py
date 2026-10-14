"""Synthetic metric OSM fixtures and vertex/parallel road topology."""
from dataclasses import replace
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin
from pyproj import CRS as OSMCRS
from shapely.geometry import LineString, Point, box

from src.data.osm import OSMBundle
from src.graph.damage import assess_damage
from src.graph.road_graph import build_road_graph
from src.graph.isolation import analyze_isolation

A, B, C, D = (300000., 3100000.), (300200., 3100000.), (300400., 3100000.), (300400., 3100100.)
CRS_METRIC = CRS.from_epsg(32645)


def frame(rows, extra=()):
    columns = list(dict.fromkeys(["osm_id", "name", "in_aoi", *extra, "geometry", *(key for row in rows for key in row)]))
    return gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs=CRS_METRIC)


def road(identifier, coordinates, **flags):
    return dict(osm_id=f"way/{identifier}", segment_id=f"way/{identifier}#0", name=f"Road {identifier}",
                highway="primary", is_bridge=False, is_tunnel=False, is_ford=False,
                in_aoi=True, geometry=LineString(coordinates), **{key: value for key, value in flags.items() if key not in {"geometry"}})


def bundle(roads=None, subjects=None, hospitals=None, towns=True, buildings=True):
    if roads is None:
        first = road(1, [A, B]); first["is_bridge"] = True
        roads = [first, road(2, [B, C]), road(3, [C, D])]
    if subjects is None:
        subjects = [dict(osm_id="node/v", name="Village", place="village", in_aoi=True, geometry=Point(C)),
                    dict(osm_id="node/h", name="Hamlet", place="hamlet", in_aoi=True, geometry=Point(D))]
    if towns:
        subjects = [*subjects, dict(osm_id="node/t", name="Town", place="town", in_aoi=False, geometry=Point(A))]
    if hospitals is None:
        hospitals = [dict(osm_id="node/hospital", name="Facility", tier="hospital", in_aoi=False, geometry=Point(A))]
    building_rows = [dict(osm_id="way/building", name=None, in_aoi=True, geometry=box(C[0] + 1, C[1] + 1, C[0] + 5, C[1] + 5))] if buildings else []
    return OSMBundle(roads=frame(roads, ["segment_id", "highway", "is_bridge", "is_tunnel", "is_ford"]),
                     hospitals=frame(hospitals, ["tier"]), settlements=frame(subjects, ["place"]),
                     buildings=frame(building_rows), snapshot_date="2026-08-10T00:00:00Z", snapshot_verified=True,
                     newest_edit="2026-08-09T00:00:00Z", endpoint_used="fixture", crs=OSMCRS.from_user_input(CRS_METRIC),
                     bbox=(84.9, 27.8, 85.3, 28.1), quality={"include_buildings": buildings}, warnings=[])


def flood(tier=3, empty=False):
    mask = np.zeros((200, 200), bool)
    if not empty:
        mask[100, 5:15] = True
    return SimpleNamespace(flood_mask=mask, confidence=mask.astype(np.uint8) * tier,
                           valid_mask=np.ones_like(mask), transform=from_origin(300000, 3101000, 10, 10),
                           crs=CRS_METRIC, resolution=10., post_vv_db=np.full(mask.shape, -20., np.float32),
                           stats={"flood_area_km2": float(mask.sum() / 10000)}, params={}, flags=[], warnings=[], cache_path=None)


def pipeline(osm=None, fm=None, **params):
    osm, fm = osm or bundle(), fm or flood()
    damage = assess_damage(osm, fm, approach_buffer_m=0)
    network = build_road_graph(osm, damage, snap_max_m=50)
    isolated = analyze_isolation(network, damage.buildings, **params)
    return osm, damage, network, isolated


def test_shared_way_vertex_splits_but_geometric_crossing_does_not_connect():
    roads = [road(1, [(0, 0), (100, 0), (200, 0)]), road(2, [(100, 0), (100, 100)]),
             road(3, [(50, -100), (50, 100)])]
    result = build_road_graph(bundle(roads, subjects=[], hospitals=[], towns=False), snap_max_m=1)
    assert (100., 0.) in result.graph and (50., 0.) not in result.graph
    assert result.graph.number_of_edges() == 4
    assert result.quality["components"] == 2


def test_parallel_members_merge_and_preserve_all_ids_and_bridge():
    first, second = road(1, [A, B]), road(2, [A, B])
    second["is_bridge"] = True
    result = build_road_graph(bundle([first, second]))
    edge = result.graph[A][B]
    assert result.graph.number_of_edges() == 1 and edge["is_bridge"]
    assert edge["osm_ids"] == ["way/1", "way/2"] and len(edge["members"]) == 2
    assert edge["segment_ids"] == ["way/1#0", "way/2#0"]
    assert edge["travel_time_s"] == pytest.approx(200 / (50 / 3.6))


def test_vertex_rounding_drivable_filter_and_snap_bound():
    first, second = road(1, [A, B]), road(2, [(B[0] + .001, B[1]), C])
    foot = road(3, [C, D]); foot["highway"] = "path"
    osm = bundle([first, second, foot])
    result = build_road_graph(osm, snap_max_m=10)
    assert result.quality["components"] == 1 and result.quality["excluded_road_segments"] == 1
    hamlet = next(row for row in result.subjects if row["name"] == "Hamlet")
    assert not hamlet["snapped"] and "exceeds" in hamlet["snap_reason"]
    assert result.params["oneway_ignored"]


def test_context_roads_and_destination_are_kept():
    osm = bundle()
    osm.roads.loc[0, "in_aoi"] = False
    result = build_road_graph(osm, snap_max_m=50)
    assert result.graph.has_edge(A, B)
    assert any(row["destination_class"] == "hospital" and not row["in_aoi"] for row in result.destinations)


def test_empty_graph_is_safe_and_fragmented_is_flagged():
    result = build_road_graph(bundle([], hospitals=[], towns=False))
    assert result.quality["nodes"] == 0 and not any(row["snapped"] for row in result.subjects)
    assert result.flags[0]["code"] == "osm_road_fragmentation"


def test_closed_way_does_not_add_artificial_junctions():
    result = build_road_graph(bundle([road(1, [A, B, C, A])], subjects=[], hospitals=[], towns=False))
    assert result.graph.number_of_nodes() == 1 and result.graph.has_edge(A, A)
