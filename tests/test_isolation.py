"""Baseline-safe loss of hospital/town access, detours, and cluster restoration."""
import pytest

from src.graph.isolation import analyze_isolation, scenario_graph
from src.graph.road_graph import build_road_graph
from test_road_graph import A, B, C, D, bundle, flood, pipeline, road


def test_only_bridge_cutoff_cluster_and_sufficient_restoration():
    _, _, network, result = pipeline()
    assert all(row["status"] == "CUT OFF" for row in result.settlements)
    assert len(result.clusters) == 1 and len(result.clusters[0]["member_settlements"]) == 2
    bridge = network.graph[A][B]["link_id"]
    assert result.clusters[0]["responsible_links"] == [{"link_id": bridge, "sufficient_to_restore": True}]
    assert result.clusters[0]["total_mapped_buildings"] == 1
    assert all(bridge in row["responsible_links"] and row["confidence_tier"] == 3 for row in result.settlements)


def test_tier_one_loss_is_uncertain_not_cutoff():
    _, _, _, result = pipeline(fm=flood(tier=1))
    assert all(row["status"] == "UNCERTAIN" for row in result.settlements)
    assert all(not row["cut_from"] and row["possible_cut_from"] for row in result.settlements)


def test_baseline_missing_and_snap_failed_never_cutoff():
    _, _, _, result = pipeline(osm=bundle(hospitals=[]))
    assert not any(row["baseline_connected"]["hospital"] for row in result.settlements)
    # Town access can still be cut; hospital access cannot.
    assert all("hospital" not in row["cut_from"] for row in result.settlements)
    _, _, _, empty = pipeline(osm=bundle([], hospitals=[], towns=False))
    assert all(row["status"] == "UNCERTAIN" for row in empty.settlements)
    assert any(flag["code"] == "snap_far_settlements" for flag in empty.flags)
    _, _, _, baseline = pipeline(osm=bundle(hospitals=[], towns=False), fm=flood(empty=True))
    assert all(row["status"] == "UNCERTAIN" and "no baseline road access" in row["reason"] for row in baseline.settlements)


def test_outside_hospital_reachable_then_lost():
    _, _, _, baseline = pipeline(fm=flood(empty=True))
    assert all(row["status"] == "CONNECTED" for row in baseline.settlements)
    assert all(row["nearest_baseline"]["hospital"]["name"] == "Facility" for row in baseline.settlements)
    _, _, _, damaged = pipeline()
    assert all("hospital" in row["cut_from"] for row in damaged.settlements)


def test_alternative_route_detour_factor_and_full_threshold():
    osm = bundle()
    alternate = road(4, [A, (300000, 3100300), (300200, 3100300), B])
    osm.roads.loc[len(osm.roads)] = alternate
    osm.roads.set_crs(osm.crs, allow_override=True, inplace=True)
    _, _, _, result = pipeline(osm=osm)
    village = next(row for row in result.settlements if row["name"] == "Village")
    assert village["status"] == "CONNECTED" and village["access_detail"] == "degraded"
    assert village["detour_factors"]["hospital"] == pytest.approx(2.5)
    assert village["alternative_route_exists"] and village["nearest_after"]["hospital"]["geometry"]["type"] == "LineString"
    _, _, _, generous = pipeline(osm=osm, degraded_factor=3)
    assert all(row["access_detail"] == "full" for row in generous.settlements)


def test_town_subject_excludes_itself_as_destination():
    osm = bundle(towns=False)
    osm.settlements.loc[0, "place"] = "town"
    network = build_road_graph(osm, snap_max_m=50)
    result = analyze_isolation(network, osm.buildings)
    town = next(row for row in result.settlements if row["place"] == "town")
    assert not town["baseline_connected"]["town"] and town["nearest_baseline"]["town"] is None


def test_partially_hit_parallel_link_survives():
    _, _, network, _ = pipeline()
    edge = network.graph[A][B]
    unaffected = dict(edge["members"][0], member_id="safe", hit=False, damage_tier=0)
    edge["members"].append(unaffected)
    graph, removed = scenario_graph(network.graph, 2)
    assert graph.has_edge(A, B) and not removed
    assert len(graph[A][B]["members"]) == 1


def test_buildings_assigned_once_with_nearby_subjects():
    _, _, _, result = pipeline(catchment_m=300)
    assert sum(row["mapped_buildings_nearby"] for row in result.settlements) == 1
