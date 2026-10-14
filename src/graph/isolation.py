"""Baseline-aware, scenario-specific mapped road access and restoration evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
from rasterio.warp import transform_geom
from shapely.geometry import LineString, mapping
from shapely.strtree import STRtree

from src.graph.road_graph import RoadNetwork, update_link

CLASSES = ("hospital", "clinic", "town")


def scenario_graph(graph: nx.Graph, minimum_tier: int) -> tuple[nx.Graph, list[dict[str, Any]]]:
    """Remove a link only when every parallel member is hit at the scenario tier."""
    result = graph.copy()
    removed = []
    for u, v, attributes in sorted(graph.edges(data=True), key=lambda row: row[2]["link_id"]):
        survivors = [member for member in attributes["members"]
                     if not member["hit"] or member["damage_tier"] < minimum_tier]
        if not survivors:
            removed.append(dict(attributes, u=u, v=v))
            result.remove_edge(u, v)
        else:
            update_link(result[u][v], survivors)
    return result, removed


def access_matrix(network: RoadNetwork, graph: nx.Graph) -> dict[str, dict[str, bool]]:
    """Connected-component reachability, excluding a town subject itself."""
    components = list(nx.connected_components(graph))
    membership = {node: index for index, component in enumerate(components) for node in component}
    targets = {(index, kind): set() for index in range(len(components)) for kind in CLASSES}
    for destination in network.destinations:
        if destination["snapped"] and destination["node"] in membership:
            targets[membership[destination["node"]], destination["destination_class"]].add(destination["osm_id"])
    return {subject["osm_id"]: {
        kind: bool(targets.get((membership.get(subject["node"]), kind), set())
                   - ({subject["osm_id"]} if kind == "town" else set())) if subject["snapped"] else False
        for kind in CLASSES} for subject in network.subjects}


class Routes:
    """Cache multi-source Dijkstra by destination class and self-town exclusion."""

    def __init__(self, network: RoadNetwork, graph: nx.Graph):
        self.network, self.graph, self.cache = network, graph, {}

    def nearest(self, subject: dict[str, Any], kind: str) -> dict[str, Any] | None:
        if not subject["snapped"]:
            return None
        exclusion = subject["osm_id"] if kind == "town" and subject["place"] in ("town", "city") else None
        key = kind, exclusion
        destinations = [row for row in self.network.destinations if row["snapped"]
                        and row["destination_class"] == kind and row["osm_id"] != exclusion]
        if key not in self.cache:
            sources = sorted({row["node"] for row in destinations})
            self.cache[key] = nx.multi_source_dijkstra(self.graph, sources, weight="travel_time_s") if sources else ({}, {})
        times, paths = self.cache[key]
        if subject["node"] not in times:
            return None
        path = list(reversed(paths[subject["node"]]))
        destination = min((row for row in destinations if row["node"] == path[-1]), key=lambda row: row["osm_id"])
        coordinates, link_ids, length = [], [], 0.
        for u, v in zip(path, path[1:]):
            edge = self.graph[u][v]
            points = list(edge["geometry"].coords)
            if np.linalg.norm(np.asarray(points[0]) - u) > np.linalg.norm(np.asarray(points[-1]) - u):
                points.reverse()
            coordinates.extend(points if not coordinates else points[1:])
            link_ids.append(edge["link_id"])
            length += edge["length_m"]
        geometry = (transform_geom(self.network.graph.graph["crs"], "EPSG:4326", mapping(LineString(coordinates)))
                    if len(coordinates) > 1 else None)
        return dict(osm_id=destination["osm_id"], name=destination.get("name"), travel_time_s=float(times[subject["node"]]),
                    length_m=float(length), link_ids=link_ids, geometry=geometry)


def building_counts(network: RoadNetwork, buildings: Any, catchment_m: float) -> dict[str, int]:
    """Assign each footprint to its nearest subject within catchment; counts are size proxies."""
    counts = {row["osm_id"]: 0 for row in network.subjects}
    subjects = network.subjects
    if not subjects or not len(buildings):
        return counts
    tree = STRtree([row["geometry"] for row in subjects])
    for geometry in buildings.geometry:
        indices = tree.query_nearest(geometry, all_matches=True)
        index = min(indices, key=lambda value: subjects[value]["osm_id"])
        if geometry.distance(subjects[index]["geometry"]) <= catchment_m:
            counts[subjects[index]["osm_id"]] += 1
    return counts


def link_evidence(link: dict[str, Any], pixel_area_m2: float) -> dict[str, Any]:
    return dict(link_id=link["link_id"], osm_ids=link["osm_ids"],
                tier=min((member["damage_tier"] for member in link["members"] if member["hit"]), default=0),
                parallel_member_count=len(link["members"]),
                flood_pixel_area_m2=pixel_area_m2,
                members=[{key: member.get(key) for key in ("member_id", "segment_id", "osm_id", "fraction_in_flood",
                                                         "length_in_flood_m", "damage_tier", "is_ford", "approach_hit")}
                         for member in link["members"]])


@dataclass
class IsolationResult:
    settlements: list[dict[str, Any]]
    clusters: list[dict[str, Any]]
    graphs: dict[str, nx.Graph]
    removed: dict[str, list[dict[str, Any]]]
    params: dict[str, Any]
    flags: list[dict[str, str]]


def analyze_isolation(network: RoadNetwork, buildings: Any, *, degraded_factor: float = 1.5,
                      catchment_m: float = 300, baseline_warning_share: float = .2,
                      flood_pixel_area_m2: float = 100) -> IsolationResult:
    """Lost baseline hospital/town access is CUT OFF only in the likely scenario.

    Missing baseline mapping remains UNCERTAIN even when no flood is detected.
    Clinics are tracked separately and never substitute for a hospital. Snapping
    is a proximity heuristic, not evidence of a driveway or traversable entrance.
    """
    if (not np.isfinite(degraded_factor) or degraded_factor < 1 or not np.isfinite(catchment_m) or catchment_m < 0
            or not 0 <= baseline_warning_share <= 1 or not np.isfinite(flood_pixel_area_m2) or flood_pixel_area_m2 <= 0):
        raise ValueError("Invalid isolation detour/catchment/baseline/pixel-area parameters")
    likely, removed_likely = scenario_graph(network.graph, 2)
    possible, removed_possible = scenario_graph(network.graph, 1)
    graphs = {"baseline": network.graph, "likely": likely, "possible": possible}
    removed = {"likely": removed_likely, "possible": removed_possible}
    routing = {name: Routes(network, graph) for name, graph in graphs.items()}
    counts = building_counts(network, buildings, catchment_m)
    results = []
    for subject in network.subjects:
        routes = {name: {kind: service.nearest(subject, kind) for kind in CLASSES} for name, service in routing.items()}
        connected = {kind: routes["baseline"][kind] is not None for kind in CLASSES}
        lost = {name: [kind for kind in ("hospital", "town") if connected[kind] and routes[name][kind] is None]
                for name in ("likely", "possible")}
        missing = [kind for kind in ("hospital", "town") if not connected[kind]]
        if not subject["snapped"]:
            status, reason = "UNCERTAIN", subject["snap_reason"]
        elif lost["likely"]:
            status, reason = "CUT OFF", "lost baseline mapped road access in the likely inundation scenario"
        elif lost["possible"]:
            status, reason = "UNCERTAIN", "access lost only with uncertain-tier inundation evidence"
        elif missing:
            status, reason = "UNCERTAIN", "no baseline road access in OSM: " + ", ".join(missing)
        else:
            status, reason = "CONNECTED", "baseline mapped hospital and town access retained"
        detours = {}
        detail = {}
        for kind in CLASSES:
            before, after = routes["baseline"][kind], routes["likely"][kind]
            ratio = (after["travel_time_s"] / before["travel_time_s"] if before and after and before["travel_time_s"] > 0
                     else (1. if before and after else None))
            detours[kind] = ratio
            detail[kind] = "unmapped" if not before else "none" if not after else "degraded" if ratio > degraded_factor else "full"
        scenario = "likely" if lost["likely"] else "possible"
        baseline_links = {link for kind in ("hospital", "town") for link in
                          (routes["baseline"][kind]["link_ids"] if routes["baseline"][kind] else [])}
        responsible = [dict(link, u=u, v=v) for u, v, link in network.graph.edges(data=True)
                       if link["link_id"] in baseline_links
                       and min(link["members"], key=lambda member: (member["travel_time_s"], member["member_id"]))["hit"]]
        # A tier-1-only disturbance on a baseline route remains uncertain even if rerouting retains access.
        if status == "CONNECTED" and responsible and all(link_evidence(link, flood_pixel_area_m2)["tier"] == 1 for link in responsible):
            status, reason = "UNCERTAIN", "baseline route has only uncertain-tier inundation evidence"
            scenario = "possible"
        evidence = [link_evidence(link, flood_pixel_area_m2) for link in responsible]
        responsible_tier = min((item["tier"] for item in evidence), default=0)
        access_detail = ("unmapped" if not subject["snapped"] else "none" if lost["likely"] else
                         "unmapped" if missing else "degraded" if any(detail[kind] == "degraded" for kind in ("hospital", "town")) else "full")
        results.append(dict(osm_id=subject["osm_id"], name=subject.get("name"), place=subject["place"],
                            geometry=subject["geometry"], node=subject["node"], snapped=subject["snapped"],
                            snap_distance_m=subject["snap_distance_m"], status=status, reason=reason,
                            baseline_connected=connected, access_detail=access_detail, access_by_class=detail,
                            nearest_baseline=routes["baseline"], nearest_after=routes["likely"],
                            nearest_possible=routes["possible"], detour_factors=detours, cut_from=lost["likely"],
                            possible_cut_from=lost["possible"], responsible_links=[item["link_id"] for item in evidence],
                            number_of_broken_routes=sum(bool(set(routes["baseline"][kind]["link_ids"]) &
                                                            {link["link_id"] for link in responsible})
                                                       for kind in ("hospital", "town") if routes["baseline"][kind]),
                            alternative_route_exists=bool(responsible and all(routes[scenario][kind] is not None
                                                                        for kind in ("hospital", "town") if connected[kind])),
                            confidence_tier=responsible_tier, evidence=evidence,
                            mapped_buildings_nearby=counts[subject["osm_id"]], cluster_mapped_buildings=counts[subject["osm_id"]],
                            size_proxy_label="Mapped building count; size proxy only", cluster_ids=[]))
    components = list(nx.connected_components(likely))
    membership = {node: index for index, component in enumerate(components) for node in component}
    grouped = {}
    for row in results:
        for kind in row["cut_from"]:
            group = grouped.setdefault(membership[row["node"]], {})
            group[row["osm_id"]] = row
    clusters = []
    for component_index, member_dict in sorted(grouped.items()):
        members = list(member_dict.values())
        lost_classes = sorted({kind for row in members for kind in row["cut_from"]})
        component = components[component_index]
        boundary = [link for link in removed_likely if (link["u"] in component) != (link["v"] in component)]
        cluster_id = f"cluster_{component_index}"
        buildings_total = sum(row["mapped_buildings_nearby"] for row in members)
        verification = []
        for link in boundary:
            restored = likely.copy()
            restored.add_edge(link["u"], link["v"], **{key: value for key, value in link.items() if key not in {"u", "v"}})
            access = access_matrix(network, restored)
            verification.append(dict(link_id=link["link_id"], sufficient_to_restore=all(
                access[row["osm_id"]][kind] for row in members for kind in row["cut_from"])))
        for row in members:
            row["cluster_ids"].append(cluster_id)
            row["cluster_mapped_buildings"] = max(row["cluster_mapped_buildings"], buildings_total)
            row["responsible_links"] = sorted(set(row["responsible_links"]) | {link["link_id"] for link in boundary})
            existing = {item["link_id"] for item in row["evidence"]}
            row["evidence"].extend(link_evidence(link, flood_pixel_area_m2) for link in boundary if link["link_id"] not in existing)
            row["confidence_tier"] = min(item["tier"] for item in row["evidence"]) if row["evidence"] else 0
        clusters.append(dict(cluster_id=cluster_id, lost_classes=lost_classes,
                             member_settlements=[row["osm_id"] for row in members],
                             total_mapped_buildings=buildings_total,
                             responsible_links=verification))
    flags = []
    missing_share = sum(not all(row["baseline_connected"][kind] for kind in ("hospital", "town")) for row in results) / len(results) if results else 0.
    if missing_share > baseline_warning_share:
        flags.append(dict(code="baseline_disconnected_share", severity="caution",
                          message=f"{missing_share:.1%} of subjects lack baseline mapped hospital or town access; this cannot establish flood isolation"))
    if any(not row["snapped"] for row in results):
        flags.append(dict(code="snap_far_settlements", severity="caution", message="Some subject settlements cannot snap within snap_max_m"))
    return IsolationResult(results, clusters, graphs, removed,
                           dict(degraded_factor=degraded_factor, catchment_m=catchment_m,
                                baseline_warning_share=baseline_warning_share, flood_pixel_area_m2=flood_pixel_area_m2,
                                likely_min_tier=2, possible_min_tier=1), flags)
