"""Vertex-connected, undirected drivable OSM roads with explicit parallel members."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
from rasterio.crs import CRS
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

SPEED_KPH = {"motorway": 80., "trunk": 60., "primary": 50., "secondary": 40., "tertiary": 30.,
             "unclassified": 25., "residential": 20., "living_street": 10., "service": 15., "track": 10.,
             "motorway_link": 40., "trunk_link": 30., "primary_link": 25., "secondary_link": 20., "tertiary_link": 15.}


@dataclass
class RoadNetwork:
    graph: nx.Graph
    subjects: list[dict[str, Any]]
    destinations: list[dict[str, Any]]
    quality: dict[str, Any]
    params: dict[str, Any]
    flags: list[dict[str, str]]


def update_link(attributes: dict[str, Any], members: list[dict[str, Any]]) -> None:
    """Use the fastest surviving member; capacity loss is not modeled."""
    preferred = min(members, key=lambda row: (row["travel_time_s"], row["member_id"]))
    for key in ("length_m", "travel_time_s", "geometry", "highway", "name"):
        attributes[key] = preferred[key]
    attributes.update(members=members, segment_ids=sorted({row["segment_id"] for row in members}),
                      osm_ids=sorted({row["osm_id"] for row in members}),
                      is_bridge=any(row["is_bridge"] for row in members),
                      is_ford=any(row["is_ford"] for row in members),
                      is_tunnel=any(row["is_tunnel"] for row in members),
                      damage_tier_max=max(row.get("damage_tier", 0) for row in members))


def snap_points(frame: gpd.GeoDataFrame, graph: nx.Graph, snap_max_m: float) -> list[dict[str, Any]]:
    """Nearest node within a metric bound, with deterministic tie handling."""
    nodes = sorted(graph.nodes)
    tree = STRtree([Point(node) for node in nodes]) if nodes else None
    output = []
    for row in frame.sort_values("osm_id").to_dict("records"):
        row = dict(row)
        point = row["geometry"]
        row.update(snapped=False, node=None, snap_distance_m=None, snap_reason="road graph is empty")
        if tree is not None:
            indices = tree.query_nearest(point, all_matches=True)
            index = min(indices, key=lambda index: nodes[index])
            node = nodes[index]
            distance = float(point.distance(Point(node)))
            row.update(snap_distance_m=distance, snap_reason="nearest road node exceeds snap_max_m")
            if distance <= snap_max_m:
                row.update(snapped=True, node=node, snap_reason=None)
        output.append(row)
    return output


def build_road_graph(osm: Any, damage: Any = None, *, snap_max_m: float = 1500,
                     coordinate_precision_m: float = .01, speed_kph: dict[str, float] | None = None,
                     fragmentation_threshold: float = .7) -> RoadNetwork:
    """Exclude path/footway/steps/cycleway/unknown classes; ignore oneway and vehicle restrictions.

    Nodes occur at endpoints and vertices shared by different OSM ways. Geometric
    crossings without a shared vertex never connect. Parallel members remain
    individually auditable; a partially hit parallel link remains usable.
    Travel speeds are illustrative constant class speeds, not observed traffic.
    """
    if (not np.isfinite(snap_max_m) or snap_max_m < 0 or not np.isfinite(coordinate_precision_m)
            or coordinate_precision_m <= 0 or not 0 <= fragmentation_threshold <= 1):
        raise ValueError("Invalid snapping/coordinate/fragmentation parameters")
    speeds = dict(SPEED_KPH)
    if speed_kph:
        if set(speed_kph) - set(speeds):
            raise ValueError("Speed overrides must use supported drivable highway classes")
        speeds.update(speed_kph)
    if any(not np.isfinite(value) or value <= 0 for value in speeds.values()):
        raise ValueError("Travel speeds must be finite and positive")
    crs = CRS.from_user_input(osm.crs)
    if not crs.is_projected or crs.linear_units_factor[1] != 1:
        raise ValueError("Road graph requires a working CRS in meters")
    roads = osm.roads.loc[osm.roads.highway.isin(speeds)].sort_values("segment_id")
    owners: dict[tuple, set] = defaultdict(set)
    lines = []
    for row in roads.to_dict("records"):
        if row["geometry"].geom_type != "LineString":
            raise ValueError("OSM road segments must be LineStrings")
        coords = [tuple(round(float(value) / coordinate_precision_m) * coordinate_precision_m
                        for value in xy[:2]) for xy in row["geometry"].coords]
        coords = [xy for index, xy in enumerate(coords) if not index or xy != coords[index - 1]]
        if len(coords) < 2:
            continue
        lines.append((row, coords))
        for xy in set(coords):
            owners[xy].add(row["osm_id"])
    graph = nx.Graph()
    graph.graph["crs"] = str(osm.crs)
    for row, coords in lines:
        cuts = [0] + [index for index in range(1, len(coords) - 1) if len(owners[coords[index]]) > 1] + [len(coords) - 1]
        for part, (start, end) in enumerate(zip(cuts, cuts[1:])):
            geometry = LineString(coords[start:end + 1])
            u, v = coords[start], coords[end]
            # Closed ways without a shared junction retain their endpoint self loop;
            # no artificial midpoint nodes or geometric crossing junctions are added.
            sections = [(u, v, geometry)]
            for sub, (u, v, geometry) in enumerate(sections):
                if geometry.length == 0:
                    continue
                member = {key: row.get(key) for key in ("segment_id", "osm_id", "highway", "name")}
                member.update(member_id=f"{row['segment_id']}:{part}:{sub}", geometry=geometry,
                              length_m=float(geometry.length), travel_time_s=float(geometry.length / (speeds[row["highway"]] / 3.6)),
                              **{key: bool(row.get(key, False)) for key in ("is_bridge", "is_ford", "is_tunnel")})
                member.update(damage.exposure.measure_line(geometry, is_bridge=member["is_bridge"],
                              is_ford=member["is_ford"], is_tunnel=member["is_tunnel"]) if damage else
                              dict(hit=False, damage_tier=0, fraction_in_flood=0., length_in_flood_m=0.))
                endpoints = sorted((u, v))
                link_id = "link_" + hashlib.sha256(repr(endpoints).encode()).hexdigest()[:12]
                if not graph.has_edge(u, v):
                    graph.add_edge(u, v, link_id=link_id, members=[])
                attrs = graph[u][v]
                update_link(attrs, [*attrs["members"], member])
    components = list(nx.connected_components(graph))
    total = sum(sum(member["length_m"] for member in data["members"]) for _, _, data in graph.edges(data=True))
    lengths = [sum(sum(member["length_m"] for member in data["members"])
                   for _, _, data in graph.subgraph(component).edges(data=True)) for component in components]
    largest_share = max(lengths) / total if total else 0.
    subjects = snap_points(osm.settlements.loc[osm.settlements.in_aoi], graph, snap_max_m)
    hospital_points = osm.hospitals.copy()
    hospital_points["destination_class"] = hospital_points.tier
    town_points = osm.settlements.loc[osm.settlements.place.isin(["city", "town"])].copy()
    town_points["destination_class"] = "town"
    destination_rows = hospital_points.to_dict("records") + town_points.to_dict("records")
    destinations = snap_points(gpd.GeoDataFrame(destination_rows,
                                              columns=list(dict.fromkeys([*hospital_points.columns, *town_points.columns])),
                                              geometry="geometry", crs=osm.crs), graph, snap_max_m)
    flags = []
    if largest_share < fragmentation_threshold:
        flags.append({"code": "osm_road_fragmentation", "severity": "caution",
                      "message": f"Largest road component contains {largest_share:.1%} of mapped road length"})
    quality = dict(nodes=graph.number_of_nodes(), links=graph.number_of_edges(), components=len(components),
                   largest_component_road_length_share=largest_share,
                   destinations_snapped={kind: sum(row["snapped"] and row["destination_class"] == kind for row in destinations)
                                         for kind in ("hospital", "clinic", "town")},
                   subjects_snapped=sum(row["snapped"] for row in subjects), excluded_road_segments=len(osm.roads) - len(roads),
                   parallel_links=sum(len(data["members"]) > 1 for _, _, data in graph.edges(data=True)))
    return RoadNetwork(graph, subjects, destinations, quality,
                       dict(snap_max_m=snap_max_m, coordinate_precision_m=coordinate_precision_m,
                            speed_kph=speeds, fragmentation_threshold=fragmentation_threshold,
                            drivable_classes=sorted(speeds), oneway_ignored=True, vehicle_restrictions_ignored=True,
                            partial_parallel_capacity_ignored=True), flags)
