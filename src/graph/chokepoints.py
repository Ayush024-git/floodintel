"""Single-link restoration leverage and latent mapped bridge access risks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from src.graph.isolation import CLASSES, IsolationResult, access_matrix, link_evidence
from src.graph.road_graph import RoadNetwork


@dataclass
class ChokepointResult:
    damaged: list[dict[str, Any]]
    latent: list[dict[str, Any]]
    params: dict[str, Any]
    flags: list[dict[str, str]]


def _restored_access(network: RoadNetwork, link: dict[str, Any],
                     components: dict[Any, int], targets: dict[tuple[int, str], set]) -> dict[str, dict[str, bool]]:
    """A single restored link only unions two precomputed components."""
    joined = {components[link["u"]], components[link["v"]]}
    result = {}
    for row in network.subjects:
        component = components.get(row["node"])
        reachable = joined if component in joined else {component}
        result[row["osm_id"]] = {kind: bool(set().union(*(targets.get((index, kind), set()) for index in reachable))
                                           - ({row["osm_id"]} if kind == "town" else set())) if row["snapped"] else False
                                 for kind in CLASSES}
    return result


def analyze_chokepoints(network: RoadNetwork, isolation: IsolationResult, *, include_latent: bool = True,
                        latent_limit: int = 200) -> ChokepointResult:
    """Measure access change, not physical failure; latent work is capped by stable link id.

    Latent bridge-link failure removes the entire merged link, including any
    parallel members. It represents corridor loss, not failure of one deck.
    """
    if type(latent_limit) is not int or latent_limit < 0:
        raise ValueError("latent_limit must be a nonnegative integer")
    baseline, after = access_matrix(network, network.graph), access_matrix(network, isolation.graphs["likely"])
    rows = {row["osm_id"]: row for row in isolation.settlements}
    graph = isolation.graphs["likely"]
    component_list = list(nx.connected_components(graph))
    components = {node: index for index, component in enumerate(component_list) for node in component}
    targets = {(index, kind): set() for index in range(len(component_list)) for kind in CLASSES}
    for destination in network.destinations:
        if destination["snapped"]:
            targets[components[destination["node"]], destination["destination_class"]].add(destination["osm_id"])

    def record(link: dict[str, Any], before: dict, changed: dict, damaged: bool) -> dict[str, Any]:
        changes = {identifier: [kind for kind in ("hospital", "town") if
                               baseline[identifier][kind] and not before[identifier][kind] and changed[identifier][kind]]
                   for identifier in rows} if damaged else {
                                   identifier: [kind for kind in ("hospital", "town") if before[identifier][kind] and not changed[identifier][kind]]
                                   for identifier in rows}
        identifiers = sorted(identifier for identifier in rows if changes[identifier])
        hospital_count = sum("hospital" in changes[identifier] for identifier in identifiers)
        town_count = sum("town" in changes[identifier] for identifier in identifiers)
        evidence = link_evidence(link, isolation.params["flood_pixel_area_m2"]) if damaged else None
        result = dict(link_id=link["link_id"], osm_ids=link["osm_ids"], name=link.get("name"), is_bridge=link["is_bridge"],
                      damaged=damaged, settlements_affected=[rows[value].get("name") or value for value in identifiers],
                      settlement_ids=identifiers, count=len(identifiers),
                      mapped_buildings=sum(rows[value]["mapped_buildings_nearby"] for value in identifiers),
                      hospital_access_lost=hospital_count,
                      hospital_access_reconnected=hospital_count if damaged else 0,
                      town_access_lost=town_count, town_access_reconnected=town_count if damaged else 0,
                      damage_tier=evidence["tier"] if evidence else 0, evidence=evidence, geometry=link["geometry"], sentence=None)
        if identifiers:
            action = "Restoring mapped link" if damaged else "Loss of mapped bridge link"
            outcome = "reconnects" if damaged else "disconnects"
            result["sentence"] = f"{action} {link.get('name') or link['link_id']} {outcome} {len(identifiers)} settlements: {hospital_count} hospital and {town_count} town accesses."
        return result

    damaged = []
    for link in isolation.removed["likely"]:
        restored = _restored_access(network, link, components, targets)
        damaged.append(record(link, after, restored, True))
    latent, flags = [], []
    removed_ids = {link["link_id"] for link in isolation.removed["possible"]}
    if include_latent:
        bridges = [dict(network.graph[u][v], u=u, v=v) for u, v in nx.bridges(network.graph)
                   if network.graph[u][v]["is_bridge"] and network.graph[u][v]["link_id"] not in removed_ids
                   and not any(member["hit"] for member in network.graph[u][v]["members"])]
        bridges.sort(key=lambda link: link["link_id"])
        if len(bridges) > latent_limit:
            flags.append(dict(code="latent_bridge_work_cap", severity="caution",
                              message=f"Latent bridge analysis capped at {latent_limit} of {len(bridges)} candidate links"))
        for link in bridges[:latent_limit]:
            view = nx.subgraph_view(network.graph, filter_edge=lambda u, v, link=link:
                                    {u, v} != {link["u"], link["v"]})
            latent.append(record(link, baseline, access_matrix(network, view), False))
    rank = lambda row: (-row["count"], -row["hospital_access_lost"],
                        -row["mapped_buildings"], row["link_id"])
    return ChokepointResult(sorted(damaged, key=rank), sorted(latent, key=rank),
                            dict(include_latent=include_latent, latent_limit=latent_limit,
                                 latent_parallel_policy="Entire merged bridge link/corridor lost"), flags)
