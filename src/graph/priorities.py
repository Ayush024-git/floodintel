"""Transparent map/evidence priorities; no inference of humanitarian need."""
from __future__ import annotations

import math
from typing import Any

SETTLEMENT_WEIGHTS = dict(status=1., hospital=50., town=20., buildings=10., evidence=5., detour=5.)
LINK_WEIGHTS = dict(settlements=10., buildings=5., hospital=20., town=10., evidence=3.)
DISCLAIMER = "Ranking reflects satellite- and map-derived evidence only, not casualties or humanitarian need."


def rank_priorities(isolation: Any, chokepoints: Any, *, settlement_weights: dict[str, float] | None = None,
                    link_weights: dict[str, float] | None = None, detour_cap: float = 10.,
                    building_log_cap: float = 3., status_scores: dict[str, float] | None = None) -> dict[str, Any]:
    """Status 100/40/20/0, access losses, capped log10(buildings+1), tier, detour.

    Settlement eligibility requires CUT OFF, a hit-associated detour, or
    tier-1/possible-scenario loss of established baseline access. Baseline-only
    uncertainty is a data gap, never a flood priority. Link scores require
    nonzero restoration leverage or latent access at risk.
    """
    def weights(defaults: dict, overrides: dict | None) -> dict:
        if overrides and set(overrides) - set(defaults):
            raise ValueError("Unknown priority weight")
        values = {**defaults, **(overrides or {})}
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("Priority weights must be finite and nonnegative")
        return values

    sw, lw = weights(SETTLEMENT_WEIGHTS, settlement_weights), weights(LINK_WEIGHTS, link_weights)
    statuses = weights({"CUT OFF": 100., "UNCERTAIN": 40., "degraded": 20., "CONNECTED": 0.}, status_scores)
    if not math.isfinite(detour_cap) or detour_cap < 1 or not math.isfinite(building_log_cap) or building_log_cap < 0:
        raise ValueError("Invalid priority detour/building caps")

    def finalize(row: dict, components: dict, values: dict, weights: dict, identifier: str, tier: int) -> dict:
        breakdown = {key: dict(value=values[key], transformed=components[key], weight=weights[key],
                              contribution=components[key] * weights[key]) for key in components}
        return dict(id=identifier, name=row.get("name"), score=float(sum(value["contribution"] for value in breakdown.values())),
                    components=breakdown, confidence={3: "strong", 2: "probable"}.get(tier, "uncertain"),
                    why=[f"{key}={values[key]} contributes {breakdown[key]['contribution']:.3f}" for key in components])

    settlements, data_gaps = [], []
    for row in isolation.settlements:
        evidence = [item for item in row.get("evidence", []) if item.get("tier", 0) > 0
                    and item.get("link_id") in row.get("responsible_links", [])]
        has_hit = bool(evidence)
        baseline = row.get("baseline_connected", {})
        baseline_complete = all(baseline.get(kind, False) for kind in ("hospital", "town"))
        possible_loss = any(baseline.get(kind, False) for kind in row.get("possible_cut_from", []))
        degraded = (row["access_detail"] == "degraded"
                    or "degraded" in row.get("access_by_class", {}).values())
        attributable = (row["status"] == "CUT OFF"
                        or (row.get("snapped", False) and has_hit and degraded)
                        or (row["status"] == "UNCERTAIN" and row.get("snapped", False) and has_hit
                            and (possible_loss or (baseline_complete and any(item["tier"] == 1 for item in evidence)))))
        if not attributable:
            if row["status"] == "UNCERTAIN" and (not row.get("snapped", False) or not baseline_complete):
                data_gaps.append(dict(id=row["osm_id"], name=row.get("name"), status=row["status"],
                                      reason=row.get("reason") or "Baseline road access or snapping is missing in OSM",
                                      snapped=row.get("snapped", False), baseline_connected=baseline))
            continue
        status = "degraded" if row["status"] == "CONNECTED" and row["access_detail"] == "degraded" else row["status"]
        detour = max((value for value in row["detour_factors"].values() if value is not None), default=1.)
        values = dict(status=status, hospital=int("hospital" in row["cut_from"]), town=int("town" in row["cut_from"]),
                      buildings=row["cluster_mapped_buildings"], evidence=row["confidence_tier"], detour=detour)
        transformed = {**values, "status": statuses[status], "buildings": min(math.log10(values["buildings"] + 1), building_log_cap),
                       "detour": max(0., min(detour, detour_cap) - 1)}
        settlements.append(finalize(row, transformed, values, sw, row["osm_id"], row["confidence_tier"]))
    links, damaged_no_leverage = [], []
    for row in [*chokepoints.damaged, *chokepoints.latent]:
        if row["count"] <= 0:
            if row["damaged"]:
                damaged_no_leverage.append({key: value for key, value in row.items() if key != "geometry"})
            continue
        values = dict(settlements=row["count"], buildings=row["mapped_buildings"],
                      hospital=row["hospital_access_reconnected"] if row["damaged"] else row["hospital_access_lost"],
                      town=row["town_access_reconnected"] if row["damaged"] else row["town_access_lost"], evidence=row["damage_tier"])
        transformed = {**values, "buildings": min(math.log10(values["buildings"] + 1), building_log_cap)}
        result = finalize(row, transformed, values, lw, row["link_id"], row["damage_tier"])
        result["action"] = "restoration inspection" if row["damaged"] else "latent corridor inspection"
        links.append(result)
    for collection in (settlements, links):
        collection.sort(key=lambda row: (-row["score"], row["id"]))
        for rank, row in enumerate(collection, 1):
            row["rank"] = rank
    data_gaps.sort(key=lambda row: row["id"])
    damaged_no_leverage.sort(key=lambda row: row["link_id"])
    return dict(settlements=settlements, links=links, data_gaps=data_gaps,
                damaged_no_leverage=damaged_no_leverage,
                message=None if settlements else "No flood-attributable priorities", disclaimer=DISCLAIMER,
                params=dict(settlement_weights=sw, link_weights=lw, detour_cap=detour_cap,
                            building_log_cap=building_log_cap, status_scores=statuses))
