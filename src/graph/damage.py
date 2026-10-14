"""Potential inundation of mapped assets, never structural damage confirmation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union


class FloodExposure:
    """Indexed raster-cell polygons plus half-pixel line sampling in metric CRS."""

    def __init__(self, flood: Any, min_overlap_m: float = 20, min_fraction: float = .1,
                 approach_buffer_m: float = 30, sample_spacing_m: float | None = None):
        self.flood = flood
        self.params = dict(min_overlap_m=min_overlap_m, min_fraction=min_fraction,
                           approach_buffer_m=approach_buffer_m,
                           sample_spacing_m=sample_spacing_m or flood.resolution / 2)
        if (any(not np.isfinite(value) or value < 0 for value in self.params.values())
                or self.params["sample_spacing_m"] <= 0 or not 0 <= min_fraction <= 1):
            raise ValueError("Invalid exposure overlap/fraction/buffer/sampling parameters")
        if not flood.crs.is_projected or flood.crs.linear_units_factor[1] != 1:
            raise ValueError("Exposure requires a working CRS in meters")
        if flood.confidence.shape != flood.flood_mask.shape:
            raise ValueError("Flood mask/confidence shapes disagree")
        if np.any(flood.flood_mask & ((flood.confidence < 1) | (flood.confidence > 3))):
            raise ValueError("Flood pixels require confidence tiers 1..3")
        rows = [{"tier": int(tier), "geometry": shape(geometry)}
                for geometry, tier in shapes(flood.confidence.astype(np.uint8),
                                             mask=flood.flood_mask, transform=flood.transform,
                                             connectivity=4)]
        self.polygons = gpd.GeoDataFrame(rows, columns=["tier", "geometry"], crs=flood.crs)
        self.index = self.polygons.sindex

    def intersect(self, geometry: Any) -> tuple[Any, int]:
        indices = self.index.query(geometry, predicate="intersects")
        if not len(indices):
            return geometry.intersection(shape({"type": "GeometryCollection", "geometries": []})), 0
        rows = self.polygons.iloc[indices]
        return geometry.intersection(unary_union(rows.geometry.to_numpy())), int(rows.tier.max())

    def sample(self, line: Any) -> np.ndarray:
        distance = np.linspace(0, line.length, max(2, int(np.ceil(line.length / self.params["sample_spacing_m"])) + 1))
        points = [line.interpolate(float(value)) for value in distance]
        inverse = ~self.flood.transform
        coordinates = np.array([inverse @ (point.x, point.y) for point in points])
        cols, rows = np.floor(coordinates).astype(np.int64).T
        height, width = self.flood.confidence.shape
        inside = (rows >= 0) & (cols >= 0) & (rows < height) & (cols < width)
        result = np.zeros(len(points), np.uint8)
        result[inside] = np.where(self.flood.flood_mask[rows[inside], cols[inside]],
                                  self.flood.confidence[rows[inside], cols[inside]], 0)
        return result

    def measure_line(self, geometry: Any, *, is_bridge: bool = False,
                     is_tunnel: bool = False, is_ford: bool = False) -> dict[str, Any]:
        """Tunnels require flooded endpoint approaches; bridges include full buffered footprint."""
        intersection, polygon_tier = self.intersect(geometry)
        length = float(intersection.length)
        fraction = length / geometry.length if geometry.length else 0.
        sampled = self.sample(geometry)
        sampled = sampled[sampled > 0]
        maximum = int(sampled.max()) if len(sampled) else polygon_tier
        median = float(np.median(sampled)) if len(sampled) else float(polygon_tier)
        approach_hit, approach_tier = False, 0
        if is_bridge or is_tunnel:
            if is_tunnel:
                footprint = unary_union([geometry.interpolate(0).buffer(self.params["approach_buffer_m"]),
                                         geometry.interpolate(geometry.length).buffer(self.params["approach_buffer_m"])])
            else:
                footprint = geometry.buffer(self.params["approach_buffer_m"])
            _, approach_tier = self.intersect(footprint)
            approach_hit = approach_tier > 0
        direct_hit = length > 0 and (length >= self.params["min_overlap_m"] or fraction >= self.params["min_fraction"])
        hit = approach_hit if is_tunnel else direct_hit or (is_bridge and approach_hit)
        tier = max(maximum, approach_tier) if hit else 0
        if is_ford and hit:
            tier = 1  # Expected inundation of a ford is low evidence of loss of access.
        return dict(length_in_flood_m=length, fraction_in_flood=fraction, max_tier=maximum,
                    median_tier=median, damage_tier=tier, hit=bool(hit),
                    bridge_approach_hit=bool(is_bridge and approach_hit), approach_hit=bool(approach_hit),
                    low_evidence_ford=bool(is_ford and hit),
                    evidence_label="potentially affected / inundated" if hit else "no qualifying inundation hit")

    def polygon_tiers(self, frame: gpd.GeoDataFrame) -> np.ndarray:
        """Bulk spatial-index intersections scale to large building layers."""
        tiers = np.zeros(len(frame), np.uint8)
        if len(frame) and len(self.polygons):
            matches = self.index.query(frame.geometry.to_numpy(), predicate="intersects")
            np.maximum.at(tiers, matches[0], self.polygons.tier.to_numpy()[matches[1]])
        return tiers


@dataclass
class DamageResult:
    roads: gpd.GeoDataFrame
    buildings: gpd.GeoDataFrame
    hospitals: gpd.GeoDataFrame
    settlements: gpd.GeoDataFrame
    summary: dict[str, Any]
    params: dict[str, Any]
    exposure: FloodExposure

    @property
    def bridges(self) -> gpd.GeoDataFrame:
        return self.roads.loc[self.roads.is_bridge]


def assess_damage(osm: Any, flood: Any, *, include_buildings: bool = True, **params: Any) -> DamageResult:
    """Measure flood-extent exposure; intersections do not demonstrate road failure."""
    exposure = FloodExposure(flood, **params)
    frames = {}
    for name in ("roads", "buildings", "hospitals", "settlements"):
        frame = getattr(osm, name).to_crs(flood.crs).copy().reset_index(drop=True)
        if name == "buildings" and not include_buildings:
            frame = frame.iloc[:0].copy()
        frames[name] = frame
    road_values = [exposure.measure_line(row.geometry, is_bridge=bool(row.is_bridge),
                                       is_tunnel=bool(row.is_tunnel), is_ford=bool(row.is_ford))
                   for row in frames["roads"].itertuples()]
    for key in ("length_in_flood_m", "fraction_in_flood", "max_tier", "median_tier", "damage_tier",
                "hit", "bridge_approach_hit", "approach_hit", "low_evidence_ford", "evidence_label"):
        frames["roads"][key] = [row[key] for row in road_values]
    for name in ("buildings", "hospitals", "settlements"):
        tiers = exposure.polygon_tiers(frames[name])
        frames[name]["damage_tier"] = tiers
        frames[name]["in_flood"] = tiers > 0
        frames[name]["hit"] = tiers > 0
    roads = frames["roads"]
    summary = {"counts": {name: int(frame.hit.sum()) for name, frame in frames.items()},
               "by_tier": {name: {str(tier): int((frame.damage_tier == tier).sum()) for tier in (1, 2, 3)}
                           for name, frame in frames.items()},
               "road_km_by_highway": {str(highway): float(group.length.sum() / 1000)
                                      for highway, group in roads.loc[roads.hit.astype(bool)].groupby("highway")},
               "inundated_road_km_by_highway": {str(highway): float(group.length_in_flood_m.sum() / 1000)
                                               for highway, group in roads.groupby("highway")}}
    summary["counts"]["bridges"] = int((roads.hit.astype(bool) & roads.is_bridge).sum())
    return DamageResult(**frames, summary=summary, params=exposure.params, exposure=exposure)
