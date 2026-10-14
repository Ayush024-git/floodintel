"""Cached-only infrastructure exposure and mapped-access analysis."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import struct
import time
from typing import Any
import zlib

import geopandas as gpd
import numpy as np
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely import make_valid
from shapely.geometry import LineString, box, mapping, shape

from src.analysis.flood_map import FLOAT_BANDS, MASK_BANDS, FloodMap, load_flood_map
from src.data.incident import load_incident, load_layers
from src.graph.chokepoints import analyze_chokepoints
from src.graph.damage import assess_damage
from src.graph.isolation import analyze_isolation
from src.graph.priorities import rank_priorities
from src.graph.road_graph import SPEED_KPH, build_road_graph

VERSION = 2
BANNER = "SYNTHETIC TEST INPUT, NOT A MODEL RESULT"
OUTPUTS = ("damaged_assets.geojson", "settlements.geojson", "routes.geojson", "clusters.json",
           "chokepoints.json", "chokepoints.geojson", "priorities.json", "impact_meta.json", "quicklook.png")
DEBRIS_FLAG = dict(code="debris_candidate_input", severity="caution",
                   message="Debris candidates are backscatter-change near drainage, unconfirmed")


def _debris_evidence(flood: Any, directory: str | Path) -> tuple[Any, str]:
    """Rasterize the local geographic candidates as tier 1; never alter the flood map."""
    path = Path(directory) / "debris_candidates.geojson"
    if not path.is_file():
        raise FileNotFoundError(f"Debris candidate polygons are missing: {path}")
    payload = path.read_bytes()
    collection = json.loads(payload)
    if collection.get("type") != "FeatureCollection":
        raise ValueError("Debris candidates must be a geographic GeoJSON FeatureCollection")
    if collection.get("crs"):
        raise ValueError("Debris candidate GeoJSON must use EPSG:4326 without a legacy CRS override")
    polygons = []
    repaired = 0
    for feature in collection.get("features", []):
        geometry = shape(feature["geometry"]) if feature.get("geometry") else None
        if geometry is None or geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Debris candidate features must contain nonempty polygon geometry")
        if not geometry.is_valid:
            geometry = make_valid(geometry)
            repaired += 1
        if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Repaired debris candidate geometry must remain polygonal")
        polygons.append(transform_geom("EPSG:4326", flood.crs, mapping(geometry)))
    mask = (rasterize([(geometry, 1) for geometry in polygons], out_shape=flood.flood_mask.shape,
                      transform=flood.transform, dtype="uint8", all_touched=False).astype(bool)
            if polygons else np.zeros_like(flood.flood_mask))
    mask &= flood.valid_mask
    evidence = copy.copy(flood)
    evidence.flood_mask = mask
    evidence.confidence = mask.astype(np.uint8)  # Input confidence properties cannot raise this tier.
    evidence.params = dict(flood.params, include_debris_candidates=True, debris_pixel_inclusion="center")
    evidence.flags = [*flood.flags, dict(DEBRIS_FLAG)]
    evidence.warnings = [*flood.warnings, DEBRIS_FLAG["message"],
                         "Candidate evidence uses the exported polygons, including their simplification and feature caps; it is not confirmed inundation."]
    if repaired:
        evidence.warnings.append(f"Repaired {repaired} invalid debris candidate geometries.")
    return evidence, hashlib.sha256(payload).hexdigest()


def _merge_evidence(flood: dict[str, Any], debris: dict[str, Any]) -> dict[str, Any]:
    """Qualify each source independently: candidates can neither boost overlap nor tier."""
    result = dict(flood)
    result.update(flood_hit=bool(flood["hit"]), debris_candidate_hit=bool(debris["hit"]),
                  debris_length_in_candidate_m=debris["length_in_flood_m"],
                  debris_fraction_in_candidate=debris["fraction_in_flood"],
                  evidence_source="flood" if flood["hit"] else "debris_candidate" if debris["hit"] else "none")
    if not flood["hit"] and debris["hit"]:
        result.update(hit=True, damage_tier=1,
                      evidence_label="potentially affected by an unconfirmed debris candidate",
                      approach_hit=debris["approach_hit"], bridge_approach_hit=debris["bridge_approach_hit"])
    return result


class _CandidateExposure:
    """Reuse independent exposure measurements for every split graph member."""

    def __init__(self, flood: Any, debris: Any):
        self.flood, self.debris = flood, debris

    def measure_line(self, geometry: Any, **params: Any) -> dict[str, Any]:
        return _merge_evidence(self.flood.measure_line(geometry, **params), self.debris.measure_line(geometry, **params))


def _add_candidate_damage(damage: Any, candidates: Any) -> None:
    """Extend exposure in place at tier 1, retaining water-only inundation measurements."""
    for name in ("roads", "buildings", "hospitals", "settlements"):
        frame, extra = getattr(damage, name), getattr(candidates, name)
        frame["flood_hit"] = frame.hit.astype(bool)
        frame["debris_candidate_hit"] = extra.hit.astype(bool)
        frame["evidence_source"] = np.where(frame.hit, "flood", np.where(extra.hit, "debris_candidate", "none"))
        if name == "roads":
            merged = [_merge_evidence(original, candidate) for original, candidate in
                      zip(frame.to_dict("records"), extra.to_dict("records"), strict=True)]
            for key in ("hit", "damage_tier", "approach_hit", "bridge_approach_hit", "evidence_label",
                        "debris_length_in_candidate_m", "debris_fraction_in_candidate"):
                frame[key] = [row[key] for row in merged]
        else:
            added = ~frame.hit.astype(bool) & extra.hit.astype(bool)
            frame.loc[added, "damage_tier"] = 1
            frame["hit"] = frame.hit.astype(bool) | extra.hit.astype(bool)
    frames = {name: getattr(damage, name) for name in ("roads", "buildings", "hospitals", "settlements")}
    damage.summary["counts"] = {name: int(frame.hit.sum()) for name, frame in frames.items()}
    damage.summary["counts"]["bridges"] = int((damage.roads.hit.astype(bool) & damage.roads.is_bridge).sum())
    damage.summary["by_tier"] = {name: {str(tier): int((frame.damage_tier == tier).sum()) for tier in (1, 2, 3)}
                                  for name, frame in frames.items()}
    damage.summary["road_km_by_highway"] = {str(highway): float(group.length.sum() / 1000) for highway, group in
                                             damage.roads.loc[damage.roads.hit.astype(bool)].groupby("highway")}
    damage.summary["debris_candidate_counts"] = candidates.summary["counts"]
    damage.params = dict(damage.params, include_debris_candidates=True, debris_candidate_tier=1,
                         overlap_rule="Each source must qualify independently; candidate pixels cannot increase flood evidence tier")
    damage.exposure = _CandidateExposure(damage.exposure, candidates.exposure)


@dataclass
class ImpactResult:
    metadata: dict[str, Any]
    settlements: list[dict[str, Any]]
    clusters: list[dict[str, Any]]
    chokepoints: dict[str, Any]
    priorities: dict[str, Any]
    cache_path: str


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def _features(rows: list[dict[str, Any]], crs: Any) -> dict[str, Any]:
    features = []
    for row in rows:
        values = dict(row)
        geometry = values.pop("geometry")
        features.append(dict(type="Feature", geometry=transform_geom(crs, "EPSG:4326", mapping(geometry)),
                             properties=_safe(values)))
    return dict(type="FeatureCollection", features=features)


def synthetic_flood(stack: Any, bounds: tuple[float, ...]) -> FloodMap:
    """Explicit test-polygon raster, center-pixel inclusion; no model or terrain filtering."""
    if len(bounds) != 4 or not all(np.isfinite(bounds)) or not (-180 <= bounds[0] < bounds[2] <= 180 and -90 <= bounds[1] < bounds[3] <= 90):
        raise ValueError("test-polygon requires a valid W S E N box in EPSG:4326")
    polygon = transform_geom("EPSG:4326", stack.crs, mapping(box(*bounds)))
    mask = rasterize([(polygon, 1)], out_shape=stack.valid_mask.shape, transform=stack.transform,
                     fill=0, dtype="uint8", all_touched=False).astype(bool) & stack.valid_mask
    values = {name: np.zeros(mask.shape, np.float32) for name in FLOAT_BANDS}
    values.update({name: np.zeros(mask.shape, bool) for name in MASK_BANDS})
    values.update(p_post=np.where(stack.valid_mask, mask.astype(np.float32), np.nan),
                  p_pre=np.where(stack.valid_mask, 0., np.nan).astype(np.float32),
                  post_vv_db=stack.array[list(stack.band_names).index("post_vv")], flood_mask=mask,
                  confidence=mask.astype(np.uint8) * 3, valid_mask=stack.valid_mask.copy(),
                  invalid_mask=~stack.valid_mask, new_water=mask.copy(), water_post=mask.copy())
    return FloodMap(**values, transform=stack.transform, crs=stack.crs, resolution=stack.resolution,
                    bbox=stack.bbox, stats=dict(flood_area_km2=float(mask.sum() * stack.resolution ** 2 / 1e6)),
                    params=dict(test_polygon=list(bounds), synthetic=True, pixel_inclusion="center"),
                    warnings=[BANNER], flags=[dict(code="synthetic_flood_input", severity="limiting", message=BANNER)])


def suggest_test_boxes(osm: Any, *, limit: int = 3, buffer_m: float = 30) -> list[dict[str, Any]]:
    """Geographic boxes around the longest mapped AOI bridge segments, for debugging only."""
    if type(limit) is not int or limit < 1 or not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("Invalid suggested-box count/buffer")
    roads = osm.roads.loc[osm.roads.is_bridge & osm.roads.in_aoi].copy()
    roads["length_m"] = roads.length
    roads = roads.sort_values(["length_m", "segment_id"], ascending=[False, True]).head(limit)
    results = []
    for row in roads.itertuples():
        bounds = gpd.GeoSeries([row.geometry.buffer(buffer_m).envelope], crs=osm.crs).to_crs(4326).total_bounds
        results.append(dict(osm_id=row.osm_id, segment_id=row.segment_id, name=row.name,
                            highway=row.highway, included_in_drivable_graph=row.highway in SPEED_KPH,
                            length_m=float(row.length_m), bbox=[float(value) for value in bounds], buffer_m=buffer_m,
                            label=BANNER))
    return results


def list_critical_bridges(osm: Any, *, snap_max_m: float = 1500., catchment_m: float = 300.) -> list[dict[str, Any]]:
    """Top ten positive-leverage latent bridges from the undamaged graph, sorted by count/id."""
    if not osm.snapshot_verified:
        raise ValueError("Cannot use an unverified OSM snapshot")
    network = build_road_graph(osm, snap_max_m=snap_max_m)
    isolation = analyze_isolation(network, osm.buildings, catchment_m=catchment_m)
    bridge_count = sum(data["is_bridge"] for _, _, data in network.graph.edges(data=True))
    candidates = analyze_chokepoints(network, isolation, latent_limit=bridge_count).latent
    candidates = sorted((row for row in candidates if row["count"] > 0), key=lambda row: (-row["count"], row["link_id"]))[:10]
    output = []
    for row in candidates:
        center = gpd.GeoSeries([row["geometry"].centroid], crs=osm.crs).to_crs(4326).iloc[0]
        output.append(dict(link_id=row["link_id"], name=row["name"], lon=float(center.x), lat=float(center.y),
                           settlements_affected=row["settlements_affected"], settlement_ids=row["settlement_ids"], count=row["count"]))
    return output


def test_link_box(osm: Any, link_id: str, *, buffer_m: float = 30., **graph_params: Any) -> tuple[float, ...]:
    """Geographic box around the baseline link geometry with a documented metric buffer."""
    if not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("Invalid test-link box buffer")
    network = build_road_graph(osm, **graph_params)
    link = next((data for _, _, data in network.graph.edges(data=True) if data["link_id"] == link_id), None)
    if link is None:
        raise ValueError(f"Unknown baseline link_id: {link_id}")
    geometry = link["geometry"].buffer(buffer_m).envelope if buffer_m else link["geometry"].envelope
    return tuple(float(value) for value in gpd.GeoSeries([geometry], crs=osm.crs).to_crs(4326).total_bounds)


def _fingerprint(incident: Any, osm: Any, flood: FloodMap, params: dict) -> str:
    digest = hashlib.sha256(json.dumps(_safe(dict(version=VERSION, incident=incident.to_dict(), params=params,
                                                 flood_params=flood.params, flood_flags=flood.flags,
                                                 osm_quality=osm.quality)), sort_keys=True).encode())
    for array in (flood.flood_mask, flood.confidence, flood.valid_mask):
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    digest.update(str((flood.transform, flood.crs, flood.resolution)).encode())
    path = Path(osm.cache_path) if osm.cache_path else None
    if path and path.is_file():
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        for name in ("roads", "hospitals", "settlements", "buildings"):
            frame = getattr(osm, name)
            for row in frame.sort_values("osm_id").to_dict("records"):
                geometry = row.pop("geometry")
                digest.update(geometry.wkb)
                digest.update(json.dumps(_safe(row), sort_keys=True).encode())
    return digest.hexdigest()[:24]


def _quicklook(flood: FloodMap, osm: Any, damage: Any, isolation: Any, path: Path) -> None:
    """Use the flood-map NumPy/stdlib PNG approach; categorical colors are in metadata."""
    gray = (np.nan_to_num(np.clip((flood.post_vv_db + 35) / 40, 0, 1), nan=0) * 255).astype(np.uint8)
    image = np.repeat(gray[..., None], 3, axis=2)
    image[~flood.valid_mask] = 0
    objects = []
    for row in damage.roads.itertuples():
        objects.append((row.geometry.buffer(flood.resolution * (.35 if not row.hit else 1.5 if row.is_bridge else .7)),
                        4 if row.hit and getattr(row, "evidence_source", None) == "debris_candidate" else 2 if row.hit else 1))
    palette = {1: (150, 150, 150), 2: (255, 25, 25), 3: (255, 25, 25), 4: (255, 175, 0), 5: (30, 220, 70), 6: (255, 255, 255)}
    for row in isolation.settlements:
        objects.append((row["geometry"].buffer(flood.resolution * 2.5), {"CUT OFF": 3, "UNCERTAIN": 4, "CONNECTED": 5}[row["status"]]))
    for point in osm.hospitals.geometry:
        r = flood.resolution * 3
        cross = LineString([(point.x - r, point.y), (point.x + r, point.y)]).buffer(flood.resolution / 2)
        cross = cross.union(LineString([(point.x, point.y - r), (point.x, point.y + r)]).buffer(flood.resolution / 2))
        objects.append((cross, 6))
    if objects:
        overlay = rasterize(objects, out_shape=flood.flood_mask.shape, transform=flood.transform, dtype="uint8")
        for value, color in palette.items():
            image[overlay == value] = color
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack("!I", len(payload)) + kind + payload + struct.pack("!I", zlib.crc32(kind + payload))
    raw = b"".join(b"\x00" + row.tobytes() for row in image)
    header = struct.pack("!2I5B", image.shape[1], image.shape[0], 8, 2, 0, 0, 0)
    note = ("SYNTHETIC TEST INPUT; " if flood.params.get("synthetic") else "") + "Grey roads; red potential inundation/cut-off; amber uncertain; green connected; white facility crosses"
    if "debris_candidate_hit" in damage.roads:
        note += "; amber candidate-only road evidence"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"tEXt", b"Description\x00" + note.encode())
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _load_result(directory: Path) -> ImpactResult:
    metadata = json.loads((directory / "impact_meta.json").read_text())
    metadata["cache_hit"] = True
    settlements = [feature["properties"] for feature in json.loads((directory / "settlements.geojson").read_text())["features"]]
    return ImpactResult(metadata, settlements, json.loads((directory / "clusters.json").read_text()),
                        json.loads((directory / "chokepoints.json").read_text()),
                        json.loads((directory / "priorities.json").read_text()), str(directory.resolve()))


def analyze_incident_impact(
    incident_id: str, flood_dir: str | Path | None = None, *, cache_dir: str | Path = "data/cache",
    test_polygon: tuple[float, ...] | None = None, test_link: str | None = None,
    include_debris_candidates: bool = False,
    min_overlap_m: float = 20., min_fraction: float = .1,
    approach_buffer_m: float = 30., sample_spacing_m: float | None = None, snap_max_m: float = 1500.,
    coordinate_precision_m: float = .01, speed_kph: dict[str, float] | None = None,
    fragmentation_threshold: float = .7, degraded_factor: float = 1.5, catchment_m: float = 300.,
    baseline_warning_share: float = .2, include_latent: bool = True, latent_limit: int = 200,
    settlement_weights: dict[str, float] | None = None, link_weights: dict[str, float] | None = None,
    detour_cap: float = 10., building_log_cap: float = 3., status_scores: dict[str, float] | None = None,
) -> ImpactResult:
    """Run local caches only; synthetic inputs are isolated under impact_test/.

    An empty flood map never establishes missing baseline access. Unmapped or
    unsnapped subjects remain UNCERTAIN; mapped connected subjects stay CONNECTED.
    Opt-in candidate polygons qualify independently at tier 1, only affect the
    possible scenario, and use impact_debris/ without changing ordinary caches.
    """
    started = time.perf_counter()
    requested = {key: value for key, value in locals().copy().items() if key in _SIGNATURE.parameters
                 and key not in {"incident_id", "flood_dir", "cache_dir"}}
    if not include_debris_candidates:
        requested.pop("include_debris_candidates")  # Preserve ordinary cache keys.
    elif test_polygon is not None or test_link is not None:
        raise ValueError("Debris candidate mode and synthetic test modes are mutually exclusive")
    if test_link is not None and test_polygon is not None:
        raise ValueError("test_link and test_polygon are mutually exclusive")
    if flood_dir is not None and (test_link is not None or test_polygon is not None):
        raise ValueError("Synthetic test modes and flood_dir are mutually exclusive")
    incident = load_incident(incident_id, cache_dir=cache_dir)
    if incident.status in {"blocked", "failed"}:
        raise ValueError(f"Cannot analyze impact of {incident.status} incident")
    layers = load_layers(incident)
    osm = layers.osm
    if not osm.snapshot_verified:
        raise ValueError("Cannot use an unverified OSM snapshot")
    root = Path(incident.cache_path).parent
    if test_link is not None:
        test_polygon = test_link_box(osm, test_link, buffer_m=approach_buffer_m, snap_max_m=snap_max_m,
                                     coordinate_precision_m=coordinate_precision_m, speed_kph=speed_kph,
                                     fragmentation_threshold=fragmentation_threshold)
    if test_polygon is not None:
        if flood_dir is not None:
            raise ValueError("test_polygon and flood_dir are mutually exclusive")
        flood = synthetic_flood(layers.s1_stack, test_polygon)
        if test_link is not None:
            flood.params.update(test_link=test_link, test_link_buffer_m=approach_buffer_m)
            if not flood.flood_mask.any():
                raise ValueError("Test link box contains no valid S1 pixels; choose a link within the cached stack grid")
    else:
        if flood_dir is None:
            candidates = [path.parent for path in (root / "flood").glob("*/flood_meta.json") if path.with_name("flood.tif").is_file()]
            if not candidates:
                raise FileNotFoundError("No complete cached flood map; run src.analysis.flood_map first")
            flood_dir = max(candidates, key=lambda path: (path.joinpath("flood_meta.json").stat().st_mtime_ns, str(path)))
        flood = load_flood_map(flood_dir)
        if flood.params.get("synthetic") or any(flag["code"] == "synthetic_flood_input" for flag in flood.flags):
            raise ValueError("Synthetic flood input requires test_polygon mode; never write it under impact/")
    if CRS.from_user_input(osm.crs) != CRS.from_user_input(flood.crs):
        raise ValueError("OSM and flood map must share the working CRS")
    candidate_evidence = None
    if include_debris_candidates:
        candidate_evidence, digest = _debris_evidence(flood, flood_dir)
        requested["debris_candidates_sha256"] = digest
    key = _fingerprint(incident, osm, flood, requested)
    directory = root / ("impact_test" if test_polygon is not None else "impact_debris" if include_debris_candidates else "impact") / key
    if all((directory / name).is_file() for name in OUTPUTS):
        return _load_result(directory)
    include_buildings = osm.quality.get("include_buildings", incident.provenance.get("request", {}).get("include_buildings", True))
    damage = assess_damage(osm, flood, include_buildings=include_buildings, min_overlap_m=min_overlap_m,
                           min_fraction=min_fraction, approach_buffer_m=approach_buffer_m, sample_spacing_m=sample_spacing_m)
    if candidate_evidence is not None:
        candidates = assess_damage(osm, candidate_evidence, include_buildings=include_buildings,
                                   min_overlap_m=min_overlap_m, min_fraction=min_fraction,
                                   approach_buffer_m=approach_buffer_m, sample_spacing_m=sample_spacing_m)
        _add_candidate_damage(damage, candidates)
    network = build_road_graph(osm, damage, snap_max_m=snap_max_m, coordinate_precision_m=coordinate_precision_m,
                               speed_kph=speed_kph, fragmentation_threshold=fragmentation_threshold)
    isolation = analyze_isolation(network, damage.buildings, degraded_factor=degraded_factor, catchment_m=catchment_m,
                                  baseline_warning_share=baseline_warning_share, flood_pixel_area_m2=flood.resolution ** 2)
    if candidate_evidence is not None:
        members = {member["member_id"]: member for _, _, link in network.graph.edges(data=True) for member in link["members"]}
        for row in isolation.settlements:
            candidate_responsible = False
            for evidence in row["evidence"]:
                for member in evidence["members"]:
                    details = members[member["member_id"]]
                    member.update({key: details[key] for key in ("evidence_source", "flood_hit", "debris_candidate_hit",
                                  "debris_length_in_candidate_m", "debris_fraction_in_candidate")})
                    candidate_responsible |= details["evidence_source"] == "debris_candidate"
            if row["status"] == "UNCERTAIN" and candidate_responsible:
                row["reason"] = "possible-scenario road access includes unconfirmed debris-candidate evidence"
    chokepoints = analyze_chokepoints(network, isolation, include_latent=include_latent, latent_limit=latent_limit)
    priorities = rank_priorities(isolation, chokepoints, settlement_weights=settlement_weights, link_weights=link_weights,
                                 detour_cap=detour_cap, building_log_cap=building_log_cap, status_scores=status_scores)
    flags = []
    for flag in [*incident.quality.get("flags", []), *flood.flags, *network.flags, *isolation.flags, *chokepoints.flags]:
        flag = {key: flag[key] for key in ("code", "severity", "message")}
        if flag not in flags:
            flags.append(flag)
    def flag(code: str, severity: str, message: str) -> None:
        if not any(value["code"] == code for value in flags):
            flags.append(dict(code=code, severity=severity, message=message))
    flag("road_graph_ignores_oneway_and_vehicle_type", "info", "Road graph is undirected and ignores oneway, access restrictions and vehicle suitability")
    flag("travel_time_heuristic", "info", "Travel times use fixed highway-class speeds, not traffic or road-condition measurements")
    flag("damage_is_inundation_not_structural", "caution", "Flood intersection means potentially affected / inundated; access removal is a scenario, not confirmed physical failure")
    if include_debris_candidates:
        flag(**DEBRIS_FLAG)
    if not include_buildings:
        flag("buildings_deferred", "info", "Buildings deferred in the incident; building exposure and size proxies are unavailable")
    if osm.quality.get("road_density_km_per_km2", 1.) < .3:
        flag("osm_sparse_network", "caution", "Mapped road density is below 0.3 km per km2")
    if (osm.quality.get("roads_edited_over_5_years_share") or 0.) > .5:
        flag("osm_stale", "caution", "More than half of mapped road ways were last edited over five years before the snapshot")
    if network.quality["destinations_snapped"]["hospital"] == 0:
        flag("osm_no_hospitals", "limiting", "No snapped hospital destination; hospital isolation cannot be established")
        if network.quality["destinations_snapped"]["clinic"]:
            flag("only_clinics_no_hospitals", "caution", "Only clinics are mapped/snapped; clinics do not substitute for hospital access")
    if not network.subjects:
        flag("osm_no_settlements", "limiting", "No subject settlements mapped in the AOI")
    if not flood.flood_mask.any():
        flag("no_flood_detected", "info", "No flood pixels; unconfirmed candidates may still affect the possible scenario" if include_debris_candidates
             else "No flood pixels; no inundation links removed, baseline mapping limits still apply")
    if not network.graph.number_of_edges():
        flag("road_graph_empty", "limiting", "No drivable road links are mapped; access cannot be established")
    warnings = list(dict.fromkeys([*incident.quality.get("warnings", []), *osm.warnings, *flood.warnings,
                                  *(candidate_evidence.warnings if candidate_evidence is not None else []),
                                  *(value["message"] for value in flags),
                                  "A snapped point is proximity to a road node, not a verified entrance; unmapped roads and paths may provide access.",
                                  "Partial parallel-link capacity loss is ignored; latent bridge-link loss represents the whole merged corridor.",
                                  "Only single-link restoration is tested; multi-link repairs may be required."]))
    if test_polygon is None and flood_dir is not None:
        warnings.append(f"Using cached flood map: {Path(flood_dir).resolve()}")
    cautions = sum(flag["severity"] == "caution" for flag in flags)
    limiting = any(flag["severity"] == "limiting" for flag in flags)
    trust = "insufficient" if test_polygon is not None else "uncertain" if limiting or cautions >= 2 else "probable" if cautions else "strong"
    metadata = dict(version=VERSION, incident_id=incident.incident_id, cache_key=key, cache_hit=False,
                    banner=BANNER if test_polygon is not None else None,
                    counts=damage.summary, status_tallies={status: sum(row["status"] == status for row in isolation.settlements)
                                                         for status in ("CUT OFF", "UNCERTAIN", "CONNECTED")},
                    graph_quality=network.quality, flags=flags, warnings=warnings, trust_summary=trust,
                    params=dict(requested=requested, damage=damage.params, graph=network.params, isolation=isolation.params,
                                chokepoints=chokepoints.params, priorities=priorities["params"]),
                    provenance_pointer=incident.cache_path, flood_path=flood.cache_path, flood_params=flood.params,
                    output_path=str(directory.resolve()), run_seconds=0.,
                    quicklook_legend=dict(roads="grey", potentially_inundated_roads="red", bridge_hits="thick red",
                                          cut_off="red dots", uncertain="amber dots", connected="green dots", hospitals="white crosses"),
                    disclaimer=priorities["disclaimer"])
    if include_debris_candidates:
        metadata["debris_candidates"] = dict(path=str((Path(flood_dir) / "debris_candidates.geojson").resolve()),
                                             sha256=requested["debris_candidates_sha256"], tier=1,
                                             pixel_inclusion="center", qualifying_pixels=int(candidate_evidence.flood_mask.sum()))
        metadata["quicklook_legend"]["candidate_only_road_hits"] = "amber"
    directory.mkdir(parents=True, exist_ok=True)
    assets = []
    for name, frame in (("road", damage.roads), ("building", damage.buildings)):
        for row in frame.loc[frame.hit.astype(bool)].to_dict("records"):
            row = dict(row, asset_class="bridge" if name == "road" and row["is_bridge"] else name,
                       tier=int(row["damage_tier"]), evidence_label="potentially affected by an unconfirmed debris candidate"
                       if row.get("evidence_source") == "debris_candidate" else "potentially affected / inundated")
            row.pop("population", None)
            assets.append(row)
    _write(directory / "damaged_assets.geojson", _features(assets, osm.crs))
    settlement_rows, routes = [], []
    for row in isolation.settlements:
        record = dict(row)
        record.pop("node")
        for key, scenario in (("nearest_baseline", "baseline"), ("nearest_after", "likely"), ("nearest_possible", "possible")):
            record[key] = {}
            for kind, route in row[key].items():
                if route and route["geometry"]:
                    routes.append(dict(type="Feature", geometry=route["geometry"],
                                       properties=dict(settlement_id=row["osm_id"], scenario=scenario,
                                                       destination_class=kind, destination_id=route["osm_id"],
                                                       travel_time_s=route["travel_time_s"], length_m=route["length_m"])))
                record[key][kind] = {name: value for name, value in route.items() if name != "geometry"} if route else None
        settlement_rows.append(record)
    _write(directory / "settlements.geojson", _features(settlement_rows, osm.crs))
    _write(directory / "routes.geojson", dict(type="FeatureCollection", features=routes))
    _write(directory / "clusters.json", isolation.clusters)
    point_rows = [*chokepoints.damaged, *chokepoints.latent]
    _write(directory / "chokepoints.geojson", _features(point_rows, osm.crs))
    choke_json = {name: [{key: value for key, value in row.items() if key != "geometry"} for row in getattr(chokepoints, name)]
                  for name in ("damaged", "latent")}
    choke_json["damaged_no_leverage"] = priorities["damaged_no_leverage"]
    choke_json["damaged"] = [row for row in choke_json["damaged"] if row["count"] > 0]
    _write(directory / "chokepoints.json", choke_json)
    _write(directory / "priorities.json", priorities)
    _quicklook(flood, osm, damage, isolation, directory / "quicklook.png")
    metadata["run_seconds"] = time.perf_counter() - started
    _write(directory / "impact_meta.json", metadata)
    return ImpactResult(metadata, [row for row in settlement_rows], isolation.clusters, choke_json, priorities, str(directory.resolve()))


_SIGNATURE = inspect.signature(analyze_incident_impact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incident", required=True)
    parser.add_argument("--flood-dir")
    parser.add_argument("--include-debris-candidates", action="store_true",
                        help="Unconfirmed candidate polygons at tier 1 only; write to impact_debris/")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test-polygon", nargs=4, type=float)
    mode.add_argument("--test-link")
    mode.add_argument("--list-critical-bridges", action="store_true")
    mode.add_argument("--suggest-test-boxes", action="store_true")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--min-overlap-m", type=float, default=20)
    parser.add_argument("--approach-buffer-m", type=float, default=30)
    parser.add_argument("--snap-max-m", type=float, default=1500)
    parser.add_argument("--catchment-m", type=float, default=300)
    parser.add_argument("--degraded-factor", type=float, default=1.5)
    parser.add_argument("--no-latent", action="store_true")
    parser.add_argument("--latent-limit", type=int, default=200)
    args = parser.parse_args()
    try:
        if args.suggest_test_boxes or args.list_critical_bridges:
            receipt = load_incident(args.incident, cache_dir=args.cache_dir)
            osm = load_layers(receipt).osm
            results = (list_critical_bridges(osm, snap_max_m=args.snap_max_m, catchment_m=args.catchment_m)
                       if args.list_critical_bridges else suggest_test_boxes(osm))
            print(json.dumps(_safe(results), indent=2))
            return
        result = analyze_incident_impact(args.incident, args.flood_dir, cache_dir=args.cache_dir,
                                        test_polygon=tuple(args.test_polygon) if args.test_polygon else None,
                                        test_link=args.test_link,
                                        include_debris_candidates=args.include_debris_candidates,
                                        min_overlap_m=args.min_overlap_m, approach_buffer_m=args.approach_buffer_m,
                                        snap_max_m=args.snap_max_m, catchment_m=args.catchment_m,
                                        degraded_factor=args.degraded_factor, include_latent=not args.no_latent,
                                        latent_limit=args.latent_limit)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(_safe(dict(counts=result.metadata["counts"]["counts"], status_tallies=result.metadata["status_tallies"],
                                top_chokepoints=[{key: row[key] for key in ("link_id", "count", "mapped_buildings", "hospital_access_reconnected", "damage_tier")}
                                                for row in result.chokepoints["damaged"][:5]],
                                top_priorities=[{key: row[key] for key in ("id", "name", "rank", "score", "confidence")}
                                                for row in result.priorities["settlements"][:5]],
                                flags=result.metadata["flags"], cache_path=result.cache_path)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
