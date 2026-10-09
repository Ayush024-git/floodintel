"""Acquire and verify a historical OSM snapshot, with limited routing context."""

import argparse
from dataclasses import dataclass
from datetime import datetime, time as day_time, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import time
from typing import Any, Callable
import zlib

import geopandas as gpd
import numpy as np
from pyproj import CRS
import requests
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely import make_valid
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from src.data.aoi import AOI, BBox, build_aoi
from src.data.s1_preprocess import S1Stack


ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)
USER_AGENT = "floodintel/0.1 (open-source flood response prototype)"
QUERY_VERSION = 1
ROAD_CLASSES = (
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "service", "track", "path", "living_street", "road",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
)
CONTEXT_ROADS = (
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
)
PLACES = ("city", "town", "village", "hamlet", "suburb", "isolated_dwelling", "locality")
LAYER_COLUMNS = {
    "roads": ["osm_id", "segment_id", "highway", "name", "surface", "bridge", "tunnel",
              "oneway", "ford", "is_bridge", "is_tunnel", "is_ford", "timestamp", "in_aoi"],
    "buildings": ["osm_id", "building", "name", "timestamp", "in_aoi"],
    "hospitals": ["osm_id", "name", "amenity", "healthcare", "tier", "timestamp", "in_aoi"],
    "settlements": ["osm_id", "name", "place", "population", "timestamp", "in_aoi"],
}
COVERAGE_WARNING = (
    "OSM coverage in this region is volunteer-mapped and may be incomplete; "
    "absence of a road/village does not mean it does not exist"
)


class OSMError(RuntimeError):
    """Historical OSM acquisition or storage failed."""


class OSMSnapshotError(OSMError):
    """Returned timestamps cannot verify the requested historical snapshot."""


@dataclass(frozen=True)
class OSMBundle:
    """Historical feature layers, provenance, and AOI quality metrics."""

    roads: gpd.GeoDataFrame
    buildings: gpd.GeoDataFrame
    hospitals: gpd.GeoDataFrame
    settlements: gpd.GeoDataFrame
    snapshot_date: str
    snapshot_verified: bool
    newest_edit: str | None
    endpoint_used: str
    crs: CRS
    bbox: BBox
    quality: dict[str, Any]
    warnings: list[str]
    cache_path: str | None = None

    @property
    def bridges(self) -> gpd.GeoDataFrame:
        """Expose the original road rows; bridges have no independent geometry layer."""
        return self.roads.loc[self.roads["is_bridge"]]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata without embedding geometries."""
        return {
            "snapshot_date": self.snapshot_date,
            "snapshot_verified": self.snapshot_verified,
            "newest_edit": self.newest_edit,
            "endpoint_used": self.endpoint_used,
            "crs": self.crs.to_string(), "bbox": list(self.bbox),
            "quality": self.quality, "warnings": list(self.warnings),
            "cache_path": self.cache_path,
        }


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: Any) -> datetime | None:
    """Reject missing, malformed, or unzoned edit timestamps."""
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def snapshot_datetime(aoi: AOI, offset_days: int = 0) -> datetime:
    """Use flood-date UTC midnight, optionally moved earlier by whole days."""
    if type(offset_days) is not int or offset_days < 0:
        raise ValueError("snapshot_offset_days must be a nonnegative integer")
    return datetime.combine(aoi.flood_date - timedelta(days=offset_days), day_time(), timezone.utc)


def context_bbox(bbox: BBox, buffer_km: float) -> BBox:
    """Expand bounds at the center latitude without wrapping geographic limits."""
    if isinstance(buffer_km, bool) or not math.isfinite(buffer_km) or buffer_km < 0:
        raise ValueError("context_buffer_km must be finite and nonnegative")
    west, south, east, north = bbox
    delta_lat = buffer_km / 111.32
    delta_lon = delta_lat / math.cos(math.radians((south + north) / 2))
    result = west - delta_lon, south - delta_lat, east + delta_lon, north + delta_lat
    if not (-180 <= result[0] < result[2] <= 180 and -90 < result[1] < result[3] < 90):
        raise ValueError("Buffered bbox exceeds geographic limits; split the AOI or reduce context")
    return result


def tile_bbox(bbox: BBox, max_side_km: float = 10) -> list[BBox]:
    """Partition bounds into sequential query tiles no larger than about 10 km."""
    if not math.isfinite(max_side_km) or max_side_km <= 0:
        raise ValueError("max_side_km must be positive")
    west, south, east, north = bbox
    lon_scale = 111.32 * math.cos(math.radians(max(abs(south), abs(north))))
    nx = max(1, math.ceil((east - west) * lon_scale / max_side_km))
    ny = max(1, math.ceil((north - south) * 111.32 / max_side_km))
    return [
        (west + (east - west) * x / nx, south + (north - south) * y / ny,
         west + (east - west) * (x + 1) / nx, south + (north - south) * (y + 1) / ny)
        for y in range(ny) for x in range(nx)
    ]


def overpass_query(tile: BBox, aoi_bbox: BBox, snapshot: datetime) -> str:
    """Query full AOI features, buffered major roads/points, and dependency metadata."""
    def bounds_string(bounds: BBox) -> str:
        west, south, east, north = bounds
        return f"({south:.10f},{west:.10f},{north:.10f},{east:.10f})"

    extent = bounds_string(tile)
    shapes = [f'way["highway"~"^({"|".join(CONTEXT_ROADS)})$"]{extent};']
    inside = box(*tile).intersection(box(*aoi_bbox))
    if inside.area > 0:
        local = bounds_string(inside.bounds)
        shapes.extend([
            f'way["highway"~"^({"|".join(ROAD_CLASSES)})$"]{local};',
            f'way["building"]{local};', f'relation["building"]{local};',
        ])
    points = []
    for kind in ("node", "way"):
        points.extend([
            f'{kind}["amenity"~"^(hospital|clinic|doctors)$"]{extent};',
            f'{kind}["healthcare"~"^(hospital|clinic)$"]{extent};',
        ])
    points.append(f'nwr["place"~"^({"|".join(PLACES)})$"]{extent};')
    return (
        f'[out:json][timeout:180][date:"{_iso(snapshot)}"];\n'
        f'({"".join(shapes)})->.shapes;\n'
        f'({"".join(points)})->.points;\n'
        '.shapes out meta geom;\n.points out meta center;\n'
        '(.shapes; .points;); >>; out meta;\n'
    )


class OverpassClient:
    """Polite HTTP requests with bounded retries; no current-data fallback."""

    def __init__(self, session: Any = None, post_fn: Callable[..., Any] | None = None):
        self.session = session if session is not None else requests.Session()
        self.owns_session = session is None
        self.post = post_fn if post_fn is not None else self.session.post

    def close(self) -> None:
        if self.owns_session:
            self.session.close()

    def request(self, endpoint: str, query: str) -> dict[str, Any]:
        """Retry transient failures up to four times on this endpoint."""
        last_error = "unknown response"
        for attempt in range(4):
            retry = True
            try:
                response = self.post(
                    endpoint, data={"data": query},
                    headers={"User-Agent": USER_AGENT}, timeout=(15, 210),
                )
                if response.status_code == 200:
                    payload = response.json()
                    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
                        last_error = "malformed Overpass JSON"
                    elif payload.get("remark"):
                        last_error = "Overpass reported an incomplete or failed query"
                    elif any(not isinstance(element, dict) for element in payload["elements"]):
                        last_error = "malformed Overpass elements"
                    else:
                        return payload
                else:
                    last_error = f"HTTP {response.status_code}"
                    retry = response.status_code in {429, 502, 503, 504}
            except (requests.Timeout, requests.ConnectionError):
                last_error = "connection failure or timeout"
            except (ValueError, requests.RequestException):
                last_error = "invalid HTTP/JSON response"
            if not retry or attempt == 3:
                break
            time.sleep(2 ** attempt)
        raise OSMError(f"{endpoint}: {last_error} after at most four attempts")


def verify_timestamps(elements: list[dict], snapshot: datetime) -> dict[str, Any]:
    """Audit every returned record, including objects excluded from feature layers."""
    timestamps = [_timestamp(element.get("timestamp")) for element in elements]
    known = [value for value in timestamps if value is not None]
    missing = len(timestamps) - len(known)
    newer = sum(value > snapshot for value in known)
    verified = bool(known) and missing == 0 and newer == 0
    return {
        "snapshot_verified": verified,
        "newest_edit": _iso(max(known)) if known else None,
        "oldest_edit": _iso(min(known)) if known else None,
        "missing_timestamps": missing, "newer_elements": newer,
        "returned_records": len(elements),
    }


def _verify_payload(payload: dict, snapshot: datetime) -> dict[str, Any]:
    """Also reject databases that have not yet reached the requested snapshot."""
    audit = verify_timestamps(payload["elements"], snapshot)
    metadata = payload.get("osm3s")
    base = _timestamp(metadata.get("timestamp_osm_base")) if isinstance(metadata, dict) else None
    audit["database_timestamp"] = _iso(base) if base else None
    audit["database_behind_snapshot"] = base is not None and base < snapshot
    if audit["database_behind_snapshot"]:
        audit["snapshot_verified"] = False
    return audit


def _key(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]


def _acquire(
    aoi: AOI, snapshot: datetime, extent: BBox, cache_dir: Path,
    client: OverpassClient, strict: bool, endpoints: tuple[str, ...],
) -> tuple[list[dict], str, dict[str, Any], list[str]]:
    """Verify each tile, retaining successful tiles when another endpoint fails."""
    records, warnings = [], []
    used: dict[str, int] = {}
    database_dates: dict[str, str | None] = {}
    behind_snapshot = False
    tiles = tile_bbox(extent)
    for tile in tiles:
        paths = [
            (endpoint, cache_dir / "osm" / "tiles" / (
                _key([tile, aoi.bbox, _iso(snapshot), QUERY_VERSION, endpoint]) + ".json"
            ))
            for endpoint in endpoints
        ]
        selected = None
        # A verified response from any mirror can be reused without another request.
        for endpoint, path in paths:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text())
                elements = payload.get("elements")
                if not isinstance(elements, list) or payload.get("remark"):
                    continue
                if any(not isinstance(element, dict) for element in elements):
                    continue
                audit = _verify_payload(payload, snapshot)
                if audit["database_behind_snapshot"]:
                    warnings.append(
                        f"Skipped {endpoint}: its database date {audit['database_timestamp']} "
                        "predates the requested snapshot."
                    )
                elif not elements or audit["snapshot_verified"]:
                    selected = endpoint, payload
                    break
            except (OSError, ValueError, AttributeError):
                warnings.append("Ignored an unreadable tile cache; fetched a fresh historical response.")
        if selected is None:
            errors = []
            snapshot_failures = 0
            query = overpass_query(tile, aoi.bbox, snapshot)
            for endpoint, path in paths:
                try:
                    payload = client.request(endpoint, query)
                    audit = _verify_payload(payload, snapshot)
                    if audit["database_behind_snapshot"] or (payload["elements"] and not audit["snapshot_verified"]):
                        if strict:
                            raise OSMSnapshotError(
                                f"{endpoint} could not verify {_iso(snapshot)}: "
                                f"{audit['newer_elements']} newer edits, "
                                f"{audit['missing_timestamps']} missing timestamps; "
                                f"database date={audit['database_timestamp']}."
                            )
                        warnings.append(
                            "UNVERIFIED SNAPSHOT: newer/missing edit timestamps or a database "
                            "that predates the requested snapshot; "
                            "unsuitable for competition inference."
                        )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(payload))
                    selected = endpoint, payload
                    time.sleep(0.25)
                    break
                except OSMSnapshotError as exc:
                    snapshot_failures += 1
                    errors.append(str(exc))
                except (OSMError, OSError) as exc:
                    errors.append(str(exc))
            warnings.extend(errors)
            if selected is None:
                detail = "; ".join(errors)
                if snapshot_failures:
                    raise OSMSnapshotError(
                        "No endpoint supplied a complete, timestamp-verified pre-event "
                        f"snapshot for tile {tile}. {detail}"
                    )
                raise OSMError(
                    "All Overpass endpoints failed; no current OSM was substituted. "
                    f"Failed tile {tile}. {detail}"
                )
        endpoint, payload = selected
        audit = _verify_payload(payload, snapshot)
        behind_snapshot |= audit["database_behind_snapshot"]
        base = audit["database_timestamp"]
        previous = database_dates.get(endpoint)
        database_dates[endpoint] = min(previous, base) if previous and base else previous or base
        if base is None:
            warnings.append(
                f"{endpoint} omitted its database timestamp; verification uses returned edit timestamps only."
            )
        records.extend(payload["elements"])
        used[endpoint] = used.get(endpoint, 0) + 1
    audit = verify_timestamps(records, snapshot)
    audit["query_tiles"] = len(tiles)
    audit["endpoint_tile_counts"] = used
    audit["oldest_database_timestamp_by_endpoint"] = database_dates
    audit["database_behind_snapshot"] = behind_snapshot
    if behind_snapshot:
        audit["snapshot_verified"] = False
    if not audit["snapshot_verified"]:
        if strict:
            raise OSMSnapshotError("No endpoint supplied a complete, timestamp-verified pre-event snapshot")
        warnings.append(
            "UNVERIFIED SNAPSHOT: missing/newer timestamps or no valid timestamp evidence; "
            "unsuitable for competition inference."
        )
    return records, "; ".join(used), audit, warnings


def _deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """Preserve geometry/centers from repeated output modes and detect conflicts."""
    elements, conflicts = {}, set()
    for element in records:
        kind, number = element.get("type"), element.get("id")
        if kind not in {"node", "way", "relation"} or not isinstance(number, int):
            raise OSMError("Overpass element has no valid type/id")
        key = f"{kind}/{number}"
        previous = elements.get(key)
        if previous is None:
            elements[key] = dict(element)
        elif previous.get("timestamp") != element.get("timestamp") or previous.get("version") != element.get("version"):
            conflicts.add(key)
            if (element.get("timestamp") or "") > (previous.get("timestamp") or ""):
                elements[key] = dict(element)
        else:
            elements[key] = {**previous, **element}
    return [elements[key] for key in sorted(elements)], len(conflicts)


def _point(element: dict) -> Point | None:
    location = element if element.get("type") == "node" else element.get("center") or {}
    try:
        point = Point(float(location["lon"]), float(location["lat"]))
        return point if point.is_valid and not point.is_empty else None
    except (KeyError, TypeError, ValueError):
        return None


def _parts(geometry: Any, kind: str) -> list[Any]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == kind:
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for child in geometry.geoms for part in _parts(child, kind)]
    return []


def _flag(tags: dict, key: str) -> bool:
    return key in tags and str(tags[key]).lower() != "no"


def _population(value: Any) -> int | None:
    try:
        text = str(value).strip().replace(",", "").replace(" ", "")
        parsed = int(text)
        return parsed if parsed >= 0 else None
    except (ValueError, TypeError):
        return None


def _frame(rows: list[dict], layer: str, crs: CRS) -> gpd.GeoDataFrame:
    columns = [*LAYER_COLUMNS[layer], "geometry"]
    frame = gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs="EPSG:4326")
    for column in ("is_bridge", "is_tunnel", "is_ford", "in_aoi"):
        if column in frame:
            frame[column] = frame[column].fillna(False).astype(bool)
    if layer == "settlements":
        frame["population"] = frame["population"].astype("Int64")
    return frame.to_crs(crs)


def _extract(
    elements: list[dict], bbox: BBox, extent: BBox, crs: CRS,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[str, int]]:
    """Clip AOI layers and keep only major roads and point features in context."""
    aoi_box, context = box(*bbox), box(*extent)
    rows = {name: [] for name in LAYER_COLUMNS}
    dropped, repaired, invalid = 0, 0, 0
    for element in elements:
        tags = element.get("tags") or {}
        if not isinstance(tags, dict):
            continue
        kind = element["type"]
        common = {"osm_id": f"{kind}/{element['id']}", "timestamp": element.get("timestamp"), "name": tags.get("name")}
        if kind == "relation" and "building" in tags:
            dropped += 1
        coordinates = []
        if kind == "way" and (tags.get("highway") in ROAD_CLASSES or "building" in tags):
            try:
                coordinates = [(float(p["lon"]), float(p["lat"])) for p in element.get("geometry", [])]
            except (TypeError, KeyError, ValueError):
                coordinates = []
            if any(not (-180 <= x <= 180 and -90 <= y <= 90) for x, y in coordinates):
                coordinates = []
        if kind == "way" and tags.get("highway") in ROAD_CLASSES:
            if len(coordinates) < 2:
                invalid += 1
            else:
                line = LineString(coordinates)
                pieces = [(part, True) for part in _parts(line.intersection(aoi_box), "LineString")]
                if tags["highway"] in CONTEXT_ROADS:
                    outside = line.intersection(context).difference(aoi_box)
                    pieces.extend((part, False) for part in _parts(outside, "LineString"))
                pieces.sort(key=lambda value: (not value[1], value[0].wkb_hex))
                for index, (geometry, in_aoi) in enumerate(pieces):
                    if not geometry.is_valid or geometry.length == 0:
                        invalid += 1
                        continue
                    rows["roads"].append({
                        **common, **{key: tags.get(key) for key in ("highway", "surface", "bridge", "tunnel", "oneway", "ford")},
                        "segment_id": f"{common['osm_id']}#{index}",
                        "is_bridge": _flag(tags, "bridge"), "is_tunnel": _flag(tags, "tunnel"),
                        "is_ford": _flag(tags, "ford"), "in_aoi": in_aoi, "geometry": geometry,
                    })
        if kind == "way" and "building" in tags:
            if len(coordinates) < 4 or coordinates[0] != coordinates[-1]:
                invalid += 1
            else:
                polygon = Polygon(coordinates)
                if not polygon.is_valid:
                    polygon = make_valid(polygon)
                    repaired += 1
                parts = _parts(polygon.intersection(aoi_box), "Polygon")
                if parts:
                    geometry = parts[0] if len(parts) == 1 else MultiPolygon(parts)
                    if geometry.is_valid:
                        rows["buildings"].append({**common, "building": tags["building"], "in_aoi": True, "geometry": geometry})
                    else:
                        invalid += 1
        hospital = kind in {"node", "way"} and (
            tags.get("amenity") in {"hospital", "clinic", "doctors"}
            or tags.get("healthcare") in {"hospital", "clinic"}
        )
        settlement = tags.get("place") in PLACES
        if hospital or settlement:
            point = _point(element)
            if point is None:
                invalid += 1
                continue
            if not context.covers(point):
                continue
            record = {**common, "geometry": point, "in_aoi": aoi_box.covers(point)}
            if hospital:
                tier = "hospital" if "hospital" in (tags.get("amenity"), tags.get("healthcare")) else "clinic"
                rows["hospitals"].append({**record, "amenity": tags.get("amenity"), "healthcare": tags.get("healthcare"), "tier": tier})
            if settlement:
                rows["settlements"].append({**record, "place": tags["place"], "population": _population(tags.get("population"))})
    frames = {name: _frame(values, name, crs) for name, values in rows.items()}
    _verify_scope(frames, bbox, extent)
    return frames, {"dropped_building_relations": dropped, "repaired_polygons": repaired, "invalid_features_dropped": invalid}


def _verify_scope(frames: dict[str, gpd.GeoDataFrame], bbox: BBox, extent: BBox) -> None:
    """Check clipped AOI features and retained context geometries after reprojection."""
    aoi_box, context = box(*bbox).buffer(1e-8), box(*extent).buffer(1e-8)
    for name, frame in frames.items():
        geographic = frame.to_crs(4326)
        for row in geographic.itertuples():
            boundary = aoi_box if row.in_aoi else context
            if not row.geometry.is_valid or row.geometry.is_empty or not boundary.covers(row.geometry):
                raise OSMError(f"{name} contains a feature outside its verified AOI/context bounds")


def _quality(
    frames: dict[str, gpd.GeoDataFrame], elements: list[dict], bbox: BBox,
    snapshot: datetime, audit: dict[str, Any], cleaning: dict[str, int],
) -> tuple[dict[str, Any], list[str]]:
    """Compute density inside the AOI, keeping context counts separate."""
    west, south, east, north = bbox
    zone = min(60, int(((west + east) / 2 + 180) // 6) + 1)
    metric = CRS.from_epsg((32600 if (south + north) / 2 >= 0 else 32700) + zone)
    area = float(gpd.GeoSeries([box(*bbox)], crs=4326).to_crs(metric).area.iloc[0] / 1_000_000)
    roads = frames["roads"].loc[frames["roads"].in_aoi]
    road_km = float(roads.to_crs(metric).length.sum() / 1000)
    counts = {name: int(frame.in_aoi.sum()) for name, frame in frames.items()}
    counts["bridges"] = int((roads.is_bridge).sum())
    edits = [_timestamp(element.get("timestamp")) for element in elements]
    ages = [(snapshot - value).total_seconds() / (365.25 * 86400) for value in edits if value is not None]
    unique_roads = roads.drop_duplicates("osm_id")
    road_dates = [_timestamp(value) for value in unique_roads.timestamp]
    stale = [((snapshot - value).total_seconds() > 5 * 365.25 * 86400) for value in road_dates if value is not None]
    quality = {
        "counts": counts, "aoi_area_km2": area, "total_road_km": road_km,
        "road_density_km_per_km2": road_km / area,
        "building_density_per_km2": counts["buildings"] / area,
        "settlement_count": counts["settlements"], "hospital_count": counts["hospitals"],
        "bridge_count": counts["bridges"],
        "context_counts": {name: int((~frame.in_aoi).sum()) for name, frame in frames.items()},
        "newest_edit": audit["newest_edit"], "oldest_edit": audit["oldest_edit"],
        "median_element_age_years": statistics.median(ages) if ages else None,
        "roads_edited_over_5_years_share": statistics.mean(stale) if stale else None,
        "timestamp_audit": audit, **cleaning,
    }
    warnings = [COVERAGE_WARNING]
    if road_km / area < 0.3:
        warnings.append(f"Sparse road network: density={road_km / area:.3f} km/km2 is below 0.3.")
    for name, message in (("hospitals", "No hospitals/clinics found in the AOI."),
                          ("settlements", "No settlements found in the AOI."),
                          ("buildings", "Zero buildings found in the AOI.")):
        if counts[name] == 0:
            warnings.append(message)
    if cleaning["dropped_building_relations"]:
        warnings.append(f"Dropped {cleaning['dropped_building_relations']} building relations/multipolygons; v1 supports building ways only.")
    if cleaning["repaired_polygons"]:
        warnings.append(f"Repaired {cleaning['repaired_polygons']} invalid building polygons with make_valid.")
    if cleaning["invalid_features_dropped"]:
        warnings.append(f"Dropped {cleaning['invalid_features_dropped']} features with missing/invalid geometry.")
    return quality, warnings


def save_osm(bundle: OSMBundle, path: str | Path) -> None:
    """Write four base layers; bridges are reconstructed as a road view."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    storage = "gpkg"
    try:
        for name in LAYER_COLUMNS:
            getattr(bundle, name).to_file(path, layer=name, driver="GPKG", index=False)
    except Exception as exc:
        storage = "geojson"
        warning = f"GeoPackage writing failed ({type(exc).__name__}); saved one GeoJSON per base layer instead."
        if warning not in bundle.warnings:
            bundle.warnings.append(warning)
        for name in LAYER_COLUMNS:
            (path.parent / f"{name}.geojson").write_text(getattr(bundle, name).to_json(drop_id=True))
    metadata = {**bundle.to_dict(), "storage_format": storage}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False))


def load_osm(path: str | Path) -> OSMBundle:
    """Restore cached layers and independently recheck the saved timestamp audit."""
    path = Path(path)
    if path.is_dir():
        path = path / "osm.gpkg"
    metadata = json.loads(path.with_suffix(".json").read_text())
    crs = CRS.from_user_input(metadata["crs"])
    frames = {}
    for name in LAYER_COLUMNS:
        if metadata["storage_format"] == "gpkg":
            frame = gpd.read_file(path, layer=name)
        else:
            payload = json.loads((path.parent / f"{name}.geojson").read_text())
            if not payload["features"]:
                frame = _frame([], name, crs)
            else:
                frame = gpd.GeoDataFrame.from_features(payload["features"], crs=crs)
        for column in ("is_bridge", "is_tunnel", "is_ford", "in_aoi"):
            if column in frame:
                frame[column] = frame[column].fillna(False).astype(bool)
        if name == "settlements":
            frame["population"] = frame["population"].astype("Int64")
        frames[name] = frame.to_crs(crs)
    snapshot = _timestamp(metadata["snapshot_date"])
    audit = metadata["quality"]["timestamp_audit"]
    newest = _timestamp(audit["newest_edit"])
    verified = bool(
        snapshot is not None and newest is not None and newest <= snapshot
        and audit["missing_timestamps"] == 0 and audit["newer_elements"] == 0
        and audit["snapshot_verified"] and metadata["snapshot_verified"]
    )
    for frame in frames.values():
        for value in frame.timestamp:
            edit = _timestamp(value)
            if edit is None or snapshot is None or edit > snapshot:
                verified = False
    extent = context_bbox(tuple(metadata["bbox"]), metadata["quality"]["context_buffer_km"])
    _verify_scope(frames, tuple(metadata["bbox"]), extent)
    warnings = metadata["warnings"]
    if not verified and not any("UNVERIFIED SNAPSHOT" in value for value in warnings):
        warnings.append("UNVERIFIED SNAPSHOT: cache timestamps do not verify the requested snapshot.")
    return OSMBundle(
        **frames, snapshot_date=metadata["snapshot_date"], snapshot_verified=verified,
        newest_edit=audit["newest_edit"], endpoint_used=metadata["endpoint_used"],
        crs=crs, bbox=tuple(metadata["bbox"]), quality=metadata["quality"], warnings=warnings,
        cache_path=str(path.resolve()),
    )


def write_quicklook(bundle: OSMBundle, path: str | Path) -> None:
    """Render roads by class, red bridges, buildings, dots and hospital crosses."""
    extent = context_bbox(bundle.bbox, bundle.quality["context_buffer_km"])
    boundary = gpd.GeoSeries([box(*extent)], crs=4326).to_crs(bundle.crs).iloc[0]
    west, south, east, north = boundary.bounds
    width = 1200
    height = max(1, round(width * (north - south) / (east - west)))
    affine = from_bounds(west, south, east, north, width, height)
    pixel = max((east - west) / width, (north - south) / height)
    image = np.full((height, width, 3), 250, dtype="uint8")

    def paint(geometries: Any, color: tuple[int, int, int]) -> None:
        shapes = [(geometry, 1) for geometry in geometries if not geometry.is_empty]
        if shapes:
            mask = rasterize(shapes, out_shape=(height, width), transform=affine, all_touched=True)
            image[mask.astype(bool)] = color

    paint(bundle.buildings.geometry, (215, 215, 215))
    colors = {"motorway": (215, 120, 20), "trunk": (215, 120, 20), "primary": (220, 165, 30),
              "secondary": (60, 110, 170), "tertiary": (90, 135, 185)}
    for highway, frame in bundle.roads.groupby("highway", sort=True):
        key = highway.removesuffix("_link")
        paint(frame.geometry.buffer(pixel * (0.7 if key in colors else 0.3)), colors.get(key, (120, 120, 120)))
    paint(bundle.bridges.geometry.buffer(pixel), (230, 20, 20))
    paint(bundle.settlements.geometry.buffer(pixel * 2.5), (20, 120, 50))
    crosses = []
    for point in bundle.hospitals.geometry:
        crosses.extend([
            LineString([(point.x - pixel * 5, point.y), (point.x + pixel * 5, point.y)]).buffer(pixel),
            LineString([(point.x, point.y - pixel * 5), (point.x, point.y + pixel * 5)]).buffer(pixel),
        ])
    paint(crosses, (115, 15, 100))
    aoi_boundary = gpd.GeoSeries([box(*bundle.bbox).boundary], crs=4326).to_crs(bundle.crs).iloc[0]
    paint([aoi_boundary.buffer(pixel * 0.3)], (60, 60, 60))
    rows = np.zeros((height, width * 3 + 1), dtype="uint8")
    rows[:, 1:] = image.reshape(height, -1)

    def chunk(kind: bytes, value: bytes) -> bytes:
        return struct.pack("!I", len(value)) + kind + value + struct.pack("!I", zlib.crc32(kind + value))

    header = struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                         + chunk(b"IDAT", zlib.compress(rows.tobytes())) + chunk(b"IEND", b""))


def build_osm(
    aoi: AOI, stack: S1Stack | None = None, cache_dir: str | Path = "data/cache",
    snapshot_offset_days: int = 0, strict: bool = True, context_buffer_km: float = 10,
    session: Any = None, crs: Any = None, post_fn: Callable[..., Any] | None = None,
    endpoints: tuple[str, ...] = ENDPOINTS,
) -> OSMBundle:
    """Build a verified historical bundle; context has major roads and points only."""
    if type(strict) is not bool:
        raise ValueError("strict must be explicitly True or False")
    snapshot = snapshot_datetime(aoi, snapshot_offset_days)
    extent = context_bbox(aoi.bbox, context_buffer_km)
    target_crs = CRS.from_user_input(crs if crs is not None else (stack.crs if stack else "EPSG:32645"))
    cache_dir = Path(cache_dir).resolve()
    key = _key([aoi.bbox, _iso(snapshot), float(context_buffer_km), target_crs.to_string(), QUERY_VERSION])
    path = cache_dir / "osm" / key / "osm.gpkg"
    if path.with_suffix(".json").exists():
        cached = load_osm(path)
        if cached.snapshot_verified or not strict:
            if not path.with_name("quicklook.png").exists():
                write_quicklook(cached, path.with_name("quicklook.png"))
            return cached
    client = OverpassClient(session=session, post_fn=post_fn)
    try:
        records, endpoint, audit, acquisition_warnings = _acquire(
            aoi, snapshot, extent, cache_dir, client, strict, endpoints,
        )
    finally:
        client.close()
    elements, conflicts = _deduplicate(records)
    if conflicts and strict:
        raise OSMSnapshotError("Conflicting object versions were returned for a single date-locked snapshot")
    frames, cleaning = _extract(elements, aoi.bbox, extent, target_crs)
    quality, warnings = _quality(frames, elements, aoi.bbox, snapshot, audit, cleaning)
    quality["context_buffer_km"] = context_buffer_km
    quality["context_road_classes"] = list(CONTEXT_ROADS)
    warnings = [*aoi.warnings, *acquisition_warnings, *warnings]
    warnings.append("Way/relation point features use Overpass bbox centers, not facility entrances or population centroids.")
    if stack is None and crs is None:
        warnings.append("No S1 stack or explicit CRS supplied; using default EPSG:32645.")
    if conflicts:
        audit["snapshot_verified"] = False
        warnings.append("UNVERIFIED SNAPSHOT: conflicting historical object versions; selected the newest returned version.")
    if crs is not None and stack is not None and target_crs != CRS.from_user_input(stack.crs):
        warnings.append("Explicit CRS overrides the S1 grid CRS; feature coordinates are not in the radar grid CRS.")
    bundle = OSMBundle(
        **frames, snapshot_date=_iso(snapshot), snapshot_verified=audit["snapshot_verified"],
        newest_edit=audit["newest_edit"], endpoint_used=endpoint, crs=target_crs, bbox=aoi.bbox,
        quality=quality, warnings=list(dict.fromkeys(warnings)), cache_path=str(path),
    )
    save_osm(bundle, path)
    write_quicklook(bundle, path.with_name("quicklook.png"))
    return bundle


def _cached_crs(aoi: AOI, cache_dir: Path) -> str | None:
    """Inspect S1 sidecars without loading raster pixels or contacting a catalog."""
    for path in sorted((cache_dir / "s1").glob("*/stack.json")):
        metadata = json.loads(path.read_text())
        if tuple(metadata.get("bbox", [])) == aoi.bbox:
            return metadata.get("crs")
    return None


def main() -> None:
    """Print verified pre-event OSM metadata and save a map preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    area = parser.add_mutually_exclusive_group(required=True)
    area.add_argument("--place")
    area.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--date", required=True, dest="flood_date")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--snapshot-offset-days", type=int, default=0)
    parser.add_argument("--context-buffer-km", type=float, default=10)
    args = parser.parse_args()
    started = time.perf_counter()
    aoi = build_aoi(place=args.place, bbox=args.bbox, flood_date=args.flood_date)
    try:
        bundle = build_osm(
            aoi, cache_dir=args.cache_dir, crs=_cached_crs(aoi, Path(args.cache_dir)),
            snapshot_offset_days=args.snapshot_offset_days, context_buffer_km=args.context_buffer_km,
        )
    except OSMError as exc:
        parser.exit(1, f"OSM error: {exc}\n")
    metadata = bundle.to_dict()
    metadata["runtime_seconds"] = time.perf_counter() - started
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
