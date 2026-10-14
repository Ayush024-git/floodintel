"""Offline-capable static dashboard and portable incident package."""
from __future__ import annotations

import argparse
import base64
import hashlib
from html import escape
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.features import shapes
from shapely.geometry import box, mapping, shape
from shapely import transform as transform_coordinates

from src.data.osm import load_osm
from src.report.build_report import (BANNER, MISSING, build_report_model, clean_text,
                                     comparison_table_html, format_number, read_json, render_report_html, render_report_md)

# Display geometry is approximate. Original uncapped GeoJSON remains in the ZIP.
MAPLIBRE_VERSION = "4.7.1"
HTML_BUDGET_BYTES = 12 * 1024 * 1024
DEFAULT_CAPS = dict(roads=2000, hospitals=500, settlements=600, assets=1000,
                    buildings=250, flood=500, preexisting=250, excluded=250,
                    debris=250, chokepoints=150, routes=100)
VIEWS = ("IMPACT", "INFRASTRUCTURE", "ISOLATION", "CHOKEPOINTS", "PRIORITIES", "TRUST", "PROVENANCE")
MAP_LAYER_ORDER = ("preexisting", "excluded", "flood", "debris", "buildings", "roads", "assets", "chokepoints", "hospitals", "settlements", "routes")
COLORS = dict(roads="#566576", hospitals="#eef6ff", settlements="#46c894", assets="#ef626b",
              buildings="#ff9d94", flood="#328eff", preexisting="#9ba9b4", excluded="#f8a84f",
              debris="#e4ce50", chokepoints="#bdb2ff", routes="#b1d3f5")


def _views(data: dict[str, Any]) -> tuple[str, ...]:
    return (*VIEWS, "VALIDATION") if data["validation_supplied"] else VIEWS


def _project(function: Any, geometry: Any) -> Any:
    return transform_coordinates(geometry, lambda coordinates: np.column_stack(function(coordinates[:, 0], coordinates[:, 1])))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _safe_script(value: Any) -> str:
    return (_json(value).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(type="FeatureCollection", features=features)


def _round_geometry(geometry: dict[str, Any]) -> dict[str, Any]:
    """Round display coordinates only; source geometry and measurements stay intact."""
    def coordinates(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [coordinates(item) for item in value]
        return round(value, 6) or 0 if isinstance(value, (int, float)) else value

    return {key: coordinates(value) if key == "coordinates" else
            [_round_geometry(part) for part in value] if key == "geometries" else value
            for key, value in geometry.items()}


def _identifier(feature: dict[str, Any]) -> str:
    props = feature.get("properties", {})
    return str(props.get("osm_id") or props.get("link_id") or props.get("component_id") or _json(feature.get("geometry")))


def _simplify_layer(collection: dict[str, Any], metric: CRS, tolerance: float, cap: int,
                    *, source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    to_metric = Transformer.from_crs(4326, metric, always_xy=True).transform
    to_geo = Transformer.from_crs(metric, 4326, always_xy=True).transform
    candidates, invalid = [], 0
    for feature in collection.get("features", []):
        try:
            geometry = _project(to_metric, shape(feature["geometry"]))
            if geometry.is_empty or not geometry.is_valid:
                invalid += 1
                continue
            importance = geometry.area if "Polygon" in geometry.geom_type else geometry.length
            simplified = geometry.simplify(tolerance, preserve_topology=True)
            if simplified.is_empty or not simplified.is_valid:
                invalid += 1
                continue
            rounded = _round_geometry(mapping(_project(to_geo, simplified)))
            if not shape(rounded).is_valid or shape(rounded).is_empty:
                invalid += 1
                continue
            candidates.append((importance, _identifier(feature), dict(type="Feature", properties=feature.get("properties", {}),
                                                                   geometry=rounded)))
        except (KeyError, TypeError, ValueError):
            invalid += 1
    candidates.sort(key=lambda row: (-row[0], row[1]))
    kept = [row[2] for row in candidates[:cap]]
    return _collection(kept), dict(source=source, input_features=len(collection.get("features", [])),
                                  kept_features=len(kept), capped_features=max(0, len(candidates) - cap),
                                  invalid_features=invalid, simplify_m=tolerance, coordinate_decimals=6, cap=cap)


def _raster_layers(directory: Path) -> dict[str, dict[str, Any]]:
    """Read saved categorical bands; no model, raster processing or new inference."""
    path = directory / "flood_masks.tif"
    if not path.is_file():
        path = directory / "flood.tif"
    if not path.is_file():
        return {}
    results = {}
    with rasterio.open(path) as dataset:
        projector = Transformer.from_crs(dataset.crs, 4326, always_xy=True).transform
        descriptions = {name: index for index, name in enumerate(dataset.descriptions, 1)}
        for layer, band in (("flood", "confidence"), ("preexisting", "preexisting_water"), ("excluded", "terrain_excluded")):
            if band not in descriptions:
                continue
            values = dataset.read(descriptions[band])
            values = np.where(np.isfinite(values), values, 0).astype(np.uint8)
            features = []
            for geometry, value in shapes(values, mask=values > 0, transform=dataset.transform):
                features.append(dict(type="Feature", geometry=mapping(_project(projector, shape(geometry))),
                                     properties=dict(tier=int(value), layer=layer)))
            results[layer] = _collection(features)
    return results


def _cached_basemap(model: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Reuse the audited OSM cache; never contact its endpoint."""
    value = model["incident"].get("layers", {}).get("osm")
    empty = _collection([])
    if not value:
        return empty, empty, ["Full cached OSM basemap is not available; potentially affected roads remain visible."]
    path = Path(value)
    if not path.is_absolute():
        path = Path(model["paths"]["incident_dir"]) / path
    try:
        bundle = load_osm(path)
        if not bundle.snapshot_verified:
            return empty, empty, ["Cached OSM snapshot is unverified; full basemap omitted."]
        layers = []
        for frame, columns in ((bundle.roads, ("osm_id", "segment_id", "highway", "name", "is_bridge")),
                               (bundle.hospitals, ("osm_id", "name", "tier", "in_aoi"))):
            selected = frame[[column for column in columns if column in frame] + ["geometry"]].to_crs(4326)
            layers.append(json.loads(selected.to_json(drop_id=True)))
        return *layers, []
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return empty, empty, [f"Full cached OSM basemap is not available: {type(error).__name__}. Potentially affected roads remain visible."]


def build_dashboard_data(model: dict[str, Any], *, simplify_m: float = 5.0,
                         caps: dict[str, int] | None = None) -> dict[str, Any]:
    """Cap display geometry; budget recovery may explicitly omit other display blocks.

    Default HTML hard budget is twelve MiB including the SVG fallback. Geometry
    caps select longest lines/largest polygons, then stable ids; routes keep
    baseline/likely geometry only. Capping never changes reported source totals.
    """
    limits = {**DEFAULT_CAPS, **(caps or {})}
    if set(limits) != set(DEFAULT_CAPS) or simplify_m < 0 or any(type(value) is not int or value < 0 for value in limits.values()):
        raise ValueError("Invalid display caps or simplification tolerance")
    bbox = model["incident"]["aoi"]["bbox"]
    longitude, latitude = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    metric = CRS.from_epsg((32600 if latitude >= 0 else 32700) + min(60, max(1, int((longitude + 180) / 6) + 1)))
    roads, hospitals, warnings = _cached_basemap(model)
    geo = model["geo"]
    assets = geo["damaged_assets"]["features"]
    raw = dict(roads=roads, hospitals=hospitals, settlements=geo["settlements"],
               assets=_collection([row for row in assets if row["properties"].get("asset_class") != "building"]),
               buildings=_collection([row for row in assets if row["properties"].get("asset_class") == "building"]),
               chokepoints=geo["chokepoints"], flood=geo.get("flood_extent", _collection([])),
               preexisting=_collection([]), excluded=_collection([]), debris=geo.get("debris_candidates", _collection([])))
    gap_ids = {row["id"] for row in model["priorities"]["data_gaps"]}
    raw["settlements"] = _collection([dict(feature, properties=dict(feature["properties"],
                                   data_gap=feature["properties"].get("osm_id") in gap_ids))
                                   for feature in geo["settlements"]["features"]])
    raster_names = set()
    if model["synthetic"]:
        bounds = model["impact"].get("flood_params", {}).get("test_polygon")
        if bounds:
            raw["flood"] = _collection([dict(type="Feature", geometry=mapping(box(*bounds)),
                                             properties=dict(tier=3, label="Synthetic test-box boundary"))])
    elif model["paths"]["flood_dir"]:
        try:
            raster = _raster_layers(Path(model["paths"]["flood_dir"]))
            raw.update(raster)
            raster_names = set(raster)
            if "flood" not in raster:
                warnings.append("Flood polygons use component maximum confidence; pixel-tier geometry is not available.")
            for name in ("preexisting", "excluded"):
                if name not in raster:
                    warnings.append(f"{name} geometry is not available: saved categorical raster missing.")
        except (OSError, ValueError) as error:
            warnings.append(f"Saved categorical raster could not be read ({type(error).__name__}); component vectors used instead.")
    route_path = Path(model["paths"]["impact_dir"]) / "routes.geojson"
    route_collection = read_json(route_path) if route_path.is_file() else _collection([])
    all_routes = route_collection.get("features", [])
    raw["routes"] = _collection([row for row in all_routes if row.get("properties", {}).get("scenario") in ("baseline", "likely")])
    layers, metadata = {}, {}
    for name in DEFAULT_CAPS:
        layers[name], metadata[name] = _simplify_layer(raw[name], metric, simplify_m, limits[name],
                                                      source="saved raster bands" if name in raster_names else "cached vector features")
    metadata["routes"]["omitted_possible_routes"] = len(all_routes) - len(raw["routes"]["features"])
    # Feature properties used by the map are compact; complete evidence lives in records.
    for name, layer in layers.items():
        for feature in layer["features"]:
            props = feature["properties"]
            feature["properties"] = {key: props[key] for key in ("osm_id", "segment_id", "link_id", "name", "highway", "asset_class", "is_bridge",
                                                                "tier", "max_confidence", "status", "data_gap", "settlement_id", "scenario", "destination_class", "label") if key in props}
    subjects = [row["properties"] for row in geo["settlements"]["features"]]
    return dict(report=model["report"].copy(), synthetic=model["synthetic"], no_flood=model["no_flood"],
                validation_supplied=model["validation"] is not None, bbox=[round(value, 6) for value in bbox],
                pair=model["incident"]["pair_summary"], layers=layers, settlements=subjects,
                assets=[row["properties"] for row in assets], clusters=model["clusters"],
                chokepoints=model["chokepoints"].copy(), priorities=model["priorities"].copy(), flags=model["flags"],
                warnings=[clean_text(value) for value in dict.fromkeys([*model["notices"], *warnings,
                          *model["impact"].get("warnings", []), *model["flood"].get("warnings", [])])],
                graph_quality=model["impact"].get("graph_quality", {}), counts=model["impact"].get("counts", {}),
                status_tallies=model["impact"].get("status_tallies", {}), flood_stats=model["flood"].get("stats", {}),
                osm_quality=model["osm_metadata"].get("quality", {}), provenance=model["incident"]["provenance"],
                parameters=dict(flood=model["impact"].get("flood_params", {}), impact=model["impact"].get("params", {})),
                display_metadata=dict(html_budget_bytes=HTML_BUDGET_BYTES, geometry_layers=metadata,
                                      originals="Uncapped original JSON/GeoJSON in the ZIP; display geometry is approximate.",
                                      context="Context roads and destinations are included from the audited OSM cache.",
                                      generation="Incident receipt timestamp; no new analysis or capture date is implied."))


def _color(layer: str, properties: dict[str, Any]) -> str:
    if layer == "settlements":
        if properties.get("data_gap"):
            return "#9daec1"
        return {"CUT OFF": "#ed6270", "UNCERTAIN": "#efb44c", "CONNECTED": "#48c799"}.get(properties.get("status"), "#8c9bac")
    if layer == "flood":
        return {1: "#80aef3", 2: "#3689e6", 3: "#1251d6"}.get(properties.get("tier", properties.get("max_confidence")), COLORS[layer])
    return COLORS[layer]


def _legend() -> str:
    """Label each categorical marker and line style in both map renderers."""
    entries = [("strong flood", "#1251d6", "area"), ("probable flood", "#3689e6", "area"),
               ("uncertain flood", "#80aef3", "area"), ("pre-existing water", "#9ba9b4", "area"),
               ("terrain excluded", "#f8a84f", "area"), ("debris candidates", "#e4ce50", "area"),
               ("settlement CUT OFF", "#ed6270", "dot"), ("settlement UNCERTAIN (flood evidence)", "#efb44c", "dot"),
               ("settlement CONNECTED", "#48c799", "dot"), ("data gap (no baseline access in OSM)", "#9daec1", "diamond"),
               ("hospital", "#eef6ff", "cross"), ("clinic", "#eef6ff", "square"),
               ("mapped road", "#566576", "line"), ("potentially affected road / bridge", "#ef626b", "line"),
               ("potentially affected building", "#ff9d94", "area"), ("chokepoint / latent bridge", "#bdb2ff", "line"),
               ("baseline route", "#ffffff", "dash"), ("post-flood detour", "#6ae6c3", "line"),
               ("responsible link", "#ffdf5a", "line")]
    markers = dict(dot='<circle cx="8" cy="8" r="4"/>', area='<rect x="3" y="3" width="10" height="10"/>',
                   diamond='<path d="M8,2 L14,8 L8,14 L2,8 Z" fill="none"/>',
                   cross='<path d="M2,8 L14,8 M8,2 L8,14" fill="none"/>',
                   square='<rect x="3" y="3" width="10" height="10" fill="none"/>',
                   line='<path d="M1,8 L15,8"/>', dash='<path d="M1,8 L15,8" stroke-dasharray="3,2"/>')
    return ''.join(f'<span><svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" fill="{color}" stroke="{color}" stroke-width="2">'
                   + markers[marker] + '</svg> ' + escape(label) + '</span>' for label, color, marker in entries)


def static_svg(data: dict[str, Any]) -> str:
    """An embedded, keyboard-clickable SVG uses exactly the capped map layers."""
    features = [(name, feature) for name in MAP_LAYER_ORDER for feature in data["layers"][name]["features"]]
    metric = Transformer.from_crs(4326, 3857, always_xy=True).transform
    geometries = [(name, feature, _project(metric, shape(feature["geometry"]))) for name, feature in features]
    extent = _project(metric, box(*data["bbox"]))
    xmin, ymin, xmax, ymax = extent.bounds
    for _, _, geometry in geometries:
        left, bottom, right, top = geometry.bounds
        xmin, ymin, xmax, ymax = min(xmin, left), min(ymin, bottom), max(xmax, right), max(ymax, top)
    scale = min(960 / max(xmax - xmin, 1), 600 / max(ymax - ymin, 1))

    def point(x: float, y: float) -> tuple[float, float]:
        return 20 + (x - xmin) * scale, 620 - (y - ymin) * scale

    def line(coords: Any) -> str:
        return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in (point(*coord[:2]) for coord in coords))

    def draw(geometry: Any, layer: str, props: dict[str, Any]) -> str:
        if geometry.geom_type == "Point":
            x, y = point(geometry.x, geometry.y)
            if layer == "hospitals":
                if props.get("tier") == "clinic":
                    return f'<rect data-marker="clinic-square" x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8" fill="none" stroke-width="2"/>'
                return f'<path data-marker="hospital-cross" d="M{x-5:.1f},{y:.1f} L{x+5:.1f},{y:.1f} M{x:.1f},{y-5:.1f} L{x:.1f},{y+5:.1f}" fill="none" stroke-width="2"/>'
            if layer == "settlements" and props.get("data_gap"):
                return f'<path data-marker="data-gap-diamond" d="M{x:.1f},{y-5:.1f} L{x+5:.1f},{y:.1f} L{x:.1f},{y+5:.1f} L{x-5:.1f},{y:.1f} Z" fill="none" stroke-width="2"/>'
            return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5"/>'
        if geometry.geom_type in ("LineString", "LinearRing"):
            return f'<path d="{line(geometry.coords)}" fill="none"/>'
        if geometry.geom_type == "Polygon":
            paths = line(geometry.exterior.coords) + " Z" + "".join(line(ring.coords) + " Z" for ring in geometry.interiors)
            return f'<path d="{paths}" fill-rule="evenodd"/>'
        return "".join(draw(part, layer, props) for part in geometry.geoms)

    output = ['<svg id="offline-svg" viewBox="0 0 1000 640" role="img" aria-label="Offline incident map"><rect width="1000" height="640" fill="#101a28"/>']
    for name, feature, geometry in geometries:
        props = feature["properties"]
        identifier = str(props.get("osm_id") or props.get("link_id") or props.get("segment_id") or "")
        width = 3 if props.get("is_bridge") and name == "assets" else 1
        color = _color(name, props)
        extra = ''
        if name == "routes":
            color = "#ffffff" if props.get("scenario") == "baseline" else "#6ae6c3"
            extra = f' data-route-subject="{escape(str(props.get("settlement_id", "")), quote=True)}" style="display:none"'
            width = 3
        if props.get("link_id"):
            extra += f' data-link-id="{escape(str(props["link_id"]), quote=True)}"'
        output.append(f'<g data-map-layer="{name}"{extra} fill="{color}" stroke="{color}" stroke-width="{width}" opacity=".8">')
        if identifier:
            output.append(f'<a href="#evidence" data-select="{escape(identifier, quote=True)}"><title>{escape(clean_text(props.get("name") or identifier))}</title>{draw(geometry, name, props)}</a>')
        else:
            output.append(draw(geometry, name, props))
        output.append('</g>')
    output.append('</svg>')
    return "".join(output)


def _table(headers: tuple[str, ...], rows: list[list[Any]]) -> str:
    return '<div class="table-wrap"><table><thead><tr>' + "".join(f'<th scope="col">{escape(header)}</th>' for header in headers) + '</tr></thead><tbody>' + "".join(
        '<tr>' + "".join(f'<td>{escape(clean_text(value))}</td>' for value in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def _select_button(identifier: str, name: Any) -> str:
    return f'<button class="select-object" data-select="{escape(identifier, quote=True)}">{escape(clean_text(name or identifier))}</button>'


def _object_table(headers: tuple[str, ...], records: list[dict[str, Any]], fields: tuple[str, ...], id_key: str) -> str:
    body = '<div class="table-wrap"><table><thead><tr>' + "".join(f'<th scope="col">{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>'
    for row in records:
        body += '<tr><td>' + _select_button(row[id_key], row.get("name")) + '</td>'
        for field in fields:
            value = row.get(field)
            if isinstance(value, (int, float)):
                value = format_number(value)
            if isinstance(value, list):
                value = "; ".join(clean_text(item) for item in value)
            body += '<td>' + escape(clean_text(value)) + '</td>'
        body += '</tr>'
    return body + '</tbody></table></div>'


def _panels(data: dict[str, Any]) -> str:
    report = data["report"]
    cards = {}
    cards["IMPACT"] = ''.join(f'<p>{escape(clean_text(value))}</p>' for value in report["findings"][:3]) + '<p>Debris layers are candidates requiring inspection.</p>'
    assets = data["assets"]
    cards["INFRASTRUCTURE"] = '<p>Potentially affected / inundated; structural condition is not established.</p>' + _object_table(
        ("Asset", "Class", "Tier", "Overlap m"), assets, ("asset_class", "tier", "length_in_flood_m"), "osm_id")
    road_km = data["counts"].get("road_km_by_highway", {})
    inundated = data["counts"].get("inundated_road_km_by_highway", {})
    cards["INFRASTRUCTURE"] += _table(("Highway", "Hit segment km", "Inundated overlap km"), [[key, format_number(road_km.get(key)), format_number(inundated.get(key))] for key in sorted(set(road_km) | set(inundated))])
    cards["ISOLATION"] = '<p>' + escape(report["findings"][4]) + '</p>' if len(report["findings"]) > 4 else ''
    if report["data_gap_note"]:
        cards["ISOLATION"] += '<p>' + escape(report["data_gap_note"]) + '</p>'
    cards["ISOLATION"] += _object_table(("Settlement", "Status", "Access", "Reason"), data["settlements"], ("status", "access_detail", "reason"), "osm_id")
    cards["ISOLATION"] += '<h3>Clusters</h3>' + _table(("Cluster", "Members", "Mapped buildings", "Lost access"), [[row["cluster_id"], "; ".join(row["member_settlements"]), row["total_mapped_buildings"], "; ".join(row["lost_classes"])] for row in data["clusters"]])
    cards["CHOKEPOINTS"] = ''
    for title, key in (("Damaged links with restoration leverage", "damaged"), ("Latent critical bridges — baseline risk", "latent"), ("Damaged links with no single-link leverage", "damaged_no_leverage")):
        cards["CHOKEPOINTS"] += '<h3>' + title + '</h3>' + _object_table(("Link", "Settlements", "Mapped buildings", "Evidence"), data["chokepoints"][key], ("count", "mapped_buildings", "sentence"), "link_id")
    cards["PRIORITIES"] = '<p>' + escape(report["priorities_note"]) + '</p><p>Mapped and satellite evidence only; not a measure of humanitarian need.</p>'
    if any(row["name"] == "priorities.settlements" for row in data["display_metadata"].get("omitted_blocks", [])):
        cards["PRIORITIES"] += '<p>Flood-attributable priority table omitted for display size; consult original priorities.json in the ZIP.</p>'
    elif not data["priorities"]["settlements"]:
        cards["PRIORITIES"] += '<p>No flood-attributable priorities</p>'
    for title, key in (("Flood-attributable settlements", "settlements"), ("Damaged links with restoration leverage", "links")):
        cards["PRIORITIES"] += '<h3>' + title + '</h3>' + _object_table(("Object", "Score", "Confidence", "Why"), data["priorities"][key], ("score", "confidence", "why"), "id")
    cards["PRIORITIES"] += '<details><summary>Data gaps — never flood priorities</summary>' + _object_table(("Settlement", "Reason"), data["priorities"]["data_gaps"], ("reason",), "id") + '</details>'
    cards["TRUST"] = '<p>' + escape(report["incident_lines"][-1]) + '</p><h3>What not to trust</h3>' + ''.join(f'<p>{escape(clean_text(value))}</p>' for value in report["limitations"][:2])
    for severity in ("limiting", "caution", "info"):
        cards["TRUST"] += f'<h3>{severity.title()}</h3><ul>' + ''.join(f'<li>{escape(clean_text(row["message"]))}</li>' for row in data["flags"] if row["severity"] == severity) + '</ul>'
    for title, values in (("Graph baseline quality", data["graph_quality"]), ("Flood quality", data["flood_stats"]), ("OSM quality", data["osm_quality"]), ("Documented thresholds and heuristics", data["parameters"])):
        cards["TRUST"] += f'<details><summary>{title}</summary><pre>{escape(clean_text(json.dumps(values, ensure_ascii=False, indent=2)))}</pre></details>'
    cards["TRUST"] += '<details><summary>Display simplification and caps</summary><p>Display geometry is simplified and capped; original source files remain uncapped. Per-layer kept, capped and invalid-feature counts are recorded in the embedded display metadata and dashboard_meta.json in the package.</p></details>'
    cards["TRUST"] += '<details><summary>All warnings and unavailable layers</summary><ul>' + ''.join(f'<li>{escape(clean_text(value))}</li>' for value in data["warnings"]) + '</ul></details>'
    cards["PROVENANCE"] = '<p>Policy requires allowed inference sources and a verified pre-event OSM snapshot. The receipt below shows what was recorded.</p>' + '<pre>' + escape(clean_text(json.dumps(data["provenance"], ensure_ascii=False, indent=2))) + '</pre><p>OpenStreetMap contributors · Open Database License. Preserve source license notes in the package.</p>'
    comparison = report.get("comparison", {})
    comparison_html = ''.join(f'<p>{escape(clean_text(value))}</p>' for value in comparison.get("paragraphs", [])) + comparison_table_html(comparison)
    if data["validation_supplied"]:
        cards["VALIDATION"] = comparison_html
    else:
        cards["TRUST"] += '<p>No validation comparison supplied.</p>'
    return ''.join(f'<section class="view-panel" id="view-{view.lower()}" aria-labelledby="heading-{view.lower()}"><h2 id="heading-{view.lower()}">{view}</h2>{cards[view]}</section>' for view in _views(data))


CSS = r'''
:root{color-scheme:dark;--bg:#0a1019;--panel:#111c2a;--muted:#9daec1;--line:#29384c;--accent:#84bbff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#edf4fc;font:13px/1.5 system-ui,sans-serif}header{padding:14px 20px;border-bottom:1px solid var(--line)}h1{font-size:22px;margin:0}h2{font-size:17px}h3{font-size:13px;color:var(--accent);margin-top:20px}p{margin:8px 0}a{color:var(--accent)}button,input{font:inherit}button{cursor:pointer;background:#1b2c41;color:inherit;border:1px solid #38506a;border-radius:4px;padding:7px 10px}button:hover,button[aria-pressed=true]{background:#274f7b}button:focus-visible,a:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #f0c35b;outline-offset:3px}.muted{color:var(--muted)}.badge{border:1px solid #dfa854;border-radius:4px;padding:3px 8px;color:#f0c35b}.banner{position:sticky;top:0;z-index:5;background:#a82030;color:#fff;padding:10px 20px;font-weight:800;letter-spacing:.04em}.layout{display:grid;grid-template-columns:160px minmax(360px,1fr) 310px;min-height:calc(100vh - 130px)}nav{border-right:1px solid var(--line);padding:14px 10px}nav button{display:block;width:100%;text-align:left;margin-bottom:6px}.workspace{min-width:0;padding:14px}.map-shell{position:relative;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:#101a28}#map{height:430px;display:none}#offline-fallback svg{width:100%;height:430px}#offline-fallback p{padding:0 10px}.map-ready #map{display:block}.map-ready #offline-fallback{display:none}.toggles{display:flex;flex-wrap:wrap;gap:8px 14px;padding:9px 0}.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:12px}.legend b{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.view-panel{border-top:1px solid var(--line);margin-top:16px;padding:10px 0}.js .view-panel{display:none}.js .view-panel.active{display:block}aside{padding:15px;border-left:1px solid var(--line);background:var(--panel);min-width:0}aside h2{margin-top:0}.table-wrap{max-height:430px;overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th{position:sticky;top:0;background:#182436;text-align:left}th,td{padding:7px;border-bottom:1px solid var(--line);vertical-align:top}td{overflow-wrap:anywhere}.select-object{padding:3px 6px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.5 ui-monospace,monospace}details{padding:8px 0}summary{cursor:pointer;color:var(--accent)}footer{padding:12px 20px;border-top:1px solid var(--line)}.evidence-field{padding:8px 0;border-bottom:1px solid var(--line)}.evidence-field strong{display:block;color:var(--muted);font-size:11px;text-transform:uppercase}.evidence-field span{white-space:pre-wrap;overflow-wrap:anywhere}.downloads{display:flex;gap:14px;flex-wrap:wrap}@media(max-width:1000px){.layout{grid-template-columns:140px minmax(0,1fr)}aside{grid-column:1/-1;border-top:1px solid var(--line)}#map,#offline-fallback svg{height:360px}}@media(max-width:600px){.layout{display:block}nav{display:flex;gap:5px;overflow-x:auto;border-bottom:1px solid var(--line)}nav button{width:auto;white-space:nowrap;margin:0}.workspace{padding:9px}header{padding:12px}h1{font-size:19px}#map,#offline-fallback svg{height:300px}.table-wrap{max-height:360px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;animation:none!important;transition:none!important}}
'''

JS = r'''
'use strict';
const D=JSON.parse(document.getElementById('incident-data').textContent);
document.documentElement.classList.add('js');
let map=null, selected=null;
const tidy=v=>v===null||v===undefined?'not available':typeof v==='object'?JSON.stringify(v,null,2):String(v);
function activate(name){document.querySelectorAll('.view-panel').forEach(p=>p.classList.toggle('active',p.id==='view-'+name.toLowerCase()));document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===name)));}
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>activate(b.dataset.view)));activate('IMPACT');
function field(title,value){const div=document.createElement('div');div.className='evidence-field';const heading=document.createElement('strong');heading.textContent=title;const content=document.createElement('span');content.textContent=tidy(value);div.append(heading,content);document.getElementById('evidence-content').append(div);}
function showEvidence(id){selected=id;const output=document.getElementById('evidence-content');output.replaceChildren();const settlement=D.settlements.find(s=>s.osm_id===id);const link=[...D.chokepoints.damaged,...D.chokepoints.latent,...D.chokepoints.damaged_no_leverage].find(s=>s.link_id===id);const asset=D.assets.find(s=>s.osm_id===id||s.segment_id===id);const row=settlement||link||asset;if(!row){field('Object',id);field('Evidence','not available');return;}document.getElementById('evidence-title').textContent=row.name||id;field('Stable id',id);
if(settlement){field('Status',row.status);field('Reason',row.reason);field('Access detail',row.access_detail);for(const kind of ['hospital','clinic','town']){field('Nearest '+kind+' at baseline',row.nearest_baseline?.[kind]);field('Nearest '+kind+' after likely scenario',row.nearest_after?.[kind]);}field('Cut from',row.cut_from);field('Possible scenario access losses',row.possible_cut_from);field('Alternative route exists',row.alternative_route_exists);field('Detour factors',row.detour_factors);field('Confidence tier',row.confidence_tier);field('Confidence explanation',row.reason);const responsible=(row.responsible_links||[]).map(link_id=>{const evidence=(row.evidence||[]).find(e=>e.link_id===link_id);const restores=D.clusters.filter(c=>(c.member_settlements||[]).includes(id)).flatMap(c=>c.responsible_links||[]).filter(x=>x.link_id===link_id);return {link_id,osm_ids:evidence?.osm_ids||'not available',sufficient_to_restore:restores.length?restores.map(x=>x.sufficient_to_restore):'not available',flood_pixel_area_m2:evidence?.flood_pixel_area_m2??'not available'};});field('Responsible links and restoration test',responsible);field('Damage fractions, tiers and flood pixel area',row.evidence);field('Mapped buildings — size proxy only',row.mapped_buildings_nearby);}
else{for(const key of ['asset_class','is_bridge','damage_tier','tier','count','settlements_affected','hospital_access_reconnected','town_access_reconnected','length_in_flood_m','fraction_in_flood','bridge_approach_hit','approach_hit','evidence','sentence'])field(key.replaceAll('_',' '),row[key]);field('OSM ids',row.osm_ids||row.osm_id);field('Restoration test',D.clusters.flatMap(c=>c.responsible_links||[]).filter(x=>x.link_id===id));}
field('Run-level flags — not object-specific proof',D.flags.map(f=>({code:f.code,severity:f.severity,message:f.message})));field('Applicable evidence',row.evidence||'not available');
const ids=settlement?.responsible_links||[id];document.querySelectorAll('[data-route-subject]').forEach(g=>g.style.display=g.dataset.routeSubject===id?'':'none');document.querySelectorAll('[data-link-id]').forEach(g=>{g.style.stroke=ids.includes(g.dataset.linkId)?'#ffdf5a':'';g.style.strokeWidth=ids.includes(g.dataset.linkId)?'5':'';});
if(map?.getSource('selected-routes')){const routes=D.layers.routes.features.filter(f=>f.properties.settlement_id===id);map.getSource('selected-routes').setData({type:'FeatureCollection',features:routes});const links=D.layers.chokepoints.features.filter(f=>ids.includes(f.properties.link_id));map.getSource('selected-links').setData({type:'FeatureCollection',features:links});}else{field('Route geometry display','Capped baseline/likely routes are highlighted in the static SVG when available. Full routes.geojson is in the ZIP.');}
}
document.querySelectorAll('[data-select]').forEach(b=>b.addEventListener('click',event=>{event.preventDefault();showEvidence(b.dataset.select);document.getElementById('evidence').focus({preventScroll:true});}));
document.querySelectorAll('[data-layer]').forEach(input=>input.addEventListener('change',()=>{const name=input.dataset.layer;document.querySelectorAll('[data-map-layer="'+name+'"]').forEach(g=>g.style.display=input.checked?'':'none');if(map)for(const suffix of ['fill','line','point']){const id=name+'-'+suffix;if(map.getLayer(id))map.setLayoutProperty(id,'visibility',input.checked?'visible':'none');}}));
function fallback(reason){document.querySelector('.map-shell').classList.remove('map-ready');document.getElementById('map').removeAttribute('style');document.getElementById('map-status').textContent=reason+' Static SVG and full tables remain available.';}
function markerImages(){
const size=18, center=(size-1)/2;
for(const kind of ['hospital-cross','clinic-square','data-gap-diamond','settlement-dot']){
 const pixels=new Uint8Array(size*size*4);
 for(let y=0;y<size;y++)for(let x=0;x<size;x++){
  const dx=Math.abs(x-center),dy=Math.abs(y-center);
  const on=kind==='hospital-cross'?Math.max(dx,dy)<=7&&Math.min(dx,dy)<=1.5:
   kind==='clinic-square'?Math.max(dx,dy)<=6&&Math.max(dx,dy)>=4:
   kind==='data-gap-diamond'?dx+dy>=6&&dx+dy<=8:dx*dx+dy*dy<=16;
  if(on)pixels.set([255,255,255,255],(y*size+x)*4);
 }
 map.addImage(kind,{width:size,height:size,data:pixels},{sdf:true});
}}
function startMap(){if(!window.maplibregl){fallback('Map library unavailable offline.');return;}const holder=document.getElementById('map');holder.style.display='block';holder.style.position='absolute';holder.style.width='100%';holder.style.visibility='hidden';try{map=new maplibregl.Map({container:'map',style:{version:8,sources:{},layers:[{id:'background',type:'background',paint:{'background-color':'#101a28'}}]},center:[(D.bbox[0]+D.bbox[2])/2,(D.bbox[1]+D.bbox[3])/2],zoom:10,attributionControl:false});map.on('error',()=>fallback('Interactive map failed.'));map.on('load',()=>{markerImages();for(const name of D.map_layer_order){if(name==='routes')continue;const collection=D.layers[name];map.addSource(name,{type:'geojson',data:collection});const color=name==='settlements'?['case',['==',['get','data_gap'],true],'#9daec1',['match',['get','status'],'CUT OFF','#ed6270','UNCERTAIN','#efb44c','CONNECTED','#48c799','#8c9bac']]:name==='flood'?['match',['coalesce',['get','tier'],['get','max_confidence']],1,'#80aef3',2,'#3689e6',3,'#1251d6','#328eff']:D.colors[name];const visibility=document.querySelector('[data-layer="'+name+'"]').checked?'visible':'none';map.addLayer({id:name+'-fill',source:name,type:'fill',filter:['==',['geometry-type'],'Polygon'],layout:{visibility},paint:{'fill-color':color,'fill-opacity':0.5}});map.addLayer({id:name+'-line',source:name,type:'line',filter:['!=',['geometry-type'],'Point'],layout:{visibility},paint:{'line-color':color,'line-width':['case',['==',['get','is_bridge'],true],4,2]}});
const icon=name==='hospitals'?['match',['get','tier'],'clinic','clinic-square','hospital-cross']:['case',['==',['get','data_gap'],true],'data-gap-diamond','settlement-dot'];
map.addLayer({id:name+'-point',source:name,type:'symbol',filter:['==',['geometry-type'],'Point'],layout:{visibility,'icon-image':icon,'icon-allow-overlap':true},paint:{'icon-color':color}});
for(const suffix of ['fill','line','point']){const layer=name+'-'+suffix;map.on('click',layer,e=>{const p=e.features?.[0]?.properties;if(p)showEvidence(p.link_id||p.segment_id||p.osm_id||p.label||name);});map.on('mousemove',layer,e=>{map.getCanvas().style.cursor='pointer';const p=e.features?.[0]?.properties;if(p)document.getElementById('map-status').textContent=p.name||p.osm_id||p.link_id||p.label||name;});map.on('mouseleave',layer,()=>map.getCanvas().style.cursor='');}}
for(const name of ['selected-routes','selected-links'])map.addSource(name,{type:'geojson',data:{type:'FeatureCollection',features:[]}});map.addLayer({id:'baseline-route',type:'line',source:'selected-routes',filter:['==',['get','scenario'],'baseline'],paint:{'line-color':'#fff','line-width':3,'line-dasharray':[2,2]}});map.addLayer({id:'detour-route',type:'line',source:'selected-routes',filter:['==',['get','scenario'],'likely'],paint:{'line-color':'#6ae6c3','line-width':3}});map.addLayer({id:'responsible-link',type:'line',source:'selected-links',paint:{'line-color':'#ffdf5a','line-width':6}});holder.removeAttribute('style');document.querySelector('.map-shell').classList.add('map-ready');map.resize();map.fitBounds([[D.bbox[0],D.bbox[1]],[D.bbox[2],D.bbox[3]]],{padding:30,duration:0});document.getElementById('map-status').textContent='Select a mapped object for evidence. Hospital crosses, clinic squares, and data-gap diamonds are labelled in the legend.';if(selected)showEvidence(selected);});}catch(error){fallback('WebGL map unavailable.');}}
const loader=document.createElement('script');loader.src=D.maplibre_js;loader.crossOrigin='anonymous';loader.onload=startMap;loader.onerror=()=>fallback('Map library unavailable offline.');document.head.append(loader);
'''


def _render_page(model: dict[str, Any], data: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """Assemble the page and measure actual UTF-8 JSON, SVG and table bytes."""
    report = data["report"]
    script_hash = base64.b64encode(hashlib.sha256(JS.encode()).digest()).decode()
    csp = (f"default-src 'none'; script-src 'sha256-{script_hash}' https://unpkg.com; style-src 'unsafe-inline' https://unpkg.com; "
           "img-src data: blob:; worker-src blob:; child-src blob:; connect-src 'none'; font-src 'none'; base-uri 'none'; object-src 'none'")
    banner = f'<div class="banner" role="alert">{BANNER}</div>' if data["synthetic"] else ''
    nav = ''.join(f'<button data-view="{view}" aria-controls="view-{view.lower()}" aria-pressed="false">{view}</button>' for view in _views(data))
    toggles = ''.join(f'<label><input type="checkbox" data-layer="{name}" checked> {name.replace("excluded", "terrain excluded").replace("debris", "debris candidates").replace("preexisting", "pre-existing water")}</label>' for name in data["layers"] if name != "routes")
    legend = _legend()
    header = escape(report["title"])
    pair = escape(report["incident_lines"][1])
    trust = escape(report["incident_lines"][-1])
    omissions = data["display_metadata"].get("omitted_blocks", [])
    notice = ('<div class="budget-note" role="status" style="background:#442c12;color:#ffe0a4;padding:12px">Display blocks omitted to meet the HTML size limit: '
              + escape(", ".join(row["name"] for row in omissions))
              + '. Source findings are unchanged. Original JSON/GeoJSON remain uncapped in the package.</div>') if omissions else ''
    svg, panels, embedded = static_svg(data), _panels(data), _safe_script(data)
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="Content-Security-Policy" content="{escape(csp, quote=True)}"><title>FloodIntel · {header}</title><link rel="stylesheet" href="https://unpkg.com/maplibre-gl@{MAPLIBRE_VERSION}/dist/maplibre-gl.css"><style>{CSS}</style></head><body>{banner}<header><h1>{header} · {escape(clean_text(model['incident']['aoi'].get('flood_date')))}</h1><div class="muted">{escape(report['incident'])} · {pair} <span class="badge">{trust}</span></div><p>{escape(report['disclaimer'])}</p><div class="downloads"><a href="situation_report.html">Print-ready report</a><a href="situation_report.md" download>Text report</a><a href="incident_package.zip" download>Incident package</a></div></header>{notice}<div class="layout"><nav aria-label="Situation views">{nav}</nav><main class="workspace">{'<p>No flood detected. Baseline access, data quality and latent bridge risks remain visible.</p>' if data['no_flood'] else ''}<div class="map-shell"><div id="map" aria-label="Interactive incident map"></div><div id="offline-fallback">{svg}<p>Offline static preview — same capped layers. Use the tables to inspect evidence.</p></div></div><p id="map-status" class="muted" aria-live="polite">Static preview is available while the map library loads.</p><div class="toggles" aria-label="Map layers">{toggles}</div><div class="legend">{legend}</div>{panels}</main><aside id="evidence" tabindex="-1" aria-labelledby="evidence-heading"><h2 id="evidence-heading">EVIDENCE</h2><h3 id="evidence-title">Select a settlement, link or bridge</h3><div id="evidence-content"><p>Choose an object in a table or on the map. Unavailable evidence is shown explicitly.</p></div></aside></div><footer>{escape(report['disclaimer'])} OpenStreetMap contributors · ODbL. {escape(clean_text(data['display_metadata']['originals']))}</footer><script type="application/json" id="incident-data">{embedded}</script><script>{JS}</script></body></html>'''
    return html, dict(html_bytes=len(html.encode("utf-8")), embedded_json_bytes=len(embedded.encode("utf-8")),
                      static_svg_bytes=len(svg.encode("utf-8")), table_html_bytes=len(panels.encode("utf-8")))


def _embedded_sizes(data: dict[str, Any]) -> dict[str, int]:
    """Measure escaped JSON payload bytes by named, non-overlapping data block."""
    blocks = {}
    for name, value in data.items():
        if name in ("layers", "report", "priorities", "chokepoints"):
            blocks.update({f"{name}.{key}": len(_safe_script(item).encode("utf-8")) for key, item in value.items()})
        else:
            blocks[name] = len(_safe_script(value).encode("utf-8"))
    return dict(sorted(blocks.items()))


def _drop_display_block(data: dict[str, Any], name: str, size: int) -> None:
    parts = name.split(".")
    parent = data if len(parts) == 1 else data[parts[0]]
    value = parent[parts[-1]]
    if parts[0] == "layers":
        parent[parts[-1]] = _collection([])
        metadata = data["display_metadata"]["geometry_layers"][parts[-1]]
        metadata["budget_omitted_features"] = metadata["kept_features"]
        metadata["kept_features"] = 0
        data["display_metadata"]["omitted_layers"].append(parts[-1])
    else:
        parent[parts[-1]] = [] if isinstance(value, list) else {} if isinstance(value, dict) else MISSING
    note = f"Display block {name} omitted to meet the HTML size limit; original source artifacts remain uncapped in the ZIP."
    data["display_metadata"]["omitted_blocks"].append(dict(name=name, embedded_bytes=size, reason="HTML byte budget"))
    data["warnings"].append(note)


def render_dashboard(model: dict[str, Any], *, caps: dict[str, int] | None = None) -> tuple[str, dict[str, Any]]:
    """Fit a page by dropping largest display blocks; original data never changes.

    Geometry goes through lower feature caps and coordinate rounding first.
    If necessary, tables/evidence blocks can also be omitted with an explicit
    notice. Measurements and the recovery diagnostic are saved in the sidecar.
    """
    data = build_dashboard_data(model, caps=caps)
    data.update(colors=COLORS, map_layer_order=MAP_LAYER_ORDER,
                maplibre_js=f"https://unpkg.com/maplibre-gl@{MAPLIBRE_VERSION}/dist/maplibre-gl.js")
    data["display_metadata"].update(omitted_layers=[], omitted_blocks=[])
    initial_blocks = _embedded_sizes(data)
    html, sizes = _render_page(model, data)
    initial_size = sizes["html_bytes"]
    protected = {"bbox", "pair", "synthetic", "no_flood", "validation_supplied", "colors", "map_layer_order", "maplibre_js", "display_metadata",
                 "report.title", "report.incident", "report.banner", "report.disclaimer", "report.incident_lines", "report.data_gap_note", "report.priorities_note"}
    removed = set()
    while sizes["html_bytes"] > HTML_BUDGET_BYTES:
        blocks = _embedded_sizes(data)
        candidates = [(size, name) for name, size in blocks.items() if name not in protected and name not in removed
                      and size > len(_safe_script(_collection([])).encode())]
        if not candidates:
            # Pathological input prose can itself exceed the ceiling. Keep a
            # small, clearly labelled index rather than losing every artifact.
            data["display_metadata"]["omitted_blocks"].append(dict(name="full dashboard", reason="HTML byte budget"))
            html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>FloodIntel</title><body>'
                    + (f'<p style="background:#a82030;color:white">{BANNER}</p>' if data["synthetic"] else '')
                    + '<h1>Dashboard display omitted to meet the HTML size limit</h1>'
                    + '<p>See dashboard_meta.json for byte measurements and omitted blocks. '
                    'The reports and original JSON/GeoJSON remain in incident_package.zip.</p>'
                    + '<p>Educational prototype. Not operational emergency infrastructure.</p></body></html>')
            sizes = dict(html_bytes=len(html.encode()), embedded_json_bytes=0, static_svg_bytes=0, table_html_bytes=0)
            break
        size, name = max(candidates, key=lambda row: (row[0], row[1]))
        _drop_display_block(data, name, size)
        removed.add(name)
        html, sizes = _render_page(model, data)
    diagnostic = None
    if initial_size > HTML_BUDGET_BYTES:
        breakdown = "; ".join(f"{name}={size} bytes" for name, size in sorted(initial_blocks.items(), key=lambda row: (-row[1], row[0])))
        diagnostic = (f"Dashboard exceeded the HTML byte budget ({initial_size} bytes). Embedded blocks: {breakdown}. "
                      + "Omitted: " + ", ".join(row["name"] for row in data["display_metadata"]["omitted_blocks"])
                      + f". Page written within the limit ({sizes['html_bytes']} bytes).")
    metadata = {**data["display_metadata"], **sizes, "initial_html_bytes": initial_size,
                "initial_embedded_block_bytes": initial_blocks,
                "embedded_block_bytes": _embedded_sizes(data) if sizes["embedded_json_bytes"] else {},
                "budget_diagnostic": diagnostic}
    return html, metadata


def _package(out: Path, model: dict[str, Any], metadata: dict[str, Any]) -> None:
    entries = {name: (out / name).read_bytes() for name in ("index.html", "situation_report.html", "situation_report.md")}
    for role in ("impact", "flood"):
        value = model["paths"].get(role + "_dir")
        if value:
            for path in sorted(Path(value).iterdir()):
                if path.is_file() and path.suffix in (".json", ".geojson"):
                    entries[f"{role}/{path.name}"] = path.read_bytes()
    entries["provenance.json"] = json.dumps(model["incident"]["provenance"], ensure_ascii=False, indent=2, allow_nan=False).encode()
    entries["dashboard_meta.json"] = json.dumps(metadata, indent=2).encode()
    if model["paths"].get("validation_json"):
        entries["validation.json"] = Path(model["paths"]["validation_json"]).read_bytes()
    readme = ("FloodIntel static incident package\n\n" + (BANNER + "\n\n" if model["synthetic"] else "")
              + "index.html: dashboard, inlined data and offline SVG. MapLibre is optional and fetched from a pinned CDN only when opened online. No map tiles or API keys.\n"
              + "situation_report.html: print to PDF; situation_report.md: same situation facts as text.\n"
              + "provenance.json: unchanged provenance block from incident.json. dashboard_meta.json: display simplification and dropped/capped geometry counts.\n"
              + "impact/impact_meta.json: computed counts, flags, parameters, graph quality and source pointer.\n"
              + "impact/damaged_assets.geojson: potentially affected road, bridge and building features.\n"
              + "impact/settlements.geojson: mapped access status and evidence; impact/routes.geojson: baseline and scenario route geometry.\n"
              + "impact/clusters.json: shared access-loss clusters and restoration tests.\n"
              + "impact/chokepoints.json and .geojson: restoration leverage and latent bridge access risk.\n"
              + "impact/priorities.json: source rankings and data gaps; legacy non-attributable rows are hidden in the dashboard.\n"
              + "flood/flood_meta.json: original flood statistics, thresholds and uncertainty. flood/flood_extent.geojson: flood components; flood/debris_candidates.geojson: unconfirmed debris candidates.\n"
              + "Synthetic impact packages have no model flood artifacts; test-box boundaries are approximate display geometry.\n"
              + "TIFFs and PNGs are excluded for bandwidth. Original JSON/GeoJSON are uncapped; dashboard geometry is simplified/capped. Any budget-driven table/evidence omissions are labelled on the page and in dashboard_meta.json.\n"
              + "OpenStreetMap contributors, Open Database License. Retain input license notes.\n" + model["report"]["disclaimer"] + "\n")
    if model["paths"].get("validation_json"):
        readme += "validation.json: supplied post-hoc comparison only; it was not used by the analysis.\n"
    entries["README.txt"] = readme.encode()
    with zipfile.ZipFile(out / "incident_package.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, payload in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)


def build_dashboard(incident: str | Path, impact_dir: str | Path | None = None,
                    flood_dir: str | Path | None = None, out: str | Path | None = None, *,
                    cache_dir: str | Path = "data/cache", caps: dict[str, int] | None = None,
                    validation_json: str | Path | None = None) -> dict[str, Any]:
    """Write only static artifacts; inference, fetching and training never run."""
    model = build_report_model(incident, impact_dir, flood_dir, cache_dir=cache_dir, validation_json=validation_json)
    html, metadata = render_dashboard(model, caps=caps)
    directory = Path(out) if out else Path(model["paths"]["incident_dir"]) / "report"
    # Protect source JSON/GeoJSON from an accidental output path selection.
    if directory.resolve() in {Path(value).resolve() for value in model["paths"].values() if value}:
        raise ValueError("Report output must be separate from incident, impact and flood source directories")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index.html").write_text(html, encoding="utf-8")
    (directory / "situation_report.html").write_text(render_report_html(model), encoding="utf-8")
    (directory / "situation_report.md").write_text(render_report_md(model), encoding="utf-8")
    (directory / "dashboard_meta.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    _package(directory, model, metadata)
    return dict(output_path=str(directory.resolve()), synthetic=model["synthetic"], no_flood=model["no_flood"],
                metadata=metadata, warnings=[*model["notices"], *[f"Display block omitted: {row['name']}" for row in metadata["omitted_blocks"]]])


def main(argv: list[str] | None = None) -> int:
    """Build a report from existing local incident artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incident", required=True)
    parser.add_argument("--impact-dir")
    parser.add_argument("--flood-dir")
    parser.add_argument("--out")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--validation-json", help="Optional local post-hoc comparison JSON; reporting only")
    args = parser.parse_args(argv)
    try:
        result = build_dashboard(args.incident, args.impact_dir, args.flood_dir, args.out, cache_dir=args.cache_dir,
                                 validation_json=args.validation_json)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"Report error: {error}\n")
    if result["metadata"]["budget_diagnostic"]:
        print("Report warning: " + result["metadata"]["budget_diagnostic"], file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
