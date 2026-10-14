"""Offline, post-inference flood-mask agreement with a local reference polygon.

This standalone script never imports inference modules or changes their output.
Legacy top-level numbers use valid flood-grid pixels. Variant comparisons use
areaOfInterestA minus notAnalysedA on that grid. Polygon cell inclusion uses
rasterio's default centre rule (all_touched=False); tolerance is declared, not fitted.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import re
import struct
from typing import Any
import xml.etree.ElementTree as ET
import zlib

# Coordinate transforms must use installed local projection resources only.
os.environ["PROJ_NETWORK"] = "OFF"

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely import make_valid
from shapely.ops import unary_union

EVAL_DIR = Path(__file__).resolve().parent
REFERENCE = EVAL_DIR / "emsr927" / "EMSR927_AOI03_GRA_PRODUCT_observedEventA_v1.shp"
REFERENCE_XML = EVAL_DIR / "emsr927" / "EMSR927_AOI03_GRA_PRODUCT_source_v1.xml"


def _day(value: str) -> str:
    """Accept an ISO date or timestamp and return its calendar date."""
    text = value.strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T.*)?", text):
        raise ValueError(f"Invalid acquisition date: {value!r}; expected YYYY-MM-DD or an ISO timestamp")
    return date.fromisoformat(text[:10]).isoformat()


def _tag(element: ET.Element) -> str:
    return re.sub(r"[^a-z]", "", element.tag.rsplit("}", 1)[-1].lower())


def reference_dates(path: str | Path) -> dict[str, Any]:
    """Extract acquisition fields or temporal extents, never creation/publication dates.

    A metadata date stamp is reported separately. Multiple distinct acquisition
    days stay explicit rather than silently selecting an image from a mosaic.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Reference source XML is missing: {path}")
    root = ET.parse(path).getroot()
    acquired: set[str] = set()
    stamp = None
    acquisition_tags = {"acquisitiondate", "acquisitiondatetime", "acquisitiontime", "imagedate", "imagingdate", "sensingtime"}

    def walk(element: ET.Element, ancestors: tuple[str, ...] = ()) -> None:
        nonlocal stamp
        tag = _tag(element)
        context = (*ancestors, tag)
        text = " ".join(element.itertext()).strip()
        if tag == "datestamp":
            values = re.findall(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)", text)
            stamp = _day(values[0]) if values else None
        acquisition = tag in acquisition_tags
        temporal = tag in {"timeposition", "beginposition", "endposition"} and any(
            value in {"extemporalextent", "temporalelement", "miacquisitioninformation", "mievent"} for value in ancestors)
        event_time = tag in {"time", "datetime"} and "mievent" in ancestors
        if acquisition or temporal or event_time:
            acquired.update(_day(value) for value in re.findall(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)", text))
        for child in element:
            walk(child, context)

    walk(root)
    values = sorted(acquired)
    return dict(acquisition_date=values[0] if len(values) == 1 else None,
                acquisition_dates=values, metadata_date_stamp=stamp)


def _post_date(metadata: dict[str, Any]) -> tuple[str | None, str | None]:
    """Read explicit post-acquisition fields; never infer a day from the flood date."""
    locations = [((key,), metadata.get(key)) for key in ("post_date", "post_image_date", "post_acquisition_date")]
    for container in ("params", "pair_summary"):
        values = metadata.get(container, {})
        locations.extend(((container, key), values.get(key)) for key in ("post_date", "post_image_date", "post_acquisition_date"))
    for container in ("pair", "pair_metadata"):
        post = metadata.get(container, {}).get("post", {})
        if isinstance(post, dict):
            locations.append(((container, "post", "acquisition_date"), post.get("acquisition_date")))
    locations.append((("acquisition_dates", "post"), metadata.get("acquisition_dates", {}).get("post")))
    found = [(".".join(path), _day(value)) for path, value in locations if isinstance(value, str)]
    if len({value for _, value in found}) > 1:
        raise ValueError("Conflicting post-image acquisition dates in flood_meta.json")
    return (found[0][1], f"flood_meta.json:{found[0][0]}") if found else (None, None)


def _polygons(geometry: Any) -> list[Any]:
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [polygon for part in geometry.geoms for polygon in _polygons(part)]
    return []


def _binary(dataset: Any, name: str, domain: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if dataset.descriptions.count(name) != 1:
        raise ValueError(f"Flood raster must contain exactly one '{name}' band description")
    values = dataset.read(dataset.descriptions.index(name) + 1, masked=True)
    finite = ~np.ma.getmaskarray(values) & np.isfinite(values.data)
    included = finite if domain is None else finite & domain
    if not np.isin(values.data[included], (0, 1)).all():
        raise ValueError(f"'{name}' must contain binary values, not probabilities")
    return np.asarray(values.data == 1), finite


def _layer_path(reference: Path, layer: str, explicit: str | Path | None) -> Path | None:
    """Find a local layer for the same reference AOI, never a different AOI."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"{layer} shapefile is missing: {path}")
        return path
    prefix = re.match(r"(.+?_AOI\d+)_", reference.stem, flags=re.IGNORECASE)
    matches = sorted(path for path in reference.parent.glob("*.shp")
                     if layer.lower() in path.stem.lower()
                     and (prefix is None or path.stem.lower().startswith(prefix[1].lower() + "_")))
    if layer == "notAnalysedA":
        gra = [path for path in matches if "_gra_product_" in path.stem.lower()]
        if gra:
            matches = gra  # The grading product's exclusions take precedence over other products.
    if len(matches) > 1:
        raise ValueError(f"Multiple {layer} files; select one explicitly")
    return matches[0] if matches else None


def _read_polygons(path: Path, crs: CRS, warnings: list[str], label: str) -> tuple[list[Any], CRS]:
    """Read the local .prj and repair polygon geometry before reprojection."""
    for suffix in (".prj", ".dbf", ".shx"):
        if not path.with_suffix(suffix).is_file():
            raise FileNotFoundError(f"{label} shapefile companion is missing: {path.with_suffix(suffix)}")
    native_crs = CRS.from_wkt(path.with_suffix(".prj").read_text(encoding="utf-8"))
    frame = gpd.read_file(path).set_crs(native_crs, allow_override=True)
    geometries, repaired, dropped = [], 0, 0
    for geometry in frame.geometry:
        if geometry is not None and not geometry.is_valid:
            geometry = make_valid(geometry)
            repaired += 1
        parts = _polygons(geometry)
        geometries.extend(parts)
        dropped += not bool(parts)
    if repaired:
        warnings.append(f"Repaired {repaired} invalid {label} geometries before rasterization.")
    if dropped:
        warnings.append(f"Ignored {dropped} empty or non-polygon {label} features.")
    projected = gpd.GeoSeries(geometries, crs=native_crs).to_crs(crs)
    return list(projected), native_crs


def reference_event_types(path: Path) -> list[str]:
    """Retain unique event_type and obj_desc labels; never infer a hazard category."""
    frame = gpd.read_file(path)
    columns = {name.lower(): name for name in frame.columns}
    values = set()
    for key in ("event_type", "obj_desc"):
        if key in columns:
            values.update(str(value).strip() for value in frame[columns[key]].dropna() if str(value).strip())
    return sorted(values)


def _raster_mask(polygons: list[Any], dataset: Any) -> np.ndarray:
    return (rasterize([(polygon, 1) for polygon in polygons], out_shape=(dataset.height, dataset.width),
                      transform=dataset.transform, fill=0, dtype="uint8", all_touched=False).astype(bool)
            if polygons else np.zeros((dataset.height, dataset.width), dtype=bool))


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _strict(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray, pixel_area: float) -> dict[str, Any]:
    tp = int(np.count_nonzero(valid & prediction & reference))
    fp = int(np.count_nonzero(valid & prediction & ~reference))
    fn = int(np.count_nonzero(valid & ~prediction & reference))
    tn = int(np.count_nonzero(valid & ~prediction & ~reference))
    return dict(counts=dict(TP=tp, FP=fp, FN=fn, TN=tn, valid_pixels=int(valid.sum()),
                            ignored_pixels=int(valid.size - valid.sum())),
                metrics=dict(iou=_divide(tp, tp + fp + fn), precision=_divide(tp, tp + fp),
                             recall=_divide(tp, tp + fn), f1=_divide(2 * tp, 2 * tp + fp + fn)),
                areas_km2=dict(ours=(tp + fp) * pixel_area / 1e6,
                               reference=(tp + fn) * pixel_area / 1e6, overlap=tp * pixel_area / 1e6))


def _within(mask: np.ndarray, sampling: tuple[float, float], tolerance_m: float) -> np.ndarray:
    """Euclidean distance between pixel centres; an empty target has no neighbours."""
    return (distance_transform_edt(~mask, sampling=sampling) <= tolerance_m
            if mask.any() else np.zeros_like(mask))


def _png(path: Path, prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> None:
    """Lossless RGBA overlay at the original grid size; invalid pixels transparent."""
    rgba = np.empty((*valid.shape, 4), dtype=np.uint8)
    rgba[..., :3] = (14, 24, 35)
    rgba[..., 3] = valid.astype(np.uint8) * 255
    rgba[prediction & valid, :3] = (40, 130, 255)
    rgba[reference & valid, :3] = (245, 151, 41)
    rgba[prediction & reference & valid, :3] = (255, 255, 255)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack("!I", len(payload)) + kind + payload + struct.pack("!I", zlib.crc32(kind + payload))

    height, width = valid.shape
    header = struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)
    raw = b"".join(b"\x00" + row.tobytes() for row in rgba)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _markdown(summary: dict[str, Any]) -> str:
    """Render the same counts, metric values and dates stored in the JSON."""
    counts, metrics, areas = summary["counts"], summary["metrics"], summary["areas_km2"]
    lines = ["# Flood polygon agreement", "", f"Incident: {summary['incident_id']}", "",
             f"Reference acquisition date: {summary['reference_acquisition_date'] or 'not available'}",
             f"Reference metadata date stamp (not acquisition): {summary['reference_metadata_date_stamp'] or 'not available'}",
             f"Our post-image acquisition date: {summary['our_post_image_date'] or 'not available'}",
             f"Absolute date gap in days: {summary['date_gap_days'] if summary['date_gap_days'] is not None else 'not available'}", "",
             f"Grid: {summary['grid']['height']} rows × {summary['grid']['width']} columns; {summary['grid']['crs']}.",
             f"Grid transform: {json.dumps(summary['grid']['transform'])}",
             f"Pixel area: {summary['grid']['pixel_area_m2']} m²; reference polygons used: {summary['reference_polygon_count']}.",
             f"Valid pixels: {counts['valid_pixels']}; ignored pixels: {counts['ignored_pixels']}.",
             f"TP: {counts['TP']}; FP: {counts['FP']}; FN: {counts['FN']}; TN: {counts['TN']}.", "",
             *[f"{label}: {metrics[key]}" for key, label in (("iou", "IoU"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))], "",
             f"Our area: {areas['ours']} km²; reference area: {areas['reference']} km²; overlap: {areas['overlap']} km².", "",
             "The numbers above retain the legacy full-valid-grid flood-only domain.",
             f"Analysed polygon domain: {summary['domain']['area_km2']} km²; valid analysed grid: {summary['domain']['valid_area_km2']} km².",
             f"Share of our valid flood area outside the analysed domain: {summary['domain']['our_flood_outside_share']}.",
             f"Declared pixel-centre tolerance: {summary['params']['tolerance_m']} m (not fitted to this reference).", "",
             "All variants below use the same analysed, valid domain. Debris is an unconfirmed candidate, not a water classification.", "",
             "| Variant | TP | FP | FN | TN | IoU | Precision | Recall | F1 | Ours km² | Reference km² | Overlap km² | Precision tol | Recall tol |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
             *["| " + " | ".join(str(value) for value in [name,
                *[variant['counts'][key] for key in ('TP', 'FP', 'FN', 'TN')],
                *[variant['metrics'][key] for key in ('iou', 'precision', 'recall', 'f1')],
                *[variant['areas_km2'][key] for key in ('ours', 'reference', 'overlap')],
                variant['tolerance_metrics']['precision_tol'], variant['tolerance_metrics']['recall_tol']]) + " |"
               for name, variant in summary['variants'].items()], "",
             "Areas use valid grid cells only. Zero metric denominators return zero.",
             "Overlay: ours blue, reference orange, overlap white; neither dark, ignored pixels transparent.",
             summary["agreement_note"], "", "Warnings:", *[f"- {warning}" for warning in summary["warnings"]]]
    return "\n".join(lines) + "\n"


def compare_flood(flood: str | Path, reference: str | Path = REFERENCE, reference_xml: str | Path = REFERENCE_XML, *,
                  incident_id: str, results_root: str | Path = EVAL_DIR / "results",
                  post_date: str | None = None, reference_date: str | None = None,
                  aoi: str | Path | None = None, not_analysed: str | Path | None = None,
                  tolerance_m: float = 50.) -> dict[str, Any]:
    """Compare local files, then write exclusively below eval/results/<incident_id>.

    Explicit date overrides are recorded as operator input. Missing acquisition
    metadata remains unknown; a product creation date is not an image date.
    Legacy top-level counts retain the full valid grid. Variants use the same
    AOI-minus-exclusions domain and pixel-centre distance tolerance, never tuned.
    """
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError("tolerance_m must be finite and nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", incident_id):
        raise ValueError("incident_id must be a single safe folder name")
    root = Path(results_root).resolve()
    if root.name != "results" or root.parent.name != "eval":
        raise ValueError("Comparison results must be written under eval/results/")
    output = (root / incident_id).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Output folder must remain inside eval/results/")
    paths = [output / name for name in ("comparison.json", "comparison.md", "overlay.png")]
    if any(path.is_symlink() for path in paths):
        raise ValueError("Comparison output files must not be symlinks")
    flood, reference = Path(flood), Path(reference)
    if flood.is_dir():
        flood /= "flood.tif"
    if not flood.is_file():
        raise FileNotFoundError(f"Our flood raster is missing: {flood}")
    if not reference.is_file():
        raise FileNotFoundError(f"Reference shapefile is missing: {reference}")
    prj = reference.with_suffix(".prj")
    if not prj.is_file():
        raise FileNotFoundError(f"Reference CRS .prj file is missing: {prj}")
    for suffix in (".dbf", ".shx"):
        if not reference.with_suffix(suffix).is_file():
            raise FileNotFoundError(f"Reference shapefile companion is missing: {reference.with_suffix(suffix)}")
    metadata_path = flood.with_name("flood_meta.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Flood metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded_incident = metadata.get("params", {}).get("incident_id")
    if recorded_incident and recorded_incident != incident_id:
        raise ValueError("Flood metadata belongs to a different incident")
    dates = reference_dates(reference_xml)
    reference_acquired = dates["acquisition_date"]
    reference_date_source = "source XML acquisition/temporal field" if reference_acquired else None
    ours_date, ours_date_source = _post_date(metadata)
    warnings: list[str] = []
    if reference_date is not None:
        reference_acquired, reference_date_source = _day(reference_date), "explicit --reference-date override"
        warnings.append("Reference acquisition date supplied explicitly; it is not inferred from the source XML.")
    if post_date is not None:
        ours_date, ours_date_source = _day(post_date), "explicit --post-date override"
        warnings.append("Our post-image date supplied explicitly; it is not inferred from flood metadata.")
    if not reference_acquired:
        warnings.append("Reference image acquisition date is absent or ambiguous in the XML; its metadata creation/date stamp is not used as acquisition.")
    if not ours_date:
        warnings.append("Post-image acquisition date is not recorded in flood_meta.json; the acquisition-date gap cannot be checked.")
    gap = abs((date.fromisoformat(reference_acquired) - date.fromisoformat(ours_date)).days) if reference_acquired and ours_date else None
    if gap is not None and gap > 3:
        warnings.append(f"Reference and post-image acquisition dates differ by {gap} days, more than 3 days; water extent may have changed.")
    with rasterio.open(flood) as dataset:
        if dataset.crs is None or not dataset.crs.is_projected:
            raise ValueError("Flood grid must have a projected CRS to compute km² areas")
        if ([dataset.height, dataset.width] != metadata.get("shape") or dataset.crs != CRS.from_user_input(metadata.get("crs"))
                or len(metadata.get("transform", [])) != 6
                or not np.allclose(list(dataset.transform)[:6], metadata["transform"], rtol=0, atol=1e-9)):
            raise ValueError("Flood raster grid disagrees with flood_meta.json")
        if "valid_mask" in dataset.descriptions:
            saved_valid, finite_valid = _binary(dataset, "valid_mask")
            valid = saved_valid & finite_valid
        else:
            valid = np.ones((dataset.height, dataset.width), dtype=bool)
            warnings.append("valid_mask band missing; using finite, non-nodata flood_mask pixels only.")
        prediction, finite_prediction = _binary(dataset, "flood_mask", valid)
        valid &= finite_prediction
        debris, finite_debris = _binary(dataset, "debris_candidate", valid)
        metres_per_unit = dataset.crs.linear_units_factor[1]
        pixel_area = abs(dataset.transform.a * dataset.transform.e - dataset.transform.b * dataset.transform.d) * metres_per_unit ** 2
        grid = dict(height=dataset.height, width=dataset.width, crs=dataset.crs.to_string(),
                    transform=list(dataset.transform)[:6], pixel_area_m2=float(pixel_area))
        polygons, reference_crs = _read_polygons(reference, dataset.crs, warnings, "reference")
        reference_mask = _raster_mask(polygons, dataset)
        aoi_path = _layer_path(reference, "areaOfInterestA", aoi)
        if aoi_path is None:
            raise FileNotFoundError("Required areaOfInterestA polygon is missing; add the local AOI shapefile or pass --aoi. The observed extent is not an evaluation domain.")
        aoi_polygons, _ = _read_polygons(aoi_path, dataset.crs, warnings, "areaOfInterestA")
        exclusion_path = _layer_path(reference, "notAnalysedA", not_analysed)
        exclusions = []
        if exclusion_path is None:
            warnings.append("notAnalysedA layer is absent; unanalysed exclusions are unknown, and only the supplied areaOfInterestA bounds the domain.")
        else:
            exclusions, _ = _read_polygons(exclusion_path, dataset.crs, warnings, "notAnalysedA")
        analysed_geometry = unary_union(aoi_polygons).difference(unary_union(exclusions))
        domain_mask = _raster_mask(_polygons(analysed_geometry), dataset)
        analysed_valid = valid & finite_debris & domain_mask
        if np.any(valid & domain_mask & ~finite_debris):
            warnings.append("Non-finite debris pixels excluded from the common domain of all variants; legacy flood-only numbers are unchanged.")
        if not analysed_valid.any():
            warnings.append("No valid pixels inside the analysed domain; variant metrics are zero and cannot establish agreement.")
        transform = dataset.transform
        column = np.array([transform.a, transform.d]) * metres_per_unit
        row = np.array([transform.b, transform.e]) * metres_per_unit
        if not np.isclose(np.dot(column, row), 0., atol=1e-8):
            raise ValueError("Pixel-centre tolerance requires an orthogonal raster grid, not a sheared transform")
        sampling = (float(np.linalg.norm(row)), float(np.linalg.norm(column)))
        within_reference = _within(reference_mask & analysed_valid, sampling, tolerance_m)
        variants = {}
        for name, mask in (("flood_mask", prediction), ("debris_candidate", debris), ("union", prediction | debris)):
            variant = _strict(mask, reference_mask, analysed_valid, pixel_area)
            predicted = mask & analysed_valid
            actual = reference_mask & analysed_valid
            precision_matched = int((predicted & within_reference).sum())
            recall_matched = int((actual & _within(predicted, sampling, tolerance_m)).sum())
            variant['tolerance_metrics'] = dict(precision_tol=_divide(precision_matched, int(predicted.sum())),
                                               recall_tol=_divide(recall_matched, int(actual.sum())),
                                               predicted_matched_pixels=precision_matched, reference_matched_pixels=recall_matched)
            variant['params'] = dict(tolerance_m=tolerance_m, domain="areaOfInterestA minus notAnalysedA, common valid grid pixels",
                                     distance="Euclidean pixel-centre distance", all_touched=False)
            variants[name] = variant
        outside = int((prediction & valid & ~domain_mask).sum())
        domain = dict(area_km2=float(analysed_geometry.area * metres_per_unit ** 2 / 1e6),
                      on_grid_area_km2=float(domain_mask.sum() * pixel_area / 1e6),
                      valid_area_km2=float(analysed_valid.sum() * pixel_area / 1e6),
                      our_flood_outside_area_km2=outside * pixel_area / 1e6,
                      our_flood_outside_share=_divide(outside, int((prediction & valid).sum())),
                      not_analysed_present=exclusion_path is not None)
    if not np.any(reference_mask & valid):
        warnings.append("Empty reference on valid grid pixels; IoU is zero and agreement cannot establish missing-reference completeness.")
    if not valid.any():
        warnings.append("No valid flood pixels; all counts and areas are zero and agreement metrics are unavailable in practice.")
    summary = dict(incident_id=incident_id, **_strict(prediction, reference_mask, valid, pixel_area),
                   variants=variants, domain=domain,
                   reference_acquisition_date=reference_acquired, reference_acquisition_dates=dates["acquisition_dates"],
                   reference_metadata_date_stamp=dates["metadata_date_stamp"], reference_date_source=reference_date_source,
                   our_post_image_date=ours_date, our_post_date_source=ours_date_source, date_gap_days=gap,
                   reference_polygon_count=len(polygons), grid=grid, warnings=warnings,
                   reference_event_types=reference_event_types(reference),
                   params=dict(all_touched=False, date_gap_warning_days=3, area_domain="valid flood-grid pixels (legacy top-level only)",
                               tolerance_m=tolerance_m, zero_denominator=0),
                   inputs=dict(flood=str(flood.resolve()), flood_metadata=str(metadata_path.resolve()), reference=str(reference.resolve()),
                               reference_xml=str(Path(reference_xml).resolve()), reference_crs=reference_crs.to_string(), flood_params=metadata.get("params", {}),
                               area_of_interest=str(aoi_path.resolve()), not_analysed=str(exclusion_path.resolve()) if exclusion_path else None),
                   output_dir=str(output), agreement_note="Pixel agreement with this reference is not a claim of model accuracy. Variant comparisons exclude pixels outside the supplied analysed domain. Legacy top-level numbers retain the full valid grid; survey completeness is not established.")
    output.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    paths[1].write_text(_markdown(summary), encoding="utf-8")
    _png(paths[2], prediction, reference_mask, valid)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incident", required=True)
    parser.add_argument("--flood", required=True, help="Path to flood.tif or its containing directory")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--reference-xml", type=Path, default=REFERENCE_XML)
    parser.add_argument("--post-date", help="Explicit acquisition date if not stored in flood_meta.json")
    parser.add_argument("--reference-date", help="Explicit acquisition date if not recorded in the source XML")
    parser.add_argument("--aoi", type=Path, help="Local areaOfInterestA shapefile; otherwise discover beside the reference")
    parser.add_argument("--not-analysed", type=Path, help="Local notAnalysedA shapefile; otherwise discover beside the reference")
    parser.add_argument("--tolerance-m", type=float, default=50., help="Declared pixel-centre tolerance in meters; never fitted to the reference")
    args = parser.parse_args(argv)
    try:
        summary = compare_flood(args.flood, args.reference, args.reference_xml, incident_id=args.incident,
                                post_date=args.post_date, reference_date=args.reference_date,
                                aoi=args.aoi, not_analysed=args.not_analysed, tolerance_m=args.tolerance_m)
    except (OSError, ValueError, ET.ParseError) as error:
        parser.exit(2, f"Comparison error: {error}\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
