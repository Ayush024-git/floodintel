"""Offline comparison with tiny rasters and locally written reference shapefiles."""
import hashlib
import json
from pathlib import Path
import struct
import zlib

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import unary_union

from eval import compare_emsr927 as compare


def inputs(tmp_path, *, reference_crs="EPSG:3857", empty=False, valid_band=True):
    directory = tmp_path / "inference" / "flood"
    directory.mkdir(parents=True)
    prediction = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], np.float32)
    valid = np.ones((3, 3), np.float32)
    valid[2, 2] = 0
    transform = from_origin(1000, 2000, 10, 10)
    flood = directory / "flood.tif"
    # Include unrelated probability bands so the reader must use descriptions.
    array = np.stack([np.full((3, 3), 0.25, np.float32), prediction, valid]) if valid_band else np.stack([np.full((3, 3), 0.25, np.float32), prediction])
    array = np.concatenate([array, np.zeros((1, 3, 3), np.float32)])
    with rasterio.open(flood, "w", driver="GTiff", height=3, width=3, count=len(array), dtype="float32", crs="EPSG:3857", transform=transform, nodata=np.nan) as raster:
        raster.write(array)
        raster.descriptions = (("p_post", "flood_mask", "valid_mask") if valid_band else ("p_post", "flood_mask")) + ("debris_candidate",)
    metadata = dict(shape=[3, 3], transform=list(transform)[:6], crs="EPSG:3857", resolution=10,
                    params=dict(incident_id="inc_synthetic", post_date="2026-08-12", threshold=0.6))
    (directory / "flood_meta.json").write_text(json.dumps(metadata))
    reference = tmp_path / "reference" / "observed.shp"
    reference.parent.mkdir()
    geometries = [] if empty else [unary_union([box(1000 + column * 10, 2000 - (row + 1) * 10,
                                                     1000 + (column + 1) * 10, 2000 - row * 10)
                                              for row, column in ((0, 0), (1, 0), (1, 1), (2, 2))])]
    frame = gpd.GeoDataFrame({"id": list(range(len(geometries)))}, geometry=geometries, crs="EPSG:3857").to_crs(reference_crs)
    frame.to_file(reference, driver="ESRI Shapefile", geometry_type="Polygon" if empty else "MultiPolygon")
    gpd.GeoDataFrame({"id": [1]}, geometry=[box(1000, 1970, 1030, 2000)], crs="EPSG:3857").to_crs(reference_crs).to_file(
        reference.parent / "PRODUCT_areaOfInterestA_v1.shp", driver="ESRI Shapefile")
    xml = reference.parent / "source.xml"
    xml.write_text('<gmd:MD_Metadata xmlns:gmd="http://example.invalid/gmd" xmlns:gco="http://example.invalid/gco">'
                   '<gmd:dateStamp><gco:Date>2026-08-29</gco:Date></gmd:dateStamp>'
                   '<gmd:acquisitionDate><gco:DateTime>2026-08-11T12:30:00Z</gco:DateTime></gmd:acquisitionDate>'
                   '</gmd:MD_Metadata>')
    root = tmp_path / "eval" / "results"
    return flood, reference, xml, root


def run(paths, **kwargs):
    flood, reference, xml, root = paths
    return compare.compare_flood(flood, reference, xml, incident_id="inc_synthetic", results_root=root, **kwargs)


def test_counts_metrics_areas_ignore_and_overlay(tmp_path):
    paths = inputs(tmp_path)
    summary = run(paths)
    assert summary["counts"] == dict(TP=2, FP=1, FN=1, TN=4, valid_pixels=8, ignored_pixels=1)
    assert summary["metrics"]["iou"] == 0.5
    assert summary["metrics"]["precision"] == pytest.approx(2 / 3)
    assert summary["metrics"]["recall"] == pytest.approx(2 / 3)
    assert summary["metrics"]["f1"] == pytest.approx(2 / 3)
    assert summary["areas_km2"] == dict(ours=0.0003, reference=0.0003, overlap=0.0002)
    assert summary["reference_acquisition_date"] == "2026-08-11"
    assert summary["our_post_image_date"] == "2026-08-12"
    assert summary["date_gap_days"] == 1
    overlay = Path(summary["output_dir"]) / "overlay.png"
    png = overlay.read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack("!II", png[16:24])
    compressed, cursor = [], 8
    while cursor < len(png):
        length = struct.unpack("!I", png[cursor:cursor+4])[0]
        if png[cursor+4:cursor+8] == b"IDAT":
            compressed.append(png[cursor+8:cursor+8+length])
        cursor += 12 + length
    scanlines = np.frombuffer(zlib.decompress(b"".join(compressed)), dtype=np.uint8).reshape(height, 1 + width * 4)
    assert (scanlines[:, 0] == 0).all()
    rgba = scanlines[:, 1:].reshape(height, width, 4)
    assert rgba.shape == (3, 3, 4)
    assert list(rgba[0, 0, :3]) == [255, 255, 255]
    assert list(rgba[0, 1, :3]) == [40, 130, 255]
    assert list(rgba[1, 1, :3]) == [245, 151, 41]
    assert rgba[2, 2, 3] == 0


def test_reprojected_reference_matches_same_pixels(tmp_path):
    first = tmp_path / "same"
    second = tmp_path / "different"
    same = run(inputs(first))
    projected = run(inputs(second, reference_crs="EPSG:4326"))
    assert same["counts"] == projected["counts"]
    assert same["metrics"] == projected["metrics"]
    assert same["areas_km2"] == projected["areas_km2"]
    assert projected["inputs"]["reference_crs"] == "EPSG:4326"


def test_missing_reference_is_clear_and_writes_nothing(tmp_path):
    flood, _, xml, root = inputs(tmp_path)
    with pytest.raises(FileNotFoundError, match="Reference shapefile is missing"):
        compare.compare_flood(flood, tmp_path / "missing.shp", xml, incident_id="inc_synthetic", results_root=root)
    assert not root.exists()


def test_empty_reference_zero_iou_and_warning(tmp_path):
    summary = run(inputs(tmp_path, empty=True))
    assert summary["metrics"]["iou"] == 0
    assert summary["counts"]["TP"] == 0
    assert summary["counts"]["FP"] == 3
    assert summary["reference_polygon_count"] == 0
    assert summary["areas_km2"]["reference"] == 0
    assert any("Empty reference" in warning for warning in summary["warnings"])


def test_files_only_under_results_and_inputs_unchanged(tmp_path):
    paths = inputs(tmp_path)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.rglob("*") if path.is_file()}
    summary = run(paths)
    root = paths[-1]
    new_files = {path for path in tmp_path.rglob("*") if path.is_file()} - set(before)
    assert {path.name for path in new_files} == {"comparison.json", "comparison.md", "overlay.png"}
    assert all(path.is_relative_to(root / "inc_synthetic") for path in new_files)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items())
    output = Path(summary["output_dir"])
    assert json.loads((output / "comparison.json").read_text()) == summary
    md = (output / "comparison.md").read_text()
    for value in summary["metrics"].values():
        assert str(value) in md
    for value in summary["areas_km2"].values():
        assert str(value) in md
    assert not any(word in md.lower() for word in ("casualt", "population", "severity"))
    with pytest.raises(ValueError, match="eval/results"):
        compare.compare_flood(*paths[:3], incident_id="inc_synthetic", results_root=tmp_path / "inference")
    with pytest.raises(ValueError, match="safe folder"):
        compare.compare_flood(*paths[:3], incident_id="../escape", results_root=root)


def test_date_gap_and_missing_dates_never_use_creation(tmp_path):
    paths = inputs(tmp_path)
    summary = run(paths, reference_date="2026-08-20")
    assert summary["date_gap_days"] == 8
    assert any("more than 3 days" in warning for warning in summary["warnings"])
    flood, _, xml, _ = paths
    xml.write_text('<Metadata><dateStamp>2026-08-29</dateStamp><CI_Date><date>2026-08-29</date>'
                   '<CI_DateTypeCode codeListValue="creation">Creation</CI_DateTypeCode></CI_Date></Metadata>')
    metadata = json.loads(flood.with_name("flood_meta.json").read_text())
    metadata["params"].pop("post_date")
    flood.with_name("flood_meta.json").write_text(json.dumps(metadata))
    summary = run(paths)
    assert summary["reference_acquisition_date"] is None
    assert summary["reference_metadata_date_stamp"] == "2026-08-29"
    assert summary["our_post_image_date"] is None
    assert summary["date_gap_days"] is None
    assert any("acquisition date is absent" in warning for warning in summary["warnings"])
    assert any("not recorded" in warning for warning in summary["warnings"])


def test_temporal_extent_and_multiple_reference_dates(tmp_path):
    xml = tmp_path / "source.xml"
    xml.write_text('<Metadata><EX_TemporalExtent><TimePeriod><beginPosition>2026-08-12T00:00:00Z</beginPosition>'
                   '<endPosition>2026-08-12T23:00:00Z</endPosition></TimePeriod></EX_TemporalExtent></Metadata>')
    assert compare.reference_dates(xml)["acquisition_date"] == "2026-08-12"
    xml.write_text('<Metadata><acquisitionDate>2026-08-12</acquisitionDate><acquisitionDate>2026-08-14</acquisitionDate></Metadata>')
    assert compare.reference_dates(xml)["acquisition_date"] is None
    assert compare.reference_dates(xml)["acquisition_dates"] == ["2026-08-12", "2026-08-14"]


def test_missing_valid_band_explicit_warning(tmp_path):
    summary = run(inputs(tmp_path, valid_band=False))
    assert summary["counts"]["valid_pixels"] == 9
    assert summary["counts"]["TP"] == 3
    assert any("valid_mask band missing" in warning for warning in summary["warnings"])


def test_cli_prints_json_with_synthetic_inputs(tmp_path, monkeypatch, capsys):
    paths = inputs(tmp_path)
    actual = compare.compare_flood
    monkeypatch.setattr(compare, "compare_flood", lambda *args, **kwargs: actual(*args, **kwargs, results_root=paths[-1]))
    assert compare.main(["--incident", "inc_synthetic", "--flood", str(paths[0]), "--reference", str(paths[1]), "--reference-xml", str(paths[2])]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["reference_acquisition_date"] == "2026-08-11"
    assert printed["metrics"]["iou"] == 0.5


def test_nodata_and_outside_valid_pixels_are_ignored(tmp_path):
    paths = inputs(tmp_path)
    with rasterio.open(paths[0], "r+") as raster:
        prediction = raster.read(2)
        prediction[0, 0] = np.nan
        prediction[2, 2] = 99  # irrelevant because saved valid_mask is false here
        raster.write(prediction, 2)
    summary = run(paths)
    assert summary["counts"] == dict(TP=1, FP=1, FN=1, TN=4, valid_pixels=7, ignored_pixels=2)


def test_grid_mismatch_and_missing_crs_fail_before_writing(tmp_path):
    paths = inputs(tmp_path)
    metadata_path = paths[0].with_name("flood_meta.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["transform"][2] += 10
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="grid disagrees"):
        run(paths)
    assert not paths[-1].exists()
    paths[1].with_suffix(".prj").unlink()
    with pytest.raises(FileNotFoundError, match="CRS .prj"):
        run(paths)


def write_domain(paths, *, aoi=None, excluded=None, crs="EPSG:3857"):
    if aoi is not None:
        gpd.GeoDataFrame({"id": [1]}, geometry=[aoi], crs="EPSG:3857").to_crs(crs).to_file(
            paths[1].parent / "PRODUCT_areaOfInterestA_v1.shp", driver="ESRI Shapefile")
    if excluded is not None:
        gpd.GeoDataFrame({"id": [1]}, geometry=[excluded], crs="EPSG:3857").to_crs(crs).to_file(
            paths[1].parent / "PRODUCT_notAnalysedA_v1.shp", driver="ESRI Shapefile")


def test_analysed_domain_clips_aoi_and_excludes_unanalysed_preserving_legacy(tmp_path):
    paths = inputs(tmp_path)
    baseline = run(paths)
    write_domain(paths, aoi=box(1000, 1970, 1020, 2000), excluded=box(1000, 1990, 1010, 2000), crs="EPSG:4326")
    summary = run(paths)
    for key in ("counts", "metrics", "areas_km2"):
        assert summary[key] == baseline[key]
    assert summary["domain"]["area_km2"] == pytest.approx(0.0005)
    assert summary["domain"]["valid_area_km2"] == pytest.approx(0.0005)
    assert summary["domain"]["our_flood_outside_share"] == pytest.approx(1 / 3)
    assert summary["domain"]["our_flood_outside_area_km2"] == 0.0001
    assert summary["domain"]["not_analysed_present"] is True
    assert summary["variants"]["flood_mask"]["counts"] == dict(TP=1, FP=1, FN=1, TN=2, valid_pixels=5, ignored_pixels=4)


def test_union_and_tolerance_pixel_centre_math(tmp_path):
    paths = inputs(tmp_path)
    with rasterio.open(paths[0], "r+") as raster:
        debris = np.zeros((3, 3), np.float32)
        debris[0, 1] = debris[1, 2] = 1
        raster.write(debris, raster.descriptions.index("debris_candidate") + 1)
    summary = run(paths, tolerance_m=10)
    assert summary["params"]["tolerance_m"] == 10
    flood, debris, union = (summary["variants"][key] for key in ("flood_mask", "debris_candidate", "union"))
    assert debris["counts"]["TP"] == 0 and debris["counts"]["FP"] == 2
    assert union["counts"]["TP"] == 2 and union["counts"]["FP"] == 2
    assert flood["tolerance_metrics"]["precision_tol"] == 1
    assert flood["tolerance_metrics"]["recall_tol"] == 1
    assert debris["tolerance_metrics"]["precision_tol"] == 1
    assert debris["tolerance_metrics"]["recall_tol"] == pytest.approx(2 / 3)
    assert union["tolerance_metrics"]["precision_tol"] == 1
    assert union["tolerance_metrics"]["recall_tol"] == 1
    exact = run(paths, tolerance_m=0)
    for variant in exact["variants"].values():
        assert variant["tolerance_metrics"]["precision_tol"] == variant["metrics"]["precision"]
        assert variant["tolerance_metrics"]["recall_tol"] == variant["metrics"]["recall"]
    assert run(paths, tolerance_m=9.9)["variants"]["debris_candidate"]["tolerance_metrics"]["precision_tol"] == 0


def test_tolerance_cannot_use_predictions_or_reference_outside_domain(tmp_path):
    paths = inputs(tmp_path)
    write_domain(paths, aoi=box(1010, 1990, 1020, 2000))  # prediction only; reference is outside
    variant = run(paths, tolerance_m=50)["variants"]["flood_mask"]
    assert variant["counts"]["FP"] == 1
    assert variant["tolerance_metrics"]["precision_tol"] == 0
    assert variant["tolerance_metrics"]["recall_tol"] == 0


def test_all_variants_in_markdown_no_winner_and_missing_exclusions_warning(tmp_path):
    summary = run(inputs(tmp_path))
    assert list(summary["variants"]) == ["flood_mask", "debris_candidate", "union"]
    assert any("notAnalysedA layer is absent" in warning for warning in summary["warnings"])
    md = (Path(summary["output_dir"]) / "comparison.md").read_text()
    assert "| Variant | TP | FP | FN | TN |" in md
    for name in summary["variants"]:
        assert f"| {name} |" in md
    assert "best" not in md.lower()
    assert "best" not in json.dumps(summary).lower()


def test_missing_aoi_never_uses_extent_as_domain_and_bad_tolerance_rejected(tmp_path):
    paths = inputs(tmp_path)
    paths[1].parent.joinpath("PRODUCT_areaOfInterestA_v1.shp").unlink()
    with pytest.raises(FileNotFoundError, match="Required areaOfInterestA"):
        run(paths)
    assert not paths[-1].exists()
    for tolerance in (-1, np.nan, np.inf):
        with pytest.raises(ValueError, match="tolerance_m"):
            run(paths, tolerance_m=tolerance)


def test_empty_domain_and_debris_nodata_are_explicit(tmp_path):
    paths = inputs(tmp_path)
    with rasterio.open(paths[0], "r+") as raster:
        debris = raster.read(4)
        debris[0, 0] = np.nan
        raster.write(debris, 4)
    summary = run(paths)
    assert summary["counts"]["valid_pixels"] == 8
    assert all(row["counts"]["valid_pixels"] == 7 for row in summary["variants"].values())
    assert any("Non-finite debris" in warning for warning in summary["warnings"])
    write_domain(paths, excluded=box(1000, 1970, 1030, 2000))
    summary = run(paths)
    assert summary["domain"]["area_km2"] == 0
    assert all(row["counts"]["valid_pixels"] == 0 for row in summary["variants"].values())
    assert any("No valid pixels inside" in warning for warning in summary["warnings"])


def test_reference_event_types_retains_unique_attribute_labels(tmp_path):
    paths = inputs(tmp_path)
    frame = gpd.read_file(paths[1])
    geometry = frame.geometry.iloc[0]
    gpd.GeoDataFrame(dict(event_type=["Flood", "Debris flow", "Flood", None],
                         obj_desc=["Observed event", "Debris flow", "", None]),
                     geometry=[geometry] * 4, crs=frame.crs).to_file(paths[1], driver="ESRI Shapefile")
    summary = run(paths)
    assert summary["reference_event_types"] == ["Debris flow", "Flood", "Observed event"]
    assert json.loads((Path(summary["output_dir"]) / "comparison.json").read_text())["reference_event_types"] == summary["reference_event_types"]


def test_absent_reference_event_attributes_are_not_inferred(tmp_path):
    assert run(inputs(tmp_path))["reference_event_types"] == []
