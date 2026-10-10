"""Offline orchestration tests using synthetic cached stage outputs."""

from collections import Counter
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import requests
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.data import incident as mod
from src.data.aoi import build_aoi
from src.data.s1_search import PairSearchResult, ScenePair, S1Pass
from src.data.s1_preprocess import S1Stack, save_stack, load_stack
from src.data.terrain import TerrainStack, save_terrain, load_terrain
from src.data.osm import LAYER_COLUMNS, OSMBundle, save_osm, load_osm


BBOX = (85, 27.9, 85.001, 27.901)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Incident tests must not contact the network")
    monkeypatch.setattr(requests.Session, "request", forbidden)


@pytest.fixture
def fake_stages(monkeypatch, tmp_path):
    state = SimpleNamespace(
        calls=Counter(), errors={}, blocked=False, warnings={}, days_after=2,
        valid_fraction=1.0, s1_source="rtc", terrain_source="cop-dem-glo-30",
        hand_fraction=0.0, density=1.0, hospitals=1, settlements=1, stale=0.0,
        verified=True, widened=False, resolution=10, filter_window=5,
        actual_buffer=5, osm_buildings=[],
    )

    def enter(name):
        state.calls[name] += 1
        if name in state.errors:
            raise state.errors[name]

    def aoi_stage(place=None, bbox=None, flood_date=..., **kwargs):
        enter("aoi")
        state.aoi = build_aoi(bbox=bbox or BBOX, flood_date=flood_date)
        if place:
            state.aoi = replace(state.aoi, name=place)
        return replace(state.aoi, warnings=list(state.warnings.get("aoi", [])))

    def search_stage(aoi):
        enter("s1_search")
        if state.blocked:
            return PairSearchResult(None, [], "no_same_orbit_pair", "Different orbits; cannot analyze this area/date.", 1, 1, [])
        passes = []
        for role, days in (("pre", -10), ("post", state.days_after)):
            day = aoi.flood_date + timedelta(days=days)
            passes.append(S1Pass("descending", 19, day, [f"grd-{role}"], ["sentinel-1d"], 1.0,
                                 f"{day}T01:00:00Z", f"{day}T01:00:00Z"))
        pair = ScenePair(*passes, "descending", 19, 10, state.days_after, 1.0, [])
        warnings = list(state.warnings.get("s1_search", []))
        if state.widened:
            warnings.append("Post window was widened from 2026-08-22 to 2026-09-03 for one retry (capped at today); pre window unchanged.")
        state.pair = PairSearchResult(pair, [], "ok", "Found a comparable pair.", 1, 1, warnings)
        return state.pair

    def preprocess_stage(aoi, pair_result, cache_dir):
        enter("preprocess")
        path = Path(cache_dir) / "s1" / "synthetic" / "stack.tif"
        if path.exists():
            return load_stack(path)
        array = np.ones((4, 2, 2), dtype="float32") * -12
        metadata = pair_result.pair.to_dict()
        metadata["input_item_ids"] = {"pre": ["rtc-pre"], "post": ["rtc-post"]}
        stack = S1Stack(array, np.ones((2, 2), dtype=bool), from_origin(303000, 3090000, state.resolution, state.resolution),
                        CRS.from_epsg(32645), state.resolution, aoi.bbox,
                        ("pre_vv", "pre_vh", "post_vv", "post_vh"), state.s1_source,
                        metadata, state.valid_fraction, list(state.warnings.get("preprocess", [])),
                        state.filter_window, str(path))
        save_stack(stack, path)
        path.with_name("quicklook.png").write_bytes(b"synthetic test preview")
        return stack

    def terrain_stage(aoi, stack, cache_dir):
        enter("terrain")
        path = Path(cache_dir) / "terrain" / "synthetic" / "terrain.tif"
        if path.exists():
            return load_terrain(path)
        array = np.ones((2, 2), dtype="float32")
        result = TerrainStack(array, array, array, np.zeros((2, 2), dtype=bool), stack.transform,
                              stack.crs, aoi.bbox, state.terrain_source, 30, 1.0, state.actual_buffer,
                              list(state.warnings.get("terrain", [])), ["dem-tile-1"], str(path))
        save_terrain(result, path)
        path.with_name("quicklook.png").write_bytes(b"synthetic test preview")
        # Controlled metadata lets threshold tests exercise values between tiny-grid fractions.
        class TerrainOutput:
            cache_path = result.cache_path
            def to_dict(self):
                data = result.to_dict()
                data["hand_nan_fraction"] = state.hand_fraction
                return data
        return TerrainOutput()

    def osm_stage(aoi, stack, cache_dir, include_buildings):
        enter("osm")
        assert stack.crs == CRS.from_epsg(32645)
        state.osm_buildings.append(include_buildings)
        path = Path(cache_dir) / "osm" / f"synthetic-{include_buildings}" / "osm.gpkg"
        if path.exists():
            return load_osm(path)
        frames = {}
        for name, columns in LAYER_COLUMNS.items():
            frame = gpd.GeoDataFrame(columns=[*columns, "geometry"], geometry="geometry", crs=4326)
            for column in ("is_bridge", "is_tunnel", "is_ford", "in_aoi"):
                if column in frame:
                    frame[column] = frame[column].astype(bool)
            if name == "settlements":
                frame["population"] = frame["population"].astype("Int64")
            frames[name] = frame.to_crs(stack.crs)
        snapshot = f"{aoi.flood_date}T00:00:00Z"
        newest = f"{aoi.flood_date - timedelta(days=1)}T00:00:00Z"
        quality = {"road_density_km_per_km2": state.density, "hospital_count": state.hospitals,
                   "settlement_count": state.settlements, "roads_edited_over_5_years_share": state.stale,
                   "context_buffer_km": 10, "counts": {name: 0 for name in LAYER_COLUMNS},
                   "timestamp_audit": {"newest_edit": newest, "newer_elements": 0, "missing_timestamps": 0,
                                       "snapshot_verified": state.verified}}
        bundle = OSMBundle(**frames, snapshot_date=snapshot, snapshot_verified=state.verified,
                           newest_edit=newest, endpoint_used="https://overpass.example.invalid/api/interpreter",
                           crs=stack.crs, bbox=aoi.bbox, quality=quality,
                           warnings=list(state.warnings.get("osm", [])), cache_path=str(path))
        save_osm(bundle, path)
        path.with_name("quicklook.png").write_bytes(b"synthetic test preview")
        return bundle

    for name, function in (("build_aoi", aoi_stage), ("find_best_pair", search_stage),
                           ("preprocess_pair", preprocess_stage), ("build_terrain", terrain_stage),
                           ("build_osm", osm_stage)):
        monkeypatch.setattr(mod, name, function)
    state.prepare = lambda **kwargs: mod.prepare_incident(bbox=BBOX, flood_date="2026-08-10", cache_dir=tmp_path, **kwargs)
    return state


def codes(incident):
    return {flag["code"]: flag for flag in incident.quality["flags"]}


def test_happy_path_receipt_and_metadata(fake_stages):
    result = fake_stages.prepare()
    assert result.status == "ready"
    assert [stage["status"] for stage in result.stages] == ["done"] * 5
    assert all(not stage["cache_hit"] for stage in result.stages)
    assert set(result.layers) == {"s1_stack", "s1_stack_metadata", "s1_stack_preview",
                                 "terrain", "terrain_metadata", "terrain_preview", "osm", "osm_metadata", "osm_preview"}
    assert all(Path(path).exists() for path in result.layers.values())
    assert result.pair_summary["relative_orbit"] == 19
    inputs = {item["role"]: item for item in result.provenance["inputs"]}
    assert inputs["s1_pre"]["item_ids"] == ["rtc-pre"]  # Actual inputs, not the GRD search ids.
    assert inputs["s1_post"]["source"] == "sentinel-1-rtc"
    assert inputs["dem"]["tile_ids"] == ["dem-tile-1"]
    assert inputs["osm"]["snapshot_verified"]
    assert all(item["license_note"] for item in inputs.values())
    assert result.provenance["disclaimer"] == mod.DISCLAIMER
    assert fake_stages.osm_buildings == [False]
    assert Path(result.cache_path).with_name("summary.md").exists()
    assert result.quality["trust_summary"] == "strong"
    json.dumps(result.to_dict(), allow_nan=False)


def test_blocked_pair_preserves_reason_and_skips_downstream(fake_stages):
    fake_stages.blocked = True
    result = fake_stages.prepare()
    assert result.status == "blocked"
    assert result.pair_summary["reason"] == "Different orbits; cannot analyze this area/date."
    assert [stage["status"] for stage in result.stages] == ["done", "blocked", "skipped", "skipped", "skipped"]
    assert set(fake_stages.calls) == {"aoi", "s1_search"}
    assert codes(result)["no_same_orbit_pair"]["severity"] == "limiting"
    assert result.quality["trust_summary"] == "insufficient"


@pytest.mark.parametrize("failure,independent", [("terrain", "osm"), ("osm", "terrain")])
def test_partial_stages_are_independent(fake_stages, failure, independent):
    fake_stages.errors[failure] = RuntimeError("Stage service unavailable")
    result = fake_stages.prepare()
    assert result.status == "partial"
    stages = {stage["name"]: stage for stage in result.stages}
    assert stages[failure]["status"] == "failed"
    assert stages[failure]["error"] == "RuntimeError: Stage service unavailable"
    assert stages[independent]["status"] == "done"
    assert fake_stages.calls[independent] == 1


@pytest.mark.parametrize("failure", ["aoi", "s1_search", "preprocess"])
def test_required_stage_failure_captured_and_skipped(fake_stages, failure):
    fake_stages.errors[failure] = ValueError("Invalid input")
    result = fake_stages.prepare()
    assert result.status == "failed"
    index = mod.STAGES.index(failure)
    assert result.stages[index]["error"] == "ValueError: Invalid input"
    assert all(stage["status"] == "skipped" for stage in result.stages[index + 1:])


def test_progress_is_ordered_and_errors_are_ignored(fake_stages, tmp_path):
    events = []
    result = fake_stages.prepare(progress=lambda *event: events.append(event))
    assert len(events) == 10
    assert [(event[0], event[1], event[2], event[3]) for event in events] == [
        (index + 1, 5, name, status) for index, name in enumerate(mod.STAGES) for status in ("running", "done")]
    def broken_callback(*event):
        raise RuntimeError("UI callback failed")
    cached = fake_stages.prepare(progress=broken_callback)
    assert cached.status == result.status == "ready"


def test_skipped_stages_have_start_end_progress(fake_stages):
    fake_stages.blocked = True
    events = []
    fake_stages.prepare(progress=lambda *event: events.append(event))
    assert len(events) == 10
    assert [event[3] for event in events[1::2]] == ["done", "blocked", "skipped", "skipped", "skipped"]


def test_fast_cache_calls_no_stages_and_force_reruns(fake_stages):
    first = fake_stages.prepare()
    fake_stages.calls.clear()
    events = []
    cached = fake_stages.prepare(progress=lambda *event: events.append(event))
    assert fake_stages.calls == Counter()
    assert cached.incident_id == first.incident_id
    assert all(stage["cache_hit"] for stage in cached.stages)
    assert all(stage["seconds"] == 0 for stage in cached.stages)
    assert [event[3] for event in events[1::2]] == ["cached"] * 5
    assert "cache_hit=True" in Path(cached.cache_path).with_name("summary.md").read_text()
    forced = fake_stages.prepare(force=True)
    assert forced.incident_id == first.incident_id
    assert fake_stages.calls == Counter(dict.fromkeys(mod.STAGES, 1))
    assert all(stage["cache_hit"] for stage in forced.stages[2:])
    assert any("force cache bypass unsupported" in warning for warning in forced.quality["warnings"])


def test_partial_receipt_resumes_failed_stage(fake_stages):
    fake_stages.errors["terrain"] = RuntimeError("Retry later")
    partial = fake_stages.prepare()
    assert partial.status == "partial"
    del fake_stages.errors["terrain"]
    fake_stages.calls.clear()
    ready = fake_stages.prepare()
    assert ready.status == "ready"
    assert ready.incident_id == partial.incident_id
    assert fake_stages.calls["terrain"] == 1
    assert ready.stages[2]["cache_hit"] and ready.stages[4]["cache_hit"]


def test_missing_layer_invalidates_fast_receipt(fake_stages):
    first = fake_stages.prepare()
    Path(first.layers["terrain"]).unlink()
    fake_stages.calls.clear()
    result = fake_stages.prepare()
    assert result.status == "ready"
    assert fake_stages.calls == Counter(dict.fromkeys(mod.STAGES, 1))
    assert result.stages[2]["cache_hit"] and not result.stages[3]["cache_hit"]


@pytest.mark.parametrize("change", ["bbox", "date", "buildings", "resolution", "filter", "buffer"])
def test_id_deterministic_and_changes_with_inputs(fake_stages, tmp_path, change):
    original = fake_stages.prepare()
    assert fake_stages.prepare().incident_id == original.incident_id
    bbox, day, buildings = BBOX, "2026-08-10", False
    if change == "bbox": bbox = (85.0001, 27.9, 85.001, 27.901)
    if change == "date": day = "2026-08-11"
    if change == "buildings": buildings = True
    if change == "resolution": fake_stages.resolution = 20
    if change == "filter": fake_stages.filter_window = 7
    if change == "buffer": fake_stages.actual_buffer = 8
    changed = mod.prepare_incident(bbox=bbox, flood_date=day, include_buildings=buildings, cache_dir=tmp_path / "other")
    assert changed.incident_id != original.incident_id
    assert len(changed.incident_id) == 16 and changed.incident_id.startswith("inc_")
    if change == "buildings": assert fake_stages.osm_buildings[-1] is True


@pytest.mark.parametrize("field,value,code,severity", [
    ("valid_fraction", 0.89, "s1_low_valid_fraction", "caution"),
    ("valid_fraction", 0.6, "s1_low_valid_fraction", "caution"),
    ("valid_fraction", 0.59, "s1_low_valid_fraction", "limiting"),
    ("days_after", 7, "acquisition_gap_long", "caution"),
    ("days_after", 0, "post_on_flood_date", "caution"),
    ("s1_source", "grd_fallback", "s1_grd_fallback", "limiting"),
    ("hand_fraction", 0.021, "hand_nan_present", "caution"),
    ("density", 0.29, "osm_sparse_network", "caution"),
    ("hospitals", 0, "osm_no_hospitals", "limiting"),
    ("settlements", 0, "osm_no_settlements", "limiting"),
    ("stale", 0.51, "osm_stale", "caution"),
    ("verified", False, "osm_snapshot_unverified", "limiting"),
    ("widened", True, "widened_post_window", "caution"),
])
def test_structured_flag_triggers(fake_stages, field, value, code, severity):
    setattr(fake_stages, field, value)
    result = fake_stages.prepare()
    assert codes(result)[code]["severity"] == severity
    if field == "verified":
        assert result.status == "partial"
        assert "ProvenanceError" in result.stages[4]["error"]
    if field == "s1_source":
        assert {entry["source"] for entry in result.provenance["inputs"] if entry["role"].startswith("s1_")} == {"sentinel-1-grd"}


@pytest.mark.parametrize("field,value,absent", [("valid_fraction", 0.9, "s1_low_valid_fraction"),
    ("days_after", 6, "acquisition_gap_long"), ("hand_fraction", 0.02, "hand_nan_present"),
    ("density", 0.3, "osm_sparse_network"), ("stale", 0.5, "osm_stale")])
def test_threshold_boundaries(fake_stages, field, value, absent):
    setattr(fake_stages, field, value)
    assert absent not in codes(fake_stages.prepare())


def test_information_flags_and_buildings_mode(fake_stages):
    result = fake_stages.prepare()
    assert {"dem_surface_model", "dem_resampled_30m", "buildings_deferred"} <= codes(result).keys()
    assert all(flag["severity"] == "info" for flag in result.quality["flags"])
    included = fake_stages.prepare(include_buildings=True)
    assert "buildings_deferred" not in codes(included)
    assert fake_stages.osm_buildings == [False, True]


@pytest.mark.parametrize("severities,status,expected", [([], "ready", "strong"),
    (["info"], "ready", "strong"), (["caution"], "ready", "probable"),
    (["caution", "caution"], "ready", "uncertain"), (["limiting"], "ready", "uncertain"),
    (["limiting", "caution"], "blocked", "insufficient")])
def test_trust_rules(severities, status, expected):
    assert mod.trust_summary([{"severity": value} for value in severities], status) == expected


@pytest.mark.parametrize("violation", ["source", "unverified", "later_date", "same_day_later_time", "training"])
def test_provenance_gate(fake_stages, violation):
    result = fake_stages.prepare()
    mod.assert_provenance_allowed(result)
    receipt = result.provenance["inputs"][-1]
    if violation == "source": receipt["source"] = "unapproved-vendor"
    if violation == "unverified": receipt["snapshot_verified"] = False
    if violation == "later_date": receipt["snapshot_date"] = "2026-08-11T00:00:00Z"
    if violation == "same_day_later_time": receipt["snapshot_date"] = "2026-08-10T01:00:00Z"
    if violation == "training": receipt.update(source="permitted-training-datasets", collection="not-registered")
    with pytest.raises(mod.ProvenanceError):
        mod.assert_provenance_allowed(result)


def test_disallowed_terrain_does_not_stop_osm(fake_stages):
    fake_stages.terrain_source = "unapproved-vendor"
    result = fake_stages.prepare()
    assert result.status == "partial"
    assert result.stages[3]["status"] == "failed"
    assert result.stages[4]["status"] == "done"


def test_warning_dedup_keeps_first_stage_prefix(fake_stages):
    fake_stages.warnings = {name: ["Shared warning", "Shared warning"] for name in mod.STAGES}
    fake_stages.warnings["terrain"].append("Terrain warning")
    result = fake_stages.prepare()
    assert result.quality["warnings"] == ["aoi: Shared warning", "terrain: Terrain warning"]


def test_roundtrip_and_lazy_layers(fake_stages, tmp_path, monkeypatch):
    result = fake_stages.prepare()
    restored = mod.load_incident(result.incident_id, cache_dir=tmp_path)
    assert restored.to_dict() == result.to_dict() == mod.Incident.from_dict(result.to_dict()).to_dict()
    assert mod.load_incident(Path(result.cache_path).parent).to_dict() == result.to_dict()
    calls = Counter()
    for name, function in (("load_stack", load_stack), ("load_terrain", load_terrain), ("load_osm", load_osm)):
        def reader(path, name=name, function=function):
            calls[name] += 1
            return function(path)
        monkeypatch.setattr(mod, name, reader)
    layers = mod.load_layers(restored)
    assert not calls
    assert layers.s1_stack.array.shape == (4, 2, 2)
    assert layers.s1_stack is layers.s1_stack
    assert layers.terrain.elevation.shape == (2, 2)
    assert layers.osm.snapshot_verified
    assert calls == Counter({"load_stack": 1, "load_terrain": 1, "load_osm": 1})
    Path(result.layers["s1_stack"]).unlink()
    with pytest.raises(FileNotFoundError, match="cache was deleted"):
        mod.load_layers(restored)
    with pytest.raises(FileNotFoundError, match="cache was deleted"):
        _ = layers.s1_stack


def test_cli_live_progress_and_summary_offline(fake_stages, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["incident", "--place", "Trishuli, Nepal", "--date", "2026-08-10", "--cache-dir", str(tmp_path)])
    mod.main()
    output = capsys.readouterr().out
    assert "[1/5] aoi ... running" in output
    assert "[5/5] osm ... done" in output
    summary = json.loads(output[output.index('{'):])
    assert summary["status"] == "ready"
    assert summary["flags"] == [{"code": flag["code"], "severity": flag["severity"]}
                                for flag in mod.load_incident(summary["cache_path"]).quality["flags"]]
    assert Path(summary["cache_path"]).with_name("summary.md").exists()


def test_out_of_range_fraction_is_a_failed_stage(fake_stages):
    fake_stages.valid_fraction = 1.2
    result = fake_stages.prepare()
    assert result.status == "failed"
    assert "valid_fraction" in result.stages[2]["error"]
    assert [stage["status"] for stage in result.stages[3:]] == ["skipped", "skipped"]


def test_invalid_osm_cannot_be_loaded_but_s1_remains_usable(fake_stages):
    fake_stages.verified = False
    result = fake_stages.prepare()
    layers = mod.load_layers(result)
    assert layers.s1_stack.array.shape == (4, 2, 2)
    with pytest.raises(mod.ProvenanceError, match="unsuccessful osm stage"):
        _ = layers.osm


def test_complete_receipt_requires_all_source_roles(fake_stages):
    result = fake_stages.prepare()
    result.provenance["inputs"] = []
    with pytest.raises(mod.ProvenanceError, match="Incomplete provenance"):
        mod.assert_provenance_allowed(result)


def test_receipt_write_failure_is_reported(fake_stages, monkeypatch):
    def cannot_save(*args, **kwargs):
        raise OSError("Disk full")
    monkeypatch.setattr(mod, "save_incident", cannot_save)
    result = fake_stages.prepare()
    assert result.status == "failed"
    assert codes(result)["incident_save_failed"]["message"] == "OSError: Disk full"


def test_supported_force_parameter_is_forwarded(fake_stages, monkeypatch):
    original = mod.build_terrain
    received = []
    def terrain_with_force(aoi, stack, cache_dir, force=False):
        received.append(force)
        return original(aoi, stack, cache_dir)
    monkeypatch.setattr(mod, "build_terrain", terrain_with_force)
    assert fake_stages.prepare(force=True).status == "ready"
    assert received == [True]


def test_id_uses_rounded_bbox_and_equivalent_numeric_params():
    request = {"flood_date": "2026-08-10"}
    aoi = {"bbox": list(BBOX), "flood_date": request["flood_date"]}
    same = {**aoi, "bbox": [value + 0.0000001 for value in BBOX]}
    assert mod._id(aoi, {"resolution": 10}, False, request) == mod._id(same, {"resolution": 10.0}, False, request)


@pytest.mark.parametrize("key", ["s1_stack", "terrain", "osm"])
def test_lazy_loader_rechecks_actual_source_metadata(fake_stages, key):
    result = fake_stages.prepare()
    path = Path(result.layers[key + "_metadata"])
    metadata = json.loads(path.read_text())
    if key == "osm":
        metadata["snapshot_verified"] = False
    else:
        metadata["source"] = "unapproved-vendor"
    path.write_text(json.dumps(metadata))
    layers = mod.load_layers(result)
    with pytest.raises(mod.ProvenanceError):
        getattr(layers, key)


def test_geojson_fallback_tracks_actual_files_even_with_partial_gpkg(fake_stages, monkeypatch):
    original = gpd.GeoDataFrame.to_file
    calls = 0
    def driver_fails_after_one_layer(frame, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError("Synthetic driver failure")
        return original(frame, *args, **kwargs)
    monkeypatch.setattr(gpd.GeoDataFrame, "to_file", driver_fails_after_one_layer)
    result = fake_stages.prepare()
    assert result.status == "ready"
    assert Path(result.layers["osm"]).is_dir()
    assert (Path(result.layers["osm"]) / "osm.gpkg").exists()
    assert all(f"osm_{name}" in result.layers for name in LAYER_COLUMNS)
    assert mod.load_layers(result).osm.snapshot_verified
    Path(result.layers["osm_hospitals"]).unlink()
    with pytest.raises(FileNotFoundError, match="cache was deleted"):
        mod.load_layers(result)
