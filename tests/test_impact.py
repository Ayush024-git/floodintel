"""Synthetic end-to-end output contracts; every loader is replaced locally."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json

import numpy as np
import pytest
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape

from src.graph import impact
from src.graph.road_graph import build_road_graph
from test_road_graph import bundle, flood


def setup(monkeypatch, tmp_path, *, empty=False, buildings=True, no_roads=False, clinics=False):
    osm = bundle(roads=[] if no_roads else None, buildings=buildings)
    if clinics:
        osm.hospitals.loc[:, "tier"] = "clinic"
    fm = flood(empty=empty)
    receipt = SimpleNamespace(incident_id="inc_123456789abc", status="ready", cache_path=str(tmp_path / "incident.json"),
                              provenance={"request": {"include_buildings": buildings}}, quality={"flags": [], "warnings": []})
    receipt.to_dict = lambda: dict(incident_id=receipt.incident_id, status=receipt.status)
    stack = SimpleNamespace(valid_mask=fm.valid_mask, transform=fm.transform, crs=fm.crs, resolution=fm.resolution,
                            bbox=osm.bbox, array=np.full((4, *fm.flood_mask.shape), -20, np.float32),
                            band_names=("pre_vv", "pre_vh", "post_vv", "post_vh"))
    monkeypatch.setattr(impact, "load_incident", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(impact, "load_layers", lambda _: SimpleNamespace(osm=osm, s1_stack=stack))
    monkeypatch.setattr(impact, "load_flood_map", lambda _: fm)
    return receipt, osm, fm


def test_end_to_end_files_valid_geojson_and_cache(monkeypatch, tmp_path):
    receipt, osm, fm = setup(monkeypatch, tmp_path)
    result = impact.analyze_incident_impact(receipt.incident_id, flood_dir=tmp_path / "fake", approach_buffer_m=0, snap_max_m=50)
    path = Path(result.cache_path)
    assert set(impact.OUTPUTS) <= {file.name for file in path.iterdir()}
    assert result.metadata["status_tallies"]["CUT OFF"] == 2
    assert result.metadata["counts"]["counts"]["bridges"] == 1
    for filename in impact.OUTPUTS:
        if filename.endswith(".geojson"):
            collection = json.loads((path / filename).read_text())
            assert collection["type"] == "FeatureCollection"
            assert all(shape(feature["geometry"]).is_valid for feature in collection["features"])
        elif filename.endswith(".json"):
            json.loads((path / filename).read_text())
    assert (path / "quicklook.png").read_bytes().startswith(b"\x89PNG")
    forbidden = Mock(side_effect=AssertionError("cache should skip graph computation"))
    monkeypatch.setattr(impact, "assess_damage", forbidden)
    again = impact.analyze_incident_impact(receipt.incident_id, flood_dir=tmp_path / "fake", approach_buffer_m=0, snap_max_m=50)
    assert again.metadata["cache_hit"] and again.cache_path == result.cache_path
    forbidden.assert_not_called()


def test_synthetic_polygon_writes_only_test_output_with_banner(monkeypatch, tmp_path):
    receipt, _, fm = setup(monkeypatch, tmp_path)
    coordinates = shape(transform_geom(fm.crs, "EPSG:4326", mapping(box(300050, 3099990, 300150, 3100010)))).bounds
    result = impact.analyze_incident_impact(receipt.incident_id, test_polygon=coordinates, approach_buffer_m=0, snap_max_m=50)
    assert Path(result.cache_path).is_relative_to(tmp_path / "impact_test")
    assert not (tmp_path / "impact").exists()
    assert result.metadata["banner"] == impact.BANNER
    assert any(flag["code"] == "synthetic_flood_input" and flag["severity"] == "limiting" for flag in result.metadata["flags"])
    assert result.metadata["status_tallies"]["CUT OFF"] == 2


def test_no_flood_connected_and_deferred_building_flag(monkeypatch, tmp_path):
    receipt, _, _ = setup(monkeypatch, tmp_path, empty=True, buildings=False)
    result = impact.analyze_incident_impact(receipt.incident_id, flood_dir=tmp_path / "fake", snap_max_m=50)
    assert result.metadata["status_tallies"]["CONNECTED"] == 2
    assert result.metadata["counts"]["counts"]["buildings"] == 0
    assert {"no_flood_detected", "buildings_deferred"} <= {flag["code"] for flag in result.metadata["flags"]}


def test_empty_graph_and_only_clinics_flags_no_crash(monkeypatch, tmp_path):
    receipt, _, _ = setup(monkeypatch, tmp_path, empty=True, no_roads=True)
    result = impact.analyze_incident_impact(receipt.incident_id, flood_dir=tmp_path / "fake")
    assert result.metadata["status_tallies"]["UNCERTAIN"] == 2
    priorities = json.loads((Path(result.cache_path) / "priorities.json").read_text())
    assert not priorities["settlements"] and len(priorities["data_gaps"]) == 2
    assert priorities["message"] == "No flood-attributable priorities"
    assert "road_graph_empty" in {flag["code"] for flag in result.metadata["flags"]}
    receipt, _, _ = setup(monkeypatch, tmp_path, empty=True, clinics=True)
    result = impact.analyze_incident_impact(receipt.incident_id, flood_dir=tmp_path / "fake")
    assert "only_clinics_no_hospitals" in {flag["code"] for flag in result.metadata["flags"]}


def test_suggest_boxes_and_synthetic_cannot_enter_real_directory(monkeypatch, tmp_path):
    receipt, osm, fm = setup(monkeypatch, tmp_path)
    suggested = impact.suggest_test_boxes(osm)
    assert suggested[0]["osm_id"] == "way/1" and len(suggested[0]["bbox"]) == 4
    fm.params["synthetic"] = True
    with pytest.raises(ValueError, match="never write it under impact"):
        impact.analyze_incident_impact(receipt.incident_id, flood_dir=tmp_path / "fake")


def test_test_link_box_and_test_only_outputs(monkeypatch, tmp_path):
    receipt, osm, fm = setup(monkeypatch, tmp_path)
    network = build_road_graph(osm, snap_max_m=50)
    link_id = next(data["link_id"] for _, _, data in network.graph.edges(data=True) if data["is_bridge"])
    bounds = impact.test_link_box(osm, link_id, buffer_m=30, snap_max_m=50)
    assert len(bounds) == 4 and bounds[0] < bounds[2] and bounds[1] < bounds[3]
    result = impact.analyze_incident_impact(receipt.incident_id, test_link=link_id, snap_max_m=50)
    assert Path(result.cache_path).is_relative_to(tmp_path / "impact_test")
    assert not (tmp_path / "impact").exists()
    assert result.metadata["banner"] == impact.BANNER
    assert result.metadata["flood_params"]["test_link"] == link_id
    assert result.metadata["flood_params"]["test_link_buffer_m"] == 30
    assert result.metadata["status_tallies"]["CUT OFF"] == 2
    assert any(flag["code"] == "synthetic_flood_input" for flag in result.metadata["flags"])


def test_unknown_or_conflicting_test_links_fail_clearly(monkeypatch, tmp_path):
    receipt, _, _ = setup(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="Unknown baseline link_id"):
        impact.analyze_incident_impact(receipt.incident_id, test_link="missing")
    with pytest.raises(ValueError, match="mutually exclusive"):
        impact.analyze_incident_impact(receipt.incident_id, test_link="missing", test_polygon=(85, 27, 86, 28))
    with pytest.raises(ValueError, match="mutually exclusive"):
        impact.analyze_incident_impact(receipt.incident_id, test_link="missing", flood_dir="fake")


def test_test_link_outside_valid_grid_fails_without_fake_success(monkeypatch, tmp_path):
    receipt, osm, fm = setup(monkeypatch, tmp_path)
    fm.valid_mask[:] = False
    link_id = impact.list_critical_bridges(osm, snap_max_m=50)[0]["link_id"]
    with pytest.raises(ValueError, match="no valid S1 pixels"):
        impact.analyze_incident_impact(receipt.incident_id, test_link=link_id)
    assert not (tmp_path / "impact_test").exists()


def test_critical_bridge_list_uses_baseline_and_stable_count_id_order():
    osm = bundle()
    result = impact.list_critical_bridges(osm, snap_max_m=50)
    assert len(result) == 1 and result[0]["count"] == 2
    assert result[0]["settlements_affected"] == ["Hamlet", "Village"]
    assert 84 < result[0]["lon"] < 90 and 27 < result[0]["lat"] < 30
    assert result == impact.list_critical_bridges(osm, snap_max_m=50)
    assert impact.list_critical_bridges(bundle([], hospitals=[], towns=False)) == []


def test_zero_leverage_links_not_in_top_chokepoints_or_ranked_links(monkeypatch, tmp_path, capsys):
    import sys
    receipt, _, _ = setup(monkeypatch, tmp_path)
    analyze = impact.analyze_chokepoints

    def no_leverage(*args, **kwargs):
        result = analyze(*args, **kwargs)
        for row in result.damaged:
            row.update(count=0, settlement_ids=[], settlements_affected=[], sentence=None)
        return result

    monkeypatch.setattr(impact, "analyze_chokepoints", no_leverage)
    monkeypatch.setattr(sys, "argv", ["impact", "--incident", receipt.incident_id, "--flood-dir", "fake", "--no-latent"])
    impact.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["top_chokepoints"] == []
    directory = Path(summary["cache_path"])
    choke_json = json.loads((directory / "chokepoints.json").read_text())
    priorities = json.loads((directory / "priorities.json").read_text())
    assert not choke_json["damaged"] and len(choke_json["damaged_no_leverage"]) == 1
    assert not priorities["links"] and len(priorities["damaged_no_leverage"]) == 1


def test_list_critical_bridges_cli_never_loads_flood_or_analyzes_incident(monkeypatch, tmp_path, capsys):
    import sys
    receipt, _, _ = setup(monkeypatch, tmp_path)
    forbidden = Mock(side_effect=AssertionError("list must use baseline graph only"))
    monkeypatch.setattr(impact, "load_flood_map", forbidden)
    monkeypatch.setattr(impact, "analyze_incident_impact", forbidden)
    monkeypatch.setattr(sys, "argv", ["impact", "--incident", receipt.incident_id, "--list-critical-bridges"])
    impact.main()
    results = json.loads(capsys.readouterr().out)
    assert results[0]["count"] == 2
    forbidden.assert_not_called()


def test_test_link_cli_passes_id_and_modes_are_exclusive(monkeypatch, tmp_path, capsys):
    import sys
    receipt, _, _ = setup(monkeypatch, tmp_path)
    results = impact.list_critical_bridges(impact.load_layers(receipt).osm)
    monkeypatch.setattr(sys, "argv", ["impact", "--incident", receipt.incident_id, "--test-link", results[0]["link_id"]])
    impact.main()
    output = json.loads(capsys.readouterr().out)
    assert "/impact_test/" in output["cache_path"]
    assert any(flag["code"] == "synthetic_flood_input" for flag in output["flags"])
    monkeypatch.setattr(sys, "argv", ["impact", "--incident", receipt.incident_id, "--test-link", "id", "--list-critical-bridges"])
    with pytest.raises(SystemExit) as error:
        impact.main()
    assert error.value.code == 2


def candidate_file(tmp_path, fm, *, geometry=None):
    directory = tmp_path / "candidate_flood"
    directory.mkdir(exist_ok=True)
    geometry = geometry if geometry is not None else box(300050, 3099990, 300150, 3100010)
    geographic = transform_geom(fm.crs, "EPSG:4326", mapping(geometry))
    payload = dict(type="FeatureCollection", features=[dict(type="Feature", geometry=geographic,
                   properties=dict(max_confidence=3, label="unconfirmed candidate"))])
    (directory / "debris_candidates.geojson").write_text(json.dumps(payload))
    return directory


def test_debris_only_is_uncertain_never_cut_off_separate_cache_and_unchanged_default(monkeypatch, tmp_path):
    receipt, _, fm = setup(monkeypatch, tmp_path, empty=True)
    directory = candidate_file(tmp_path, fm)
    default = impact.analyze_incident_impact(receipt.incident_id, directory, approach_buffer_m=0, snap_max_m=50)
    assert default.metadata["status_tallies"] == {"CUT OFF": 0, "UNCERTAIN": 0, "CONNECTED": 2}
    assert Path(default.cache_path).is_relative_to(tmp_path / "impact")
    assert "debris_candidate_input" not in {flag["code"] for flag in default.metadata["flags"]}
    result = impact.analyze_incident_impact(receipt.incident_id, directory, include_debris_candidates=True,
                                           approach_buffer_m=0, snap_max_m=50)
    assert result.metadata["status_tallies"] == {"CUT OFF": 0, "UNCERTAIN": 2, "CONNECTED": 0}
    assert Path(result.cache_path).is_relative_to(tmp_path / "impact_debris")
    assert result.cache_path != default.cache_path
    assert impact.DEBRIS_FLAG in result.metadata["flags"]
    assert result.metadata["params"]["damage"]["debris_candidate_tier"] == 1
    assert result.metadata["debris_candidates"]["tier"] == 1
    assert all(row["confidence_tier"] == 1 and row["possible_cut_from"] and not row["cut_from"] for row in result.settlements)
    for row in result.settlements:
        assert "unconfirmed debris-candidate" in row["reason"]
        assert row["evidence"][0]["members"][0]["evidence_source"] == "debris_candidate"
    assets = json.loads((Path(result.cache_path) / "damaged_assets.geojson").read_text())["features"]
    assert assets and all(feature["properties"]["tier"] == 1 for feature in assets)
    assert all("unconfirmed debris" in feature["properties"]["evidence_label"] for feature in assets)
    assert not fm.flood_mask.any() and not fm.confidence.any() and fm.params == {}
    loader = Mock(side_effect=AssertionError("candidate cache must skip damage computation"))
    monkeypatch.setattr(impact, "assess_damage", loader)
    again = impact.analyze_incident_impact(receipt.incident_id, directory, include_debris_candidates=True,
                                         approach_buffer_m=0, snap_max_m=50)
    assert again.metadata["cache_hit"]
    loader.assert_not_called()


def test_candidate_export_change_invalidates_cache_and_invalid_pixels_are_excluded(monkeypatch, tmp_path):
    receipt, _, fm = setup(monkeypatch, tmp_path, empty=True)
    directory = candidate_file(tmp_path, fm)
    first = impact.analyze_incident_impact(receipt.incident_id, directory, include_debris_candidates=True, approach_buffer_m=0, snap_max_m=50)
    candidate_file(tmp_path, fm, geometry=box(301000, 3100500, 301100, 3100600))
    second = impact.analyze_incident_impact(receipt.incident_id, directory, include_debris_candidates=True, approach_buffer_m=0, snap_max_m=50)
    assert first.cache_path != second.cache_path
    assert second.metadata["status_tallies"]["CONNECTED"] == 2
    fm.valid_mask[:] = False
    evidence, _ = impact._debris_evidence(fm, directory)
    assert not evidence.flood_mask.any()


def test_mixed_evidence_does_not_promote_subthreshold_flood_to_cut_off(monkeypatch, tmp_path):
    receipt, osm, fm = setup(monkeypatch, tmp_path, empty=True)
    # A non-bridge 200 m road: 10 m tier-3 flood is below 20 m / 0.1 thresholds.
    osm.roads.loc[:, "is_bridge"] = False
    fm.flood_mask[100, 5] = True
    fm.confidence[100, 5] = 3
    directory = candidate_file(tmp_path, fm)
    result = impact.analyze_incident_impact(receipt.incident_id, directory, include_debris_candidates=True,
                                           approach_buffer_m=0, snap_max_m=50)
    assert result.metadata["status_tallies"]["CUT OFF"] == 0
    assert result.metadata["status_tallies"]["UNCERTAIN"] == 2
    assert all(row["confidence_tier"] == 1 for row in result.settlements)


def test_qualified_flood_keeps_its_tier_when_candidates_also_present(monkeypatch, tmp_path):
    receipt, _, fm = setup(monkeypatch, tmp_path)
    directory = candidate_file(tmp_path, fm)
    result = impact.analyze_incident_impact(receipt.incident_id, directory, include_debris_candidates=True,
                                           approach_buffer_m=0, snap_max_m=50)
    assert result.metadata["status_tallies"]["CUT OFF"] == 2
    assert all(row["confidence_tier"] == 3 for row in result.settlements)


def test_candidate_mode_missing_file_is_clear_and_cannot_combine_with_synthetic(monkeypatch, tmp_path):
    receipt, _, _ = setup(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="Debris candidate polygons are missing"):
        impact.analyze_incident_impact(receipt.incident_id, tmp_path / "missing", include_debris_candidates=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        impact.analyze_incident_impact(receipt.incident_id, test_link="any", include_debris_candidates=True)
    assert not (tmp_path / "impact_debris").exists()


def test_candidate_cli_writes_only_candidate_output(monkeypatch, tmp_path, capsys):
    import sys
    receipt, _, fm = setup(monkeypatch, tmp_path, empty=True)
    directory = candidate_file(tmp_path, fm)
    monkeypatch.setattr(sys, "argv", ["impact", "--incident", receipt.incident_id, "--flood-dir", str(directory),
                                     "--include-debris-candidates", "--approach-buffer-m", "0", "--snap-max-m", "50"])
    impact.main()
    summary = json.loads(capsys.readouterr().out)
    assert Path(summary["cache_path"]).is_relative_to(tmp_path / "impact_debris")
    assert not (tmp_path / "impact").exists()
    assert summary["status_tallies"]["CUT OFF"] == 0
    assert impact.DEBRIS_FLAG in summary["flags"]


def test_previous_priority_cache_is_not_reused(monkeypatch, tmp_path):
    receipt, _, _ = setup(monkeypatch, tmp_path, empty=True)
    current_version = impact.VERSION
    monkeypatch.setattr(impact, "VERSION", 1)
    previous = impact.analyze_incident_impact(receipt.incident_id, flood_dir="fake", snap_max_m=50)
    (Path(previous.cache_path) / "priorities.json").write_text(json.dumps({"settlements": [{"id": "old-baseline-only-priority"}]}))
    monkeypatch.setattr(impact, "VERSION", current_version)
    current = impact.analyze_incident_impact(receipt.incident_id, flood_dir="fake", snap_max_m=50)
    assert current.cache_path != previous.cache_path and not current.metadata["cache_hit"]
    assert current.priorities["settlements"] == []
    assert current.priorities["message"] == "No flood-attributable priorities"


def test_critical_bridge_list_caps_ten_and_orders_equal_counts_by_id(monkeypatch):
    osm = bundle()
    baseline_rows = impact.list_critical_bridges(osm)
    network = build_road_graph(osm)
    edge = next(data for _, _, data in network.graph.edges(data=True) if data["is_bridge"])
    candidates = [dict(link_id=f"link_{index:02d}", name=None, geometry=edge["geometry"], count=2,
                       settlements_affected=["Hamlet", "Village"], settlement_ids=["node/h", "node/v"])
                  for index in reversed(range(12))]
    candidates.append(dict(candidates[0], link_id="link_zero", count=0))
    monkeypatch.setattr(impact, "analyze_chokepoints", lambda *args, **kwargs: SimpleNamespace(latent=candidates))
    results = impact.list_critical_bridges(osm)
    assert len(results) == 10
    assert [row["link_id"] for row in results] == [f"link_{index:02d}" for index in range(10)]
    assert results[0]["lon"] == baseline_rows[0]["lon"]
