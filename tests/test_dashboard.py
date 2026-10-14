"""Static report packaging, mapping fallbacks and CLI using synthetic caches."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import shape

from src.report import dashboard
from src.report.build_report import BANNER, build_report_model, numbers_trace
from test_report import comparison_input, fc, feature, inputs, visible, write


def data_block(html):
    return json.loads(re.search(r'<script type="application/json" id="incident-data">(.*?)</script>', html, flags=re.S).group(1))


def test_one_file_all_views_drawer_and_offline(tmp_path):
    directory, impact, _ = inputs(tmp_path)
    model = build_report_model(directory, impact)
    html, meta = dashboard.render_dashboard(model)
    data = data_block(html)
    assert data["settlements"][0]["nearest_baseline"]["hospital"]["travel_time_s"] == 300
    assert data["clusters"][0]["responsible_links"][0]["sufficient_to_restore"]
    for view in dashboard.VIEWS:
        assert view in html
    for marker in ("offline-fallback", "offline-svg", "showEvidence", "sufficient_to_restore", "baseline-route", "detour-route", "selected-links", "prefers-reduced-motion", "Content-Security-Policy"):
        assert marker in html
    assert "fetch(" not in html
    assert 'connect-src &#x27;none&#x27;' in html
    assert "maplibre-gl@4.7.1" in html
    assert "tile.openstreetmap" not in html
    # Only downloads use sibling paths; scripts/styles/images need no local assets.
    parser = AssetReferences()
    parser.feed(html)
    assert all(url.startswith(("https://unpkg.com/", "data:")) for url in parser.urls)
    assert len(html.encode()) <= dashboard.HTML_BUDGET_BYTES
    assert data["display_metadata"]["originals"]
    assert numbers_trace(visible(html), model["trace_sources"])["untraced"] == []


class AssetReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("img", "script") and "src" in attrs:
            self.urls.append(attrs["src"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.urls.append(attrs["href"])


def test_hostile_json_is_inert_and_text_escaped(tmp_path):
    directory, impact, _ = inputs(tmp_path, hostile=True)
    html, _ = dashboard.render_dashboard(build_report_model(directory, impact))
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    data = data_block(html)
    assert 'गाउँ <script>alert("x")</script>' in data["settlements"][0]["name"]
    assert "गाउँ" in visible(html)
    assert "textContent=title" in html
    assert "innerHTML" not in html
    assert not any(word in visible(html).lower() for word in ("destroyed", "casualt", "death", "population"))


def test_caps_simplification_budget_and_determinism(tmp_path):
    directory, impact, _ = inputs(tmp_path)
    # A mid-size vector fixture with many redundant vertices.
    features = []
    for index in range(1200):
        coordinates = [[85 + step * 0.00001, 27.85 + index * 0.000001] for step in range(30)]
        features.append(feature(dict(osm_id=f"way/{index}", name="Road", asset_class="road", tier=2), dict(type="LineString", coordinates=coordinates)))
    write(impact / "damaged_assets.geojson", fc(features))
    model = build_report_model(directory, impact)
    first, metadata = dashboard.render_dashboard(model, caps=dict(assets=100))
    second, _ = dashboard.render_dashboard(model, caps=dict(assets=100))
    assert first == second
    layer = metadata["geometry_layers"]["assets"]
    assert layer["input_features"] == 1200
    assert layer["kept_features"] == 100
    assert layer["capped_features"] == 1100
    assert layer["simplify_m"] == 5
    data = data_block(first)
    assert len(data["assets"]) == 1200  # complete table, only map capped
    assert all(len(feature["geometry"]["coordinates"]) == 2 for feature in data["layers"]["assets"]["features"])
    assert len(first.encode()) < 2 * 1024 * 1024
    assert all(shape(feature["geometry"]).is_valid for layer in data["layers"].values() for feature in layer["features"])
    with pytest.raises(ValueError, match="caps"):
        dashboard.render_dashboard(model, caps=dict(assets=-1))


def test_raster_categories_add_preexisting_and_exclusions(tmp_path):
    directory, impact, flood = inputs(tmp_path)
    transform = from_origin(85.05, 27.95, 0.01, 0.01)
    array = np.zeros((3, 8, 8), np.uint8)
    array[0, 1:3, 1:3] = 2
    array[0, 3:5, 3:5] = 3
    array[1, 5:7, 1:3] = 1
    array[2, 5:7, 5:7] = 1
    with rasterio.open(flood / "flood_masks.tif", "w", driver="GTiff", height=8, width=8, count=3, dtype="uint8", crs="EPSG:4326", transform=transform) as raster:
        raster.write(array)
        raster.descriptions = ("confidence", "preexisting_water", "terrain_excluded")
    html, _ = dashboard.render_dashboard(build_report_model(directory, impact))
    data = data_block(html)
    assert {feature["properties"]["tier"] for feature in data["layers"]["flood"]["features"]} == {2, 3}
    assert data["layers"]["preexisting"]["features"]
    assert data["layers"]["excluded"]["features"]


def test_synthetic_banner_persistent_all_views_and_package(tmp_path):
    directory, impact, _ = inputs(tmp_path, synthetic=True)
    result = dashboard.build_dashboard(directory, impact)
    output = Path(result["output_path"])
    html = (output / "index.html").read_text()
    assert BANNER in html
    assert html.index('class="banner"') < html.index('class="layout"')
    assert data_block(html)["synthetic"]
    assert len(data_block(html)["layers"]["flood"]["features"]) == 1
    for filename in ("situation_report.html", "situation_report.md"):
        assert BANNER in (output / filename).read_text()
    with zipfile.ZipFile(output / "incident_package.zip") as package:
        assert not any(name.startswith("flood/") for name in package.namelist())
        assert BANNER in package.read("README.txt").decode()


def test_zip_contains_original_artifacts_and_provenance(tmp_path):
    directory, impact, flood = inputs(tmp_path)
    result = dashboard.build_dashboard(directory, impact)
    output = Path(result["output_path"])
    expected = {"index.html", "situation_report.html", "situation_report.md", "README.txt", "provenance.json", "dashboard_meta.json"}
    expected |= {"impact/" + path.name for path in impact.iterdir() if path.suffix in (".json", ".geojson")}
    expected |= {"flood/" + path.name for path in flood.iterdir() if path.suffix in (".json", ".geojson")}
    with zipfile.ZipFile(output / "incident_package.zip") as package:
        assert set(package.namelist()) == expected
        assert json.loads(package.read("provenance.json")) == json.loads((directory / "incident.json").read_text())["provenance"]
        assert package.read("impact/routes.geojson") == (impact / "routes.geojson").read_bytes()
    original = (output / "incident_package.zip").read_bytes()
    dashboard.build_dashboard(directory, impact)
    assert original == (output / "incident_package.zip").read_bytes()


def test_cli_synthetic_end_to_end_and_missing_error(tmp_path, capsys):
    directory, impact, _ = inputs(tmp_path)
    out = tmp_path / "report"
    assert dashboard.main(["--incident", str(directory), "--impact-dir", str(impact), "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["output_path"] == str(out)
    assert (out / "index.html").is_file()
    with pytest.raises(SystemExit) as exc:
        dashboard.main(["--incident", str(directory), "--impact-dir", str(tmp_path / "missing")])
    assert exc.value.code == 2
    assert "Impact directory is missing" in capsys.readouterr().err


def test_default_uses_real_impact_and_linked_flood_not_newer_unrelated(tmp_path):
    directory, impact, flood = inputs(tmp_path)
    write(directory / "impact_test" / "new" / "impact_meta.json", dict(incident_id=directory.name, banner=BANNER))
    write(directory / "flood" / "unrelated" / "flood_meta.json", dict(stats=dict(flood_area_km2=999)))
    model = build_report_model(directory)
    assert model["paths"]["impact_dir"] == str(impact)
    assert model["paths"]["flood_dir"] == str(flood)
    assert not model["synthetic"]


def test_no_flood_dashboard_shows_gaps_without_ranked_settlements(tmp_path):
    directory, impact, _ = inputs(tmp_path, no_flood=True)
    html, _ = dashboard.render_dashboard(build_report_model(directory, impact))
    data = data_block(html)
    assert "No flood detected" in html
    assert "No flood-attributable priorities" in html
    assert data["priorities"]["settlements"] == []
    assert data["priorities"]["data_gaps"]
    assert data["chokepoints"]["latent"]


def test_reuses_cached_osm_and_preserves_context_destinations(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    directory, impact, _ = inputs(tmp_path)
    model = build_report_model(directory, impact)
    model["incident"]["layers"]["osm"] = "../osm/osm.gpkg"
    roads = gpd.GeoDataFrame([dict(osm_id="way/88", highway="primary", name="Context road", is_bridge=False,
                                   geometry=LineString([(85.1, 27.9), (85.25, 27.9)]))], crs=4326)
    hospitals = gpd.GeoDataFrame([dict(osm_id="node/22", name="Outside hospital", tier="hospital", in_aoi=False,
                                      geometry=Point(85.25, 27.9))], crs=4326)
    loader = Mock(return_value=SimpleNamespace(roads=roads, hospitals=hospitals, snapshot_verified=True))
    monkeypatch.setattr(dashboard, "load_osm", loader)
    html, _ = dashboard.render_dashboard(model)
    data = data_block(html)
    loader.assert_called_once()
    assert data["layers"]["roads"]["features"][0]["properties"]["osm_id"] == "way/88"
    assert data["layers"]["hospitals"]["features"][0]["properties"]["osm_id"] == "node/22"
    loader.return_value.snapshot_verified = False
    data = dashboard.build_dashboard_data(model)
    assert not data["layers"]["roads"]["features"]
    assert any("unverified" in warning for warning in data["warnings"])


def test_six_decimal_display_coordinates_and_lower_defaults(tmp_path):
    directory, impact, _ = inputs(tmp_path)
    coordinates = [[85.05123456789, 27.85678912345], [85.15198765432, 27.95298765432]]
    collection = fc([feature(dict(osm_id="way/999", asset_class="road", tier=2), dict(type="LineString", coordinates=coordinates))])
    write(impact / "damaged_assets.geojson", collection)
    original = (impact / "damaged_assets.geojson").read_bytes()
    model = build_report_model(directory, impact)
    html, metadata = dashboard.render_dashboard(model)
    displayed = data_block(html)["layers"]["assets"]["features"][0]["geometry"]["coordinates"]
    assert all(round(number, 6) == number for point in displayed for number in point)
    assert (impact / "damaged_assets.geojson").read_bytes() == original
    assert model["geo"]["damaged_assets"] == collection
    assert dashboard.DEFAULT_CAPS["roads"] == 2000
    assert dashboard.DEFAULT_CAPS["routes"] == 100
    assert metadata["geometry_layers"]["assets"]["coordinate_decimals"] == 6
    embedded = re.search(r'<script type="application/json" id="incident-data">(.*?)</script>', html, re.S).group(1)
    assert metadata["embedded_json_bytes"] == len(embedded.encode())
    assert metadata["embedded_block_bytes"]["layers.assets"] == len(dashboard._safe_script(data_block(html)["layers"]["assets"]).encode())


def test_largest_geometry_dropped_with_visible_note_and_byte_diagnostic(tmp_path, monkeypatch, capsys):
    directory, impact, _ = inputs(tmp_path)
    coordinates = [[85.05 + step * 0.000001, 27.9 + (step % 2) * 0.0002] for step in range(8000)]
    route = feature(dict(settlement_id="node/11", scenario="baseline", destination_class="hospital"), dict(type="LineString", coordinates=coordinates))
    write(impact / "routes.geojson", fc([route]))
    original = (impact / "routes.geojson").read_bytes()
    monkeypatch.setattr(dashboard, "HTML_BUDGET_BYTES", 150_000)
    assert dashboard.main(["--incident", str(directory), "--impact-dir", str(impact)]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert "layers.routes=" in captured.err and "bytes" in captured.err
    assert "Omitted: layers.routes" in captured.err
    output = Path(result["output_path"])
    html = (output / "index.html").read_text()
    metadata = json.loads((output / "dashboard_meta.json").read_text())
    assert len(html.encode()) <= dashboard.HTML_BUDGET_BYTES
    assert "layers.routes" in visible(html)
    assert 'class="budget-note"' in html
    assert metadata["omitted_layers"] == ["routes"]
    assert metadata["geometry_layers"]["routes"]["budget_omitted_features"] == 1
    assert metadata["geometry_layers"]["routes"]["kept_features"] == 0
    assert metadata["initial_embedded_block_bytes"]["layers.routes"] > metadata["embedded_block_bytes"]["layers.routes"]
    assert data_block(html)["layers"]["routes"]["features"] == []
    assert metadata["html_bytes"] == len(html.encode())
    with zipfile.ZipFile(output / "incident_package.zip") as package:
        assert package.read("impact/routes.geojson") == original
        assert json.loads(package.read("dashboard_meta.json")) == metadata


@pytest.mark.parametrize("synthetic", [False, True])
def test_mid_size_with_large_evidence_always_writes_under_budget(tmp_path, synthetic):
    directory, impact, _ = inputs(tmp_path, synthetic=synthetic)
    # Moderately many features, but bulky non-map evidence reproduces the failure
    # that geometry caps alone cannot solve. Source files must remain complete.
    features = [feature(dict(osm_id=f"way/{index}", name=f"Road {index}", asset_class="road", tier=3,
                             evidence_note="source evidence " + "x" * 25_000),
                        dict(type="LineString", coordinates=[[85.05123456789, 27.85678912345], [85.15198765432, 27.95298765432]]))
                for index in range(600)]
    write(impact / "damaged_assets.geojson", fc(features))
    original = (impact / "damaged_assets.geojson").read_bytes()
    result = dashboard.build_dashboard(directory, impact)
    output = Path(result["output_path"])
    html = (output / "index.html").read_text()
    assert len(html.encode()) <= 12 * 1024 * 1024
    assert result["metadata"]["initial_html_bytes"] > dashboard.HTML_BUDGET_BYTES
    assert result["metadata"]["omitted_blocks"][0]["name"] == "assets"
    assert "assets" in visible(html)
    assert (BANNER in html) == synthetic
    with zipfile.ZipFile(output / "incident_package.zip") as package:
        assert package.read("impact/damaged_assets.geojson") == original
        assert len(json.loads(package.read("impact/damaged_assets.geojson"))["features"]) == 600


def test_irreducible_prose_still_writes_a_labelled_index(tmp_path, monkeypatch):
    directory, impact, _ = inputs(tmp_path, synthetic=True)
    model = build_report_model(directory, impact)
    title = model["report"]["title"]
    model["report"]["title"] = title + "x" * 200_000
    monkeypatch.setattr(dashboard, "HTML_BUDGET_BYTES", 80_000)
    monkeypatch.setattr(dashboard, "build_report_model", lambda *args, **kwargs: model)
    result = dashboard.build_dashboard(directory, impact)
    html = (Path(result["output_path"]) / "index.html").read_text()
    assert len(html.encode()) <= dashboard.HTML_BUDGET_BYTES
    assert BANNER in html
    assert "Dashboard display omitted" in html
    assert result["metadata"]["omitted_blocks"][-1]["name"] == "full dashboard"
    assert result["metadata"]["embedded_block_bytes"] == {}
    assert model["report"]["title"] == title + "x" * 200_000
    assert (Path(result["output_path"]) / "incident_package.zip").is_file()


def test_status_gap_note_complete_legend_and_distinct_offline_markers(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import geopandas as gpd
    from shapely.geometry import Point
    directory, impact, _ = inputs(tmp_path)
    model = build_report_model(directory, impact)
    model["incident"]["layers"]["osm"] = "local.gpkg"
    hospitals = gpd.GeoDataFrame([dict(osm_id="hospital", name="Hospital", tier="hospital", geometry=Point(85.11, 27.9)),
                                  dict(osm_id="clinic", name="Clinic", tier="clinic", geometry=Point(85.12, 27.9))], crs=4326)
    roads = gpd.GeoDataFrame({"geometry": []}, crs=4326)
    monkeypatch.setattr(dashboard, "load_osm", lambda _: SimpleNamespace(roads=roads, hospitals=hospitals, snapshot_verified=True))
    html, _ = dashboard.render_dashboard(model)
    data = data_block(html)
    assert "1 settlements are UNCERTAIN because mapped OSM roads show no baseline access to a hospital or town." in visible(html)
    assert "This is a data gap, not flood damage." in visible(html)
    assert data["layers"]["settlements"]["features"][1]["properties"]["data_gap"] is True
    assert data["layers"]["settlements"]["features"][0]["properties"]["data_gap"] is False
    for label in ("settlement CUT OFF", "settlement UNCERTAIN", "settlement CONNECTED", "data gap (no baseline access in OSM)",
                  "hospital", "clinic", "strong flood", "probable flood", "uncertain flood", "pre-existing water", "terrain excluded", "debris candidates"):
        assert label in visible(html)
    for marker in ("hospital-cross", "clinic-square", "data-gap-diamond"):
        assert f'data-marker="{marker}"' in html
        assert marker in dashboard.JS
    assert "type:'symbol'" in dashboard.JS
    assert "light dots" not in html
    assert dashboard._color("settlements", dict(status="UNCERTAIN", data_gap=True)) != dashboard._color("settlements", dict(status="UNCERTAIN", data_gap=False))
    assert 'data_gap' not in model["geo"]["settlements"]["features"][1]["properties"]
    assert "CUT OFF settlements: 1. This is not an isolation ranking." in visible(html)


def test_validation_optional_view_all_rows_trace_and_original_in_zip(tmp_path):
    directory, impact, _ = inputs(tmp_path)
    comparison_path, comparison = comparison_input(tmp_path)
    original = comparison_path.read_bytes()
    result = dashboard.build_dashboard(directory, impact, validation_json=comparison_path)
    output = Path(result["output_path"])
    html = (output / "index.html").read_text()
    model = build_report_model(directory, impact, validation_json=comparison_path)
    data = data_block(html)
    assert data["validation_supplied"]
    assert 'data-view="VALIDATION"' in html
    assert html.count("flood_mask</td>") == html.count("debris_candidate</td>") == html.count("union</td>") == 1
    assert "Post-hoc comparison, not used by the analysis." in visible(html)
    assert "proxy check, not a flood-extent validation" in visible(html)
    assert "best" not in visible(html).lower()
    assert "No validation comparison supplied" not in visible(html)
    assert numbers_trace(visible(html), model["trace_sources"])["untraced"] == []
    assert numbers_trace(visible((output / "situation_report.html").read_text()), model["trace_sources"])["untraced"] == []
    assert "<script>alert" not in html
    assert comparison_path.read_bytes() == original
    with zipfile.ZipFile(output / "incident_package.zip") as archive:
        assert archive.read("validation.json") == original
        assert json.loads(archive.read("validation.json")) == comparison
        assert "not used by the analysis" in archive.read("README.txt").decode()
    plain, _ = dashboard.render_dashboard(build_report_model(directory, impact))
    assert 'data-view="VALIDATION"' not in plain
    assert "No validation comparison supplied." in visible(plain)


def test_validation_cli_and_zero_cut_off_wording(tmp_path, capsys):
    directory, impact, _ = inputs(tmp_path, no_flood=True)
    path, _ = comparison_input(tmp_path)
    assert dashboard.main(["--incident", str(directory), "--impact-dir", str(impact), "--validation-json", str(path)]) == 0
    out = Path(json.loads(capsys.readouterr().out)["output_path"])
    for filename in ("index.html", "situation_report.html", "situation_report.md"):
        text = (out / filename).read_text()
        assert "No settlement is CUT OFF in this run." in text
        assert "CUT OFF settlements: 0." in text
        assert "Post-hoc comparison" in text
