"""Small cached receipts; reporting never invokes inference or network clients."""
from copy import deepcopy
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path

import pytest
from shapely.geometry import box, mapping

from src.report.build_report import (BANNER, build_report_model, format_number, numbers_trace,
                                     render_report_html, render_report_md)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def feature(properties, geometry=None):
    return dict(type="Feature", properties=properties, geometry=geometry or dict(type="Point", coordinates=[85.1, 27.9]))


def fc(features):
    return dict(type="FeatureCollection", features=features)


def inputs(tmp_path, *, synthetic=False, no_flood=False, hostile=False, legacy=False):
    directory = tmp_path / "incidents" / "inc_123456789abc"
    impact, flood = directory / ("impact_test" if synthetic else "impact") / "cache", directory / "flood" / "cache"
    name = 'गाउँ <script>alert("x")</script> & "नेपाल"' if hostile else "Valley village"
    evidence = dict(link_id="link_bridge", tier=3, osm_ids=["way/77"], flood_pixel_area_m2=100,
                    members=[dict(fraction_in_flood=0.2, length_in_flood_m=30, damage_tier=3)])
    subject = dict(osm_id="node/11", name=name, place="village", snapped=True, status="CUT OFF", reason="Hospital access lost in likely scenario",
                   baseline_connected=dict(hospital=True, town=True, clinic=False), access_detail="none", access_by_class=dict(hospital="none", town="full"),
                   nearest_baseline=dict(hospital=dict(osm_id="node/21", name="Clinic <north>", travel_time_s=300, length_m=500, link_ids=["link_bridge"]), town=None),
                   nearest_after=dict(hospital=None, town=None), nearest_possible=dict(hospital=None),
                   detour_factors=dict(hospital=None, town=1), cut_from=["hospital"], possible_cut_from=["hospital"],
                   responsible_links=["link_bridge"], number_of_broken_routes=1, alternative_route_exists=False,
                   confidence_tier=3, evidence=[evidence], mapped_buildings_nearby=4, cluster_mapped_buildings=4, cluster_ids=["cluster_A"])
    gap = dict(osm_id="node/12", name="Baseline gap", place="hamlet", snapped=False, status="UNCERTAIN", reason="no baseline road access in OSM",
               baseline_connected=dict(hospital=False, town=False), access_detail="unmapped", responsible_links=[], evidence=[], cut_from=[], possible_cut_from=[])
    if no_flood:
        subject.update(status="CONNECTED", reason="Baseline access retained", access_detail="full", cut_from=[], possible_cut_from=[], responsible_links=[], evidence=[])
    choke = dict(link_id="link_bridge", osm_ids=["way/77"], name="Bridge नदी", is_bridge=True, damaged=True,
                 settlements_affected=[name], settlement_ids=["node/11"], count=1, mapped_buildings=4,
                 hospital_access_lost=1, hospital_access_reconnected=1, town_access_lost=0, town_access_reconnected=0,
                 damage_tier=3, evidence=evidence, sentence="Restoring mapped link reconnects 1 settlement and 1 hospital access.")
    latent = {**choke, "link_id": "link_latent", "damaged": False, "name": "Latent bridge", "sentence": "Loss of mapped bridge link disconnects 1 settlement.", "damage_tier": 0}
    zero = {**choke, "link_id": "link_zero", "count": 0, "sentence": None}
    priority = dict(id="node/11", name=name, score=174.125, rank=1, confidence="strong", components=dict(hospital=dict(value=1, contribution=50)), why=["hospital=1 contributes 50.000"])
    priorities = dict(settlements=[priority], links=[{**priority, "id": "link_bridge"}], data_gaps=[dict(id="node/12", name="Baseline gap", reason=gap["reason"])], message=None)
    if legacy:
        priorities["settlements"].append({**priority, "id": "node/12", "name": "Baseline gap", "score": 200, "rank": 1})
        priorities.pop("data_gaps")
    receipt = dict(incident_id=directory.name, created_utc="2026-08-12T12:34:56Z", status="ready",
                   aoi=dict(name="Trishuli, Nepal", flood_date="2026-08-10", bbox=[85, 27.8, 85.2, 28], area_km2=400),
                   pair_summary=dict(orbit_state="descending", relative_orbit=19, pre_date="2026-07-31", post_date="2026-08-12", days_after_flood=2),
                   layers={}, stages=[], quality=dict(flags=[], trust_summary="probable"),
                   provenance=dict(inputs=[dict(role="s1_post", source="sentinel-1-rtc", acquisition_date="2026-08-12", item_ids=["S1D"], license_note="Copernicus open access"),
                                          dict(role="osm", source="pre-event-openstreetmap", endpoint="https://example.invalid", snapshot_date="2026-08-10T00:00:00Z", snapshot_verified=True, newest_edit="2026-08-09T00:00:00Z")],
                                   allowed_sources=["sentinel-1-rtc", "cop-dem-glo-30", "pre-event-openstreetmap"]),
                   timings_total_seconds=10, cache_path=str(directory / "incident.json"))
    flags = [dict(code="osm_stale", severity="caution", message="Road edits are stale."),
             dict(code="domain_gap", severity="caution", message="Model trained on Sen1Floods11; processing domain gap."),
             dict(code="probabilities_uncalibrated", severity="info", message="Probabilities are uncalibrated.")]
    if synthetic:
        flags.append(dict(code="synthetic_flood_input", severity="limiting", message=BANNER))
    if no_flood:
        flags.append(dict(code="no_flood_detected", severity="info", message="No flood detected."))
    meta = dict(version=2, incident_id=directory.name, counts=dict(counts=dict(roads=1, bridges=1, buildings=0, hospitals=0, settlements=0),
                                                                by_tier=dict(roads={"3": 1}), road_km_by_highway=dict(primary=0.04), inundated_road_km_by_highway=dict(primary=0.03)),
                status_tallies={"CUT OFF": 0 if no_flood else 1, "UNCERTAIN": 1, "CONNECTED": 1 if no_flood else 0},
                graph_quality=dict(nodes=4, links=3, components=1, largest_component_road_length_share=1), flags=flags,
                warnings=[], trust_summary="insufficient" if synthetic else "uncertain", params=dict(damage=dict(min_overlap_m=20)),
                flood_path=None if synthetic else str(flood / "flood.tif"),
                flood_params=dict(synthetic=True, test_polygon=[85.05, 27.85, 85.15, 27.95]) if synthetic else dict(cache_key="cache"), banner=BANNER if synthetic else None)
    write(directory / "incident.json", receipt)
    write(impact / "impact_meta.json", meta)
    write(impact / "priorities.json", priorities)
    write(impact / "chokepoints.json", dict(damaged=[choke, zero], latent=[latent], damaged_no_leverage=[]))
    write(impact / "clusters.json", [dict(cluster_id="cluster_A", lost_classes=["hospital"], member_settlements=["node/11"], total_mapped_buildings=4,
                                         responsible_links=[dict(link_id="link_bridge", sufficient_to_restore=True)])])
    line = dict(type="LineString", coordinates=[[85.05, 27.85], [85.15, 27.95]])
    write(impact / "damaged_assets.geojson", fc([feature(dict(osm_id="way/77", segment_id="way/77#0", name="Bridge नदी", asset_class="bridge", tier=3, is_bridge=True,
                                                            highway="primary", length_in_flood_m=30, fraction_in_flood=0.2), line)]))
    write(impact / "settlements.geojson", fc([feature(subject), feature(gap, dict(type="Point", coordinates=[85.12, 27.91]))]))
    write(impact / "chokepoints.geojson", fc([feature(choke, line), feature(latent, line)]))
    write(impact / "routes.geojson", fc([feature(dict(settlement_id="node/11", scenario="baseline", destination_class="hospital", destination_id="node/21", travel_time_s=300, length_m=500), line)]))
    stats = dict(flood_area_km2=0 if no_flood else 0.0123, valid_area_km2=400, preexisting_water_km2=0.3,
                 terrain_excluded_km2=0.1, debris_candidate_km2=0.02, valid_fraction=0.95,
                 area_per_confidence_tier_km2=dict(strong=0.01, probable=0.002, uncertain=0.0003))
    write(flood / "flood_meta.json", dict(stats=stats, params=dict(cache_key="cache", incident_id=directory.name), flags=[], warnings=[]))
    write(flood / "flood_extent.geojson", fc([] if no_flood else [feature(dict(area_km2=0.0123, max_confidence=3, median_confidence=2), mapping(box(85.08, 27.88, 85.11, 27.91)))]))
    write(flood / "debris_candidates.geojson", fc([]))
    return directory, impact, flood


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def visible(html):
    parser = VisibleText()
    parser.feed(html)
    return " ".join(parser.parts)


def test_facts_and_numbers_trace(tmp_path):
    directory, impact, _ = inputs(tmp_path)
    model = build_report_model(directory, impact)
    assert model["flood"]["stats"]["flood_area_km2"] == 0.0123
    assert "roads 1" in model["report"]["findings"][3]
    assert model["priorities"]["settlements"][0]["id"] == "node/11"
    md, html = render_report_md(model), render_report_html(model)
    assert numbers_trace(unescape(md), model["trace_sources"])["untraced"] == []
    assert numbers_trace(visible(html), model["trace_sources"])["untraced"] == []
    assert numbers_trace("999999999999", model["trace_sources"])["untraced"]
    assert len(model["report"]["findings"]) == 8
    assert "@page{size:A4" in html
    assert [section for section in model["report"]["limitations"] if "Road edits" in section]


def test_formatter_missing_values_and_percent():
    assert format_number(0.0123456789) == "0.0123"
    assert format_number(0.125, percent=True) == "12.5%"
    assert format_number(float("nan")) == "not available"
    assert format_number(None) == "not available"


def test_legacy_gaps_and_zero_leverage_not_ranked(tmp_path):
    directory, impact, _ = inputs(tmp_path, legacy=True)
    model = build_report_model(directory, impact)
    assert [row["id"] for row in model["priorities"]["settlements"]] == ["node/11"]
    assert [row["id"] for row in model["priorities"]["data_gaps"]] == ["node/12"]
    assert [row["link_id"] for row in model["chokepoints"]["damaged"]] == ["link_bridge"]
    assert model["chokepoints"]["damaged_no_leverage"][0]["link_id"] == "link_zero"


def test_no_flood_keeps_baseline_latent_and_trust(tmp_path):
    directory, impact, _ = inputs(tmp_path, no_flood=True)
    model = build_report_model(directory, impact)
    assert model["no_flood"]
    assert not model["priorities"]["settlements"]
    assert "No flood detected" in render_report_md(model)
    assert "No flood-attributable priorities" in render_report_md(model)
    assert model["chokepoints"]["latent"]


def test_synthetic_never_borrows_real_statistics(tmp_path):
    directory, impact, flood = inputs(tmp_path, synthetic=True)
    model = build_report_model(directory, impact)
    assert model["flood"] == {}
    assert BANNER in render_report_html(model)
    assert BANNER in render_report_md(model)
    assert "not available" in model["report"]["findings"][0]
    with pytest.raises(ValueError, match="Synthetic"):
        build_report_model(directory, impact, flood)


def test_missing_mismatched_receipts(tmp_path):
    directory, impact, flood = inputs(tmp_path)
    with pytest.raises(FileNotFoundError, match="Impact directory"):
        build_report_model(directory, tmp_path / "missing")
    meta = json.loads((flood / "flood_meta.json").read_text())
    meta["params"]["cache_key"] = "different"
    write(flood / "flood_meta.json", meta)
    with pytest.raises(ValueError, match="does not match"):
        build_report_model(directory, impact, flood)


def test_escaped_unicode_and_no_outcome_claims(tmp_path):
    directory, impact, _ = inputs(tmp_path, hostile=True)
    model = build_report_model(directory, impact)
    for text in (render_report_md(model), render_report_html(model)):
        assert "गाउँ" in text
        assert "<script>alert" not in text
        assert "&lt;script&gt;" in text
        assert not any(word in text.lower() for word in ("destroyed", "casualt", "death", "population"))


def comparison_input(tmp_path):
    path = tmp_path / "provided_comparison.json"
    comparison = dict(incident_id="inc_123456789abc", params=dict(tolerance_m=50),
                      domain=dict(area_km2=123.456789, our_flood_outside_share=0.234567),
                      reference_event_types=["Debris flow", "Observed event"],
                      warnings=['Acquisition date unknown; गाउँ <script>alert("x")</script>.'], variants={})
    for index, name in enumerate(("flood_mask", "debris_candidate", "union"), 1):
        comparison["variants"][name] = dict(metrics=dict(iou=index * 0.11345, precision=index * 0.22456,
                                                         recall=index * 0.31234, f1=index * 0.25678),
                                              areas_km2=dict(ours=index * 0.45678, reference=1.23456, overlap=index * 0.12345),
                                              tolerance_metrics=dict(precision_tol=index * 0.24567, recall_tol=index * 0.34567))
    write(path, comparison)
    return path, comparison


def test_gap_clarity_and_priorities_note_follow_inputs(tmp_path):
    directory, impact, _ = inputs(tmp_path)
    model = build_report_model(directory, impact)
    note = "1 settlements are UNCERTAIN because mapped OSM roads show no baseline access to a hospital or town. This is a data gap, not flood damage."
    assert model["report"]["data_gap_note"] == note
    for rendered in (render_report_md(model), visible(render_report_html(model))):
        assert note in rendered
        assert "Ranked from flood-evidence and OSM-derived access scores. CUT OFF settlements: 1. This is not an isolation ranking." in rendered
        assert "No settlement is CUT OFF" not in rendered
    priorities = json.loads((impact / "priorities.json").read_text())
    priorities["data_gaps"] = []
    write(impact / "priorities.json", priorities)
    settlements = json.loads((impact / "settlements.geojson").read_text())
    settlements["features"] = settlements["features"][:1]
    write(impact / "settlements.geojson", settlements)
    assert build_report_model(directory, impact)["report"]["data_gap_note"] is None


def test_no_cut_off_note_and_comparison_absent_are_plain(tmp_path):
    directory, impact, _ = inputs(tmp_path, no_flood=True)
    model = build_report_model(directory, impact)
    for rendered in (render_report_md(model), visible(render_report_html(model))):
        assert "CUT OFF settlements: 0." in rendered
        assert "No settlement is CUT OFF in this run." in rendered
        assert "No validation comparison supplied." in rendered


def test_comparison_shared_table_all_rows_proxy_note_and_numbers_trace(tmp_path):
    directory, impact, _ = inputs(tmp_path)
    path, comparison = comparison_input(tmp_path)
    model = build_report_model(directory, impact, validation_json=path)
    assert model["validation"] == comparison
    assert comparison in model["trace_sources"]
    assert [row[0] for row in model["report"]["comparison"]["rows"]] == ["flood_mask", "debris_candidate", "union"]
    html, md = render_report_html(model), render_report_md(model)
    assert html.count("<table>") == 1
    assert "Tolerance precision" in html and "Tolerance recall" in md
    for text in (md, visible(html)):
        assert "Post-hoc comparison, not used by the analysis." in text
        assert "all rows are shown" in text
        assert "These reference polygons are labelled Debris flow, Observed event; this is a proxy check, not a flood-extent validation." in text
        assert "123.4568" in text and "23.4567%" in text
        assert "Acquisition date unknown" in text
        assert numbers_trace(unescape(text), model["trace_sources"])["untraced"] == []
        assert "best" not in text.lower()
    assert "<script>alert" not in html
    assert "गाउँ" in html
    assert model["report"]["findings"] == build_report_model(directory, impact)["report"]["findings"]
    with pytest.raises(FileNotFoundError, match="Required report input"):
        build_report_model(directory, impact, validation_json=tmp_path / "missing.json")


def test_gap_count_cardinality_is_traceable_without_matching_numeric_scalar(tmp_path):
    directory, impact, _ = inputs(tmp_path)
    priorities = json.loads((impact / "priorities.json").read_text())
    priorities["data_gaps"] = [dict(id=f"gap_{letter}", name=letter, reason="No baseline route") for letter in "abcdefg"]
    write(impact / "priorities.json", priorities)
    model = build_report_model(directory, impact)
    assert model["report"]["data_gap_note"].startswith("8 settlements")  # seven listed plus one missing from the list
    assert numbers_trace(render_report_md(model), model["trace_sources"])["untraced"] == []
