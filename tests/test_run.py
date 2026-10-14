"""Offline runner contracts with four fake stages and tiny local output files."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import run


def stages(tmp_path, monkeypatch, *, status="ready", cached=False):
    root = tmp_path / "cache"
    directory = root / "incidents" / "inc_123456789abc"
    directory.mkdir(parents=True)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake checkpoint; never loaded")
    incident = SimpleNamespace(incident_id=directory.name, cache_path=str(directory / "incident.json"), status=status,
                               layers={"s1_stack": "stack.tif", "terrain": "terrain.tif", "osm": "osm.gpkg"},
                               pair_summary={}, stages=[dict(name="osm", status="done", error=None, cache_hit=cached)],
                               quality={"flags": []})
    if status == "blocked":
        incident.pair_summary = {"reason": "No same-orbit pair exists for these observation windows"}
        incident.quality["flags"] = [dict(code="no_same_orbit_pair", severity="limiting", message="No usable pair")]
    flood_path = directory / "flood" / "hash" / "flood.tif"
    if cached:
        flood_path.parent.mkdir(parents=True)
        flood_path.write_bytes(b"saved flood")
    impact_path = directory / "impact" / "hash"
    report_path = directory / "report"
    calls = []

    def prepare(**kwargs):
        calls.append(("incident", kwargs))
        kwargs["progress"](5, 5, "osm", "running", "Starting")
        kwargs["progress"](5, 5, "osm", "done", "Finished")
        return incident

    def flood(*args, **kwargs):
        calls.append(("flood", dict(args=args, **kwargs)))
        if not cached:
            flood_path.parent.mkdir(parents=True)
            flood_path.write_bytes(b"saved flood")
        return SimpleNamespace(cache_path=str(flood_path))

    def impact(*args, **kwargs):
        calls.append(("impact", dict(args=args, **kwargs)))
        impact_path.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(cache_path=str(impact_path), metadata={"cache_hit": cached})

    def dashboard(*args, **kwargs):
        calls.append(("dashboard", dict(args=args, **kwargs)))
        report_path.mkdir()
        for name in ("index.html", "situation_report.html", "situation_report.md"):
            (report_path / name).write_text("fake static report")
        return dict(output_path=str(report_path), warnings=[])

    for name, function in (("prepare_incident", prepare), ("analyze_incident_flood", flood),
                           ("analyze_incident_impact", impact), ("build_dashboard", dashboard)):
        monkeypatch.setattr(run, name, function)
    argv = ["--place", "Trishuli, Nepal", "--date", "2026-08-26", "--checkpoint", str(checkpoint), "--cache-dir", str(root)]
    return argv, calls, incident, flood_path, report_path


def test_happy_path_order_paths_timings_and_osm_note(tmp_path, monkeypatch, capsys):
    argv, calls, incident, flood, report = stages(tmp_path, monkeypatch)
    assert run.main(argv) == 0
    assert [name for name, _ in calls] == ["incident", "flood", "impact", "dashboard"]
    assert calls[0][1]["include_buildings"] is False
    assert calls[1][1]["args"][0] == incident.cache_path
    assert calls[2][1]["flood_dir"] == flood.parent
    assert calls[3][1]["flood_dir"] == flood.parent
    assert calls[3][1]["impact_dir"] == str(report.parent / "impact" / "hash")
    output = capsys.readouterr().out
    for index, name in enumerate(("incident", "flood map", "impact", "dashboard"), 1):
        assert f"[{index}/4] {name} ... done" in output
    assert output.count(run.OSM_NOTE) == 1
    assert output.index(run.OSM_NOTE) < output.index("[1/4] incident ... done")
    assert str(report / "index.html") in output
    assert str(report / "situation_report.html") in output
    assert "(cached)" not in output


def test_blocked_incident_reason_flags_and_no_downstream_calls(tmp_path, monkeypatch, capsys):
    argv, calls, _, _, _ = stages(tmp_path, monkeypatch, status="blocked")
    assert run.main(argv) == 2
    assert [name for name, _ in calls] == ["incident"]
    output = capsys.readouterr()
    assert "No same-orbit pair" in output.err
    assert "no_same_orbit_pair (limiting)" in output.err
    assert "[1/4] incident ... blocked" in output.out
    assert "[2/4]" not in output.out


@pytest.mark.parametrize("status", ["failed", "partial"])
def test_incomplete_incident_and_missing_layers_stop_early(tmp_path, monkeypatch, capsys, status):
    argv, calls, incident, _, _ = stages(tmp_path, monkeypatch, status=status)
    incident.stages = [dict(name="terrain", status="failed", error="ValueError: missing DEM", cache_hit=False)]
    incident.layers.pop("terrain")
    assert run.main(argv) == 1
    assert [name for name, _ in calls] == ["incident"]
    assert "missing DEM" in capsys.readouterr().err


def test_ready_receipt_with_missing_required_layer_is_not_success(tmp_path, monkeypatch, capsys):
    argv, calls, incident, _, _ = stages(tmp_path, monkeypatch)
    incident.layers.pop("osm")
    assert run.main(argv) == 1
    assert [name for name, _ in calls] == ["incident"]
    assert "Required layers are missing: osm" in capsys.readouterr().err


@pytest.mark.parametrize("url", [None, "https://example.invalid/weights/best.pt"])
def test_missing_checkpoint_preflight_is_clear_without_stage_calls(tmp_path, monkeypatch, capsys, url):
    if url:
        monkeypatch.setenv("FLOODINTEL_WEIGHTS_URL", url)
    else:
        monkeypatch.delenv("FLOODINTEL_WEIGHTS_URL", raising=False)
    forbidden = Mock(side_effect=AssertionError("must stop before incident preparation"))
    monkeypatch.setattr(run, "prepare_incident", forbidden)
    checkpoint = tmp_path / "missing.pt"
    assert run.main(["--place", "Trishuli, Nepal", "--date", "2026-08-26", "--checkpoint", str(checkpoint)]) == 2
    error = capsys.readouterr().err
    assert str(checkpoint) in error
    assert (url or "docs/TRAINING.md") in error
    forbidden.assert_not_called()


def test_explicit_comparison_passes_only_to_dashboard_without_runner_read(tmp_path, monkeypatch):
    argv, calls, _, _, _ = stages(tmp_path, monkeypatch)
    path = tmp_path / "operator_selected.json"
    # The runner must not parse or search for comparison input itself.
    path.write_text("not parsed by runner")
    assert run.main([*argv, "--validation-json", str(path), "--with-buildings", "--force"]) == 0
    assert calls[0][1]["include_buildings"] is True
    assert calls[0][1]["force"] is True
    for _, kwargs in calls[:3]:
        assert "validation_json" not in kwargs
    for _, kwargs in calls[1:]:
        assert "force" not in kwargs
    assert calls[3][1]["validation_json"] == path


def test_cache_labels_observe_reuse_and_reports_are_rebuilt(tmp_path, monkeypatch, capsys):
    argv, _, _, _, _ = stages(tmp_path, monkeypatch, cached=True)
    assert run.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len([line for line in lines if "(cached)" in line]) == 3
    assert "(cached)" not in next(line for line in lines if "[4/4] dashboard ... done" in line)


@pytest.mark.parametrize("function,previous", [("prepare_incident", []), ("analyze_incident_flood", ["incident"]),
                                                ("analyze_incident_impact", ["incident", "flood"]),
                                                ("build_dashboard", ["incident", "flood", "impact"])])
def test_stage_exceptions_return_nonzero_and_stop_later_calls(tmp_path, monkeypatch, capsys, function, previous):
    argv, calls, _, _, _ = stages(tmp_path, monkeypatch)
    monkeypatch.setattr(run, function, Mock(side_effect=RuntimeError("stage unavailable")))
    assert run.main(argv) == 1
    assert [name for name, _ in calls] == previous
    output = capsys.readouterr()
    assert "RuntimeError: stage unavailable" in output.err
    assert " ... failed " in output.out
    assert "Dashboard:" not in output.out


def test_bbox_takes_precedence_without_geocoding(tmp_path, monkeypatch, capsys):
    argv, calls, _, _, _ = stages(tmp_path, monkeypatch)
    assert run.main([*argv, "--bbox", "84.99", "27.79", "85.30", "28.06"]) == 0
    assert calls[0][1]["place"] is None
    assert calls[0][1]["bbox"] == (84.99, 27.79, 85.30, 28.06)
    assert "--place is ignored" in capsys.readouterr().out


def test_argument_errors_exit_nonzero_without_running_stages(monkeypatch):
    forbidden = Mock(side_effect=AssertionError("no stage calls"))
    monkeypatch.setattr(run, "prepare_incident", forbidden)
    for argv in (["--date", "2026-08-26"], ["--place", "Trishuli, Nepal"],
                 ["--date", "2026-08-26", "--bbox", "0", "1"]):
        with pytest.raises(SystemExit) as error:
            run.main(argv)
        assert error.value.code == 2
    forbidden.assert_not_called()
