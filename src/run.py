"""Run the existing incident, flood, impact and static-report stages in order."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

from src.analysis.flood_map import analyze_incident_flood
from src.data.incident import Incident, prepare_incident
from src.graph.impact import analyze_incident_impact
from src.report.dashboard import build_dashboard

OSM_NOTE = "OSM download can take 5-25 minutes for a new area; cached areas are instant."


class IncidentUnavailable(RuntimeError):
    """An incomplete incident cannot supply all downstream prerequisites."""

    def __init__(self, incident: Incident, reason: str):
        self.incident = incident
        super().__init__(f"Incident {incident.incident_id} is {incident.status}: {reason}")


def _require_incident(incident: Incident) -> Incident:
    reasons = [incident.pair_summary.get("reason"),
               *[f"{stage['name']}: {stage['error']}" for stage in incident.stages if stage.get("error")]]
    missing = [name for name in ("s1_stack", "terrain", "osm") if not incident.layers.get(name)]
    if missing:
        reasons.append("Required layers are missing: " + ", ".join(missing))
    if incident.status != "ready" or missing:
        raise IncidentUnavailable(incident, "; ".join(dict.fromkeys(reason for reason in reasons if reason))
                                  or "Required incident stages did not complete; inspect the incident receipt.")
    if not incident.cache_path:
        raise IncidentUnavailable(incident, "The saved incident manifest path is missing.")
    return incident


def _stage(index: int, name: str, action: Callable[[], Any],
           cached: Callable[[Any], bool] = lambda _: False) -> Any:
    """Time one public stage; report a failure before propagating it to the CLI."""
    print(f"[{index}/4] {name} ... running", flush=True)
    started = time.perf_counter()
    try:
        result = action()
        cache_hit = cached(result)
    except Exception as error:
        status = "blocked" if isinstance(error, IncidentUnavailable) and error.incident.status == "blocked" else "failed"
        print(f"[{index}/4] {name} ... {status} {time.perf_counter() - started:.1f}s", flush=True)
        raise
    print(f"[{index}/4] {name} ... done {time.perf_counter() - started:.1f}s" + (" (cached)" if cache_hit else ""), flush=True)
    return result


def _stamp(path: Path) -> tuple[int, int] | None:
    """Observe writes for cache reporting; never choose or invalidate a cache entry."""
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    """Chain public APIs; comparison input is passed only to the final report stage.

    Force is forwarded to incident preparation. Other public stages provide no
    force option: content-addressed caches remain active and reports rebuild.
    If both area arguments are supplied, the explicit bbox takes precedence.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--place")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--date", required=True, dest="flood_date")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/expA/best.pt"))
    parser.add_argument("--with-buildings", action="store_true")
    parser.add_argument("--validation-json", type=Path, help="Explicit optional comparison input for reporting only")
    parser.add_argument("--force", action="store_true", help="Refresh incident receipt; downstream stage caches remain active")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    args = parser.parse_args(argv)
    if args.place is None and args.bbox is None:
        parser.error("Provide --place or --bbox")
    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_file():
        location = os.environ.get("FLOODINTEL_WEIGHTS_URL", "").strip()
        guidance = (f"Download the checkpoint from {location} and save it at {checkpoint}." if location
                    else "See docs/TRAINING.md for checkpoint instructions, then save the checkpoint at this path.")
        print(f"Checkpoint file is missing: {checkpoint}. {guidance}", file=sys.stderr)
        return 2
    if args.place is not None and args.bbox is not None:
        print("Using --bbox for the area; --place is ignored.", flush=True)
    if args.force:
        print("Force refreshes the incident receipt; existing data-stage caches remain active. "
              "Flood and impact retain their caches; the dashboard is rebuilt.", flush=True)
    osm_noted = False

    def progress(index: int, total: int, name: str, status: str, detail: str) -> None:
        nonlocal osm_noted
        if name == "osm" and status == "running" and detail != "Checking prerequisites" and not osm_noted:
            print(OSM_NOTE, flush=True)
            osm_noted = True

    try:
        incident = _stage(1, "incident", lambda: _require_incident(prepare_incident(
            place=None if args.bbox is not None else args.place,
            bbox=tuple(args.bbox) if args.bbox is not None else None, flood_date=args.flood_date,
            cache_dir=args.cache_dir, include_buildings=args.with_buildings, progress=progress, force=args.force)),
            cached=lambda value: bool(value.stages) and all(stage.get("cache_hit") for stage in value.stages))
        # The flood API has no cache-hit field; unchanged saved output proves reuse.
        previous = {path.resolve(): _stamp(path) for path in Path(incident.cache_path).parent.glob("flood/*/flood.tif")}

        def flood_cached(value: Any) -> bool:
            if not value.cache_path:
                raise RuntimeError("Flood stage returned no saved raster path")
            path = Path(value.cache_path).resolve()
            after = _stamp(path)
            if after is None:
                raise FileNotFoundError(f"Flood raster output is missing: {path}")
            return previous.get(path) == after

        flood = _stage(2, "flood map", lambda: analyze_incident_flood(
            incident.cache_path, checkpoint, cache_dir=args.cache_dir), cached=flood_cached)
        flood_dir = Path(flood.cache_path).parent
        impact = _stage(3, "impact", lambda: analyze_incident_impact(
            incident.cache_path, flood_dir=flood_dir, cache_dir=args.cache_dir),
            cached=lambda value: bool(value.metadata.get("cache_hit")))
        dashboard = _stage(4, "dashboard", lambda: build_dashboard(
            incident.cache_path, impact_dir=impact.cache_path, flood_dir=flood_dir,
            cache_dir=args.cache_dir, validation_json=args.validation_json))
    except IncidentUnavailable as error:
        print(str(error), file=sys.stderr)
        flags = error.incident.quality.get("flags", [])
        print("Incident flags:", file=sys.stderr)
        for flag in flags:
            print(f"- {flag.get('code', 'unknown')} ({flag.get('severity', 'unknown')}): {flag.get('message', '')}", file=sys.stderr)
        if not flags:
            print("- No incident flags recorded.", file=sys.stderr)
        return 2 if error.incident.status == "blocked" else 1
    except Exception as error:
        print(f"Pipeline stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    for warning in dashboard.get("warnings", []):
        print(f"Report warning: {warning}", file=sys.stderr)
    output = Path(dashboard["output_path"])
    print(f"Incident: {incident.incident_id}")
    print(f"Dashboard: {output / 'index.html'}")
    print(f"Situation report: {output / 'situation_report.html'}")
    print(f"Text report: {output / 'situation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
