"""Assemble a resumable incident receipt from the existing data stages."""

import argparse
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import time as clock
from typing import Any, Callable, Literal

from src.data.aoi import AOI, BBox, build_aoi
from src.data.s1_search import PairSearchResult, find_best_pair
from src.data.s1_preprocess import S1Stack, load_stack, preprocess_pair
from src.data.terrain import TerrainStack, build_terrain, load_terrain
from src.data.osm import LAYER_COLUMNS, OSMBundle, build_osm, load_osm


STAGES = ("aoi", "s1_search", "preprocess", "terrain", "osm")
ALLOWED_SOURCES = (
    "sentinel-1-rtc", "sentinel-1-grd", "sentinel-2", "cop-dem-glo-30",
    "pre-event-openstreetmap", "permitted-training-datasets",
)
# Training inputs remain disabled until specific permitted datasets are registered.
PERMITTED_TRAINING_DATASETS: tuple[str, ...] = ()
DISCLAIMER = "Educational prototype. Not operational emergency infrastructure."
Progress = Callable[[int, int, str, str, str], None]


def _default(function: Callable, name: str) -> Any:
    return inspect.signature(function).parameters[name].default


PARAMETERS = {
    "resolution": _default(preprocess_pair, "resolution"),
    "filter_window": _default(preprocess_pair, "filter_window"),
    "filter": "simple_lee", "db_clip": [-35, 5],
    "buffer_km": _default(build_terrain, "buffer_km"),
    "hydro_resolution": _default(build_terrain, "hydro_resolution"),
    "stream_threshold_km2": _default(build_terrain, "stream_threshold_km2"),
    "context_buffer_km": _default(build_osm, "context_buffer_km"),
    "snapshot_offset_days": _default(build_osm, "snapshot_offset_days"),
    "strict": True,
    "scene_collection": _default(find_best_pair, "collection"),
    "widen_post_days": _default(find_best_pair, "widen_post_days"),
    "pre_days": _default(build_aoi, "pre_days"),
    "post_days": _default(build_aoi, "post_days"),
    "max_side_km": _default(build_aoi, "max_side_km"),
}


class ProvenanceError(ValueError):
    """An incident contains inputs outside the permitted source/date policy."""


@dataclass
class Incident:
    """Serializable run state and paths; raster/vector data remain in stage caches."""

    incident_id: str
    created_utc: str
    status: Literal["ready", "blocked", "partial", "failed"]
    aoi: dict[str, Any]
    pair_summary: dict[str, Any]
    layers: dict[str, str]
    stages: list[dict[str, Any]]
    quality: dict[str, Any]
    provenance: dict[str, Any]
    timings_total_seconds: float
    cache_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible receipt."""
        return deepcopy(vars(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Incident":
        """Restore a receipt without loading any layers."""
        return cls(**deepcopy(data))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _datetime(value: str) -> datetime:
    if len(value) == 10:
        return datetime.combine(date.fromisoformat(value), time(), timezone.utc)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ProvenanceError("Snapshot timestamps must specify a timezone")
    return parsed.astimezone(timezone.utc)


def assert_provenance_allowed(incident: Incident) -> None:
    """Check the fixed allowlist and the UTC-midnight pre-event OSM cutoff."""
    for entry in incident.provenance["inputs"]:
        source = entry.get("source")
        if source not in ALLOWED_SOURCES:
            raise ProvenanceError(f"Disallowed input source: {source!r}")
        if source == "permitted-training-datasets" and entry.get("collection") not in PERMITTED_TRAINING_DATASETS:
            raise ProvenanceError("Training dataset is not explicitly registered as permitted")
        if source == "pre-event-openstreetmap":
            if entry.get("snapshot_verified") is not True:
                raise ProvenanceError("OSM snapshot_verified is false; pre-event input cannot be verified")
            try:
                snapshot = _datetime(entry["snapshot_date"])
                cutoff = _datetime(incident.aoi["flood_date"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProvenanceError(f"Invalid OSM snapshot/flood date: {exc}") from exc
            if snapshot > cutoff:
                raise ProvenanceError("OSM snapshot_date is later than the flood-date UTC-midnight cutoff")
    if incident.status == "ready":
        roles = {entry.get("role") for entry in incident.provenance["inputs"]}
        missing = {"s1_pre", "s1_post", "dem", "osm"} - roles
        if missing:
            raise ProvenanceError(f"Incomplete provenance receipt: missing {', '.join(sorted(missing))}")


def trust_summary(flags: list[dict[str, str]], status: str) -> str:
    """Coarse run trust: limiting -> insufficient if blocked, otherwise uncertain.

    Without limiting flags, two or more caution flags mean uncertain, one means
    probable, and none means strong. Informational flags do not lower trust.
    This label describes the run, not the confidence of individual results.
    """
    if any(flag["severity"] == "limiting" for flag in flags):
        return "insufficient" if status == "blocked" else "uncertain"
    cautions = sum(flag["severity"] == "caution" for flag in flags)
    return "uncertain" if cautions >= 2 else "probable" if cautions == 1 else "strong"


def _pair_summary(result: PairSearchResult, aoi: AOI) -> dict[str, Any]:
    if result.status != "ok" or result.pair is None:
        return {"status": result.status, "reason": result.reason}
    pair = result.pair
    # Step 5 exposes widening only as this explicit receipt; arbitrary text is not classified.
    receipt = re.compile(
        r"Post window was widened from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2}) "
        r"for one retry \(capped at today\); pre window unchanged\."
    )
    widened = pair.post.acquisition_date > aoi.post_window[1]
    for warning in result.warnings:
        match = receipt.fullmatch(warning)
        if match:
            widened |= date.fromisoformat(match[2]) > date.fromisoformat(match[1])
    return {
        "orbit_state": pair.orbit_state, "relative_orbit": pair.relative_orbit,
        "pre_date": pair.pre.acquisition_date.isoformat(),
        "post_date": pair.post.acquisition_date.isoformat(),
        "days_before_flood": pair.days_before_flood, "days_after_flood": pair.days_after_flood,
        "coverage": pair.min_coverage, "pre_coverage": pair.pre.coverage,
        "post_coverage": pair.post.coverage, "widened_post_window": widened,
    }


def _quality(
    incident: Incident, outputs: dict[str, dict], warnings: dict[str, list[str]],
    include_buildings: bool,
) -> dict[str, Any]:
    """Generate flags from typed metadata, numeric thresholds, and stage states."""
    flags: list[dict[str, str]] = []

    def flag(code: str, stage: str, severity: str, message: str) -> None:
        flags.append({"code": code, "stage": stage, "severity": severity, "message": message})

    pair = incident.pair_summary
    if pair.get("status") in {"no_same_orbit_pair", "no_scenes"}:
        flag("no_same_orbit_pair", "s1_search", "limiting", pair["reason"])
    if pair.get("widened_post_window"):
        flag("widened_post_window", "s1_search", "caution", "The post-event search window was extended.")
    if pair.get("days_after_flood", 0) > 6:
        flag("acquisition_gap_long", "s1_search", "caution", f"Post acquisition is {pair['days_after_flood']} days after the flood.")
    if "days_after_flood" in pair and pair["days_after_flood"] == 0:
        flag("post_on_flood_date", "s1_search", "caution", "Post acquisition is on the flood date and may precede the event.")
    s1 = outputs.get("preprocess")
    if s1:
        fraction = s1["valid_fraction"]
        if fraction < 0.9:
            flag("s1_low_valid_fraction", "preprocess", "limiting" if fraction < 0.6 else "caution",
                 f"Common S1 valid fraction is {fraction:.4f}.")
        if s1["source"] == "grd_fallback":
            flag("s1_grd_fallback", "preprocess", "limiting", "GRD fallback has no radiometric/terrain correction.")
    terrain = outputs.get("terrain")
    if terrain:
        if terrain["source"] == "cop-dem-glo-30":
            flag("dem_surface_model", "terrain", "info", "Copernicus DEM includes buildings and canopy.")
            if s1 and s1["resolution"] < 30:
                flag("dem_resampled_30m", "terrain", "info", "Native 30 m DEM is resampled to the finer S1 grid; no detail is added.")
        fraction = terrain["hand_nan_fraction"]
        if fraction > 0.02:
            flag("hand_nan_present", "terrain", "caution", f"HAND is unavailable for {fraction:.2%} of grid pixels.")
    osm = outputs.get("osm")
    if osm:
        quality = osm["quality"]
        if quality["road_density_km_per_km2"] < 0.3:
            flag("osm_sparse_network", "osm", "caution", f"Road density is {quality['road_density_km_per_km2']:.3f} km/km2.")
        if quality["hospital_count"] == 0:
            flag("osm_no_hospitals", "osm", "limiting", "No hospitals/clinics are mapped in the AOI; isolation analysis is limited.")
        if quality["settlement_count"] == 0:
            flag("osm_no_settlements", "osm", "limiting", "No settlements are mapped in the AOI.")
        stale = quality["roads_edited_over_5_years_share"]
        if stale is not None and stale > 0.5:
            flag("osm_stale", "osm", "caution", f"{stale:.1%} of road ways were last edited more than five years before the snapshot.")
        if osm["snapshot_verified"] is not True:
            flag("osm_snapshot_unverified", "osm", "limiting", "The pre-event OSM snapshot is unverified.")
    if not include_buildings:
        flag("buildings_deferred", "osm", "info", "Building footprints are deferred until a later local flood-mask pull.")
    for stage in incident.stages:
        if stage["status"] == "failed":
            flag("stage_failed", stage["name"], "limiting", stage["error"])
    merged, seen = [], set()
    for stage in STAGES:
        for message in warnings.get(stage, []):
            if message not in seen:
                seen.add(message)
                merged.append(f"{stage}: {message}")
    return {"warnings": merged, "flags": flags, "trust_summary": trust_summary(flags, incident.status)}


def _path(incident: Incident, value: str) -> Path:
    path = Path(value)
    base = Path(incident.cache_path).parent if incident.cache_path else Path("data/cache/incidents") / incident.incident_id
    return path if path.is_absolute() else base / path


def _check_layers(incident: Incident) -> None:
    for name, value in incident.layers.items():
        path = _path(incident, value)
        if not path.exists():
            raise FileNotFoundError(f"Incident {incident.incident_id}: {name} cache file is missing (cache was deleted): {path}")


def _layer_paths(output: Any, key: str) -> dict[str, str]:
    if not output.cache_path:
        raise ValueError(f"{key} returned no on-disk cache_path")
    path = Path(output.cache_path).resolve()
    metadata = path.with_suffix(".json")
    if not metadata.is_file():
        raise FileNotFoundError(f"{key} metadata cache is missing: {metadata}")
    paths = {key: str(path), f"{key}_metadata": str(metadata)}
    if key == "osm":
        saved = json.loads(metadata.read_text())
        if saved.get("storage_format") == "geojson":
            paths[key] = str(path.parent)
            paths.update({f"osm_{name}": str(path.parent / f"{name}.geojson") for name in LAYER_COLUMNS})
        elif not path.is_file():
            raise FileNotFoundError(f"OSM cache is missing: {path}")
    elif not path.is_file():
        raise FileNotFoundError(f"{key} cache is missing: {path}")
    preview = path.with_name("quicklook.png")
    if preview.exists():
        paths[f"{key}_preview"] = str(preview)
    for value in paths.values():
        if not Path(value).exists():
            raise FileNotFoundError(f"{key} cache file is missing: {value}")
    return paths


@dataclass
class IncidentLayers:
    """Lazy S1, terrain and OSM readers; each property checks its cache files."""

    incident: Incident
    _loaded: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def _load(self, key: str, loader: Callable) -> Any:
        if key not in self.incident.layers:
            raise FileNotFoundError(f"Incident {self.incident.incident_id} has no {key} layer; check its stage status")
        stage_name = "preprocess" if key == "s1_stack" else key
        stage = next((stage for stage in self.incident.stages if stage["name"] == stage_name), None)
        if stage is None or stage["status"] != "done":
            raise ProvenanceError(f"Cannot load {key} from an unsuccessful {stage_name} stage")
        roles = {"s1_stack": {"s1_pre", "s1_post"}, "terrain": {"dem"}, "osm": {"osm"}}[key]
        receipt = {**self.incident.provenance,
                   "inputs": [entry for entry in self.incident.provenance["inputs"] if entry["role"] in roles]}
        assert_provenance_allowed(replace(self.incident, status="partial", provenance=receipt))
        for name, value in self.incident.layers.items():
            if name == key or name.startswith(key + "_"):
                path = _path(self.incident, value)
                if not path.exists():
                    raise FileNotFoundError(f"{name} cache file is missing (cache was deleted): {path}")
        if key not in self._loaded:
            loaded = loader(_path(self.incident, self.incident.layers[key]))
            actual = deepcopy(receipt)
            if key == "osm":
                for entry in actual["inputs"]:
                    entry.update(snapshot_date=loaded.snapshot_date, snapshot_verified=loaded.snapshot_verified)
            else:
                source = ({"rtc": "sentinel-1-rtc", "grd_fallback": "sentinel-1-grd"}.get(loaded.source, loaded.source)
                          if key == "s1_stack" else loaded.source)
                for entry in actual["inputs"]:
                    entry["source"] = source
            assert_provenance_allowed(replace(self.incident, status="partial", provenance=actual))
            self._loaded[key] = loaded
        return self._loaded[key]

    @property
    def s1_stack(self) -> S1Stack:
        return self._load("s1_stack", load_stack)

    @property
    def terrain(self) -> TerrainStack:
        return self._load("terrain", load_terrain)

    @property
    def osm(self) -> OSMBundle:
        return self._load("osm", load_osm)


def load_layers(incident: Incident) -> IncidentLayers:
    """Check referenced paths now, loading data only on property access."""
    _check_layers(incident)
    return IncidentLayers(incident)


def _summary(incident: Incident) -> str:
    lines = [
        f"# Incident {incident.incident_id}", "",
        f"Status: {incident.status}; trust: {incident.quality['trust_summary']}",
        f"Flood date: {incident.aoi.get('flood_date', 'unavailable')}",
        f"BBox: {incident.aoi.get('bbox', 'unavailable')}",
        f"Pair: {json.dumps(incident.pair_summary, ensure_ascii=False)}",
        f"Total seconds: {incident.timings_total_seconds:.3f}",
        f"Cache: {incident.cache_path}", "", "Stages:",
    ]
    for stage in incident.stages:
        lines.append(f"- {stage['name']}: {stage['status']}, {stage['seconds']:.3f}s, cache_hit={stage['cache_hit']}"
                     + (f"; {stage['error']}" if stage['error'] else ""))
    lines.extend(["", "Flags:"])
    lines.extend(f"- {flag['code']} ({flag['stage']}): {flag['severity']}" for flag in incident.quality["flags"])
    lines.extend(["", DISCLAIMER, ""])
    return "\n".join(lines)


def save_incident(incident: Incident, cache_dir: str | Path = "data/cache") -> None:
    """Atomically save the JSON receipt and a concise human-readable summary."""
    path = Path(incident.cache_path) if incident.cache_path else Path(cache_dir) / "incidents" / incident.incident_id / "incident.json"
    path = path.resolve()
    incident.cache_path = str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(incident.to_dict(), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    path.with_name("summary.md").write_text(_summary(incident), encoding="utf-8")


def load_incident(incident_id_or_path: str | Path, cache_dir: str | Path = "data/cache") -> Incident:
    """Load by id, manifest path or incident directory without loading layers."""
    value = str(incident_id_or_path)
    if re.fullmatch(r"inc_[a-f0-9]{12}", value):
        path = Path(cache_dir) / "incidents" / value / "incident.json"
    else:
        path = Path(value)
        if path.is_dir() or path.suffix != ".json":
            path = path / "incident.json"
    if not path.is_file():
        raise FileNotFoundError(f"Incident manifest is missing (cache was deleted): {path}")
    incident = Incident.from_dict(json.loads(path.read_text(encoding="utf-8")))
    incident.cache_path = str(path.resolve())
    return incident


def _id(aoi: dict, parameters: dict, include_buildings: bool, request: dict) -> str:
    bbox = [round(float(value), 5) or 0.0 for value in aoi["bbox"]] if aoi else None
    # Equivalent numeric defaults (10 and 10.0) must identify the same processing.
    parameters = {name: int(value) if isinstance(value, float) and value.is_integer() else value
                  for name, value in parameters.items()}
    inputs = {"bbox": bbox, "flood_date": aoi.get("flood_date", request["flood_date"]),
              "include_buildings": include_buildings, "parameters": parameters}
    if not aoi:
        inputs["unresolved_request"] = request
    return "inc_" + hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12]


def _notify(progress: Progress | None, index: int, status: str, detail: str) -> None:
    if progress is not None:
        try:
            progress(index + 1, len(STAGES), STAGES[index], status, detail)
        except Exception:
            pass


def _fingerprint(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def prepare_incident(
    place: str | None = None, bbox: BBox | None = None, flood_date: str | date = ...,
    cache_dir: str | Path = "data/cache", include_buildings: bool = False,
    progress: Progress | None = None, force: bool = False,
) -> Incident:
    """Run five stages; incomplete runs retry while complete receipts return fast.

    Force bypasses the incident receipt. It passes force=True only to stages
    exposing that parameter. Current stages expose no such option: unsupported
    cache bypass is skipped, their own caches remain active, and warnings say so.
    """
    started = clock.perf_counter()
    root = Path(cache_dir).resolve()
    request = json.loads(json.dumps({"place": place, "bbox": bbox, "flood_date": flood_date,
                                    "include_buildings": include_buildings,
                                    "parameters": PARAMETERS}, default=str))
    if not force:
        for path in sorted((root / "incidents").glob("inc_*/incident.json")):
            try:
                cached = load_incident(path)
                if cached.provenance.get("request") != request or cached.status != "ready":
                    continue
                if not all(key in cached.layers for key in ("s1_stack", "terrain", "osm")):
                    continue
                if len(cached.stages) != 5 or any(stage["status"] != "done" for stage in cached.stages):
                    continue
                _check_layers(cached)
                assert_provenance_allowed(cached)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            for index, stage in enumerate(cached.stages):
                _notify(progress, index, "running", "Checking saved incident")
                stage.update(cache_hit=True, seconds=0.0)
                _notify(progress, index, "cached", "0.0s (cached incident)")
            cached.timings_total_seconds = clock.perf_counter() - started
            try:
                save_incident(cached)
            except Exception as exc:
                cached.status = "failed"
                cached.quality["warnings"].append(f"incident: {type(exc).__name__}: {exc}")
                cached.quality["flags"].append({"code": "incident_save_failed", "stage": "incident",
                                               "severity": "limiting", "message": str(exc)})
                cached.quality["trust_summary"] = trust_summary(cached.quality["flags"], cached.status)
            return cached
    incident = Incident(
        incident_id="", created_utc=_utc(), status="failed", aoi={}, pair_summary={},
        layers={}, stages=[], quality={}, timings_total_seconds=0.0,
        provenance={"inputs": [], "allowed_sources": list(ALLOWED_SOURCES),
                    "disclaimer": DISCLAIMER, "request": request,
                    "parameters": deepcopy(PARAMETERS)},
    )
    outputs: dict[str, dict] = {}
    warnings: dict[str, list[str]] = {name: [] for name in STAGES}
    before = {}
    for pattern in ("s1/*/stack.tif", "terrain/*/terrain.tif", "osm/*/osm.json"):
        for path in root.glob(pattern):
            before[str(path)] = _fingerprint(path)
            before[str(path.with_suffix(".json"))] = _fingerprint(path.with_suffix(".json"))

    def call(index: int, function: Callable, **kwargs: Any) -> Any:
        if force:
            if "force" in inspect.signature(function).parameters:
                kwargs["force"] = True
            else:
                warnings[STAGES[index]].append(
                    f"force cache bypass unsupported by {STAGES[index]} API; rerunning with its existing cache policy."
                )
        return function(**kwargs)

    def run(index: int, action: Callable, accept: Callable) -> Any:
        name = STAGES[index]
        _notify(progress, index, "running", "Starting")
        stage = {"name": name, "status": "failed", "seconds": 0.0, "cache_hit": False, "error": None}
        then = clock.perf_counter()
        value = None
        try:
            value = action()
            stage["status"] = accept(value) or "done"
            if stage["status"] == "blocked":
                stage["error"] = value.reason
            else:
                layer = {2: "s1_stack", 3: "terrain", 4: "osm"}.get(index)
                if layer:
                    path = Path(incident.layers[layer])
                    primary = Path(incident.layers[f"{layer}_metadata"]) if layer == "osm" else path
                    metadata = Path(incident.layers[f"{layer}_metadata"])
                    stage["cache_hit"] = all(
                        before.get(str(file)) is not None and before[str(file)] == _fingerprint(file)
                        for file in (primary, metadata)
                    )
        except Exception as exc:
            stage["status"] = "failed"
            stage["error"] = f"{type(exc).__name__}: {exc}"
            warnings[name].append(stage["error"])
            value = None
        stage["seconds"] = clock.perf_counter() - then
        incident.stages.append(stage)
        status = "cached" if stage["status"] == "done" and stage["cache_hit"] else stage["status"]
        detail = stage["error"] or f"{stage['seconds']:.1f}s" + (" (cached)" if stage["cache_hit"] else "")
        _notify(progress, index, status, detail)
        return value

    def skip(index: int, reason: str) -> None:
        _notify(progress, index, "running", "Checking prerequisites")
        incident.stages.append({"name": STAGES[index], "status": "skipped", "seconds": 0.0,
                                "cache_hit": False, "error": None})
        warnings[STAGES[index]].append(reason)
        _notify(progress, index, "skipped", reason)

    def area_output(aoi: AOI) -> None:
        incident.aoi = aoi.to_dict()
        warnings["aoi"].extend(aoi.warnings)

    def pair_output(result: PairSearchResult) -> str:
        if result.status not in {"ok", "no_scenes", "no_same_orbit_pair"}:
            raise ValueError(f"Unknown pair-search status: {result.status}")
        if result.status == "ok" and result.pair is None:
            raise ValueError("Pair search reported ok without a pair")
        incident.pair_summary = _pair_summary(result, aoi)
        warnings["s1_search"].extend(result.warnings)
        if result.pair:
            warnings["s1_search"].extend(result.pair.warnings)
        if result.status != "ok" or result.pair is None:
            warnings["s1_search"].append(result.reason)
            return "blocked"
        return "done"

    def layer_output(output: Any, stage: str, key: str) -> None:
        metadata = output.to_dict()
        # Access the fields consumed by quality inside the stage exception boundary.
        if stage == "preprocess":
            fraction = metadata["valid_fraction"]
            if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("S1 valid_fraction must be finite and within [0, 1]")
        elif stage == "terrain":
            fraction = metadata["hand_nan_fraction"]
            if not math.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("HAND NaN fraction must be finite and within [0, 1]")
        else:
            for name in ("road_density_km_per_km2", "hospital_count", "settlement_count",
                         "roads_edited_over_5_years_share"):
                value = metadata["quality"][name]
                if value is None and name == "roads_edited_over_5_years_share":
                    continue
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"OSM {name} must be finite and nonnegative")
        warnings[stage].extend(metadata["warnings"])
        incident.layers.update(_layer_paths(output, key))
        parameters = incident.provenance["parameters"]
        first_input = len(incident.provenance["inputs"])
        if stage == "preprocess":
            parameters.update(resolution=metadata["resolution"], filter_window=metadata["filter_window"])
            source = {"rtc": "sentinel-1-rtc", "grd_fallback": "sentinel-1-grd"}.get(metadata["source"], metadata["source"])
            for role, pass_ in (("pre", result.pair.pre), ("post", result.pair.post)):
                ids = metadata["pair_metadata"]["input_item_ids"][role]
                incident.provenance["inputs"].append({
                    "role": f"s1_{role}", "source": source, "collection": source,
                    "provider": "Planetary Computer", "item_ids": ids,
                    "acquisition_date": pass_.acquisition_date.isoformat(),
                    "license_note": "License not captured by the S1 cache; consult Planetary Computer collection metadata.",
                })
        elif stage == "terrain":
            parameters.update({name: metadata[name] for name in ("buffer_km", "hydro_resolution", "stream_threshold_km2")})
            incident.provenance["inputs"].append({
                "role": "dem", "source": metadata["source"], "collection": metadata["source"],
                "provider": "Planetary Computer", "tile_ids": metadata["tile_ids"],
                "license_note": "License not captured by the DEM cache; consult Planetary Computer collection metadata.",
            })
        else:
            parameters["context_buffer_km"] = metadata["quality"]["context_buffer_km"]
            incident.provenance["inputs"].append({
                "role": "osm", "source": "pre-event-openstreetmap", "endpoint": metadata["endpoint_used"],
                "snapshot_date": metadata["snapshot_date"], "snapshot_verified": metadata["snapshot_verified"],
                "newest_edit": metadata["newest_edit"],
                "license_note": "OpenStreetMap contributors; Open Database License (ODbL); preserve attribution.",
            })
        receipt = {**incident.provenance, "inputs": incident.provenance["inputs"][first_input:]}
        outputs[stage] = metadata
        assert_provenance_allowed(replace(incident, provenance=receipt))

    aoi = run(0, lambda: call(0, build_aoi, place=place, bbox=bbox, flood_date=flood_date), area_output)
    result = None
    stack = None
    if aoi is None:
        skip(1, "AOI stage failed.")
    else:
        result = run(1, lambda: call(1, find_best_pair, aoi=aoi), pair_output)
    if result is None or result.status != "ok" or result.pair is None:
        skip(2, "No usable S1 pair; preprocessing requires a successful search.")
    else:
        stack = run(2, lambda: call(2, preprocess_pair, aoi=aoi, pair_result=result, cache_dir=root),
                    lambda output: layer_output(output, "preprocess", "s1_stack"))
    if stack is None:
        skip(3, "Terrain requires a successful S1 stack.")
        skip(4, "OSM requires a successful S1 stack.")
    else:
        run(3, lambda: call(3, build_terrain, aoi=aoi, stack=stack, cache_dir=root),
            lambda output: layer_output(output, "terrain", "terrain"))
        run(4, lambda: call(4, build_osm, aoi=aoi, stack=stack, cache_dir=root, include_buildings=include_buildings),
            lambda output: layer_output(output, "osm", "osm"))
    if result is not None and (result.status != "ok" or result.pair is None):
        incident.status = "blocked"
    elif stack is not None:
        incident.status = "ready" if all(stage["status"] == "done" for stage in incident.stages) else "partial"
    if incident.status == "ready":
        assert_provenance_allowed(incident)
    incident.quality = _quality(incident, outputs, warnings, include_buildings)
    incident.incident_id = _id(incident.aoi, incident.provenance["parameters"], include_buildings, request)
    incident.cache_path = str(root / "incidents" / incident.incident_id / "incident.json")
    incident.timings_total_seconds = clock.perf_counter() - started
    try:
        save_incident(incident)
    except Exception as exc:
        incident.status = "failed"
        message = f"{type(exc).__name__}: {exc}"
        incident.quality["warnings"].append(f"incident: Could not save receipt: {message}")
        incident.quality["flags"].append({"code": "incident_save_failed", "stage": "incident",
                                          "severity": "limiting", "message": message})
        incident.quality["trust_summary"] = trust_summary(incident.quality["flags"], incident.status)
    return incident


def main() -> None:
    """Print stage progress and a compact receipt, with nonzero exit for incomplete runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    area = parser.add_mutually_exclusive_group(required=True)
    area.add_argument("--place")
    area.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--date", required=True, dest="flood_date")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--with-buildings", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    def progress(index: int, total: int, name: str, status: str, detail: str) -> None:
        display = "done" if status == "cached" else status
        suffix = "" if status == "running" else f" {detail}"
        print(f"[{index}/{total}] {name} ... {display}{suffix}", flush=True)

    incident = prepare_incident(
        place=args.place, bbox=args.bbox, flood_date=args.flood_date, cache_dir=args.cache_dir,
        include_buildings=args.with_buildings, progress=progress, force=args.force,
    )
    print(json.dumps({
        "incident_id": incident.incident_id, "status": incident.status,
        "trust_summary": incident.quality["trust_summary"], "pair_summary": incident.pair_summary,
        "stages": incident.stages, "timings_total_seconds": incident.timings_total_seconds,
        "flags": [{"code": flag["code"], "severity": flag["severity"]} for flag in incident.quality["flags"]],
        "cache_path": incident.cache_path,
    }, indent=2))
    if incident.status != "ready":
        parser.exit(2 if incident.status == "blocked" else 1)


if __name__ == "__main__":
    main()
