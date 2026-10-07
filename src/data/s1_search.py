"""Search Sentinel-1 metadata and choose comparable passes around a flood."""

import argparse
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import json
from typing import Any, Literal

import planetary_computer
from pystac_client import Client
from shapely.errors import GEOSException
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from src.data.aoi import AOI, BBox, build_aoi


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
Scene = dict[str, Any]


@dataclass(frozen=True)
class S1Pass:
    """Frames from one orbit and UTC acquisition date, with union coverage."""

    orbit_state: str
    relative_orbit: int
    acquisition_date: date
    item_ids: list[str]
    platforms: list[str]
    coverage: float
    earliest_datetime: str
    latest_datetime: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible pass metadata."""
        return {
            "orbit_state": self.orbit_state,
            "relative_orbit": self.relative_orbit,
            "acquisition_date": self.acquisition_date.isoformat(),
            "item_ids": list(self.item_ids),
            "platforms": list(self.platforms),
            "coverage": self.coverage,
            "earliest_datetime": self.earliest_datetime,
            "latest_datetime": self.latest_datetime,
        }


@dataclass(frozen=True)
class ScenePair:
    """A pre/post pair with matching viewing geometry."""

    pre: S1Pass
    post: S1Pass
    orbit_state: str
    relative_orbit: int
    days_before_flood: int
    days_after_flood: int
    min_coverage: float
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible pair metadata."""
        return {
            "pre": self.pre.to_dict(),
            "post": self.post.to_dict(),
            "orbit_state": self.orbit_state,
            "relative_orbit": self.relative_orbit,
            "days_before_flood": self.days_before_flood,
            "days_after_flood": self.days_after_flood,
            "min_coverage": self.min_coverage,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PairSearchResult:
    """Best pair, up to three other candidates, or an honest absence."""

    pair: ScenePair | None
    alternatives: list[ScenePair]
    status: Literal["ok", "no_same_orbit_pair", "no_scenes"]
    reason: str
    n_pre_scenes: int
    n_post_scenes: int
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible search results."""
        return {
            "pair": self.pair.to_dict() if self.pair is not None else None,
            "alternatives": [pair.to_dict() for pair in self.alternatives],
            "status": self.status,
            "reason": self.reason,
            "n_pre_scenes": self.n_pre_scenes,
            "n_post_scenes": self.n_post_scenes,
            "warnings": list(self.warnings),
        }


def _timestamp(value: Any) -> datetime | None:
    """Parse a timestamp into UTC; treat unzoned values as UTC."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    try:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _orbit(state: Any, relative: Any) -> tuple[str, int] | None:
    """Reject unknown direction or missing/nonintegral orbit metadata."""
    if not isinstance(state, str) or state.lower() not in {"ascending", "descending"}:
        return None
    if isinstance(relative, bool) or not isinstance(relative, (int, str)):
        return None
    try:
        number = int(relative)
    except ValueError:
        return None
    return (state.lower(), number) if number > 0 else None


def _footprint(value: Any) -> BaseGeometry | None:
    """Accept only valid, nonempty polygon footprints."""
    try:
        geometry = value if isinstance(value, BaseGeometry) else shape(value)
        if (
            geometry.geom_type in {"Polygon", "MultiPolygon"}
            and geometry.is_valid
            and geometry.area > 0
        ):
            return geometry
    except (AttributeError, TypeError, ValueError, KeyError, IndexError, GEOSException):
        pass
    return None


def search_s1(
    aoi: AOI, collection: str = "sentinel-1-grd", client: Any = None
) -> list[Scene]:
    """Search inclusive windows and normalize usable IW, VV+VH scene metadata."""
    if client is None:
        client = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    interval = (
        f"{aoi.pre_window[0].isoformat()}T00:00:00Z/"
        f"{aoi.post_window[1].isoformat()}T23:59:59.999999Z"
    )
    items = client.search(
        collections=[collection], bbox=list(aoi.bbox), datetime=interval
    ).items()
    scenes: dict[str, Scene] = {}
    for item in items:
        if isinstance(item, dict):
            properties = item.get("properties") or {}
            item_id, geometry = item.get("id"), item.get("geometry")
        else:
            properties = getattr(item, "properties", None) or {}
            item_id, geometry = getattr(item, "id", None), getattr(item, "geometry", None)
        if not isinstance(properties, dict):
            continue
        polarizations = properties.get("sar:polarizations")
        if (
            properties.get("sar:instrument_mode") != "IW"
            or not isinstance(polarizations, (list, tuple))
        ):
            continue
        if "VV" not in polarizations or "VH" not in polarizations:
            continue
        orbit = _orbit(properties.get("sat:orbit_state"), properties.get("sat:relative_orbit"))
        timestamp = _timestamp(properties.get("datetime"))
        footprint = _footprint(geometry)
        if (
            not isinstance(item_id, str) or not item_id
            or orbit is None or timestamp is None or footprint is None
        ):
            continue
        if not aoi.pre_window[0] <= timestamp.date() <= aoi.post_window[1]:
            continue
        platform = properties.get("platform")
        scenes[item_id] = {
            "id": item_id,
            "datetime": timestamp.isoformat(),
            "orbit_state": orbit[0],
            "relative_orbit": orbit[1],
            "platform": platform if isinstance(platform, str) else None,
            "geometry": footprint,
        }
    return sorted(scenes.values(), key=lambda scene: (scene["datetime"], scene["id"]))


def group_into_passes(scenes: list[Scene], bbox: BBox) -> list[S1Pass]:
    """Union same-date, same-orbit footprints, clipped to the AOI rectangle."""
    rectangle = box(*bbox)
    if not rectangle.is_valid or rectangle.area <= 0:
        raise ValueError("bbox must have positive area")
    groups: dict[tuple[str, int, date], dict[str, Scene]] = {}
    for scene in scenes:
        orbit = _orbit(scene.get("orbit_state"), scene.get("relative_orbit"))
        timestamp = _timestamp(scene.get("datetime"))
        footprint = _footprint(scene.get("geometry"))
        item_id = scene.get("id")
        if (
            orbit is None or timestamp is None or footprint is None
            or not isinstance(item_id, str) or not item_id
        ):
            continue
        key = (*orbit, timestamp.date())
        groups.setdefault(key, {})[item_id] = {**scene, "datetime": timestamp, "geometry": footprint}
    passes = []
    for (state, relative, day), frames in sorted(groups.items()):
        clipped = [scene["geometry"].intersection(rectangle) for scene in frames.values()]
        coverage = unary_union(clipped).area / rectangle.area
        timestamps = [scene["datetime"] for scene in frames.values()]
        passes.append(S1Pass(
            orbit_state=state,
            relative_orbit=relative,
            acquisition_date=day,
            item_ids=sorted(frames),
            platforms=sorted({
                scene["platform"] for scene in frames.values()
                if isinstance(scene.get("platform"), str)
            }),
            coverage=max(0.0, min(1.0, coverage)),
            earliest_datetime=min(timestamps).isoformat(),
            latest_datetime=max(timestamps).isoformat(),
        ))
    return passes


def select_pairs(aoi: AOI, passes: list[S1Pass]) -> PairSearchResult:
    """Rank matching-orbit pairs by coverage bucket, then post/pre proximity."""
    pre = [p for p in passes if aoi.pre_window[0] <= p.acquisition_date <= aoi.pre_window[1]]
    post = [p for p in passes if aoi.post_window[0] <= p.acquisition_date <= aoi.post_window[1]]
    n_pre, n_post = sum(len(p.item_ids) for p in pre), sum(len(p.item_ids) for p in post)
    candidates = []
    for before in pre:
        if before.coverage < 0.5:
            continue
        for after in post:
            if after.coverage < 0.5 or (
                before.orbit_state, before.relative_orbit
            ) != (after.orbit_state, after.relative_orbit):
                continue
            days_before = (aoi.flood_date - before.acquisition_date).days
            days_after = (after.acquisition_date - aoi.flood_date).days
            coverage = min(before.coverage, after.coverage)
            warnings = []
            if coverage < 0.95:
                warnings.append(f"Partial coverage: min_coverage={coverage:.6f} ({coverage:.2%}) is below 0.95.")
            if days_after == 0:
                warnings.append("Post pass is on the flood date; acquisition time may precede the event.")
            if days_after > 6:
                warnings.append(f"Post pass is {days_after} days after the flood; flood may have receded; revisit gap.")
            candidates.append(ScenePair(
                pre=before,
                post=after,
                orbit_state=before.orbit_state,
                relative_orbit=before.relative_orbit,
                days_before_flood=days_before,
                days_after_flood=days_after,
                min_coverage=coverage,
                warnings=warnings,
            ))
    candidates.sort(key=lambda pair: (
        pair.min_coverage < 0.95,
        pair.days_after_flood,
        pair.days_before_flood,
        pair.orbit_state,
        pair.relative_orbit,
        pair.pre.item_ids,
        pair.post.item_ids,
    ))
    if candidates:
        status = "ok"
        reason = (
            "Selected a same-direction, same-relative-orbit pair with at least "
            "50% coverage in each pass."
        )
    elif not passes:
        status = "no_scenes"
        reason = (
            "No usable IW scenes with VV and VH and complete orbit, time, "
            "and footprint metadata were found."
        )
    else:
        status = "no_same_orbit_pair"
        reason = (
            "Scenes were found, but no pre/post passes in the observation windows "
            "share direction and relative orbit with at least 50% coverage each."
        )
    best = candidates[0] if candidates else None
    return PairSearchResult(
        pair=best,
        alternatives=candidates[1:4],
        status=status,
        reason=reason,
        n_pre_scenes=n_pre,
        n_post_scenes=n_post,
        warnings=[*aoi.warnings, *(best.warnings if best else [])],
    )


def find_best_pair(
    aoi: AOI,
    collection: str = "sentinel-1-grd",
    client: Any = None,
    widen_post_days: int = 12,
) -> PairSearchResult:
    """Search and select, retrying once with only the post end extended."""
    if type(widen_post_days) is not int or widen_post_days < 0:
        raise ValueError("widen_post_days must be a nonnegative integer")
    scenes = search_s1(aoi, collection, client)
    result = select_pairs(aoi, group_into_passes(scenes, aoi.bbox))
    if result.status == "ok":
        return result
    old_end = aoi.post_window[1]
    today = date.today()
    extension = min(widen_post_days, max(0, (today - old_end).days))
    new_end = min(old_end, today) + timedelta(days=extension)
    if new_end > old_end:
        warning = (
            f"Post window was widened from {old_end.isoformat()} to {new_end.isoformat()} "
            "for one retry (capped at today); pre window unchanged."
        )
    else:
        warning = (
            f"Post-window widening requested for one retry; end remains {new_end.isoformat()} "
            "because no extension is available within today or widen_post_days is zero; "
            "pre window unchanged."
        )
    widened = replace(aoi, post_window=(aoi.post_window[0], new_end))
    scenes = search_s1(widened, collection, client)
    result = select_pairs(widened, group_into_passes(scenes, widened.bbox))
    return replace(result, warnings=[*result.warnings, warning])


def main() -> None:
    """Print a scene-pair search as pretty JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    area = parser.add_mutually_exclusive_group(required=True)
    area.add_argument("--place")
    area.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--date", required=True, dest="flood_date")
    args = parser.parse_args()
    aoi = build_aoi(place=args.place, bbox=args.bbox, flood_date=args.flood_date)
    print(json.dumps(find_best_pair(aoi).to_dict(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
