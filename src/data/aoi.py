"""Build a bounded geographic area and observation windows for a flood."""

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
import json
import math
from typing import Any


BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class AOI:
    """Area of interest in EPSG:4326, with inclusive observation windows."""

    name: str
    bbox: BBox
    flood_date: date
    pre_window: tuple[date, date]
    post_window: tuple[date, date]
    area_km2: float
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible values, with dates in ISO format."""
        return {
            "name": self.name,
            "bbox": list(self.bbox),
            "flood_date": self.flood_date.isoformat(),
            "pre_window": [day.isoformat() for day in self.pre_window],
            "post_window": [day.isoformat() for day in self.post_window],
            "area_km2": self.area_km2,
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        """Serialize this area as pretty JSON."""
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AOI":
        """Restore an area from the dictionary produced by ``to_dict``."""
        pre_start, pre_end = data["pre_window"]
        post_start, post_end = data["post_window"]
        return cls(
            name=data["name"],
            bbox=_validate_bbox(data["bbox"]),
            flood_date=date.fromisoformat(data["flood_date"]),
            pre_window=(date.fromisoformat(pre_start), date.fromisoformat(pre_end)),
            post_window=(date.fromisoformat(post_start), date.fromisoformat(post_end)),
            area_km2=float(data["area_km2"]),
            warnings=list(data["warnings"]),
        )


def geocode_place(place: str, point_buffer_km: float = 15.0):
    import math
    import osmnx
    try:
        return _validate_bbox(osmnx.geocode_to_gdf(place).total_bounds)
    except TypeError:
        # result is a point/line (town, river), not a polygon -> buffer it
        lat, lon = osmnx.geocode(place)
        dlat = point_buffer_km / 111.32
        dlon = point_buffer_km / (111.32 * math.cos(math.radians(lat)))
        return _validate_bbox((lon - dlon, lat - dlat, lon + dlon, lat + dlat))


def _validate_bbox(bbox: tuple) -> BBox:
    """Validate and normalize four finite longitude/latitude coordinates."""
    try:
        west, south, east, north = (float(value) for value in bbox)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bbox must contain four numeric coordinates") from exc
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise ValueError("bbox coordinates must be finite")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox requires -180 <= west < east <= 180 and -90 <= south < north <= 90")
    return west, south, east, north


def build_aoi(
    place: str | None = None,
    bbox: tuple | None = None,
    flood_date: str | date = ...,
    pre_days: int = 60,
    post_days: int = 12,
    max_side_km: float = 50.0,
) -> AOI:
    """Build an area, capping each side and truncating future observations.

    Distances use 111.32 km per latitude degree and the cosine of the
    center latitude for longitude degrees. The area is their product.
    """
    if (place is None) == (bbox is None):
        raise ValueError("Provide exactly one of place or bbox")
    if place is not None and (not isinstance(place, str) or not place.strip()):
        raise ValueError("place must be a nonempty string")
    if type(pre_days) is not int or pre_days < 1:
        raise ValueError("pre_days must be a positive integer")
    if type(post_days) is not int or post_days < 0:
        raise ValueError("post_days must be a nonnegative integer")
    try:
        max_side_km = float(max_side_km)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_side_km must be finite and positive") from exc
    if not math.isfinite(max_side_km) or max_side_km <= 0:
        raise ValueError("max_side_km must be finite and positive")

    today = date.today()
    if isinstance(flood_date, str):
        try:
            parsed_date = date.fromisoformat(flood_date)
        except ValueError as exc:
            raise ValueError("flood_date must be a date or YYYY-MM-DD string") from exc
        if parsed_date.isoformat() != flood_date:
            raise ValueError("flood_date must use YYYY-MM-DD format")
        flood_date = parsed_date
    if type(flood_date) is not date:
        raise ValueError("flood_date must be a date or YYYY-MM-DD string")
    if flood_date > today:
        raise ValueError("flood_date cannot be in the future")
    try:
        pre_window = (flood_date - timedelta(days=pre_days), flood_date - timedelta(days=1))
        post_end = flood_date + timedelta(days=post_days)
    except OverflowError as exc:
        raise ValueError("Observation windows exceed the supported date range") from exc

    west, south, east, north = _validate_bbox(
        geocode_place(place) if place is not None else bbox
    )
    center_lon = (west + east) / 2
    center_lat = (south + north) / 2
    lon_km = 111.32 * math.cos(math.radians(center_lat))
    width = (east - west) * lon_km
    height = (north - south) * 111.32
    warnings = []
    if width > max_side_km or height > max_side_km:
        original_width, original_height = width, height
        if width > max_side_km:
            half_width = max_side_km / lon_km / 2
            west, east = center_lon - half_width, center_lon + half_width
        if height > max_side_km:
            half_height = max_side_km / 111.32 / 2
            south, north = center_lat - half_height, center_lat + half_height
        width, height = (east - west) * lon_km, (north - south) * 111.32
        warnings.append(
            f"BBox clamped from {original_width:.3f} x {original_height:.3f} km "
            f"to {width:.3f} x {height:.3f} km (side limit {max_side_km:g} km)."
        )
    if post_end > today:
        warnings.append(
            f"Post window extends past today; end shortened from {post_end.isoformat()} "
            f"to {today.isoformat()}."
        )
        post_end = today
    return AOI(
        name=place if place is not None else "Custom bbox",
        bbox=(west, south, east, north),
        flood_date=flood_date,
        pre_window=pre_window,
        post_window=(flood_date, post_end),
        area_km2=width * height,
        warnings=warnings,
    )


def main() -> None:
    """Print an area of interest as JSON from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    area = parser.add_mutually_exclusive_group(required=True)
    area.add_argument("--place")
    area.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--date", required=True, dest="flood_date")
    args = parser.parse_args()
    print(build_aoi(place=args.place, bbox=args.bbox, flood_date=args.flood_date).to_json())


if __name__ == "__main__":
    main()
