"""Build an aligned, filtered Sentinel-1 pre/post backscatter stack."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Literal
import zlib

import numpy as np
import planetary_computer
from pystac_client import Client
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, GCPTransformer, from_origin
from rasterio.warp import reproject, transform, transform_bounds
from rasterio.windows import Window, from_bounds
from scipy.ndimage import uniform_filter

from src.data.aoi import AOI, BBox, build_aoi
from src.data.s1_search import PairSearchResult, S1Pass, STAC_URL, find_best_pair


BAND_NAMES = ("pre_vv", "pre_vh", "post_vv", "post_vh")
FALLBACK_WARNING = (
    "no radiometric/terrain correction applied (GRD fallback): mountain "
    "layover/shadow effects are not corrected; treat results as lower confidence"
)


@dataclass(frozen=True)
class RasterBand:
    """An AOI-only source window with affine or shifted GCP georeferencing."""

    array: np.ndarray
    transform: Affine | None
    crs: CRS
    gcps: tuple[GroundControlPoint, ...] = ()


@dataclass(frozen=True)
class Grid:
    """Shared UTM destination grid for both acquisitions."""

    transform: Affine
    crs: CRS
    width: int
    height: int
    resolution: float
    warnings: list[str]


@dataclass(frozen=True)
class S1Stack:
    """Four dB bands and their common validity mask and provenance."""

    array: np.ndarray
    valid_mask: np.ndarray
    transform: Affine
    crs: CRS
    resolution: float
    bbox: BBox
    band_names: tuple[str, ...]
    source: Literal["rtc", "grd_fallback"]
    pair_metadata: dict[str, Any]
    valid_fraction: float
    warnings: list[str]
    filter_window: int = 5
    cache_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return metadata only, in JSON-compatible form."""
        return {
            "shape": list(self.array.shape),
            "transform": list(self.transform)[:6],
            "crs": self.crs.to_string(),
            "resolution": self.resolution,
            "bbox": list(self.bbox),
            "band_names": list(self.band_names),
            "source": self.source,
            "pair_metadata": self.pair_metadata,
            "valid_fraction": self.valid_fraction,
            "warnings": list(self.warnings),
            "filter_window": self.filter_window,
            "cache_path": self.cache_path,
        }


def _properties(item: Any) -> dict[str, Any]:
    """Read properties from either a STAC item or a test dictionary."""
    return (item.get("properties") if isinstance(item, dict) else item.properties) or {}


def _item_id(item: Any) -> str:
    """Read an item identifier."""
    return item["id"] if isinstance(item, dict) else item.id


def _assets(item: Any) -> dict[str, Any]:
    """Read an item's assets."""
    return (item.get("assets") if isinstance(item, dict) else item.assets) or {}


def find_rtc_items(aoi: AOI, pass_: S1Pass, client: Any = None) -> list[Any]:
    """Find dual-polarization RTC frames on exactly the requested UTC pass."""
    if client is None:
        client = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    day = pass_.acquisition_date.isoformat()
    items = client.search(
        collections=["sentinel-1-rtc"], bbox=list(aoi.bbox),
        datetime=f"{day}T00:00:00Z/{day}T23:59:59.999999Z",
    ).items()
    matches = {}
    for item in items:
        properties = _properties(item)
        try:
            timestamp = datetime.fromisoformat(properties.get("datetime", ""))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            orbit = properties.get("sat:relative_orbit")
            if isinstance(orbit, bool) or not isinstance(orbit, (int, str)):
                continue
            matches_pass = (
                properties.get("sat:orbit_state") == pass_.orbit_state
                and int(orbit) == pass_.relative_orbit
                and timestamp.astimezone(timezone.utc).date() == pass_.acquisition_date
            )
        except (TypeError, ValueError, OverflowError):
            continue
        if matches_pass and {"vv", "vh"}.issubset(_assets(item)):
            matches[_item_id(item)] = item
    return [matches[key] for key in sorted(matches)]


def _grd_items(pass_: S1Pass, client: Any) -> list[Any]:
    """Resolve every original frame; never substitute another acquisition."""
    items = client.search(collections=["sentinel-1-grd"], ids=pass_.item_ids).items()
    by_id = {_item_id(item): item for item in items}
    missing = set(pass_.item_ids) - by_id.keys()
    if missing:
        raise ValueError(f"GRD frames are missing from the catalog: {sorted(missing)}")
    selected = [by_id[key] for key in sorted(set(pass_.item_ids))]
    if not selected or any(not {"vv", "vh"}.issubset(_assets(item)) for item in selected):
        raise ValueError("GRD fallback requires vv and vh assets for every selected frame")
    return selected


def _bbox_edges(bbox: BBox) -> tuple[np.ndarray, np.ndarray]:
    """Densify all edges before inverting a nonlinear GCP transform."""
    west, south, east, north = bbox
    xs, ys = np.linspace(west, east, 33), np.linspace(south, north, 33)
    return (
        np.concatenate((xs, xs, np.full(33, west), np.full(33, east))),
        np.concatenate((np.full(33, south), np.full(33, north), ys, ys)),
    )


def load_band(item: Any, asset_key: str, aoi: AOI) -> RasterBand:
    """Read only intersecting AOI pixels, retaining affine or GCP geometry."""
    asset = _assets(item).get(asset_key)
    if asset is None:
        raise ValueError(f"Item {_item_id(item)} has no {asset_key} asset")
    href = asset.get("href") if isinstance(asset, dict) else asset.href
    fields = asset if isinstance(asset, dict) else asset.extra_fields
    raster_bands = fields.get("raster:bands") or [{}]
    metadata = raster_bands[0]
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_CONNECTTIMEOUT="15", GDAL_HTTP_TIMEOUT="120",
    ), rasterio.open(href) as src:
        gcps, gcp_crs = src.gcps
        crs = gcp_crs if gcps else src.crs
        if crs is None:
            raise ValueError(f"Item {_item_id(item)} has neither CRS nor georeferenced GCPs")
        if gcps:
            xs, ys = _bbox_edges(aoi.bbox)
            xs, ys = transform("EPSG:4326", crs, xs.tolist(), ys.tolist())
            with GCPTransformer(gcps) as transformer:
                rows, cols = transformer.rowcol(xs, ys, op=lambda value: value)
            if not np.isfinite(rows).all() or not np.isfinite(cols).all():
                raise ValueError("AOI cannot be located using the source GCPs")
            left, top = math.floor(min(cols)), math.floor(min(rows))
            right, bottom = math.ceil(max(cols)), math.ceil(max(rows))
        else:
            bounds = transform_bounds("EPSG:4326", crs, *aoi.bbox, densify_pts=21)
            window = from_bounds(*bounds, transform=src.transform)
            left, top = math.floor(window.col_off), math.floor(window.row_off)
            right = math.ceil(window.col_off + window.width)
            bottom = math.ceil(window.row_off + window.height)
        left, top = max(0, left), max(0, top)
        right, bottom = min(src.width, right), min(src.height, bottom)
        if right <= left or bottom <= top:
            return RasterBand(np.empty((0, 0), dtype="float32"), src.transform, crs)
        window = Window(left, top, right - left, bottom - top)
        data = src.read(1, window=window, masked=True).astype("float32").filled(np.nan)
        nodata = metadata.get("nodata")
        if isinstance(nodata, (int, float)):
            data[data == nodata] = np.nan
        scale = metadata.get("scale", src.scales[0])
        offset = metadata.get("offset", src.offsets[0])
        data = data * scale + offset
        units = (src.units[0], metadata.get("unit"), src.tags(1).get("Scale"))
        if any(str(value).lower() == "db" for value in units):
            raise ValueError("Expected linear power/amplitude input, but asset declares dB values")
        data[~np.isfinite(data) | (data <= 0)] = np.nan
        shifted = tuple(GroundControlPoint(
            row=gcp.row - top, col=gcp.col - left,
            x=gcp.x, y=gcp.y, z=gcp.z, id=gcp.id, info=gcp.info,
        ) for gcp in gcps)
        return RasterBand(data, None if gcps else src.window_transform(window), crs, shifted)


def make_grid(aoi: AOI, resolution: float = 10) -> Grid:
    """Snap the AOI to its center UTM zone and bound output dimensions."""
    if isinstance(resolution, bool) or not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and positive")
    west, south, east, north = aoi.bbox
    if south < -80 or north > 84:
        raise ValueError("UTM preprocessing supports AOIs between 80°S and 84°N")
    zone = min(60, int(((west + east) / 2 + 180) // 6) + 1)
    crs = CRS.from_epsg((32600 if (south + north) / 2 >= 0 else 32700) + zone)
    bounds = transform_bounds("EPSG:4326", crs, *aoi.bbox, densify_pts=21)
    warnings = []

    def dimensions(spacing: float) -> tuple[Affine, int, int]:
        left = math.floor(bounds[0] / spacing) * spacing
        bottom = math.floor(bounds[1] / spacing) * spacing
        right = math.ceil(bounds[2] / spacing) * spacing
        top = math.ceil(bounds[3] / spacing) * spacing
        return (
            from_origin(left, top, spacing, spacing),
            round((right - left) / spacing), round((top - bottom) / spacing),
        )

    affine, width, height = dimensions(resolution)
    if max(width, height) > 4000 and resolution < 20:
        warnings.append(
            f"Grid {width}x{height} exceeds 4000 pixels per side; "
            f"resolution auto-coarsened from {resolution:g} to 20 m."
        )
        resolution = 20.0
        affine, width, height = dimensions(resolution)
    if max(width, height) > 4000:
        raise ValueError("AOI grid exceeds 4000 pixels per side; use a smaller AOI or coarser resolution")
    if int((west + 180) // 6) != int((east + 180) // 6):
        warnings.append("AOI spans UTM zones; the grid uses the center's zone.")
    return Grid(affine, crs, width, height, float(resolution), warnings)


def _warp(band: RasterBand, grid: Grid) -> np.ndarray:
    """Resample into the common pre grid using bilinear interpolation."""
    output = np.full((grid.height, grid.width), np.nan, dtype="float32")
    if band.array.size:
        geometry = {"gcps": band.gcps} if band.gcps else {"src_transform": band.transform}
        reproject(
            source=band.array, destination=output, src_crs=band.crs,
            src_nodata=np.nan, dst_transform=grid.transform, dst_crs=grid.crs,
            dst_nodata=np.nan, resampling=Resampling.bilinear, **geometry,
        )
    return output


def load_grd_fallback(item: Any, asset_key: str, aoi: AOI, grid: Grid) -> np.ndarray:
    """Warp uncalibrated amplitude using GCPs, then square it into DN power."""
    band = load_band(item, asset_key, aoi)
    if band.array.size and not band.gcps:
        raise ValueError("GRD fallback requires source GCPs; affine georeferencing is insufficient")
    amplitude = _warp(band, grid)
    return np.square(amplitude, dtype="float32")


def mosaic_pass(
    items: list[Any], asset_key: str, aoi: AOI, grid: Grid,
    source: Literal["rtc", "grd_fallback"] = "rtc",
) -> np.ndarray:
    """Mosaic frames on one grid; the first valid pixel wins overlaps."""
    mosaic = np.full((grid.height, grid.width), np.nan, dtype="float32")
    for item in sorted(items, key=_item_id):
        if source == "grd_fallback":
            warped = load_grd_fallback(item, asset_key, aoi, grid)
        else:
            band = load_band(item, asset_key, aoi)
            if band.gcps:
                raise ValueError("RTC input must already have terrain-corrected affine georeferencing")
            warped = _warp(band, grid)
        valid = np.isfinite(warped) & (warped > 0) & ~np.isfinite(mosaic)
        mosaic[valid] = warped[valid]
    return mosaic


def lee_filter(power: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply a NaN-aware multiplicative Lee filter with local noise estimation."""
    if type(window_size) is not int or window_size < 1 or window_size % 2 == 0:
        raise ValueError("filter window size must be a positive odd integer")
    valid = np.isfinite(power) & (power > 0)
    values = np.where(valid, power, 0).astype("float64")
    count = uniform_filter(valid.astype("float64"), size=window_size, mode="reflect")
    safe_count = np.maximum(count, np.finfo("float64").eps)
    mean = uniform_filter(values, size=window_size, mode="reflect") / safe_count
    second = uniform_filter(values * values, size=window_size, mode="reflect") / safe_count
    variance = np.maximum(second - mean * mean, 0)
    usable = valid & (mean > 0)
    noise = float(np.quantile(variance[usable] / mean[usable] ** 2, 0.25)) if usable.any() else 0.0
    signal_variance = np.maximum((variance - noise * mean * mean) / (1 + noise), 0)
    weight = np.divide(signal_variance, variance, out=np.zeros_like(variance), where=variance > 0)
    filtered = mean + weight * (values - mean)
    filtered[~valid] = np.nan
    return filtered.astype("float32")


def power_to_db(power: np.ndarray) -> np.ndarray:
    """Convert positive linear power to dB, preserving invalid pixels as NaN."""
    result = np.full(power.shape, np.nan, dtype="float32")
    valid = np.isfinite(power) & (power > 0)
    result[valid] = np.clip(10 * np.log10(power[valid]), -35, 5)
    return result


def write_quicklook(stack: S1Stack, path: str | Path) -> None:
    """Write post-VV as a grayscale PNG: -35 to 5 dB, invalid pixels black."""
    data = stack.array[2]
    image = np.zeros(data.shape, dtype="uint8")
    valid = np.isfinite(data)
    image[valid] = 1 + np.rint((np.clip(data[valid], -35, 5) + 35) * (254 / 40)).astype("uint8")
    rows = np.zeros((image.shape[0], image.shape[1] + 1), dtype="uint8")
    rows[:, 1:] = image

    def chunk(kind: bytes, value: bytes) -> bytes:
        return struct.pack("!I", len(value)) + kind + value + struct.pack("!I", zlib.crc32(kind + value))

    header = struct.pack("!2I5B", image.shape[1], image.shape[0], 8, 0, 0, 0, 0)
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows.tobytes())) + chunk(b"IEND", b"")
    )


def save_stack(stack: S1Stack, path: str | Path) -> None:
    """Save the float32 GeoTIFF, band descriptions, and JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", count=4, height=stack.array.shape[1],
        width=stack.array.shape[2], dtype="float32", crs=stack.crs,
        transform=stack.transform, nodata=np.nan, tiled=True, compress="deflate",
    ) as dst:
        dst.write(stack.array.astype("float32"))
        for index, name in enumerate(stack.band_names, 1):
            dst.set_band_description(index, name)
            dst.set_band_unit(index, "dB")
    path.with_suffix(".json").write_text(json.dumps(stack.to_dict(), indent=2, allow_nan=False))


def load_stack(path: str | Path) -> S1Stack:
    """Restore a cached cube and recompute the common validity mask."""
    path = Path(path)
    if path.is_dir():
        path = path / "stack.tif"
    metadata = json.loads(path.with_suffix(".json").read_text())
    with rasterio.open(path) as src:
        array = src.read(masked=True).astype("float32").filled(np.nan)
        affine, crs, names = src.transform, src.crs, src.descriptions
    if array.shape[0] != 4 or tuple(names) != tuple(metadata["band_names"]):
        raise ValueError("Cached stack must contain the four described pre/post bands")
    valid = np.isfinite(array).all(axis=0)
    return S1Stack(
        array=array, valid_mask=valid, transform=affine, crs=crs,
        resolution=metadata["resolution"], bbox=tuple(metadata["bbox"]),
        band_names=tuple(names), source=metadata["source"],
        pair_metadata=metadata["pair_metadata"], valid_fraction=float(valid.mean()),
        warnings=metadata["warnings"], filter_window=metadata["filter_window"],
        cache_path=str(path.resolve()),
    )


def preprocess_pair(
    aoi: AOI, pair_result: PairSearchResult, cache_dir: str | Path = "data/cache",
    client: Any = None, resolution: float = 10, filter_window: int = 5,
) -> S1Stack:
    """Resolve RTC or paired GRD inputs and cache an aligned four-band cube."""
    if pair_result.status != "ok" or pair_result.pair is None:
        raise ValueError(f"Cannot preprocess pair: status={pair_result.status}; {pair_result.reason}")
    if type(filter_window) is not int or filter_window < 1 or filter_window % 2 == 0:
        raise ValueError("filter window size must be a positive odd integer")
    pair = pair_result.pair
    if (pair.pre.orbit_state, pair.pre.relative_orbit) != (pair.post.orbit_state, pair.post.relative_orbit):
        raise ValueError("Pre/post passes must share orbit direction and relative orbit")
    grid = make_grid(aoi, resolution)
    if client is None:
        client = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    pre_items = find_rtc_items(aoi, pair.pre, client)
    post_items = find_rtc_items(aoi, pair.post, client)
    warnings = [*aoi.warnings, *pair_result.warnings, *pair.warnings, *grid.warnings]
    source: Literal["rtc", "grd_fallback"] = "rtc"
    if not pre_items or not post_items:
        missing = " and ".join(name for name, items in (("pre", pre_items), ("post", post_items)) if not items)
        warnings.extend([
            f"RTC lacks usable {missing} pass assets; both passes use GRD "
            "to avoid mixing gamma0 and uncalibrated DN power.",
            FALLBACK_WARNING,
            "GRD dB values are uncalibrated DN power, not physical backscatter; clipping may saturate them.",
        ])
        source = "grd_fallback"
        pre_items, post_items = _grd_items(pair.pre, client), _grd_items(pair.post, client)
    inputs = {
        "version": 1, "bbox": aoi.bbox,
        "pre_ids": sorted(pair.pre.item_ids), "post_ids": sorted(pair.post.item_ids),
        "input_ids": [[_item_id(item) for item in items] for items in (pre_items, post_items)],
        "resolution": grid.resolution, "requested_resolution": resolution,
        "filter": "simple_lee", "filter_window": filter_window,
        "db_clip": [-35, 5], "source": source,
    }
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(cache_dir).resolve() / "s1" / key / "stack.tif"
    if path.exists() and path.with_suffix(".json").exists():
        stack = load_stack(path)
        if not path.with_name("quicklook.png").exists():
            write_quicklook(stack, path.with_name("quicklook.png"))
        return stack
    array = np.empty((4, grid.height, grid.width), dtype="float32")
    for index, (items, band) in enumerate((
        (pre_items, "vv"), (pre_items, "vh"), (post_items, "vv"), (post_items, "vh"),
    )):
        power = lee_filter(mosaic_pass(items, band, aoi, grid, source), filter_window)
        valid = np.isfinite(power) & (power > 0)
        clipped = valid & ((power < 10 ** (-3.5)) | (power > 10 ** 0.5))
        if clipped.any():
            warnings.append(
                f"{BAND_NAMES[index]}: {clipped.sum() / valid.sum():.2%} "
                "of valid pixels clipped to [-35, 5] dB."
            )
        array[index] = power_to_db(power)
    valid_mask = np.isfinite(array).all(axis=0)
    fraction = float(valid_mask.mean())
    if fraction < 0.9:
        warnings.append(f"Common valid_fraction={fraction:.4f} ({fraction:.2%}) is below 0.9.")
    metadata = pair.to_dict()
    metadata["input_item_ids"] = {
        "pre": [_item_id(item) for item in pre_items],
        "post": [_item_id(item) for item in post_items],
    }
    stack = S1Stack(
        array=array, valid_mask=valid_mask, transform=grid.transform, crs=grid.crs,
        resolution=grid.resolution, bbox=aoi.bbox, band_names=BAND_NAMES,
        source=source, pair_metadata=metadata, valid_fraction=fraction,
        warnings=list(dict.fromkeys(warnings)), filter_window=filter_window,
        cache_path=str(path),
    )
    save_stack(stack, path)
    write_quicklook(stack, path.with_name("quicklook.png"))
    return stack


def main() -> None:
    """Run live scene search and preprocessing, then print output metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    area = parser.add_mutually_exclusive_group(required=True)
    area.add_argument("--place")
    area.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--date", required=True, dest="flood_date")
    parser.add_argument("--resolution", type=float, default=10)
    parser.add_argument("--filter-window", type=int, default=5)
    parser.add_argument("--cache-dir", default="data/cache")
    args = parser.parse_args()
    aoi = build_aoi(place=args.place, bbox=args.bbox, flood_date=args.flood_date)
    stack = preprocess_pair(
        aoi, find_best_pair(aoi), cache_dir=args.cache_dir,
        resolution=args.resolution, filter_window=args.filter_window,
    )
    print(json.dumps(stack.to_dict(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
