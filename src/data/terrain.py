"""Buffered elevation, slope and drainage height on the Sentinel-1 grid."""

import argparse
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct
import time
from typing import Any
import zlib

import numpy as np
import planetary_computer
from pystac_client import Client
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import Affine, array_bounds, from_origin
from rasterio.warp import reproject, transform_bounds
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, binary_erosion, find_objects, label
from scipy.spatial import QhullError

from src.data.aoi import AOI, BBox, build_aoi
from src.data.s1_search import STAC_URL, find_best_pair
from src.data.s1_preprocess import S1Stack, preprocess_pair


BAND_NAMES = ("elevation", "slope_deg", "hand_m", "stream_mask")
NEIGHBORS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
BACKEND_WARNING = (
    "Self-contained priority-flood/D8 backend used: pysheds failed its compatibility "
    "probe because NumPy no longer provides in1d."
)
SURFACE_WARNING = (
    "Copernicus DEM is a surface model including buildings and canopy; "
    "HAND and slope can be biased in dense settlements and forest."
)


@dataclass(frozen=True)
class DEMData:
    """Windowed native DEM mosaic and its coverage warnings."""

    elevation: np.ndarray
    transform: Affine
    crs: CRS
    warnings: list[str]
    tile_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HANDResult:
    """Drainage height, threshold-derived streams, and hydrologic warnings."""

    hand_m: np.ndarray
    stream_mask: np.ndarray
    warnings: list[str]


@dataclass(frozen=True)
class TerrainStack:
    """Terrain layers aligned exactly with the input radar cube."""

    elevation: np.ndarray
    slope_deg: np.ndarray
    hand_m: np.ndarray
    stream_mask: np.ndarray
    transform: Affine
    crs: CRS
    bbox: BBox
    source: str
    hydro_resolution: float
    stream_threshold_km2: float
    buffer_km: float
    warnings: list[str]
    tile_ids: list[str] = field(default_factory=list)
    cache_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible grid, provenance, and summary statistics."""
        return {
            "shape": list(self.elevation.shape),
            "transform": list(self.transform)[:6],
            "crs": self.crs.to_string(),
            "bbox": list(self.bbox),
            "source": self.source,
            "hydro_resolution": self.hydro_resolution,
            "stream_threshold_km2": self.stream_threshold_km2,
            "buffer_km": self.buffer_km,
            "hydrology_backend": "priority_flood_d8",
            "band_names": list(BAND_NAMES),
            "stats": {
                "elevation": _stats(self.elevation),
                "slope_deg": _stats(self.slope_deg),
                "hand_m": _stats(self.hand_m),
            },
            "hand_nan_fraction": float((~np.isfinite(self.hand_m)).mean()),
            "stream_cell_count": int(self.stream_mask.sum()),
            "tile_ids": list(self.tile_ids),
            "warnings": list(self.warnings),
            "cache_path": self.cache_path,
        }


def _stats(array: np.ndarray) -> dict[str, float | None]:
    """Summarize finite samples without inventing statistics for missing data."""
    values = array[np.isfinite(array)]
    if not values.size:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean(dtype="float64")),
    }


def _positive(value: float, name: str, allow_zero: bool = False) -> None:
    """Validate a finite numeric parameter."""
    if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def expanded_bbox(bbox: BBox, buffer_km: float) -> BBox:
    """Expand geographic bounds using distances at the center latitude."""
    _positive(buffer_km, "buffer_km", allow_zero=True)
    west, south, east, north = bbox
    if not all(math.isfinite(value) for value in bbox) or not (
        -180 <= west < east <= 180 and -90 < south < north < 90
    ):
        raise ValueError("bbox must contain ordered longitude/latitude bounds")
    latitude = buffer_km / 111.32
    longitude = latitude / math.cos(math.radians((south + north) / 2))
    result = west - longitude, south - latitude, east + longitude, north + latitude
    if result[0] < -180 or result[2] > 180 or result[1] <= -90 or result[3] >= 90:
        raise ValueError("Buffered bbox crosses geographic limits; reduce the buffer or split the AOI")
    return result


def fill_small_holes(dem: np.ndarray, max_cells: int = 16) -> tuple[np.ndarray, list[str]]:
    """Linearly interpolate enclosed NaN components up to max_cells native pixels."""
    output = np.array(dem, dtype="float32", copy=True)
    output[~np.isfinite(output)] = np.nan
    components, _ = label(~np.isfinite(output), structure=np.ones((3, 3)))
    sizes = np.bincount(components.ravel())
    filled = 0
    height, width = output.shape
    for component, slices in enumerate(find_objects(components), 1):
        if slices is None or sizes[component] > max_cells:
            continue
        rows, cols = slices
        if rows.start == 0 or cols.start == 0 or rows.stop == height or cols.stop == width:
            continue
        region = (slice(rows.start - 1, rows.stop + 1), slice(cols.start - 1, cols.stop + 1))
        local = output[region]
        points = np.argwhere(np.isfinite(local))
        missing = np.argwhere(components[region] == component)
        try:
            values = griddata(points, local[tuple(points.T)], missing, method="linear")
        except QhullError:
            continue
        usable = np.isfinite(values)
        local[tuple(missing[usable].T)] = values[usable]
        filled += int(usable.sum())
    warnings = []
    if filled:
        warnings.append(
            f"Interpolated {filled} DEM hole pixels in enclosed components "
            f"of at most {max_cells} cells."
        )
    remaining = int((~np.isfinite(output)).sum())
    if remaining:
        warnings.append(
            f"{remaining} DEM pixels remain missing; "
            "no large holes or missing edge coverage were invented."
        )
    return output, warnings


def fetch_dem(
    bbox: BBox, buffer_km: float = 5, client: Any = None,
    collection: str = "cop-dem-glo-30",
) -> DEMData:
    """Window-read and mosaic every intersecting tile on its native geographic grid."""
    bounds = expanded_bbox(bbox, buffer_km)
    if client is None:
        client = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = list(client.search(collections=[collection], bbox=list(bounds)).items())
    if not items:
        raise ValueError("No DEM tiles intersect the buffered bbox")
    items.sort(key=lambda item: item["id"] if isinstance(item, dict) else item.id)
    ids, warnings = [], []
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_CONNECTTIMEOUT="15", GDAL_HTTP_TIMEOUT="120",
    ), ExitStack() as opened:
        sources = []
        for item in items:
            item_id = item["id"] if isinstance(item, dict) else item.id
            assets = item.get("assets", {}) if isinstance(item, dict) else item.assets
            if "data" not in assets:
                raise ValueError(f"DEM tile {item_id} has no data asset")
            asset = assets["data"]
            href = asset["href"] if isinstance(asset, dict) else asset.href
            source = opened.enter_context(rasterio.open(href))
            if source.crs != CRS.from_epsg(4326) or source.transform.b or source.transform.d:
                raise ValueError("Native DEM tiles must have an unrotated EPSG:4326 grid")
            sources.append(source)
            ids.append(item_id)
        spacing = min(source.res[0] for source in sources), min(source.res[1] for source in sources)
        if any(not np.allclose(source.res, spacing) for source in sources):
            warnings.append(
                "DEM tile spacings differ; the native mosaic uses the finest "
                "spacing before 30 m hydrology."
            )
        origin = sources[0].transform
        west = origin.c + math.floor((bounds[0] - origin.c) / spacing[0]) * spacing[0]
        east = origin.c + math.ceil((bounds[2] - origin.c) / spacing[0]) * spacing[0]
        south = origin.f - math.ceil((origin.f - bounds[1]) / spacing[1]) * spacing[1]
        north = origin.f - math.floor((origin.f - bounds[3]) / spacing[1]) * spacing[1]
        mosaic, affine = merge(
            sources, bounds=(west, south, east, north), res=spacing,
            nodata=np.nan, dtype="float32", method="first", resampling=Resampling.bilinear,
        )
    elevation, hole_warnings = fill_small_holes(mosaic[0])
    if not np.isfinite(elevation).any():
        raise ValueError("The buffered DEM contains no valid elevation pixels")
    return DEMData(elevation, affine, CRS.from_epsg(4326), [*warnings, *hole_warnings], ids)


def compute_slope(dem: np.ndarray, resolution: float) -> np.ndarray:
    """Return slope angle in degrees using meter-based centered differences."""
    _positive(resolution, "resolution")
    if dem.ndim != 2 or min(dem.shape) < 2:
        raise ValueError("Slope requires a two-dimensional DEM with at least two cells per side")
    dy, dx = np.gradient(dem.astype("float64"), resolution)
    slope = np.degrees(np.arctan(np.hypot(dx, dy))).astype("float32")
    slope[~np.isfinite(dem)] = np.nan
    return slope


def _priority_flood(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill depressions and retain acyclic outlet routes across exact flats."""
    height, width = dem.shape
    valid = np.isfinite(dem)
    filled = dem.astype("float64").copy()
    edges = valid & ~binary_erosion(valid, structure=np.ones((3, 3)), border_value=0)
    visited = edges.copy()
    parents = np.full(dem.size, -1, dtype="int64")
    indices = np.flatnonzero(edges)
    queue = [(float(filled.flat[index]), int(index)) for index in indices]
    heapq.heapify(queue)
    while queue:
        level, index = heapq.heappop(queue)
        row, col = divmod(index, width)
        for dr, dc in NEIGHBORS:
            r, c = row + dr, col + dc
            if r < 0 or r >= height or c < 0 or c >= width or visited[r, c] or not valid[r, c]:
                continue
            visited[r, c] = True
            neighbor = r * width + c
            value = max(level, float(filled[r, c]))
            filled[r, c] = value
            parents[neighbor] = index
            heapq.heappush(queue, (value, neighbor))
    return filled, parents


def _resolve_flats(filled: np.ndarray, routes: np.ndarray, steepest: np.ndarray) -> None:
    """Route equal-height flats toward their nearest lower exit by breadth first search."""
    height, width = filled.shape
    valid = np.isfinite(filled)
    assigned = steepest > 0
    flats = valid & ~assigned
    seeds = assigned & binary_dilation(flats, structure=np.ones((3, 3)))
    queue = deque(int(index) for index in np.flatnonzero(seeds))
    while queue:
        index = queue.popleft()
        row, col = divmod(index, width)
        for dr, dc in NEIGHBORS:
            r, c = row + dr, col + dc
            if (
                0 <= r < height and 0 <= c < width and valid[r, c]
                and not assigned[r, c] and filled[r, c] == filled[row, col]
            ):
                assigned[r, c] = True
                routes[r, c] = index
                queue.append(r * width + c)


def _accumulate(receivers: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Accumulate contributing cells in topological order, checking for cycles."""
    targets = receivers[receivers >= 0]
    indegree = np.bincount(targets, minlength=receivers.size)
    queue = deque(int(index) for index in np.flatnonzero(valid & (indegree == 0)))
    accumulation = valid.astype("float64")
    order = []
    while queue:
        index = queue.popleft()
        order.append(index)
        target = receivers[index]
        if target >= 0:
            accumulation[target] += accumulation[index]
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(int(target))
    if len(order) != int(valid.sum()):
        raise ValueError("D8 routing contains a cycle; hydrology cannot be computed honestly")
    return accumulation, order


def d8_hydrology(dem: np.ndarray, resolution: float, stream_threshold_km2: float) -> HANDResult:
    """Route D8 slopes, resolve flats, accumulate upstream area and trace streams."""
    filled, receivers = _priority_flood(dem)
    height, width = dem.shape
    indices = np.arange(dem.size, dtype="int64").reshape(dem.shape)
    steepest = np.zeros(dem.shape, dtype="float64")
    routes = receivers.reshape(dem.shape)
    for dr, dc in NEIGHBORS:
        rows = slice(max(0, -dr), min(height, height - dr))
        cols = slice(max(0, -dc), min(width, width - dc))
        neighbors = (slice(max(0, dr), min(height, height + dr)),
                     slice(max(0, dc), min(width, width + dc)))
        slope = (filled[rows, cols] - filled[neighbors]) / (resolution * math.hypot(dr, dc))
        better = slope > steepest[rows, cols]
        routes[rows, cols][better] = indices[neighbors][better]
        steepest[rows, cols][better] = slope[better]
    _resolve_flats(filled, routes, steepest)
    accumulation, order = _accumulate(receivers, np.isfinite(dem).ravel())
    streams = (accumulation * (resolution ** 2 / 1_000_000) >= stream_threshold_km2)
    streams &= np.isfinite(dem).ravel()
    drainage = np.full(dem.size, -1, dtype="int64")
    for index in reversed(order):
        if streams[index]:
            drainage[index] = index
        elif receivers[index] >= 0:
            drainage[index] = drainage[receivers[index]]
    hand = np.full(dem.size, np.nan, dtype="float32")
    reached = drainage >= 0
    values = dem.ravel()
    hand[reached] = values[reached] - values[drainage[reached]]
    warnings = [BACKEND_WARNING]
    raised = np.isfinite(dem) & (filled > dem + 1e-6)
    if raised.any():
        warnings.append(
            f"Priority-flood raised {int(raised.sum())} depression pixels for routing "
            f"(maximum fill {float((filled - dem)[raised].max()):.2f} m); reported elevation remains original."
        )
    negative = np.isfinite(hand) & (hand < 0)
    if negative.any():
        warnings.append(
            f"{int(negative.sum())} original elevations lay below their downstream drainage "
            "after depression filling; negative HAND values were clamped to zero."
        )
        hand[negative] = 0
    unknown = int((~np.isfinite(dem)).sum())
    if unknown:
        warnings.append(
            f"{unknown} missing/outside-coverage DEM pixels act as drainage "
            "boundaries; flow paths can be truncated."
        )
    unreached = int((np.isfinite(dem).ravel() & ~reached).sum())
    if unreached:
        warnings.append(f"{unreached} valid hydrology pixels never reach a threshold stream; their HAND is NaN.")
    return HANDResult(hand.reshape(dem.shape), streams.reshape(dem.shape), warnings)


def compute_hand(
    dem: np.ndarray, resolution: float, stream_threshold_km2: float = 1.0,
) -> HANDResult:
    """Compute flow-path HAND; smaller area thresholds produce more streams."""
    _positive(resolution, "resolution")
    _positive(stream_threshold_km2, "stream_threshold_km2")
    if dem.ndim != 2 or min(dem.shape) < 2:
        raise ValueError("HAND requires a two-dimensional DEM with at least two cells per side")
    return d8_hydrology(dem, resolution, stream_threshold_km2)


def _resample(
    array: np.ndarray, source_transform: Affine, source_crs: CRS,
    target_transform: Affine, target_crs: CRS, shape: tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Warp a continuous layer or categorical mask onto an explicit grid."""
    output = np.full(shape, np.nan, dtype="float32")
    reproject(
        source=array.astype("float32"), destination=output, src_transform=source_transform,
        src_crs=source_crs, src_nodata=np.nan, dst_transform=target_transform,
        dst_crs=target_crs, dst_nodata=np.nan, resampling=resampling,
    )
    return output


def save_terrain(terrain: TerrainStack, path: str | Path) -> None:
    """Save four float32 bands, descriptions, units, and the JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", count=4, height=terrain.elevation.shape[0],
        width=terrain.elevation.shape[1], dtype="float32", crs=terrain.crs,
        transform=terrain.transform, nodata=np.nan, tiled=True, compress="deflate",
    ) as dst:
        for index, (array, name, unit) in enumerate(zip(
            (terrain.elevation, terrain.slope_deg, terrain.hand_m, terrain.stream_mask),
            BAND_NAMES, ("m", "degree", "m", "1"),
        ), 1):
            dst.write(array.astype("float32"), index)
            dst.set_band_description(index, name)
            dst.set_band_unit(index, unit)
    path.with_suffix(".json").write_text(json.dumps(terrain.to_dict(), indent=2, allow_nan=False))


def load_terrain(path: str | Path) -> TerrainStack:
    """Restore terrain arrays and provenance from the cached pair of files."""
    path = Path(path)
    if path.is_dir():
        path = path / "terrain.tif"
    metadata = json.loads(path.with_suffix(".json").read_text())
    with rasterio.open(path) as src:
        if src.count != 4 or src.descriptions != BAND_NAMES:
            raise ValueError("Terrain cache must contain four correctly described bands")
        arrays = src.read(masked=True).astype("float32").filled(np.nan)
        affine, crs = src.transform, src.crs
    return TerrainStack(
        elevation=arrays[0], slope_deg=arrays[1], hand_m=arrays[2],
        stream_mask=np.isfinite(arrays[3]) & (arrays[3] > 0.5),
        transform=affine, crs=crs, bbox=tuple(metadata["bbox"]), source=metadata["source"],
        hydro_resolution=metadata["hydro_resolution"],
        stream_threshold_km2=metadata["stream_threshold_km2"], buffer_km=metadata["buffer_km"],
        warnings=metadata["warnings"], tile_ids=metadata["tile_ids"], cache_path=str(path.resolve()),
    )


def write_quicklook(terrain: TerrainStack, path: str | Path) -> None:
    """Render elevation in grayscale with threshold-derived streams in blue."""
    elevation = terrain.elevation
    valid = np.isfinite(elevation)
    gray = np.zeros(elevation.shape, dtype="uint8")
    if valid.any():
        low, high = np.quantile(elevation[valid], [0.02, 0.98])
        gray[valid] = 1 + (np.clip((elevation[valid] - low) / max(high - low, 1), 0, 1) * 254).astype("uint8")
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    rgb[terrain.stream_mask] = (0, 100, 255)
    rows = np.zeros((rgb.shape[0], rgb.shape[1] * 3 + 1), dtype="uint8")
    rows[:, 1:] = rgb.reshape(rgb.shape[0], -1)

    def chunk(kind: bytes, value: bytes) -> bytes:
        return struct.pack("!I", len(value)) + kind + value + struct.pack("!I", zlib.crc32(kind + value))

    header = struct.pack("!2I5B", rgb.shape[1], rgb.shape[0], 8, 2, 0, 0, 0)
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows.tobytes())) + chunk(b"IEND", b"")
    )


def build_terrain(
    aoi: AOI, stack: S1Stack, cache_dir: str | Path = "data/cache", client: Any = None,
    buffer_km: float = 5, hydro_resolution: float = 30, stream_threshold_km2: float = 1.0,
) -> TerrainStack:
    """Run buffered native-resolution hydrology, then align terrain to the radar grid."""
    _positive(buffer_km, "buffer_km", allow_zero=True)
    _positive(hydro_resolution, "hydro_resolution")
    _positive(stream_threshold_km2, "stream_threshold_km2")
    if hydro_resolution < 30:
        raise ValueError(
            "hydro_resolution must be at least 30 m; "
            "the DEM does not support finer hydrologic detail"
        )
    if not stack.crs.is_projected or stack.crs.linear_units_factor[1] != 1:
        raise ValueError("The S1 grid CRS must be projected in meters")
    if tuple(aoi.bbox) != tuple(stack.bbox):
        raise ValueError("AOI bbox must match the S1 stack bbox")
    shape = stack.array.shape[-2:]
    inputs = {
        "version": 1, "bbox": aoi.bbox, "buffer_km": buffer_km,
        "hydro_resolution": hydro_resolution, "stream_threshold_km2": stream_threshold_km2,
        "transform": list(stack.transform)[:6], "crs": stack.crs.to_string(), "shape": list(shape),
    }
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(cache_dir).resolve() / "terrain" / key / "terrain.tif"
    if path.exists() and path.with_suffix(".json").exists():
        terrain = load_terrain(path)
        if not path.with_name("quicklook.png").exists():
            write_quicklook(terrain, path.with_name("quicklook.png"))
        return terrain
    dem = fetch_dem(aoi.bbox, buffer_km, client=client)
    native_bounds = array_bounds(*dem.elevation.shape, dem.transform)
    bounds = transform_bounds(dem.crs, stack.crs, *native_bounds, densify_pts=21)
    left = math.floor(bounds[0] / hydro_resolution) * hydro_resolution
    bottom = math.floor(bounds[1] / hydro_resolution) * hydro_resolution
    right = math.ceil(bounds[2] / hydro_resolution) * hydro_resolution
    top = math.ceil(bounds[3] / hydro_resolution) * hydro_resolution
    hydro_transform = from_origin(left, top, hydro_resolution, hydro_resolution)
    hydro_shape = (round((top - bottom) / hydro_resolution), round((right - left) / hydro_resolution))
    metric = _resample(dem.elevation, dem.transform, dem.crs, hydro_transform, stack.crs, hydro_shape)
    hydrology = compute_hand(metric, hydro_resolution, stream_threshold_km2)
    slope = compute_slope(metric, hydro_resolution)
    elevation, slope_deg, hand = [
        _resample(layer, hydro_transform, stack.crs, stack.transform, stack.crs, shape)
        for layer in (metric, slope, hydrology.hand_m)
    ]
    streams = _resample(
        hydrology.stream_mask, hydro_transform, stack.crs, stack.transform, stack.crs,
        shape, Resampling.nearest,
    ) > 0.5
    reachable = _resample(
        np.isfinite(hydrology.hand_m), hydro_transform, stack.crs, stack.transform, stack.crs,
        shape, Resampling.nearest,
    ) > 0.5
    hand[~reachable] = np.nan
    hand[streams] = 0
    missing_elevation = ~np.isfinite(elevation)
    slope_deg[missing_elevation] = np.nan
    hand[missing_elevation] = np.nan
    streams[missing_elevation] = False
    warnings = [*aoi.warnings, *dem.warnings, *hydrology.warnings, SURFACE_WARNING]
    spacing = math.hypot(stack.transform.a, stack.transform.d)
    if spacing < 30:
        warnings.append(f"DEM is native ~30 m; resampling onto the {spacing:g} m S1 grid adds no terrain detail.")
    if hydro_resolution > 30:
        warnings.append(f"Hydrology was coarsened from native ~30 m to {hydro_resolution:g} m.")
    warnings.extend([
        f"Flow accumulation is limited to the {buffer_km:g} km buffered DEM; "
        "upstream catchments beyond it are truncated.",
        "Stream mask is an area-threshold D8 drainage network, not a mapped water extent.",
    ])
    for name, layer in (("elevation", elevation), ("slope", slope_deg), ("HAND", hand)):
        missing = int((~np.isfinite(layer)).sum())
        if missing:
            warnings.append(
                f"{name} is NaN for {missing}/{layer.size} output pixels "
                f"({missing / layer.size:.2%})."
            )
    terrain = TerrainStack(
        elevation=elevation, slope_deg=slope_deg, hand_m=hand, stream_mask=streams,
        transform=stack.transform, crs=stack.crs, bbox=aoi.bbox, source="cop-dem-glo-30",
        hydro_resolution=hydro_resolution, stream_threshold_km2=stream_threshold_km2,
        buffer_km=buffer_km, warnings=list(dict.fromkeys(warnings)),
        tile_ids=dem.tile_ids, cache_path=str(path),
    )
    save_terrain(terrain, path)
    write_quicklook(terrain, path.with_name("quicklook.png"))
    return terrain


def main() -> None:
    """Build terrain from live metadata and print statistics and cache locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    area = parser.add_mutually_exclusive_group(required=True)
    area.add_argument("--place")
    area.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--date", required=True, dest="flood_date")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--buffer-km", type=float, default=5)
    parser.add_argument("--hydro-resolution", type=float, default=30)
    parser.add_argument("--stream-threshold-km2", type=float, default=1.0)
    args = parser.parse_args()
    started = time.perf_counter()
    aoi = build_aoi(place=args.place, bbox=args.bbox, flood_date=args.flood_date)
    stack = preprocess_pair(aoi, find_best_pair(aoi), cache_dir=args.cache_dir)
    terrain = build_terrain(
        aoi, stack, cache_dir=args.cache_dir, buffer_km=args.buffer_km,
        hydro_resolution=args.hydro_resolution, stream_threshold_km2=args.stream_threshold_km2,
    )
    metadata = terrain.to_dict()
    metadata["runtime_seconds"] = time.perf_counter() - started
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
