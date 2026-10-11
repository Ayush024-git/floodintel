"""Classified pre/post water difference, terrain heuristics, and auditable exclusions."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import date
import hashlib
import inspect
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
from typing import Any
import zlib

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

try:
    from scipy import ndimage
except ImportError:
    ndimage = None

from src.model.predict import load_checkpoint, predict_probabilities, stack_to_model_input


DOMAIN_GAP = (
    "Model trained on Sen1Floods11 (post-flood chips, different processing chain); "
    "behaviour on terrain-corrected, Lee-filtered input is unvalidated"
)
DEBRIS_NOTE = (
    "candidate: Sentinel-1 backscatter change near drainage; cannot confirm debris; requires inspection"
)
FLOAT_BANDS = ("p_post", "p_pre", "change_db_vv", "change_db_vh", "post_vv_db")
MASK_BANDS = (
    "flood_mask", "confidence", "preexisting_water", "terrain_excluded", "debris_candidate",
    "valid_mask", "invalid_mask", "water_post", "water_pre", "new_water", "change_supports",
    "hand_missing", "slope_missing", "cleanup_removed", "holes_filled", "hole_fill_blocked",
    "debris_removed", "flood_vector_omitted", "debris_vector_omitted",
)


@dataclass
class FloodMap:
    """All classifications and exclusions on the original metric radar grid."""

    p_post: np.ndarray
    p_pre: np.ndarray
    flood_mask: np.ndarray
    confidence: np.ndarray
    preexisting_water: np.ndarray
    terrain_excluded: np.ndarray
    change_db_vv: np.ndarray
    change_db_vh: np.ndarray
    debris_candidate: np.ndarray
    valid_mask: np.ndarray
    invalid_mask: np.ndarray
    water_post: np.ndarray
    water_pre: np.ndarray
    new_water: np.ndarray
    change_supports: np.ndarray
    hand_missing: np.ndarray
    slope_missing: np.ndarray
    cleanup_removed: np.ndarray
    holes_filled: np.ndarray
    hole_fill_blocked: np.ndarray
    debris_removed: np.ndarray
    flood_vector_omitted: np.ndarray
    debris_vector_omitted: np.ndarray
    post_vv_db: np.ndarray
    transform: Affine
    crs: CRS
    resolution: float
    bbox: tuple[float, ...]
    stats: dict[str, Any]
    params: dict[str, Any]
    warnings: list[str]
    flags: list[dict[str, str]]
    cache_path: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.p_post.shape

    def to_dict(self) -> dict[str, Any]:
        """Metadata only; GeoTIFF bands contain the arrays."""
        return {
            "format_version": 1, "shape": list(self.shape), "crs": str(self.crs),
            "transform": list(self.transform)[:6], "resolution": self.resolution,
            "bbox": list(self.bbox), "stats": self.stats, "params": self.params,
            "warnings": self.warnings, "flags": self.flags, "cache_path": self.cache_path,
            "checkpoint_path": self.params["checkpoint_path"],
            "model_config": self.params["model_config"],
            "float_bands": list(FLOAT_BANDS), "mask_bands": list(MASK_BANDS),
            "storage": "flood.tif: all bands float32; flood_masks.tif: categorical bands uint8",
            "vector_area_basis": "Raster pixel area before metric-grid geometry simplification",
        }


def _bands(stack: Any) -> dict[str, np.ndarray]:
    if stack.array.ndim != 3 or len(stack.band_names) != stack.array.shape[0]:
        raise ValueError("Stack array and band_names do not match")
    bands = dict(zip(stack.band_names, stack.array))
    if len(bands) != len(stack.band_names):
        raise ValueError("Stack band names must be unique")
    for name in ("pre_vv", "pre_vh", "post_vv", "post_vh"):
        if name not in bands:
            raise ValueError(f"Missing required radar band: {name}")
    return bands


def pre_as_post(stack: Any) -> Any:
    """Independent two-band stack-like view; pre VV/VH occupy the model's post slots."""
    bands = _bands(stack)
    values = dict(vars(stack))
    values.update(array=np.stack((bands["pre_vv"], bands["pre_vh"])).astype(np.float32),
                  band_names=("post_vv", "post_vh"),
                  valid_mask=np.array(stack.valid_mask, dtype=bool, copy=True))
    return SimpleNamespace(**values)


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Eight-connected components, with a slower dependency-free fallback."""
    if ndimage is not None:
        return ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    labels = np.zeros(mask.shape, dtype=np.int32)
    count = 0
    height, width = mask.shape
    for y, x in np.argwhere(mask):
        if labels[y, x]:
            continue
        count += 1
        labels[y, x] = count
        queue = deque([(int(y), int(x))])
        while queue:
            row, col = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = row + dy, col + dx
                    if (0 <= ny < height and 0 <= nx < width
                            and mask[ny, nx] and not labels[ny, nx]):
                        labels[ny, nx] = count
                        queue.append((ny, nx))
    return labels, count


def _remove_small(mask: np.ndarray, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    labels, _ = _label(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum
    keep[0] = False
    cleaned = keep[labels]
    return cleaned, mask & ~cleaned


def _fill_holes(mask: np.ndarray, eligible: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if maximum == 0:
        return mask.copy(), np.zeros_like(mask), np.zeros_like(mask)
    labels, count = _label(~mask)
    sizes = np.bincount(labels.ravel())
    enclosed = sizes < maximum
    enclosed[0] = False
    edges = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    enclosed[edges] = False
    forbidden = np.bincount(labels[~eligible], minlength=count + 1) > 0
    fill = (enclosed & ~forbidden)[labels]
    blocked = (enclosed & forbidden)[labels]
    return mask | fill, fill, blocked


def _near_stream(stream: np.ndarray, buffer_m: float, resolution: float) -> np.ndarray:
    if not stream.any():
        return np.zeros(stream.shape, dtype=bool)
    if ndimage is not None:
        return ndimage.distance_transform_edt(~stream, sampling=resolution) <= buffer_m
    near = np.zeros_like(stream, dtype=bool)
    radius = math.ceil(buffer_m / resolution)
    height, width = stream.shape
    for dy in range(-min(radius, height - 1), min(radius, height - 1) + 1):
        for dx in range(-min(radius, width - 1), min(radius, width - 1) + 1):
            if math.hypot(dx, dy) * resolution > buffer_m:
                continue
            sy, sx = max(0, -dy), max(0, -dx)
            ty, tx = max(0, dy), max(0, dx)
            h, w = height - abs(dy), width - abs(dx)
            near[ty:ty + h, tx:tx + w] |= stream[sy:sy + h, sx:sx + w]
    return near


def _calibration_present(meta: dict[str, Any]) -> bool:
    """Diagnostics present is distinct from established transfer calibration."""
    if not isinstance(meta, dict):
        return False
    data = meta.get("calibration")
    if isinstance(data, dict):
        ece = data.get("ece")
        if isinstance(ece, (float, int)) and math.isfinite(ece):
            return True
        if any(bin.get("count", 0) > 0 for bin in data.get("reliability", []) if isinstance(bin, dict)):
            return True
    return any(_calibration_present(value) for value in meta.values() if isinstance(value, dict))


def _grid(stack: Any, terrain: Any) -> CRS:
    if stack.array.ndim != 3 or min(stack.array.shape[1:]) < 1:
        raise ValueError("S1 grid must have nonempty height and width")
    crs = CRS.from_user_input(stack.crs)
    if not crs.is_projected or not math.isclose(crs.linear_units_factor[1], 1):
        raise ValueError("Flood area and drainage buffers require a projected grid in meters")
    t, resolution = stack.transform, stack.resolution
    if (not math.isfinite(resolution) or resolution <= 0
            or not math.isclose(math.hypot(t.a, t.d), resolution, rel_tol=1e-6)
            or not math.isclose(math.hypot(t.b, t.e), resolution, rel_tol=1e-6)
            or not math.isclose(t.a * t.b + t.d * t.e, 0, abs_tol=1e-6)):
        raise ValueError("Stack transform must describe square, orthogonal metric pixels at its resolution")
    if (terrain.transform != t or CRS.from_user_input(terrain.crs) != crs
            or np.shape(stack.valid_mask) != stack.array.shape[1:]
            or any(np.shape(getattr(terrain, name)) != stack.array.shape[1:]
                   for name in ("hand_m", "slope_deg", "stream_mask"))):
        raise ValueError("Terrain must match the exact S1 transform, CRS, and shape")
    return crs


def build_flood_map(
    stack: Any, terrain: Any, model: Any, meta: dict[str, Any], *,
    threshold: float | None = None, pre_threshold: float | None = None,
    hand_max_m: float = 15.0, slope_max_deg: float = 20.0, change_db_min: float = 3.0,
    min_pixels: int = 50, fill_hole_pixels: int = 50, tta: bool = False,
    debris_buffer_m: float = 200.0, debris_slope_min_deg: float = 0.0,
    debris_slope_max_deg: float = 15.0, confidence_post_min: float = .8,
    confidence_pre_max: float = .2, terrain_exclusion_warning_fraction: float = .3,
    water_overlap_warning_fraction: float = .5, low_valid_caution: float = .9,
    low_valid_limiting: float = .6, tile: int = 512, overlap: int = 128,
    batch_size: int = 2, device: str | None = None, simplify_cells: float = 1.0,
    max_vector_features: int = 1000,
) -> FloodMap:
    """Subtract classified pre-water, gate terrain, and record every excluded pixel.

    Defaults are exploratory heuristics: HAND<=15 m, slope<=20°, darkening>=3 dB,
    50-pixel component floor and holes strictly smaller than 50 pixels. Holes are
    filled only if wholly valid, plausible, and not pre-water; additions are marked.
    Tier 3: flood AND post>=max(threshold,0.8) AND pre<0.2 AND VV/VH darkening AND
    plausible terrain. Tier 2: flood AND new water AND plausible AND (darkening OR
    post>=0.8). Other flood pixels are tier 1; non-flood is 0. These tiers are rules,
    not measured accuracy or calibrated probabilities. Defaults 0.8/0.2 are tunable.
    Debris is separate: non-post-water, absolute VV/VH change>=3 dB, within 200 m
    of mapped stream-cell centers, deposit slope 0..15°, and no flood overlap.
    """
    supplied_threshold, supplied_pre_threshold = threshold, pre_threshold
    threshold = meta.get("best_threshold") if threshold is None else threshold
    pre_threshold = threshold if pre_threshold is None else pre_threshold
    proportions = (threshold, pre_threshold, confidence_post_min, confidence_pre_max,
                   terrain_exclusion_warning_fraction, water_overlap_warning_fraction,
                   low_valid_caution, low_valid_limiting)
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in proportions):
        raise ValueError("Water/confidence/warning thresholds must be finite in [0,1]")
    if (any(not math.isfinite(x) or x < 0 for x in (hand_max_m, debris_buffer_m, simplify_cells))
            or not 0 <= slope_max_deg <= 90 or not 0 <= debris_slope_min_deg <= debris_slope_max_deg <= 90
            or not math.isfinite(change_db_min) or change_db_min <= 0
            or low_valid_limiting > low_valid_caution):
        raise ValueError("Invalid HAND/slope/change/buffer/simplification/validity thresholds")
    if (any(type(x) is not int or x < 0 for x in (min_pixels, fill_hole_pixels))
            or type(max_vector_features) is not int or max_vector_features < 1
            or type(tile) is not int or tile < 2 or type(overlap) is not int or not 0 <= overlap < tile
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError("Invalid component/hole/vector counts or tile/batch parameters")
    if len(meta.get("channels", [])) != 2 or set(meta["channels"]) != {"post_vv", "post_vh"}:
        raise ValueError("Pre/post subtraction requires a two-band post_vv/post_vh water model")
    bands = _bands(stack)
    crs = _grid(stack, terrain)
    post_input, post_valid = stack_to_model_input(stack, meta)
    pre_input, pre_valid = stack_to_model_input(pre_as_post(stack), meta)
    prediction_options = dict(tile=tile, overlap=overlap, batch_size=batch_size, device=device, tta=tta)
    p_post = (predict_probabilities(model, post_input, post_valid, **prediction_options)
              if post_valid.any() else np.full(post_valid.shape, np.nan, np.float32))
    p_pre = (predict_probabilities(model, pre_input, pre_valid, **prediction_options)
             if pre_valid.any() else np.full(pre_valid.shape, np.nan, np.float32))
    p_post, p_pre = np.array(p_post, dtype=np.float32, copy=True), np.array(p_pre, dtype=np.float32, copy=True)
    if p_post.shape != post_valid.shape or p_pre.shape != pre_valid.shape:
        raise ValueError("Probability rasters must match the S1 grid")
    valid = np.asarray(stack.valid_mask, bool) & post_valid & pre_valid & np.isfinite(p_post) & np.isfinite(p_pre)
    if np.any(valid & ((p_post < 0) | (p_post > 1) | (p_pre < 0) | (p_pre > 1))):
        raise ValueError("Finite probabilities must lie in [0,1]")
    p_post[~valid] = p_pre[~valid] = np.nan
    water_post, water_pre = valid & (p_post >= threshold), valid & (p_pre >= pre_threshold)
    preexisting = water_post & water_pre
    new = water_post & ~water_pre
    hand_missing = valid & ~np.isfinite(terrain.hand_m)
    slope_missing = valid & ~np.isfinite(terrain.slope_deg)
    plausible = (np.isfinite(terrain.hand_m) & np.isfinite(terrain.slope_deg)
                 & (terrain.hand_m <= hand_max_m) & (terrain.slope_deg <= slope_max_deg))
    excluded = new & ~plausible
    vv = np.subtract(bands["post_vv"], bands["pre_vv"], dtype=np.float32)
    vh = np.subtract(bands["post_vh"], bands["pre_vh"], dtype=np.float32)
    vv[~valid] = vh[~valid] = np.nan
    supports = valid & ((vv <= -change_db_min) | (vh <= -change_db_min))
    cleaned, removed = _remove_small(new & plausible & valid, min_pixels)
    flood, filled, blocked = _fill_holes(cleaned, valid & plausible & ~water_pre, fill_hole_pixels)
    removed &= ~flood
    blocked &= valid
    confidence = np.zeros(valid.shape, dtype=np.uint8)
    confidence[flood] = 1
    confidence[flood & new & plausible & (supports | (p_post >= confidence_post_min))] = 2
    confidence[flood & plausible & supports & (p_post >= max(threshold, confidence_post_min))
               & (p_pre < confidence_pre_max)] = 3
    near = _near_stream(np.asarray(terrain.stream_mask, bool), debris_buffer_m, stack.resolution)
    debris_raw = (valid & ~water_post & ~flood & ((np.abs(vv) >= change_db_min) | (np.abs(vh) >= change_db_min))
                  & near & np.isfinite(terrain.slope_deg) & (terrain.slope_deg >= debris_slope_min_deg)
                  & (terrain.slope_deg <= debris_slope_max_deg))
    debris, debris_removed = _remove_small(debris_raw, min_pixels)
    warnings = [*stack.warnings, *terrain.warnings, *meta.get("warnings", []),
                "Classified pre/post water difference can include seasonal water changes and model errors; "
                "pre-existing water is not a confirmed permanent-water inventory.",
                "Radar cannot reliably identify debris. " + DEBRIS_NOTE]
    flags = []

    def flag(code: str, severity: str, message: str) -> None:
        flags.append({"code": code, "severity": severity, "message": message})
        warnings.append(message)

    flag("domain_gap", "caution", DOMAIN_GAP)
    calibrated = _calibration_present(meta)
    if not calibrated:
        flag("probabilities_uncalibrated", "info", "Checkpoint contains no calibration diagnostics; probabilities are uncalibrated")
    else:
        warnings.append("Calibration diagnostics exist; they do not establish calibration after transfer to this incident")
    flag("terrain_heuristic", "info", "HAND/slope thresholds are heuristics, not validated for this region")
    count_new, count_post = int(new.sum()), int(water_post.sum())
    if count_new and excluded.sum() / count_new > terrain_exclusion_warning_fraction:
        flag("mountain_valley_hand_caution", "caution", f"Terrain excludes {excluded.sum() / count_new:.1%} of new-water pixels; "
             "mountain-valley HAND and surface-model biases can remove plausible inundation")
    fraction = float(valid.mean())
    inherited_fraction = min(fraction, float(np.asarray(stack.valid_mask, bool).mean()),
                             float(getattr(stack, "valid_fraction", fraction)))
    if inherited_fraction < low_valid_caution:
        flag("low_valid_fraction", "limiting" if inherited_fraction < low_valid_limiting else "caution",
             f"Common valid_fraction={fraction:.4f}; inherited stack fraction={inherited_fraction:.4f}")
    if not valid.any():
        flag("no_valid_pixels", "limiting", "No jointly valid pre/post probabilities; no flood extent can be assessed")
    if count_post and preexisting.sum() / count_post > water_overlap_warning_fraction:
        flag("pre_post_water_overlap_high", "caution", f"Pre-existing water is {preexisting.sum() / count_post:.1%} of classified post-water")
    if not flood.any():
        flag("no_flood_detected", "info", "No flood pixels remain under these thresholds; this does not establish absence of flooding")
    flag("debris_is_candidate_only", "info", DEBRIS_NOTE)
    if ndimage is None:
        flag("numpy_morphology_fallback", "caution", "scipy.ndimage unavailable; NumPy component/buffer fallback may be slow")
    labels, components = _label(flood)
    sizes = np.bincount(labels.ravel())[1:]
    pixel_km2 = float(stack.resolution ** 2 / 1e6)
    stats = {
        "flood_area_km2": float(flood.sum() * pixel_km2),
        "preexisting_water_km2": float(preexisting.sum() * pixel_km2),
        "terrain_excluded_km2": float(excluded.sum() * pixel_km2),
        "debris_candidate_km2": float(debris.sum() * pixel_km2),
        "valid_area_km2": float(valid.sum() * pixel_km2), "valid_fraction": fraction,
        "flood_fraction_of_valid": float(flood.sum() / valid.sum()) if valid.any() else None,
        "area_per_confidence_tier_km2": {name: float((confidence == tier).sum() * pixel_km2)
                                         for tier, name in ((1, "uncertain"), (2, "probable"), (3, "strong"))},
        "component_count": components, "largest_component_km2": float(sizes.max() * pixel_km2) if len(sizes) else 0.0,
        "hand_missing_px": int(hand_missing.sum()), "slope_missing_px": int(slope_missing.sum()),
        "new_water_pixels": count_new, "water_post_pixels": count_post,
        "water_pre_pixels": int(water_pre.sum()), "invalid_pixels": int((~valid).sum()),
        "preexisting_water_pixels": int(preexisting.sum()), "terrain_excluded_pixels": int(excluded.sum()),
        "cleanup_removed_pixels": int(removed.sum()), "cleanup_removed_km2": float(removed.sum() * pixel_km2),
        "holes_filled_pixels": int(filled.sum()), "hole_fill_blocked_pixels": int(blocked.sum()),
        "debris_removed_pixels": int(debris_removed.sum()), "debris_removed_km2": float(debris_removed.sum() * pixel_km2),
        "pixel_area_m2": float(stack.resolution ** 2),
    }
    params = {name: value for name, value in locals().copy().items()
              if name in _BUILD_SIGNATURE.parameters and name not in {"stack", "terrain", "model", "meta"}}
    params.update(threshold_source="checkpoint" if supplied_threshold is None else "override",
                  pre_threshold_source="post_threshold" if supplied_pre_threshold is None else "override",
                  connectivity=8, morphology_backend="scipy.ndimage" if ndimage is not None else "numpy",
                  checkpoint_path=meta.get("checkpoint_path"), checkpoint_hash=meta.get("checkpoint_hash"),
                  model_config=meta["model_config"], calibration_data_present=calibrated,
                  hole_fill_policy="Only enclosed holes wholly valid/plausible and not classified pre-water; additions are explicit",
                  debris_label=DEBRIS_NOTE, stream_distance_basis="Euclidean stream-cell center distance in meters")
    return FloodMap(
        p_post=p_post, p_pre=p_pre, flood_mask=flood, confidence=confidence,
        preexisting_water=preexisting, terrain_excluded=excluded, change_db_vv=vv, change_db_vh=vh,
        debris_candidate=debris, valid_mask=valid, invalid_mask=~valid, water_post=water_post,
        water_pre=water_pre, new_water=new, change_supports=supports, hand_missing=hand_missing,
        slope_missing=slope_missing, cleanup_removed=removed, holes_filled=filled,
        hole_fill_blocked=blocked, debris_removed=debris_removed,
        flood_vector_omitted=np.zeros_like(valid), debris_vector_omitted=np.zeros_like(valid),
        post_vv_db=np.array(bands["post_vv"], dtype=np.float32, copy=True),
        transform=stack.transform, crs=crs, resolution=float(stack.resolution), bbox=tuple(stack.bbox),
        stats=stats, params=params, warnings=list(dict.fromkeys(warnings)), flags=flags,
    )


_BUILD_SIGNATURE = inspect.signature(build_flood_map)


def _vectors(fm: FloodMap, mask: np.ndarray, kind: str) -> dict[str, Any]:
    """Capped component polygons; omitted components remain explicit raster layers."""
    labels, count = _label(mask)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    ranked = sorted(range(1, count + 1), key=lambda value: (-int(sizes[value]), value))
    selected = set(ranked[:fm.params["max_vector_features"]])
    omitted = mask & ~np.isin(labels, list(selected))
    setattr(fm, f"{kind}_vector_omitted", omitted)
    fm.stats[f"{kind}_vector_omitted_pixels"] = int(omitted.sum())
    fm.stats[f"{kind}_vector_omitted_km2"] = float(omitted.sum() * fm.resolution ** 2 / 1e6)
    fm.stats[f"{kind}_vector_omitted_components"] = count - len(selected)
    if omitted.any():
        code = f"{kind}_vector_feature_cap"
        message = f"{kind} vector feature cap omitted {count - len(selected)} components; raster masks retain all pixels"
        if not any(flag["code"] == code for flag in fm.flags):
            fm.flags.append({"code": code, "severity": "info", "message": message})
            fm.warnings.append(message)
    parts: dict[int, list[Any]] = {value: [] for value in selected}
    # Four-connected polygon pieces avoid self-touching rings; union groups them
    # into the original eight-connected component, including diagonal contacts.
    for geometry, value in shapes(labels.astype(np.int32), mask=mask & ~omitted,
                                  connectivity=4, transform=fm.transform):
        parts[int(value)].append(shape(geometry))
    confidence_counts = np.bincount((labels.ravel().astype(np.int64) * 4 + fm.confidence.ravel()),
                                    minlength=(count + 1) * 4).reshape(-1, 4)
    features = []
    for value in ranked[:len(selected)]:
        geometry = make_valid(unary_union(parts[value]))
        geometry = geometry.simplify(fm.resolution * fm.params["simplify_cells"], preserve_topology=True)
        projected = make_valid(shape(transform_geom(fm.crs, "EPSG:4326", mapping(geometry))))
        histogram = confidence_counts[value]
        cumulative = np.cumsum(histogram)
        middle = ((int(sizes[value]) - 1) // 2, int(sizes[value]) // 2)
        median = float(np.mean([np.searchsorted(cumulative, position, side="right") for position in middle]))
        features.append({
            "type": "Feature", "id": f"{kind}/{value}", "geometry": mapping(projected),
            "properties": {
                "component_id": value, "pixel_count": int(sizes[value]),
                "area_km2": float(sizes[value] * fm.resolution ** 2 / 1e6),
                "max_confidence": int(np.flatnonzero(histogram)[-1]) if kind == "flood" else None,
                "median_confidence": median if kind == "flood" else None,
                "label": "classified new water with terrain/cleanup heuristics" if kind == "flood" else DEBRIS_NOTE,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _quicklook(fm: FloodMap, path: Path) -> None:
    """Post VV background with explicitly labeled classifications and candidates."""
    colors = {1: (77, 166, 255), 2: (26, 102, 230), 3: (5, 31, 179)}
    gray = (np.nan_to_num(np.clip((fm.post_vv_db + 35) / 40, 0, 1), nan=0) * 255).astype(np.uint8)
    image = np.repeat(gray[..., None], 3, axis=2)
    image[~fm.valid_mask] = 0

    def blend(mask: np.ndarray, color: tuple[int, ...]) -> None:
        image[mask] = (.2 * image[mask] + .8 * np.array(color)).astype(np.uint8)

    blend(fm.preexisting_water, (166, 166, 166))
    blend(fm.terrain_excluded, (255, 128, 13))
    for tier, color in colors.items():
        blend(fm.confidence == tier, color)
    inside = fm.debris_candidate.copy()
    padded = np.pad(fm.debris_candidate, 1)
    for dy, dx in ((0, 1), (1, 0), (1, 2), (2, 1)):
        inside &= padded[dy:dy + fm.shape[0], dx:dx + fm.shape[1]]
    image[fm.debris_candidate & ~inside] = (255, 255, 0)
    # Bitmap text keeps the PNG independent of optional plotting/font packages.
    glyphs = {
        "A": (14, 17, 17, 31, 17, 17, 17), "B": (30, 17, 17, 30, 17, 17, 30),
        "C": (14, 17, 16, 16, 16, 17, 14), "D": (30, 17, 17, 17, 17, 17, 30),
        "E": (31, 16, 16, 30, 16, 16, 31), "F": (31, 16, 16, 30, 16, 16, 16),
        "G": (14, 17, 16, 23, 17, 17, 14), "H": (17, 17, 17, 31, 17, 17, 17),
        "I": (31, 4, 4, 4, 4, 4, 31), "L": (16, 16, 16, 16, 16, 16, 31),
        "N": (17, 25, 21, 19, 17, 17, 17), "O": (14, 17, 17, 17, 17, 17, 14),
        "P": (30, 17, 17, 30, 16, 16, 16), "R": (30, 17, 17, 30, 20, 18, 17),
        "S": (15, 16, 16, 14, 1, 1, 30), "T": (31, 4, 4, 4, 4, 4, 4),
        "U": (17, 17, 17, 17, 17, 17, 14), "V": (17, 17, 17, 17, 17, 10, 4),
        "W": (17, 17, 17, 21, 21, 21, 10), "Y": (17, 17, 10, 4, 4, 4, 4),
        "X": (17, 17, 10, 4, 10, 17, 17),
    }
    width = max(fm.shape[1], 640)
    scale = max(2, width // 800)
    row_height = 11 * scale
    canvas = np.full((fm.shape[0] + 6 * row_height, width, 3), 30, np.uint8)
    canvas[:fm.shape[0], :fm.shape[1]] = image

    def text(value: str, y: int, x: int) -> None:
        for letter in value:
            for row, bits in enumerate(glyphs.get(letter, (0,) * 7)):
                for column in range(5):
                    if bits & (1 << (4 - column)):
                        canvas[y + row * scale:y + (row + 1) * scale,
                               x + column * scale:x + (column + 1) * scale] = 240
            x += 6 * scale

    start = fm.shape[0] + scale
    text("WATER CHANGE HEURISTICS", start, 4 * scale)
    entries = [(colors[3], "STRONG RULE"), (colors[2], "PROBABLE RULE"),
               (colors[1], "UNCERTAIN RULE"), ((166, 166, 166), "PRE WATER"),
               ((255, 128, 13), "TERRAIN EXCLUDED"), ((255, 255, 0), "DEBRIS CANDIDATE")]
    for index, (color, label) in enumerate(entries):
        y = start + (index // 2 + 1) * row_height
        x = (index % 2) * (width // 2) + 4 * scale
        canvas[y:y + 7 * scale, x:x + 7 * scale] = color
        text(label, y, x + 10 * scale)
    text("DEBRIS CANDIDATE ONLY", start + 4 * row_height, 4 * scale)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack("!I", len(payload)) + kind + payload + struct.pack("!I", zlib.crc32(kind + payload))

    raw = b"".join(b"\x00" + row.tobytes() for row in canvas)
    header = struct.pack("!2I5B", width, len(canvas), 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def save_flood_map(fm: FloodMap, out_dir: str | Path) -> Path:
    """Save probabilities, categorical audit layers, polygons, metadata, and preview."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collections = {"flood_extent.geojson": _vectors(fm, fm.flood_mask, "flood"),
                   "debris_candidates.geojson": _vectors(fm, fm.debris_candidate, "debris")}
    fm.cache_path = str((out_dir / "flood.tif").resolve())
    profile = dict(driver="GTiff", height=fm.shape[0], width=fm.shape[1], crs=fm.crs,
                   transform=fm.transform, compress="deflate")
    # A GeoTIFF has one dtype for all bands. Store lossless float32 audit bands
    # together, and native uint8 categorical bands in an aligned companion file.
    for filename, names, dtype, nodata in (
        ("flood.tif", FLOAT_BANDS + MASK_BANDS, "float32", np.nan),
        ("flood_masks.tif", MASK_BANDS, "uint8", None),
    ):
        with rasterio.open(out_dir / filename, "w", count=len(names), dtype=dtype, nodata=nodata,
                           **profile) as dataset:
            for band, name in enumerate(names, 1):
                dataset.write(np.asarray(getattr(fm, name), dtype=dtype), band)
                dataset.set_band_description(band, name)
    for filename, collection in collections.items():
        (out_dir / filename).write_text(json.dumps(collection, allow_nan=False), encoding="utf-8")
    _quicklook(fm, out_dir / "quicklook.png")
    metadata = out_dir / "flood_meta.json"
    temporary = metadata.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(fm.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(metadata)
    return out_dir / "flood.tif"


def load_flood_map(path: str | Path) -> FloodMap:
    """Restore all raster dtypes and audit metadata from a saved map."""
    path = Path(path)
    if path.is_dir():
        path /= "flood.tif"
    if not path.is_file():
        raise FileNotFoundError(f"Flood raster missing: {path}")
    metadata = json.loads(path.with_name("flood_meta.json").read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported flood-map format version")
    with rasterio.open(path) as dataset:
        if (dataset.descriptions != FLOAT_BANDS + MASK_BANDS
                or [dataset.height, dataset.width] != metadata["shape"]
                or list(dataset.transform)[:6] != metadata["transform"]
                or dataset.crs != CRS.from_user_input(metadata["crs"])):
            raise ValueError("Flood raster grid/bands disagree with metadata")
        values = {name: dataset.read(index) for index, name in enumerate(dataset.descriptions, 1)}
        transform, crs = dataset.transform, dataset.crs
    for name in MASK_BANDS:
        value = values[name]
        allowed = (0, 1, 2, 3) if name == "confidence" else (0, 1)
        if not np.isin(value, allowed).all():
            raise ValueError(f"Invalid saved categorical values in {name}")
        values[name] = value.astype(np.uint8 if name == "confidence" else bool)
    return FloodMap(**values, transform=transform, crs=crs, resolution=metadata["resolution"],
                    bbox=tuple(metadata["bbox"]), stats=metadata["stats"], params=metadata["params"],
                    warnings=metadata["warnings"], flags=metadata["flags"], cache_path=str(path.resolve()))


def flood_cache_key(checkpoint_hash: str, params: dict[str, Any], inputs: str = "") -> str:
    """Content-addressed checkpoint, requested parameters, and input fingerprint."""
    payload = {"version": 1, "checkpoint_hash": checkpoint_hash, "params": params, "inputs": inputs}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_hash(stack: Any, terrain: Any) -> str:
    digest = hashlib.sha256(json.dumps({"crs": str(stack.crs), "transform": list(stack.transform)[:6],
                                     "resolution": stack.resolution, "bands": list(stack.band_names),
                                     "bbox": list(stack.bbox), "source": stack.source,
                                     "valid_fraction": stack.valid_fraction, "stack_warnings": stack.warnings,
                                     "terrain_crs": str(terrain.crs), "terrain_transform": list(terrain.transform)[:6],
                                     "terrain_warnings": terrain.warnings}, sort_keys=True).encode())
    for array in (stack.array, stack.valid_mask, terrain.hand_m, terrain.slope_deg, terrain.stream_mask):
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def analyze_incident_flood(incident_id_or_path: str | Path, checkpoint: str | Path, **params: Any) -> FloodMap:
    """Analyze cached incident layers only; reuse content-addressed map outputs."""
    from src.data.incident import load_incident, load_layers

    cache_dir = params.pop("cache_dir", "data/cache")
    incident = load_incident(incident_id_or_path, cache_dir=cache_dir)
    if incident.status in {"blocked", "failed"}:
        reason = incident.pair_summary.get("reason", incident.status)
        raise ValueError(f"Cannot analyze {incident.status} incident: {reason}")
    if not incident.layers.get("s1_stack"):
        raise ValueError("Incident has no S1 stack; run prepare_incident first")
    if not incident.layers.get("terrain"):
        raise ValueError("Incident has no terrain layers; run prepare_incident first")
    layers = load_layers(incident)
    stack, terrain = layers.s1_stack, layers.terrain
    bound = _BUILD_SIGNATURE.bind(stack, terrain, None, {}, **params)
    bound.apply_defaults()
    requested = {name: value for name, value in bound.arguments.items()
                 if name not in {"stack", "terrain", "model", "meta"}}
    checkpoint = Path(checkpoint)
    checksum = _file_hash(checkpoint)
    key = flood_cache_key(checksum, requested, _input_hash(stack, terrain))
    directory = (Path(incident.cache_path).parent if incident.cache_path else
                 Path(cache_dir) / "incidents" / incident.incident_id) / "flood" / key
    expected = ("flood.tif", "flood_masks.tif", "flood_meta.json", "flood_extent.geojson",
                "debris_candidates.geojson", "quicklook.png")
    if all((directory / name).is_file() for name in expected):
        return load_flood_map(directory)
    model, meta = load_checkpoint(checkpoint, device=requested["device"])
    meta = dict(meta, checkpoint_path=str(checkpoint.resolve()), checkpoint_hash=checksum)
    fm = build_flood_map(stack, terrain, model, meta, **params)
    fm.params.update(cache_key=key, incident_id=incident.incident_id)
    save_flood_map(fm, directory)
    return fm


def resolve_cached_incident(place: str, flood_date: str, cache_dir: str | Path = "data/cache") -> Any:
    """Resolve saved request/AOI names and dates without geocoding or fetching."""
    from src.data.aoi import AOI
    from src.data.incident import load_incident

    requested_date = date.fromisoformat(flood_date)
    matches = []
    for path in sorted((Path(cache_dir) / "incidents").glob("*/incident.json")):
        incident = load_incident(path)
        aoi = AOI.from_dict(incident.aoi) if incident.aoi else None
        request = incident.provenance.get("request", {})
        names = [request.get("place"), aoi.name if aoi else None]
        if (aoi and aoi.flood_date == requested_date
                and any(name and name.strip().casefold() == place.strip().casefold() for name in names)):
            matches.append(incident)
    if not matches:
        raise FileNotFoundError(f"No cached incident for {place} on {flood_date}; run prepare_incident first")
    if len(matches) != 1:
        raise ValueError("Multiple cached incidents match; use --incident with an explicit incident id")
    return matches[0]


def main() -> None:
    """Cached-only command-line flood analysis; never fetches missing inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--incident")
    selection.add_argument("--place")
    parser.add_argument("--date")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--pre-threshold", type=float)
    parser.add_argument("--hand-max", type=float, default=15)
    parser.add_argument("--slope-max", type=float, default=20)
    parser.add_argument("--change-db-min", type=float, default=3)
    parser.add_argument("--min-pixels", type=int, default=50)
    parser.add_argument("--fill-hole-pixels", type=int, default=50)
    parser.add_argument("--debris-buffer-m", type=float, default=200)
    parser.add_argument("--debris-slope-min", type=float, default=0)
    parser.add_argument("--debris-slope-max", type=float, default=15)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()
    try:
        if args.place and not args.date:
            raise ValueError("--date is required with --place")
        incident = (resolve_cached_incident(args.place, args.date, args.cache_dir).cache_path
                    if args.place else args.incident)
        fm = analyze_incident_flood(
            incident, args.checkpoint, cache_dir=args.cache_dir,
            threshold=args.threshold, pre_threshold=args.pre_threshold, hand_max_m=args.hand_max,
            slope_max_deg=args.slope_max, change_db_min=args.change_db_min, min_pixels=args.min_pixels,
            fill_hole_pixels=args.fill_hole_pixels, debris_buffer_m=args.debris_buffer_m,
            debris_slope_min_deg=args.debris_slope_min, debris_slope_max_deg=args.debris_slope_max,
            tile=args.tile, overlap=args.overlap, batch_size=args.batch_size, device=args.device, tta=args.tta,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps({"areas_km2": {name: value for name, value in fm.stats.items() if name.endswith("_area_km2")
                                  or name in {"preexisting_water_km2", "terrain_excluded_km2", "debris_candidate_km2"}},
                      "confidence_tiers_km2": fm.stats["area_per_confidence_tier_km2"],
                      "flags": [{"code": flag["code"], "severity": flag["severity"]} for flag in fm.flags],
                      "cache_path": fm.cache_path}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
