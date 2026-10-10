"""Streaming valid-pixel confusion, water calibration, and threshold selection."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch


DATA_CAVEAT = (
    "Official val/test share regions with train; chip IDs alone are disjoint. "
    "Bolivia is the separate region holdout and the closest generalization measure, "
    "but is small (15 chips in the official dataset). Test and Bolivia are never merged. "
    "Water segmentation includes permanent water; it does not by itself identify new flooding."
)


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _batch(value: Any) -> np.ndarray:
    array = _numpy(value)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError("Expected (H,W), (B,H,W), or (B,1,H,W)")
    return array


def confusion_metrics(counts: Sequence[int]) -> dict[str, Any]:
    """Zero denominators give zero; no valid pixels gives None for all scores."""
    tp, fp, fn, tn = map(int, counts)
    total = tp + fp + fn + tn

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else (0.0 if total else None)

    return {"iou": ratio(tp, tp + fp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "accuracy": ratio(tp + tn, total), "valid_pixels": total,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def _counts(probability: np.ndarray, label: np.ndarray, threshold: float) -> np.ndarray:
    prediction, water = probability >= threshold, label == 1
    return np.array([(prediction & water).sum(), (prediction & ~water).sum(),
                     (~prediction & water).sum(), (~prediction & ~water).sum()], dtype=np.int64)


class MetricAccumulator:
    """O(bins + regions + thresholds) storage, independent of dataset pixel count.

    Reliability bins compare mean water probability with observed water frequency;
    ECE is their count-weighted absolute difference. Calibration is not a guarantee
    that a satellite observation establishes damage or operational safety.
    """

    def __init__(self, threshold: float = 0.5, thresholds: Sequence[float] | None = None,
                 n_bins: int = 10) -> None:
        values = np.asarray(list(thresholds) if thresholds is not None else np.linspace(.05, .95, 19))
        if (not 0 <= threshold <= 1 or n_bins < 1 or not len(values)
                or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))):
            raise ValueError("Thresholds must be finite in [0,1]; bins must be positive")
        self.threshold, self.thresholds, self.n_bins = threshold, values, n_bins
        self.counts = np.zeros(4, dtype=np.int64)
        self.sweep = np.zeros((len(values), 4), dtype=np.int64)
        self.regions: dict[str, np.ndarray] = {}
        self.bin_count = np.zeros(n_bins, dtype=np.int64)
        self.bin_probability = np.zeros(n_bins, dtype=np.float64)
        self.bin_water = np.zeros(n_bins, dtype=np.float64)
        self.ignored = 0

    def update(self, probabilities: Any, labels: Any, meta: dict[str, Any] | None = None) -> None:
        p, y = _batch(probabilities), _batch(labels)
        if p.shape != y.shape:
            raise ValueError("Probability and label shapes differ")
        regions = (meta or {}).get("region", ["unknown"] * len(p))
        if isinstance(regions, str):
            regions = [regions] * len(p)
        if len(regions) != len(p):
            raise ValueError("One region name is required per chip")
        for probability, label, region in zip(p, y, regions):
            valid = label != 255
            self.ignored += int((~valid).sum())
            probability, label = probability[valid], label[valid]
            if (not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))
                    or np.any((label != 0) & (label != 1))):
                raise ValueError("Valid pixels require probabilities in [0,1] and binary labels")
            counts = _counts(probability, label, self.threshold)
            self.counts += counts
            self.regions.setdefault(str(region), np.zeros(4, dtype=np.int64))[:] += counts
            for i, threshold in enumerate(self.thresholds):
                self.sweep[i] += _counts(probability, label, threshold)
            bins = np.minimum((probability * self.n_bins).astype(int), self.n_bins - 1)
            self.bin_count += np.bincount(bins, minlength=self.n_bins)
            self.bin_probability += np.bincount(bins, weights=probability, minlength=self.n_bins)
            self.bin_water += np.bincount(bins, weights=label, minlength=self.n_bins)

    def compute(self) -> dict[str, Any]:
        reliability = []
        error = 0.0
        total = int(self.bin_count.sum())
        for i, count in enumerate(self.bin_count):
            confidence = float(self.bin_probability[i] / count) if count else None
            frequency = float(self.bin_water[i] / count) if count else None
            if count:
                error += int(count) * abs(confidence - frequency)
            reliability.append({"lower": i / self.n_bins, "upper": (i + 1) / self.n_bins,
                                "count": int(count), "mean_probability": confidence,
                                "water_frequency": frequency})
        sweep = [{"threshold": float(t), **confusion_metrics(c)}
                 for t, c in zip(self.thresholds, self.sweep)]
        best = max(sweep, key=lambda entry: (entry["f1"] if entry["f1"] is not None else -1,
                                            -abs(entry["threshold"] - .5), -entry["threshold"]))
        return {**confusion_metrics(self.counts), "threshold": self.threshold,
                "ignored_pixels": self.ignored,
                "per_region": {region: confusion_metrics(c) for region, c in sorted(self.regions.items())},
                "calibration": {"ece": error / total if total else None, "reliability": reliability},
                "best_threshold": best["threshold"] if total else None, "threshold_sweep": sweep}


def threshold_sweep(probabilities: Any, labels: Any,
                    thresholds: Sequence[float] | None = None) -> dict[str, Any]:
    accumulator = MetricAccumulator(thresholds=thresholds)
    accumulator.update(probabilities, labels)
    result = accumulator.compute()
    return {"best_threshold": result["best_threshold"], "sweep": result["threshold_sweep"]}


def calibration(probabilities: Any, labels: Any, n_bins: int = 10) -> dict[str, Any]:
    accumulator = MetricAccumulator(n_bins=n_bins)
    accumulator.update(probabilities, labels)
    return accumulator.compute()["calibration"]
