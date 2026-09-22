from __future__ import annotations

import numpy as np


def volatility_threshold_from_training(
    values,
    *,
    quantile: float = 0.75,
    default: float = 0.02,
) -> float:
    """Derive a regime volatility threshold from training observations only."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float(default)
    q = float(np.clip(quantile, 0.0, 1.0))
    threshold = float(np.quantile(array, q))
    if not np.isfinite(threshold):
        return float(default)
    return threshold


def aggregate_oos_training_thresholds(
    thresholds,
    *,
    default: float = 0.02,
) -> float:
    """Aggregate fold-local train-only thresholds without reading OOS values."""
    array = np.asarray(thresholds, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float(default)
    threshold = float(np.median(array))
    if not np.isfinite(threshold):
        return float(default)
    return threshold
