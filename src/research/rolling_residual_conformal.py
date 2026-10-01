from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal upper quantile using a conservative rank."""
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1-D array")
    n = int(scores.size)
    rank = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), n)
    return float(np.partition(scores, rank - 1)[rank - 1])


def rolling_residual_conformal_interval(
    y_history: Iterable[float],
    point_history: Iterable[float],
    point_current: Iterable[float] | np.ndarray,
    *,
    alpha: float = 0.10,
    min_history: int = 200,
    window: int = 252,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Build symmetric return intervals from strictly prior pseudo-OOS residuals.

    The history must contain only outcomes known before the current forecast.
    Only the most recent window finite residuals are used.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0,1)")
    if int(min_history) < 20:
        raise ValueError("min_history must be >= 20")
    if int(window) < int(min_history):
        raise ValueError("window must be >= min_history")

    y = np.asarray(list(y_history), dtype=float)
    hist_pred = np.asarray(list(point_history), dtype=float)
    current = np.asarray(point_current, dtype=float)
    if y.ndim != 1 or hist_pred.ndim != 1 or current.ndim != 1:
        raise ValueError("all inputs must be 1-D")
    if len(y) != len(hist_pred):
        raise ValueError("history labels and predictions must align")
    if not np.isfinite(current).all():
        raise ValueError("current point predictions must be finite")

    finite = np.isfinite(y) & np.isfinite(hist_pred)
    residuals = np.abs(y[finite] - hist_pred[finite])
    if residuals.size < int(min_history):
        return np.full((len(current), 2), np.nan, dtype=float), {
            "status": "INSUFFICIENT_HISTORY",
            "history_rows": float(residuals.size),
            "window_rows": float(min(residuals.size, int(window))),
            "alpha": float(alpha),
            "mean_width": float("nan"),
        }

    residuals = residuals[-int(window):]
    radius = _conformal_quantile(residuals, float(alpha))
    lower = current - radius
    upper = current + radius
    interval = np.column_stack([lower, upper])
    return interval, {
        "status": "EXECUTED_PRIOR_OOS",
        "history_rows": float(len(y)),
        "window_rows": float(len(residuals)),
        "alpha": float(alpha),
        "radius": float(radius),
        "mean_width": float(2.0 * radius),
    }


def interval_score(
    y_true: Iterable[float],
    interval: np.ndarray,
    *,
    alpha: float = 0.10,
) -> float:
    """Compute the interval score for a central (1-alpha) prediction interval."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0,1)")
    y = np.asarray(list(y_true), dtype=float)
    arr = np.asarray(interval, dtype=float)
    if arr.ndim != 2 or arr.shape != (len(y), 2):
        raise ValueError("interval must have shape (n, 2)")
    if not np.isfinite(y).all() or not np.isfinite(arr).all():
        raise ValueError("interval score inputs must be finite")
    lower = np.minimum(arr[:, 0], arr[:, 1])
    upper = np.maximum(arr[:, 0], arr[:, 1])
    score = upper - lower
    below = y < lower
    above = y > upper
    scale = 2.0 / float(alpha)
    score = score + scale * np.where(below, lower - y, 0.0)
    score = score + scale * np.where(above, y - upper, 0.0)
    return float(np.mean(score))


def interval_diagnostics(
    y_true: Iterable[float],
    interval: np.ndarray,
    *,
    alpha: float = 0.10,
) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=float)
    arr = np.asarray(interval, dtype=float)
    if arr.ndim != 2 or arr.shape != (len(y), 2):
        raise ValueError("interval must have shape (n, 2)")
    lower = np.minimum(arr[:, 0], arr[:, 1])
    upper = np.maximum(arr[:, 0], arr[:, 1])
    contains = (y >= lower) & (y <= upper)
    misses = ~contains
    clipped = np.clip(y, lower, upper)
    severe_miss = misses & (np.abs(y - clipped) >= 0.05)
    return {
        "coverage": float(np.mean(contains)),
        "mean_width": float(np.mean(upper - lower)),
        "interval_score": interval_score(y, arr, alpha=alpha),
        "miss_rate": float(np.mean(misses)),
        "severe_miss_rate_ge_5pct": float(np.mean(severe_miss)),
        "n_test": float(len(y)),
    }


__all__ = [
    "rolling_residual_conformal_interval",
    "interval_score",
    "interval_diagnostics",
]
