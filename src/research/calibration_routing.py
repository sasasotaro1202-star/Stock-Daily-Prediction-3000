from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


CALIBRATION_METHODS = ("platt", "beta", "isotonic", "temperature")


def select_temporal_calibration_method(
    history: Mapping[str, Sequence[Mapping[str, float]]],
    current_fold: int,
    *,
    min_history_folds: int = 2,
    half_life_folds: float = 4.0,
    stability_penalty: float = 0.25,
    default_method: str = "platt",
) -> str:
    """Select a calibration family using only prior chronological OOS folds.

    Each history row must contain fold, logloss, ece and brier. Recent folds
    receive exponentially larger weight. The current fold must not be present
    in history; callers update history only after scoring the current fold.
    """
    if min_history_folds < 1:
        raise ValueError("min_history_folds must be >= 1")
    if not np.isfinite(half_life_folds) or half_life_folds <= 0:
        raise ValueError("half_life_folds must be > 0")
    if not np.isfinite(stability_penalty) or stability_penalty < 0:
        raise ValueError("stability_penalty must be >= 0")
    if default_method not in CALIBRATION_METHODS:
        raise ValueError(f"unsupported default calibration method: {default_method}")

    candidates = {}
    for method in CALIBRATION_METHODS:
        rows = []
        for row in history.get(method, ()):
            try:
                fold = int(row["fold"])
                logloss = float(row["logloss"])
                ece = float(row["ece"])
                brier = float(row["brier"])
            except (KeyError, TypeError, ValueError):
                continue
            if fold >= current_fold:
                raise ValueError(
                    "temporal calibration history contains a non-prior fold"
                )
            if not all(np.isfinite(x) for x in (logloss, ece, brier)):
                continue
            rows.append((fold, logloss, ece, brier))

        if len(rows) < min_history_folds:
            continue

        folds = np.asarray([row[0] for row in rows], dtype=float)
        logloss = np.asarray([row[1] for row in rows], dtype=float)
        ece = np.asarray([row[2] for row in rows], dtype=float)
        brier = np.asarray([row[3] for row in rows], dtype=float)
        distance = np.maximum(float(current_fold) - folds, 0.0)
        weights = np.power(0.5, distance / float(half_life_folds))
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0 or not math.isfinite(weight_sum):
            continue

        mean_logloss = float(np.average(logloss, weights=weights))
        variance = float(
            np.average((logloss - mean_logloss) ** 2, weights=weights)
        )
        score = mean_logloss + stability_penalty * math.sqrt(max(variance, 0.0))
        candidates[method] = {
            "score": score,
            "ece": float(np.average(ece, weights=weights)),
            "brier": float(np.average(brier, weights=weights)),
        }

    if not candidates:
        return default_method

    return min(
        candidates,
        key=lambda method: (
            candidates[method]["score"],
            candidates[method]["ece"],
            candidates[method]["brier"],
            CALIBRATION_METHODS.index(method),
        ),
    )
