from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

SELECTIVE_CONFIDENCE_THRESHOLDS = (0.03, 0.05, 0.075, 0.10, 0.15)
SELECTIVE_RETAINED_WEIGHTS = (0.25, 0.50, 0.75, 1.0)


def apply_confidence_shrinkage(
    probability,
    *,
    base_rate: float,
    confidence_threshold: float,
    retained_weight: float,
) -> np.ndarray:
    """Shrink only low-confidence probabilities toward a causal base rate."""
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(float(base_rate)):
        raise ValueError("base_rate must be finite")
    if not 0.0 <= float(confidence_threshold) <= 0.5:
        raise ValueError("confidence_threshold must be in [0, 0.5]")
    if not 0.0 <= float(retained_weight) <= 1.0:
        raise ValueError("retained_weight must be in [0, 1]")
    if p.ndim != 1:
        raise ValueError("probability must be a one-dimensional array")
    if not np.isfinite(p).all():
        raise ValueError("probability must contain only finite values")
    base = float(np.clip(base_rate, 1e-5, 1.0 - 1e-5))
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    low = np.abs(p - 0.5) < float(confidence_threshold)
    shrunk = (
        float(retained_weight) * p
        + (1.0 - float(retained_weight)) * base
    )
    return np.where(low, shrunk, p)


def select_confidence_shrinkage_parameters(
    y_calibration,
    calibrated_probability,
    *,
    base_rate: float,
) -> dict[str, float]:
    """Select shrinkage settings using only the fold-local calibration slice."""
    y = np.asarray(y_calibration, dtype=int)
    p = np.asarray(calibrated_probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("calibration labels/probabilities must be aligned 1-D arrays")
    if len(y) < 20:
        raise ValueError("calibration slice is too small for selective tuning")
    if len(np.unique(y)) < 2:
        raise ValueError("calibration labels must contain both classes")
    if not np.isfinite(p).all():
        raise ValueError("calibration probabilities must be finite")
    baseline = np.clip(p, 1e-5, 1.0 - 1e-5)
    candidates = []
    for threshold in SELECTIVE_CONFIDENCE_THRESHOLDS:
        for weight in SELECTIVE_RETAINED_WEIGHTS:
            adjusted = apply_confidence_shrinkage(
                baseline,
                base_rate=base_rate,
                confidence_threshold=threshold,
                retained_weight=weight,
            )
            ll = float(log_loss(y, adjusted, labels=[0, 1]))
            brier = float(brier_score_loss(y, adjusted))
            candidates.append(
                (ll, brier, -float(weight), float(threshold), float(weight))
            )
    best = min(candidates)
    return {
        "confidence_threshold": best[3],
        "retained_weight": best[4],
        "validation_logloss": best[0],
        "validation_brier": best[1],
    }
