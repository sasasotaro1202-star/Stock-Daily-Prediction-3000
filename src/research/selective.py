from __future__ import annotations

import numpy as np


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
    base = float(np.clip(base_rate, 1e-5, 1.0 - 1e-5))
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    low = np.abs(p - 0.5) < float(confidence_threshold)
    shrunk = (
        float(retained_weight) * p
        + (1.0 - float(retained_weight)) * base
    )
    return np.where(low, shrunk, p)
