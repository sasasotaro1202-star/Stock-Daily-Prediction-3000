from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SYMBOL_FALLBACK_FEATURES = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "volatility_20",
    "volume_ratio_20",
    "price_vs_sma60",
    "trend_r2_20",
    "gap_pct",
    "return_z20",
)


def make_symbol_logistic_model():
    """Strongly regularized symbol-level expert for the research fallback."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=0.25,
            max_iter=1200,
            class_weight="balanced",
            random_state=42,
        ),
    )


def blend_probabilities(
    probabilities,
    weights,
    *,
    minimum: float = 1e-5,
    maximum: float = 1.0 - 1e-5,
) -> float:
    probs = np.asarray(probabilities, dtype=float)
    w = np.asarray(weights, dtype=float)
    if probs.ndim != 1 or w.ndim != 1 or len(probs) != len(w) or len(probs) == 0:
        raise ValueError("probabilities and weights must be aligned 1-D arrays")
    if not np.isfinite(probs).all() or not np.isfinite(w).all():
        raise ValueError("probabilities and weights must be finite")
    if (w < 0).any() or float(w.sum()) <= 0.0:
        raise ValueError("weights must be non-negative with positive sum")
    probs = np.clip(probs, minimum, maximum)
    normalized = w / float(w.sum())
    return float(np.clip(np.dot(probs, normalized), minimum, maximum))
