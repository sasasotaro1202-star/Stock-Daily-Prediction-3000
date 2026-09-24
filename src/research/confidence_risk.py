from __future__ import annotations

"""Research-only temporal confidence-risk layer.

The layer estimates the probability that a prediction will be incorrect using
only prior out-of-sample predictions and prediction-time context. It is a
meta-risk score, not a directional signal: high predicted error risk causes
bounded confidence shrinkage rather than an automatic contrarian flip.
"""

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


def _clip_probability(p) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    return np.clip(arr, EPS, 1.0 - EPS)


def confidence_risk_features(
    probability,
    expert_probabilities,
    context,
) -> np.ndarray:
    """Build finite, bounded meta-risk features from prediction-time inputs.

    Context columns, when present:
      0 volatility_20
      1 volume_ratio_20
      2 abs(gap_pct)
      3 breadth_distance_from_0.5
      4 market_dispersion_1d
      5 market_dispersion_vs_20d
      6 vix_level_lag1
      7 return_z20
      8 drawdown_from_high_20
      9 cs_ret_1d_rank
     10 cs_vol_rank
    """
    p = _clip_probability(probability)
    experts = _clip_probability(expert_probabilities)
    if p.ndim != 1:
        raise ValueError("probability must be 1-D")
    if experts.ndim != 2 or len(experts) != len(p) or experts.shape[1] < 2:
        raise ValueError("expert_probabilities must be 2-D with matching rows")
    ctx = np.asarray(context, dtype=float)
    if ctx.ndim != 2 or len(ctx) != len(p):
        raise ValueError("context must be 2-D with matching rows")

    confidence = np.abs(p - 0.5)
    entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    disagreement = np.std(experts, axis=1)
    expert_range = np.max(experts, axis=1) - np.min(experts, axis=1)
    direction_votes = np.mean(experts >= 0.5, axis=1)
    vote_conflict = np.minimum(direction_votes, 1.0 - direction_votes)

    cols = [confidence, entropy, disagreement, expert_range, vote_conflict]
    if ctx.shape[1]:
        for j in range(min(ctx.shape[1], 11)):
            cols.append(np.nan_to_num(ctx[:, j], nan=0.0, posinf=0.0, neginf=0.0))

    out = np.column_stack(cols)
    # Robustly clip only for the meta model; missing context was already made
    # explicit to the caller and is not interpreted as directional information.
    out = np.where(np.isfinite(out), out, 0.0)
    return np.clip(out, -10.0, 10.0)


def fit_temporal_confidence_risk(
    meta_features,
    correctness,
    *,
    min_rows: int = 240,
):
    """Fit a low-capacity correctness model on prior chronological OOS rows."""
    X = np.asarray(meta_features, dtype=float)
    y = np.asarray(correctness, dtype=int)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        return None
    if len(y) < int(min_rows) or len(np.unique(y)) < 2:
        return None
    if not np.isfinite(X).all():
        return None
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=20260925,
                ),
            ),
        ]
    ).fit(X, y)


def predicted_error_risk(model, meta_features) -> np.ndarray:
    X = np.asarray(meta_features, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("meta_features must be finite 2-D")
    if model is None:
        return np.full(len(X), 0.5, dtype=float)
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and int(expected) != X.shape[1]:
        return np.full(len(X), 0.5, dtype=float)
    return np.clip(1.0 - model.predict_proba(X)[:, 1], 0.0, 1.0)


def apply_confidence_risk_shrinkage(
    probability,
    error_risk,
    *,
    base_rate: float,
    risk_threshold: float = 0.60,
    max_shrink: float = 0.35,
) -> np.ndarray:
    """Shrink high-risk predictions toward the causal training base rate.

    This never reverses the predicted direction. Risk is not treated as a
    signal to bet against the model.
    """
    p = _clip_probability(probability)
    risk = np.asarray(error_risk, dtype=float)
    if risk.shape != p.shape:
        raise ValueError("error_risk/probability shape mismatch")
    base = float(np.clip(base_rate, 1e-5, 1.0 - 1e-5))
    threshold = float(risk_threshold)
    shrink_cap = float(max_shrink)
    if not 0.0 <= threshold < 1.0:
        raise ValueError("risk_threshold must be in [0,1)")
    if not 0.0 <= shrink_cap <= 1.0:
        raise ValueError("max_shrink must be in [0,1]")

    risk = np.nan_to_num(risk, nan=1.0, posinf=1.0, neginf=1.0)
    active = np.clip((risk - threshold) / max(1.0 - threshold, EPS), 0.0, 1.0)
    amount = shrink_cap * active
    adjusted = (1.0 - amount) * p + amount * base
    return _clip_probability(adjusted)


def risk_bins(error_risk, y_true, probability):
    """Return fixed-bin diagnostics without tuning on the evaluated fold."""
    risk = np.asarray(error_risk, dtype=float)
    y = np.asarray(y_true, dtype=int)
    p = _clip_probability(probability)
    if risk.ndim != 1 or y.ndim != 1 or p.ndim != 1 or not (len(risk) == len(y) == len(p)):
        raise ValueError("risk/y/probability shape mismatch")

    out = []
    for lo, hi in ((0.0, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.01)):
        mask = (risk >= lo) & (risk < hi)
        if not mask.any():
            continue
        pred = p[mask] >= 0.5
        out.append(
            {
                "risk_lo": lo,
                "risk_hi": hi,
                "n": int(mask.sum()),
                "error_rate": float(np.mean(pred != y[mask])),
                "mean_risk": float(np.mean(risk[mask])),
            }
        )
    return out


__all__ = [
    "confidence_risk_features",
    "fit_temporal_confidence_risk",
    "predicted_error_risk",
    "apply_confidence_risk_shrinkage",
    "risk_bins",
]
