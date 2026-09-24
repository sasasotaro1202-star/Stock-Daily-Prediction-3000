from __future__ import annotations

import numpy as np

from src.research.confidence_risk import (
    apply_confidence_risk_shrinkage,
    confidence_risk_features,
    fit_temporal_confidence_risk,
    predicted_error_risk,
    risk_bins,
)


def test_feature_shape_and_finiteness():
    p = np.asarray([0.90, 0.55, 0.10, 0.45])
    experts = np.asarray([
        [0.90, 0.80, 0.95],
        [0.52, 0.55, 0.58],
        [0.10, 0.20, 0.05],
        [0.40, 0.50, 0.48],
    ])
    ctx = np.asarray([
        [0.03, 3.5, 0.04, 0.40, 0.06, 0.25, 32.0, 2.0, -0.20, 0.95, 0.10],
        [0.01, 1.0, 0.00, 0.02, 0.01, 0.00, 15.0, 0.2, -0.03, 0.60, 0.40],
        [0.04, 5.0, 0.06, 0.45, 0.08, 0.50, 40.0, -3.0, -0.40, 0.05, 0.90],
        [np.nan, 1.2, np.nan, np.nan, 0.02, np.nan, np.nan, 0.0, -0.02, 0.50, 0.50],
    ])
    out = confidence_risk_features(p, experts, ctx)
    assert out.shape == (4, 27)
    assert np.all(np.isfinite(out))
    assert out[0, 2] > out[1, 2]
    # Each context value is paired with an explicit missingness indicator.
    assert out[3, 6 + 2 * 0 + 1] == 0.0  # volatility_20 is observed on row 4? no: NaN is flagged below
    assert out[3, 7] == 0.0


def test_temporal_risk_model_and_shrinkage():
    rng = np.random.default_rng(20260925)
    n = 320
    p = np.clip(rng.uniform(0.15, 0.85, n), 1e-4, 1 - 1e-4)
    experts = np.column_stack([
        np.clip(p + rng.normal(0, 0.04, n), 1e-4, 1 - 1e-4),
        np.clip(p + rng.normal(0, 0.10, n), 1e-4, 1 - 1e-4),
        np.clip(p + rng.normal(0, 0.06, n), 1e-4, 1 - 1e-4),
    ])
    ctx = rng.normal(0, 1, size=(n, 11))
    meta = confidence_risk_features(p, experts, ctx)
    correctness = (meta[:, 1] + 0.8 * meta[:, 2] + rng.normal(0, 0.6, n)
                   < np.median(meta[:, 1] + 0.8 * meta[:, 2])).astype(int)

    model = fit_temporal_confidence_risk(meta, correctness, min_rows=240)
    assert model is not None
    risk = predicted_error_risk(model, meta[-20:])
    assert risk.shape == (20,)
    assert np.all((risk >= 0) & (risk <= 1))

    adjusted = apply_confidence_risk_shrinkage(
        np.asarray([0.90, 0.60, 0.10]),
        np.asarray([0.90, 0.20, 0.95]),
        base_rate=0.52,
    )
    assert adjusted[0] < 0.90
    assert adjusted[1] == 0.60
    assert adjusted[2] > 0.10


def test_risk_bins_are_fixed_and_safe():
    bins = risk_bins(
        np.asarray([0.2, 0.5, 0.7, 0.9]),
        np.asarray([1, 1, 0, 1]),
        np.asarray([0.7, 0.6, 0.4, 0.3]),
    )
    assert len(bins) == 4
    assert all(0.0 <= row["error_rate"] <= 1.0 for row in bins)
