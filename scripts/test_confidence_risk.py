from __future__ import annotations

import numpy as np

from src.research.confidence_risk import (
    apply_confidence_risk_shrinkage,
    confidence_risk_features,
    fit_temporal_confidence_risk,
    predicted_error_risk,
    risk_bins,
)


def main() -> None:
    p = np.asarray([0.9, 0.55, 0.1, 0.45])
    experts = np.asarray([
        [0.9, 0.8, 0.95],
        [0.52, 0.55, 0.58],
        [0.1, 0.2, 0.05],
        [0.40, 0.50, 0.48],
    ])
    ctx = np.asarray([
        [0.03, 3.5, 0.04, 0.40, 0.06, 0.25, 32.0, 2.0, -0.20, 0.95, 0.10],
        [0.01, 1.0, 0.00, 0.02, 0.01, 0.00, 15.0, 0.2, -0.03, 0.60, 0.40],
        [0.04, 5.0, 0.06, 0.45, 0.08, 0.50, 40.0, -3.0, -0.40, 0.05, 0.90],
        [np.nan, 1.2, np.nan, np.nan, 0.02, np.nan, np.nan, 0.0, -0.02, 0.50, 0.50],
    ])
    features = confidence_risk_features(p, experts, ctx)
    assert features.shape == (4, 16)
    assert np.all(np.isfinite(features))
    assert features[0, 2] > features[1, 2]

    rng = np.random.default_rng(20260925)
    n = 320
    base_p = np.clip(rng.uniform(0.15, 0.85, n), 1e-4, 1 - 1e-4)
    expert_matrix = np.column_stack([
        np.clip(base_p + rng.normal(0, 0.04, n), 1e-4, 1 - 1e-4),
        np.clip(base_p + rng.normal(0, 0.10, n), 1e-4, 1 - 1e-4),
        np.clip(base_p + rng.normal(0, 0.06, n), 1e-4, 1 - 1e-4),
    ])
    meta_ctx = rng.normal(0, 1, size=(n, 11))
    meta = confidence_risk_features(base_p, expert_matrix, meta_ctx)
    # Make correctness depend on observable meta-risk structure.
    score = 1.2 * meta[:, 1] + 0.8 * meta[:, 2] + rng.normal(0, 0.6, n)
    correct = (score < np.median(score)).astype(int)
    model = fit_temporal_confidence_risk(meta, correct, min_rows=240)
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
    assert np.all((adjusted > 0) & (adjusted < 1))

    bins = risk_bins(
        np.asarray([0.2, 0.5, 0.7, 0.9]),
        np.asarray([1, 1, 0, 1]),
        np.asarray([0.7, 0.6, 0.4, 0.3]),
    )
    assert len(bins) == 4
    print("CONFIDENCE_RISK_RESEARCH_SMOKE=PASS")


if __name__ == "__main__":
    main()
