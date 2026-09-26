import numpy as np
import pandas as pd
import pytest

from src.research.innovative_control_v2 import (
    build_disagreement_features,
    block_bootstrap_ci,
    compute_distribution_drift,
    evaluate_innovative_v2,
    selective_metrics,
)


def _bank(y, p):
    p = np.asarray(p, dtype=float)
    n, m = p.shape
    risk = np.zeros((n, 11), dtype=float)
    return {
        "session_dates": np.arange(n),
        "y": np.asarray(y, dtype=int),
        "predictions": {f"m{i}": p[:, i] for i in range(m)},
        "risk_context": risk,
        "asset_classes": np.array(["equity"] * n),
        "situations": np.array(["normal"] * n),
    }


def test_disagreement_features_are_target_free():
    p = np.array([[0.1, 0.2, 0.9], [0.8, 0.7, 0.6]], dtype=float)
    out = build_disagreement_features(p)
    assert list(out.columns) == [
        "mean_probability",
        "std_probability",
        "min_probability",
        "max_probability",
        "probability_range",
        "prediction_entropy",
        "top_class_agreement_rate",
        "majority_margin",
        "rank_disagreement",
        "pairwise_disagreement",
    ]
    assert np.isclose(out.loc[0, "mean_probability"], 0.4)
    assert np.isclose(out.loc[0, "probability_range"], 0.8)


def test_disagreement_rejects_nonfinite_inputs():
    with pytest.raises(ValueError):
        build_disagreement_features(np.array([[0.1, np.nan]], dtype=float))


def test_distribution_drift_is_zero_for_identical_reference():
    x = np.arange(40, dtype=float).reshape(20, 2)
    assert np.isclose(compute_distribution_drift(x, x), 0.0)


def test_selective_prediction_reports_requested_coverages():
    y = np.array([0, 1, 0, 1] * 10)
    p = np.array([0.51, 0.49, 0.55, 0.45] * 10)
    pred = np.full(len(y), 0.8)
    out = selective_metrics(y, p, pred)
    assert all(key in out for key in ["100%", "95%", "90%", "80%", "70%"])
    assert out["100%"]["coverage"] == 1.0


def test_block_bootstrap_ci_is_finite_and_ordered():
    lo, hi = block_bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], block_length=2, n_boot=100)
    assert np.isfinite(lo)
    assert np.isfinite(hi)
    assert lo <= hi


def test_meta_layer_requires_chronological_history():
    banks = {}
    rng = np.random.default_rng(42)
    for fold in range(6):
        y = rng.integers(0, 2, size=120)
        p = np.column_stack(
            [
                np.clip(0.35 + 0.3 * y + rng.normal(0, 0.05, size=120), 0.02, 0.98),
                np.clip(0.55 - 0.25 * y + rng.normal(0, 0.05, size=120), 0.02, 0.98),
            ]
        )
        banks[fold] = _bank(y, p)
    out = evaluate_innovative_v2(banks)
    assert out["status"] == "OOS_COMPLETE"
    assert out["pit_audit"] == "PASS"
    assert out["meta_leakage"] == "PASS"
    assert out["promotion"] == "HOLD"


def test_meta_layer_fails_closed_on_nonfinite_oos():
    banks = {}
    for fold in range(6):
        y = np.array([0, 1] * 60)
        p = np.column_stack([np.full(120, 0.4), np.full(120, 0.6)])
        if fold == 5:
            p[0, 0] = np.nan
        banks[fold] = _bank(y, p)
    with pytest.raises(ValueError):
        evaluate_innovative_v2(banks)


def test_no_accuracy_only_promotion():
    banks = {}
    rng = np.random.default_rng(7)
    for fold in range(6):
        y = rng.integers(0, 2, size=120)
        p = rng.uniform(0.05, 0.95, size=(120, 2))
        banks[fold] = _bank(y, p)
    out = evaluate_innovative_v2(banks)
    assert out["promotion"] == "HOLD"
