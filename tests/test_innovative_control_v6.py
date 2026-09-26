import numpy as np
import pandas as pd
import pytest

from src.research.innovative_control_v6 import (
    _ece,
    block_bootstrap_ci,
    disagreement_features,
    evaluate_v6,
    information_shock_score,
)


def bank(fold: int, n: int = 80):
    rng = np.random.default_rng(100 + fold)
    y = rng.integers(0, 2, size=n)
    p0 = np.clip(0.35 + 0.30 * y + rng.normal(0, 0.03, size=n), 0.02, 0.98)
    p1 = np.clip(0.55 - 0.20 * y + rng.normal(0, 0.03, size=n), 0.02, 0.98)
    risk = np.column_stack([
        np.full(n, 0.02),
        np.full(n, 1.2),
        np.zeros(n),
    ])
    return {
        "session_dates": np.arange(fold * n, (fold + 1) * n),
        "y": y,
        "predictions": {"lr": p0, "hgb": p1},
        "risk_context": risk,
        "symbols": np.array([f"S{i % 20}" for i in range(n)]),
        "regimes": np.array(["normal" if fold % 2 == 0 else "trend"] * n),
        "asset_classes": np.array(["equity"] * n),
        "situations": np.array(["normal"] * n),
    }


def test_disagreement_schema_and_values():
    p = np.array([[0.1, 0.9], [0.7, 0.8]], dtype=float)
    out = disagreement_features(p)
    assert np.isclose(out.loc[0, "mean_probability"], 0.5)
    assert np.isclose(out.loc[0, "probability_range"], 0.8)
    assert {"js_divergence", "l1_disagreement", "l2_disagreement", "cosine_disagreement"} <= set(out.columns)


def test_disagreement_rejects_nonfinite():
    with pytest.raises(ValueError):
        disagreement_features(np.array([[0.1, np.nan]]))


def test_ece_zero_for_perfect_calibration_case():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.01, 0.99, 0.99])
    assert _ece(y, p) < 0.02


def test_information_shock_is_bounded():
    rc = np.array([[0.0, 0.0, 0.0], [0.10, 8.0, 0.0]], dtype=float)
    s = information_shock_score(rc)
    assert np.all((s >= 0.0) & (s <= 1.0))
    assert s[1] > s[0]


def test_block_bootstrap_is_deterministic():
    a = block_bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], draws=200)
    b = block_bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], draws=200)
    assert a == b


def test_v6_has_locked_oos_and_no_production_change():
    folds = {i: bank(i) for i in range(6)}
    out = evaluate_v6(folds, locked_folds=2)
    assert out["status"] == "OOS_COMPLETE"
    assert out["production_changed"] is False
    assert out["locked_oos_untouched_for_tuning"] is True
    assert out["promotion"] in {"HOLD", "CANDIDATE"}


def test_v6_contains_required_architecture_matrix():
    folds = {i: bank(i) for i in range(6)}
    out = evaluate_v6(folds, locked_folds=2)
    assert set(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                "K_error_corr", "L_regime_transition", "M_retrieval",
                "N_tta", "O_adaptive_compute", "P_full_v6"]) <= set(out["architectures"])


def test_v6_fails_closed_on_invalid_probability():
    folds = {i: bank(i) for i in range(5)}
    folds[4]["predictions"]["lr"][0] = np.nan
    with pytest.raises(ValueError):
        evaluate_v6(folds)


def test_meta_audit_never_marks_current_labels_as_training():
    folds = {i: bank(i) for i in range(6)}
    out = evaluate_v6(folds, locked_folds=2)
    assert all(not row["current_labels_used_for_meta_training"] for row in out["per_fold_meta_audit"])


def test_retrieval_is_history_only():
    folds = {i: bank(i) for i in range(6)}
    out = evaluate_v6(folds, locked_folds=2)
    assert out["retrieval"]["history_only"] is True


def test_source_reliability_is_not_fabricated():
    folds = {i: bank(i) for i in range(6)}
    out = evaluate_v6(folds, locked_folds=2)
    assert out["production_artifact"] == "UNCHANGED"
