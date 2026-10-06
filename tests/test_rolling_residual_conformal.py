from __future__ import annotations

import numpy as np
import pytest

from src.research.rolling_residual_conformal import (
    interval_diagnostics,
    interval_score,
    rolling_residual_conformal_interval,
)


def test_rolling_conformal_uses_recent_prior_residuals_only():
    y_history = np.asarray([0.01] * 200 + [0.20] * 252, dtype=float)
    pred_history = np.zeros_like(y_history)
    current = np.asarray([0.0, 0.10], dtype=float)

    interval, diag = rolling_residual_conformal_interval(
        y_history,
        pred_history,
        current,
        alpha=0.10,
        min_history=200,
        window=252,
    )

    assert diag["status"] == "EXECUTED_PRIOR_OOS"
    assert diag["window_rows"] == 252.0
    assert diag["radius"] == pytest.approx(0.20)
    assert np.allclose(interval[:, 0], current - 0.20)
    assert np.allclose(interval[:, 1], current + 0.20)


def test_rolling_conformal_defers_without_enough_history():
    interval, diag = rolling_residual_conformal_interval(
        [0.01] * 199,
        [0.0] * 199,
        [0.0, 0.1],
        min_history=200,
    )

    assert diag["status"] == "INSUFFICIENT_HISTORY"
    assert interval.shape == (2, 2)
    assert np.isnan(interval).all()


def test_interval_score_penalizes_out_of_interval_observations():
    interval = np.asarray([[-0.02, 0.02], [-0.02, 0.02]], dtype=float)
    y = np.asarray([0.0, 0.10], dtype=float)

    score = interval_score(y, interval, alpha=0.10)
    assert score > np.mean(interval[:, 1] - interval[:, 0])


def test_interval_diagnostics_reports_coverage_and_severe_miss_rate():
    interval = np.asarray([[-0.05, 0.05], [-0.05, 0.05]], dtype=float)
    y = np.asarray([0.01, 0.20], dtype=float)

    result = interval_diagnostics(y, interval, alpha=0.10)

    assert result["coverage"] == pytest.approx(0.5)
    assert result["mean_width"] == pytest.approx(0.10)
    assert result["miss_rate"] == pytest.approx(0.5)
    assert result["severe_miss_rate_ge_5pct"] == pytest.approx(0.5)
