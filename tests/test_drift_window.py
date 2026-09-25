from __future__ import annotations

import pytest

from src.research.drift_window import (
    robust_distribution_shift_score,
    select_drift_aware_window,
)


def test_shift_score_detects_distribution_change():
    ref = [[0.0], [0.1], [0.2], [0.3]] * 10
    qry = [[1.0], [1.1], [1.2], [1.3]] * 10
    assert robust_distribution_shift_score(ref, qry) > 1.0


def test_window_router_uses_only_prior_folds():
    history = {
        252: [
            {"fold": 0, "logloss": 0.60, "drift": 0.20},
            {"fold": 1, "logloss": 0.59, "drift": 0.25},
        ],
        756: [
            {"fold": 0, "logloss": 0.58, "drift": 1.10},
            {"fold": 1, "logloss": 0.57, "drift": 1.00},
        ],
    }
    selected, diagnostics = select_drift_aware_window(
        history,
        current_fold=2,
        current_drift_by_window={252: 0.22, 756: 1.02},
        min_history_folds=2,
    )
    assert selected == 252
    assert diagnostics[252]["history_folds"] == 2


def test_window_router_rejects_current_or_future_history():
    history = {
        252: [
            {"fold": 2, "logloss": 0.50, "drift": 0.20},
            {"fold": 1, "logloss": 0.60, "drift": 0.25},
        ]
    }
    with pytest.raises(ValueError):
        select_drift_aware_window(
            history,
            current_fold=2,
            current_drift_by_window={252: 0.2},
        )
