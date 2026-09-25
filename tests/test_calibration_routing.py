from __future__ import annotations

import pytest

from src.research.calibration_routing import select_temporal_calibration_method


def _history(rows):
    return {name: list(rows) for name in ("platt", "beta", "isotonic", "temperature")}


def test_prior_history_selects_recently_better_method():
    history = _history([])
    history["platt"] = [
        {"fold": 0, "logloss": 0.60, "ece": 0.04, "brier": 0.22},
        {"fold": 1, "logloss": 0.59, "ece": 0.04, "brier": 0.22},
    ]
    history["temperature"] = [
        {"fold": 0, "logloss": 0.61, "ece": 0.05, "brier": 0.23},
        {"fold": 1, "logloss": 0.56, "ece": 0.03, "brier": 0.21},
    ]
    chosen = select_temporal_calibration_method(
        history, current_fold=2, min_history_folds=2, half_life_folds=2.0
    )
    assert chosen == "temperature"


def test_insufficient_history_fails_back_to_platt():
    history = _history([])
    history["temperature"] = [
        {"fold": 0, "logloss": 0.10, "ece": 0.01, "brier": 0.02},
    ]
    assert select_temporal_calibration_method(
        history, current_fold=1, min_history_folds=2
    ) == "platt"


def test_non_prior_fold_fails_closed():
    history = _history([])
    history["temperature"] = [
        {"fold": 2, "logloss": 0.10, "ece": 0.01, "brier": 0.02},
        {"fold": 1, "logloss": 0.11, "ece": 0.01, "brier": 0.02},
    ]
    with pytest.raises(ValueError):
        select_temporal_calibration_method(history, current_fold=2, min_history_folds=2)
