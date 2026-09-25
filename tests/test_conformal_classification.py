from __future__ import annotations
import numpy as np
import pytest
from src.research.conformal_classification import (
    conformal_prediction_sets, conformal_prediction_set_metrics
)

def test_prediction_sets_flag_ambiguous_and_confident_cases():
    y_cal = np.asarray([0, 1] * 20, dtype=int)
    p_cal = np.asarray([0.2, 0.8] * 20, dtype=float)
    p_test = np.asarray([0.5, 0.95, 0.05], dtype=float)
    result = conformal_prediction_sets(y_cal, p_cal, p_test, alpha=0.10)
    assert result["set_size"][0] == 0
    assert result["set_size"][1] == 1
    assert result["set_size"][2] == 1

def test_metrics_are_bounded():
    y_cal = np.asarray([0, 1] * 25, dtype=int)
    p_cal = np.asarray([0.2, 0.8] * 25, dtype=float)
    y_test = np.asarray([0, 1, 0, 1], dtype=int)
    p_test = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=float)
    metrics = conformal_prediction_set_metrics(
        y_cal, p_cal, p_test, y_test, alpha=0.10
    )
    assert 0.0 <= metrics["set_coverage"] <= 1.0
    assert 0.0 <= metrics["singleton_rate"] <= 1.0
    assert 0.0 <= metrics["empty_rate"] <= 1.0
    assert 0.0 <= metrics["mean_set_size"] <= 2.0
    assert 0.0 <= metrics["singleton_accuracy"] <= 1.0

def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        conformal_prediction_sets([0, 1], [0.2, 0.8], [0.5], alpha=0.0)
    with pytest.raises(ValueError):
        conformal_prediction_sets([0, 1] * 10, [0.2] * 19, [0.5], alpha=0.1)
    with pytest.raises(ValueError):
        conformal_prediction_set_metrics(
            [0, 1] * 10, [0.2, 0.8] * 10, [0.5, 0.5], [0, 2], alpha=0.1
        )
