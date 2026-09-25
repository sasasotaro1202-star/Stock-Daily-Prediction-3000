from __future__ import annotations

import pytest

from src.research.sequential_selection import (
    chronological_policy_oos,
    select_prior_oos_model,
)


def _rows(values):
    return [
        {"fold": float(i), "logloss": float(value)}
        for i, value in enumerate(values)
    ]


def test_selector_uses_only_prior_folds():
    rows = {
        "slow": _rows([0.50, 0.51]),
        "fast": _rows([0.60, 0.61]),
    }
    selected, diagnostics = select_prior_oos_model(
        rows,
        current_fold=2,
        min_history_folds=2,
    )
    assert selected == "slow"
    assert diagnostics["slow"]["history_folds"] == 2


def test_selector_rejects_current_fold_when_directly_supplied():
    rows = {
        "model": _rows([0.60, 0.59, 0.50]),
    }
    with pytest.raises(ValueError):
        select_prior_oos_model(rows, current_fold=2, min_history_folds=2)


def test_selector_rejects_current_or_future_history():
    rows = {
        "model": _rows([0.60, 0.59, 0.50]),
    }
    with pytest.raises(ValueError):
        select_prior_oos_model(
            rows,
            current_fold=2,
            min_history_folds=2,
        )


def test_chronological_policy_scores_after_each_selection():
    rows = {
        "slow": _rows([0.50, 0.51, 0.70, 0.69]),
        "fast": _rows([0.60, 0.61, 0.55, 0.54]),
    }
    result = chronological_policy_oos(
        rows,
        min_history_folds=2,
        baseline_model="slow",
    )
    assert result["status"] == "EVALUATED"
    assert result["folds"] == 2
    assert result["selected_model_by_fold"][0]["fold"] == 2
    assert result["selected_model_by_fold"][0]["selected_model"] == "slow"
    assert result["selected_model_by_fold"][1]["fold"] == 3
    assert result["selected_model_by_fold"][1]["selected_model"] == "fast"
    assert result["baseline_model"] == "slow"
