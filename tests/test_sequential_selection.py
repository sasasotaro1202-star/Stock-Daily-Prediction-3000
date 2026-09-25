from __future__ import annotations

import pytest

from src.research.nested_policy import nested_sequential_policy_oos
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



def test_nested_policy_reserves_later_outer_block_and_adapts_prequentially():
    rows = {
        "slow": _rows([0.50, 0.51, 0.52, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70]),
        "fast": _rows([0.60, 0.59, 0.58, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35]),
    }
    result = nested_sequential_policy_oos(
        rows,
        outer_start_fold=3,
        min_history_folds=3,
        min_outer_folds=5,
        baseline_model="slow",
    )
    assert result["status"] == "EVALUATED"
    assert result["inner_folds"] == 3
    assert result["outer_folds"] == 6
    assert result["selection_hyperparameters_frozen_before_outer"] is True
    assert result["selected_model_by_outer_fold"][0]["fold"] == 3
    assert result["selected_model_by_outer_fold"][0]["selected_model"] == "slow"
    assert result["selected_model_by_outer_fold"][0]["history_max_fold"] == 2
    assert result["selected_model_by_outer_fold"][1]["fold"] == 4
    assert result["selected_model_by_outer_fold"][1]["selected_model"] == "fast"
    assert result["selected_model_by_outer_fold"][1]["history_max_fold"] == 3


def test_nested_policy_requires_enough_outer_folds():
    rows = {
        "slow": _rows([0.50, 0.51, 0.52, 0.70, 0.70, 0.70]),
        "fast": _rows([0.60, 0.59, 0.58, 0.40, 0.40, 0.40]),
    }
    result = nested_sequential_policy_oos(
        rows,
        outer_start_fold=3,
        min_history_folds=3,
        min_outer_folds=5,
    )
    assert result["status"] == "INSUFFICIENT_OOS"


def test_nested_policy_auto_baseline_is_selected_from_inner_block_only():
    rows = {
        "slow": _rows([0.50, 0.51, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70]),
        "fast": _rows([0.60, 0.59, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35]),
    }
    result = nested_sequential_policy_oos(
        rows,
        outer_start_fold=3,
        min_history_folds=3,
        min_outer_folds=5,
        baseline_model="auto_inner_static",
    )
    assert result["status"] == "EVALUATED"
    assert result["baseline_model"] == "slow"
    assert result["baseline_model_policy"] == "auto_inner_static"
