from __future__ import annotations

import numpy as np

from src.research.nested_ranking import nested_prequential_ranking_oos


def _fixture():
    predictions = {}
    returns = {}
    model_rows = {
        "model_a": [],
        "model_b": [],
    }
    for fold in range(5):
        y = np.asarray([0, 1, 0, 1], dtype=int)
        target_returns = np.asarray([-0.02, 0.03, -0.01, 0.04], dtype=float)
        predictions[fold] = {
            "y": y,
            "session_dates": np.asarray(
                [f"2026-01-0{fold + 1}"] * 4,
                dtype=str,
            ),
            "asset_classes": np.asarray(["jp_stock"] * 4, dtype=str),
            "predictions": {
                "model_a": np.asarray([0.20, 0.80, 0.35, 0.75]),
                "model_b": np.asarray([0.70, 0.30, 0.65, 0.25]),
            },
        }
        returns[fold] = {
            "q50": {
                "y": target_returns,
                "pred": np.asarray([-0.01, 0.025, -0.005, 0.03]),
                "interval": np.asarray(
                    [
                        [-0.04, 0.02],
                        [-0.01, 0.06],
                        [-0.03, 0.02],
                        [0.00, 0.07],
                    ]
                ),
            },
            "mean": {
                "y": target_returns,
                "pred": np.asarray([-0.015, 0.02, -0.002, 0.025]),
                "interval": np.asarray(
                    [
                        [-0.04, 0.02],
                        [-0.01, 0.06],
                        [-0.03, 0.02],
                        [0.00, 0.07],
                    ]
                ),
            },
        }
        model_rows["model_a"].append({"fold": fold, "logloss": 0.60 - 0.01 * fold})
        model_rows["model_b"].append({"fold": fold, "logloss": 0.80 - 0.005 * fold})
    return predictions, returns, model_rows


def test_nested_ranking_is_prequential_and_research_only():
    predictions, returns, model_rows = _fixture()
    result = nested_prequential_ranking_oos(
        predictions,
        returns,
        model_rows,
        min_history_folds=2,
    )

    assert result["status"] == "EVALUATED"
    assert result["research_only"] is True
    assert result["production_changed"] is False
    assert result["promotion_allowed"] is False
    assert result["same_oos_global_model_or_window_reuse"] is False
    assert result["ranking_weight_selection_prequential"] is True
    assert result["model_selection_prequential"] is True
    assert result["return_estimator_selection_prequential"] is True
    assert len(result["outer_metrics"]) >= 3


def test_current_fold_outcome_does_not_change_current_fold_selection():
    predictions, returns, model_rows = _fixture()
    baseline = nested_prequential_ranking_oos(
        predictions,
        returns,
        model_rows,
        min_history_folds=2,
    )

    changed_predictions = {fold: dict(bank) for fold, bank in predictions.items()}
    changed_returns = {
        fold: {
            name: dict(bank)
            for name, bank in per_fold.items()
        }
        for fold, per_fold in returns.items()
    }
    # Alter only the final fold's realized outcome and model log-loss. There
    # is no later fold, so this information must not affect final-fold choices.
    changed_predictions[4] = dict(changed_predictions[4])
    changed_predictions[4]["y"] = np.asarray([1, 0, 1, 0], dtype=int)
    changed_returns[4] = {
        name: {
            **bank,
            "y": np.asarray([0.20, -0.10, 0.15, -0.08], dtype=float),
        }
        for name, bank in changed_returns[4].items()
    }
    changed_model_rows = {
        name: list(rows)
        for name, rows in model_rows.items()
    }
    changed_model_rows["model_a"][-1] = {"fold": 4, "logloss": 9.0}
    changed_model_rows["model_b"][-1] = {"fold": 4, "logloss": 0.01}

    changed = nested_prequential_ranking_oos(
        changed_predictions,
        changed_returns,
        changed_model_rows,
        min_history_folds=2,
    )

    assert baseline["selected_model_by_fold"][4] == changed["selected_model_by_fold"][4]
    assert baseline["selected_return_estimator_by_fold"][4] == changed[
        "selected_return_estimator_by_fold"
    ][4]
    assert baseline["selected_ranking_parameters_by_fold"]["4"] == changed[
        "selected_ranking_parameters_by_fold"
    ]["4"]


def test_nested_ranking_requires_bootstrap_evidence_for_positive_candidate():
    predictions, returns, model_rows = _fixture()
    result = nested_prequential_ranking_oos(
        predictions,
        returns,
        model_rows,
        min_history_folds=2,
    )
    assert "bootstrap_probability_improvement" in result
    assert "bootstrap_p05_improvement" in result
    assert result["bootstrap_method"] == "moving_block"
    assert result["research_positive"] is False or (
        result["bootstrap_probability_improvement"] >= 0.90
        and result["bootstrap_p05_improvement"] > 0.0
    )


def test_production_identity_alignment_is_fail_closed():
    predictions, returns, model_rows = _fixture()
    result = nested_prequential_ranking_oos(
        predictions,
        returns,
        model_rows,
        min_history_folds=2,
        production_identity={
            "selected_model": "not_the_final_model",
            "classifier_training_window_sessions": 504,
            "return_estimator": "not_the_final_estimator",
            "rank_probability_weight": 0.99,
            "rank_uncertainty_penalty": 0.99,
        },
        prediction_generation_training_window_sessions=0,
    )

    alignment = result["production_identity_alignment"]
    assert alignment["provided"] is True
    assert alignment["aligned"] is False
    assert alignment["checks"]["classifier_training_window_matches_prediction_generation"] is False

