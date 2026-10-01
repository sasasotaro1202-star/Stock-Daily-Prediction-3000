from __future__ import annotations

import pytest

from src.research.experience_candidate_evaluator import evaluate_experience_candidates


def _plan(*candidates):
    return {
        "schema_version": 1,
        "candidates": list(candidates),
        "safety_contract": {
            "research_only": True,
            "production_changed": False,
            "frozen_holdout_allowed": False,
        },
    }


def _candidate(dimension, segment, cid):
    return {
        "candidate_id": cid,
        "action": "test",
        "source_dimension": dimension,
        "source_segment": segment,
        "evidence_n": 100,
        "impact_vs_global_logloss": 0.1,
    }


def test_evaluates_asset_candidate_against_same_scope_parent():
    plan = _plan(_candidate("by_asset_class", "us_stock", "c1"))
    metrics = {
        "selected_model": "hgb",
        "asset_class_selected_models": {"us_stock": "hgb"},
        "asset_class_metrics": {
            "us_stock": {
                "hgb": {
                    "folds": 5,
                    "n_test_min": 100,
                    "logloss": 0.70,
                    "logloss_std": 0.04,
                    "accuracy": 0.55,
                    "brier": 0.24,
                    "ece": 0.08,
                },
                "extra_trees": {
                    "folds": 5,
                    "n_test_min": 100,
                    "logloss": 0.67,
                    "logloss_std": 0.03,
                    "accuracy": 0.58,
                    "brier": 0.22,
                    "ece": 0.07,
                },
            }
        }
    }

    result = evaluate_experience_candidates(plan, metrics)

    row = result["evaluations"][0]
    assert row["status"] == "EVALUATED"
    assert row["parent_model"] == "hgb"
    assert row["best_model"] == "extra_trees"
    assert row["relative_score_improvement"] > 0.03
    assert row["candidate_passes_screen"] is True
    assert row["promotion_allowed"] is False
    assert row["fresh_oos_required"] is True


def test_blocks_candidate_without_direct_oos_scope():
    plan = _plan(_candidate("error_types", "high_confidence_wrong", "c2"))
    result = evaluate_experience_candidates(
        plan,
        {
            "selected_model": "hgb",
            "results": {},
        },
    )

    row = result["evaluations"][0]
    assert row["status"] == "NO_DIRECT_OOS_SCOPE"
    assert row["promotion_allowed"] is False
    assert row["fresh_oos_required"] is True


def test_requires_comparable_parent_on_same_scope():
    plan = _plan(_candidate("by_regime", "high_vol", "c3"))
    result = evaluate_experience_candidates(
        plan,
        {
            "selected_model": "hgb",
            "regime_selected_models": {"high_vol": "logistic"},
            "regime_metrics": {
                "high_vol": {
                    "hgb": {
                        "folds": 5,
                        "n_test_min": 100,
                        "logloss": 0.70,
                        "logloss_std": 0.04,
                    }
                }
            },
        },
    )

    assert result["evaluations"][0]["status"] == "NO_COMPARABLE_PARENT"


def test_rejects_unsafe_candidate_plan():
    plan = _plan(_candidate("by_asset_class", "us_stock", "c4"))
    plan["safety_contract"]["production_changed"] = True
    with pytest.raises(ValueError, match="production mutation"):
        evaluate_experience_candidates(plan, {})


def test_rejects_too_few_folds():
    plan = _plan(_candidate("by_asset_class", "us_stock", "c5"))
    with pytest.raises(ValueError, match=">=3 OOS folds"):
        evaluate_experience_candidates(plan, {}, min_folds=2)
