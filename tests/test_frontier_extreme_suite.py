from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.frontier_extreme_suite import run_extreme_pattern_suite


def _toy_bank():
    rng = np.random.default_rng(271828)
    models = ["a", "b", "c", "d", "e"]
    bank = []
    for fold in range(6):
        n = 20
        y = np.asarray([(i + fold) % 2 for i in range(n)], dtype=int)
        latent = 0.35 + 0.30 * y + rng.normal(0.0, 0.04, n)
        predictions = {}
        for j, model in enumerate(models):
            predictions[model] = np.clip(
                latent
                + (j - 2) * 0.012
                + rng.normal(0.0, 0.02, n),
                0.02,
                0.98,
            )
        frame = pd.DataFrame({
            "session_date": [f"2026-02-{fold + 1:02d}"] * n,
            "symbol": [f"T{i:03d}" for i in range(n)],
            "regime": ["normal" if fold < 3 else "stress"] * n,
        })
        risk = np.column_stack([
            np.linspace(0.0, 1.0, n),
            np.where(np.arange(n) % 4 == 0, 4.0, 1.0),
            np.where(np.arange(n) % 6 == 0, 0.05, 0.01),
            np.where(np.arange(n) % 9 == 0, np.nan, 0.9),
        ])
        bank.append({
            "fold": fold,
            "y": y,
            "predictions": predictions,
            "frame": frame,
            "risk_matrix": risk,
        })
    return bank


def test_extreme_suite_has_broad_mechanism_coverage():
    result = run_extreme_pattern_suite(
        _toy_bank(),
        locked_folds=2,
        min_folds=5,
        minimum_patterns=100,
    )
    assert result["status"] == "EXECUTED_EXTREME_PATTERN_MATRIX"
    assert result["research_contract_id"] == "extreme-frontier-pattern-ecology-v1"
    assert result["execution_failures"] == []
    assert result["pattern_count"] >= 100
    assert result["research_only"] is True
    assert result["production_changed"] is False
    assert result["promotion_allowed"] is False
    names = {row["name"] for row in result["patterns"]}
    for name in (
        "probit_mean",
        "minimax_t0.05",
        "cal_platt_equal",
        "cal_temperature_quality",
        "rank_hybrid_50",
        "vote_100",
        "shrink_base_rate_20",
        "case_high_uncertainty_20",
        "agreement_strength_logit_50",
        "anchor_mix_logit_base_rate_50",
    ):
        assert name in names


def test_extreme_suite_uses_same_locked_cases():
    result = run_extreme_pattern_suite(
        _toy_bank(),
        locked_folds=2,
        min_folds=5,
        minimum_patterns=100,
    )
    locked_cases = {row["locked_cases"] for row in result["patterns"]}
    assert locked_cases == {40}
    assert result["baseline"]["locked_metrics"]["n"] == 40
    assert result["session_cluster"]["n_clusters"] == 2


def test_extreme_suite_never_opens_production_or_holdout():
    result = run_extreme_pattern_suite(
        _toy_bank(),
        locked_folds=2,
        min_folds=5,
        minimum_patterns=100,
    )
    contracts = result["contracts"]
    assert contracts["current_fold_outcomes_used_for_pattern_tuning"] is False
    assert contracts["learned_patterns_fit_only_on_strictly_prior_folds"] is True
    assert contracts["calibration_fit_only_on_strictly_prior_rows"] is True
    assert contracts["selective_thresholds_fit_only_on_prior_state"] is True
    assert contracts["frozen_holdout_used"] is False
    assert contracts["production_changed"] is False
    assert contracts["promotion_allowed"] is False
    assert contracts["locked_outcomes_used_for_selection"] is False
    assert contracts["best_research_pattern_selected_from_development_only"] is True
    assert contracts["winner_selection_is_prequential_development_only"] is True
    assert result["selection"]["source"] == "prequential_development_only"
    assert "prequential_selection_stability" in result["selection"]
    assert result["selection"]["prequential_selection_stability"]["decision_count"] >= 0
    final_sel = result["selection"]["final_prequential_selection"]
    decisions = result["selection"]["prequential_decisions"]
    assert final_sel["source"] == "last_prequential_development_decision"
    assert final_sel["decision_outcome_used_for_selection"] is False
    assert final_sel["decision_fold"] == decisions[-1]["fold"]
    assert result["selection"]["selected_name"] == decisions[-1]["selected_name"]


def test_extreme_suite_blocks_short_oos_history():
    result = run_extreme_pattern_suite(
        _toy_bank()[:4],
        locked_folds=2,
        min_folds=5,
        minimum_patterns=100,
    )
    assert result["status"] == "BLOCKED"
    assert result["promotion_allowed"] is False


def test_extreme_suite_preserves_generation_failures(monkeypatch):
    import src.research.frontier_extreme_suite as module

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic-generation-failure")

    monkeypatch.setattr(module, "_candidate_family", fail)
    result = run_extreme_pattern_suite(
        _toy_bank(),
        locked_folds=2,
        min_folds=5,
        minimum_patterns=100,
    )
    assert result["execution_failures"]
    failure = result["execution_failures"][0]
    assert failure["error_type"] == "RuntimeError"
    assert "synthetic-generation-failure" in failure["error"]
    assert result["status"] == "BLOCKED_INSUFFICIENT_PATTERN_BREADTH"
    assert result["pattern_count"] < 100
    assert result["promotion_allowed"] is False
