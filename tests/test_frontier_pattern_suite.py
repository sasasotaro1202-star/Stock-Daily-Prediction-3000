from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.frontier_pattern_suite import run_frontier_pattern_suite


def _toy_bank():
    rng = np.random.default_rng(13013)
    bank = []
    models = ["model_a", "model_b", "model_c", "model_d"]
    for fold in range(6):
        n = 18
        y = np.asarray([(i + fold) % 2 for i in range(n)], dtype=int)
        base = 0.35 + 0.30 * y + rng.normal(0.0, 0.05, n)
        predictions = {}
        for j, name in enumerate(models):
            drift = (j - 1.5) * 0.015 + (0.01 if fold >= 4 and j % 2 == 0 else 0.0)
            predictions[name] = np.clip(base + drift + rng.normal(0.0, 0.025, n), 0.02, 0.98)
        frame = pd.DataFrame({
            "session_date": [f"2026-01-{fold+1:02d}"] * n,
            "symbol": [f"S{i:03d}" for i in range(n)],
            "regime": ["normal" if fold < 3 else "stress"] * n,
        })
        risk = np.column_stack([
            np.linspace(0.0, 1.0, n),
            np.ones(n) * 0.2,
            np.where(np.arange(n) % 5 == 0, 0.06, 0.01),
            np.where(np.arange(n) % 7 == 0, np.nan, 0.8),
        ])
        bank.append({
            "fold": fold,
            "y": y,
            "predictions": predictions,
            "frame": frame,
            "risk_matrix": risk,
        })
    return bank


def test_frontier_suite_executes_broad_matrix_without_production_mutation():
    result = run_frontier_pattern_suite(_toy_bank(), locked_folds=2, min_folds=5)
    assert result["status"] == "EXECUTED_RESEARCH_PATTERN_MATRIX"
    assert result["research_only"] is True
    assert result["production_changed"] is False
    assert result["promotion_allowed"] is False
    assert result["frozen_holdout_used"] is False
    assert result["pattern_count"] >= 50
    names = {row["name"] for row in result["patterns"]}
    assert "winsorized_logit_mean" in names
    assert "regime_recent_mix_50" in names
    assert "difficulty_shrink_40" in names
    assert "risk_ood_shrink" in names
    assert "logit_median" in names
    assert "winsorized_probability_mean" in names
    assert "prior_outcome_base_rate_shrink_25" in names
    assert "selective_difficulty_q60" in names
    assert "selective_difficulty_q95" in names


def test_frontier_suite_keeps_same_locked_case_count_for_all_patterns():
    result = run_frontier_pattern_suite(_toy_bank(), locked_folds=2, min_folds=5)
    locked_cases = {row["locked_cases"] for row in result["patterns"]}
    assert locked_cases == {36}
    assert result["baseline"]["locked_metrics"]["n"] == 36
    assert result["diagnostics"]["oracle_best_per_case_logloss"] is not None


def test_frontier_suite_learned_components_are_prior_only_by_contract():
    result = run_frontier_pattern_suite(_toy_bank(), locked_folds=2, min_folds=5)
    contracts = result["contracts"]
    assert contracts["current_fold_outcomes_used_for_pattern_tuning"] is False
    assert contracts["learned_patterns_fit_only_on_strictly_prior_folds"] is True
    assert contracts["calibration_fit_only_on_strictly_prior_rows"] is True
    assert contracts["selective_thresholds_fit_only_on_prior_state"] is True
    assert contracts["locked_outcomes_used_for_tuning"] is False
    assert contracts["locked_outcomes_used_for_selection"] is False
    assert contracts["best_research_pattern_selected_from_development_only"] is True
    assert result["selection"]["source"] == "development_only"


def test_frontier_suite_blocks_insufficient_chronological_folds():
    result = run_frontier_pattern_suite(_toy_bank()[:4], locked_folds=2, min_folds=5)
    assert result["status"] == "BLOCKED"
    assert result["production_changed"] is False
    assert result["promotion_allowed"] is False
