from __future__ import annotations

from src.research.expert_loss_routing_oos import analyze_expert_loss_routing


MODELS = ["expert_good", "expert_bad"]


def _case_row(fold: int, row: int, *, locked: bool) -> dict:
    return {
        "fold": fold,
        "row": row,
        "is_locked": locked,
        "prediction_time": "2026-01-02T00:00:00+00:00",
        "available_at": "2026-01-01T23:00:00+00:00",
        "pit_status": "PASS",
        "case_predictability": 0.15 + 0.70 * ((row % 10) / 9.0),
        "case_ood": 0.05 + 0.80 * ((row % 8) / 7.0),
        "case_failure_risk": 0.10 + 0.70 * ((row % 6) / 5.0),
        "case_disagreement": 0.05 + 0.40 * ((row % 5) / 4.0),
    }


def _make_data(locked_outcomes: list[int], *, locked_folds: int = 1):
    ledger = []
    folds = []
    for fold in range(4):
        n = 80
        is_locked = fold >= 4 - int(locked_folds)
        if is_locked:
            y = locked_outcomes
        else:
            y = [i % 2 for i in range(n)]
        # Keep expert predictions independent of the locked outcomes so tests
        # isolate PIT/freeze behavior rather than changing the prediction inputs.
        p_good = [0.9 if (i % 10) < 8 else 0.1 for i in range(n)]
        p_bad = [0.1 if (i % 10) < 8 else 0.9 for i in range(n)]
        for i, _outcome in enumerate(y):
            ledger.append(_case_row(fold, i, locked=is_locked))
        folds.append(
            {
                "y": y,
                "predictions": {
                    "expert_good": p_good,
                    "expert_bad": p_bad,
                },
            }
        )
    return ledger, folds


def test_expert_loss_routing_is_research_only_and_causal():
    ledger, folds = _make_data([i % 2 for i in range(80)])
    result = analyze_expert_loss_routing(
        ledger,
        folds,
        models=MODELS,
        locked_folds=1,
        min_training_rows=60,
    )
    assert result["status"] == "EVALUATED"
    assert result["research_only"] is True
    assert result["production_changed"] is False
    assert result["promotion_allowed"] is False
    assert result["frozen_holdout_used"] is False
    assert result["contracts"]["current_fold_outcomes_used_for_routing"] is False
    assert result["contracts"]["current_fold_outcomes_used_for_expert_loss_fit"] is False
    assert result["contracts"]["all_expert_loss_models_fit_only_on_prior_folds"] is True
    assert result["locked_metrics"]["expert_loss_routing"]["n"] > 0


def test_locked_outcomes_do_not_change_locked_routing_weights():
    ledger_a, folds_a = _make_data([i % 2 for i in range(80)])
    ledger_b, folds_b = _make_data([1 - (i % 2) for i in range(80)])
    result_a = analyze_expert_loss_routing(
        ledger_a,
        folds_a,
        models=MODELS,
        locked_folds=1,
        min_training_rows=60,
    )
    result_b = analyze_expert_loss_routing(
        ledger_b,
        folds_b,
        models=MODELS,
        locked_folds=1,
        min_training_rows=60,
    )
    assert result_a["status"] == "EVALUATED"
    assert result_b["status"] == "EVALUATED"
    a = result_a["fold_results"][-1]
    b = result_b["fold_results"][-1]
    assert a["predicted_loss_mean"] == b["predicted_loss_mean"]
    assert a["weight_means"] == b["weight_means"]


def test_missing_case_lineage_fails_closed():
    ledger, folds = _make_data([i % 2 for i in range(80)])
    ledger[0]["available_at"] = "2026-01-03T00:00:00+00:00"
    result = analyze_expert_loss_routing(
        ledger,
        folds,
        models=MODELS,
        locked_folds=1,
        min_training_rows=60,
    )
    assert result["status"] == "BLOCKED_INVALID_CASE_LINEAGE"


def test_missing_expert_prediction_fails_closed():
    ledger, folds = _make_data([i % 2 for i in range(80)])
    del folds[2]["predictions"]["expert_bad"]
    result = analyze_expert_loss_routing(
        ledger,
        folds,
        models=MODELS,
        locked_folds=1,
        min_training_rows=60,
    )
    assert result["status"] == "BLOCKED_MISSING_EXPERT_PREDICTION"


def test_expert_loss_routing_reports_deterministic_paired_bootstrap():
    ledger, folds = _make_data([i % 2 for i in range(80)])
    first = analyze_expert_loss_routing(
        ledger,
        folds,
        models=MODELS,
        locked_folds=1,
        min_training_rows=60,
    )
    second = analyze_expert_loss_routing(
        ledger,
        folds,
        models=MODELS,
        locked_folds=1,
        min_training_rows=60,
    )
    bootstrap = first["paired_bootstrap"]
    assert bootstrap == second["paired_bootstrap"]
    assert bootstrap["research_only"] is True
    assert bootstrap["selection_allowed"] is False
    assert bootstrap["same_oos_cases"] is True
    assert bootstrap["ci_95_low"] <= bootstrap["observed_delta_candidate_minus_dynamic"] <= bootstrap["ci_95_high"]


def test_paired_bootstrap_is_deterministic_and_fail_closed():
    ledger, folds = _make_data([i % 2 for i in range(80)])
    result = analyze_expert_loss_routing(
        ledger,
        folds,
        models=MODELS,
        locked_folds=1,
        min_training_rows=60,
    )
    bootstrap = result["paired_bootstrap"]
    assert bootstrap["status"] == "EXECUTED_PAIRED_BOOTSTRAP"
    assert bootstrap["research_only"] is True
    assert bootstrap["selection_allowed"] is False
    assert bootstrap["same_oos_cases"] is True
    assert bootstrap["ci_95_low"] <= bootstrap["observed_delta_candidate_minus_dynamic"] <= bootstrap["ci_95_high"]


def test_locked_suffix_is_frozen_across_multiple_locked_folds():
    ledger_a, folds_a = _make_data([i % 2 for i in range(80)], locked_folds=2)
    ledger_b, folds_b = _make_data([1 - (i % 2) for i in range(80)], locked_folds=2)
    result_a = analyze_expert_loss_routing(
        ledger_a,
        folds_a,
        models=MODELS,
        locked_folds=2,
        min_training_rows=60,
    )
    result_b = analyze_expert_loss_routing(
        ledger_b,
        folds_b,
        models=MODELS,
        locked_folds=2,
        min_training_rows=60,
    )
    assert result_a["status"] == "EVALUATED"
    assert result_b["status"] == "EVALUATED"
    assert result_a["contracts"]["locked_suffix_routing_is_frozen_across_all_locked_folds"] is True
    # The second locked fold must not be retrained on the first locked fold's outcome.
    a_second = result_a["fold_results"][-1]
    b_second = result_b["fold_results"][-1]
    assert a_second["predicted_loss_mean"] == b_second["predicted_loss_mean"]
    assert a_second["weight_means"] == b_second["weight_means"]


def test_locked_flag_mismatch_fails_closed():
    ledger, folds = _make_data([i % 2 for i in range(80)], locked_folds=2)
    ledger[160]["is_locked"] = False
    result = analyze_expert_loss_routing(
        ledger,
        folds,
        models=MODELS,
        locked_folds=2,
        min_training_rows=60,
    )
    assert result["status"] == "BLOCKED_LOCKED_FLAG_MISMATCH"
