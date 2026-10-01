from __future__ import annotations

import math

from src.research.case_risk_oos import analyze_case_risk, case_risk_score


def _row(fold: int, row: int, *, locked: bool, risk: float, result: int, p: float) -> dict:
    return {
        "fold": fold,
        "row": row,
        "is_locked": locked,
        "prediction": p,
        "result": result,
        "case_predictability": 1.0 - risk,
        "case_ood": risk,
        "case_failure_risk": risk,
        "case_disagreement": risk,
    }


def test_case_risk_score_is_outcome_free_and_bounded():
    row = _row(0, 0, locked=True, risk=0.8, result=0, p=0.9)
    score_a = case_risk_score({**row, "result": 0})
    score_b = case_risk_score({**row, "result": 1})
    assert score_a == score_b
    assert math.isclose(score_a, 0.8)
    assert 0.0 <= score_a <= 1.0


def test_case_risk_fails_closed_on_missing_component():
    row = _row(0, 0, locked=True, risk=0.8, result=0, p=0.9)
    row.pop("case_ood")
    assert case_risk_score(row) is None


def test_legacy_fold_level_signals_cannot_substitute_for_case_signals():
    row = _row(0, 0, locked=True, risk=0.8, result=0, p=0.9)
    row.update(
        {
            "predictability": row.pop("case_predictability"),
            "ood": row.pop("case_ood"),
            "failure_risk": row.pop("case_failure_risk"),
            "disagreement": row.pop("case_disagreement"),
        }
    )
    assert case_risk_score(row) is None


def test_threshold_uses_all_strictly_prior_oos_scores():
    rows = [
        _row(0, 0, locked=False, risk=0.1, result=0, p=0.9),
        _row(0, 1, locked=False, risk=0.2, result=0, p=0.1),
        _row(1, 0, locked=False, risk=0.2, result=0, p=0.9),
        _row(1, 1, locked=False, risk=0.3, result=1, p=0.1),
        _row(2, 0, locked=True, risk=0.05, result=0, p=0.9),
        _row(2, 1, locked=True, risk=0.9, result=1, p=0.9),
    ]
    result = analyze_case_risk(rows, risk_quantile=0.75)
    assert result["status"] == "EVALUATED"
    assert len(result["per_fold"]) == 1
    fold = result["per_fold"][0]
    assert fold["threshold_source"] == "strictly_prior_oos_scores"
    assert fold["prior_score_count"] == 4
    assert math.isclose(fold["high_risk_threshold"], 0.225)
    assert fold["high_risk_coverage"] == 0.5


def test_only_locked_rows_are_evaluated_and_contracts_are_explicit():
    rows = [
        _row(0, 0, locked=False, risk=0.95, result=1, p=0.95),
        _row(0, 1, locked=True, risk=0.2, result=0, p=0.9),
        _row(1, 0, locked=True, risk=0.7, result=0, p=0.9),
        _row(1, 1, locked=True, risk=0.7, result=1, p=0.9),
    ]
    result = analyze_case_risk(rows, risk_quantile=0.75)
    assert result["locked_rows"] == 3
    assert result["scored_rows"] == 3
    assert result["contracts"]["only_locked_oos_rows_evaluated"] is True
    assert result["contracts"]["threshold_uses_strictly_prior_oos_scores"] is True
    assert result["contracts"]["outcomes_used_for_threshold"] is False
    assert result["promotion_allowed"] is False
