from __future__ import annotations

import math

from src.research.learned_case_risk_oos import analyze_learned_case_risk


def _row(fold: int, row: int, *, locked: bool, risk: float, result: int, p: float) -> dict:
    return {
        "fold": fold,
        "row": row,
        "is_locked": locked,
        "prediction": p,
        "result": result,
        "failed": int((p >= 0.5) != bool(result)),
        "prediction_time": "2026-01-02T00:00:00+00:00",
        "available_at": "2026-01-01T23:00:00+00:00",
        "pit_status": "PASS",
        "case_predictability": 1.0 - risk,
        "case_ood": risk,
        "case_failure_risk": risk,
        "case_disagreement": risk,
    }


def test_learned_case_risk_is_frozen_on_locked_suffix():
    rows = []
    for fold in range(4):
        for i in range(80):
            risk = 0.05 if i < 40 else 0.90
            result = 1 if i < 40 else 0
            p = 0.90 if result == 0 else 0.10
            rows.append(_row(fold, i, locked=False, risk=risk, result=result, p=p))
    for i in range(80):
        risk = 0.90 if i < 40 else 0.05
        result = 0 if i < 40 else 1
        p = 0.90 if result == 0 else 0.10
        rows.append(_row(4, i, locked=True, risk=risk, result=result, p=p))
    result = analyze_learned_case_risk(rows)
    assert result["status"] == "EVALUATED"
    assert result["frozen_holdout_used"] is True
    assert result["contracts"]["locked_suffix_model_is_frozen"] is True
    assert result["contracts"]["locked_outcomes_used_for_fit_or_threshold"] is False
    assert result["promotion_allowed"] is False


def test_learned_case_risk_blocks_invalid_locked_pit():
    row = _row(4, 0, locked=True, risk=0.8, result=0, p=0.9)
    row["available_at"] = "2026-01-02T01:00:00+00:00"
    result = analyze_learned_case_risk(
        [_row(0, i, locked=False, risk=0.2, result=i % 2, p=0.9 if i % 2 == 0 else 0.1) for i in range(240)]
        + [row]
    )
    assert result["status"] == "BLOCKED_INVALID_LOCKED_CASES"
    assert result["scored_rows"] == 0


def test_learned_case_risk_requires_explicit_case_features():
    rows = [
        _row(0, i, locked=False, risk=0.2, result=i % 2, p=0.9 if i % 2 == 0 else 0.1)
        for i in range(240)
    ]
    broken = _row(1, 0, locked=True, risk=0.8, result=0, p=0.9)
    broken.pop("case_disagreement")
    rows.append(broken)
    result = analyze_learned_case_risk(rows)
    assert result["status"] == "BLOCKED_INVALID_LOCKED_CASES"
