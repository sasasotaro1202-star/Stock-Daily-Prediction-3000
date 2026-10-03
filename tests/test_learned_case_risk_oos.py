from __future__ import annotations

from src.research.learned_case_risk_oos import analyze_learned_case_risk


def _row(
    fold: int,
    row: int,
    *,
    locked: bool,
    risk: float,
    result: int,
    p: float,
) -> dict:
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


def _development_rows() -> list[dict]:
    rows = []
    for fold in range(6):
        for i in range(80):
            risk = 0.05 if i < 40 else 0.90
            result = i % 2
            correct = i % 10 < (8 if risk < 0.5 else 2)
            predicted_class = result if correct else 1 - result
            p = 0.90 if predicted_class == 1 else 0.10
            rows.append(
                _row(
                    fold,
                    i,
                    locked=False,
                    risk=risk,
                    result=result,
                    p=p,
                )
            )
    return rows


def test_learned_case_risk_is_frozen_on_locked_suffix():
    rows = _development_rows()
    for i in range(80):
        risk = 0.90 if i < 40 else 0.05
        result = i % 2
        correct = i % 10 >= (8 if risk < 0.5 else 2)
        predicted_class = result if correct else 1 - result
        p = 0.90 if predicted_class == 1 else 0.10
        rows.append(
            _row(
                6,
                i,
                locked=True,
                risk=risk,
                result=result,
                p=p,
            )
        )
    result = analyze_learned_case_risk(rows)
    assert result["status"] == "EVALUATED"
    assert result["frozen_holdout_used"] is True
    assert result["contracts"]["locked_suffix_model_is_frozen"] is True
    assert result["contracts"]["locked_outcomes_used_for_fit_or_threshold"] is False
    assert result["promotion_allowed"] is False
    for method in ("fixed_case_risk", "learned_case_risk"):
        metrics = result[method]["risk_metrics"]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["logloss"] >= 0.0
        assert metrics["brier"] >= 0.0
        assert 0.0 <= metrics["ece"] <= 1.0
    assert len(result["development_fold_metrics"]) == 3
    assert result["stability"]["development_fold_count"] == 3
    assert result["stability"]["development_learned_logloss"]["folds"] == 3
    assert result["stability"]["development_fixed_logloss"]["folds"] == 3
    assert len(result["fold_metrics"]) == 1
    assert result["stability"]["locked_fold_count"] == 1
    assert result["stability"]["locked_learned_logloss"]["folds"] == 1
    assert result["stability"]["locked_fixed_logloss"]["folds"] == 1


def test_learned_case_risk_blocks_invalid_locked_pit():
    rows = _development_rows()
    row = _row(4, 0, locked=True, risk=0.8, result=0, p=0.9)
    row["available_at"] = "2026-01-02T01:00:00+00:00"
    rows.append(row)
    result = analyze_learned_case_risk(rows)
    assert result["status"] == "BLOCKED_INVALID_LOCKED_CASES"
    assert result["scored_rows"] == 0


def test_learned_case_risk_requires_explicit_case_features():
    rows = _development_rows()
    broken = _row(4, 0, locked=True, risk=0.8, result=0, p=0.9)
    broken.pop("case_disagreement")
    rows.append(broken)
    result = analyze_learned_case_risk(rows)
    assert result["status"] == "BLOCKED_INVALID_LOCKED_CASES"


def test_learned_case_risk_rejects_future_publication_timestamp():
    rows = _development_rows()
    broken = _row(4, 0, locked=True, risk=0.8, result=0, p=0.9)
    broken["published_at"] = "2026-01-02T01:00:00+00:00"
    rows.append(broken)
    result = analyze_learned_case_risk(rows)
    assert result["status"] == "BLOCKED_INVALID_LOCKED_CASES"


def test_learned_case_risk_rejects_impossible_retrieval_order():
    rows = _development_rows()
    broken = _row(4, 0, locked=True, risk=0.8, result=0, p=0.9)
    broken["retrieved_at"] = "2026-01-01T22:00:00+00:00"
    rows.append(broken)
    result = analyze_learned_case_risk(rows)
    assert result["status"] == "BLOCKED_INVALID_LOCKED_CASES"


def test_learned_case_risk_rejects_malformed_optional_pit_timestamp():
    rows = _development_rows()
    broken = _row(4, 0, locked=True, risk=0.8, result=0, p=0.9)
    broken["published_at"] = "not-a-timestamp"
    rows.append(broken)
    result = analyze_learned_case_risk(rows)
    assert result["status"] == "BLOCKED_INVALID_LOCKED_CASES"



def test_learned_case_risk_rejects_post_cutoff_retrieval():
    rows = _development_rows()
    broken = _row(4, 0, locked=True, risk=0.8, result=0, p=0.9)
    broken["retrieved_at"] = "2026-01-02T01:00:00+00:00"
    rows.append(broken)
    result = analyze_learned_case_risk(rows)
    assert result["status"] == "BLOCKED_INVALID_LOCKED_CASES"


def test_learned_case_risk_rejects_retrieval_before_publication():
    rows = _development_rows()
    broken = _row(4, 0, locked=True, risk=0.8, result=0, p=0.9)
    broken["published_at"] = "2026-01-01T23:30:00+00:00"
    broken["retrieved_at"] = "2026-01-01T23:15:00+00:00"
    rows.append(broken)
    result = analyze_learned_case_risk(rows)
    assert result["status"] == "BLOCKED_INVALID_LOCKED_CASES"

def test_learned_case_risk_fails_closed_on_invalid_development_pit():
    rows = _development_rows()
    rows.append(_row(6, 0, locked=True, risk=0.8, result=0, p=0.9))
    rows[0]["available_at"] = "2026-01-03T00:00:00+00:00"
    result = analyze_learned_case_risk(rows)
    assert result["status"] == "BLOCKED_INVALID_DEVELOPMENT_PIT"
    assert result["invalid_development_pit_rows"] == 1
    assert result["scored_rows"] == 0


def test_learned_case_risk_fails_closed_on_malformed_development_pit():
    rows = _development_rows()
    rows.append(_row(6, 0, locked=True, risk=0.8, result=0, p=0.9))
    rows[1]["retrieved_at"] = "invalid"
    result = analyze_learned_case_risk(rows)
    assert result["status"] == "BLOCKED_INVALID_DEVELOPMENT_PIT"
    assert result["invalid_development_pit_rows"] == 1
