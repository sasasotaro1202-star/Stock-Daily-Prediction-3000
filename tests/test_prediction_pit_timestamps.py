from __future__ import annotations

import pandas as pd

from src.validation.leakage import audit_prediction_snapshot_timestamps


def _frame(available: str, retrieved: str | None = None) -> pd.DataFrame:
    data = {"available_at": [available]}
    if retrieved is not None:
        data["retrieved_at"] = [retrieved]
    return pd.DataFrame(data)


def test_live_snapshot_requires_retrieval_provenance() -> None:
    result = audit_prediction_snapshot_timestamps(
        _frame("2026-09-30T08:00:00Z"),
        pd.Timestamp("2026-09-30T09:00:00Z"),
    )
    assert result.ok is False
    assert "missing:retrieved_at" in result.violations


def test_live_snapshot_accepts_causally_ordered_timestamps() -> None:
    result = audit_prediction_snapshot_timestamps(
        _frame("2026-09-30T08:00:00Z", "2026-09-30T08:05:00Z"),
        pd.Timestamp("2026-09-30T09:00:00Z"),
    )
    assert result.ok is True


def test_live_snapshot_rejects_future_retrieval() -> None:
    result = audit_prediction_snapshot_timestamps(
        _frame("2026-09-30T08:00:00Z", "2026-09-30T10:00:00Z"),
        pd.Timestamp("2026-09-30T09:00:00Z"),
    )
    assert result.ok is False
    assert "retrieved_at_after_prediction_time:1" in result.violations


def test_live_snapshot_rejects_available_after_retrieval() -> None:
    result = audit_prediction_snapshot_timestamps(
        _frame("2026-09-30T09:00:00Z", "2026-09-30T08:00:00Z"),
        pd.Timestamp("2026-09-30T10:00:00Z"),
    )
    assert result.ok is False
    assert "available_at_after_retrieved_at:1" in result.violations


def test_live_snapshot_rejects_na_or_future_available_timestamp() -> None:
    frame = pd.DataFrame(
        {
            "available_at": [None, "2026-09-30T10:00:00Z"],
            "retrieved_at": ["2026-09-30T08:00:00Z", "2026-09-30T10:05:00Z"],
        }
    )
    result = audit_prediction_snapshot_timestamps(
        frame,
        pd.Timestamp("2026-09-30T09:00:00Z"),
    )
    assert result.ok is False
    assert "invalid_available_at:1" in result.violations
    assert "available_at_after_prediction_time:1" in result.violations
    assert "retrieved_at_after_prediction_time:1" in result.violations
