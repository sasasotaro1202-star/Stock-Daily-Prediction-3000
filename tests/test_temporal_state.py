from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.temporal_state import (
    TEMPORAL_STATE_COLUMNS,
    audit_temporal_state_causality,
    build_temporal_state_features,
)


def _frame(values: list[float]) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame(
        {
            "symbol": ["A"] * n,
            "market_family": ["JP_stock"] * n,
            "session_date": pd.date_range("2026-01-01", periods=n, freq="D"),
            "open": values,
            "high": np.asarray(values) * 1.01,
            "low": np.asarray(values) * 0.99,
            "close": values,
            "volume": np.arange(100, 100 + n, dtype=float) * 10,
        }
    )


def test_temporal_state_is_causal() -> None:
    base = _frame([100, 101, 103, 102, 105, 107, 106, 109, 110, 111, 112, 114, 115, 116, 118])
    changed = base.copy()
    changed.loc[changed.index[-1], ["close", "volume"]] = [99999.0, 999999.0]

    left = build_temporal_state_features(base).iloc[:-1][list(TEMPORAL_STATE_COLUMNS)].reset_index(drop=True)
    right = build_temporal_state_features(changed).iloc[:-1][list(TEMPORAL_STATE_COLUMNS)].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=0, atol=1e-12)


def test_temporal_state_has_expected_schema() -> None:
    out = build_temporal_state_features(_frame(list(np.linspace(100, 160, 60))))
    assert list(TEMPORAL_STATE_COLUMNS) == list(out[TEMPORAL_STATE_COLUMNS].columns)
    assert int(out[TEMPORAL_STATE_COLUMNS].notna().all(axis=1).sum()) > 0


def test_temporal_state_audit_is_research_only() -> None:
    audit = audit_temporal_state_causality(_frame([100 + i for i in range(25)]))
    assert audit["status"] == "PASS"
    assert audit["research_only"] is True
    assert audit["production_changed"] is False
    assert audit["future_row_reference"] is False
