from __future__ import annotations

import pandas as pd

from src.validation.independent_audit import audit_raw_inputs


def test_independent_audit_reports_source_level_pit_coverage() -> None:
    prices = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "asset_class": ["stock", "stock", "stock"],
            "session_date": pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-02"]),
            "available_at": pd.to_datetime(
                [
                    "2026-01-02T08:00:00Z",
                    "2026-01-03T08:00:00Z",
                    "2026-01-02T08:00:00Z",
                ],
                utc=True,
            ),
            "retrieved_at": pd.to_datetime(
                [
                    "2026-01-02T08:05:00Z",
                    None,
                    "2026-01-02T08:05:00Z",
                ],
                utc=True,
            ),
            "source": ["alpha", "alpha", "beta"],
            "provider_symbol": ["AAA", "AAA", "BBB"],
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    context = pd.DataFrame(
        {
            "family": ["macro", "macro"],
            "session_date": pd.to_datetime(["2026-01-02", "2026-01-03"]),
            "available_at": pd.to_datetime(
                ["2026-01-02T07:00:00Z", "2026-01-03T07:00:00Z"], utc=True
            ),
        }
    )

    result = audit_raw_inputs(
        prices,
        context,
        now=pd.Timestamp("2026-01-04T00:00:00Z"),
    )
    assert result.ok is True
    alpha = result.provenance["prices_by_source"]["groups"]["alpha"]
    beta = result.provenance["prices_by_source"]["groups"]["beta"]
    assert alpha["rows"] == 2
    assert alpha["available_at_coverage"] == 1.0
    assert alpha["retrieved_at_coverage"] == 0.5
    assert beta["available_at_le_retrieved_at_coverage"] == 1.0


def test_independent_audit_reports_missing_group_column_without_failure() -> None:
    prices = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "asset_class": ["stock"],
            "session_date": pd.to_datetime(["2026-01-02"]),
            "available_at": pd.to_datetime(["2026-01-02T08:00:00Z"], utc=True),
            "source": ["alpha"],
            "provider_symbol": ["AAA"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "volume": [1.0],
        }
    )
    context = pd.DataFrame(
        {
            "family": ["macro"],
            "session_date": pd.to_datetime(["2026-01-02"]),
            "available_at": pd.to_datetime(["2026-01-02T07:00:00Z"], utc=True),
        }
    )
    result = audit_raw_inputs(
        prices,
        context,
        now=pd.Timestamp("2026-01-03T00:00:00Z"),
    )
    assert result.provenance["market_context_by_family"]["available"] is True
    assert result.provenance["prices_by_source"]["available"] is True
