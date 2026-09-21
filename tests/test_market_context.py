from __future__ import annotations

import pandas as pd

from src.features.context import add_market_context


def test_market_context_uses_latest_pit_observation():
    prices=pd.DataFrame({
        "symbol":["AAA","AAA"],
        "session_date":pd.to_datetime(
            ["2026-09-22","2026-09-22"]
        ).date,
        "available_at":pd.to_datetime([
            "2026-09-22T07:00:00Z",
            "2026-09-22T09:00:00Z",
        ]),
        "ret_1d":[0.01,0.02],
        "volatility_20":[0.10,0.11],
    })
    context=pd.DataFrame({
        "session_date":pd.to_datetime([
            "2026-09-21","2026-09-22"
        ]).date,
        "family":["sp500","sp500"],
        "available_at":pd.to_datetime([
            "2026-09-22T05:30:00Z",
            "2026-09-23T05:30:00Z",
        ]),
        "ret_1d":[0.10,0.20],
        "volatility_20":[0.40,0.50],
        "close":[100.0,110.0],
    })
    out=add_market_context(prices,context)
    assert out.loc[0,"sp500_ret_1d_lag1"]==0.10
    assert out.loc[0,"sp500_ret_1d_lag1"] != 0.20
    assert out.loc[1,"sp500_ret_1d_lag1"]==0.10


def test_market_context_keeps_price_row_count():
    prices=pd.DataFrame({
        "symbol":["AAA","BBB"],
        "session_date":pd.to_datetime(
            ["2026-09-22","2026-09-22"]
        ).date,
        "available_at":pd.to_datetime([
            "2026-09-22T07:00:00Z",
            "2026-09-22T07:00:00Z",
        ]),
        "ret_1d":[0.01,-0.01],
        "volatility_20":[0.10,0.20],
    })
    context=pd.DataFrame({
        "session_date":pd.to_datetime(
            ["2026-09-21","2026-09-22"]
        ).date,
        "family":["nikkei","nikkei"],
        "available_at":pd.to_datetime([
            "2026-09-21T07:30:00Z",
            "2026-09-22T07:30:00Z",
        ]),
        "ret_1d":[0.01,0.02],
        "volatility_20":[0.30,0.31],
        "close":[100.0,101.0],
    })
    out=add_market_context(prices,context)
    assert len(out)==len(prices)


def test_market_context_requires_available_at():
    prices=pd.DataFrame({
        "symbol":["AAA"],
        "session_date":[pd.Timestamp("2026-09-22").date()],
        "ret_1d":[0.01],
        "volatility_20":[0.10],
    })
    context=pd.DataFrame({
        "session_date":[pd.Timestamp("2026-09-21").date()],
        "family":["sp500"],
        "available_at":[pd.Timestamp("2026-09-22T05:30:00Z")],
        "ret_1d":[0.10],
        "volatility_20":[0.40],
        "close":[100.0],
    })
    try:
        add_market_context(prices,context)
    except ValueError as exc:
        assert "available_at" in str(exc)
    else:
        raise AssertionError("expected missing available_at to fail")
