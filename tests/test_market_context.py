from __future__ import annotations

import pandas as pd

from src.features.context import add_market_context


def test_market_context_is_lagged_by_one_published_session():
    dates=pd.to_datetime(["2026-09-21","2026-09-22","2026-09-23"]).date
    prices=pd.DataFrame({
        "symbol":["AAA"]*3,
        "session_date":dates,
        "ret_1d":[0.01,0.02,0.03],
        "volatility_20":[0.10,0.11,0.12],
    })
    context=pd.DataFrame({
        "session_date":dates,
        "family":["sp500"]*3,
        "ret_1d":[0.10,0.20,0.30],
        "volatility_20":[0.40,0.50,0.60],
        "close":[100.0,110.0,120.0],
    })
    out=add_market_context(prices,context)
    assert pd.isna(out.loc[0,"sp500_ret_1d_lag1"])
    assert out.loc[1,"sp500_ret_1d_lag1"]==0.10
    assert out.loc[2,"sp500_ret_1d_lag1"]==0.20
    assert out.loc[1,"sp500_ret_1d_lag1"] != context.loc[1,"ret_1d"]


def test_market_context_keeps_price_row_count():
    prices=pd.DataFrame({
        "symbol":["AAA","BBB"],
        "session_date":pd.to_datetime(["2026-09-22","2026-09-22"]).date,
        "ret_1d":[0.01,-0.01],
        "volatility_20":[0.10,0.20],
    })
    context=pd.DataFrame({
        "session_date":pd.to_datetime(["2026-09-21","2026-09-22"]).date,
        "family":["nikkei","nikkei"],
        "ret_1d":[0.01,0.02],
        "volatility_20":[0.30,0.31],
        "close":[100.0,101.0],
    })
    out=add_market_context(prices,context)
    assert len(out)==len(prices)
