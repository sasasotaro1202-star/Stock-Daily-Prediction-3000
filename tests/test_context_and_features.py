from src.features.context import add_cross_sectional_context
import numpy as np
import pandas as pd


def test_market_relative_factors_are_causal_and_numeric():
    from src.features.context import add_market_context, add_cross_sectional_context

    rows=[]
    for asset, market_col in [("jp_stock","nikkei"),("us_stock","sp500")]:
        for i in range(25):
            d=pd.Timestamp("2026-01-01")+pd.Timedelta(days=i)
            rows.append({
                "symbol": f"{asset}_A",
                "asset_class": asset,
                "session_date": d,
                "available_at": pd.Timestamp(d).tz_localize("UTC")+pd.Timedelta(hours=16),
                "open":100+i,
                "high":101+i,
                "low":99+i,
                "close":100+i,
                "volume":1000,
                "ret_1d": 0.01 if i else np.nan,
                "volatility_20":0.02,
            })
    frame=pd.DataFrame(rows)
    # Minimal context family rows with distinct availability timestamps.
    ctx=[]
    for family in ("nikkei","sp500"):
        for i in range(25):
            d=pd.Timestamp("2026-01-01")+pd.Timedelta(days=i)
            ctx.append({
                "family":family,
                "session_date":d,
                "available_at":pd.Timestamp(d).tz_localize("UTC")+pd.Timedelta(hours=15),
                "ret_1d":0.005,
                "volatility_20":0.015,
                "close":100,
            })
    out=add_market_context(frame,pd.DataFrame(ctx))
    out=add_cross_sectional_context(out)
    assert {"market_beta_20","market_corr_20","market_residual_ret_1d"} <= set(out.columns)
    assert out["market_beta_20"].dtype.kind == "f"
