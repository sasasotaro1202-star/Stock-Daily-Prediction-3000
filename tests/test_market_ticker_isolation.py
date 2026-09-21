from pathlib import Path

import pandas as pd

from src.features.technical import add_technical_features
from src.prediction.targets import add_targets


def test_same_ticker_isolated_by_asset_class():
    rows=[]
    for asset in ("jp_stock","us_stock"):
        for i in range(70):
            close=100+i+(10 if asset=="us_stock" else 0)
            rows.append({
                "symbol":"SAME",
                "asset_class":asset,
                "session_date":pd.Timestamp("2025-01-01")+pd.Timedelta(days=i),
                "open":close-0.2,
                "high":close+1.0,
                "low":close-1.0,
                "close":close,
                "volume":1000+i,
            })
    out=add_technical_features(pd.DataFrame(rows))
    assert len(out)==140
    for asset in ("jp_stock","us_stock"):
        part=out[out.asset_class.eq(asset)]
        assert part.ret_1d.iloc[0] != part.ret_1d.iloc[1] or pd.isna(part.ret_1d.iloc[0])
        assert part.volatility_20.notna().any()


def test_same_ticker_target_isolated_by_asset_class():
    rows=[]
    for asset in ("jp_stock","us_stock"):
        for i in range(3):
            close=100+i+(100 if asset=="us_stock" else 0)
            rows.append({
                "symbol":"SAME",
                "asset_class":asset,
                "session_date":pd.Timestamp("2025-01-01")+pd.Timedelta(days=i),
                "close":close,
                "stock_splits":[0.0,0.0,0.0][i],
            })
    out=add_targets(pd.DataFrame(rows))
    jp=out[out.asset_class.eq("jp_stock")]
    us=out[out.asset_class.eq("us_stock")]
    assert jp.target_ret_1d.iloc[0] == us.target_ret_1d.iloc[0]
