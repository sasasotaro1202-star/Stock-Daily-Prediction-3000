from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1d","ret_5d","ret_20d","sma_5","sma_20","sma_60","ema_20",
    "rsi_14","atr_14","volatility_20","volume_ratio_20","range_pct",
    "gap_pct","price_vs_sma20","price_vs_sma60",
]

def add_technical_features(df: pd.DataFrame, group_col: str = "symbol") -> pd.DataFrame:
    out=df.copy()
    required={"close","high","low","volume",group_col,"session_date"}
    missing=required-set(out.columns)
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    out=out.sort_values([group_col,"session_date"]).reset_index(drop=True)
    g=out.groupby(group_col,sort=False)
    close=g["close"]; volume=g["volume"]
    out["ret_1d"]=close.pct_change()
    out["ret_5d"]=close.pct_change(5)
    out["ret_20d"]=close.pct_change(20)
    out["sma_5"]=close.transform(lambda s:s.rolling(5,min_periods=5).mean())
    out["sma_20"]=close.transform(lambda s:s.rolling(20,min_periods=20).mean())
    out["sma_60"]=close.transform(lambda s:s.rolling(60,min_periods=60).mean())
    out["ema_20"]=close.transform(lambda s:s.ewm(span=20,adjust=False,min_periods=20).mean())
    delta=close.diff()
    gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    avg_gain=gain.transform(lambda s:s.rolling(14,min_periods=14).mean())
    avg_loss=loss.transform(lambda s:s.rolling(14,min_periods=14).mean())
    rs=avg_gain/avg_loss.replace(0,np.nan)
    out["rsi_14"]=100-(100/(1+rs))
    prev=out.groupby(group_col)["close"].shift(1)
    tr=pd.concat([(out["high"]-out["low"]),(out["high"]-prev).abs(),(out["low"]-prev).abs()],axis=1).max(axis=1)
    out["atr_14"]=tr.groupby(out[group_col]).transform(lambda s:s.rolling(14,min_periods=14).mean())
    out["volatility_20"]=out["ret_1d"].groupby(out[group_col]).transform(lambda s:s.rolling(20,min_periods=20).std())
    out["volume_ratio_20"]=volume.transform(lambda s:s/s.rolling(20,min_periods=20).mean())
    out["range_pct"]=(out["high"]-out["low"])/out["close"].replace(0,np.nan)
    out["gap_pct"]=(out["open"]/prev)-1.0
    out["price_vs_sma20"]=out["close"]/out["sma_20"]-1.0
    out["price_vs_sma60"]=out["close"]/out["sma_60"]-1.0
    return out
