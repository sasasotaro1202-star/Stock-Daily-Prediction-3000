import pandas as pd

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    required={"symbol","session_date","close","volume"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    out=df.sort_values(["symbol","session_date"]).copy()
    g=out.groupby("symbol",group_keys=False)
    out["ret_1d"]=g["close"].pct_change()
    out["ret_5d"]=g["close"].pct_change(5)
    out["ret_20d"]=g["close"].pct_change(20)
    out["sma_20"]=g["close"].transform(lambda s:s.rolling(20,min_periods=20).mean())
    out["volatility_20"]=g["ret_1d"].transform(lambda s:s.rolling(20,min_periods=20).std())
    out["volume_ratio_20"]=g["volume"].transform(lambda s:s/s.rolling(20,min_periods=20).mean())
    return out
