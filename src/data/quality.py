from __future__ import annotations
import pandas as pd

def validate_daily_bars(frame:pd.DataFrame)->dict[str,object]:
    required={"symbol","session_date","open","high","low","close","volume"}
    missing=required-set(frame.columns)
    if missing: return {"ok":False,"reason":f"missing:{sorted(missing)}"}
    dup=frame.duplicated(["symbol","session_date"]).sum()
    bad=((frame["high"]<frame[["open","close"]].max(axis=1))|
         (frame["low"]>frame[["open","close"]].min(axis=1))|
         (frame["volume"]<0)).sum()
    nulls=frame[list(required)].isna().any(axis=1).sum()
    return {"ok":dup==0 and bad==0 and nulls==0,"duplicates":int(dup),
            "bad_ohlc_or_volume":int(bad),"required_null_rows":int(nulls),"rows":int(len(frame))}
