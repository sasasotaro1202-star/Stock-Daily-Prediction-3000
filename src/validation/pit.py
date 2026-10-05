from __future__ import annotations
import pandas as pd
REQUIRED={"session_date","available_at"}

def assert_pit(frame:pd.DataFrame,prediction_time:pd.Timestamp)->None:
    missing=REQUIRED-set(frame.columns)
    if missing: raise ValueError(f"missing PIT columns: {sorted(missing)}")
    available=pd.to_datetime(frame["available_at"],utc=True,errors="coerce")
    if available.isna().any(): raise ValueError("invalid available_at")
    if (available>prediction_time).any(): raise ValueError("future information detected")

def drop_unavailable(frame:pd.DataFrame,prediction_time:pd.Timestamp)->pd.DataFrame:
    out=frame.copy()
    out["available_at"]=pd.to_datetime(out["available_at"],utc=True,errors="coerce")
    return out.loc[out["available_at"]<=prediction_time].copy()
