from __future__ import annotations

import pandas as pd


def cap_training_rows(
    df: pd.DataFrame,
    *,
    max_rows: int = 300_000,
    recent_sessions: int = 252,
) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    if "session_date" not in df.columns:
        raise ValueError("session_date is required for deterministic sampling")

    out=df.copy()
    dates=sorted(pd.to_datetime(out["session_date"],errors="coerce").dt.date.dropna().unique())
    recent=set(dates[-recent_sessions:])
    recent_df=out[pd.to_datetime(out["session_date"],errors="coerce").dt.date.isin(recent)].copy()
    old_df=out[~pd.to_datetime(out["session_date"],errors="coerce").dt.date.isin(recent)].copy()

    budget=max(0,max_rows-len(recent_df))
    if budget==0:
        return recent_df.sort_values(["session_date","symbol"]).tail(max_rows)

    if len(old_df) > budget:
        stride=max(2,(len(old_df)+budget-1)//budget)
        old_df=old_df.sort_values(["session_date","symbol"]).iloc[::stride]
        if len(old_df)>budget:
            old_df=old_df.tail(budget)

    result=pd.concat([old_df,recent_df],ignore_index=True)
    if len(result)>max_rows:
        result=result.sort_values(["session_date","symbol"]).tail(max_rows)
    return result.sort_values(["session_date","symbol"]).reset_index(drop=True)
