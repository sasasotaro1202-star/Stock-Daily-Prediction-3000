from __future__ import annotations

import pandas as pd


def restrict_to_lookback(
    df: pd.DataFrame,
    lookback_sessions: int | None,
    date_column: str = "session_date",
) -> pd.DataFrame:
    if lookback_sessions in (None, 0):
        return df.copy()
    n = int(lookback_sessions)
    if n < 1:
        raise ValueError("lookback_sessions must be >= 1, 0, or None")
    if date_column not in df.columns:
        raise ValueError(f"missing date column: {date_column}")
    dates = sorted(pd.to_datetime(df[date_column], errors="coerce").dropna().dt.date.unique())
    keep = set(dates[-n:])
    return df.loc[pd.to_datetime(df[date_column], errors="coerce").dt.date.isin(keep)].copy()
