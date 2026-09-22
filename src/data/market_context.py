from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

CONTEXT_SYMBOLS={
    "nikkei":"^N225",
    "topix":"998405.T",
    "sp500":"^GSPC",
    "nasdaq":"^IXIC",
    "vix":"^VIX",
    "usd_jpy":"USDJPY=X",
    "us10y":"^TNX",
    "dxy":"DX-Y.NYB",
    "gold":"GC=F",
    "oil":"CL=F",
    "hyg":"HYG",
}


def _available_at(session_date, family: str) -> pd.Timestamp:
    if family in {"nikkei","topix"}:
        dt=datetime.combine(
            session_date,
            time(16,30),
            tzinfo=ZoneInfo("Asia/Tokyo"),
        )
    elif family in {"sp500","nasdaq","vix","us10y"}:
        dt=datetime.combine(
            session_date,
            time(16,30),
            tzinfo=ZoneInfo("America/New_York"),
        )
    elif family in {"dxy","gold","oil","hyg"}:
        # Conservative U.S. cross-asset availability: after the regular
        # U.S. session, avoiding any same-day close look-ahead for Japan.
        dt=datetime.combine(
            session_date,
            time(18,0),
            tzinfo=ZoneInfo("America/New_York"),
        )
    else:
        dt=datetime.combine(
            session_date,
            time(18,0),
            tzinfo=ZoneInfo("Asia/Tokyo"),
        )
    return pd.Timestamp(dt).tz_convert("UTC")


def download_market_context(period: str="5y") -> pd.DataFrame:
    symbols=list(CONTEXT_SYMBOLS.values())
    raw=yf.download(
        symbols,
        period=period,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    frames=[]
    if isinstance(raw.columns,pd.MultiIndex):
        for family, symbol in CONTEXT_SYMBOLS.items():
            if symbol not in raw.columns.get_level_values(0):
                continue
            part=raw[symbol].reset_index().rename(columns=str.lower)
            if "date" not in part or "close" not in part:
                continue
            part["session_date"]=pd.to_datetime(part["date"]).dt.date
            part["close"]=pd.to_numeric(part["close"],errors="coerce")
            part=part.dropna(subset=["close"]).sort_values("session_date")
            part["ret_1d"]=part["close"].pct_change()
            part["volatility_20"]=part["ret_1d"].rolling(
                20,min_periods=20
            ).std()
            part["family"]=family
            part["provider_symbol"]=symbol
            part["available_at"]=part["session_date"].map(
                lambda d:_available_at(d,family)
            )
            frames.append(
                part[
                    [
                        "session_date","family","provider_symbol",
                        "close","ret_1d","volatility_20","available_at"
                    ]
                ]
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames,ignore_index=True)


def write_market_context(path: str="data/market_context.parquet", period: str="5y") -> int:
    df=download_market_context(period=period)
    if df.empty:
        raise SystemExit("DEFERRED: market context returned no rows")
    out=Path(path)
    out.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(out,index=False)
    return len(df)
