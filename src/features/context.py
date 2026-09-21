from __future__ import annotations

import pandas as pd


def add_market_context(
    df:pd.DataFrame,
    context:pd.DataFrame,
) -> pd.DataFrame:
    out=df.copy()
    required={"session_date","family","ret_1d","volatility_20","close"}
    if context.empty or not required.issubset(context.columns):
        raise ValueError("market context is missing required columns")
    ctx=context.copy()
    ctx["session_date"]=pd.to_datetime(ctx["session_date"],errors="coerce").dt.date
    for family in sorted(ctx["family"].dropna().unique()):
        part=ctx[ctx["family"].eq(family)].sort_values("session_date").copy()
        # Lag by one published session: never use the same-day context value.
        part["ret_lag1"]=part["ret_1d"].shift(1)
        part["vol_lag1"]=part["volatility_20"].shift(1)
        part["close_lag1"]=part["close"].shift(1)
        cols=["session_date","ret_lag1","vol_lag1","close_lag1"]
        part=part[cols].rename(columns={
            "ret_lag1":f"__{family}_ret",
            "vol_lag1":f"__{family}_vol",
            "close_lag1":f"__{family}_close",
        })
        out=out.merge(part,on="session_date",how="left")
    mapping={
        "nikkei_ret_1d_lag1":"__nikkei_ret",
        "topix_ret_1d_lag1":"__topix_ret",
        "sp500_ret_1d_lag1":"__sp500_ret",
        "nasdaq_ret_1d_lag1":"__nasdaq_ret",
        "vix_level_lag1":"__vix_close",
        "usd_jpy_ret_1d_lag1":"__usd_jpy_ret",
        "jp_market_volatility_lag1":"__nikkei_vol",
        "us_market_volatility_lag1":"__sp500_vol",
    }
    for target,source in mapping.items():
        out[target]=out[source]
    drop=[c for c in out.columns if c.startswith("__")]
    return out.drop(columns=drop)


def add_cross_sectional_context(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy().sort_values(["session_date","symbol"]).reset_index(drop=True)
    g=out.groupby("session_date",sort=False)
    out["cs_ret_1d_rank"]=g["ret_1d"].rank(method="average",pct=True)
    out["cs_vol_rank"]=g["volatility_20"].rank(method="average",pct=True,ascending=False)
    daily=g.agg(
        median_vol=("volatility_20","median"),
        median_ret=("ret_1d","median"),
        breadth_up=("ret_1d",lambda s:float((s>0).mean())),
    ).reset_index()
    out=out.merge(daily,on="session_date",how="left")
    if "sector" in out.columns:
        sg=out.groupby(["session_date","sector"],sort=False)
        out["sector_ret_1d"]=sg["ret_1d"].transform("mean")
        out["relative_strength_1d"]=out["ret_1d"]-out["sector_ret_1d"]
        out["sector_rank"]=sg["ret_1d"].rank(method="average",pct=True)
    if "market_ret_1d" in out.columns:
        out["market_relative_ret_1d"]=out["ret_1d"]-out["market_ret_1d"]
    return out
