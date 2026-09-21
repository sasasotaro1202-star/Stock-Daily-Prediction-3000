from __future__ import annotations
import pandas as pd

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
