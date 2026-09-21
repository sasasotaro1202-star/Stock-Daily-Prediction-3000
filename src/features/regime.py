from __future__ import annotations
import pandas as pd

def add_market_regime(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy().sort_values("session_date")
    daily=out.groupby("session_date").agg(
        median_vol=("volatility_20","median"),
        median_ret=("ret_1d","median"),
        breadth_up=("ret_1d",lambda s:float((s>0).mean())),
    ).reset_index()
    out=out.merge(daily,on="session_date",how="left",suffixes=("","_market"))
    out["market_regime"]=pd.Series("normal",index=out.index)
    high=(out["median_vol"]>=out["median_vol"].rolling(60,min_periods=20).quantile(0.75))
    trend=out["median_ret"].rolling(20,min_periods=20).mean().abs()>=0.005
    out.loc[high.fillna(False),"market_regime"]="high_vol"
    out.loc[(~high.fillna(False))&trend.fillna(False),"market_regime"]="trend"
    out.loc[out["breadth_up"].between(0.25,0.75),"market_regime"]="balanced"
    return out
