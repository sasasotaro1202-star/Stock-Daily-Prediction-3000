from __future__ import annotations
import pandas as pd

def add_targets(df:pd.DataFrame,group_col:str="symbol")->pd.DataFrame:
    out=df.copy().sort_values([group_col,"session_date"]).reset_index(drop=True)
    next_close=out.groupby(group_col,sort=False)["adj_close"].shift(-1)
    close5=out.groupby(group_col,sort=False)["adj_close"].shift(-5)
    base=out["adj_close"]
    r1=next_close/base-1.0
    r5=close5/base-1.0
    raw_next=out.groupby(group_col,sort=False)["close"].shift(-1)
    out["raw_price_return_1d"]=raw_next/out["close"]-1.0
    out["target_ret_1d"]=r1
    out["target_up_1d"]=(r1>0).astype("float").where(r1.notna())
    out["target_ret_5d"]=r5
    out["target_up_5d_3pct"]=(r5>0.03).astype("float").where(r5.notna())
    out["target_down_5d_3pct"]=(r5<-0.03).astype("float").where(r5.notna())
    return out
