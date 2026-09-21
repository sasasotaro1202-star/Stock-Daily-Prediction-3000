from __future__ import annotations

import pandas as pd


def add_targets(df: pd.DataFrame, group_col: str = "symbol") -> pd.DataFrame:
    out=df.copy().sort_values([group_col,"session_date"]).reset_index(drop=True)
    next_close=out.groupby(group_col,sort=False)["close"].shift(-1)
    close5=out.groupby(group_col,sort=False)["close"].shift(-5)
    base=out["close"]
    r1=next_close/base-1.0
    r5=close5/base-1.0
    raw_next=next_close
    out["raw_price_return_1d"]=r1

    # A split changes the quoted price scale. Exclude observations whose
    # current or next session is a split so raw-price returns stay economic.
    if "stock_splits" in out.columns:
        current_split=out["stock_splits"].fillna(0).ne(0)
        next_split=out.groupby(group_col,sort=False)["stock_splits"].shift(-1).fillna(0).ne(0)
        split_affected=current_split|next_split
        r1=r1.mask(split_affected)
        r5=r5.mask(current_split)

    out["target_ret_1d"]=r1
    out["target_up_1d"]=(r1>0).astype("float").where(r1.notna())
    out["target_ret_5d"]=r5
    out["target_up_5d_3pct"]=(r5>0.03).astype("float").where(r5.notna())
    out["target_down_5d_3pct"]=(r5<-0.03).astype("float").where(r5.notna())
    return out
