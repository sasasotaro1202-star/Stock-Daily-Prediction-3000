from __future__ import annotations
import pandas as pd

def cross_sectional_rank(df:pd.DataFrame,probability_col="p_up_1d",return_col="expected_return_1d")->pd.DataFrame:
    out=df.copy()
    if probability_col not in out or return_col not in out: raise ValueError("required prediction columns are missing")
    out["rank_probability"]=out.groupby("prediction_date")[probability_col].rank(method="average",ascending=False,pct=True)
    out["rank_expected_return"]=out.groupby("prediction_date")[return_col].rank(method="average",ascending=False,pct=True)
    out["rank_score"]=0.5*out["rank_probability"]+0.5*out["rank_expected_return"]
    return out.sort_values(["prediction_date","rank_score"],ascending=[True,False])
