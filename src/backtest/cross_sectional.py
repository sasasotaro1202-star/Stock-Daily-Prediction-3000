from __future__ import annotations
import numpy as np
import pandas as pd

def evaluate_predictions(predictions:pd.DataFrame,bars:pd.DataFrame,top_quantile:float=0.1,cost_bps:float=5.0)->dict:
    p=predictions.copy(); b=bars.copy()
    p["prediction_date"]=pd.to_datetime(p["prediction_date"]).dt.date
    b=b.sort_values(["symbol","session_date"])
    b["forward_return_1d"]=b.groupby("symbol")["close"].shift(-1)/b["close"]-1.0
    outcome=b[["symbol","session_date","forward_return_1d","available_at"]].rename(columns={"session_date":"prediction_date","available_at":"outcome_available_at"})
    outcome["prediction_date"]=pd.to_datetime(outcome["prediction_date"]).dt.date
    m=p.merge(outcome,on=["symbol","prediction_date"],how="inner")
    if "prediction_time" in m:
        m["prediction_time"]=pd.to_datetime(m["prediction_time"],utc=True)
        m["outcome_available_at"]=pd.to_datetime(m["outcome_available_at"],utc=True)
        m=m[m["outcome_available_at"]>m["prediction_time"]].copy()
    rows=[]
    for date,g in m.groupby("prediction_date"):
        g=g.dropna(subset=["expected_return_1d","forward_return_1d"])
        if len(g)<20:continue
        q=g["expected_return_1d"].rank(pct=True)
        long=g.loc[q>=1-top_quantile,"forward_return_1d"].mean()
        short=g.loc[q<=top_quantile,"forward_return_1d"].mean()
        spread=long-short-(2*cost_bps/10000)
        rows.append({"date":str(date),"n":len(g),"long_return":float(long),"short_return":float(short),"long_short_return":float(spread)})
    if not rows:return {"status":"DEFERRED","rows":0}
    r=pd.DataFrame(rows); equity=(1+r["long_short_return"].fillna(0)).cumprod()
    peak=equity.cummax(); drawdown=equity/peak-1
    result={"status":"PASS","rows":int(len(r)),"mean_daily_spread":float(r["long_short_return"].mean()),
            "vol_daily_spread":float(r["long_short_return"].std(ddof=0)),
            "cumulative_spread":float(equity.iloc[-1]-1),
            "max_drawdown":float(drawdown.min())}
    return result
