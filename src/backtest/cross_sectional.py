from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_predictions(
    predictions: pd.DataFrame,
    bars: pd.DataFrame,
    top_quantile: float = 0.1,
    cost_bps: float = 5.0,
) -> dict:
    p=predictions.copy()
    b=bars.copy()

    required_pred={"symbol","session_date","expected_return_1d"}
    required_bar={"symbol","session_date","close","available_at"}
    if not required_pred.issubset(p.columns):
        return {
            "status":"DEFERRED",
            "rows":0,
            "reason":"prediction_schema_missing",
        }
    if not required_bar.issubset(b.columns):
        return {
            "status":"DEFERRED",
            "rows":0,
            "reason":"bar_schema_missing",
        }

    p["session_date"]=pd.to_datetime(
        p["session_date"],errors="coerce"
    ).dt.date
    b["session_date"]=pd.to_datetime(
        b["session_date"],errors="coerce"
    ).dt.date
    b=b.sort_values(["symbol","session_date"])

    # The prediction is made from the last known session bar. Joining on that
    # bar keeps the backtest correct for Japan/U.S. market-clock differences.
    b["forward_return_1d"]=(
        b.groupby("symbol")["close"].shift(-1)/b["close"]-1.0
    )
    if "stock_splits" in b.columns:
        current_split=b["stock_splits"].fillna(0).ne(0)
        next_split=b.groupby("symbol")["stock_splits"].shift(-1).fillna(0).ne(0)
        b.loc[current_split|next_split,"forward_return_1d"]=np.nan
    outcome=b[
        ["symbol","session_date","forward_return_1d","available_at"]
    ].rename(columns={"available_at":"outcome_available_at"})

    m=p.merge(
        outcome,
        on=["symbol","session_date"],
        how="inner",
        suffixes=("_prediction","_outcome"),
    )
    if "prediction_time" in m:
        m["prediction_time"]=pd.to_datetime(
            m["prediction_time"],utc=True,errors="coerce"
        )
        m["outcome_available_at"]=pd.to_datetime(
            m["outcome_available_at"],utc=True,errors="coerce"
        )
        m=m[
            m["outcome_available_at"].gt(m["prediction_time"])
        ].copy()

    rows=[]
    for date,g in m.groupby("prediction_date"):
        g=g.dropna(
            subset=["expected_return_1d","forward_return_1d"]
        )
        if len(g)<20:
            continue
        q=g["expected_return_1d"].rank(pct=True)
        long=g.loc[
            q>=1-top_quantile,"forward_return_1d"
        ].mean()
        short=g.loc[
            q<=top_quantile,"forward_return_1d"
        ].mean()
        spread=long-short-(2*cost_bps/10000)
        rows.append({
            "date":str(date),
            "n":len(g),
            "long_return":float(long),
            "short_return":float(short),
            "long_short_return":float(spread),
        })

    if not rows:
        return {
            "status":"DEFERRED",
            "rows":0,
            "reason":"no_valid_outcomes",
        }

    r=pd.DataFrame(rows)
    equity=(1+r["long_short_return"].fillna(0)).cumprod()
    peak=equity.cummax()
    drawdown=equity/peak-1
    result={
        "status":"PASS",
        "rows":int(len(r)),
        "mean_daily_spread":float(r["long_short_return"].mean()),
        "vol_daily_spread":float(r["long_short_return"].std(ddof=0)),
        "cumulative_spread":float(equity.iloc[-1]-1),
        "max_drawdown":float(drawdown.min()),
        "cost_bps_per_side":float(cost_bps),
    }
    return result
