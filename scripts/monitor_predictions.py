from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.metrics import classification_metrics

PRED_DIR=Path("data/predictions")
BARS=Path("data/prices")
OUT=Path("data/research/monitor_latest.json")


def load_predictions() -> pd.DataFrame:
    files=sorted(glob.glob(str(PRED_DIR/"prediction_*.parquet")))
    if not files:
        return pd.DataFrame()
    frames=[pd.read_parquet(p) for p in files]
    out=pd.concat(frames,ignore_index=True)
    out["prediction_time"]=pd.to_datetime(
        out["prediction_time"],utc=True,errors="coerce"
    )
    out=out.sort_values("prediction_time").drop_duplicates(
        ["symbol","session_date","prediction_time"],
        keep="last",
    )
    return out


def main():
    pred=load_predictions()
    if pred.empty or not BARS.exists():
        payload={
            "status":"NO_BASELINE",
            "evaluated":0,
            "reason":"no timestamped production predictions or price history",
        }
    else:
        bars=pd.read_parquet(BARS).sort_values(
            ["symbol","session_date"]
        ).copy()
        bars["session_date"]=pd.to_datetime(
            bars["session_date"],errors="coerce"
        ).dt.date
        series_keys=["asset_class","symbol"] if "asset_class" in bars.columns else ["symbol"]
        bars["forward_return_1d"]=(
            bars.groupby(series_keys)["close"].shift(-1)
            /bars["close"]-1.0
        )
        if "stock_splits" in bars.columns:
            current_split=bars["stock_splits"].fillna(0).ne(0)
            next_split=bars.groupby(series_keys)["stock_splits"].shift(-1).fillna(0).ne(0)
            bars.loc[current_split|next_split,"forward_return_1d"]=np.nan
        bars["outcome_available_at"]=pd.to_datetime(
            bars["available_at"],utc=True,errors="coerce"
        )
        outcome=bars[
            series_keys
            + [
                "session_date",
                "forward_return_1d",
                "outcome_available_at",
            ]
        ]
        m=pred.merge(
            outcome,
            on=series_keys+["session_date"],
            how="inner",
        )
        m["prediction_time"]=pd.to_datetime(
            m["prediction_time"],utc=True,errors="coerce"
        )
        m=m[
            m["outcome_available_at"].gt(m["prediction_time"])
        ].dropna(
            subset=[
                "forward_return_1d",
                "p_up_1d",
                "expected_return_1d",
            ]
        )

        n=len(m)
        if n<250:
            payload={
                "status":"WARMUP",
                "evaluated":int(n),
                "reason":"insufficient completed outcomes",
            }
        else:
            def evaluate(frame:pd.DataFrame)->dict:
                y=(frame["forward_return_1d"]>0).astype(int).to_numpy()
                p=np.clip(
                    frame["p_up_1d"].to_numpy(dtype=float),
                    1e-6,
                    1-1e-6,
                )
                metrics=classification_metrics(y,p)
                base=classification_metrics(
                    y,np.full(len(frame),float(y.mean()))
                )
                errors=(
                    frame["expected_return_1d"]
                    -frame["forward_return_1d"]
                ).to_numpy(dtype=float)
                return {
                    "rows":int(len(frame)),
                    "metrics":metrics,
                    "baseline":base,
                    "beats_baseline":bool(
                        metrics["logloss"]<base["logloss"]
                    ),
                    "return_mae":float(np.mean(np.abs(errors))),
                    "return_rmse":float(np.mean(errors**2)**0.5),
                }

            latest_dates=sorted(m["session_date"].unique())[-20:]
            recent=m[m["session_date"].isin(latest_dates)].copy()
            overall=evaluate(m)
            recent_eval=evaluate(recent)
            valid_recent=(
                recent_eval["rows"]>=250
                and recent_eval["beats_baseline"]
                and recent_eval["metrics"]["ece"]<=0.25
            )
            valid_overall=(
                overall["beats_baseline"]
                and overall["metrics"]["ece"]<=0.25
            )
            payload={
                "status":"PASS"
                if valid_recent and valid_overall
                else "FAIL",
                "evaluated":int(n),
                "overall":overall,
                "recent_20_sessions":recent_eval,
                "latest_outcome_date":str(max(m["session_date"])),
                "route_counts":(
                    m["model_id"].value_counts(dropna=False).to_dict()
                    if "model_id" in m.columns else {}
                ),
                "scope_counts":(
                    m["training_scope"].value_counts(dropna=False).to_dict()
                    if "training_scope" in m.columns else {}
                ),
                "version_counts":(
                    m["model_version"].value_counts(dropna=False).to_dict()
                    if "model_version" in m.columns else {}
                ),
            }

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(
        json.dumps(payload,indent=2,default=str),
        encoding="utf-8",
    )
    print(json.dumps(payload,indent=2,default=str))




if __name__=="__main__":
    main()
