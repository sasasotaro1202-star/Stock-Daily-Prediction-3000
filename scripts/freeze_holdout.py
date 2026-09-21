from __future__ import annotations
import argparse,json,os
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cutoff-date",required=True); args=ap.parse_args()
    cfg=Path("config/frozen_holdout.json")
    if cfg.exists(): raise SystemExit("FAIL: frozen holdout is already locked")
    metrics=Path("data/research/latest_metrics.json")
    prices=Path("data/prices")
    if not metrics.exists() or not prices.exists(): raise SystemExit("DEFERRED: OOS metrics and price data are required")
    payload=json.loads(metrics.read_text(encoding="utf-8"))
    selected=payload.get("selected_model")
    if not selected: raise SystemExit("FAIL: no OOS-selected model to freeze")
    df=pd.read_parquet(prices)
    dates=sorted(pd.to_datetime(df["session_date"]).dt.date.unique())
    cutoff=pd.Timestamp(args.cutoff_date).date()
    if cutoff not in dates: raise SystemExit(f"FAIL: cutoff date not in dataset: {cutoff}")
    idx=dates.index(cutoff); holdout_dates=dates[idx+1:]
    if idx+1<252 or len(holdout_dates)<21: raise SystemExit("FAIL: require >=252 pre-cutoff and >=21 holdout sessions")
    lock={"cutoff_date":str(cutoff),"holdout_start":str(holdout_dates[0]),"holdout_end":str(holdout_dates[-1]),
          "frozen_at":datetime.now(timezone.utc).isoformat(),"git_sha":os.getenv("GITHUB_SHA"),
          "selected_model":selected,"regime_selected_models":payload.get("regime_selected_models",{}),
          "status":"FROZEN","selection_source":"chronological OOS only"}
    cfg.parent.mkdir(parents=True,exist_ok=True); cfg.write_text(json.dumps(lock,indent=2),encoding="utf-8")
    print(json.dumps(lock,indent=2))

if __name__=="__main__": main()
