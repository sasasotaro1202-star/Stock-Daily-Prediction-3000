from __future__ import annotations
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
import os,pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cutoff-date",required=True); args=ap.parse_args()
    path=Path("config/frozen_holdout.json")
    if path.exists(): raise SystemExit("FAIL: frozen holdout cutoff is already locked")
    prices=Path("data/prices")
    if not prices.exists(): raise SystemExit("DEFERRED: price data is required")
    df=pd.read_parquet(prices)
    dates=sorted(pd.to_datetime(df["session_date"]).dt.date.unique())
    cutoff=pd.Timestamp(args.cutoff_date).date()
    if cutoff not in dates: raise SystemExit(f"FAIL: cutoff date not in dataset: {cutoff}")
    idx=dates.index(cutoff); holdout_dates=dates[idx+1:]
    if idx+1<252 or len(holdout_dates)<21: raise SystemExit("FAIL: require >=252 pre-cutoff and >=21 holdout sessions")
    lock={"cutoff_date":str(cutoff),"holdout_start":str(holdout_dates[0]),"holdout_end":str(holdout_dates[-1]),
          "cutoff_frozen_at":datetime.now(timezone.utc).isoformat(),"cutoff_frozen_git_sha":os.getenv("GITHUB_SHA"),
          "status":"CUTOFF_FROZEN_PENDING_MODEL","selected_model":None,"regime_selected_models":{}}
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(lock,indent=2),encoding="utf-8")
    print(json.dumps(lock,indent=2))

if __name__=="__main__": main()
