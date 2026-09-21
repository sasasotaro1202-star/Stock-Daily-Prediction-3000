from __future__ import annotations
import json
import os
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

def main():
    lock=Path("config/frozen_holdout.json")
    if lock.exists():
        print("frozen-holdout: already locked")
        return
    root=Path("data/prices")
    if not root.exists():
        raise SystemExit("DEFERRED: price data is required")
    df=pd.read_parquet(root)
    dates=sorted(pd.to_datetime(df["session_date"]).dt.date.unique())
    holdout_sessions=63
    if len(dates)<252+holdout_sessions+1:
        raise SystemExit(f"DEFERRED: insufficient sessions ({len(dates)})")
    cutoff=dates[-(holdout_sessions+1)]
    holdout=dates[dates.index(cutoff)+1:]
    payload={
        "cutoff_date":str(cutoff),
        "holdout_start":str(holdout[0]),
        "holdout_end":str(holdout[-1]),
        "cutoff_frozen_at":datetime.now(timezone.utc).isoformat(),
        "cutoff_frozen_git_sha":os.getenv("GITHUB_SHA"),
        "status":"CUTOFF_FROZEN_PENDING_MODEL",
        "selected_model":None,
        "regime_selected_models":{},
        "selection_source":"automatic latest-minus-63-session cutoff; holdout observations excluded from selection",
    }
    lock.parent.mkdir(parents=True,exist_ok=True)
    lock.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    main()
