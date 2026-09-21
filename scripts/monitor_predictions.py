from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from src.research.metrics import classification_metrics

PRED=Path("data/predictions/latest.parquet")
BARS=Path("data/prices")
OUT=Path("data/research/monitoring_latest.json")

def main():
    if not PRED.exists() or not BARS.exists():
        payload={"status":"NO_BASELINE","evaluated":0,"reason":"no prior production predictions"}
    else:
        pred=pd.read_parquet(PRED).copy()
        bars=pd.read_parquet(BARS).sort_values(["symbol","session_date"]).copy()
        bars["forward_return_1d"]=bars.groupby("symbol")["close"].shift(-1)/bars["close"]-1.0
        bars["outcome_available_at"]=pd.to_datetime(bars["available_at"],utc=True,errors="coerce")
        m=pred.merge(bars[["symbol","session_date","forward_return_1d","outcome_available_at"]],on=["symbol","session_date"],how="inner")
        m["prediction_time"]=pd.to_datetime(m["prediction_time"],utc=True,errors="coerce")
        m=m[m["outcome_available_at"]>m["prediction_time"]].dropna(subset=["forward_return_1d","p_up_1d"])
        n=len(m)
        if n<250:
            payload={"status":"WARMUP","evaluated":int(n),"reason":"insufficient completed outcomes"}
        else:
            y=(m["forward_return_1d"]>0).astype(int).to_numpy()
            p=np.clip(m["p_up_1d"].to_numpy(dtype=float),1e-6,1-1e-6)
            mm=classification_metrics(y,p)
            base=float(y.mean())
            baseline=classification_metrics(y,np.full(n,base))
            cfg=Path("config/pipeline.yml").read_text(encoding="utf-8")
            hit=re.search(r"live_max_ece:\s*([0-9.]+)",cfg)
            max_ece=float(hit.group(1)) if hit else 0.25
            payload={"status":"PASS" if mm["logloss"]<baseline["logloss"] and mm["ece"]<=max_ece else "FAIL",
                     "evaluated":int(n),"metrics":mm,"baseline":baseline,"beats_baseline":bool(mm["logloss"]<baseline["logloss"]),"max_ece":max_ece}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))

if __name__=="__main__":
    main()
