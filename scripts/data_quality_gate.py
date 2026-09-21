from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

REQUIRED={"symbol","asset_class","session_date","available_at","open","high","low","close","volume"}

def main():
    root=Path("data/prices")
    if not root.exists(): raise SystemExit("DEFERRED: price directory missing")
    files=sorted(root.glob("*.parquet"))
    if not files: raise SystemExit("DEFERRED: no price partitions")
    frames=[pd.read_parquet(p) for p in files]
    df=pd.concat(frames,ignore_index=True)
    reasons=[]
    missing=REQUIRED-set(df.columns)
    if missing: reasons.append(f"missing_columns:{sorted(missing)}")
    if df.empty: reasons.append("empty_dataset")
    if not df.empty:
        dup=int(df.duplicated(["symbol","session_date"]).sum())
        bad_ohlc=int(((df["high"]<df["low"])|(df["high"]<df["open"])|(df["high"]<df["close"])|
                      (df["low"]>df["open"])|(df["low"]>df["close"])|
                      (df[["open","high","low","close"]]<=0).any(axis=1)).sum())
        neg_vol=int((df["volume"]<0).sum())
        invalid_avail=int(pd.to_datetime(df["available_at"],utc=True,errors="coerce").isna().sum())
        reasons += ([f"duplicates:{dup}"] if dup else [])
        reasons += ([f"bad_ohlc:{bad_ohlc}"] if bad_ohlc else [])
        reasons += ([f"negative_volume:{neg_vol}"] if neg_vol else [])
        reasons += ([f"invalid_available_at:{invalid_avail}"] if invalid_avail else [])
    result={"status":"PASS" if not reasons else "DEFERRED","rows":int(len(df)),"files":len(files),"reasons":reasons}
    Path("data/research").mkdir(parents=True,exist_ok=True)
    Path("data/research/data_quality.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if reasons and any(x.startswith(("missing_columns","bad_ohlc","duplicates")) for x in reasons):
        raise SystemExit("FAIL: critical price quality issue")

if __name__=="__main__": main()
