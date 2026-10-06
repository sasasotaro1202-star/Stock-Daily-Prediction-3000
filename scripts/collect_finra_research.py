from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from src.data.finra_api import collect_finra_short_sale
from src.research.finra_features import FINRA_FEATURE_COLUMNS

PRICE=Path("data/prices/canonical.parquet")
OUT=Path("data/research/finra_short_sale.parquet")
META=Path("data/research/finra_short_sale.json")

def main() -> None:
    if not PRICE.exists(): raise SystemExit("DEFERRED: canonical price dataset missing")
    prices=pd.read_parquet(PRICE,columns=["session_date","symbol","asset_class"])
    us=prices[prices.asset_class.astype(str).eq("us_stock")]
    if us.empty: raise SystemExit("DEFERRED: no US stock symbols in price data")
    end=pd.to_datetime(us.session_date,errors="coerce").max().date()
    start=end-timedelta(days=365)
    symbols=sorted(us.symbol.astype(str).str.upper().unique())
    frame=collect_finra_short_sale(start,end,symbols)
    if frame.empty: raise SystemExit("DEFERRED: FINRA returned no rows")
    OUT.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(OUT,index=False)
    payload={"status":"OOS_READY","source":"FINRA Reg SHO Daily Short Sale Volume","research_only":True,
             "production_changed":False,"rows":int(len(frame)),"symbols":int(frame.symbol.nunique()),
             "start_date":str(frame.session_date.min()),"end_date":str(frame.session_date.max()),
             "feature_count":len(FINRA_FEATURE_COLUMNS),
             "pit_policy":{"fail_closed":True,"available_at_method":"next_calendar_day_1200_jst_conservative"}}
    META.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
