from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.data.market_context import write_market_context

OUT=Path("data/market_context.parquet")
RESULT=Path("data/research/market_context_quality.json")


def main():
    rows=write_market_context(str(OUT),period="5y")
    df=pd.read_parquet(OUT)
    families=sorted(df["family"].dropna().unique().tolist())
    required={"nikkei","topix","sp500","nasdaq","vix","usd_jpy"}
    missing=sorted(required-set(families))
    result={
        "status":"PASS" if not missing else "DEFERRED",
        "rows":int(rows),
        "families":families,
        "missing_families":missing,
    }
    RESULT.parent.mkdir(parents=True,exist_ok=True)
    RESULT.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if result["status"]!="PASS":
        raise SystemExit("DEFERRED: incomplete market context")


if __name__=="__main__":
    main()
