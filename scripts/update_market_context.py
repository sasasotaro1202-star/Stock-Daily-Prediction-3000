from __future__ import annotations

import json
from pathlib import Path

from src.data.market_context import write_market_context

OUT=Path("data/market_context.parquet")
RESULT=Path("data/research/market_context_quality.json")


def main():
    rows=write_market_context(str(OUT),period="5y")
    df=__import__("pandas").read_parquet(OUT)
    families=sorted(df["family"].dropna().unique().tolist())
    result={
        "status":"PASS" if len(families)>=5 else "DEFERRED",
        "rows":int(rows),
        "families":families,
    }
    RESULT.parent.mkdir(parents=True,exist_ok=True)
    RESULT.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if result["status"]!="PASS":
        raise SystemExit("DEFERRED: incomplete market context")


if __name__=="__main__":
    main()
