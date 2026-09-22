from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.market_context import download_market_context

OUT=Path("data/market_context.parquet")
RESULT=Path("data/research/market_context_quality.json")

REQUIRED={
    "nikkei","topix","sp500","nasdaq","vix","usd_jpy",
    "us10y","dxy","gold","oil","hyg",
}
MAX_STALENESS_DAYS=10


def main():
    period="5y" if not OUT.exists() else "10d"
    fresh=download_market_context(period=period)
    if fresh.empty:
        raise SystemExit("DEFERRED: market context returned no rows")

    if OUT.exists() and period=="10d":
        old=pd.read_parquet(OUT)
        df=pd.concat([old,fresh],ignore_index=True)
    else:
        df=fresh

    df["session_date"]=pd.to_datetime(
        df["session_date"],errors="coerce"
    ).dt.date
    df["available_at"]=pd.to_datetime(
        df["available_at"],utc=True,errors="coerce"
    )
    df=(
        df.dropna(subset=["session_date","available_at"])
        .drop_duplicates(["family","session_date"],keep="last")
        .sort_values(["family","session_date"])
        .reset_index(drop=True)
    )

    OUT.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(OUT,index=False)

    families=sorted(df["family"].dropna().unique().tolist())
    missing=sorted(REQUIRED-set(families))
    stale=[]
    now_jst=pd.Timestamp.now(tz="Asia/Tokyo").date()
    stale_cutoff=(pd.Timestamp(now_jst)-pd.Timedelta(days=MAX_STALENESS_DAYS)).date()
    latest_by_family=(
        df.groupby("family")["session_date"].max().to_dict()
        if "family" in df.columns else {}
    )
    for family in sorted(REQUIRED):
        latest=latest_by_family.get(family)
        if latest is None:
            continue
        if latest < stale_cutoff:
            stale.append(family)
    result={
        "status":"PASS" if not missing and not stale else "DEFERRED",
        "rows":int(len(df)),
        "families":families,
        "missing_families":missing,
        "stale_families":sorted(stale),
        "latest_session_by_family":{k:str(v) for k,v in latest_by_family.items()},
        "max_staleness_days":MAX_STALENESS_DAYS,
        "update_mode":"initial_full" if period=="5y" else "incremental_10d",
    }
    RESULT.parent.mkdir(parents=True,exist_ok=True)
    RESULT.write_text(
        json.dumps(result,indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result,indent=2))

    if result["status"]!="PASS":
        raise SystemExit("DEFERRED: incomplete or stale market context")


if __name__=="__main__":
    main()
