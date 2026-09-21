from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json

import pandas as pd

REQUIRED={
    "symbol","asset_class","session_date","available_at",
    "open","high","low","close","volume",
}
UNIVERSE=Path("data/universe/latest.json")


def main():
    root=Path("data/prices")
    if not root.exists():
        raise SystemExit("DEFERRED: price directory missing")
    files=sorted(root.glob("*.parquet"))
    if not files:
        raise SystemExit("DEFERRED: no price partitions")
    if not UNIVERSE.exists():
        raise SystemExit("DEFERRED: current PayPay universe snapshot missing")

    frames=[pd.read_parquet(p) for p in files]
    df=pd.concat(frames,ignore_index=True)
    reasons=[]
    missing=REQUIRED-set(df.columns)
    if missing:
        reasons.append(f"missing_columns:{sorted(missing)}")

    if df.empty:
        reasons.append("empty_dataset")
    else:
        dup=int(df.duplicated(["symbol","session_date"]).sum())
        bad_ohlc=int(
            (
                (df["high"]<df["low"])
                | (df["high"]<df["open"])
                | (df["high"]<df["close"])
                | (df["low"]>df["open"])
                | (df["low"]>df["close"])
                | (df[["open","high","low","close"]]<=0).any(axis=1)
            ).sum()
        )
        neg_vol=int((df["volume"]<0).sum())
        invalid_avail=int(
            pd.to_datetime(df["available_at"],utc=True,errors="coerce").isna().sum()
        )
        reasons += [f"duplicates:{dup}"] if dup else []
        reasons += [f"bad_ohlc:{bad_ohlc}"] if bad_ohlc else []
        reasons += [f"negative_volume:{neg_vol}"] if neg_vol else []
        reasons += [f"invalid_available_at:{invalid_avail}"] if invalid_avail else []

        snap=json.loads(UNIVERSE.read_text(encoding="utf-8"))
        expected={
            (row["asset_class"],row["symbol"])
            for row in snap.get("records",[])
            if row.get("tradeable") is True
        }
        observed={
            (row["asset_class"],row["symbol"])
            for row in df[["asset_class","symbol"]].drop_duplicates().to_dict("records")
        }
        missing_now=sorted(expected-observed)
        now=pd.Timestamp(datetime.now(ZoneInfo("Asia/Tokyo")))
        avail=pd.to_datetime(df["available_at"],utc=True,errors="coerce")
        session_dates=pd.to_datetime(df["session_date"],errors="coerce").dt.date
        current_mask=avail.le(now.tz_convert("UTC"))
        latest_by_symbol=(
            df.loc[current_mask & session_dates.notna(),["asset_class","symbol","session_date"]]
            .assign(session_date=session_dates[current_mask & session_dates.notna()])
            .groupby(["asset_class","symbol"])["session_date"].max()
        )
        stale_cutoff=(now-pd.Timedelta(days=10)).date()
        stale=[
            key for key,value in latest_by_symbol.items()
            if value < stale_cutoff
        ]
        missing_latest=sorted(expected-set(latest_by_symbol.index))

        if missing_now:
            reasons.append(f"universe_symbols_missing_from_price_history:{len(missing_now)}")
        if missing_latest:
            reasons.append(f"universe_symbols_without_current_pit_row:{len(missing_latest)}")
        if stale:
            reasons.append(f"universe_symbols_stale_over_10d:{len(stale)}")

    critical_prefixes=(
        "missing_columns","empty_dataset","duplicates","bad_ohlc",
        "universe_symbols_missing_from_price_history",
        "universe_symbols_without_current_pit_row",
        "universe_symbols_stale_over_10d",
        "negative_volume",
    )
    result={
        "status":"PASS" if not reasons else "DEFERRED",
        "rows":int(len(df)),
        "files":len(files),
        "reasons":reasons,
    }
    Path("data/research").mkdir(parents=True,exist_ok=True)
    Path("data/research/data_quality.json").write_text(
        json.dumps(result,indent=2),encoding="utf-8"
    )
    print(json.dumps(result,indent=2))
    if reasons and any(
        x.startswith(critical_prefixes) for x in reasons
    ):
        raise SystemExit("FAIL: critical price/universe quality issue")


if __name__=="__main__":
    main()
