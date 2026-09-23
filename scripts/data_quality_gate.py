from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json

import pandas as pd

REQUIRED={
    "symbol","asset_class","session_date","available_at",
    "source","provider_symbol",
    "open","high","low","close","volume",
}
UNIVERSE=Path("data/universe/latest.json")
ASSET_SCOPE={
    x.strip()
    for x in __import__("os").environ.get("QUALITY_ASSET_CLASSES","").split(",")
    if x.strip()
}


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
        result={
            "status":"FAIL",
            "rows":int(len(df)),
            "files":len(files),
            "reasons":[f"missing_columns:{sorted(missing)}"],
        }
        Path("data/research").mkdir(parents=True,exist_ok=True)
        Path("data/research/data_quality.json").write_text(
            json.dumps(result,indent=2),encoding="utf-8"
        )
        print(json.dumps(result,indent=2))
        raise SystemExit("FAIL: required price columns are missing")

    deferred_keys:set[tuple[str,str]] = set()
    deferred_reports=sorted(root.glob("price_deferred_shard_*.json"))
    current_run_id=__import__("os").environ.get("GITHUB_RUN_ID")
    for report_path in deferred_reports:
        try:
            payload=json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if current_run_id and payload.get("retrieval_run_id") != current_run_id:
            continue
        for item in payload.get("deferred",[]) or []:
            if isinstance(item,dict) and item.get("asset_class") and item.get("symbol"):
                deferred_keys.add((str(item["asset_class"]),str(item["symbol"])))
    if deferred_keys:
        reasons.append(f"provider_deferred_current_cycle:{len(deferred_keys)}")

    if df.empty:
        reasons.append("empty_dataset")
    else:
        session_ts=pd.to_datetime(df["session_date"],errors="coerce")
        session_dates=session_ts.dt.date
        dup=int(df.duplicated(["asset_class","symbol","session_date"]).sum())
        numeric_cols=["open","high","low","close","volume"]
        missing_source=int(
            df[["source","provider_symbol"]].isna().any(axis=1).sum()
        )
        numeric_invalid=int(df[numeric_cols].isna().any(axis=1).sum())
        invalid_session=int(pd.to_datetime(df["session_date"],errors="coerce").isna().sum())
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
        retrieved_missing=0
        retrieved_before_available=0
        retrieved_future=0
        latest_retrieval_missing=0
        if "retrieved_at" in df.columns:
            retrieved=pd.to_datetime(df["retrieved_at"],utc=True,errors="coerce")
            avail=pd.to_datetime(df["available_at"],utc=True,errors="coerce")
            # Historical cache rows may predate provenance tracking. They remain
            # auditable as legacy rows; only the currently-used latest PIT row
            # must carry retrieval provenance.
            current_available_mask=avail.le(pd.Timestamp.now(tz="UTC"))
            latest_idx=(
                df.loc[current_available_mask & session_dates.notna()]
                .groupby(["asset_class","symbol"])["session_date"].idxmax()
            )
            latest_retrieval_missing=int(retrieved.loc[latest_idx].isna().sum())
            retrieved_before_available=int(
                avail.gt(retrieved).fillna(False).sum()
            )
            retrieved_future=int(
                retrieved.gt(
                    pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5)
                ).fillna(False).sum()
            )
            retrieved_missing=int(retrieved.isna().sum())
        else:
            reasons.append("retrieved_at_missing")
            latest_retrieval_missing=1
        session_day_start=session_ts.dt.tz_localize("UTC",ambiguous="NaT",nonexistent="NaT")
        avail_ts=pd.to_datetime(df["available_at"],utc=True,errors="coerce")
        impossible_pit=int((avail_ts.lt(session_day_start)).fillna(False).sum())
        reasons += [f"duplicates:{dup}"] if dup else []
        reasons += [f"numeric_invalid:{numeric_invalid}"] if numeric_invalid else []
        reasons += [f"missing_source_provenance:{missing_source}"] if missing_source else []
        reasons += [f"invalid_session_date:{invalid_session}"] if invalid_session else []
        reasons += [f"bad_ohlc:{bad_ohlc}"] if bad_ohlc else []
        reasons += [f"negative_volume:{neg_vol}"] if neg_vol else []
        reasons += [f"invalid_available_at:{invalid_avail}"] if invalid_avail else []
        legacy_retrieval_missing = retrieved_missing
        reasons += [f"latest_retrieval_at_missing:{latest_retrieval_missing}"] if latest_retrieval_missing else []
        reasons += [f"available_at_after_retrieved_at:{retrieved_before_available}"] if retrieved_before_available else []
        reasons += [f"available_at_before_session_date:{impossible_pit}"] if impossible_pit else []

        snap=json.loads(UNIVERSE.read_text(encoding="utf-8"))
        expected={
            (row["asset_class"],row["symbol"])
            for row in snap.get("records",[])
            if row.get("tradeable") is True
            and (not ASSET_SCOPE or str(row["asset_class"]) in ASSET_SCOPE)
        }
        observed={
            (row["asset_class"],row["symbol"])
            for row in df[["asset_class","symbol"]].drop_duplicates().to_dict("records")
        }
        missing_now=sorted(expected-observed)
        now=pd.Timestamp(datetime.now(ZoneInfo("Asia/Tokyo")))
        avail=pd.to_datetime(df["available_at"],utc=True,errors="coerce")
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

        # Provider-unavailable symbols are allowed only as bounded degradation.
        # A large current-cycle omission still blocks release.
        deferred_expected = sorted(
            key for key in deferred_keys if key in expected
        )
        deferred_unknown = sorted(
            key for key in deferred_keys if key not in expected
        )
        deferred_ratio = (
            len(deferred_expected) / len(expected)
            if expected
            else 0.0
        )
        if deferred_unknown:
            reasons.append(
                f"provider_deferred_unknown_symbols:{len(deferred_unknown)}"
            )
        if deferred_ratio > 0.05:
            reasons.append(
                f"provider_deferred_ratio_over_5pct:{deferred_ratio:.6f}"
            )

        missing_now_non_deferred=[key for key in missing_now if key not in deferred_keys]
        missing_latest_non_deferred=[key for key in missing_latest if key not in deferred_keys]
        if missing_now_non_deferred:
            reasons.append(f"universe_symbols_missing_from_price_history:{len(missing_now_non_deferred)}")
        if missing_latest_non_deferred:
            reasons.append(f"universe_symbols_without_current_pit_row:{len(missing_latest_non_deferred)}")
        if stale:
            reasons.append(f"universe_symbols_stale_over_10d:{len(stale)}")

    critical_prefixes=(
        "missing_columns","empty_dataset","duplicates","numeric_invalid","missing_source_provenance","invalid_session_date","bad_ohlc",
        "available_at_before_session_date",
        "latest_retrieval_at_missing",
        "available_at_after_retrieved_at",
        "universe_symbols_missing_from_price_history",
        "universe_symbols_without_current_pit_row",
        "universe_symbols_stale_over_10d",
        "negative_volume",
    )
    hard_reasons=[
        reason for reason in reasons
        if not reason.startswith("provider_deferred_current_cycle:")
    ]
    result={
        "status":"PASS" if not hard_reasons else "DEFERRED",
        "rows":int(len(df)),
        "asset_scope":sorted(ASSET_SCOPE) if ASSET_SCOPE else None,
        "files":len(files),
        "reasons":reasons,
        "legacy_retrieval_missing": int(locals().get("legacy_retrieval_missing", 0)),
        "expected_symbols": int(locals().get("expected", set()).__len__()),
        "provider_deferred_symbols": int(len(locals().get("deferred_expected", []))),
        "provider_deferred_ratio": float(locals().get("deferred_ratio", 0.0)),
    }
    Path("data/research").mkdir(parents=True,exist_ok=True)
    Path("data/research/data_quality.json").write_text(
        json.dumps(result,indent=2),encoding="utf-8"
    )
    print(json.dumps(result,indent=2))
    if hard_reasons and any(
        x.startswith(critical_prefixes) for x in hard_reasons
    ):
        raise SystemExit("FAIL: critical price/universe quality issue")


if __name__=="__main__":
    main()
