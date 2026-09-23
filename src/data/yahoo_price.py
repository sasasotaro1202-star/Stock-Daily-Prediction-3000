from __future__ import annotations

import os
import time
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


def yahoo_symbol(symbol: str, asset_class: str) -> str:
    if asset_class.startswith("jp_"):
        return f"{symbol}.T"
    # Yahoo uses hyphens for share-class tickers such as BRK.B -> BRK-B.
    return symbol.replace(".", "-")


def available_at_for(session_date, asset_class: str) -> pd.Timestamp:
    if asset_class.startswith("jp_"):
        # Conservative availability: 30 minutes after the regular close.
        dt = datetime.combine(
            session_date,
            time(16, 0),
            tzinfo=ZoneInfo("Asia/Tokyo"),
        )
    else:
        # Conservative availability: 30 minutes after the U.S. regular close.
        dt = datetime.combine(
            session_date,
            time(16, 30),
            tzinfo=ZoneInfo("America/New_York"),
        )
    return pd.Timestamp(dt).tz_convert("UTC")


def download_batch(
    records: list[dict],
    period: str = "5y",
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    symbols = [
        yahoo_symbol(r["symbol"], r["asset_class"])
        for r in records
    ]
    mapping = {
        yahoo_symbol(r["symbol"], r["asset_class"]): r
        for r in records
    }

    raw = yf.download(
        symbols,
        period=period,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
        actions=True,
    )
    retrieved_at = pd.Timestamp.now(tz="UTC")
    retrieval_run_id = os.getenv("GITHUB_RUN_ID")
    frames = []
    retrieval_run_id = os.getenv("GITHUB_RUN_ID")

    def normalize(part: pd.DataFrame, rec: dict, provider_symbol: str):
        part = part.reset_index().rename(columns=str.lower)
        if "date" not in part:
            return
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(part.columns):
            return
        # Adj Close is retained for split/dividend-robust feature/target work.
        if "adj close" not in part.columns:
            part["adj close"]=part["close"]

        part["symbol"] = rec["symbol"]
        part["asset_class"] = rec["asset_class"]
        part["session_date"] = pd.to_datetime(
            part["date"], errors="coerce"
        ).dt.date
        part["available_at"] = part["session_date"].map(
            lambda d: available_at_for(d, rec["asset_class"])
        )
        part["retrieved_at"] = retrieved_at
        part["available_at_method"] = "conservative_post_close_inferred"
        part["source"] = "yfinance"
        part["provider_symbol"] = provider_symbol
        part["retrieval_run_id"] = retrieval_run_id

        for optional in ("dividends","stock splits","capital gains"):
            if optional not in part.columns:
                part[optional]=0.0
        columns = [
            "symbol",
            "asset_class",
            "session_date",
            "available_at",
            "retrieved_at",
            "available_at_method",
            "source",
            "provider_symbol",
            "retrieval_run_id",
            "open",
            "high",
            "low",
            "close",
            "adj close",
            "dividends",
            "stock splits",
            "capital gains",
            "volume",
        ]
        part = part[columns].rename(
            columns={
                "adj close":"adj_close",
                "dividends":"dividends",
                "stock splits":"stock_splits",
                "capital gains":"capital_gains",
            }
        )
        for col in [
            "open","high","low","close","adj_close",
            "dividends","stock_splits","capital_gains","volume"
        ]:
            part[col] = pd.to_numeric(
                part[col], errors="coerce"
            )
        part = part.dropna(
            subset=["open", "high", "low", "close", "adj_close"]
        )
        part["volume"] = part["volume"].fillna(0)
        frames.append(part)

    if isinstance(raw.columns, pd.MultiIndex):
        for ysym in symbols:
            if ysym not in raw.columns.get_level_values(0):
                continue
            normalize(raw[ysym], mapping[ysym], ysym)
    else:
        normalize(raw, records[0], yahoo_symbol(
            records[0]["symbol"],
            records[0]["asset_class"],
        ))

    # Multi-ticker yfinance calls can silently omit a subset of valid tickers.
    # Recover those symbols individually before returning a partial batch.
    observed = {
        (str(frame["asset_class"].iloc[0]), str(frame["symbol"].iloc[0]))
        for frame in frames
        if not frame.empty and {"asset_class", "symbol"}.issubset(frame.columns)
    }
    missing_records = [
        rec for rec in records
        if (str(rec["asset_class"]), str(rec["symbol"])) not in observed
    ]
    for rec in missing_records:
        key = (str(rec["asset_class"]), str(rec["symbol"]))
        provider_symbol = yahoo_symbol(rec["symbol"], rec["asset_class"])
        recovered = False
        for attempt in range(3):
            try:
                single = yf.Ticker(provider_symbol).history(
                    period=period,
                    auto_adjust=False,
                    actions=True,
                )
            except Exception as exc:
                print(
                    f"price-single-retry provider_symbol={provider_symbol} "
                    f"attempt={attempt + 1}/3 error={type(exc).__name__}"
                )
                single = pd.DataFrame()
            if isinstance(single, pd.DataFrame) and not single.empty:
                before = len(frames)
                if isinstance(single.columns, pd.MultiIndex):
                    levels = single.columns.get_level_values(0)
                    if provider_symbol in levels:
                        single = single[provider_symbol]
                    elif len(set(levels)) == 1:
                        single = single.droplevel(0, axis=1)
                normalize(single, rec, provider_symbol)
                if len(frames) > before and not frames[-1].empty:
                    observed.add(key)
                    recovered = True
                    print(
                        f"price-single-recovery provider_symbol={provider_symbol} "
                        f"rows={len(frames[-1])}"
                    )
                    break
            if attempt < 2:
                time.sleep(2 ** attempt)
        if not recovered:
            print(
                f"price-single-deferred symbol={rec['symbol']} "
                f"asset_class={rec['asset_class']} provider_symbol={provider_symbol}"
            )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def upsert_batch_parquet(new_data: pd.DataFrame, path: str) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    old = (
        pd.read_parquet(path)
        if os.path.exists(path)
        else pd.DataFrame()
    )
    dedup_keys=(
        ["asset_class","symbol","session_date"]
        if "asset_class" in new_data.columns or "asset_class" in old.columns
        else ["symbol","session_date"]
    )
    combined = (
        pd.concat([old, new_data], ignore_index=True)
        .drop_duplicates(
            dedup_keys,
            keep="last",
        )
    )
    sort_keys=(
        ["asset_class","symbol","session_date"]
        if "asset_class" in combined.columns
        else ["symbol","session_date"]
    )
    combined = combined.sort_values(sort_keys)
    combined.to_parquet(path, index=False)
    return len(combined)
