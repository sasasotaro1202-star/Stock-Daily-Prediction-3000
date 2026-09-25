from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import time

import pandas as pd

from src.data.yahoo_price import download_batch, upsert_batch_parquet

U=Path("data/universe/latest.json")
ROOT=Path("data/prices")
SHARD_INDEX=int(os.getenv("PRICE_SHARD_INDEX","0"))
SHARD_COUNT=max(1,int(os.getenv("PRICE_SHARD_COUNT","1")))
MAX_WORKERS=max(1,int(os.getenv("PRICE_MAX_WORKERS","2")))
ASSET_SCOPE={
    x.strip()
    for x in os.getenv("PRICE_ASSET_CLASSES","").split(",")
    if x.strip()
}


def prune_price_batch_to_current_universe(
    path: Path,
    current_keys: set[tuple[str,str]],
) -> int:
    """Remove stale symbols left behind by dynamic-universe reordering.

    Batch ids are positional. When the universe reorders, a symbol can move to
    a different batch while its historical rows remain in the old batch. Keep
    only symbols that belong to the current batch before canonicalization.
    """
    if not path.exists():
        return 0
    stored=pd.read_parquet(path)
    required={"asset_class","symbol"}
    if stored.empty:
        return 0
    if not required.issubset(stored.columns):
        raise ValueError(
            f"stored price batch missing columns: {sorted(required-set(stored.columns))}"
        )
    keep=pd.Series(
        list(zip(
            stored["asset_class"].astype(str),
            stored["symbol"].astype(str),
        )),
        index=stored.index,
    ).isin(current_keys)
    removed=int((~keep).sum())
    if removed:
        stored.loc[keep].to_parquet(path,index=False)
    return removed




def fetch_resilient(
    records: list[dict],
    period: str,
    *,
    depth: int = 0,
    downloader=None,
    sleep_fn=time.sleep,
) -> pd.DataFrame:
    """Bounded retry + split fallback for poisoned bulk downloads."""
    if not records:
        return pd.DataFrame()

    fetcher = downloader or download_batch
    attempts = 3 if depth == 0 else 1
    for attempt in range(attempts):
        try:
            data = fetcher(records, period=period)
        except Exception as exc:
            print(
                f"price-download-retry attempt={attempt + 1}/{attempts} "
                f"period={period} rows={len(records)} depth={depth} "
                f"error={type(exc).__name__}"
            )
            data = pd.DataFrame()
        if not data.empty:
            if {"asset_class", "symbol"}.issubset(data.columns):
                observed_keys = {
                    (str(asset_class), str(symbol))
                    for asset_class, symbol in zip(
                        data["asset_class"].astype(str),
                        data["symbol"].astype(str),
                    )
                }
                missing_records = [
                    rec
                    for rec in records
                    if (str(rec["asset_class"]), str(rec["symbol"])) not in observed_keys
                ]
                if missing_records and depth < 3 and len(missing_records) > 5:
                    recovered = fetch_resilient(
                        missing_records,
                        period,
                        depth=depth + 1,
                        downloader=fetcher,
                        sleep_fn=sleep_fn,
                    )
                    if not recovered.empty:
                        print(
                            f"price-download-partial-recovery period={period} "
                            f"requested={len(records)} "
                            f"returned={len(observed_keys)} "
                            f"missing={len(missing_records)} "
                            f"recovered_rows={len(recovered)} "
                            f"depth={depth}"
                        )
                        data = pd.concat([data, recovered], ignore_index=True)
            return data
        if attempt < attempts - 1:
            sleep_fn(2 ** attempt)

    if depth < 3 and len(records) > 5:
        midpoint = len(records) // 2
        left = fetch_resilient(
            records[:midpoint],
            period,
            depth=depth + 1,
            downloader=fetcher,
            sleep_fn=sleep_fn,
        )
        right = fetch_resilient(
            records[midpoint:],
            period,
            depth=depth + 1,
            downloader=fetcher,
            sleep_fn=sleep_fn,
        )
        pieces = [frame for frame in (left, right) if not frame.empty]
        if pieces:
            recovered_rows = sum(len(frame) for frame in pieces)
            print(
                f"price-download-split-recovery period={period} "
                f"rows={len(records)} depth={depth} "
                f"recovered_rows={recovered_rows}"
            )
            return pd.concat(pieces, ignore_index=True)

    print(
        f"price-download-deferred period={period} rows={len(records)} "
        f"depth={depth}"
    )
    return pd.DataFrame()

def one(i:int,batch:list[dict])->tuple[int,int,str,list[tuple[str,str]]]:
    path=ROOT/f"batch_{i:03d}.parquet"
    ROOT.mkdir(parents=True,exist_ok=True)

    old=pd.read_parquet(path) if path.exists() else pd.DataFrame()
    old_keys=(
        set(zip(old["asset_class"].astype(str),old["symbol"].astype(str)))
        if {"asset_class","symbol"}.issubset(old.columns)
        else set()
    )
    current_keys={
        (str(r["asset_class"]),str(r["symbol"])) for r in batch
    }

    # Dynamic PayPay universes can add/remove/reorder names. Existing keys
    # only need incremental pull; new product keys require full warm-up history.
    existing=[
        r for r in batch
        if (str(r["asset_class"]),str(r["symbol"])) in old_keys
    ]
    new=[
        r for r in batch
        if (str(r["asset_class"]),str(r["symbol"])) not in old_keys
    ]


    frames=[]
    if existing:
        data=fetch_resilient(existing,period="10d")
        if not data.empty:
            frames.append(data)
    if new:
        data=fetch_resilient(new,period="5y")
        if not data.empty:
            frames.append(data)

    if not frames:
        try:
            removed=prune_price_batch_to_current_universe(path,current_keys)
        except ValueError as exc:
            raise SystemExit(f"FAIL: invalid stored price batch {path}: {exc}") from exc
        if removed:
            print(
                f"price-batch-pruned={i:03d} stale_symbol_rows={removed} "
                f"reason=dynamic_universe_reorder"
            )
        return i,0,"DEFERRED",sorted(current_keys)

    data=pd.concat(frames,ignore_index=True)
    upsert_batch_parquet(data,str(path))
    try:
        removed=prune_price_batch_to_current_universe(path,current_keys)
    except ValueError as exc:
        raise SystemExit(f"FAIL: invalid stored price batch {path}: {exc}") from exc
    if removed:
        print(
            f"price-batch-pruned={i:03d} stale_symbol_rows={removed} "
            f"reason=dynamic_universe_reorder"
        )
    stored=pd.read_parquet(path)
    rows=len(stored)
    stored_keys=set(
        zip(
            stored["asset_class"].astype(str),
            stored["symbol"].astype(str),
        )
    )
    missing_keys=sorted(current_keys-stored_keys)
    status="PASS" if not missing_keys else "DEFERRED"
    return i,rows,status,missing_keys


if __name__=="__main__":
    if not U.exists():
        raise SystemExit("FAIL: universe snapshot missing")

    records=json.loads(
        U.read_text(encoding="utf-8")
    )["records"]
    if ASSET_SCOPE:
        records=[
            row for row in records
            if str(row.get("asset_class","")) in ASSET_SCOPE
        ]
        if not records:
            raise SystemExit(
                f"DEFERRED: no universe records match PRICE_ASSET_CLASSES={sorted(ASSET_SCOPE)}"
            )

    all_batches=[records[i:i+75] for i in range(0,len(records),75)]
    selected=[
        (i,b) for i,b in enumerate(all_batches)
        if i % SHARD_COUNT == SHARD_INDEX
    ]

    completed=0
    deferred=0
    unresolved: list[dict] = []
    report_path=ROOT/f"price_deferred_shard_{SHARD_INDEX}.json"
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures=[pool.submit(one,i,b) for i,b in selected]
        for fut in as_completed(futures):
            i,rows,status,missing_keys=fut.result()
            for asset_class, symbol in missing_keys:
                unresolved.append({
                    "asset_class": asset_class,
                    "symbol": symbol,
                    "batch": i,
                    "reason": "no_price_rows_after_bounded_recovery",
                    "retrieval_run_id": os.getenv("GITHUB_RUN_ID"),
                })
            print(
                f"price-batch={i:03d} "
                f"shard={SHARD_INDEX}/{SHARD_COUNT} "
                f"status={status} rows={rows}"
            )
            if status=="PASS":
                completed+=1
            else:
                deferred+=1

    report_path.write_text(
        json.dumps(
            {
                "status": "PASS" if not unresolved else "DEFERRED",
                "shard": SHARD_INDEX,
                "shard_count": SHARD_COUNT,
                "retrieval_run_id": os.getenv("GITHUB_RUN_ID"),
                "deferred": sorted(
                    unresolved,
                    key=lambda x: (x["asset_class"], x["symbol"]),
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"price-update: PASS shard={SHARD_INDEX}/{SHARD_COUNT} "
        f"completed={completed}/{len(selected)} deferred={deferred}"
    )
