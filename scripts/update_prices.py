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

    def fetch_resilient(records: list[dict], period: str) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
        last_empty = False
        for attempt in range(3):
            try:
                data = download_batch(records, period=period)
            except Exception as exc:
                print(
                    f"price-download-retry attempt={attempt + 1}/3 "
                    f"period={period} rows={len(records)} error={type(exc).__name__}"
                )
                data = pd.DataFrame()
            if not data.empty:
                return data
            last_empty = True
            if attempt < 2:
                time.sleep(2 ** attempt)
        if last_empty:
            print(
                f"price-download-deferred period={period} rows={len(records)}"
            )
        return pd.DataFrame()

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
