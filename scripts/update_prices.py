from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,os
from src.data.yahoo_price import download_batch,upsert_batch_parquet

U=Path("data/universe/latest.json"); ROOT=Path("data/prices")
SHARD_INDEX=int(os.getenv("PRICE_SHARD_INDEX","0"))
SHARD_COUNT=max(1,int(os.getenv("PRICE_SHARD_COUNT","1")))
MAX_WORKERS=max(1,int(os.getenv("PRICE_MAX_WORKERS","2")))

def one(i:int,batch:list[dict])->tuple[int,int,str]:
    path=ROOT/f"batch_{i:03d}.parquet"
    period="10d" if path.exists() else "5y"
    data=download_batch(batch,period=period)
    if data.empty:return i,0,"DEFERRED"
    return i,upsert_batch_parquet(data,str(path)),"PASS"

if __name__=="__main__":
    if not U.exists():raise SystemExit("FAIL: universe snapshot missing")
    records=json.loads(U.read_text(encoding="utf-8"))["records"]
    all_batches=[records[i:i+75] for i in range(0,len(records),75)]
    selected=[(i,b) for i,b in enumerate(all_batches) if i % SHARD_COUNT == SHARD_INDEX]
    completed=0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures=[pool.submit(one,i,b) for i,b in selected]
        for fut in as_completed(futures):
            i,rows,status=fut.result()
            print(f"price-batch={i:03d} shard={SHARD_INDEX}/{SHARD_COUNT} status={status} rows={rows}")
            if status=="PASS":completed+=1
    if selected and completed==0:raise SystemExit(f"DEFERRED: shard {SHARD_INDEX} returned no price batches")
    print(f"price-update: PASS shard={SHARD_INDEX}/{SHARD_COUNT} completed={completed}/{len(selected)}")
