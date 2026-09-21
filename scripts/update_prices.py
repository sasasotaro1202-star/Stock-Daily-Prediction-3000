from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
from src.data.yahoo_price import download_batch,upsert_batch_parquet

U=Path("data/universe/latest.json"); ROOT=Path("data/prices")

def one(i:int,batch:list[dict])->tuple[int,int,str]:
    path=ROOT/f"batch_{i:03d}.parquet"
    period="10d" if path.exists() else "5y"
    data=download_batch(batch,period=period)
    if data.empty:return i,0,"DEFERRED"
    rows=upsert_batch_parquet(data,str(path))
    return i,rows,"PASS"

if __name__=="__main__":
    if not U.exists():raise SystemExit("FAIL: universe snapshot missing")
    records=json.loads(U.read_text(encoding="utf-8"))["records"]
    batches=[records[i:i+75] for i in range(0,len(records),75)]
    completed=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(one,i,b):i for i,b in enumerate(batches)}
        for fut in as_completed(futures):
            i,rows,status=fut.result()
            print(f"price-batch={i:03d} status={status} rows={rows}")
            if status=="PASS":completed+=1
    if completed==0:raise SystemExit("DEFERRED: no price batch returned")
    print(f"price-update: PASS batches={completed}/{len(batches)}")
