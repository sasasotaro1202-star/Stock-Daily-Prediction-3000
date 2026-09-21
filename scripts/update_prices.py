from pathlib import Path
import json
from src.data.yahoo_price import download_batch,upsert_parquet

U=Path("data/universe/latest.json"); OUT="data/prices/daily.parquet"

if __name__=="__main__":
    if not U.exists(): raise SystemExit("FAIL: universe snapshot missing")
    records=json.loads(U.read_text(encoding="utf-8"))["records"]
    total=0; returned_batches=0
    for start in range(0,len(records),75):
        batch=records[start:start+75]
        data=download_batch(batch)
        if data.empty:
            print(f"DEFERRED batch={start}:{start+len(batch)}")
            continue
        returned_batches+=1
        total=upsert_parquet(data,OUT)
        print(f"price-batch {start}:{start+len(batch)} rows={len(data)} total={total}")
    if total==0: raise SystemExit("DEFERRED: no price data returned")
    print(f"price-update: PASS batches={returned_batches} rows={total}")
