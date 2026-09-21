from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

from src.data.paypay_collector import build_snapshot

def main():
    latest = Path("data/universe/latest.json")
    snap = build_snapshot(str(latest))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    archive = Path("data/universe/snapshots") / f"{stamp}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        archive.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    history = Path("data/universe/history.ndjson")
    record = {
        "snapshot_date": stamp,
        "retrieved_at": snap["retrieved_at"],
        "record_count": snap["record_count"],
        "source_hashes": snap["source_hashes"],
        "symbols": [
            {
                "symbol": x["symbol"],
                "asset_class": x["asset_class"],
                "name": x["name"],
                "tradeable": x["tradeable"],
            }
            for x in snap["records"]
        ],
    }
    existing = history.read_text(encoding="utf-8").splitlines() if history.exists() else []
    if not any(json.loads(line).get("snapshot_date") == stamp for line in existing if line.strip()):
        history.parent.mkdir(parents=True, exist_ok=True)
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"paypay-universe: {snap['record_count']} archived={archive}")

if __name__ == "__main__":
    main()
