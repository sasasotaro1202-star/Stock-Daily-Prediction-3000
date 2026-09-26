from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.boj_api import collect_from_config, load_config

OUT = Path("data/research/boj_timeseries.parquet")
META = Path("data/research/boj_timeseries.json")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()
    frame, metadata = collect_from_config(
        load_config(),
        start_override=args.start_date,
        end_override=args.end_date,
    )
    if metadata["status"] != "OOS_READY":
        raise SystemExit("DEFERRED: no PIT-visible BOJ observations")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    META.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
