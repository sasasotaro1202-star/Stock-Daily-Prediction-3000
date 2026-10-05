from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.data.treasury_api import collect_treasury_curve

OUT = Path("data/research/treasury_curve.parquet")
META = Path("data/research/treasury_curve.json")

def main() -> None:
    year = datetime.now(timezone.utc).year
    frame = collect_treasury_curve(year - 5, year)
    if frame.empty:
        raise SystemExit("DEFERRED: Treasury yield curve returned no rows")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT, index=False)
    payload = {
        "status": "OOS_READY",
        "source": "U.S. Treasury Daily Treasury Par Yield Curve Rates",
        "research_only": True,
        "production_changed": False,
        "rows": int(len(frame)),
        "start_date": str(frame["session_date"].min()),
        "end_date": str(frame["session_date"].max()),
        "feature_count": int(sum(c.startswith("treasury_") for c in frame.columns)),
        "pit_policy": {"type": "next_weekday", "time_jst": "06:00", "fail_closed": True},
    }
    META.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
