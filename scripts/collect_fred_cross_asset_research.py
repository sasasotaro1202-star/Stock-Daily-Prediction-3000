from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml

CONFIG = Path("config/research_data_sources.yml")
DEFAULT_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
OUT_DIR = Path("data/research/fred_cross_asset")
UA = "Stock-Daily-Prediction-3000/1.0 fred-research"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch(url: str) -> tuple[bytes, dict[str, str]]:
    req = Request(
        url,
        headers={
            "Accept": "text/csv,text/plain,*/*",
            "User-Agent": UA,
        },
    )
    with urlopen(req, timeout=45) as response:
        raw = response.read()
        headers = {
            "etag": str(response.headers.get("ETag") or ""),
            "last_modified": str(response.headers.get("Last-Modified") or ""),
            "content_type": str(response.headers.get("Content-Type") or ""),
        }
    if not raw:
        raise ValueError("FRED response body is empty")
    return raw, headers


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    source = cfg.get("fred_cross_asset") or {}
    if source.get("enabled") is False:
        raise SystemExit("DEFERRED: FRED cross-asset research disabled")

    base_url = str(source.get("base_url") or DEFAULT_BASE_URL)
    series = [str(x).strip().upper() for x in source.get("series") or []]
    series = [x for x in series if x]
    if not series:
        raise SystemExit("FAIL: no FRED series configured")

    years = max(1, int(os.getenv("FRED_LOOKBACK_YEARS", "10")))
    start = datetime.now(timezone.utc).date().replace(
        year=max(1900, datetime.now(timezone.utc).year - years)
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for series_id in series:
        url = base_url + "?" + urlencode(
            {
                "id": series_id,
                "cosd": start.isoformat(),
                "coed": datetime.now(timezone.utc).date().isoformat(),
                "fq": "Daily",
            }
        )
        try:
            raw, headers = _fetch(url)
            path = OUT_DIR / f"{series_id}.csv"
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            line_count = raw.count(b"\n")
            rows.append(
                {
                    "series": series_id,
                    "status": "EVALUATED",
                    "url": url,
                    "retrieved_at": _utc(),
                    "bytes": len(raw),
                    "line_count_hint": line_count,
                    "sha256": digest,
                    "headers": headers,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "series": series_id,
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}:{exc}",
                    "retrieved_at": _utc(),
                }
            )

    success_count = sum(row["status"] == "EVALUATED" for row in rows)
    report = {
        "schema_version": 1,
        "status": "EVALUATED" if success_count == len(rows) else ("DEGRADED" if success_count else "FAILED"),
        "retrieved_at": _utc(),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "source": "Federal Reserve Bank of St. Louis FRED public graph CSV",
        "series_requested": series,
        "lookback_years": years,
        "summary": {
            "requested": len(rows),
            "successful": success_count,
            "failed": len(rows) - success_count,
        },
        "pit_status": "UNVERIFIED",
        "pit_policy": {
            "retrieval_timestamp_recorded": True,
            "vintage_timestamp_available": False,
            "source_available_at_inferred": False,
            "revision_policy_requires_local_snapshot": True,
            "fail_closed_for_production": True,
        },
        "series": rows,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "research_only": True,
        "production_changed": False,
        "pit_status": "UNVERIFIED",
        **report["summary"],
    }, indent=2))
    return 0 if success_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
