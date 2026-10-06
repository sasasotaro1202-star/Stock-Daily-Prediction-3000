from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml

CONFIG = Path("config/research_data_sources.yml")
OUT = Path("data/research/boj_frontier_research.json")
UA = "Stock-Daily-Prediction-3000/1.0 boj-frontier-research"
MAX_ATTEMPTS = max(1, int(os.getenv("BOJ_FRONTIER_MAX_ATTEMPTS", "3")))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch(url: str) -> tuple[dict, bytes]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            request = Request(
                url,
                headers={"Accept": "application/json", "User-Agent": UA},
            )
            with urlopen(request, timeout=45) as response:
                raw = response.read()
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("BOJ response root is not an object")
            return payload, raw
        except Exception as exc:
            last_error = exc
            if attempt == MAX_ATTEMPTS:
                raise
    raise RuntimeError(f"BOJ request failed: {last_error!r}")


def _resultset(payload: dict) -> list[dict]:
    direct = payload.get("RESULTSET")
    if isinstance(direct, list):
        return direct
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("RESULTSET"), list):
        return data["RESULTSET"]
    raise ValueError("FAIL: BOJ RESULTSET missing")


def _structured_rows(payload: dict, db: str) -> list[dict]:
    rows = []
    for series in _resultset(payload):
        if not isinstance(series, dict):
            continue
        values = series.get("VALUES") or {}
        dates = values.get("SURVEY_DATES") if isinstance(values, dict) else None
        numbers = values.get("VALUES") if isinstance(values, dict) else None
        if not isinstance(dates, list) or not isinstance(numbers, list):
            continue
        for raw_date, raw_value in zip(dates, numbers):
            rows.append(
                {
                    "db": db,
                    "series_code": str(series.get("SERIES_CODE") or ""),
                    "series_name": str(series.get("NAME_OF_TIME_SERIES") or ""),
                    "unit": str(series.get("UNIT") or ""),
                    "frequency": str(series.get("FREQUENCY") or ""),
                    "observation_date": str(raw_date),
                    "value": raw_value,
                }
            )
    return rows


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    source_cfg = cfg.get("boj_frontier_research") or {}
    if source_cfg.get("enabled") is False:
        raise SystemExit("DEFERRED: BOJ frontier research disabled")

    base_url = str(
        source_cfg.get("base_url")
        or "https://www.stat-search.boj.or.jp/api/v1/getDataCode"
    )
    language = str(source_cfg.get("language") or "en")
    start_dates = source_cfg.get("start_dates") or {}
    groups = source_cfg.get("series_groups") or []
    if not groups:
        raise SystemExit("FAIL: no BOJ frontier series groups configured")

    batches = []
    total_rows = 0
    total_series = 0
    for group in groups:
        db = str(group.get("db") or "")
        codes = [str(x) for x in group.get("codes") or [] if str(x).strip()]
        if not db or not codes:
            continue
        url = base_url + "?" + urlencode(
            {
                "format": "json",
                "lang": language,
                "db": db,
                "code": ",".join(codes),
                "startDate": str(start_dates.get(db) or "200001"),
            }
        )
        payload, raw = _fetch(url)
        rows = _structured_rows(payload, db)
        batch = {
            "db": db,
            "codes": codes,
            "url": url,
            "retrieved_at": _utc(),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_payload": payload,
            "rows": rows,
        }
        batches.append(batch)
        total_rows += len(rows)
        total_series += len(_resultset(payload))
    
    report = {
        "schema_version": 1,
        "status": "EVALUATED" if total_rows else "FAILED",
        "retrieved_at": _utc(),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "source": "Bank of Japan Time-Series Data Search API",
        "api_available_to_anyone": True,
        "selected_domains": [
            "monetary_base",
            "current_account_balances",
            "balance_of_payments",
            "tankan_business_conditions",
        ],
        "summary": {
            "batch_count": len(batches),
            "series_count": total_series,
            "observation_rows": total_rows,
        },
        "pit_status": "UNVERIFIED",
        "pit_policy": {
            "observation_timestamp_is_not_release_timestamp": True,
            "available_at_verified": False,
            "revision_policy_verified": False,
            "source_available_at_inferred": False,
            "fail_closed_for_production": True,
        },
        "batches": batches,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "research_only": True,
        "production_changed": False,
        "pit_status": "UNVERIFIED",
        "series_count": total_series,
        "observation_rows": total_rows,
    }, indent=2))
    return 0 if total_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
