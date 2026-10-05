from __future__ import annotations

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml


CONFIG = Path("config/research_data_sources.yml")
DEFAULT_BASE_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
OUT_DIR = Path("data/research/treasury_rate_family")
USER_AGENT = "Stock-Daily-Prediction-3000/1.0 treasury-rate-family-research"
MAX_ATTEMPTS = max(1, int(os.getenv("TREASURY_RATE_MAX_ATTEMPTS", "3")))
SLEEP_SECONDS = max(0.0, float(os.getenv("TREASURY_RATE_SLEEP_SECONDS", "0.20")))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _get_bytes(url: str) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/xml,text/xml",
            },
        )
        try:
            with urlopen(request, timeout=45) as response:
                payload = response.read()
                headers = {
                    "etag": str(response.headers.get("ETag") or ""),
                    "last_modified": str(response.headers.get("Last-Modified") or ""),
                    "content_type": str(response.headers.get("Content-Type") or ""),
                }
            return payload, headers
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_ATTEMPTS:
                raise
            time.sleep(min(8.0, 2 ** (attempt - 1)))
    raise RuntimeError(f"Treasury rate request failed: {last_error!r}")


def _parse_xml(payload: bytes, dataset: str) -> list[dict[str, object]]:
    root = ET.fromstring(payload)
    rows: list[dict[str, object]] = []
    for entry in root.iter():
        if _local_name(entry.tag) != "entry":
            continue
        props = next(
            (child for child in entry.iter() if _local_name(child.tag) == "properties"),
            None,
        )
        if props is None:
            continue
        raw_values = {
            _local_name(child.tag): (child.text or "").strip()
            for child in props
        }
        raw_date = raw_values.get("NEW_DATE") or raw_values.get("record_date")
        if not raw_date:
            continue
        row: dict[str, object] = {
            "dataset": dataset,
            "session_date": str(raw_date)[:10],
        }
        for key, value in raw_values.items():
            if key in {"NEW_DATE", "record_date"}:
                continue
            if value in {"", "N/A", "NA", "null", "None"}:
                row[key] = None
                continue
            if re.fullmatch(r"-?(?:\d+(?:\.\d*)?|\.\d+)", value):
                try:
                    row[key] = float(value)
                    continue
                except ValueError:
                    pass
            row[key] = value
        rows.append(row)
    return rows


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    source_cfg = cfg.get("treasury_rate_family") or {}
    if source_cfg.get("enabled") is False:
        raise SystemExit("DEFERRED: treasury_rate_family disabled")

    base_url = str(source_cfg.get("base_url") or DEFAULT_BASE_URL)
    datasets = [str(x) for x in source_cfg.get("datasets") or []]
    if not datasets:
        raise SystemExit("FAIL: no Treasury rate-family datasets configured")

    now = datetime.now(timezone.utc)
    lookback_years = max(1, int(source_cfg.get("lookback_years", 5)))
    try:
        start_year = now.year - lookback_years
    except Exception:
        start_year = now.year - 5

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    total_rows = 0

    for dataset in datasets:
        dataset_rows = []
        for year in range(start_year, now.year + 1):
            url = base_url + "?" + urlencode(
                {
                    "data": dataset,
                    "field_tdr_date_value": str(year),
                }
            )
            payload, headers = _get_bytes(url)
            raw_path = OUT_DIR / "raw" / f"{dataset}-{year}.xml"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(payload)
            sha = hashlib.sha256(payload).hexdigest()
            rows = _parse_xml(payload, dataset)
            dataset_rows.extend(rows)
            manifest_rows.append(
                {
                    "dataset": dataset,
                    "year": year,
                    "retrieved_at": _utc(),
                    "rows": len(rows),
                    "sha256": sha,
                    "raw_path": str(raw_path),
                    "etag": headers.get("etag"),
                    "last_modified": headers.get("last_modified"),
                }
            )
            total_rows += len(rows)
            time.sleep(SLEEP_SECONDS)

        (OUT_DIR / f"{dataset}.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "rows": dataset_rows,
                    "pit_status": "UNVERIFIED",
                    "research_only": True,
                    "production_changed": False,
                },
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "schema_version": 1,
        "status": "EVALUATED" if total_rows else "FAILED",
        "retrieved_at": _utc(),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "source": "U.S. Treasury Daily Interest Rate XML Feed",
        "base_url": base_url,
        "datasets": datasets,
        "lookback_start_year": start_year,
        "rows": manifest_rows,
        "summary": {"total_rows": total_rows, "dataset_count": len(datasets)},
        "pit_status": "UNVERIFIED",
        "pit_policy": {
            "daily_date_preserved": True,
            "exact_publication_time_verified": False,
            "source_available_at_not_inferred": True,
            "fail_closed_for_production": True,
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if total_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
