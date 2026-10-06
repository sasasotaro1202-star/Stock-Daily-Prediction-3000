from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml


CONFIG = Path("config/research_data_sources.yml")
USER_AGENT = "Stock-Daily-Prediction-3000/mof-securities-research"
DEFAULT_TIMEOUT = int(os.getenv("MOF_REQUEST_TIMEOUT_SECONDS", "30"))
ENCODINGS = ("utf-8-sig", "utf-8", "cp932", "shift_jis")


def _download(url: str) -> tuple[bytes, dict[str, str]]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/csv,text/plain,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
        body = response.read()
        headers = {
            "content_type": str(response.headers.get("Content-Type") or ""),
            "last_modified": str(response.headers.get("Last-Modified") or ""),
            "etag": str(response.headers.get("ETag") or ""),
            "final_url": str(response.geturl()),
        }
    return body, headers


def _decode(body: bytes) -> tuple[str, str]:
    for encoding in ENCODINGS:
        try:
            return body.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", body, 0, len(body), "unsupported encoding")


def _parse_csv_shape(text: str) -> dict[str, int]:
    lines = [line for line in text.splitlines() if line.strip()]
    widths = []
    for line in lines[:100]:
        widths.append(len(line.split(",")))
    return {
        "non_empty_lines": len(lines),
        "max_columns_first_100_lines": max(widths, default=0),
    }


def collect(config: dict) -> dict:
    cfg = config.get("mof_securities") or {}
    if not bool(cfg.get("enabled", True)):
        return {
            "status": "DISABLED",
            "research_only": True,
            "production_changed": False,
        }

    output_dir = Path(str(cfg.get("output_dir", "data/research/mof_securities")))
    output_dir.mkdir(parents=True, exist_ok=True)
    checked_at = datetime.now(timezone.utc).isoformat()
    sources = cfg.get("sources") or {}

    results = []
    for name, entry in sources.items():
        url = str((entry or {}).get("url") or "").strip()
        kind = str((entry or {}).get("kind") or "").strip()
        if not url:
            results.append({
                "name": name,
                "status": "FAILED",
                "error": "missing_url",
            })
            continue

        try:
            body, headers = _download(url)
            decoded, encoding = _decode(body)
            shape = _parse_csv_shape(decoded)

            raw_path = output_dir / f"{name}.csv"
            raw_path.write_bytes(body)

            results.append({
                "name": name,
                "kind": kind,
                "status": "SUCCESS",
                "url": url,
                "final_url": headers.get("final_url"),
                "content_type": headers.get("content_type"),
                "last_modified": headers.get("last_modified") or None,
                "etag": headers.get("etag") or None,
                "encoding": encoding,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "raw_path": str(raw_path),
                "csv_shape_hint": shape,
            })
        except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError) as exc:
            results.append({
                "name": name,
                "kind": kind,
                "status": "FAILED",
                "url": url,
                "error": f"{type(exc).__name__}:{exc}",
            })

    succeeded = sum(row["status"] == "SUCCESS" for row in results)
    failed = sum(row["status"] == "FAILED" for row in results)
    report = {
        "schema_version": 1,
        "status": "EVALUATED" if succeeded and not failed else "DEGRADED" if succeeded else "FAILED",
        "checked_at_utc": checked_at,
        "source": "Japan Ministry of Finance Securities Transactions",
        "source_url": "https://www.mof.go.jp/policy/international_policy/reference/itn_transactions_in_securities/data.htm",
        "research_only": True,
        "production_changed": False,
        "pit_status": "UNVERIFIED",
        "pit_policy": {
            "retrieved_at_recorded": True,
            "http_last_modified_recorded": True,
            "source_available_at_inferred": False,
            "feature_adoption_blocked_until_pit_validation": True,
        },
        "results": results,
        "summary": {
            "configured_sources": len(sources),
            "successful_sources": succeeded,
            "failed_sources": failed,
            "raw_data_preserved": succeeded > 0,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    if not CONFIG.exists():
        raise SystemExit("FAIL: research data-source config is missing")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    report = collect(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
