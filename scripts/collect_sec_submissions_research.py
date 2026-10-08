from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

from src.data.http_encoding import decode_http_body


CONFIG = Path("config/research_data_sources.yml")
UNIVERSE = Path("data/universe/latest.json")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"
OUT_DIR = Path("data/research/sec_submissions")
USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "Stock-Daily-Prediction-3000/1.0 research-contact",
)
MAX_ATTEMPTS = max(1, int(os.getenv("SEC_SUBMISSIONS_MAX_ATTEMPTS", "3")))
SLEEP_SECONDS = max(0.0, float(os.getenv("SEC_SUBMISSIONS_SLEEP_SECONDS", "0.20")))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_json(url: str) -> tuple[dict, dict[str, str]]:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=45) as response:
                body = response.read()
                content_encoding = str(response.headers.get("Content-Encoding") or "")
                body = decode_http_body(body, content_encoding)
                payload = json.loads(body.decode("utf-8-sig"))
                response_headers = {
                    "etag": str(response.headers.get("ETag") or ""),
                    "last_modified": str(response.headers.get("Last-Modified") or ""),
                    "content_type": str(response.headers.get("Content-Type") or ""),
                    "content_encoding": content_encoding,
                }
            if not isinstance(payload, dict):
                raise ValueError("SEC JSON root is not an object")
            return payload, response_headers
        except (HTTPError, URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= MAX_ATTEMPTS:
                raise
            time.sleep(min(8.0, 2 ** (attempt - 1)))
    raise RuntimeError(f"SEC request failed: {last_error!r}")


def _load_cfg() -> dict:
    if not CONFIG.exists():
        return {}
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}


def _load_current_us_symbols() -> list[str]:
    payload = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    symbols = {
        str(row.get("symbol") or "").strip().upper()
        for row in payload.get("records", [])
        if str(row.get("asset_class") or "").strip().lower() == "us_stock"
        and bool(row.get("tradeable", True))
    }
    return sorted(symbol for symbol in symbols if symbol)


def _build_ticker_map(payload: dict) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for raw in payload.values() if isinstance(payload, dict) else []:
        if not isinstance(raw, dict):
            continue
        ticker = str(raw.get("ticker") or "").strip().upper()
        cik = str(raw.get("cik_str") or "").strip()
        if not ticker or not cik:
            continue
        result[ticker] = {
            "ticker": ticker,
            "cik": str(int(cik)).zfill(10),
            "title": str(raw.get("title") or ""),
            "exchange": str(raw.get("exchange") or ""),
        }
    return result


def _recent_rows(payload: dict) -> list[dict[str, object]]:
    recent = ((payload.get("filings") or {}).get("recent") or {})
    if not isinstance(recent, dict):
        return []
    keys = sorted({key for key, value in recent.items() if isinstance(value, list)})
    n = max((len(recent[key]) for key in keys), default=0)
    rows = []
    for i in range(n):
        row = {key: recent[key][i] for key in keys if i < len(recent[key])}
        if row:
            rows.append(row)
    return rows


def _extract_pit_view(payload: dict) -> list[dict[str, object]]:
    rows = []
    for row in _recent_rows(payload):
        rows.append(
            {
                "form": row.get("form"),
                "filing_date": row.get("filingDate"),
                "acceptance_datetime": row.get("acceptanceDateTime"),
                "report_date": row.get("reportDate"),
                "acceptance_datetime_is_preserved": bool(row.get("acceptanceDateTime")),
                "accession_number": row.get("accessionNumber"),
                "primary_document": row.get("primaryDocument"),
                "is_xbrl": row.get("isXBRL"),
            }
        )
    return rows


def main() -> int:
    if not UNIVERSE.exists():
        raise SystemExit(f"FAIL: missing official universe: {UNIVERSE}")

    cfg = _load_cfg().get("sec_submissions") or {}
    if cfg and cfg.get("enabled") is False:
        raise SystemExit("DEFERRED: sec_submissions disabled")

    ticker_cfg = _get_json(TICKERS_URL)
    if isinstance(ticker_cfg, tuple):
        ticker_payload, ticker_headers = ticker_cfg
    else:
        ticker_payload, ticker_headers = ticker_cfg, {}
    ticker_map = _build_ticker_map(ticker_payload)

    symbols = _load_current_us_symbols()
    shard_count = max(
        1,
        int(os.getenv("SEC_SUBMISSIONS_SHARD_COUNT", cfg.get("shard_count", 4))),
    )
    shard_index = int(
        os.getenv("SEC_SUBMISSIONS_SHARD_INDEX", "0"),
    )
    if shard_index < 0 or shard_index >= shard_count:
        raise SystemExit("FAIL: SEC_SUBMISSIONS_SHARD_INDEX out of range")

    max_symbols = max(
        1,
        int(
            os.getenv(
                "SEC_SUBMISSIONS_MAX_SYMBOLS_PER_SHARD",
                cfg.get("max_symbols_per_shard", 75),
            )
        ),
    )
    selected = [
        symbol
        for position, symbol in enumerate(symbols)
        if position % shard_count == shard_index
    ][:max_symbols]

    template = str(cfg.get("submissions_url_template") or DEFAULT_TEMPLATE)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures: list[dict[str, str]] = []
    manifest_rows = []

    for symbol in selected:
        mapping = ticker_map.get(symbol)
        if not mapping:
            failures.append({"symbol": symbol, "reason": "SEC_TICKER_NOT_FOUND"})
            continue
        cik = mapping["cik"]
        url = template.format(cik=cik)
        try:
            payload, headers = _get_json(url)
            raw_path = OUT_DIR / f"{symbol}.json"
            raw_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            pit_rows = _extract_pit_view(payload)
            pit_path = OUT_DIR / f"{symbol}.pit.json"
            pit_path.write_text(
                json.dumps(
                    {
                        "ticker": symbol,
                        "cik": cik,
                        "acceptance_datetime_rows": pit_rows,
                        "pit_status": "UNVERIFIED",
                        "exact_publication_time_verified": False,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            manifest_rows.append(
                {
                    "symbol": symbol,
                    "cik": cik,
                    "title": mapping["title"],
                    "exchange": mapping["exchange"],
                    "retrieved_at": _utc(),
                    "etag": headers.get("etag"),
                    "last_modified": headers.get("last_modified"),
                    "raw_path": str(raw_path),
                    "pit_path": str(pit_path),
                    "acceptance_datetime_count": sum(
                        1 for row in pit_rows if row["acceptance_datetime"]
                    ),
                }
            )
            successes += 1
            time.sleep(SLEEP_SECONDS)
        except Exception as exc:
            failures.append(
                {
                    "symbol": symbol,
                    "cik": cik,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )

    manifest = {
        "schema_version": 1,
        "status": "EVALUATED" if not failures else ("DEGRADED" if successes else "FAILED"),
        "retrieved_at": _utc(),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "source": "U.S. SEC EDGAR Submissions API",
        "source_url": "https://data.sec.gov/submissions/",
        "ticker_master": {
            "url": TICKERS_URL,
            "etag": ticker_headers.get("etag"),
            "last_modified": ticker_headers.get("last_modified"),
        },
        "pit_status": "UNVERIFIED",
        "pit_policy": {
            "acceptance_datetime_preserved": True,
            "exact_publication_timestamp_verified": False,
            "source_available_at_not_inferred": True,
            "fail_closed_for_production": True,
        },
        "universe": {
            "path": str(UNIVERSE),
            "source_scope": "current_official_paypay_universe_us_stock",
            "historical_survivorship_audit_complete": False,
        },
        "shard": {
            "index": shard_index,
            "count": shard_count,
            "selected_symbols": len(selected),
        },
        "summary": {
            "successful": successes,
            "failed": len(failures),
            "acceptance_datetime_records": sum(
                row["acceptance_datetime_count"] for row in manifest_rows
            ),
        },
        "rows": manifest_rows,
        "failures": failures,
    }
    (OUT_DIR / f"manifest-{shard_index}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if selected and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
