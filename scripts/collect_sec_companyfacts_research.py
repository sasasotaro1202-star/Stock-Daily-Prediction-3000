from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


UNIVERSE = Path("data/universe/latest.json")
CONFIG = Path("config/research_data_sources.yml")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
OUT_DIR = Path("data/research/sec_companyfacts")
USER_AGENT = os.getenv(
    "SEC_COMPANYFACTS_USER_AGENT",
    "Stock-Daily-Prediction-3000/0.1",
)
TIMEOUT = int(os.getenv("SEC_COMPANYFACTS_TIMEOUT_SECONDS", "60"))
MAX_ATTEMPTS = max(1, int(os.getenv("SEC_COMPANYFACTS_MAX_ATTEMPTS", "3")))
SLEEP_SECONDS = max(0.05, float(os.getenv("SEC_COMPANYFACTS_SLEEP_SECONDS", "0.20")))


def _get_json(url: str) -> tuple[dict, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        req = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(req, timeout=TIMEOUT) as response:
                body = response.read()
                return (
                    json.loads(body.decode("utf-8-sig")),
                    {
                        "content_type": str(response.headers.get("Content-Type") or ""),
                        "last_modified": str(response.headers.get("Last-Modified") or ""),
                        "etag": str(response.headers.get("ETag") or ""),
                    },
                )
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= MAX_ATTEMPTS:
                raise
            retry_after = None
            if isinstance(exc, HTTPError):
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = max(1.0, float(retry_after)) if retry_after else 2 ** (attempt - 1)
            except (TypeError, ValueError):
                delay = 2 ** (attempt - 1)
            time.sleep(min(15.0, delay))
    raise RuntimeError(f"SEC request failed: {last_error!r}")


def _universe_symbols() -> list[str]:
    if not UNIVERSE.exists():
        raise SystemExit("DEFERRED: official PayPay universe snapshot is missing")
    payload = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    records = payload.get("records") or []
    symbols = {
        str(row.get("symbol") or "").strip().upper()
        for row in records
        if str(row.get("asset_class") or "") == "us_stock"
        and bool(row.get("tradeable", True))
    }
    return sorted(symbol for symbol in symbols if symbol)


def _official_ticker_map(payload: dict) -> dict[str, dict]:
    out = {}
    for row in (payload.values() if isinstance(payload, dict) else []):
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        cik = str(row.get("cik_str") or "").strip()
        if ticker and cik:
            out[ticker] = {
                "ticker": ticker,
                "cik": cik.zfill(10),
                "title": str(row.get("title") or "").strip(),
                "exchange": str(row.get("exchange") or "").strip(),
            }
    return out


def main() -> int:
    if not CONFIG.exists():
        raise SystemExit("FAIL: research data-source config is missing")

    config = json.loads(json.dumps({}))
    try:
        import yaml
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise SystemExit(f"FAIL: cannot load research data-source config: {type(exc).__name__}:{exc}") from exc

    cfg = config.get("sec_companyfacts") or {}
    if not bool(cfg.get("enabled", True)):
        print(json.dumps({"status": "DISABLED", "research_only": True}, indent=2))
        return 0

    max_per_shard = max(1, int(cfg.get("max_symbols_per_shard", 75)))
    shard_count = max(1, int(os.getenv("SEC_COMPANYFACTS_SHARD_COUNT", cfg.get("shard_count", 4))))
    shard_index = int(os.getenv("SEC_COMPANYFACTS_SHARD_INDEX", "0"))
    if shard_index < 0 or shard_index >= shard_count:
        raise SystemExit("FAIL: SEC_COMPANYFACTS_SHARD_INDEX is outside shard range")

    checked_at = datetime.now(timezone.utc).isoformat()
    universe_symbols = _universe_symbols()
    ticker_payload, ticker_headers = _get_json(TICKERS_URL)
    ticker_map = _official_ticker_map(ticker_payload)

    matched = [ticker_map[symbol] for symbol in universe_symbols if symbol in ticker_map]
    missing_cik = [symbol for symbol in universe_symbols if symbol not in ticker_map]

    shard_rows = [
        row for pos, row in enumerate(matched)
        if pos % shard_count == shard_index
    ][:max_per_shard]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for row in shard_rows:
        cik = row["cik"]
        ticker = row["ticker"]
        url = str(cfg["companyfacts_url_template"]).format(cik=cik)
        try:
            payload, headers = _get_json(url)
            out = OUT_DIR / f"{ticker}.json"
            raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            out.write_bytes(raw)
            results.append({
                "ticker": ticker,
                "cik": cik,
                "title": row["title"],
                "exchange": row["exchange"],
                "status": "SUCCESS",
                "source_url": url,
                "raw_path": str(out),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "last_modified": headers.get("last_modified") or None,
                "etag": headers.get("etag") or None,
            })
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            results.append({
                "ticker": ticker,
                "cik": cik,
                "status": "FAILED",
                "source_url": url,
                "error": f"{type(exc).__name__}:{exc}",
            })
        time.sleep(SLEEP_SECONDS)

    success = sum(row["status"] == "SUCCESS" for row in results)
    failed = sum(row["status"] == "FAILED" for row in results)
    report = {
        "schema_version": 1,
        "status": "EVALUATED" if success and not failed else "DEGRADED" if success else "FAILED",
        "checked_at_utc": checked_at,
        "source": "U.S. SEC EDGAR XBRL Company Facts",
        "source_url": "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
        "research_only": True,
        "production_changed": False,
        "pit_status": "UNVERIFIED",
        "pit_policy": {
            "filing_date_preserved": True,
            "exact_acceptance_timestamp_verified": False,
            "historical_available_at_inferred": False,
            "companyfacts_not_used_as_production_feature_without_local_pit_validation": True,
        },
        "ticker_master": {
            "url": TICKERS_URL,
            "sha256": hashlib.sha256(
                json.dumps(ticker_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "last_modified": ticker_headers.get("last_modified") or None,
            "etag": ticker_headers.get("etag") or None,
        },
        "universe": {
            "us_stock_symbols": len(universe_symbols),
            "matched_to_sec_cik": len(matched),
            "missing_sec_cik": missing_cik,
            "shard_index": shard_index,
            "shard_count": shard_count,
        },
        "results": results,
        "summary": {
            "attempted": len(results),
            "successful": success,
            "failed": failed,
            "raw_companyfacts_preserved": success > 0,
            "xbrl_facts_are_available_for_future_local_pit_mapping": success > 0,
        },
    }
    manifest = OUT_DIR / "manifest.json"
    manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
