from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

from src.data.http_encoding import decode_http_body


CONFIG = Path("config/research_data_sources.yml")
UNIVERSE = Path("data/universe/latest.json")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
OUT_DIR = Path("data/research/sec_company_events")
UA = os.getenv(
    "SEC_USER_AGENT",
    "Stock-Daily-Prediction-3000/1.0 company-event-research",
)
MAX_ATTEMPTS = max(1, int(os.getenv("SEC_COMPANY_EVENTS_MAX_ATTEMPTS", "3")))
SLEEP_SECONDS = max(0.0, float(os.getenv("SEC_COMPANY_EVENTS_SLEEP_SECONDS", "0.20")))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_json(url: str) -> tuple[dict, dict[str, str], bytes]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = Request(
            url,
            headers={
                "User-Agent": UA,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            },
        )
        try:
            with urlopen(request, timeout=45) as response:
                raw = response.read()
                content_encoding = str(response.headers.get("Content-Encoding") or "")
                raw = decode_http_body(raw, content_encoding)
                headers = {
                    "etag": str(response.headers.get("ETag") or ""),
                    "last_modified": str(response.headers.get("Last-Modified") or ""),
                    "content_type": str(response.headers.get("Content-Type") or ""),
                    "content_encoding": content_encoding,
                }
            payload = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(payload, dict):
                raise ValueError("SEC JSON root is not an object")
            return payload, headers, raw
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_ATTEMPTS:
                raise
            time.sleep(min(8.0, 2 ** (attempt - 1)))
    raise RuntimeError(f"SEC request failed: {last_error!r}")


def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}


def _load_us_symbols() -> list[str]:
    payload = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    return sorted(
        {
            str(row.get("symbol") or "").strip().upper()
            for row in payload.get("records", [])
            if str(row.get("asset_class") or "").strip().lower() == "us_stock"
            and bool(row.get("tradeable", True))
        }
        - {""}
    )


def _ticker_map(payload: dict) -> dict[str, dict[str, str]]:
    result = {}
    for raw in payload.values() if isinstance(payload, dict) else []:
        if not isinstance(raw, dict):
            continue
        ticker = str(raw.get("ticker") or "").strip().upper()
        cik = str(raw.get("cik_str") or "").strip()
        if ticker and cik:
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
    keys = [key for key, value in recent.items() if isinstance(value, list)]
    count = max((len(recent[key]) for key in keys), default=0)
    return [
        {key: recent[key][idx] for key in keys if idx < len(recent[key])}
        for idx in range(count)
    ]


def _date_or_none(raw: object) -> date | None:
    text = str(raw or "").strip()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


def _normalize_items(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def _event_rows(
    payload: dict,
    *,
    forms: set[str],
    cutoff_date: date,
    max_events: int,
) -> list[dict[str, object]]:
    rows = []
    for row in _recent_rows(payload):
        form = str(row.get("form") or "").strip()
        if form not in forms:
            continue
        filing_date = _date_or_none(row.get("filingDate"))
        if filing_date is None or filing_date < cutoff_date:
            continue
        rows.append(
            {
                "form": form,
                "filing_date": str(row.get("filingDate") or ""),
                "acceptance_datetime": str(row.get("acceptanceDateTime") or ""),
                "report_date": str(row.get("reportDate") or ""),
                "event_date": str(row.get("reportDate") or "")
                or str(row.get("filingDate") or ""),
                "items": _normalize_items(row.get("items")),
                "accession_number": str(row.get("accessionNumber") or ""),
                "primary_document": str(row.get("primaryDocument") or ""),
                "is_xbrl": row.get("isXBRL"),
                "is_inline_xbrl": row.get("isInlineXBRL"),
                "period": str(row.get("period") or ""),
                "act": str(row.get("act") or ""),
                "file_number": str(row.get("fileNumber") or ""),
                "film_number": str(row.get("filmNumber") or ""),
            }
        )
    rows.sort(
        key=lambda row: (
            row["acceptance_datetime"],
            row["filing_date"],
            row["accession_number"],
        ),
        reverse=True,
    )
    return rows[:max_events]


def main() -> int:
    if not UNIVERSE.exists():
        raise SystemExit(f"FAIL: missing official universe: {UNIVERSE}")

    cfg = _load_cfg().get("sec_company_events") or {}
    if cfg.get("enabled") is False:
        raise SystemExit("DEFERRED: sec_company_events disabled")

    ticker_payload, ticker_headers, ticker_raw = _get_json(TICKERS_URL)
    mapping = _ticker_map(ticker_payload)
    symbols = _load_us_symbols()

    shard_count = max(
        1,
        int(os.getenv("SEC_COMPANY_EVENTS_SHARD_COUNT", cfg.get("shard_count", 4))),
    )
    shard_index = int(os.getenv("SEC_COMPANY_EVENTS_SHARD_INDEX", "0"))
    if not 0 <= shard_index < shard_count:
        raise SystemExit("FAIL: SEC_COMPANY_EVENTS_SHARD_INDEX out of range")

    max_symbols = max(
        1,
        int(
            os.getenv(
                "SEC_COMPANY_EVENTS_MAX_SYMBOLS_PER_SHARD",
                cfg.get("max_symbols_per_shard", 75),
            )
        ),
    )
    selected = [
        symbol for idx, symbol in enumerate(symbols) if idx % shard_count == shard_index
    ][:max_symbols]

    forms = {
        str(form).strip()
        for form in cfg.get("forms") or []
        if str(form).strip()
    }
    if not forms:
        raise SystemExit("FAIL: SEC company-event forms are empty")

    lookback_days = max(1, int(cfg.get("lookback_days", 365)))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date()
    max_events = max(1, int(cfg.get("max_events_per_symbol", 40)))
    submissions_template = str(
        cfg.get("submissions_url_template") or DEFAULT_SUBMISSIONS
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = []
    total_events = 0
    manifest_rows = []

    for symbol in selected:
        info = mapping.get(symbol)
        if not info:
            failures.append({"symbol": symbol, "reason": "SEC_TICKER_NOT_FOUND"})
            continue

        try:
            payload, headers, raw = _get_json(
                submissions_template.format(cik=info["cik"])
            )
            events = _event_rows(
                payload,
                forms=forms,
                cutoff_date=cutoff,
                max_events=max_events,
            )

            raw_path = OUT_DIR / f"{symbol}.submissions.json"
            raw_path.write_bytes(raw)

            event_path = OUT_DIR / f"{symbol}.events.json"
            event_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "symbol": symbol,
                        "cik": info["cik"],
                        "lookback_days": lookback_days,
                        "cutoff_date": cutoff.isoformat(),
                        "events": events,
                        "research_only": True,
                        "production_changed": False,
                        "pit_status": "UNVERIFIED",
                        "pit_policy": {
                            "acceptance_datetime_preserved": True,
                            "filing_date_preserved": True,
                            "report_date_preserved": True,
                            "8k_item_codes_preserved": True,
                            "source_available_at_inferred": False,
                            "exact_publication_timestamp_verified": False,
                            "fail_closed_for_production": True,
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest_rows.append(
                {
                    "symbol": symbol,
                    "cik": info["cik"],
                    "retrieved_at": _utc(),
                    "submission_sha256": hashlib.sha256(raw).hexdigest(),
                    "submission_etag": headers.get("etag"),
                    "submission_last_modified": headers.get("last_modified"),
                    "event_count": len(events),
                    "raw_path": str(raw_path),
                    "event_path": str(event_path),
                    "has_8k_items": any(
                        row["form"].startswith("8-K") and bool(row["items"])
                        for row in events
                    ),
                }
            )
            successes += 1
            total_events += len(events)
            time.sleep(SLEEP_SECONDS)
        except Exception as exc:
            failures.append(
                {
                    "symbol": symbol,
                    "cik": info["cik"],
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )

    manifest = {
        "schema_version": 1,
        "status": (
            "EVALUATED"
            if selected and not failures
            else ("DEGRADED" if successes else "FAILED")
        ),
        "retrieved_at": _utc(),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "source": "U.S. SEC EDGAR current and periodic company filings",
        "source_url": "https://data.sec.gov/submissions/",
        "forms": sorted(forms),
        "lookback_days": lookback_days,
        "ticker_master": {
            "url": TICKERS_URL,
            "sha256": hashlib.sha256(ticker_raw).hexdigest(),
            "etag": ticker_headers.get("etag"),
            "last_modified": ticker_headers.get("last_modified"),
        },
        "shard": {
            "index": shard_index,
            "count": shard_count,
            "selected_symbols": len(selected),
        },
        "summary": {
            "successful_symbols": successes,
            "failed_symbols": len(failures),
            "events": total_events,
        },
        "pit_status": "UNVERIFIED",
        "pit_policy": {
            "acceptance_datetime_preserved": True,
            "filing_date_preserved": True,
            "report_date_preserved": True,
            "8k_item_codes_preserved": True,
            "exact_publication_timestamp_verified": False,
            "source_available_at_not_inferred": True,
            "fail_closed_for_production": True,
        },
        "rows": manifest_rows,
        "failures": failures,
    }
    (OUT_DIR / f"manifest-{shard_index}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "status": manifest["status"],
            "research_only": True,
            "production_changed": False,
            "pit_status": "UNVERIFIED",
            "selected_symbols": len(selected),
            "successful_symbols": successes,
            "failed_symbols": len(failures),
            "events": total_events,
        },
        indent=2,
    ))
    return 0 if selected and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
