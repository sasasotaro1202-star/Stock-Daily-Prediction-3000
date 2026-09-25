from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

UNIVERSE = Path("data/universe/latest.json")
OUT = Path("data/research/sec_filings_research.parquet")
META = Path("data/research/sec_filings_research.json")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
ALLOWED_FORMS = {
    "8-K", "10-K", "10-Q", "20-F", "6-K", "40-F",
    "S-1", "S-3", "S-4", "424B2", "DEF 14A", "SC 13D", "SC 13G",
}
USER_AGENT = "Stock-Daily-Prediction-3000/0.1 (+https://github.com/sasasotaro1202-star/Stock-Daily-Prediction-3000)"
REQUEST_TIMEOUT = 25
RATE_SLEEP_SECONDS = 0.15
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
MAX_REQUEST_ATTEMPTS = 3


def _get_json(url: str) -> object:
    last_error: Exception | None = None
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        req = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            },
        )
        try:
            with urlopen(req, timeout=REQUEST_TIMEOUT) as response:
                return json.load(response)
        except HTTPError as exc:
            last_error = exc
            if exc.code not in RETRYABLE_HTTP_CODES or attempt >= MAX_REQUEST_ATTEMPTS:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = min(15.0, max(1.0, float(retry_after))) if retry_after else float(2 ** (attempt - 1))
            except (TypeError, ValueError):
                delay = float(2 ** (attempt - 1))
            time.sleep(delay)
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= MAX_REQUEST_ATTEMPTS:
                raise
            time.sleep(float(2 ** (attempt - 1)))
    if last_error is not None:
        raise last_error
    raise RuntimeError("SEC request failed without an exception")


def _norm_ticker(value: object) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def _rows_from_submissions(
    universe_row: dict[str, object],
    ticker_info: dict[str, object],
    payload: object,
    collected_at: pd.Timestamp,
) -> list[dict[str, object]]:
    recent = payload.get("filings", {}).get("recent", {}) if isinstance(payload, dict) else {}
    if not isinstance(recent, dict):
        return []
    fields = {k: recent.get(k, []) for k in (
        "accessionNumber", "filingDate", "acceptanceDateTime", "form", "primaryDocument"
    )}
    n = max((len(v) for v in fields.values()), default=0)
    rows = []
    for i in range(n):
        form = str(fields["form"][i]) if i < len(fields["form"]) else ""
        if form not in ALLOWED_FORMS:
            continue
        accepted = pd.to_datetime(
            str(fields["acceptanceDateTime"][i]) if i < len(fields["acceptanceDateTime"]) else "",
            utc=True,
            errors="coerce",
        )
        if pd.isna(accepted) or accepted > collected_at:
            continue
        rows.append({
            "symbol": str(universe_row.get("symbol", "")),
            "asset_class": "us_stock",
            "issuer_name": str(universe_row.get("name", "")),
            "sec_title": str(ticker_info.get("title", "")),
            "cik": str(ticker_info.get("cik", "")),
            "form": form,
            "filing_date": str(fields["filingDate"][i]) if i < len(fields["filingDate"]) else "",
            "acceptance_datetime": accepted.isoformat(),
            "available_at": accepted.isoformat(),
            "accession_number": str(fields["accessionNumber"][i]) if i < len(fields["accessionNumber"]) else "",
            "primary_document": str(fields["primaryDocument"][i]) if i < len(fields["primaryDocument"]) else "",
            "source": "sec_edgar_submissions",
            "research_only": True,
            "production_changed": False,
            "collected_at": collected_at.isoformat(),
        })
    return rows


def main() -> None:
    started = pd.Timestamp.now(tz="UTC")
    if not UNIVERSE.exists():
        payload = {
            "status": "DEFERRED", "reason": "universe_snapshot_missing", "rows": 0,
            "research_only": True, "production_changed": False,
        }
        META.parent.mkdir(parents=True, exist_ok=True)
        META.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    try:
        universe = json.loads(UNIVERSE.read_text(encoding="utf-8"))
        records = [
            x for x in universe.get("records", [])
            if isinstance(x, dict)
            and x.get("asset_class") == "us_stock"
            and bool(x.get("tradeable"))
            and str(x.get("symbol", "")).strip()
        ]
        ticker_payload = _get_json(TICKERS_URL)
        ticker_map = {}
        for value in ticker_payload.values() if isinstance(ticker_payload, dict) else []:
            if not isinstance(value, dict):
                continue
            ticker = _norm_ticker(value.get("ticker"))
            if ticker and value.get("cik_str") is not None:
                ticker_map[ticker] = {
                    "cik": str(value["cik_str"]).zfill(10),
                    "title": str(value.get("title", "")),
                }
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "status": "DEFERRED",
            "reason": f"sec_metadata_fetch_failed:{getattr(exc, 'code', '')}:{type(exc).__name__}",
            "rows": 0,
            "research_only": True,
            "production_changed": False,
        }
        META.parent.mkdir(parents=True, exist_ok=True)
        META.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    collected_at = pd.Timestamp(datetime.now(timezone.utc))
    rows = []
    matched = 0
    deferred = 0
    for row in sorted(records, key=lambda x: str(x.get("symbol", ""))):
        info = ticker_map.get(_norm_ticker(row.get("symbol")))
        if not info:
            deferred += 1
            continue
        matched += 1
        try:
            payload = _get_json(f"https://data.sec.gov/submissions/CIK{info['cik']}.json")
            rows.extend(_rows_from_submissions(row, info, payload, collected_at))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            deferred += 1
        time.sleep(RATE_SLEEP_SECONDS)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(["cik", "accession_number"], keep="last")
        out = out.sort_values(["symbol", "available_at", "accession_number"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    summary = {
        "status": "OOS_READY" if not out.empty else "DEFERRED",
        "rows": int(len(out)),
        "symbols": int(len(records)),
        "matched_cik_symbols": int(matched),
        "deferred_symbols": int(deferred),
        "unique_forms": sorted(out["form"].astype(str).unique().tolist()) if not out.empty else [],
        "research_only": True,
        "production_changed": False,
        "collected_at": collected_at.isoformat(),
        "started_at": started.isoformat(),
    }
    META.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
