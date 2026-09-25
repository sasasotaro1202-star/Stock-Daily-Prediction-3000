from __future__ import annotations

import gzip
import json
import os
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
FALLBACK_TICKERS_URL = "https://raw.githubusercontent.com/jadchaar/sec-cik-mapper/7883b83389836f9bba9bdfe53031467235746334/mappings/stocks/ticker_to_cik.json"
ALLOWED_FORMS = {
    "8-K", "10-K", "10-Q", "20-F", "6-K", "40-F",
    "S-1", "S-3", "S-4", "424B2", "DEF 14A", "SC 13D", "SC 13G",
}
USER_AGENT = os.getenv("SEC_USER_AGENT", "Stock-Daily-Prediction-3000/0.1 (+https://github.com/sasasotaro1202-star/Stock-Daily-Prediction-3000)")
REQUEST_TIMEOUT = 25
RATE_SLEEP_SECONDS = 0.15
RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
MAX_REQUEST_ATTEMPTS = 3


def _curl_cffi_get_json(url: str) -> object:
    # SEC can reject the standard-library TLS/client fingerprint from shared
    # CI egress even when the User-Agent is compliant. curl_cffi is already a
    # free project dependency and can impersonate a normal Chrome client.
    from curl_cffi import get as curl_get

    response = curl_get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "From": os.getenv(
                "SEC_CONTACT_EMAIL",
                "262083466+sasasotaro1202-star@users.noreply.github.com",
            ),
        },
        impersonate="chrome",
        timeout=REQUEST_TIMEOUT,
    )
    if int(response.status_code) >= 400:
        raise HTTPError(
            url,
            int(response.status_code),
            getattr(response, "reason", None),
            getattr(response, "headers", None),
            None,
        )
    return response.json()


def _get_json(url: str) -> object:
    last_error: Exception | None = None
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        req = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
                "From": os.getenv(
                    "SEC_CONTACT_EMAIL",
                    "262083466+sasasotaro1202-star@users.noreply.github.com",
                ),
            },
        )
        try:
            with urlopen(req, timeout=REQUEST_TIMEOUT) as response:
                body = response.read()
                if str(response.headers.get("Content-Encoding", "")).lower() == "gzip":
                    body = gzip.decompress(body)
                return json.loads(body.decode("utf-8"))
        except HTTPError as exc:
            last_error = exc
            # A persistent 403 from the GitHub Actions egress can be caused by
            # the HTTP/TLS client fingerprint rather than the declared UA.
            # Switch once to curl_cffi browser impersonation before entering
            # the ordinary bounded retry path. Other 4xx responses remain
            # fail-closed and are not retried.
            if exc.code == 403:
                try:
                    return _curl_cffi_get_json(url)
                except Exception:
                    raise
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
        mapping_source = "sec_official_company_tickers"
        mapping_note = "official"
        try:
            ticker_payload = _get_json(TICKERS_URL)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            # SEC's public ticker mapping can be blocked independently from
            # data.sec.gov submissions. Fall back to a pinned public mapping
            # only for CIK discovery; filing history remains authoritative SEC
            # data and each mapped CIK is validated against the submission
            # payload's ticker list before admission.
            ticker_payload = _get_json(FALLBACK_TICKERS_URL)
            mapping_source = FALLBACK_TICKERS_URL
            mapping_note = "pinned_third_party_fallback"
        ticker_map = {}
        if mapping_source == "sec_official_company_tickers":
            for value in ticker_payload.values() if isinstance(ticker_payload, dict) else []:
                if not isinstance(value, dict):
                    continue
                ticker = _norm_ticker(value.get("ticker"))
                if ticker and value.get("cik_str") is not None:
                    ticker_map[ticker] = {
                        "cik": str(value["cik_str"]).zfill(10),
                        "title": str(value.get("title", "")),
                    }
        else:
            for ticker, cik in ticker_payload.items() if isinstance(ticker_payload, dict) else []:
                ticker = _norm_ticker(ticker)
                if ticker and cik:
                    ticker_map[ticker] = {
                        "cik": str(cik).zfill(10),
                        "title": "",
                    }
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "status": "DEFERRED",
            "reason": f"sec_metadata_fetch_failed:{getattr(exc, 'code', '')}:{type(exc).__name__}",
            "rows": 0,
            "mapping_source": "unavailable",
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
    submission_failures: dict[str, int] = {}
    ticker_mismatches = 0
    for row in sorted(records, key=lambda x: str(x.get("symbol", ""))):
        info = ticker_map.get(_norm_ticker(row.get("symbol")))
        if not info:
            deferred += 1
            continue
        matched += 1
        try:
            payload = _get_json(f"https://data.sec.gov/submissions/CIK{info['cik']}.json")
            payload_tickers = {
                _norm_ticker(value)
                for value in (payload.get("tickers", []) if isinstance(payload, dict) else [])
            }
            if payload_tickers and _norm_ticker(row.get("symbol")) not in payload_tickers:
                ticker_mismatches += 1
                deferred += 1
                continue
            rows.extend(_rows_from_submissions(row, info, payload, collected_at))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            deferred += 1
            key = f"{getattr(exc, 'code', '')}:{type(exc).__name__}"
            submission_failures[key] = submission_failures.get(key, 0) + 1
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
        "mapping_source": mapping_source,
        "mapping_note": mapping_note,
        "submission_failure_counts": submission_failures,
        "ticker_mismatch_count": int(ticker_mismatches),
    }
    META.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
