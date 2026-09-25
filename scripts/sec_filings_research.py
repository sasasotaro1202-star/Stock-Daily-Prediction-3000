from __future__ import annotations

import gzip
import json
import os
import time
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

UNIVERSE = Path("data/universe/latest.json")
OUT = Path("data/research/sec_filings_research.parquet")
META = Path("data/research/sec_filings_research.json")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FALLBACK_TICKERS_URL = "https://raw.githubusercontent.com/jadchaar/sec-cik-mapper/7883b83389836f9bba9bdfe53031467235746334/mappings/stocks/ticker_to_cik.json"
EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EFTS_PAGE_SIZE = 100
EFTS_MAX_PAGES = 20
ALLOWED_FORMS = {
    "8-K", "10-K", "10-Q", "20-F", "6-K", "40-F",
    "S-1", "S-3", "S-4", "424B2", "DEF 14A", "SC 13D", "SC 13G",
}
USER_AGENT = os.getenv("SEC_USER_AGENT", "Stock-Daily-Prediction-3000/0.1 (+https://github.com/sasasotaro1202-star/Stock-Daily-Prediction-3000)")
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
            if exc.code not in RETRYABLE_HTTP_CODES or attempt >= MAX_REQUEST_ATTEMPTS:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = min(
                    15.0,
                    max(1.0, float(retry_after)),
                ) if retry_after else float(2 ** (attempt - 1))
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



def _jina_get_json(url: str) -> object:
    """Read an official SEC JSON file through the free Jina Reader fallback."""
    reader_url = "https://r.jina.ai/" + url
    req = Request(
        reader_url,
        headers={
            "User-Agent": "Stock-Daily-Prediction-3000/0.1 (SEC research reader)",
            "Accept": "application/json,text/plain;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(req, timeout=45) as response:
        text_value = response.read().decode("utf-8", errors="replace").strip()
    if text_value.startswith("{") or text_value.startswith("["):
        payload = json.loads(text_value)
    else:
        raise ValueError("Jina SEC JSON response is not raw JSON")
    if not isinstance(payload, (dict, list)):
        raise ValueError("Jina SEC JSON payload has unexpected type")
    return payload

def _curl_cffi_get_json(url: str) -> object:
    """Free browser-like JSON fallback for an official SEC endpoint."""
    from curl_cffi import requests as curl_requests

    response = curl_requests.get(
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
        timeout=REQUEST_TIMEOUT,
        impersonate="chrome",
        allow_redirects=True,
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



def _curl_cffi_get_master_index_text(url: str) -> tuple[str, str]:
    """Retry an official SEC index through curl_cffi's browser-like client."""
    from curl_cffi import requests as curl_requests

    response = curl_requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/plain,text/*;q=0.9,*/*;q=0.8",
            "From": os.getenv(
                "SEC_CONTACT_EMAIL",
                "262083466+sasasotaro1202-star@users.noreply.github.com",
            ),
        },
        timeout=REQUEST_TIMEOUT,
        impersonate="chrome",
        allow_redirects=True,
    )
    if int(response.status_code) >= 400:
        raise HTTPError(
            url,
            int(response.status_code),
            getattr(response, "reason", None),
            getattr(response, "headers", None),
            None,
        )
    body = bytes(response.content)
    if len(body) < 1000:
        raise RuntimeError("SEC master index response unexpectedly small")
    return body.decode("latin-1", errors="replace"), "curl_cffi_chrome"


def _get_master_index_text(url: str) -> tuple[str, str]:
    """Fetch the official SEC master.idx with bounded free fallbacks."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        req = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/plain,text/*;q=0.9,*/*;q=0.8",
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
                return body.decode("latin-1", errors="replace"), "sec_direct"
        except HTTPError as exc:
            last_error = exc
            if exc.code in {400, 403}:
                break
            if exc.code not in RETRYABLE_HTTP_CODES or attempt >= MAX_REQUEST_ATTEMPTS:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = (
                    min(15.0, max(1.0, float(retry_after)))
                    if retry_after
                    else float(2 ** (attempt - 1))
                )
            except (TypeError, ValueError):
                delay = float(2 ** (attempt - 1))
            time.sleep(delay)
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= MAX_REQUEST_ATTEMPTS:
                raise
            time.sleep(float(2 ** (attempt - 1)))

    if isinstance(last_error, HTTPError) and last_error.code in {400, 403}:
        try:
            return _curl_cffi_get_master_index_text(url)
        except Exception as curl_exc:
            last_error = curl_exc

        reader_url = "https://r.jina.ai/" + url
        req = Request(
            reader_url,
            headers={
                "User-Agent": "Stock-Daily-Prediction-3000/0.1 (SEC research reader)",
                "Accept": "text/plain,text/*;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urlopen(req, timeout=45) as response:
                body = response.read()
            text_value = body.decode("utf-8", errors="replace")
            if len(text_value) < 1000:
                raise RuntimeError("Jina SEC master index response unexpectedly small")
            return text_value, "jina_reader_sec_official_url"
        except Exception:
            pass

    if last_error is not None:
        raise last_error
    raise RuntimeError("SEC master index request failed without an exception")



def _quarter_keys(start_date: date, end_date: date) -> list[tuple[int, int]]:
    year = start_date.year
    quarter = ((start_date.month - 1) // 3) + 1
    end_key = (end_date.year, ((end_date.month - 1) // 3) + 1)
    out = []
    while (year, quarter) <= end_key:
        out.append((year, quarter))
        if quarter == 4:
            year += 1
            quarter = 1
        else:
            quarter += 1
    return out


def _accession_from_filename(filename: str) -> str:
    parts = [part for part in str(filename).split("/") if part]
    candidate = parts[-2] if len(parts) >= 2 else ""
    if len(candidate) == 18 and candidate.isdigit():
        return f"{candidate[:10]}-{candidate[10:12]}-{candidate[12:]}"
    return candidate


def _norm_ticker(value: object) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def _ticker_aliases(value: object) -> set[str]:
    raw = str(value or "").strip().upper()
    aliases = {_norm_ticker(raw)}
    for marker in (".US", "-US", ":US", "/US", "_US"):
        if raw.endswith(marker):
            aliases.add(_norm_ticker(raw[: -len(marker)]))
    for marker in ("NASDAQ:", "NYSE:", "AMEX:", "ARCA:"):
        if raw.startswith(marker):
            aliases.add(_norm_ticker(raw[len(marker):]))
    return {alias for alias in aliases if alias}


def _norm_company_name(value: object) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def _resolve_ticker_info(
    ticker: object,
    exact_map: dict[str, dict[str, object]],
    alias_map: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    exact = _norm_ticker(ticker)
    if exact in exact_map:
        return exact_map[exact]
    for alias in _ticker_aliases(ticker):
        info = alias_map.get(alias)
        if info is not None:
            return info
    return None


def _rows_from_master_indexes(
    universe_records: list[dict[str, object]],
    ticker_map: dict[str, dict[str, object]],
    ticker_alias_map: dict[str, dict[str, object]] | None,
    collected_at: pd.Timestamp,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Collect one year of official SEC filing metadata via quarterly master.idx."""
    ticker_alias_map = ticker_alias_map or {}
    start_date = collected_at.to_pydatetime().date() - timedelta(days=365)
    end_date = collected_at.to_pydatetime().date()

    by_cik: dict[str, list[dict[str, object]]] = {}
    by_name_candidates: dict[str, list[dict[str, object]]] = {}
    for record in universe_records:
        info = _resolve_ticker_info(record.get("symbol"), ticker_map, ticker_alias_map)
        if info:
            cik = str(info.get("cik", "")).zfill(10)
            if cik:
                by_cik.setdefault(cik, []).append(record)
        name_key = _norm_company_name(record.get("name"))
        if name_key:
            by_name_candidates.setdefault(name_key, []).append(record)

    by_name = {
        key: values[0]
        for key, values in by_name_candidates.items()
        if len(values) == 1
    }

    rows = []
    diagnostics: dict[str, int] = {}
    tz = ZoneInfo("America/New_York")

    for year, quarter in _quarter_keys(start_date, end_date):
        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"
        body, transport = _get_master_index_text(url)
        quarter_count = 0
        for line in body.splitlines():
            line = line.rstrip("\r\n")
            if (
                not line
                or line.startswith("Description:")
                or line.startswith("Last Data Received:")
                or line.startswith("Comments:")
                or line.startswith("CIK|")
            ):
                continue
            parts = line.split("|", 4)
            if len(parts) != 5:
                continue

            _, company_name, form, filed, filename = parts
            form = form.strip()
            if form not in ALLOWED_FORMS:
                continue

            filed_ts = pd.to_datetime(filed.strip(), errors="coerce")
            if pd.isna(filed_ts):
                continue
            filed_date = filed_ts.date()
            if filed_date < start_date or filed_date > end_date:
                continue

            filename = filename.strip()
            path_parts = [part for part in filename.split("/") if part]
            cik_from_path = ""
            for i, part in enumerate(path_parts):
                if (
                    part == "data"
                    and i + 1 < len(path_parts)
                    and path_parts[i + 1].isdigit()
                ):
                    cik_from_path = str(int(path_parts[i + 1])).zfill(10)
                    break
            if not cik_from_path:
                continue

            matched_records = list(by_cik.get(cik_from_path, []))
            if not matched_records:
                name_key = _norm_company_name(company_name)
                record = by_name.get(name_key)
                if record is not None:
                    matched_records = [record]
            if not matched_records:
                continue

            available_at = pd.Timestamp(
                datetime.combine(
                    filed_date,
                    dt_time(23, 59, 59, 999999),
                    tzinfo=tz,
                )
            ).tz_convert("UTC")

            for record in matched_records:
                ticker_info = _resolve_ticker_info(
                    record.get("symbol"),
                    ticker_map,
                    ticker_alias_map,
                ) or {}
                rows.append({
                    "symbol": str(record.get("symbol", "")),
                    "asset_class": "us_stock",
                    "issuer_name": str(record.get("name", "")),
                    "sec_title": str(ticker_info.get("title", "")),
                    "cik": cik_from_path,
                    "form": form,
                    "filing_date": filed_date.isoformat(),
                    "acceptance_datetime": "",
                    "available_at": available_at.isoformat(),
                    "accession_number": _accession_from_filename(filename),
                    "primary_document": Path(filename).name,
                    "source": "sec_edgar_master_index",
                    "source_url": url,
                    "available_at_method": "filing_date_eod_conservative",
                    "retrieval_transport": transport,
                    "research_only": True,
                    "production_changed": False,
                    "collected_at": collected_at.isoformat(),
                })
                quarter_count += 1
        diagnostics[f"{year}Q{quarter}"] = quarter_count

    return rows, diagnostics



def _rows_from_efts(
    universe_records: list[dict[str, object]],
    ticker_map: dict[str, dict[str, object]],
    ticker_alias_map: dict[str, dict[str, object]],
    collected_at: pd.Timestamp,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Collect filing metadata through the official SEC EFTS JSON endpoint.

    EFTS provides filing date, CIK, form, accession, and document metadata.
    It does not expose acceptanceDateTime, so availability is conservatively
    set to filing-day 23:59:59.999999 ET. CIK filtering is used to avoid
    ambiguous company-name/ticker matches.
    """
    start_date = collected_at.to_pydatetime().date() - timedelta(days=365)
    end_date = collected_at.to_pydatetime().date()
    forms = ",".join(sorted(ALLOWED_FORMS))
    tz = ZoneInfo("America/New_York")
    rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {
        "transport_counts": {},
        "empty_symbols": [],
        "failed_symbols": {},
        "pages": 0,
        "symbols_attempted": 0,
    }

    for record in sorted(universe_records, key=lambda x: str(x.get("symbol", ""))):
        info = _resolve_ticker_info(
            record.get("symbol"),
            ticker_map,
            ticker_alias_map,
        )
        if not info:
            continue
        cik = str(info.get("cik", "")).zfill(10)
        if not cik:
            continue

        diagnostics["symbols_attempted"] = int(diagnostics["symbols_attempted"]) + 1
        symbol_rows: list[dict[str, object]] = []
        offset = 0
        transport_used = ""

        try:
            for _ in range(EFTS_MAX_PAGES):
                params = {
                    "q": "",
                    "dateRange": "custom",
                    "startdt": start_date.isoformat(),
                    "enddt": end_date.isoformat(),
                    "forms": forms,
                    "ciks": cik,
                    "from": offset,
                    "size": EFTS_PAGE_SIZE,
                }
                url = f"{EFTS_SEARCH_URL}?{urlencode(params)}"
                try:
                    payload = _get_json(url)
                    transport_used = "efts_direct"
                except HTTPError as exc:
                    if exc.code != 403:
                        raise
                    payload = _curl_cffi_get_json(url)
                    transport_used = "efts_curl_cffi_chrome"

                if not isinstance(payload, dict):
                    raise ValueError("EFTS response is not a JSON object")
                hit_block = payload.get("hits")
                if not isinstance(hit_block, dict):
                    raise ValueError("EFTS response.hits is not an object")
                hits = hit_block.get("hits", [])
                if not isinstance(hits, list):
                    raise ValueError("EFTS response.hits.hits is not a list")

                diagnostics["pages"] = int(diagnostics["pages"]) + 1
                if not hits:
                    break

                for hit in hits:
                    if not isinstance(hit, dict):
                        continue
                    source = hit.get("_source", {})
                    if not isinstance(source, dict):
                        continue

                    hit_ciks = {
                        str(value).zfill(10)
                        for value in (source.get("ciks") or [])
                        if str(value).strip()
                    }
                    if hit_ciks and cik not in hit_ciks:
                        continue

                    form = str(source.get("form", "")).strip()
                    if form not in ALLOWED_FORMS:
                        continue
                    filed = pd.to_datetime(
                        str(source.get("file_date", "")),
                        errors="coerce",
                    )
                    if pd.isna(filed):
                        continue
                    filed_date = filed.date()
                    if filed_date < start_date or filed_date > end_date:
                        continue

                    accession = str(source.get("adsh", "")).strip()
                    hit_id = str(hit.get("_id", ""))
                    if not accession:
                        accession = hit_id.split(":", 1)[0]
                    if not accession:
                        continue

                    filename = hit_id.split(":", 1)[1] if ":" in hit_id else ""
                    available_at = pd.Timestamp(
                        datetime.combine(
                            filed_date,
                            dt_time(23, 59, 59, 999999),
                            tzinfo=tz,
                        )
                    ).tz_convert("UTC")

                    symbol_rows.append({
                        "symbol": str(record.get("symbol", "")),
                        "asset_class": "us_stock",
                        "issuer_name": str(record.get("name", "")),
                        "sec_title": str(info.get("title", "")),
                        "cik": cik,
                        "form": form,
                        "filing_date": filed_date.isoformat(),
                        "acceptance_datetime": "",
                        "available_at": available_at.isoformat(),
                        "accession_number": accession,
                        "primary_document": filename or accession,
                        "source": "sec_edgar_efts",
                        "source_url": (
                            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                            f"{accession.replace('-', '')}/{filename}"
                            if filename
                            else f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                            f"{accession.replace('-', '')}/"
                        ),
                        "available_at_method": "filing_date_eod_conservative",
                        "retrieval_transport": transport_used,
                        "research_only": True,
                        "production_changed": False,
                        "collected_at": collected_at.isoformat(),
                    })

                if len(hits) < EFTS_PAGE_SIZE:
                    break
                offset += EFTS_PAGE_SIZE

            if symbol_rows:
                rows.extend(symbol_rows)
            else:
                diagnostics["empty_symbols"].append(str(record.get("symbol", "")))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            failures = diagnostics["failed_symbols"]
            assert isinstance(failures, dict)
            failures[str(record.get("symbol", ""))] = (
                f"{getattr(exc, 'code', '')}:{type(exc).__name__}"
            )

        if transport_used:
            counts = diagnostics["transport_counts"]
            assert isinstance(counts, dict)
            counts[transport_used] = int(counts.get(transport_used, 0)) + 1

        time.sleep(RATE_SLEEP_SECONDS)

    return rows, diagnostics



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
            try:
                # Same official SEC mapping via the free Jina Reader fallback.
                ticker_payload = _jina_get_json(TICKERS_URL)
                mapping_source = TICKERS_URL
                mapping_note = "official_via_jina_reader"
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
                # Third-party pinned CIK map is used only as the final discovery fallback.
                ticker_payload = _get_json(FALLBACK_TICKERS_URL)
                mapping_source = FALLBACK_TICKERS_URL
                mapping_note = "pinned_third_party_fallback"
        ticker_map = {}
        ticker_alias_map = {}
        if mapping_source == "sec_official_company_tickers":
            items = ticker_payload.values() if isinstance(ticker_payload, dict) else []
            for value in items:
                if not isinstance(value, dict):
                    continue
                raw_ticker = str(value.get("ticker", ""))
                exact = _norm_ticker(raw_ticker)
                if exact and value.get("cik_str") is not None:
                    info = {
                        "cik": str(value["cik_str"]).zfill(10),
                        "title": str(value.get("title", "")),
                    }
                    ticker_map[exact] = info
                    for alias in _ticker_aliases(raw_ticker):
                        if alias == exact:
                            continue
                        prior = ticker_alias_map.get(alias)
                        if prior is not None and prior.get("cik") != info["cik"]:
                            ticker_alias_map.pop(alias, None)
                        elif alias not in ticker_alias_map:
                            ticker_alias_map[alias] = info
        else:
            items = ticker_payload.items() if isinstance(ticker_payload, dict) else []
            for raw_ticker, cik in items:
                exact = _norm_ticker(raw_ticker)
                if exact and cik:
                    info = {
                        "cik": str(cik).zfill(10),
                        "title": "",
                    }
                    ticker_map[exact] = info
                    for alias in _ticker_aliases(raw_ticker):
                        if alias == exact:
                            continue
                        prior = ticker_alias_map.get(alias)
                        if prior is not None and prior.get("cik") != info["cik"]:
                            ticker_alias_map.pop(alias, None)
                        elif alias not in ticker_alias_map:
                            ticker_alias_map[alias] = info
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
    bulk_index_diagnostics: dict[str, int] = {}
    bulk_index_used = False
    for row in sorted(records, key=lambda x: str(x.get("symbol", ""))):
        info = _resolve_ticker_info(row.get("symbol"), ticker_map, ticker_alias_map)
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
            key = f"{getattr(exc, 'code', '')}:{type(exc).__name__}"
            submission_failures[key] = submission_failures.get(key, 0) + 1
            if getattr(exc, "code", None) == 403:
                bulk_index_used = True
                break
            deferred += 1
        time.sleep(RATE_SLEEP_SECONDS)

    efts_used = False
    efts_diagnostics: dict[str, object] = {}
    if bulk_index_used:
        try:
            efts_rows, efts_diagnostics = _rows_from_efts(
                records,
                ticker_map,
                ticker_alias_map,
                collected_at,
            )
            if efts_rows:
                efts_used = True
                rows.extend(efts_rows)
                mapped_symbols = {str(row["symbol"]) for row in efts_rows}
                deferred = len(records) - len(mapped_symbols)
            else:
                bulk_rows, bulk_index_diagnostics = _rows_from_master_indexes(
                    records,
                    ticker_map,
                    ticker_alias_map,
                    collected_at,
                )
                rows.extend(bulk_rows)
                mapped_symbols = {str(row["symbol"]) for row in bulk_rows}
                deferred = len(records) - len(mapped_symbols)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            key = f"efts:{getattr(exc, 'code', '')}:{type(exc).__name__}"
            submission_failures[key] = submission_failures.get(key, 0) + 1
            try:
                bulk_rows, bulk_index_diagnostics = _rows_from_master_indexes(
                    records,
                    ticker_map,
                    ticker_alias_map,
                    collected_at,
                )
                rows.extend(bulk_rows)
                mapped_symbols = {str(row["symbol"]) for row in bulk_rows}
                deferred = len(records) - len(mapped_symbols)
            except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as archive_exc:
                key = f"bulk_index:{getattr(archive_exc, 'code', '')}:{type(archive_exc).__name__}"
                submission_failures[key] = submission_failures.get(key, 0) + 1


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
        "ticker_alias_matches": int(
            sum(
                1
                for row in records
                if _norm_ticker(row.get("symbol")) not in ticker_map
                and _resolve_ticker_info(row.get("symbol"), ticker_map, ticker_alias_map) is not None
            )
        ),
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
        "bulk_index_used": bool(bulk_index_used),
        "efts_used": bool(efts_used),
        "efts_diagnostics": efts_diagnostics,
        "bulk_index_quarters": bulk_index_diagnostics,
        "bulk_index_retrieval_transports": sorted(
            {
                str(row.get("retrieval_transport", ""))
                for row in rows
                if row.get("source") == "sec_edgar_master_index"
                and row.get("retrieval_transport")
            }
        ),
        "available_at_method": (
            "acceptance_datetime"
            if not bulk_index_used
            else "filing_date_eod_conservative"
        ),
    }
    META.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
