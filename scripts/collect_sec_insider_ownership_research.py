from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import yaml

CONFIG = Path("config/research_data_sources.yml")
UNIVERSE = Path("data/universe/latest.json")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
OUT_DIR = Path("data/research/sec_insider_ownership")
UA = os.getenv(
    "SEC_USER_AGENT",
    "Stock-Daily-Prediction-3000/1.0 insider-ownership-research",
)
MAX_ATTEMPTS = max(1, int(os.getenv("SEC_INSIDER_MAX_ATTEMPTS", "3")))
SLEEP_SECONDS = max(0.0, float(os.getenv("SEC_INSIDER_SLEEP_SECONDS", "0.20")))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_bytes(url: str) -> tuple[bytes, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = Request(
            url,
            headers={
                "User-Agent": UA,
                "Accept": "application/xml,text/xml,text/html,*/*",
                "Accept-Encoding": "gzip, deflate",
            },
        )
        try:
            with urlopen(request, timeout=45) as response:
                raw = response.read()
                headers = {
                    "etag": str(response.headers.get("ETag") or ""),
                    "last_modified": str(response.headers.get("Last-Modified") or ""),
                    "content_type": str(response.headers.get("Content-Type") or ""),
                }
            if not raw:
                raise ValueError("SEC response body is empty")
            return raw, headers
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_ATTEMPTS:
                raise
            time.sleep(min(8.0, 2 ** (attempt - 1)))
    raise RuntimeError(f"SEC request failed: {last_error!r}")


def _get_json(url: str) -> tuple[dict, dict[str, str]]:
    raw, headers = _get_bytes(url)
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("SEC JSON root is not an object")
    return payload, headers


def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}


def _load_us_universe() -> list[str]:
    payload = json.loads(UNIVERSE.read_text(encoding="utf-8"))
    symbols = {
        str(row.get("symbol") or "").strip().upper()
        for row in payload.get("records", [])
        if str(row.get("asset_class") or "").strip().lower() == "us_stock"
        and bool(row.get("tradeable", True))
    }
    return sorted(symbol for symbol in symbols if symbol)


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
            }
    return result


def _recent_rows(payload: dict) -> list[dict[str, object]]:
    recent = ((payload.get("filings") or {}).get("recent") or {})
    if not isinstance(recent, dict):
        return []
    keys = [k for k, v in recent.items() if isinstance(v, list)]
    n = max((len(recent[k]) for k in keys), default=0)
    return [
        {key: recent[key][idx] for key in keys if idx < len(recent[key])}
        for idx in range(n)
    ]


def _candidate_rows(
    submissions: dict,
    forms: set[str],
    cutoff_date: datetime.date,
    max_filings: int,
) -> list[dict[str, object]]:
    selected = []
    for row in _recent_rows(submissions):
        form = str(row.get("form") or "").strip()
        if form not in forms:
            continue
        filing_date_raw = str(row.get("filingDate") or "").strip()
        try:
            filing_date = datetime.strptime(filing_date_raw, "%Y-%m-%d").date()
        except ValueError:
            continue
        if filing_date < cutoff_date:
            continue
        row["_filing_date_obj"] = filing_date
        selected.append(row)
    selected.sort(
        key=lambda row: (
            str(row.get("acceptanceDateTime") or ""),
            str(row.get("filingDate") or ""),
            str(row.get("accessionNumber") or ""),
        ),
        reverse=True,
    )
    return selected[:max_filings]


def _archive_url(cik: str, accession: str, primary_document: str) -> str:
    accession_compact = accession.replace("-", "")
    return (
        f"{ARCHIVE_BASE}/{int(cik)}/{accession_compact}/"
        f"{quote(primary_document, safe='/._-')}"
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _form4_transaction_view(raw: bytes) -> dict[str, object]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return {"xml_parse_status": "NOT_XML", "transactions": []}

    owner = None
    for elem in root.iter():
        if _local_name(elem.tag) == "rptOwnerName" and elem.text:
            owner = elem.text.strip()
            break

    transactions = []
    for tx in root.iter():
        if _local_name(tx.tag) != "nonDerivativeTransaction":
            continue
        item = {}
        allowed = {
            "transactionDate",
            "transactionCode",
            "transactionShares",
            "transactionPricePerShare",
            "sharesOwnedFollowingTransaction",
            "directOrIndirectOwnership",
        }
        for child in tx.iter():
            name = _local_name(child.tag)
            if name not in allowed:
                continue
            value_nodes = [
                descendant
                for descendant in child.iter()
                if _local_name(descendant.tag) == "value"
            ]
            value = next(
                (
                    descendant.text.strip()
                    for descendant in value_nodes
                    if descendant.text and descendant.text.strip()
                ),
                child.text.strip() if child.text else "",
            )
            if value:
                item[name] = value
        if item:
            transactions.append(item)
    return {
        "xml_parse_status": "PARSED",
        "reporting_owner_name": owner,
        "transactions": transactions,
    }


def main() -> int:
    if not UNIVERSE.exists():
        raise SystemExit(f"FAIL: missing official universe: {UNIVERSE}")

    cfg = _load_cfg().get("sec_insider_ownership") or {}
    if cfg.get("enabled") is False:
        raise SystemExit("DEFERRED: sec_insider_ownership disabled")

    ticker_payload, ticker_headers = _get_json(TICKERS_URL)
    mapping = _ticker_map(ticker_payload)
    symbols = _load_us_universe()

    shard_count = max(1, int(os.getenv(
        "SEC_INSIDER_SHARD_COUNT", cfg.get("shard_count", 4)
    )))
    shard_index = int(os.getenv("SEC_INSIDER_SHARD_INDEX", "0"))
    if not 0 <= shard_index < shard_count:
        raise SystemExit("FAIL: SEC_INSIDER_SHARD_INDEX out of range")

    max_symbols = max(1, int(os.getenv(
        "SEC_INSIDER_MAX_SYMBOLS_PER_SHARD",
        cfg.get("max_symbols_per_shard", 75),
    )))
    selected_symbols = [
        s for i, s in enumerate(symbols) if i % shard_count == shard_index
    ][:max_symbols]

    forms = {
        str(x).strip()
        for x in cfg.get("forms", ["4", "4/A", "13D", "13D/A", "13G", "13G/A"])
        if str(x).strip()
    }
    lookback_days = max(1, int(cfg.get("lookback_days", 180)))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date()
    submissions_template = str(
        cfg.get("submissions_url_template") or DEFAULT_SUBMISSIONS
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = []
    filings_written = 0
    manifest_rows = []

    for symbol in selected_symbols:
        info = mapping.get(symbol)
        if not info:
            failures.append({"symbol": symbol, "reason": "SEC_TICKER_NOT_FOUND"})
            continue

        try:
            submissions, submission_headers = _get_json(
                submissions_template.format(cik=info["cik"])
            )
            candidates = _candidate_rows(
                submissions,
                forms,
                cutoff,
                max(1, int(cfg.get("max_filings_per_symbol", 8))),
            )
            symbol_dir = OUT_DIR / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)

            written_for_symbol = 0
            for row in candidates:
                accession = str(row.get("accessionNumber") or "").strip()
                primary_document = str(row.get("primaryDocument") or "").strip()
                if not accession or not primary_document:
                    continue
                url = _archive_url(info["cik"], accession, primary_document)
                raw, headers = _get_bytes(url)

                ext = Path(primary_document).suffix.lower() or ".bin"
                safe_accession = accession.replace("-", "")
                raw_path = symbol_dir / f"{safe_accession}{ext}"
                raw_path.write_bytes(raw)

                metadata = {
                    "symbol": symbol,
                    "cik": info["cik"],
                    "form": row.get("form"),
                    "filing_date": row.get("filingDate"),
                    "acceptance_datetime": row.get("acceptanceDateTime"),
                    "report_date": row.get("reportDate"),
                    "accession_number": accession,
                    "primary_document": primary_document,
                    "archive_url": url,
                    "retrieved_at": _utc(),
                    "raw_path": str(raw_path),
                    "response_headers": headers,
                    "submission_response_headers": submission_headers,
                    "pit_status": "UNVERIFIED",
                    "exact_publication_timestamp_verified": False,
                    "source_available_at": None,
                    "source_available_at_method": "not_inferred",
                    "research_only": True,
                    "production_changed": False,
                }

                if str(row.get("form") or "") in {"4", "4/A"}:
                    metadata["form4_view"] = _form4_transaction_view(raw)

                meta_path = symbol_dir / f"{safe_accession}.meta.json"
                meta_path.write_text(
                    json.dumps(
                        metadata,
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                written_for_symbol += 1
                filings_written += 1
                time.sleep(SLEEP_SECONDS)

            manifest_rows.append(
                {
                    "symbol": symbol,
                    "cik": info["cik"],
                    "candidate_filings": len(candidates),
                    "written_filings": written_for_symbol,
                    "submissions_last_modified": submission_headers.get("last_modified"),
                    "retrieved_at": _utc(),
                }
            )
            successes += 1
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
            if selected_symbols and not failures
            else ("DEGRADED" if successes else "FAILED")
        ),
        "retrieved_at": _utc(),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "source": "U.S. SEC EDGAR insider and beneficial ownership filings",
        "source_url": "https://www.sec.gov/edgar/",
        "forms": sorted(forms),
        "lookback_days": lookback_days,
        "ticker_master": {
            "url": TICKERS_URL,
            "etag": ticker_headers.get("etag"),
            "last_modified": ticker_headers.get("last_modified"),
        },
        "shard": {
            "index": shard_index,
            "count": shard_count,
            "selected_symbols": len(selected_symbols),
        },
        "summary": {
            "successful_symbols": successes,
            "failed_symbols": len(failures),
            "filings_written": filings_written,
        },
        "pit_status": "UNVERIFIED",
        "pit_policy": {
            "acceptance_datetime_preserved": True,
            "transaction_or_schedule_event_date_preserved": True,
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
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if selected_symbols and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
