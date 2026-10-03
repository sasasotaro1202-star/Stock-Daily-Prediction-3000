from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml


CONFIG = Path("config/research_data_sources.yml")
OUT = Path("data/research/treasury_auctions_research.json")
USER_AGENT = "Stock-Daily-Prediction-3000/1.0 treasury-auction-research"
DEFAULT_AUCTIONS_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
DEFAULT_UPCOMING_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/upcoming_auctions"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_json(url: str, params: dict[str, str]) -> tuple[dict, dict[str, str]]:
    query = urlencode(params)
    request = Request(
        url + ("?" + query if query else ""),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
        headers = {
            "etag": str(response.headers.get("ETag") or ""),
            "last_modified": str(response.headers.get("Last-Modified") or ""),
            "content_type": str(response.headers.get("Content-Type") or ""),
        }
    if not isinstance(payload, dict):
        raise ValueError("Treasury Fiscal Data response root is not an object")
    return payload, headers


def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}


def _extract_publication_rows(payload: dict) -> list[dict]:
    rows = []
    for row in payload.get("data", []) or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "record_date": row.get("record_date"),
                "publication_date": row.get("record_date"),
                "auction_date": row.get("auction_date"),
                "issue_date": row.get("issue_date"),
                "security_type": row.get("security_type"),
                "security_term": row.get("security_term"),
                "offering_amt": row.get("offering_amt"),
                "total_accepted": row.get("total_accepted"),
                "bid_to_cover_ratio": row.get("bid_to_cover_ratio"),
                "high_yield": row.get("high_yield"),
                "int_rate": row.get("int_rate"),
                "indirect_bidder_accepted": row.get("indirect_bidder_accepted"),
                "cusip": row.get("cusip"),
            }
        )
    return rows


def main() -> int:
    cfg = _load_cfg().get("treasury_auctions") or {}
    if cfg.get("enabled") is False:
        raise SystemExit("DEFERRED: treasury_auctions disabled")

    now = datetime.now(timezone.utc)
    lookback_years = max(1, int(cfg.get("lookback_years", 5)))
    try:
        start = now.replace(year=now.year - lookback_years)
    except ValueError:
        start = now - timedelta(days=lookback_years * 365)
    start_date = start.date().isoformat()

    auctions_url = str(cfg.get("auctions_url") or DEFAULT_AUCTIONS_URL)
    upcoming_url = str(cfg.get("upcoming_url") or DEFAULT_UPCOMING_URL)

    auction_params = {
        "filter": f"record_date:gte:{start_date}",
        "sort": "-record_date",
        "page[size]": "5000",
    }
    upcoming_params = {
        "sort": "auction_date",
        "page[size]": "200",
    }

    auctions_payload, auction_headers = _get_json(auctions_url, auction_params)
    upcoming_payload, upcoming_headers = _get_json(upcoming_url, upcoming_params)

    auctions_view = _extract_publication_rows(auctions_payload)
    upcoming_view = [
        {
            "auction_date": row.get("auction_date"),
            "security_type": row.get("security_type"),
            "security_term": row.get("security_term"),
            "offering_amt": row.get("offering_amt"),
            "record_date": row.get("record_date"),
            "publication_date": row.get("record_date"),
        }
        for row in upcoming_payload.get("data", []) or []
        if isinstance(row, dict)
    ]

    payload = {
        "schema_version": 1,
        "status": "EVALUATED",
        "retrieved_at": _utc(),
        "research_only": True,
        "production_changed": False,
        "promotion_allowed": False,
        "source": "U.S. Treasury Fiscal Data",
        "sources": {
            "auctions": {
                "url": auctions_url,
                "headers": auction_headers,
                "lookback_start": start_date,
                "rows": len(auctions_view),
            },
            "upcoming": {
                "url": upcoming_url,
                "headers": upcoming_headers,
                "rows": len(upcoming_view),
            },
        },
        "pit_status": "UNVERIFIED",
        "pit_policy": {
            "record_date_is_publication_date": True,
            "exact_publication_time_verified": False,
            "source_available_at_not_inferred": True,
            "fail_closed_for_production": True,
        },
        "auctions": {
            "raw": auctions_payload,
            "publication_view": auctions_view,
        },
        "upcoming_auctions": {
            "raw": upcoming_payload,
            "event_view": upcoming_view,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "research_only": True,
        "production_changed": False,
        "pit_status": "UNVERIFIED",
        "auction_rows": len(auctions_view),
        "upcoming_rows": len(upcoming_view),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
