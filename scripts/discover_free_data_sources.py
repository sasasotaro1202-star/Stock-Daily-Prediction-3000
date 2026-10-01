from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

OUT = Path("data/research/source_discovery.json")
UA = "Stock-Daily-Prediction-SourceDiscovery/1.0"
TIMEOUT = int(os.getenv("SOURCE_DISCOVERY_TIMEOUT", "12"))

# Research-only discovery. These sources are candidates, not automatic
# production inputs. Commercial/uncertain-cost providers are excluded.
OFFICIAL_SOURCES = [
    {
        "name": "Yahoo Finance",
        "kind": "price_history",
        "url": "https://finance.yahoo.com/",
        "official": True,
        "free_status": "public_web",
    },
    {
        "name": "U.S. SEC EDGAR",
        "kind": "company_filings",
        "url": "https://www.sec.gov/edgar.shtml",
        "official": True,
        "free_status": "public",
    },
    {
        "name": "FINRA",
        "kind": "market_microstructure",
        "url": "https://www.finra.org/finra-data",
        "official": True,
        "free_status": "public",
    },
    {
        "name": "U.S. Treasury",
        "kind": "rates",
        "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates",
        "official": True,
        "free_status": "public",
    },
    {
        "name": "Federal Reserve Bank of St. Louis FRED",
        "kind": "macro",
        "url": "https://fred.stlouisfed.org/",
        "official": True,
        "free_status": "public",
    },
    {
        "name": "Bank of Japan Time-Series",
        "kind": "macro",
        "url": "https://www.stat-search.boj.or.jp/",
        "official": True,
        "free_status": "public",
    },
]

GITHUB_QUERIES = {
    "price_history": [
        "stock historical data csv",
        "equity historical prices csv",
        "japan stock historical csv",
    ],
    "company_filings": [
        "SEC companyfacts csv",
        "EDGAR financial data csv",
    ],
    "macro": [
        "FRED macroeconomic csv",
        "BOJ time series csv",
    ],
    "market_microstructure": [
        "FINRA short sale volume csv",
        "market microstructure stock csv",
    ],
}

BLOCKED_PROVIDER_TERMS = {
    "bloomberg",
    "factset",
    "lseg",
    "refinitiv",
    "morningstar",
    "sp global",
    "standard and poor",
    "pitchbook",
    "third bridge",
    "polygon.io",
    "alphavantage",
    "alpha vantage",
    "quandl",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_json(url: str, *, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8", "ignore"))


def _health(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "text/html,application/json;q=0.8,*/*;q=0.2",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            return {
                "url": url,
                "reachable": True,
                "status_code": int(response.status),
                "content_type": str(response.headers.get("content-type") or ""),
                "final_url": str(response.geturl()),
            }
    except urllib.error.HTTPError as exc:
        return {
            "url": url,
            "reachable": False,
            "status_code": int(exc.code),
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "url": url,
            "reachable": False,
            "error": repr(exc),
        }


def _blocked(text: str) -> bool:
    normalized = str(text or "").lower()
    return any(term in normalized for term in BLOCKED_PROVIDER_TERMS)


def source_score(candidate: dict, *, now: datetime | None = None) -> float:
    score = 0.0
    if candidate.get("official"):
        score += 0.65
    if candidate.get("reachable"):
        score += 0.15
    if str(candidate.get("url", "")).startswith("https://"):
        score += 0.05
    if candidate.get("has_tabular_hint"):
        score += 0.05
    if candidate.get("recent"):
        score += 0.05
    if candidate.get("pit_hint"):
        score += 0.05
    if _blocked(
        " ".join(
            str(candidate.get(k, ""))
            for k in ("name", "description", "full_name", "url")
        )
    ):
        score -= 1.0
    return round(max(0.0, min(1.0, score)), 6)


def _github_repo_leads() -> list[dict]:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": UA,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    leads: list[dict] = []
    for kind, queries in GITHUB_QUERIES.items():
        for query in queries:
            url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
                {
                    "q": query,
                    "sort": "updated",
                    "order": "desc",
                    "per_page": 8,
                }
            )
            try:
                payload = _get_json(url, headers=headers)
                for item in payload.get("items") or []:
                    updated = str(item.get("updated_at") or "")
                    try:
                        updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                        recent = datetime.now(timezone.utc) - updated_dt <= timedelta(days=180)
                    except Exception:
                        recent = False
                    full_name = str(item.get("full_name") or "")
                    description = str(item.get("description") or "")
                    candidate = {
                        "kind": kind,
                        "query": query,
                        "full_name": full_name,
                        "html_url": item.get("html_url"),
                        "description": description,
                        "updated_at": updated,
                        "fork": bool(item.get("fork")),
                        "archived": bool(item.get("archived")),
                        "recent": recent,
                        "has_tabular_hint": bool(
                            re.search(r"csv|parquet|data|dataset", f"{full_name} {description}", re.I)
                        ),
                        "pit_hint": bool(
                            re.search(r"timestamp|published|available_at|asof|as_of|release", description, re.I)
                        ),
                        "reachable": True,
                        "official": False,
                    }
                    candidate["blocked"] = _blocked(
                        " ".join(
                            str(candidate.get(k, ""))
                            for k in ("full_name", "description", "html_url")
                        )
                    )
                    candidate["score"] = source_score(candidate)
                    leads.append(candidate)
            except Exception as exc:
                leads.append(
                    {
                        "kind": kind,
                        "query": query,
                        "status": "SEARCH_FAILED",
                        "error": repr(exc),
                        "score": 0.0,
                    }
                )
    return leads


def select_for_research(candidates: list[dict], max_per_kind: int = 5) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for candidate in candidates:
        if candidate.get("status") == "SEARCH_FAILED":
            continue
        if candidate.get("blocked"):
            continue
        grouped.setdefault(str(candidate.get("kind", "unknown")), []).append(candidate)

    selected = {}
    for kind, rows in grouped.items():
        rows = sorted(
            rows,
            key=lambda x: (
                float(x.get("score", 0.0)),
                bool(x.get("recent")),
                str(x.get("updated_at", "")),
                str(x.get("full_name", "")),
            ),
            reverse=True,
        )
        seen: set[str] = set()
        unique = []
        for row in rows:
            key = str(row.get("full_name") or row.get("url") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(row)
            if len(unique) >= max_per_kind:
                break
        selected[kind] = unique
    return selected


def main() -> int:
    checked_at = _utc()
    official = []
    for row in OFFICIAL_SOURCES:
        health = _health(row["url"])
        candidate = {**row, **health}
        candidate["pit_hint"] = row["kind"] in {
            "company_filings",
            "market_microstructure",
            "rates",
            "macro",
        }
        candidate["has_tabular_hint"] = True
        candidate["recent"] = True
        candidate["score"] = source_score(candidate)
        official.append(candidate)

    github = _github_repo_leads()
    all_candidates = official + github
    selected = select_for_research(all_candidates)

    failed = sum(1 for row in github if row.get("status") == "SEARCH_FAILED")
    reachable_official = sum(1 for row in official if row.get("reachable"))
    status = "EVALUATED"
    if failed >= len(GITHUB_QUERIES) and reachable_official == 0:
        status = "FAILED"
    elif failed or reachable_official < len(official):
        status = "DEGRADED"

    report = {
        "schema_version": 1,
        "status": status,
        "checked_at_utc": checked_at,
        "free_only": True,
        "research_only": True,
        "production_changed": False,
        "discovery_does_not_adopt": True,
        "retrieval_is_not_historical_pit": True,
        "commercial_sources_require_explicit_review": True,
        "official_sources": official,
        "github_repository_leads": github,
        "selected_for_research": selected,
        "selection_policy": {
            "max_per_kind": 5,
            "sort": ["score", "recent", "updated_at", "full_name"],
            "blocked_provider_terms": sorted(BLOCKED_PROVIDER_TERMS),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
