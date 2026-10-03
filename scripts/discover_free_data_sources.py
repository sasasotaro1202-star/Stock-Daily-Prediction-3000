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
        "name": "U.S. SEC EDGAR XBRL Company Facts",
        "kind": "fundamentals",
        "url": "https://data.sec.gov/api/xbrl/companyfacts/",
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
    {
        "name": "Statistics Dashboard (e-Stat)",
        "kind": "macro",
        "url": "https://dashboard.e-stat.go.jp/",
        "official": True,
        "free_status": "public",
        "access_mode": "api_no_registration",
    },
    {
        "name": "Japan Ministry of Finance Securities Transactions",
        "kind": "capital_flows",
        "url": "https://www.mof.go.jp/policy/international_policy/reference/itn_transactions_in_securities/montha1.csv",
        "official": True,
        "free_status": "public",
        "access_mode": "direct_csv",
    },
    {
        "name": "U.S. SEC EDGAR Submissions API",
        "kind": "filing_timestamps",
        "url": "https://data.sec.gov/submissions/",
        "official": True,
        "free_status": "public",
        "access_mode": "api_no_auth",
    },
    {
        "name": "U.S. Treasury Securities Auctions Data",
        "kind": "rates_auction",
        "url": "https://fiscaldata.treasury.gov/datasets/treasury-securities-auctions-data/",
        "official": True,
        "free_status": "public",
        "access_mode": "api_no_auth",
    },
    {
        "name": "U.S. Treasury Upcoming Auctions",
        "kind": "rates_auction_events",
        "url": "https://fiscaldata.treasury.gov/datasets/treasury-securities-auctions-data/",
        "official": True,
        "free_status": "public",
        "access_mode": "api_no_auth",
    },
    {
        "name": "U.S. Treasury Daily Real Yield Curve Rates",
        "kind": "real_rates",
        "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates",
        "official": True,
        "free_status": "public_web",
        "access_mode": "api_or_feed",
    },
    {
        "name": "U.S. Treasury Daily Bill Rates",
        "kind": "rates_detail",
        "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates",
        "official": True,
        "free_status": "public_web",
        "access_mode": "api_or_feed",
    },
    {
        "name": "U.S. Treasury Daily Long-Term Rates",
        "kind": "rates_detail",
        "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates",
        "official": True,
        "free_status": "public_web",
        "access_mode": "api_or_feed",
    },
    {
        "name": "U.S. Treasury Daily Real Long-Term Rates",
        "kind": "real_rates_detail",
        "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates",
        "official": True,
        "free_status": "public_web",
        "access_mode": "api_or_feed",
    },
    {
        "name": "Japan MOF Foreign Exchange Intervention Operations",
        "kind": "fx_intervention",
        "url": "https://www.mof.go.jp/policy/international_policy/reference/feio/foreign_exchange_intervention_operations.csv",
        "official": True,
        "free_status": "public",
        "access_mode": "direct_csv",
    },
    {
        "name": "Japan MOF Official Reserve Assets",
        "kind": "official_reserves",
        "url": "https://www.mof.go.jp/policy/international_policy/reference/official_reserve_assets/historical.csv",
        "official": True,
        "free_status": "public",
        "access_mode": "direct_csv",
    },
    {
        "name": "Bank of Japan Monetary Base and Current Account Balances",
        "kind": "boj_money",
        "url": "https://www.stat-search.boj.or.jp/",
        "official": True,
        "free_status": "public",
        "access_mode": "api_no_registration",
    },
    {
        "name": "Bank of Japan Balance of Payments Time Series",
        "kind": "boj_balance_of_payments",
        "url": "https://www.stat-search.boj.or.jp/",
        "official": True,
        "free_status": "public",
        "access_mode": "api_no_registration",
    },
    {
        "name": "Bank of Japan TANKAN Time Series",
        "kind": "boj_tankan",
        "url": "https://www.stat-search.boj.or.jp/",
        "official": True,
        "free_status": "public",
        "access_mode": "api_no_registration",
    },
    {
        "name": "Federal Reserve / FRED Public Macro and Financial Series",
        "kind": "fed_cross_asset",
        "url": "https://fred.stlouisfed.org/",
        "official": True,
        "free_status": "public",
        "access_mode": "direct_csv",
    },
    {
        "name": "Tokyo Stock Exchange Daily Bulletin",
        "kind": "market_statistics",
        "url": "https://www.jpx.co.jp/markets/statistics-equities/daily/03.html",
        "official": True,
        "free_status": "public_web",
        "access_mode": "manual_only",
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
    "fundamentals": [
        "SEC companyfacts XBRL fundamentals",
        "SEC companyconcept financial statements",
    ],
    "capital_flows": [
        "Japan securities transactions capital flows CSV",
        "foreign portfolio investment Japan statistics CSV",
    ],
    "filing_timestamps": [
        "SEC EDGAR submissions acceptance datetime",
        "SEC filing acceptance timestamp XBRL",
    ],
    "rates_auction": [
        "U.S. Treasury securities auction bid to cover data",
        "Treasury auction results CSV JSON",
    ],
    "rates_auction_events": [
        "U.S. Treasury upcoming auctions schedule",
        "Treasury auction announcement schedule",
    ],
    "real_rates": [
        "U.S. Treasury real yield curve rates",
        "Treasury TIPS real yield daily",
    ],
    "rates_detail": [
        "U.S. Treasury daily bill rates",
        "U.S. Treasury long term rates XML",
    ],
    "real_rates_detail": [
        "U.S. Treasury real long term rates",
        "Treasury real long-term interest rates XML",
    ],
    "fx_intervention": [
        "Japan MOF foreign exchange intervention operations CSV",
        "foreign exchange intervention historical Japan",
    ],
    "official_reserves": [
        "Japan MOF official reserve assets historical CSV",
        "Japan foreign exchange reserves historical data",
    ],
    "boj_money": [
        "BOJ monetary base current account balances API",
        "Bank of Japan money deposits time series",
    ],
    "boj_balance_of_payments": [
        "BOJ balance of payments time series API",
        "Bank of Japan financial account net balance",
    ],
    "boj_tankan": [
        "BOJ TANKAN business conditions API",
        "Bank of Japan TANKAN time series",
    ],
    "fed_cross_asset": [
        "Federal Reserve FRED financial conditions SOFR dollar index",
        "FRED monetary financial stress macro time series",
    ],
    "market_statistics": [
        "Tokyo Stock Exchange daily bulletin CSV",
        "JPX market statistics equities",
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

# Discovery is deliberately weaker than adoption. A candidate can be useful
# enough to investigate while still having unresolved cost, access, or PIT
# evidence. These states are persisted so downstream research cannot silently
# promote a discovery result into an input source.
FREE_STATUS_VALUES = {"public", "public_web", "free", "free_registration"}
CONDITIONAL_FREE_STATUS_VALUES = {"free_noncommercial", "optional"}


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


def _lifecycle(candidate: dict) -> dict:
    """Return fail-closed discovery lifecycle metadata without claiming adoption."""
    blocked = bool(candidate.get("blocked"))
    free_status = str(candidate.get("free_status") or "").strip().lower()
    source_url = str(candidate.get("url") or candidate.get("html_url") or "").strip()
    if blocked:
        eligibility = "REJECTED_BLOCKED_PROVIDER"
    elif not source_url.startswith("https://"):
        eligibility = "REJECTED_NON_HTTPS"
    else:
        eligibility = "ELIGIBLE_FOR_RESEARCH_REVIEW"

    if free_status in FREE_STATUS_VALUES:
        cost_status = "VERIFIED_BY_DECLARATION"
    elif free_status in CONDITIONAL_FREE_STATUS_VALUES:
        cost_status = "CONDITIONALLY_FREE_REQUIRES_POLICY_REVIEW"
    else:
        cost_status = "UNCONFIRMED"

    data_status = (
        "METADATA_SUPPORTED"
        if candidate.get("reachable") and candidate.get("has_tabular_hint")
        else "UNVERIFIED"
    )

    # Discovery does not retrieve historical release/availability lineage.
    # Therefore PIT remains explicitly unverified and cannot advance the stage.
    pit_status = "UNVERIFIED"
    stage = "REJECTED" if blocked else "DISCOVERED"

    automation_status = (
        "MANUAL_ONLY"
        if str(candidate.get("access_mode") or "").strip().lower() == "manual_only"
        else "AUTOMATABLE_CANDIDATE"
    )

    return {
        "stage": stage,
        "eligibility": eligibility,
        "cost_status": cost_status,
        "data_feasibility": data_status,
        "pit_status": pit_status,
        "automation_status": automation_status,
        "adoption_status": "RESEARCH_CANDIDATE_ONLY",
        "next_test": (
            "verify_license_cost_access_and_pit_lineage"
            if not blocked
            else "no_further_research_unless_policy_changes"
        ),
    }


def _decorate_candidate(candidate: dict) -> dict:
    row = dict(candidate)
    row["lifecycle"] = _lifecycle(row)
    return row


def source_score(candidate: dict, *, now: datetime | None = None) -> float:
    score = 0.0
    if candidate.get("official"):
        score += 0.65
    if candidate.get("reachable"):
        score += 0.15
    source_url = str(candidate.get("url") or candidate.get("html_url") or "")
    if source_url.startswith("https://"):
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
        lifecycle = candidate.get("lifecycle") or _lifecycle(candidate)
        if lifecycle.get("eligibility") != "ELIGIBLE_FOR_RESEARCH_REVIEW":
            continue
        # Keep discovery broad, but only place explicitly free/public sources
        # into the research shortlist. Unknown cost is retained as a frontier
        # lead and must be manually/policy validated before acquisition.
        if lifecycle.get("cost_status") != "VERIFIED_BY_DECLARATION":
            continue
        if lifecycle.get("automation_status") == "MANUAL_ONLY":
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
    all_candidates = [_decorate_candidate(row) for row in (official + github)]
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
        "official_sources": [
            _decorate_candidate(row) for row in official
        ],
        "github_repository_leads": [
            _decorate_candidate(row) for row in github
        ],
        "selected_for_research": selected,
        "discovery_metrics": {
            "total_candidates": len(all_candidates),
            "eligible_candidates": sum(
                1
                for row in all_candidates
                if row["lifecycle"]["eligibility"] == "ELIGIBLE_FOR_RESEARCH_REVIEW"
            ),
            "rejected_candidates": sum(
                1 for row in all_candidates if row["lifecycle"]["stage"] == "REJECTED"
            ),
            "cost_unconfirmed": sum(
                1 for row in all_candidates
                if row["lifecycle"]["cost_status"] == "UNCONFIRMED"
            ),
            "pit_unverified": sum(
                1 for row in all_candidates
                if row["lifecycle"]["pit_status"] == "UNVERIFIED"
            ),
            "research_candidates_only": sum(
                1
                for row in all_candidates
                if row["lifecycle"]["adoption_status"] == "RESEARCH_CANDIDATE_ONLY"
            ),
        },
        "lifecycle_policy": {
            "ladder": [
                "DISCOVERED",
                "METADATA_CHECKED",
                "DATA_FEASIBLE",
                "PIT_VALIDATED",
                "SHADOW",
                "OOS_ROBUSTNESS_VALIDATED",
                "LIMITED_PRODUCTION",
                "STABLE_PRODUCTION",
            ],
            "discovery_never_adopts": True,
            "pit_unverified_is_fail_closed": True,
            "cost_unconfirmed_is_not_free_verified": True,
            "manual_only_sources_are_cataloged_but_not_auto_selected": True,
        },
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
