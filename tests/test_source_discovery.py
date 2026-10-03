from __future__ import annotations

import json
from pathlib import Path

from scripts.discover_free_data_sources import (
    OFFICIAL_SOURCES,
    _blocked,
    select_for_research,
    source_score,
)

ROOT = Path(__file__).resolve().parents[1]


def test_source_discovery_is_free_and_research_only():
    script = (ROOT / "scripts" / "discover_free_data_sources.py").read_text(encoding="utf-8")
    assert '"free_only": True' in script
    assert '"research_only": True' in script
    assert '"production_changed": False' in script
    assert "discovery_does_not_adopt" in script
    assert "retrieval_is_not_historical_pit" in script


def test_blocked_commercial_sources_never_score_as_selected():
    candidate = {
        "full_name": "example/factset-stock-dataset",
        "description": "FactSet historical data",
        "url": "https://example.invalid",
        "reachable": True,
        "official": False,
    }
    assert _blocked(candidate["description"])
    assert source_score(candidate) == 0.0


def test_official_sources_have_explicit_free_status():
    assert OFFICIAL_SOURCES
    assert all(row["official"] for row in OFFICIAL_SOURCES)
    assert all(row["free_status"] in {"public", "public_web"} for row in OFFICIAL_SOURCES)


def test_discovery_lifecycle_is_fail_closed():
    from scripts.discover_free_data_sources import _lifecycle, select_for_research

    verified = {
        "kind": "macro",
        "name": "Public Source",
        "url": "https://example.org/data",
        "free_status": "public",
        "reachable": True,
        "has_tabular_hint": True,
        "pit_hint": True,
        "blocked": False,
    }
    lifecycle = _lifecycle(verified)
    assert lifecycle["stage"] == "DISCOVERED"
    assert lifecycle["eligibility"] == "ELIGIBLE_FOR_RESEARCH_REVIEW"
    assert lifecycle["cost_status"] == "VERIFIED_BY_DECLARATION"
    assert lifecycle["data_feasibility"] == "METADATA_SUPPORTED"
    assert lifecycle["pit_status"] == "UNVERIFIED"
    assert lifecycle["adoption_status"] == "RESEARCH_CANDIDATE_ONLY"

    unknown_cost = dict(verified)
    unknown_cost.pop("free_status")
    unknown_lifecycle = _lifecycle(unknown_cost)
    assert unknown_lifecycle["cost_status"] == "UNCONFIRMED"
    selected = select_for_research(
        [verified, {**unknown_cost, "name": "Unknown Cost"}],
        max_per_kind=5,
    )
    assert [row["name"] for row in selected["macro"]] == ["Public Source"]


def test_github_discovery_uses_html_url_for_https_lifecycle():
    from scripts.discover_free_data_sources import _lifecycle

    github_candidate = {
        "kind": "macro",
        "full_name": "example/public-data",
        "html_url": "https://github.com/example/public-data",
        "free_status": "public",
        "reachable": True,
        "has_tabular_hint": True,
        "pit_hint": True,
        "blocked": False,
    }
    lifecycle = _lifecycle(github_candidate)
    assert lifecycle["eligibility"] == "ELIGIBLE_FOR_RESEARCH_REVIEW"
    assert lifecycle["cost_status"] == "VERIFIED_BY_DECLARATION"
    assert lifecycle["stage"] == "DISCOVERED"

    assert source_score(github_candidate) == 0.30
    selected = select_for_research([github_candidate], max_per_kind=5)
    assert [row["full_name"] for row in selected["macro"]] == ["example/public-data"]


def test_selection_is_deterministic_and_deduplicated():
    rows = [
        {
            "kind": "price_history",
            "full_name": "z/repo",
            "url": "https://example.org/z",
            "free_status": "public",
            "reachable": True,
            "score": 0.8,
            "recent": True,
            "updated_at": "2026-09-30T00:00:00Z",
        },
        {
            "kind": "price_history",
            "full_name": "a/repo",
            "url": "https://example.org/a",
            "free_status": "public",
            "reachable": True,
            "score": 0.8,
            "recent": True,
            "updated_at": "2026-09-30T00:00:00Z",
        },
        {
            "kind": "price_history",
            "full_name": "a/repo",
            "score": 0.8,
            "recent": True,
            "updated_at": "2026-09-30T00:00:00Z",
        },
    ]
    selected = select_for_research(rows, max_per_kind=5)
    assert [row["full_name"] for row in selected["price_history"]] == ["z/repo", "a/repo"]



def test_source_discovery_summary_links_frontier_without_adoption(tmp_path, monkeypatch):
    from scripts import run_daily_research as research

    report = tmp_path / "source_discovery.json"
    report.write_text(
        json.dumps(
            {
                "status": "EVALUATED",
                "checked_at_utc": "2026-10-01T04:00:00+00:00",
                "production_changed": False,
                "discovery_does_not_adopt": True,
                "selected_for_research": {
                    "macro": [{"name": "Public Source"}],
                    "rates": [],
                },
                "discovery_metrics": {
                    "eligible_candidates": 3,
                    "rejected_candidates": 1,
                    "cost_unconfirmed": 2,
                    "pit_unverified": 3,
                    "research_candidates_only": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(research, "SOURCE_DISCOVERY_REPORT", report)

    summary = research._source_discovery_summary()
    assert summary["status"] == "EVALUATED"
    assert summary["selected_for_research_count"] == 1
    assert summary["selected_kinds"] == ["macro"]
    assert summary["cost_unconfirmed"] == 2
    assert summary["pit_unverified"] == 3
    assert summary["selection_is_advisory_only"] is True
    assert summary["production_adoption"] is False
    assert summary["next_best_action"] == "validate_frontier_sources_before_any_adapter"


def test_source_discovery_summary_is_fail_closed_when_missing_or_invalid(tmp_path, monkeypatch):
    from scripts import run_daily_research as research

    report = tmp_path / "source_discovery.json"
    monkeypatch.setattr(research, "SOURCE_DISCOVERY_REPORT", report)

    missing = research._source_discovery_summary()
    assert missing["status"] == "NOT_RUN"
    assert missing["production_adoption"] is False

    report.write_text("{not-json", encoding="utf-8")
    invalid = research._source_discovery_summary()
    assert invalid["status"] == "INVALID"
    assert invalid["next_best_action"] == "rerun_discovery_fail_closed"
    assert invalid["selection_is_advisory_only"] is True


def test_official_source_catalog_covers_additional_security_data_domains():
    kinds = {row["kind"] for row in OFFICIAL_SOURCES}
    assert "fundamentals" in kinds
    assert "capital_flows" in kinds
    assert "market_statistics" in kinds
    assert "filing_timestamps" in kinds
    assert "rates_auction" in kinds
    assert "rates_auction_events" in kinds
    assert "real_rates" in kinds
    assert "rates_detail" in kinds
    assert "real_rates_detail" in kinds


def test_manual_only_jpx_source_is_cataloged_but_not_auto_selected():
    from scripts.discover_free_data_sources import _lifecycle

    source = next(row for row in OFFICIAL_SOURCES if row["kind"] == "market_statistics")
    decorated = {
        **source,
        "reachable": True,
        "has_tabular_hint": True,
        "pit_hint": False,
        "blocked": False,
    }
    lifecycle = _lifecycle(decorated)
    assert lifecycle["automation_status"] == "MANUAL_ONLY"
    selected = select_for_research([decorated], max_per_kind=5)
    assert selected.get("market_statistics", []) == []
