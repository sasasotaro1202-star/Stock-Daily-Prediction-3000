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


def test_selection_is_deterministic_and_deduplicated():
    rows = [
        {
            "kind": "price_history",
            "full_name": "z/repo",
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
