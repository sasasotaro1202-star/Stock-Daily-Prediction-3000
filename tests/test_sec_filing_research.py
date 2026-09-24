from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd


def test_sec_rows_use_acceptance_time_as_available_at():
    mod = importlib.import_module("scripts.sec_filings_research")
    collected = pd.Timestamp("2026-09-24T00:00:00Z")
    row = {"symbol": "TEST", "asset_class": "us_stock", "name": "Test Co", "tradeable": True}
    info = {"cik": "0000000001", "title": "Test Co"}
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000000001-26-000001"],
                "filingDate": ["2026-09-23"],
                "acceptanceDateTime": ["2026-09-23T20:15:00.000Z"],
                "form": ["8-K"],
                "primaryDocument": ["test.htm"],
            }
        }
    }
    rows = mod._rows_from_submissions(row, info, payload, collected)
    assert len(rows) == 1
    assert rows[0]["available_at"] == rows[0]["acceptance_datetime"]
    assert rows[0]["research_only"] is True
    assert rows[0]["production_changed"] is False


def test_unknown_or_future_acceptance_is_excluded():
    mod = importlib.import_module("scripts.sec_filings_research")
    collected = pd.Timestamp("2026-09-24T00:00:00Z")
    row = {"symbol": "TEST", "name": "Test Co"}
    info = {"cik": "0000000001", "title": "Test Co"}
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["a", "b", "c"],
                "filingDate": ["2026-09-23", "2026-09-23", "2026-09-23"],
                "acceptanceDateTime": ["", "2026-09-24T01:00:00Z", "2026-09-23T19:00:00Z"],
                "form": ["8-K", "8-K", "8-K"],
                "primaryDocument": ["a.htm", "b.htm", "c.htm"],
            }
        }
    }
    rows = mod._rows_from_submissions(row, info, payload, collected)
    assert len(rows) == 1
    assert rows[0]["accession_number"] == "c"


def test_sec_collector_is_research_only():
    source = Path("scripts/sec_filings_research.py").read_text(encoding="utf-8")
    assert "research_only" in source
    assert "production_changed" in source
    assert "load_production_artifact" not in source
