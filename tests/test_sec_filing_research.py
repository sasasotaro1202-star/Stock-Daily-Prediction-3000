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


def test_sec_request_headers_are_identified_and_rate_limit_friendly():
    mod = importlib.import_module("scripts.sec_filings_research")
    captured = {}

    class DummyResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b"{}"

    def fake_urlopen(req, timeout):
        captured["request"] = req
        captured["timeout"] = timeout
        return DummyResponse()

    original = mod.urlopen
    mod.urlopen = fake_urlopen
    try:
        assert mod._get_json(mod.TICKERS_URL) is not None
    finally:
        mod.urlopen = original

    req = captured["request"]
    assert req.get_header("User-agent") == mod.USER_AGENT
    assert "github.com/sasasotaro1202-star/Stock-Daily-Prediction-3000" in mod.USER_AGENT
    assert req.get_header("Accept") == "application/json"
    assert req.get_header("Accept-encoding") == "gzip, deflate"
    assert captured["timeout"] == mod.REQUEST_TIMEOUT


def test_sec_ablation_uses_research_feature_module():
    source = Path("scripts/run_sec_filing_ablation.py").read_text(encoding="utf-8")
    assert "from src.research.sec_features import" in source
    assert "from src.features.sec_features import" not in source
