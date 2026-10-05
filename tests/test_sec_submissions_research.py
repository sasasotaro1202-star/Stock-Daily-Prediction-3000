from __future__ import annotations

import json

from scripts import collect_sec_submissions_research as collector


def test_sec_submissions_preserves_acceptance_datetime_and_pit_gate(tmp_path, monkeypatch):
    universe = tmp_path / "data" / "universe" / "latest.json"
    universe.parent.mkdir(parents=True)
    universe.write_text(
        json.dumps(
            {
                "records": [
                    {"symbol": "AAA", "asset_class": "us_stock", "tradeable": True},
                    {"symbol": "JPX", "asset_class": "jp_stock", "tradeable": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(collector, "UNIVERSE", universe)
    monkeypatch.setattr(
        collector,
        "OUT_DIR",
        tmp_path / "data" / "research" / "sec_submissions",
    )

    def fake_get_json(url):
        if url == collector.TICKERS_URL:
            return (
                {
                    "0": {
                        "ticker": "AAA",
                        "cik_str": 123,
                        "title": "A Corp",
                        "exchange": "NASDAQ",
                    }
                },
                {"etag": '"master"', "last_modified": "today"},
            )
        return (
            {
                "filings": {
                    "recent": {
                        "form": ["10-Q"],
                        "filingDate": ["2026-09-30"],
                        "acceptanceDateTime": ["20260930143000"],
                        "reportDate": ["2026-09-28"],
                        "accessionNumber": ["0000000123-26-000001"],
                        "primaryDocument": ["a10q.htm"],
                        "isXBRL": [1],
                    }
                }
            },
            {"etag": '"submission"', "last_modified": "today"},
        )

    monkeypatch.setattr(collector, "_get_json", fake_get_json)
    monkeypatch.setenv("SEC_SUBMISSIONS_SHARD_COUNT", "1")
    monkeypatch.setenv("SEC_SUBMISSIONS_SHARD_INDEX", "0")
    monkeypatch.setenv("SEC_SUBMISSIONS_MAX_SYMBOLS_PER_SHARD", "75")
    assert collector.main() == 0

    pit = json.loads(
        (collector.OUT_DIR / "AAA.pit.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (collector.OUT_DIR / "manifest-0.json").read_text(encoding="utf-8")
    )
    assert pit["pit_status"] == "UNVERIFIED"
    assert pit["acceptance_datetime_rows"][0]["acceptance_datetime"] == "20260930143000"
    assert manifest["production_changed"] is False
    assert manifest["summary"]["acceptance_datetime_records"] == 1
