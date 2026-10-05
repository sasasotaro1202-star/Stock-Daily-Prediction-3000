from __future__ import annotations

import json

from scripts import collect_sec_company_events_research as collector


def test_normalize_items_handles_list_and_delimited_text():
    assert collector._normalize_items(["1.01", "2.03"]) == ["1.01", "2.03"]
    assert collector._normalize_items("1.01, 5.02; 8.01") == [
        "1.01",
        "5.02",
        "8.01",
    ]


def test_event_rows_preserve_acceptance_and_8k_items():
    payload = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q"],
                "filingDate": ["2026-09-30", "2026-09-29"],
                "acceptanceDateTime": ["20260930143000", "20260929120000"],
                "reportDate": ["2026-09-28", "2026-09-27"],
                "items": [["1.01", "5.02"], ""],
                "accessionNumber": [
                    "0000000001-26-000001",
                    "0000000001-26-000002",
                ],
                "primaryDocument": ["a8k.htm", "q.htm"],
            }
        }
    }
    rows = collector._event_rows(
        payload,
        forms={"8-K", "10-Q"},
        cutoff_date=collector.date(2026, 9, 1),
        max_events=10,
    )
    assert rows[0]["form"] == "8-K"
    assert rows[0]["acceptance_datetime"] == "20260930143000"
    assert rows[0]["items"] == ["1.01", "5.02"]


def test_sec_company_events_research_is_research_only(tmp_path, monkeypatch):
    universe = tmp_path / "universe.json"
    universe.write_text(
        json.dumps(
            {"records": [{"symbol": "AAA", "asset_class": "us_stock", "tradeable": True}]}
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yml"
    config.write_text(
        """
sec_company_events:
  enabled: true
  forms: ["8-K"]
  lookback_days: 365
  max_events_per_symbol: 40
  max_symbols_per_shard: 75
  shard_count: 1
  submissions_url_template: "https://example/submissions/CIK{cik}.json"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(collector, "UNIVERSE", universe)
    monkeypatch.setattr(collector, "CONFIG", config)
    monkeypatch.setattr(collector, "TICKERS_URL", "https://example/tickers.json")
    monkeypatch.setattr(collector, "OUT_DIR", tmp_path / "out")

    def fake_json(url):
        if "tickers" in url:
            return (
                {"0": {"ticker": "AAA", "cik_str": 123, "title": "A Corp", "exchange": "NYSE"}},
                {"etag": '"master"'},
                b"master",
            )
        payload = {
            "filings": {
                "recent": {
                    "form": ["8-K"],
                    "filingDate": ["2026-09-30"],
                    "acceptanceDateTime": ["20260930143000"],
                    "reportDate": ["2026-09-28"],
                    "items": ["1.01,5.02"],
                    "accessionNumber": ["0000000123-26-000001"],
                    "primaryDocument": ["a8k.htm"],
                }
            }
        }
        return payload, {"last_modified": "today"}, json.dumps(payload).encode()

    monkeypatch.setattr(collector, "_get_json", fake_json)
    monkeypatch.setenv("SEC_COMPANY_EVENTS_SHARD_COUNT", "1")
    monkeypatch.setenv("SEC_COMPANY_EVENTS_SHARD_INDEX", "0")
    assert collector.main() == 0

    event = json.loads(
        (tmp_path / "out" / "AAA.events.json").read_text(encoding="utf-8")
    )
    assert event["pit_status"] == "UNVERIFIED"
    assert event["production_changed"] is False
    assert event["events"][0]["items"] == ["1.01", "5.02"]
