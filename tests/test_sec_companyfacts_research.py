from __future__ import annotations

import json

from scripts import collect_sec_companyfacts_research as collector


def test_sec_companyfacts_research_collection_is_pit_unverified_and_sharded(
    tmp_path, monkeypatch
):
    universe = tmp_path / "data" / "universe" / "latest.json"
    universe.parent.mkdir(parents=True)
    universe.write_text(
        json.dumps(
            {
                "records": [
                    {"symbol": "AAA", "asset_class": "us_stock", "tradeable": True},
                    {"symbol": "BBB", "asset_class": "us_stock", "tradeable": True},
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
        tmp_path / "data" / "research" / "sec_companyfacts",
    )

    def fake_get_json(url):
        if url == collector.TICKERS_URL:
            return (
                {
                    "0": {
                        "ticker": "AAA",
                        "cik_str": 1,
                        "title": "A Corp",
                        "exchange": "NYSE",
                    },
                    "1": {
                        "ticker": "BBB",
                        "cik_str": 2,
                        "title": "B Corp",
                        "exchange": "NASDAQ",
                    },
                },
                {"etag": '"master"'},
            )
        return (
            {"facts": {"us-gaap": {"Revenues": {"units": {"USD": []}}}}},
            {"etag": '"company"'},
        )

    monkeypatch.setattr(collector, "_get_json", fake_get_json)
    monkeypatch.setenv("SEC_COMPANYFACTS_SHARD_COUNT", "2")
    monkeypatch.setenv("SEC_COMPANYFACTS_SHARD_INDEX", "0")

    report = collector.main()
    assert report == 0

    manifest = json.loads(
        (collector.OUT_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["research_only"] is True
    assert manifest["production_changed"] is False
    assert manifest["pit_status"] == "UNVERIFIED"
    assert manifest["universe"]["matched_to_sec_cik"] == 2
    assert manifest["summary"]["successful"] == 1
    assert (collector.OUT_DIR / "AAA.json").exists()
