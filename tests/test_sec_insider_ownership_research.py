from __future__ import annotations

import json

from scripts import collect_sec_insider_ownership_research as collector


def test_archive_url_uses_compact_accession():
    url = collector._archive_url(
        "0000123456",
        "0000123456-26-000001",
        "primary_doc.xml",
    )
    assert url.endswith("/123456/000012345626000001/primary_doc.xml")


def test_form4_view_extracts_transaction_fields():
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<ownershipDocument>
  <reportingOwner><reportingOwnerId><rptOwnerName>Jane Doe</rptOwnerName></reportingOwnerId></reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-09-30</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts><transactionShares><value>100</value></transactionShares>
        <transactionPricePerShare><value>12.50</value></transactionPricePerShare></transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""
    view = collector._form4_transaction_view(xml)
    assert view["xml_parse_status"] == "PARSED"
    assert view["reporting_owner_name"] == "Jane Doe"
    assert len(view["transactions"]) == 1
    assert view["transactions"][0]["transactionCode"] == "P"
    assert view["transactions"][0]["transactionShares"] == "100"


def test_sec_insider_research_is_research_only(tmp_path, monkeypatch):
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
sec_insider_ownership:
  enabled: true
  forms: ["4"]
  lookback_days: 180
  max_filings_per_symbol: 2
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
                {"0": {"ticker": "AAA", "cik_str": 123, "title": "A Corp"}},
                {"etag": '"master"'},
            )
        return (
            {
                "filings": {
                    "recent": {
                        "form": ["4"],
                        "filingDate": ["2026-09-30"],
                        "acceptanceDateTime": ["20260930143000"],
                        "reportDate": ["2026-09-30"],
                        "accessionNumber": ["0000000123-26-000001"],
                        "primaryDocument": ["primary_doc.xml"],
                    }
                }
            },
            {"last_modified": "today"},
        )

    xml = b"""<ownershipDocument><reportingOwner><reportingOwnerId><rptOwnerName>Jane Doe</rptOwnerName></reportingOwnerId></reportingOwner><nonDerivativeTable><nonDerivativeTransaction><transactionDate><value>2026-09-30</value></transactionDate><transactionCoding><transactionCode>P</transactionCode></transactionCoding><transactionAmounts><transactionShares><value>10</value></transactionShares></transactionAmounts></nonDerivativeTransaction></nonDerivativeTable></ownershipDocument>"""
    monkeypatch.setattr(collector, "_get_json", fake_json)
    monkeypatch.setattr(
        collector,
        "_get_bytes",
        lambda url: (xml, {"content_type": "application/xml"}),
    )
    monkeypatch.setenv("SEC_INSIDER_SHARD_COUNT", "1")
    monkeypatch.setenv("SEC_INSIDER_SHARD_INDEX", "0")
    assert collector.main() == 0

    manifest = json.loads(
        (tmp_path / "out" / "manifest-0.json").read_text(encoding="utf-8")
    )
    assert manifest["research_only"] is True
    assert manifest["production_changed"] is False
    meta_files = list((tmp_path / "out" / "AAA").glob("*.meta.json"))
    assert len(meta_files) == 1
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert meta["acceptance_datetime"] == "20260930143000"
    assert meta["pit_status"] == "UNVERIFIED"
    assert meta["form4_view"]["transactions"][0]["transactionCode"] == "P"
