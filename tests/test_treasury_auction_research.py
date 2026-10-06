from __future__ import annotations

import json

from scripts import collect_treasury_auction_research as collector


def test_treasury_auction_research_preserves_publication_date_and_pit_gate(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        collector,
        "OUT",
        tmp_path / "data" / "research" / "treasury_auctions_research.json",
    )

    def fake_get_json(url, params):
        if "upcoming_auctions" in url:
            return (
                {
                    "data": [
                        {
                            "auction_date": "2026-10-05",
                            "security_type": "Note",
                            "security_term": "10-Year",
                            "offering_amt": "39000000000",
                            "record_date": "2026-10-01",
                        }
                    ]
                },
                {"etag": '"upcoming"'},
            )
        return (
            {
                "data": [
                    {
                        "record_date": "2026-09-30",
                        "auction_date": "2026-09-30",
                        "issue_date": "2026-10-01",
                        "security_type": "Note",
                        "security_term": "10-Year",
                        "offering_amt": "39000000000",
                        "total_accepted": "39000000000",
                        "bid_to_cover_ratio": "2.50",
                        "high_yield": "4.10",
                        "int_rate": "4.00",
                        "cusip": "123456789",
                    }
                ]
            },
            {"etag": '"auction"'},
        )

    monkeypatch.setattr(collector, "_get_json", fake_get_json)
    assert collector.main() == 0

    payload = json.loads(collector.OUT.read_text(encoding="utf-8"))
    assert payload["pit_status"] == "UNVERIFIED"
    assert payload["production_changed"] is False
    assert payload["auctions"]["publication_view"][0]["publication_date"] == "2026-09-30"
    assert payload["upcoming_auctions"]["event_view"][0]["auction_date"] == "2026-10-05"
