from __future__ import annotations

import json

from scripts import collect_boj_frontier_research as collector


def test_boj_frontier_research_preserves_raw_data_and_pit_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "OUT", tmp_path / "boj_frontier.json")
    monkeypatch.setattr(
        collector,
        "CONFIG",
        tmp_path / "config.yml",
    )
    collector.CONFIG.write_text(
        """
boj_frontier_research:
  enabled: true
  base_url: https://example.org/getDataCode
  language: en
  start_dates:
    MD01: "200001"
  series_groups:
    - name: monetary_base
      db: MD01
      codes:
        - "MD01'MABS1AN11"
""",
        encoding="utf-8",
    )

    payload = {
        "RESULTSET": [
            {
                "SERIES_CODE": "MD01'MABS1AN11",
                "NAME_OF_TIME_SERIES": "Monetary Base/Average Amounts Outstanding",
                "UNIT": "100 million yen",
                "FREQUENCY": "Monthly",
                "VALUES": {
                    "SURVEY_DATES": ["202609"],
                    "VALUES": ["12345.0"],
                },
            }
        ]
    }
    captured = {}

    def fake_fetch(url):
        captured["url"] = url
        return payload, json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(collector, "_fetch", fake_fetch)
    assert collector.main() == 0
    report = json.loads(collector.OUT.read_text(encoding="utf-8"))
    assert report["pit_status"] == "UNVERIFIED"
    assert report["production_changed"] is False
    assert report["summary"]["series_count"] == 1
    assert report["batches"][0]["rows"][0]["value"] == "12345.0"

    from urllib.parse import parse_qs, urlparse

    query = parse_qs(urlparse(captured["url"]).query)
    assert query["db"] == ["MD01"]
    assert query["code"] == ["MABS1AN11"]
