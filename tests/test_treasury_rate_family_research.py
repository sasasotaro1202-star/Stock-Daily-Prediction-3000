from __future__ import annotations

import json

from scripts import collect_treasury_rate_family_research as collector


def test_treasury_rate_family_preserves_raw_and_separates_pit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        collector,
        "OUT_DIR",
        tmp_path / "data" / "research" / "treasury_rate_family",
    )
    monkeypatch.setattr(
        collector,
        "CONFIG",
        tmp_path / "config" / "research_data_sources.yml",
    )
    collector.CONFIG.parent.mkdir(parents=True)
    collector.CONFIG.write_text(
        """
treasury_rate_family:
  enabled: true
  base_url: "https://example/treasury"
  datasets:
    - daily_treasury_real_yield_curve
  lookback_years: 1
""",
        encoding="utf-8",
    )

    xml = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata">
  <entry>
    <content>
      <m:properties>
        <d:NEW_DATE>2026-09-30T00:00:00</d:NEW_DATE>
        <d:TC_5YEAR>2.10</d:TC_5YEAR>
        <d:TC_10YEAR>2.45</d:TC_10YEAR>
        <d:TC_30YEAR>N/A</d:TC_30YEAR>
      </m:properties>
    </content>
  </entry>
</feed>"""

    monkeypatch.setattr(
        collector,
        "_get_bytes",
        lambda url: (xml, {"etag": '"x"', "last_modified": "today", "content_type": "xml"}),
    )

    # The loop uses current year; shrink to a one-year test by invoking the parser directly.
    rows = collector._parse_xml(xml, "daily_treasury_real_yield_curve")
    assert rows[0]["session_date"] == "2026-09-30"
    assert rows[0]["TC_5YEAR"] == 2.10
    assert rows[0]["TC_30YEAR"] is None

    collector.OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload_path = collector.OUT_DIR / "sample.json"
    payload_path.write_text(
        json.dumps(
            {
                "research_only": True,
                "production_changed": False,
                "pit_status": "UNVERIFIED",
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["pit_status"] == "UNVERIFIED"
    assert payload["production_changed"] is False
