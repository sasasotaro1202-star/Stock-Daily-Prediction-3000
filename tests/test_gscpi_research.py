from __future__ import annotations

import json

import pandas as pd

from scripts import collect_gscpi_research as collector


def test_parse_gscpi_frame_normalizes_date_and_numeric_values(monkeypatch):
    def fake_read_excel(*args, **kwargs):
        return pd.DataFrame(
            {
                "Date": ["31-Jan-2026", "28-Feb-2026", "28-Feb-2026"],
                "GSCPI": ["0.5", 0.75, 0.80],
            }
        )

    monkeypatch.setattr(collector.pd, "read_excel", fake_read_excel)
    frame = collector._parse_frame(b"unused", sheet_name="GSCPI Monthly Data")

    assert list(frame["observation_date"]) == ["2026-01-31", "2026-02-28"]
    assert frame["gscpi"].tolist() == [0.5, 0.8]


def test_gscpi_collector_is_research_only(tmp_path, monkeypatch):
    config = tmp_path / "config.yml"
    config.write_text(
        """
gscpi_newyorkfed:
  enabled: true
  source_url: https://example.org/gscpi.xls
  sheet_name: GSCPI Monthly Data
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(collector, "CONFIG", config)
    monkeypatch.setattr(collector, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(collector, "_fetch_bytes", lambda url: (b"fake", {}))
    monkeypatch.setattr(
        collector.pd,
        "read_excel",
        lambda *args, **kwargs: pd.DataFrame(
            {"Date": ["31-Jan-2026"], "GSCPI": ["0.5"]}
        ),
    )

    assert collector.main() == 0

    report = json.loads(
        (tmp_path / "out" / "manifest.json").read_text(encoding="utf-8")
    )
    assert report["research_only"] is True
    assert report["production_changed"] is False
    assert report["pit_status"] == "UNVERIFIED"
