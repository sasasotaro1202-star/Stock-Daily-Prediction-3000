from __future__ import annotations

import json

from scripts import collect_fred_cross_asset_research as collector


def test_fred_cross_asset_preserves_snapshots_and_pit_gate(tmp_path, monkeypatch):
    config = tmp_path / "config.yml"
    config.write_text(
        """
fred_cross_asset:
  enabled: true
  base_url: https://example.org/fredgraph.csv
  series:
    - DFF
    - SOFR
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(collector, "CONFIG", config)
    monkeypatch.setattr(collector, "OUT_DIR", tmp_path / "fred")

    monkeypatch.setattr(
        collector,
        "_fetch",
        lambda url: (
            b"observation_date,value\n2026-09-30,4.00\n",
            {"content_type": "text/csv", "etag": '"x"'},
        ),
    )
    monkeypatch.setenv("FRED_LOOKBACK_YEARS", "2")
    assert collector.main() == 0

    report = json.loads(
        (tmp_path / "fred" / "manifest.json").read_text(encoding="utf-8")
    )
    assert report["pit_status"] == "UNVERIFIED"
    assert report["production_changed"] is False
    assert report["summary"]["successful"] == 2
    assert (tmp_path / "fred" / "DFF.csv").exists()


def test_fred_cross_asset_fails_closed_on_partial_failure(tmp_path, monkeypatch):
    config = tmp_path / "config.yml"
    config.write_text(
        "fred_cross_asset:\n  enabled: true\n  base_url: https://example.org/fredgraph.csv\n  series:\n    - DFF\n    - SOFR\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(collector, "CONFIG", config)
    monkeypatch.setattr(collector, "OUT_DIR", tmp_path / "fred")

    def fake_fetch(url):
        if "DFF" in url:
            return (
                b"observation_date,value\n2026-09-30,4.00\n",
                {"content_type": "text/csv"},
            )
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(collector, "_fetch", fake_fetch)
    assert collector.main() == 1
