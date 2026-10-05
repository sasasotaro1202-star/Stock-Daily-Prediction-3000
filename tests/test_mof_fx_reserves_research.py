from __future__ import annotations

import json

from scripts import collect_mof_fx_reserves_research as collector


def test_mof_fx_reserves_preserves_raw_csv_and_pit_gate(tmp_path, monkeypatch):
    config = tmp_path / "config.yml"
    config.write_text(
        """
mof_fx_intervention:
  enabled: true
  url: https://example.org/intervention.csv
  output: data/intervention.csv
  pit_policy:
    publication_schedule_verified: true
mof_official_reserves:
  enabled: true
  url: https://example.org/reserves.csv
  output: data/reserves.csv
  pit_policy:
    publication_schedule_verified: true
    revision_history_must_be_preserved: true
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(collector, "CONFIG", config)
    monkeypatch.chdir(tmp_path)

    def fake_fetch(url):
        raw = "date,value\n2026-08-31,123\n".encode("utf-8")
        return raw, {
            "content_type": "text/csv",
            "etag": '"x"',
            "last_modified": "today",
        }

    monkeypatch.setattr(collector, "_fetch", fake_fetch)
    assert collector.main() == 0

    manifest = json.loads(
        (tmp_path / "data/research/mof_fx_reserves_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["production_changed"] is False
    assert all(row["pit_status"] == "UNVERIFIED" for row in manifest["sources"])

    meta = json.loads(
        (tmp_path / "data/intervention.csv.meta.json").read_text(encoding="utf-8")
    )
    assert meta["pit_policy"]["publication_schedule_verified"] is True
    assert meta["pit_policy"]["row_level_available_at_verified"] is False
    assert (tmp_path / "data/intervention.csv").read_text(encoding="utf-8").startswith(
        "date,value"
    )
