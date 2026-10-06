from __future__ import annotations

import json

import pytest

from scripts import audit_research_pit_contract as audit


def _write_config(path):
    path.write_text(
        json.dumps({"policy": {"research_only": True}}),
        encoding="utf-8",
    )


def test_required_post_oos_pit_ledger_missing_fails_closed(monkeypatch, tmp_path):
    config_path = tmp_path / "research_data_sources.yml"
    out_path = tmp_path / "audit.json"
    _write_config(config_path)

    monkeypatch.setattr(audit, "CONFIG", config_path)
    monkeypatch.setattr(audit, "LEDGER_CANDIDATES", (tmp_path / "missing.json",))
    monkeypatch.setattr(audit, "OUT", out_path)
    monkeypatch.setenv("REQUIRE_PIT_LEDGER", "1")

    with pytest.raises(SystemExit):
        audit.main()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["prediction_ledger"]["status"] == "FAIL"
    assert "NO_LEDGER_EVIDENCE" in payload["prediction_ledger"]["violations"][0]["violations"]


def test_required_post_oos_pit_ledger_empty_fails_closed(monkeypatch, tmp_path):
    config_path = tmp_path / "research_data_sources.yml"
    ledger_path = tmp_path / "prediction_ledger.json"
    out_path = tmp_path / "audit.json"
    _write_config(config_path)
    ledger_path.write_text(json.dumps({"row_level": []}), encoding="utf-8")

    monkeypatch.setattr(audit, "CONFIG", config_path)
    monkeypatch.setattr(audit, "LEDGER_CANDIDATES", (ledger_path,))
    monkeypatch.setattr(audit, "OUT", out_path)
    monkeypatch.setenv("REQUIRE_PIT_LEDGER", "1")

    with pytest.raises(SystemExit):
        audit.main()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["prediction_ledger"]["rows"] == 0
    assert "NO_LEDGER_EVIDENCE" in payload["prediction_ledger"]["violations"][0]["violations"]


def test_pre_oos_pit_audit_can_remain_unverified_without_ledger(monkeypatch, tmp_path):
    config_path = tmp_path / "research_data_sources.yml"
    out_path = tmp_path / "audit.json"
    _write_config(config_path)

    monkeypatch.setattr(audit, "CONFIG", config_path)
    monkeypatch.setattr(audit, "LEDGER_CANDIDATES", (tmp_path / "missing.json",))
    monkeypatch.setattr(audit, "OUT", out_path)
    monkeypatch.delenv("REQUIRE_PIT_LEDGER", raising=False)

    assert audit.main() == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["status"] == "RESEARCH_ONLY_UNVERIFIED"
    assert payload["prediction_ledger"]["status"] == "NO_LEDGER_EVIDENCE"
