from __future__ import annotations

import json

from scripts.run_v13_meta_leakage_audit import main


def test_v13_meta_leakage_audit_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main() == 0
    payload = json.loads(
        (tmp_path / "data/research/v13_meta_leakage_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "PASS"
    assert all(payload["checks"].values())
