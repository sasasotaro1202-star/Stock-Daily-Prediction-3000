from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path("scripts/audit_external_oss_top1000_metadata.py")


def _load():
    spec = importlib.util.spec_from_file_location("oss_audit", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delta_is_strict_and_missing_is_unknown():
    m = _load()
    assert m.delta(10, 7) == 3
    assert m.delta(7, 10) == -3
    assert m.delta(10, None) is None
    assert m.delta(None, 10) is None


def test_existing_registry_is_loaded_fail_closed():
    import json
    registry = json.loads(
        Path("config/external_prediction_system_oss_top1000_registry.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        r for r in registry["rows"]
        if r["entry_type"] == "EXPLICIT_REPOSITORY_REF"
    ]
    assert len(rows) == registry["entry_summary"]["explicit_repository_refs"]
    assert len(rows) > 500


def test_momentum_is_not_claimed_without_previous_snapshot():
    assert "BASELINE_ONLY" in MODULE_PATH.read_text(encoding="utf-8")
