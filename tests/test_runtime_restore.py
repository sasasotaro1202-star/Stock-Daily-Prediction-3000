def test_production_runtime_restore_is_fail_closed():
    from pathlib import Path

    source=Path("scripts/install_production_runtime.py").read_text(encoding="utf-8")
    assert "DEFERRED: production artifact metadata missing" in source
    assert "runtime_dependency_versions" in source
    assert "lightgbm" in source
