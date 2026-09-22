def test_production_runtime_restore_is_fail_closed():
    from pathlib import Path

    source=Path("scripts/install_production_runtime.py").read_text(encoding="utf-8")
    assert "DEFERRED: production artifact metadata missing" in source
    assert "runtime_dependency_versions" in source
    assert "lightgbm" in source


def test_restore_skips_approved_but_mismatched_research_state(tmp_path):
    import json
    from scripts.restore_latest_research_state import _has_approved_production_state

    src = tmp_path / "data" / "research"
    src.mkdir(parents=True)
    (src / "production_model_artifact.pkl").write_bytes(b"artifact")
    (src / "release_gate.json").write_text(
        json.dumps({"approved": True}),
        encoding="utf-8",
    )
    (src / "production_model_artifact.meta.json").write_text(
        json.dumps(
            {
                "holdout_generation": 2,
                "research_code_fingerprint_sha256": "old",
                "code_fingerprint_sha256": "old-full",
            }
        ),
        encoding="utf-8",
    )

    assert not _has_approved_production_state(
        tmp_path,
        expected_holdout_generation=3,
        expected_research_fingerprint="new",
        expected_full_fingerprint="new-full",
    )


def test_restore_accepts_exact_approved_research_state(tmp_path):
    import json
    from scripts.restore_latest_research_state import _has_approved_production_state

    src = tmp_path / "data" / "research"
    src.mkdir(parents=True)
    (src / "production_model_artifact.pkl").write_bytes(b"artifact")
    (src / "release_gate.json").write_text(
        json.dumps({"approved": True}),
        encoding="utf-8",
    )
    (src / "production_model_artifact.meta.json").write_text(
        json.dumps(
            {
                "holdout_generation": 3,
                "research_code_fingerprint_sha256": "new",
                "code_fingerprint_sha256": "new-full",
            }
        ),
        encoding="utf-8",
    )

    assert _has_approved_production_state(
        tmp_path,
        expected_holdout_generation=3,
        expected_research_fingerprint="new",
        expected_full_fingerprint="new-full",
    )
